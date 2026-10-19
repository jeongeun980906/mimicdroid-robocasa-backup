#!/usr/bin/env python3
"""
Analyze IK tracking error during dataset replay.

For each step we compare the *commanded* end-effector pose (the absolute eef
target stored in the dataset action, base frame) against the eef pose actually
*observed after env.step* on the replay robot. The gap is the IK / low-level
tracking error that explains replay failures.

We also report the deviation between the dataset's recorded eef pose (what the
demo robot achieved) and the replay robot's eef pose.

Action layout (14-dim two-hand, absolute, base frame):
    [0:3] right pos, [3:6] right axis-angle, [6:9] left pos, [9:12] left aa,
    [12] gripper, [13] gripper.

Example:
    MUJOCO_GL=egl uv run python robocasa/scripts/analyze_ik_error.py \
        --hdf5_path /mnt/ddn/dataset/MimicDroidDataset/TaskDemos/CloseLeftCabinetDoor/003/demo_im128_notp.hdf5 \
        --episodes CloseLeftCabinetDoor_D_demo_1 CloseLeftCabinetDoor_D_demo_2
"""

import os
import sys
import json
import argparse

import h5py
import numpy as np
from termcolor import colored

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../icrt"))

import robocasa  # noqa: F401
import robocasa.macros as macros
from robocasa.environments.kitchen.play_env.play_env import *  # noqa: F401,F403
import robosuite.utils.transform_utils as T

from robocasa.utils.mimicdroid_utils import make_env, reset_to, resample_actions, EnvArgs

macros.SHOW_SITES = False

# Indices into the 14-dim absolute action for each hand's 6-DoF pose block.
HAND_BLOCKS = {"right": 0, "left": 6}


def quat_angle_deg(quat_a_xyzw, quat_b_xyzw):
    """Angle (deg) between two xyzw quaternions."""
    Ra = T.quat2mat(quat_a_xyzw)
    Rb = T.quat2mat(quat_b_xyzw)
    cos = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def analyze_episode(env, hdf5_path, ep_name, robots, speed):
    """Replay one episode, return per-step IK tracking errors for each hand."""
    with h5py.File(hdf5_path, "r") as f:
        ep = f["data"][ep_name]
        states = ep["states"][:]
        actions = ep["actions"][:]
        # dataset-recorded eef poses (base frame), per hand
        data_eef = {
            h: {
                "pos": ep[f"obs/robot0_base_to_{h}_eef_pos"][:],
                "quat": ep[f"obs/robot0_base_to_{h}_eef_quat"][:],  # xyzw
            }
            for h in HAND_BLOCKS
        }
        ep_meta = ep.attrs.get("ep_meta", {})
        if isinstance(ep_meta, str):
            ep_meta = json.loads(ep_meta)
        initial_state = {
            "states": states[0],
            "ep_meta": ep_meta,
            "model": ep.attrs.get("model_file", None),
            "non_robot_qpos_idx": ep.attrs.get("non_robot_qpos_idx", None),
            "qpos_state": ep.attrs.get("initial_qpos", None),
        }
        initial_state = {k: v for k, v in initial_state.items() if v is not None}

    reset_to(env, initial_state, replace_robot_joints=True, change_to_gr1="GR1" in robots)

    # NB: only analyze at 1.0x so step i aligns with dataset row i. For speed!=1
    # the dataset eef arrays no longer line up, so we only track command-vs-live.
    use_data_eef = speed == 1.0
    play_actions = resample_actions(actions, speed)

    rec = {h: {"cmd_pos_err": [], "cmd_ori_err": [], "dev_pos_err": [], "dev_ori_err": []}
           for h in HAND_BLOCKS}

    for i, action in enumerate(play_actions):
        obs, _, done, _ = env.step(action)
        for h, b in HAND_BLOCKS.items():
            # commanded target from the action (base frame)
            tgt_pos = action[b : b + 3]
            tgt_quat = T.axisangle2quat(action[b + 3 : b + 6])  # xyzw
            # observed-after-step eef pose (base frame) from live obs
            live_pos = obs[f"robot0_base_to_{h}_eef_pos"]
            live_quat = obs[f"robot0_base_to_{h}_eef_quat"]
            rec[h]["cmd_pos_err"].append(float(np.linalg.norm(live_pos - tgt_pos)))
            rec[h]["cmd_ori_err"].append(quat_angle_deg(live_quat, tgt_quat))
            if use_data_eef and i < len(data_eef[h]["pos"]):
                d_pos = data_eef[h]["pos"][i]
                d_quat = data_eef[h]["quat"][i]
                rec[h]["dev_pos_err"].append(float(np.linalg.norm(live_pos - d_pos)))
                rec[h]["dev_ori_err"].append(quat_angle_deg(live_quat, d_quat))
        if done:
            break

    success = bool(env._check_success())
    return rec, success, len(play_actions)


def summarize(rec, key):
    vals = np.array(rec[key]) if rec[key] else np.array([0.0])
    return vals.mean(), vals.max(), int(np.argmax(vals))


def main():
    ap = argparse.ArgumentParser(description="Analyze IK tracking error during replay")
    ap.add_argument("--hdf5_path", required=True)
    ap.add_argument("--episodes", nargs="+", default=None,
                    help="Episode names (default: all in file)")
    ap.add_argument("--robots", default="GR1TwoHand", choices=["DemoTwoHand", "GR1TwoHand"])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--output", default=None, help="Optional JSON path for full per-step errors")
    args = ap.parse_args()

    with h5py.File(args.hdf5_path, "r") as f:
        all_eps = sorted(f["data"].keys())
    episodes = args.episodes or all_eps

    env_args = EnvArgs(
        robots=args.robots, render=False, control_freq=20,
        controller="WHOLE_BODY_MINK_IK" if "GR1" in args.robots else None,
        use_camera_obs=False,
    )
    env, _ = make_env(file_name=args.hdf5_path, env_args=env_args)

    dump = {}
    print(colored(f"\n{'='*78}", "cyan"))
    print(colored("IK TRACKING ERROR  (commanded action target  vs  observed eef after step)", "cyan", attrs=["bold"]))
    print(colored(f"speed={args.speed}x   units: pos=meters, ori=degrees", "cyan"))
    print(colored(f"{'='*78}", "cyan"))
    header = f"{'episode':<34}{'res':<6}{'hand':<6}{'pos_mean':>9}{'pos_max':>9}{'@step':>7}{'ori_mean':>9}{'ori_max':>9}"
    print(header)
    print("-" * len(header))

    for ep in episodes:
        rec, success, n = analyze_episode(env, args.hdf5_path, ep, args.robots, args.speed)
        dump[ep] = {"success": success, "n_steps": n, "errors": rec}
        tag = colored("OK  ", "green") if success else colored("FAIL", "red")
        for h in HAND_BLOCKS:
            pm, px, pi = summarize(rec[h], "cmd_pos_err")
            om, ox, _ = summarize(rec[h], "cmd_ori_err")
            line = (f"{ep:<34}{'':<6}{h:<6}{pm:>9.4f}{px:>9.4f}{pi:>7d}{om:>9.2f}{ox:>9.2f}")
            # put the res tag only on the first hand row
            if h == "right":
                line = f"{ep:<34}" + tag + " " + line[34 + 6:]
            print(line)
        # deviation vs dataset-recorded eef (only meaningful at 1.0x)
        if args.speed == 1.0:
            for h in HAND_BLOCKS:
                dm, dx, di = summarize(rec[h], "dev_pos_err")
                print(colored(f"{'  └ vs demo eef ('+h+')':<40}{'dev_pos m':>0} mean={dm:.4f} max={dx:.4f} @step{di}", "yellow"))

    print("-" * len(header))
    print(colored("cmd_* = how far the replay robot's hand is from the commanded target (IK/tracking error)", "white"))
    print(colored("dev_* = how far the replay hand is from the demo's recorded hand pose", "white"))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(dump, f, indent=2)
        print(colored(f"\nFull per-step errors -> {args.output}", "green"))


if __name__ == "__main__":
    main()
