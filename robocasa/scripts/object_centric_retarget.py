#!/usr/bin/env python3
"""
Two-stage object-centric retargeting (MimicGen-style) for a pick-and-place task.

The trajectory is split into per-subtask segments, each with its OWN object of
interest (object_ref), and each segment is re-anchored to that object's new pose:

    A_tgt = Hb_tgt @ inv(Hb_src) @ A_src        (active arm only, per segment)

Subtasks for PnP:
    pick  : object_ref = the grasped object (e.g. "vegetable")     [0 : grasp]
    place : object_ref = the destination    (e.g. "vegetable_container") [grasp : end]

The pick->place boundary is the GRIPPER-CLOSE moment (active gripper command
crosses from open to closed). A short slerp interpolation bridges the two
segments so the hand doesn't jump. The idle arm and gripper dims are kept as
recorded.

Run:
    MUJOCO_GL=egl uv run python robocasa/scripts/object_centric_retarget.py \
        --hdf5_path .../PnPSinkToRightCounterPlate/003/demo_im128_notp.hdf5 \
        --pick_object vegetable --place_object vegetable_container \
        --active_arm right --pick_perturb 0.15 0.10 0.30 --save_videos
"""

import os
import sys
import json
import argparse

import h5py
import numpy as np
import cv2
from termcolor import colored

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../icrt"))

import robocasa  # noqa: F401
import robocasa.macros as macros
from robocasa.environments.kitchen.play_env.play_env import *  # noqa: F401,F403
import robosuite.utils.transform_utils as T

from robocasa.utils.mimicdroid_utils import make_env, reset_to, EnvArgs

macros.SHOW_SITES = False

ARM_BLOCK = {"right": 0, "left": 6}
GRIPPER_DIM = {"right": 13, "left": 12}  # 14-dim two-hand layout
active_arm_g = "right"


def body_pose(env, body_name):
    bid = env.sim.model.body_name2id(body_name)
    return T.make_pose(env.sim.data.body_xpos[bid], env.sim.data.body_xmat[bid].reshape(3, 3))


def controller_base_pose(env, arm):
    """The base frame the OSC/IK controller actually uses for base-frame actions
    (NOT the robot0_base body, which can be offset). Matches MimicGen's
    RobosuiteInterface.get_controller_base_pose."""
    pos, rot = env.robots[0].composite_controller.get_controller_base_pose(arm)
    return T.make_pose(np.array(pos), np.array(rot).reshape(3, 3))


def a6_to_pose(a6):
    return T.make_pose(a6[:3], T.quat2mat(T.axisangle2quat(a6[3:6])))


def pose_to_a6(P):
    return np.concatenate([P[:3, 3], T.quat2axisangle(T.mat2quat(P[:3, :3]))])


def detect_grasp_boundary(actions, arm):
    """First step where the active gripper command crosses open(<=0)->close(>0)."""
    g = actions[:, GRIPPER_DIM[arm]]
    for t in range(1, len(g)):
        if g[t - 1] <= 0 and g[t] > 0:
            return t
    return len(actions) // 2  # fallback


def retarget_segment(actions, idx, Hb_src, Hb_tgt, arm):
    """Re-anchor the active arm pose for rows `idx` from Hb_src to Hb_tgt."""
    out = actions.copy()
    b = ARM_BLOCK[arm]
    M = Hb_tgt @ np.linalg.inv(Hb_src)
    for t in idx:
        out[t][b : b + 6] = pose_to_a6(M @ a6_to_pose(actions[t][b : b + 6]))
    return out


def slerp_bridge(a_from, a_to, n, arm, gripper_val, idle_template):
    """n interpolated action rows from pose a_from -> a_to (active arm), gripper held."""
    b = ARM_BLOCK[arm]
    P0, P1 = a6_to_pose(a_from[b : b + 6]), a6_to_pose(a_to[b : b + 6])
    q0, q1 = T.mat2quat(P0[:3, :3]), T.mat2quat(P1[:3, :3])
    rows = []
    for k in range(1, n + 1):
        s = k / (n + 1)
        pos = (1 - s) * P0[:3, 3] + s * P1[:3, 3]
        quat = T.quat_slerp(q0, q1, s)
        row = idle_template.copy()
        row[b : b + 6] = np.concatenate([pos, T.quat2axisangle(quat)])
        row[GRIPPER_DIM[arm]] = gripper_val
        rows.append(row)
    return np.array(rows)


def capture_frame(obs):
    keys = [k for k in obs.keys() if "_image" in k]
    return np.concatenate([obs[k][::-1][..., ::-1] for k in keys], axis=1) if keys else None


def play(env, state_flat, actions, save, lift_body=None):
    env.sim.set_state_from_flattened(state_flat)
    env.sim.forward()
    env.robots[0].reset()
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    z0 = None
    if lift_body is not None:
        z0 = env.sim.data.body_xpos[env.sim.model.body_name2id(lift_body)][2]
    max_lift = 0.0
    min_eef_dist = np.inf
    frames = []
    for a in actions:
        obs, _, done, _ = env.step(a)
        if lift_body is not None:
            opos = env.sim.data.body_xpos[env.sim.model.body_name2id(lift_body)]
            max_lift = max(max_lift, opos[2] - z0)
            ek = f"robot0_{active_arm_g}_eef_pos"
            if ek in obs:
                min_eef_dist = min(min_eef_dist, float(np.linalg.norm(obs[ek] - opos)))
        if save:
            f = capture_frame(obs)
            if f is not None:
                frames.append(f)
        if done:
            break
    return bool(env._check_success()), frames, max_lift, min_eef_dist


def perturb_object(env, obj_name, B, dx, dy, dyaw):
    joint = env.objects[obj_name].joints[0]
    q = env.sim.data.get_joint_qpos(joint).copy()
    q[:3] += B[:3, :3] @ np.array([dx, dy, 0.0])
    if dyaw:
        dq = T.mat2quat(T.euler2mat([0, 0, dyaw]))
        q[3:7] = np.roll(T.quat_multiply(dq, np.roll(q[3:7], -1)), 1)
    env.sim.data.set_joint_qpos(joint, q)
    env.sim.forward()


def obj_base_pose(env, obj_name):
    B = controller_base_pose(env, active_arm_g)
    return np.linalg.inv(B) @ body_pose(env, env.objects[obj_name].root_body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5_path", required=True)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--pick_object", default="vegetable")
    ap.add_argument("--place_object", default="vegetable_container")
    ap.add_argument("--active_arm", default="right", choices=["right", "left"])
    ap.add_argument("--robots", default="DemoTwoHand")
    ap.add_argument("--pick_perturb", type=float, nargs=3, default=[0.15, 0.10, 0.30],
                    metavar=("DX", "DY", "DYAW"), help="move the pick object by this base-frame offset")
    ap.add_argument("--place_perturb", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("DX", "DY", "DYAW"))
    ap.add_argument("--bridge_steps", type=int, default=8)
    ap.add_argument("--settle_steps", type=int, default=20,
                    help="physics steps to let the perturbed object settle before retargeting")
    ap.add_argument("--reset_target", action="store_true",
                    help="use env.reset() (stable placement sampler) for the target instead of manual perturb")
    ap.add_argument("--seed", type=int, default=1, help="(reset_target) number of resets to advance")
    ap.add_argument("--save_videos", action="store_true")
    ap.add_argument("--video_dir", default="/mnt/ddn/tmp/replay_videos")
    args = ap.parse_args()

    with h5py.File(args.hdf5_path, "r") as f:
        eps = sorted(f["data"].keys())
        ep_name = args.episode or eps[0]
        ep = f["data"][ep_name]
        actions = ep["actions"][:]
        init_state = {
            "states": ep["states"][0],
            "ep_meta": json.loads(ep.attrs["ep_meta"]),
            "model": ep.attrs["model_file"],
            "non_robot_qpos_idx": ep.attrs["non_robot_qpos_idx"],
            "qpos_state": ep.attrs["initial_qpos"],
        }

    grasp = detect_grasp_boundary(actions, args.active_arm)
    pick_idx = range(0, grasp)
    place_idx = range(grasp, len(actions))
    print(colored(f"grasp(=pick→place 경계) @ step {grasp} / {len(actions)}", "cyan"))

    env_args = EnvArgs(robots=args.robots, render=False, control_freq=20,
                       controller="WHOLE_BODY_MINK_IK" if "GR1" in args.robots else None,
                       use_camera_obs=args.save_videos)
    env, _ = make_env(file_name=args.hdf5_path, env_args=env_args)

    # ---- SOURCE poses ----
    reset_to(env, init_state, replace_robot_joints=True, change_to_gr1="GR1" in args.robots)
    Hb_pick_src = obj_base_pose(env, args.pick_object)
    Hb_place_src = obj_base_pose(env, args.place_object)
    global active_arm_g
    active_arm_g = args.active_arm
    pick_body = env.objects[args.pick_object].root_body
    src_state = env.sim.get_state().flatten()
    ok_src, _, lift_src, dist_src = play(env, src_state, actions, False, pick_body)
    print(colored(f"[sanity] 원본 @ 소스 -> success={ok_src} (물체 들림 {lift_src:.3f}m, 손-물체 최소 {dist_src:.3f}m)", "green" if ok_src else "red"))

    # ---- TARGET ----
    if args.reset_target:
        # env.reset() resamples ALL object placements at STABLE rest poses via the
        # robocasa placement_initializer (same object instances), which is the
        # realistic generation distribution and avoids the rolling/settle issue of
        # manually setting a free-joint qpos.
        set_meta = env.set_ep_meta if hasattr(env, "set_ep_meta") else env.set_attrs_from_ep_meta
        for _ in range(args.seed):
            set_meta(init_state["ep_meta"])
            env.reset()
    else:
        # controlled per-object perturbation (for stress-testing larger displacements)
        reset_to(env, init_state, replace_robot_joints=True, change_to_gr1="GR1" in args.robots)
        B = controller_base_pose(env, args.active_arm)
        perturb_object(env, args.pick_object, B, *args.pick_perturb)
        if any(args.place_perturb):
            perturb_object(env, args.place_object, B, *args.place_perturb)
        for _ in range(args.settle_steps):
            env.step(actions[0])
        env.sim.forward()
    Hb_pick_tgt = obj_base_pose(env, args.pick_object)
    Hb_place_tgt = obj_base_pose(env, args.place_object)
    tgt_state = env.sim.get_state().flatten()
    print(colored(f"pick object 이동 {np.linalg.norm(Hb_pick_tgt[:3,3]-Hb_pick_src[:3,3]):.3f} m | "
                  f"place object 이동 {np.linalg.norm(Hb_place_tgt[:3,3]-Hb_place_src[:3,3]):.3f} m", "cyan"))

    # ---- build retargeted trajectory (2-stage + bridge) ----
    rt = retarget_segment(actions, pick_idx, Hb_pick_src, Hb_pick_tgt, args.active_arm)
    rt = retarget_segment(rt, place_idx, Hb_place_src, Hb_place_tgt, args.active_arm)
    gclose = actions[grasp, GRIPPER_DIM[args.active_arm]]
    bridge = slerp_bridge(rt[grasp - 1], rt[grasp], args.bridge_steps,
                          args.active_arm, gclose, rt[grasp].copy()) if args.bridge_steps else np.empty((0, actions.shape[1]))
    retargeted = np.concatenate([rt[:grasp], bridge, rt[grasp:]], axis=0)

    # also single-stage (whole trajectory rel to pick object) for comparison
    single = retarget_segment(actions, range(len(actions)), Hb_pick_src, Hb_pick_tgt, args.active_arm)

    # ---- compare on the SAME target scene ----
    ok_orig, fr_orig, lift_o, dist_o = play(env, tgt_state, actions, args.save_videos, pick_body)
    ok_single, _, lift_s, dist_s = play(env, tgt_state, single, False, pick_body)
    ok_2stage, fr_rt, lift_2, dist_2 = play(env, tgt_state, retargeted, args.save_videos, pick_body)
    print(colored(f"원본 actions         @ TARGET -> {ok_orig}  (들림 {lift_o:.3f}m, 손-물체최소 {dist_o:.3f}m)", "green" if ok_orig else "red"))
    print(colored(f"단일스테이지 retarget @ TARGET -> {ok_single}  (들림 {lift_s:.3f}m, 손-물체최소 {dist_s:.3f}m)", "green" if ok_single else "red"))
    print(colored(f"2-스테이지 retarget   @ TARGET -> {ok_2stage}  (들림 {lift_2:.3f}m, 손-물체최소 {dist_2:.3f}m)", "green" if ok_2stage else "red"))

    if args.save_videos:
        os.makedirs(args.video_dir, exist_ok=True)
        for tag, fr in [("orig", fr_orig), ("2stage_retarget", fr_rt)]:
            if fr:
                p = os.path.join(args.video_dir, f"OCR_{ep_name}_{tag}.mp4")
                vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30, (fr[0].shape[1], fr[0].shape[0]))
                for f in fr:
                    vw.write(f)
                vw.release()
                print(colored(f"video -> {p}", "green"))

    print(colored(f"\nRESULT  원본={ok_orig}  단일={ok_single}  2스테이지={ok_2stage}", "magenta", attrs=["bold"]))


if __name__ == "__main__":
    main()
