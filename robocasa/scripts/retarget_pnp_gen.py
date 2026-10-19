#!/usr/bin/env python3
"""
First object-centric retargeting generation test (MimicGen-style) for a PnP task.

Idea: the recorded action trajectory is a sequence of ABSOLUTE eef pose targets
(base frame) tied to the source object's pose. To "generate" a demo for a NEW
object placement, we express the active arm's targets relative to the source
object pose, then re-anchor them to the new object pose:

    A_tgt = Hb_tgt @ inv(Hb_src) @ A_src      (active arm only)

where Hb is the object pose expressed in the robot base frame. The idle arm and
grippers are left unchanged (these tasks are single-arm, no coordination).

We compare, on the SAME randomized target scene:
  - ORIGINAL actions   (should miss — object moved)
  - RETARGETED actions (should track the new object)

Run:
    MUJOCO_GL=egl uv run python robocasa/scripts/retarget_pnp_gen.py \
        --hdf5_path /mnt/ddn/dataset/MimicDroidDataset/TaskDemos/PnPSinkToRightCounterPlate/003/demo_im128_notp.hdf5 \
        --object vegetable --active_arm right --seed 1
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

ARM_BLOCK = {"right": 0, "left": 6}  # start index of the 6-DoF pose block per arm


def make_pose(pos, rot):
    return T.make_pose(np.asarray(pos), np.asarray(rot).reshape(3, 3))


def body_pose(env, body_name):
    bid = env.sim.model.body_name2id(body_name)
    return make_pose(env.sim.data.body_xpos[bid], env.sim.data.body_xmat[bid])


def a6_to_pose(a6):
    return T.make_pose(a6[:3], T.quat2mat(T.axisangle2quat(a6[3:6])))


def pose_to_a6(P):
    aa = T.quat2axisangle(T.mat2quat(P[:3, :3]))
    return np.concatenate([P[:3, 3], aa])


def retarget(actions, Hb_src, Hb_tgt, arm):
    """Re-anchor the active arm's targets from Hb_src to Hb_tgt (base frame)."""
    out = actions.copy()
    b = ARM_BLOCK[arm]
    Hbi = np.linalg.inv(Hb_src)
    for t in range(len(actions)):
        A = a6_to_pose(actions[t][b : b + 6])
        out[t][b : b + 6] = pose_to_a6(Hb_tgt @ Hbi @ A)
    return out


def capture_frame(obs):
    keys = [k for k in obs.keys() if "_image" in k]
    if not keys:
        return None
    return np.concatenate([obs[k][::-1][..., ::-1] for k in keys], axis=1)


def play(env, init_state_flat, actions, save_frames):
    """Restore to a captured sim state, replay actions, return (success, frames)."""
    env.sim.set_state_from_flattened(init_state_flat)
    env.sim.forward()
    env.robots[0].reset()
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    frames = []
    for a in actions:
        obs, _, done, _ = env.step(a)
        if save_frames:
            f = capture_frame(obs)
            if f is not None:
                frames.append(f)
        if done:
            break
    return bool(env._check_success()), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5_path", required=True)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--object", default="vegetable", help="manipulated object name (env.objects key)")
    ap.add_argument("--active_arm", default="right", choices=["right", "left"])
    ap.add_argument("--robots", default="GR1TwoHand")
    ap.add_argument("--seed", type=int, default=1, help="(env.reset mode) resets advanced this many times")
    ap.add_argument("--perturb", type=float, nargs=3, default=[0.08, 0.05, 0.0],
                    metavar=("DX", "DY", "DYAW"),
                    help="Controlled target: move ONLY the object of interest by this base-frame "
                         "offset (dx, dy meters; dyaw radians), keeping the same scene & instance. "
                         "This isolates the retargeting test. Use --reset_target for full resample.")
    ap.add_argument("--reset_target", action="store_true",
                    help="Use env.reset() resampling for the target instead of a controlled perturb")
    ap.add_argument("--video_dir", default="/mnt/ddn/tmp/replay_videos")
    ap.add_argument("--save_videos", action="store_true")
    args = ap.parse_args()

    with h5py.File(args.hdf5_path, "r") as f:
        eps = sorted(f["data"].keys())
        ep_name = args.episode or eps[0]
        ep = f["data"][ep_name]
        actions = ep["actions"][:]
        ep_meta = json.loads(ep.attrs["ep_meta"])
        init_state = {
            "states": ep["states"][0],
            "ep_meta": ep_meta,
            "model": ep.attrs["model_file"],
            "non_robot_qpos_idx": ep.attrs["non_robot_qpos_idx"],
            "qpos_state": ep.attrs["initial_qpos"],
        }

    env_args = EnvArgs(
        robots=args.robots, render=False, control_freq=20,
        controller="WHOLE_BODY_MINK_IK" if "GR1" in args.robots else None,
        use_camera_obs=args.save_videos,
    )
    env, _ = make_env(file_name=args.hdf5_path, env_args=env_args)
    obj_body = env.objects[args.object].root_body

    # ---- SOURCE: reset to the recorded demo, read source object pose (base frame) ----
    reset_to(env, init_state, replace_robot_joints=True, change_to_gr1="GR1" in args.robots)
    B = body_pose(env, "robot0_base")
    Hb_src = np.linalg.inv(B) @ body_pose(env, obj_body)
    print(colored(f"source {args.object} pose (base): {Hb_src[:3,3].round(3)}", "cyan"))

    # baseline: does the ORIGINAL trajectory still succeed on the SOURCE scene?
    src_state = env.sim.get_state().flatten()
    ok_src, _ = play(env, src_state, actions, save_frames=False)
    print(colored(f"[sanity] original actions on SOURCE scene -> success={ok_src}", "green" if ok_src else "red"))

    # ---- TARGET ----
    if args.reset_target:
        # full resample (confounded: all objects + possibly new instances move)
        set_meta = env.set_ep_meta if hasattr(env, "set_ep_meta") else env.set_attrs_from_ep_meta
        for _ in range(args.seed):
            set_meta(ep_meta)
            env.reset()
    else:
        # controlled: move ONLY the object of interest by a known base-frame offset,
        # keeping the same scene and the same object instance.
        reset_to(env, init_state, replace_robot_joints=True, change_to_gr1="GR1" in args.robots)
        joint = env.objects[args.object].joints[0]
        q = env.sim.data.get_joint_qpos(joint).copy()  # [x,y,z, qw,qx,qy,qz]
        dx, dy, dyaw = args.perturb
        # offset is given in base frame; rotate into world for the joint qpos
        off_base = np.array([dx, dy, 0.0])
        off_world = B[:3, :3] @ off_base
        q[:3] += off_world
        if dyaw != 0.0:
            dq = T.mat2quat(T.euler2mat([0, 0, dyaw]))  # xyzw
            cur = np.roll(q[3:7], -1)                    # wxyz->xyzw
            new = T.quat_multiply(dq, cur)
            q[3:7] = np.roll(new, 1)                     # xyzw->wxyz
        env.sim.data.set_joint_qpos(joint, q)
        env.sim.forward()

    Bt = body_pose(env, "robot0_base")
    Hb_tgt = np.linalg.inv(Bt) @ body_pose(env, obj_body)
    dR = Hb_tgt[:3, :3] @ Hb_src[:3, :3].T
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    print(colored(f"target {args.object} pose (base): {Hb_tgt[:3,3].round(3)}  "
                  f"(이동 {np.linalg.norm(Hb_tgt[:3,3]-Hb_src[:3,3]):.3f} m, 회전 {ang:.1f}deg)", "cyan"))
    tgt_state = env.sim.get_state().flatten()

    retargeted = retarget(actions, Hb_src, Hb_tgt, args.active_arm)

    # ---- compare on the SAME target scene ----
    ok_orig, fr_orig = play(env, tgt_state, actions, args.save_videos)
    print(colored(f"ORIGINAL actions   on TARGET -> success={ok_orig}", "green" if ok_orig else "red"))
    ok_rt, fr_rt = play(env, tgt_state, retargeted, args.save_videos)
    print(colored(f"RETARGETED actions on TARGET -> success={ok_rt}", "green" if ok_rt else "red"))

    if args.save_videos:
        os.makedirs(args.video_dir, exist_ok=True)
        for tag, fr in [("orig_on_target", fr_orig), ("retarget_on_target", fr_rt)]:
            if not fr:
                continue
            p = os.path.join(args.video_dir, f"PnP_gen_{ep_name}_{tag}.mp4")
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30, (fr[0].shape[1], fr[0].shape[0]))
            for f in fr:
                vw.write(f)
            vw.release()
            print(colored(f"video -> {p}", "green"))

    print(colored(f"\nRESULT: original={ok_orig}  retargeted={ok_rt}  "
                  f"(retargeting {'HELPED' if ok_rt and not ok_orig else 'no change' if ok_rt==ok_orig else 'HURT'})",
                  "magenta", attrs=["bold"]))


if __name__ == "__main__":
    main()
