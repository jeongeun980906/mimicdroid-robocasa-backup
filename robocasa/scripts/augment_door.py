#!/usr/bin/env python3
"""
Augment a Close*CabinetDoor demo by randomizing the door's initial open angle
and door-centrically retargeting the (single-arm) closing trajectory.

Unlike PnP, the manipulated thing is a FIXTURE door rotating about a fixed
vertical hinge -- not a free object. To generate a demo for a different initial
open angle theta0, we:

  1. set the hinge qpos to theta0 at reset (the rest of the scene is resampled by
     env.reset(), which also varies the visual style),
  2. remap the demo's door-angle profile theta_demo(t) -> theta_demo(t)*theta0/theta0_demo
     (both end at the closed angle 0), and
  3. rotate each active-hand eef target about the hinge's vertical axis by
     dtheta(t) = theta_new(t) - theta_demo(t), so the hand keeps the same pose
     RELATIVE to the door throughout the push.

Saving matches the original dataset format (obs incl. upright images, action_dict,
objects-joint-state, correct per-scene qpos) -- identical to augment_pnp.

Run:
    MUJOCO_GL=egl uv run python robocasa/scripts/augment_door.py \
        --hdf5_path .../CloseLeftCabinetDoor/003/demo_im128_notp.hdf5 \
        --demo_idx 0 --target 200 --hinge cab_2_main_group_leftdoorhinge \
        --active_arm left --no_segmentation --out parts/door0.hdf5
"""

import os
import sys
import json
import argparse

import h5py
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.dirname(__file__))
import augment_pnp as AP  # helpers: controller_base_pose, a6_to_pose, pose_to_a6, replay_record, merge
import robosuite.utils.transform_utils as T
from robocasa.utils.mimicdroid_utils import make_env, reset_to, EnvArgs


def Rax(axis, a):
    """Rotation matrix by angle a (rad) about an arbitrary unit axis (Rodrigues)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def door_angle_traj(env, init, states, hinge):
    """Door hinge qpos at each recorded timestep."""
    reset_to(env, init, replace_robot_joints=True, change_to_gr1=False)
    ang = []
    for s in states:
        env.sim.set_state_from_flattened(s)
        env.sim.forward()
        ang.append(float(env.sim.data.get_joint_qpos(hinge)))
    return np.array(ang)


def retarget_door(actions, arm, Bpose, axis, pivot, th_demo, th0, thend, th_new0):
    """Rotate the active arm's eef targets about the fixture joint's rotation axis
    by the per-step angle remap, so the joint is driven from th_new0 to thend
    (the demo's final value). Works for any single rotating fixture joint: a
    vertical door hinge (axis ~ +z, thend ~ 0) or a horizontal faucet handle
    (axis ~ x, thend ~ the "on" angle). pivot is the joint's world anchor (3D)."""
    b = AP.ARM_BLOCK[arm]
    out = actions.copy()
    frac = (th_demo - th0) / (thend - th0 + 1e-9)   # 0 at start, 1 at the demo's end
    th_new = th_new0 + frac * (thend - th_new0)      # start at th_new0, end at thend
    for t in range(len(actions)):
        dth = th_new[t] - th_demo[t]
        Hw = Bpose @ AP.a6_to_pose(actions[t][b : b + 6])
        R = Rax(axis, dth)
        Hw2 = T.make_pose(pivot + R @ (Hw[:3, 3] - pivot), R @ Hw[:3, :3])
        out[t][b : b + 6] = AP.pose_to_a6(np.linalg.inv(Bpose) @ Hw2)
    return out


def generate(args):
    with h5py.File(args.hdf5_path, "r") as f:
        env_args_str = f["data"].attrs["env_args"]
        eps = sorted(f["data"].keys())
        ep = f["data"][eps[args.demo_idx]]
        actions = ep["actions"][:]
        states = ep["states"][:]
        ep_meta = json.loads(ep.attrs["ep_meta"])
        src = dict(model=ep.attrs["model_file"], states0=ep["states"][0],
                   non_robot=ep.attrs["non_robot_qpos_idx"], qpos=ep.attrs["initial_qpos"],
                   ep_meta_info=ep.attrs.get("ep_meta_info", ""))
        src_name = eps[args.demo_idx]
        orig_obs_keys = set(ep["obs"].keys())

    arm = args.active_arm
    env, _ = make_env(file_name=args.hdf5_path,
                      env_args=EnvArgs(robots=args.robots, render=False, control_freq=20,
                                       use_camera_obs=True,
                                       camera_segmentations=None if args.no_segmentation else "instance"))
    set_meta = env.set_ep_meta if hasattr(env, "set_ep_meta") else env.set_attrs_from_ep_meta
    styles = args.styles or [f"{i:03d}_l1" for i in range(40)]

    # obs format (match original minus bbox, optionally minus segmentation)
    bbox_keys = {k for k in orig_obs_keys if "bbox" in k}
    seg_keys = {k for k in orig_obs_keys if "segmentation" in k} if args.no_segmentation else set()
    store_keys = orig_obs_keys - bbox_keys - seg_keys
    env_keys = store_keys - {"object", "objects-joint-state"}

    # demo door-angle profile + hinge pivot (world x,y), read once
    init0 = {"states": src["states0"], "ep_meta": ep_meta, "model": src["model"],
             "non_robot_qpos_idx": src["non_robot"], "qpos_state": src["qpos"]}
    th_demo = door_angle_traj(env, init0, states, args.hinge)
    th0 = th_demo[0]
    reset_to(env, init0, replace_robot_joints=True, change_to_gr1=False)
    jid = env.sim.model.joint_name2id(args.hinge)
    axis = env.sim.data.xaxis[jid].copy()       # joint rotation axis (world)
    pivot = env.sim.data.xanchor[jid].copy()    # joint anchor (world, 3D)
    thend = float(th_demo[-1])                   # demo's final joint value (closed / on)
    print(colored(f"[{src_name}] hinge={args.hinge} axis={axis.round(2)} pivot={pivot.round(3)} "
                  f"demo angle {th0:.3f}->{thend:.3f}", "cyan"), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = h5py.File(args.out, "w")
    od = out.create_group("data")
    od.attrs["env_args"] = env_args_str
    gi, att = 0, 0
    max_att = args.max_attempts or args.target * 4
    rng = np.random.RandomState(args.seed)
    while gi < args.target and att < max_att:
        style = styles[att % len(styles)]
        att += 1
        meta = json.loads(json.dumps(ep_meta))
        meta["style_id"] = style
        try:
            set_meta(meta)
            env.reset()
        except Exception:
            continue
        # randomize the initial open angle, set hinge, settle briefly
        th_new0 = float(rng.uniform(args.angle_lo, args.angle_hi))
        try:
            Bpose = AP.controller_base_pose(env, arm)
            env.sim.data.set_joint_qpos(args.hinge, th_new0)
            env.sim.forward()
        except Exception:
            continue
        model_xml = env.sim.model.get_xml()
        traj = retarget_door(actions, arm, Bpose, axis, pivot, th_demo, th0, thend, th_new0)

        ok, st, obs_list = AP.replay_record(env, traj, capture_obs=True)
        if ok:
            n = len(st)
            nq = env.sim.model.nq
            robot_idx = set(int(i) for i in env.robots[0]._ref_joint_pos_indexes)
            non_robot_idx = np.array(sorted(set(range(nq)) - robot_idx))
            qpos_all = np.stack([st[t][1:1 + nq] for t in range(n)])
            objects_joint_state = qpos_all[:, non_robot_idx]

            g = od.create_group(f"demo_{gi}")
            g.create_dataset("actions", data=traj.astype(np.float32))
            g.create_dataset("states", data=st)
            g.create_dataset("dones", data=np.zeros(n, np.int64))
            g.create_dataset("rewards", data=np.zeros(n, np.int64))
            og = g.create_group("obs")
            for key in env_keys:
                if key in obs_list[0]:
                    arr = np.stack([np.asarray(o[key]) for o in obs_list])
                    if "image" in key:
                        arr = arr[:, ::-1]  # bottom-origin -> upright
                    og.create_dataset(key, data=arr)
            if "object-state" in og:
                og.create_dataset("object", data=og["object-state"][()])
            og.create_dataset("objects-joint-state", data=objects_joint_state)
            g.attrs["model_file"] = model_xml
            g.attrs["ep_meta"] = json.dumps(meta)
            g.attrs["ep_meta_info"] = src["ep_meta_info"]
            g.attrs["initial_qpos"] = qpos_all[0]
            g.attrs["non_robot_qpos_idx"] = non_robot_idx
            g.attrs["num_samples"] = n
            g.attrs["source_demo"] = src_name
            g.attrs["style_id"] = style
            g.attrs["init_door_angle"] = th_new0
            gi += 1
        if att % 25 == 0 or ok:
            print(colored(f"[{src_name}] {gi}/{args.target} ({att} att, style={style}, "
                          f"th0={th_new0:.2f})", "green" if ok else "yellow"), flush=True)
    od.attrs["total"] = sum(g["actions"].shape[0] for g in od.values())
    out.close()
    try:
        import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
        DatasetUtils.extract_action_dict(dataset=args.out)
    except Exception as e:
        print(colored(f"[{src_name}] action_dict 추가 실패(무시): {e}", "yellow"), flush=True)
    print(colored(f"[{src_name}] DONE: {gi} demos / {att} attempts -> {args.out}", "magenta"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5_path", required=False)
    ap.add_argument("--demo_idx", type=int, default=0)
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--max_attempts", type=int, default=0)
    ap.add_argument("--robots", default="DemoTwoHand")
    ap.add_argument("--active_arm", default="left", choices=["left", "right"])
    ap.add_argument("--hinge", default="cab_2_main_group_leftdoorhinge",
                    help="name of the door hinge joint that closes during the demo")
    ap.add_argument("--angle_lo", type=float, default=-1.6, help="most-open initial angle (rad)")
    ap.add_argument("--angle_hi", type=float, default=-0.7, help="least-open initial angle (rad)")
    ap.add_argument("--no_segmentation", action="store_true")
    ap.add_argument("--styles", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=False)
    ap.add_argument("--merge", nargs="+", help="merge mode: list of part files")
    args = ap.parse_args()
    if args.merge:
        AP.merge(args.merge, args.out)
    else:
        generate(args)


if __name__ == "__main__":
    main()
