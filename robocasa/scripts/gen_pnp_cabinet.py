#!/usr/bin/env python3
"""PnPSinkToCabinet demo generator with CONTROLLED spawn + t=0 visibility.

Replaces augment_pnp.py's placement for the Cabinet task. augment_pnp is
RightCounterPlate-centric: its reset() sampler + wide x/y jitter produce a
trimodal, right-skewed vegetable spawn (base_y mean -0.099, std 0.16, only ~11%
within +-3cm of the original taskdemo center), which mismatches the tight
original taskdemo (base_y ~ -0.03) the policy is conditioned/evaluated on. It
also records t=0 BEFORE the object settles into view (empty-sink opening).

This generator instead:
  * anchors every spawn at the ORIGINAL taskdemo center (measured from the source
    demo in the arm-base frame) and adds only a SMALL controlled offset
    (--spawn_spread, default +-0.035 m uniform in base x,y) -> matches eval spawns,
  * WARMUP: prepends K recorded hold frames (--warmup) so the object is visible &
    settled at t=0 (fixes the empty-sink taskdemo opening),
  * keeps the proven pick retarget + blend_place-into-cabinet from augment_pnp
    (fixed fixture destination, offset decayed to 0 -> replay reproduces ~100%).

Saves in the original dataset format (same as augment_pnp). Single source demo per
run for PARALLEL subprocess generation; merge with augment_pnp.py --merge.

Run:
    MUJOCO_GL=egl uv run python robocasa/scripts/gen_pnp_cabinet.py \
        --hdf5_path .../PnPSinkToCabinet/003/demo.hdf5 --demo_idx 0 \
        --target 200 --spawn_spread 0.035 --warmup 6 \
        --out /mnt/ddn/tmp/augmented/parts/cab0.hdf5
"""
import os
import sys
import json
import argparse

import h5py
import numpy as np
from termcolor import colored

# Reuse the validated helpers + env plumbing from augment_pnp (same dir). Importing
# it also runs robocasa / play_env registration and macros.SHOW_SITES=False.
sys.path.insert(0, os.path.dirname(__file__))
from augment_pnp import (  # noqa: E402
    ARM_BLOCK,
    GRIPPER_DIM,
    controller_base_pose,
    obj_base_pose,
    detect_grasp_boundary,
    retarget_segment,
    blend_place,
    shift_xy,
    replay_record,
)
from robocasa.utils.mimicdroid_utils import make_env, reset_to, EnvArgs  # noqa: E402


def base_center_xy(env, arm, obj):
    """Object position in the arm-controller base frame (x forward, y left) — the
    frame the spawn offset is sampled in, so the spread is layout-invariant."""
    return obj_base_pose(env, arm, obj)[:2, 3].copy()


def place_controlled(env, arm, pick, container, center_xy, off_xy, hold_action, settle):
    """Force pick+container to (center_xy + off_xy) in the base frame, keeping their
    relative arrangement (both shifted by the same world delta), then settle."""
    Pb = controller_base_pose(env, arm)                    # base pose (world)
    cur = base_center_xy(env, arm, container)              # current container base xy
    tgt = center_xy + off_xy
    d_base = np.array([tgt[0] - cur[0], tgt[1] - cur[1], 0.0])
    d_world = Pb[:3, :3] @ d_base                          # base-frame delta -> world
    shift_xy(env, [pick, container], float(d_world[0]), float(d_world[1]))
    for _ in range(settle):
        env.step(hold_action)
    env.sim.forward()


def generate(args):
    with h5py.File(args.hdf5_path, "r") as f:
        eps = sorted(f["data"].keys())
        ep = f["data"][eps[args.demo_idx]]
        actions = ep["actions"][:]
        ep_meta = json.loads(ep.attrs["ep_meta"])
        env_args_str = f["data"].attrs["env_args"]
        src = dict(model=ep.attrs["model_file"], states0=ep["states"][0],
                   non_robot=ep.attrs["non_robot_qpos_idx"], qpos=ep.attrs["initial_qpos"],
                   ep_meta_info=ep.attrs.get("ep_meta_info", ""))
        src_name = eps[args.demo_idx]
        orig_obs_keys = set(ep["observations"].keys()) if "observations" in ep else set(ep["obs"].keys())

    arm = args.active_arm
    grasp = detect_grasp_boundary(actions, arm)
    hold0 = actions[0]

    env, _ = make_env(file_name=args.hdf5_path,
                      env_args=EnvArgs(robots=args.robots, render=False, control_freq=20,
                                       controller="WHOLE_BODY_MINK_IK" if "GR1" in args.robots else None,
                                       use_camera_obs=True,
                                       camera_segmentations=None if args.no_segmentation else "instance"))
    set_meta = env.set_ep_meta if hasattr(env, "set_ep_meta") else env.set_attrs_from_ep_meta
    styles = args.styles or [f"{i:03d}_l1" for i in range(40)]

    # obs keys to store (mirror the original format minus bbox; drop seg if requested)
    bbox_keys = {k for k in orig_obs_keys if "bbox" in k}
    seg_keys = {k for k in orig_obs_keys if "segmentation" in k} if args.no_segmentation else set()
    computed_keys = {"object", "objects-joint-state"}
    env_keys = (orig_obs_keys - bbox_keys - seg_keys) - computed_keys

    # ANCHOR: the original spawn center, measured from the source scene in the base
    # frame (obj_base_pose after reset_to source). This is the tight taskdemo center
    # the policy is conditioned/evaluated on; controlled spread is added around it.
    init = {"states": src["states0"], "ep_meta": ep_meta, "model": src["model"],
            "non_robot_qpos_idx": src["non_robot"], "qpos_state": src["qpos"]}
    reset_to(env, init, replace_robot_joints=True, change_to_gr1="GR1" in args.robots)
    Hps = obj_base_pose(env, arm, args.pick_object)              # source pick pose (base)
    center_xy = base_center_xy(env, arm, args.place_object)      # source container center (base)
    print(colored(f"[{src_name}] spawn center (base xy) = {center_xy.round(4)}  spread=+-{args.spawn_spread}", "cyan"), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = h5py.File(args.out, "w")
    od = out.create_group("data")
    od.attrs["env_args"] = env_args_str
    gi = 0
    att = 0
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
        # controlled spawn: uniform +-spread in base x,y around the original center
        off = rng.uniform(-args.spawn_spread, args.spawn_spread, size=2)
        try:
            place_controlled(env, arm, args.pick_object, args.place_object,
                             center_xy, off, hold0, args.settle)
        except Exception:
            continue
        Hpt = obj_base_pose(env, arm, args.pick_object)
        model_xml = env.sim.model.get_xml()
        # pick retargets to the (controlled) new object pose; place blends smoothly
        # toward the FIXED cabinet destination (offset from new grasp decayed to 0).
        rt = retarget_segment(actions, range(0, grasp), Hps, Hpt, arm)
        traj = blend_place(rt, actions, grasp, arm)
        # WARMUP: prepend K recorded hold frames so t=0 shows the settled, visible
        # object (the taskdemo opening [0:32] must not be an empty sink).
        if args.warmup > 0:
            traj = np.concatenate([np.repeat(traj[:1], args.warmup, axis=0), traj], axis=0)
        ok, states, obs_list = replay_record(env, traj, capture_obs=True)
        if ok:
            T = len(states)
            nq = env.sim.model.nq
            robot_idx = set(int(i) for i in env.robots[0]._ref_joint_pos_indexes)
            non_robot_idx = np.array(sorted(set(range(nq)) - robot_idx))
            qpos_all = np.stack([states[t][1:1 + nq] for t in range(T)])
            objects_joint_state = qpos_all[:, non_robot_idx]

            g = od.create_group(f"demo_{gi}")
            g.create_dataset("actions", data=traj.astype(np.float32))
            g.create_dataset("states", data=states)
            g.create_dataset("dones", data=np.zeros(T, np.int64))
            g.create_dataset("rewards", data=np.zeros(T, np.int64))
            og = g.create_group("obs")
            for key in env_keys:
                if key in obs_list[0]:
                    arr = np.stack([np.asarray(o[key]) for o in obs_list])
                    if "image" in key:      # robosuite cams are bottom-origin; store upright
                        arr = arr[:, ::-1]
                    og.create_dataset(key, data=arr)
            if "object-state" in og:
                og.create_dataset("object", data=og["object-state"][()])
            og.create_dataset("objects-joint-state", data=objects_joint_state)
            g.attrs["model_file"] = model_xml
            g.attrs["ep_meta"] = json.dumps(meta)
            g.attrs["ep_meta_info"] = src["ep_meta_info"]
            g.attrs["initial_qpos"] = qpos_all[0]
            g.attrs["non_robot_qpos_idx"] = non_robot_idx
            g.attrs["num_samples"] = T
            g.attrs["source_demo"] = src_name
            g.attrs["style_id"] = style
            g.attrs["spawn_off"] = off
            gi += 1
        if att % 25 == 0 or ok:
            print(colored(f"[{src_name}] {gi}/{args.target} ({att} att, style={style}, off={off.round(3)})",
                          "green" if ok else "yellow"), flush=True)
    od.attrs["total"] = sum(g["actions"].shape[0] for g in od.values())
    out.close()
    try:
        import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
        DatasetUtils.extract_action_dict(dataset=args.out)
    except Exception as e:
        print(colored(f"[{src_name}] action_dict add failed (ignored): {e}", "yellow"), flush=True)
    print(colored(f"[{src_name}] DONE: {gi} demos / {att} attempts -> {args.out}", "magenta"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5_path", required=True)
    ap.add_argument("--demo_idx", type=int, default=0)
    ap.add_argument("--pick_object", default="vegetable")
    ap.add_argument("--place_object", default="vegetable_container",
                    help="the container the pick object rests in at spawn (shifted together)")
    ap.add_argument("--no_segmentation", action="store_true")
    ap.add_argument("--active_arm", default="right", choices=["right", "left"])
    ap.add_argument("--robots", default="DemoTwoHand")
    ap.add_argument("--styles", nargs="+", default=None)
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--max_attempts", type=int, default=0)
    ap.add_argument("--spawn_spread", type=float, default=0.035,
                    help="+- uniform base-frame x,y offset (m) around the original taskdemo center")
    ap.add_argument("--warmup", type=int, default=6,
                    help="recorded hold frames prepended so t=0 shows the settled/visible object")
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
