#!/usr/bin/env python3
"""
Build a GR1 dataset by replaying already-augmented DemoTwoHand demos on GR1 and
KEEPING ONLY the ones GR1 succeeds at (the success check = reachability filter).

We do NOT retarget here -- the augmented actions are absolute eef pose targets,
GR1's WHOLE_BODY_MINK_IK follows them. ~50% transfer (GR1 reachability). For each
GR1-successful replay we capture GR1's own states + obs (upright images) + the
computed object low-dim, saved in the same format as the source augment.

Run (parallel over demo ranges):
    MUJOCO_GL=egl uv run python robocasa/scripts/replay_to_gr1.py \
        --src .../TaskDemos/PnPSinkToRightCounterPlate/003/demo_im128_notp.hdf5 \
        --aug /mnt/ddn/tmp/augmented/PnPSinkToRightCounterPlate_aug_obs.hdf5 \
        --start 0 --end 200 --out parts_gr1/rcp0.hdf5
"""
import os
import sys
import json
import argparse

import h5py
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.dirname(__file__))
import augment_pnp as AP  # merge() reused
import robosuite.utils.transform_utils as T
from robocasa.utils.mimicdroid_utils import make_env, reset_to, EnvArgs

_ARMB = {"right": 0, "left": 6}   # action pose block per arm: [Rpos3,Raa3 | Lpos3,Laa3 | grip12,grip13]


def _gr1_current_action(obs, a0):
    """GR1's CURRENT both-arm eef pose as a 14-dim action row (base frame, absolute,
    axis-angle ori — the WHOLE_BODY_MINK_IK action layout). Grips are held at ``a0``'s
    (pre-grasp/open). Used as the start of the home->demo approach bridge."""
    row = np.asarray(a0, dtype=np.float64).copy()
    for arm, key in (("right", "robot0_base_to_right_eef"), ("left", "robot0_base_to_left_eef")):
        b = _ARMB[arm]
        row[b:b + 3] = np.asarray(obs[key + "_pos"], float)
        # Use the SITE quat: it is the frame the WHOLE_BODY_MINK_IK action targets. The plain
        # body-eef quat is ~180deg flipped from it, so reading it made the approach demand a
        # bogus ~175deg wrist rotation (fly-away) instead of the true ~18deg reach.
        row[b + 3:b + 6] = T.quat2axisangle(np.asarray(obs[key + "_quat_site"], float))  # xyzw -> aa
    return row


def _approach_rows(a_cur, a0, vel_pos, vel_rot, max_steps):
    """VELOCITY-LIMITED GR1 home -> demo-start bridge. Picks step count K so no arm's
    per-step eef translation exceeds ``vel_pos`` (m) nor rotation ``vel_rot`` (rad), then
    lerp(pos)/slerp(rot) for BOTH arms over K steps (grips held at ``a0``). Recorded as
    leading actions so the GR1 demo eases in from its own home at a CONTROLLED SPEED (no
    step-0 IK jump), regardless of how far the reset pose is from the demo start."""
    dmax, rmax = 0.0, 0.0
    for arm in ("right", "left"):
        b = _ARMB[arm]
        dmax = max(dmax, float(np.linalg.norm(np.asarray(a0[b:b + 3]) - np.asarray(a_cur[b:b + 3]))))
        q0, q1 = T.axisangle2quat(a_cur[b + 3:b + 6]), T.axisangle2quat(a0[b + 3:b + 6])
        q0 = q0 / (np.linalg.norm(q0) + 1e-9); q1 = q1 / (np.linalg.norm(q1) + 1e-9)
        rmax = max(rmax, 2.0 * float(np.arccos(min(1.0, abs(float(np.dot(q0, q1)))))))
    K = int(np.ceil(max(dmax / max(vel_pos, 1e-6), rmax / max(vel_rot, 1e-6))))
    K = max(0, min(int(max_steps), K))
    if K <= 0:
        return np.zeros((0, len(a0)), np.float64)
    rows = []
    for k in range(1, K + 1):
        s = k / (K + 1)
        row = np.asarray(a0, dtype=np.float64).copy()
        for arm in ("right", "left"):
            b = _ARMB[arm]
            row[b:b + 3] = (1 - s) * np.asarray(a_cur[b:b + 3]) + s * np.asarray(a0[b:b + 3])
            q0, q1 = T.axisangle2quat(a_cur[b + 3:b + 6]), T.axisangle2quat(a0[b + 3:b + 6])
            row[b + 3:b + 6] = T.quat2axisangle(T.quat_slerp(q0, q1, s))
        rows.append(row)
    return np.asarray(rows)


def bbox_from_seg(seg, max_inst=32):
    """[x1,y1,x2,y2] bbox per instance from an (H,W,1) instance segmentation,
    matching the TaskDemos bbox/bbox_ids/bbox_mask format (padded to max_inst)."""
    s = seg[..., 0] if seg.ndim == 3 else seg
    ids = [i for i in np.unique(s) if i != 0][:max_inst]
    bbox = np.zeros((max_inst, 4), np.float64)
    bids = np.zeros((max_inst,), np.int64)
    bmask = np.zeros((max_inst,), np.int32)
    for i, vid in enumerate(ids):
        ys, xs = np.where(s == vid)
        bbox[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
        bids[i] = vid
        bmask[i] = 1
    return bbox, bids, bmask


def convert(args):
    with h5py.File(args.aug, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        ea = f["data"].attrs["env_args"]
        ea = json.loads(ea) if isinstance(ea, str) else ea
    ea.setdefault("env_kwargs", {})["robots"] = "GR1TwoHand"  # GR1 dataset env_args
    gr1_env_args = json.dumps(ea)
    with h5py.File(args.src, "r") as f:
        s_eps = sorted(f["data"].keys())
        orig_obs_keys = set(f["data"][s_eps[0]]["obs"].keys())
    bbox_keys = {k for k in orig_obs_keys if "bbox" in k}      # computed from segmentation
    seg_keys = sorted(k for k in orig_obs_keys if "segmentation" in k)
    cams = [k.replace("_segmentation_instance", "") for k in seg_keys]
    # everything the env produces directly (images, segmentation, per-object, robot,
    # object-state); object + objects-joint-state + bbox are filled in by us.
    env_keys = orig_obs_keys - bbox_keys - {"object", "objects-joint-state"}

    env, _ = make_env(file_name=args.src,
                      env_args=EnvArgs(robots="GR1TwoHand", render=False, control_freq=20,
                                       controller="WHOLE_BODY_MINK_IK", use_camera_obs=True,
                                       camera_segmentations="instance"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = h5py.File(args.out, "w")
    od = out.create_group("data")
    od.attrs["env_args"] = gr1_env_args
    gi = 0
    aug = h5py.File(args.aug, "r")["data"]

    # Diversity-maximizing replay order: round-robin across styles (so successive
    # replays differ in style), and within each style spread by object x-position.
    # Replaying in this order and keeping GR1-successes until --target makes the
    # kept set cover many styles + positions even though only ~25-50% transfer.
    from collections import defaultdict
    bystyle = defaultdict(list)
    for di in range(len(demos)):
        bystyle[aug[demos[di]].attrs.get("style_id", "")].append(di)
    for s in bystyle:
        bystyle[s].sort(key=lambda di: float(np.asarray(aug[demos[di]]["obs"]["vegetable_pos"][0])[0])
                        if "vegetable_pos" in aug[demos[di]]["obs"] else di)
    styles_sorted = sorted(bystyle.keys())
    order = []
    for i in range(max(len(v) for v in bystyle.values())):
        for s in styles_sorted:
            if i < len(bystyle[s]):
                order.append(bystyle[s][i])
    order = order[args.start:args.end]   # disjoint demo-range shard for parallel replay workers
    target = args.target or len(order)
    # Even scene-design coverage: cap kept demos per style_id so the kept set spreads across
    # styles instead of piling onto whichever styles transfer best. Auto = ceil(target/nstyles).
    nstyles = max(1, len(bystyle))
    per_style_max = args.per_style_max if args.per_style_max > 0 else int(np.ceil(target / nstyles))
    kept_style = defaultdict(int)
    tried = 0
    for di in order:
        if gi >= target:
            break
        st = aug[demos[di]].attrs.get("style_id", "")
        if args.exclude_style and st in set(args.exclude_style):
            continue                          # reserved for a separate eval-style pass
        if kept_style[st] >= per_style_max:   # this style already has its even share
            continue
        tried += 1
        g = aug[demos[di]]
        actions = g["actions"][:]
        init = {"states": g["states"][0], "ep_meta": json.loads(g.attrs["ep_meta"]),
                "model": g.attrs["model_file"], "non_robot_qpos_idx": g.attrs["non_robot_qpos_idx"],
                "qpos_state": g.attrs["initial_qpos"]}
        try:
            reset_to(env, init, replace_robot_joints=True, change_to_gr1=True)
        except Exception:
            continue
        obs = env._get_observations()
        # Render-settle (NOT recorded): robosuite's FIRST post-reset frame misses the
        # dynamically-loaded objects (empty-sink t=0), and they only appear after a step.
        # Step a few times holding GR1's CURRENT eef — RE-READ every step (closed-loop) so
        # the arm doesn't diverge (a fixed target would drag it) — so the object renders
        # while GR1 stays at its reset pose. Then record starts with the object visible.
        for _ in range(args.settle):
            try:
                obs, _, _, _ = env.step(_gr1_current_action(obs, actions[0]))
            except Exception:
                break
        # Prepend a recorded home->demo-start approach so the saved GR1 demo starts at
        # GR1's own (settled) pose (no step-0 IK jump) and reaches into the visible scene.
        a_cur = _gr1_current_action(obs, actions[0])
        seq = np.concatenate([_approach_rows(a_cur, actions[0], args.approach_vel,
                                             args.approach_rot_vel, args.approach_max), actions], axis=0)
        states, obs_list = [], []
        ok_run = True
        for a in seq:
            states.append(env.sim.get_state().flatten())
            obs_list.append({k: np.asarray(v) for k, v in obs.items()})
            try:
                obs, _, done, _ = env.step(a)
            except Exception:
                ok_run = False
                break
            if done:
                break
        if not ok_run or not env._check_success():
            if tried % 10 == 0:
                print(colored(f"[{args.tag}] {gi} kept / {tried} tried", "yellow"), flush=True)
            continue
        # GR1-successful -> save GR1's own states + obs
        states = np.array(states)
        n = len(states)
        nq = env.sim.model.nq
        robot_idx = set(int(i) for i in env.robots[0]._ref_joint_pos_indexes)
        non_robot_idx = np.array(sorted(set(range(nq)) - robot_idx))
        qpos_all = np.stack([states[t][1:1 + nq] for t in range(n)])
        objects_joint_state = qpos_all[:, non_robot_idx]

        dg = od.create_group(f"demo_{gi}")
        dg.create_dataset("actions", data=seq[:n].astype(np.float32))
        dg.create_dataset("states", data=states)
        dg.create_dataset("dones", data=np.zeros(n, np.int64))
        dg.create_dataset("rewards", data=np.zeros(n, np.int64))
        og = dg.create_group("obs")
        for key in env_keys:
            if key in obs_list[0]:
                arr = np.stack([np.asarray(o[key]) for o in obs_list])
                # camera renders (image AND segmentation) are bottom-origin -> flip upright
                if "image" in key or "segmentation" in key:
                    arr = arr[:, ::-1]
                og.create_dataset(key, data=arr)
        # bbox (12 keys) computed per camera from the upright segmentation
        for cam in cams:
            sk = cam + "_segmentation_instance"
            if sk not in og:
                continue
            seg_seq = og[sk][()]
            res = [bbox_from_seg(seg_seq[t]) for t in range(n)]
            og.create_dataset(cam + "_bbox", data=np.stack([r[0] for r in res]))
            og.create_dataset(cam + "_bbox_ids", data=np.stack([r[1] for r in res]))
            og.create_dataset(cam + "_bbox_mask", data=np.stack([r[2] for r in res]))
        if "object-state" in og:
            og.create_dataset("object", data=og["object-state"][()])
        og.create_dataset("objects-joint-state", data=objects_joint_state)
        dg.attrs["model_file"] = env.sim.model.get_xml()
        dg.attrs["ep_meta"] = g.attrs["ep_meta"]
        dg.attrs["ep_meta_info"] = g.attrs.get("ep_meta_info", "")
        dg.attrs["initial_qpos"] = qpos_all[0]
        dg.attrs["non_robot_qpos_idx"] = non_robot_idx
        dg.attrs["num_samples"] = n
        dg.attrs["source_demo"] = demos[di]
        dg.attrs["style_id"] = g.attrs.get("style_id", "")
        kept_style[g.attrs.get("style_id", "")] += 1
        gi += 1
        print(colored(f"[{args.tag}] kept {gi} (demo {di}) success", "green"), flush=True)
    od.attrs["total"] = sum(d["actions"].shape[0] for d in od.values()) if gi else 0
    out.close()
    try:
        import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
        if gi:
            DatasetUtils.extract_action_dict(dataset=args.out)
    except Exception as e:
        print(colored(f"[{args.tag}] action_dict 실패(무시): {e}", "yellow"), flush=True)
    # tried accumulated above
    print(colored(f"[{args.tag}] DONE: {gi}/{tried} GR1-success kept -> {args.out}", "magenta"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="source TaskDemos hdf5 (for env setup)")
    ap.add_argument("--aug", help="augmented DemoTwoHand hdf5 to replay on GR1")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--target", type=int, default=0, help="stop after N GR1-successful demos (0=all)")
    ap.add_argument("--approach_vel", type=float, default=0.01,
                    help="velocity-limited approach: max per-step eef translation (m). Smaller = slower. "
                         "0.01 m/step @20Hz = 0.2 m/s. Step count auto-chosen from reset->start distance.")
    ap.add_argument("--approach_rot_vel", type=float, default=0.06,
                    help="velocity-limited approach: max per-step eef rotation (rad)")
    ap.add_argument("--approach_max", type=int, default=60,
                    help="cap on approach bridge steps (safety). 0 disables the approach entirely.")
    ap.add_argument("--exclude_style", nargs="+", default=None,
                    help="style_id(s) to skip (e.g. the eval style handled by a separate concentrated pass)")
    ap.add_argument("--per_style_max", type=int, default=0,
                    help="max kept demos per style_id for even scene-design coverage "
                         "(0 = auto: ceil(target / n_styles))")
    ap.add_argument("--settle", type=int, default=3,
                    help="unrecorded closed-loop hold steps after reset (re-reads GR1's current eef each step "
                         "so the arm stays put) — lets robosuite render objects the first frame misses")
    ap.add_argument("--no_segmentation", action="store_true")
    ap.add_argument("--tag", default="GR1")
    ap.add_argument("--out", required=False)
    ap.add_argument("--merge", nargs="+")
    args = ap.parse_args()
    if args.merge:
        AP.merge(args.merge, args.out)
    else:
        convert(args)


if __name__ == "__main__":
    main()
