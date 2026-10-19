#!/usr/bin/env python3
"""
Diversity augmentation for a PnP task via 2-stage object-centric retargeting.

Generates new successful demos from ONE source demo by:
  - env.reset() -> stable randomized object placement (same instances),
  - optional x-jitter: shift pick object + its container together by a small x
    offset and let them settle (adds x-position diversity that the placement
    sampler can't, since the sink basin caps the sampling region in x),
  - style_id cycling for visual diversity,
  - 2-stage object-centric retarget (pick->pick_object, place->place_object),
  - replay + keep ONLY successes (env._check_success()).

Saves in the ORIGINAL dataset format: data.attrs['env_args'/'total'] and per-demo
groups with states/actions/dones/rewards + attrs (ep_meta, model_file,
num_samples, initial_qpos, non_robot_qpos_idx). obs images / action_dict are
omitted (regenerable from states via dataset_states_to_obs).

Single-demo per run so multiple source demos can be generated in PARALLEL
(subprocess) then merged with --merge.

Run one demo:
    MUJOCO_GL=egl uv run python robocasa/scripts/augment_pnp.py \
        --hdf5_path .../demo_im128_notp.hdf5 --demo_idx 0 --target 200 \
        --x_jitter 0.05 --out /mnt/ddn/tmp/augmented/parts/demo0.hdf5
Merge:
    ... augment_pnp.py --merge a.hdf5 b.hdf5 c.hdf5 --out final.hdf5
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

from robocasa.utils.mimicdroid_utils import make_env, reset_to, EnvArgs

macros.SHOW_SITES = False

ARM_BLOCK = {"right": 0, "left": 6}
GRIPPER_DIM = {"right": 13, "left": 12}


def body_pose(env, name):
    i = env.sim.model.body_name2id(name)
    return T.make_pose(env.sim.data.body_xpos[i], env.sim.data.body_xmat[i].reshape(3, 3))


def controller_base_pose(env, arm):
    p, r = env.robots[0].composite_controller.get_controller_base_pose(arm)
    return T.make_pose(np.array(p), np.array(r).reshape(3, 3))


def obj_base_pose(env, arm, obj):
    return np.linalg.inv(controller_base_pose(env, arm)) @ body_pose(env, env.objects[obj].root_body)


def a6_to_pose(a):
    return T.make_pose(a[:3], T.quat2mat(T.axisangle2quat(a[3:6])))


def pose_to_a6(P):
    return np.concatenate([P[:3, 3], T.quat2axisangle(T.mat2quat(P[:3, :3]))])


def detect_grasp_boundary(actions, arm):
    g = actions[:, GRIPPER_DIM[arm]]
    for t in range(1, len(g)):
        if g[t - 1] <= 0 and g[t] > 0:
            return t
    return len(actions) // 2


def retarget_segment(actions, idx, Hs, Ht, arm):
    out = actions.copy()
    b = ARM_BLOCK[arm]
    M = Ht @ np.linalg.inv(Hs)
    for t in idx:
        out[t][b:b + 6] = pose_to_a6(M @ a6_to_pose(actions[t][b:b + 6]))
    return out


def slerp_bridge(a0, a1, n, arm, gval, tmpl):
    b = ARM_BLOCK[arm]
    P0, P1 = a6_to_pose(a0[b:b + 6]), a6_to_pose(a1[b:b + 6])
    q0, q1 = T.mat2quat(P0[:3, :3]), T.mat2quat(P1[:3, :3])
    rows = []
    for k in range(1, n + 1):
        s = k / (n + 1)
        row = tmpl.copy()
        row[b:b + 6] = np.concatenate([(1 - s) * P0[:3, 3] + s * P1[:3, 3],
                                       T.quat2axisangle(T.quat_slerp(q0, q1, s))])
        row[GRIPPER_DIM[arm]] = gval
        rows.append(row)
    return np.array(rows)


def blend_pregrasp(rt, grasp, arm, a_cur):
    """v6 mirror of blend_place for the PRE-GRASP phase: anchor t0 at the robot's
    FIXED settled pose (q0) instead of the object-shifted demo start.

    Offset = (q0 eef) - (retargeted pre-grasp start eef), applied in full at t=0
    and DECAYED linearly to zero by the grasp frame: the approach starts exactly
    at q0, keeps the demo's approach shape/timing, and still arrives at the
    retargeted grasp pose on the demo's grasp frame. (v3/v4/v5 rigid-shifted the
    whole segment, which moved t0 with the object — the defect this replaces.)"""
    b = ARM_BLOCK[arm]
    Pq0 = a6_to_pose(np.asarray(a_cur, np.float64)[b:b + 6])   # fixed settled start
    P0 = a6_to_pose(rt[0][b:b + 6])                            # retargeted demo start
    dpos = Pq0[:3, 3] - P0[:3, 3]
    q_rel = T.mat2quat(Pq0[:3, :3] @ P0[:3, :3].T)
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    out = rt.copy()
    for t in range(grasp):
        w = 1.0 - t / max(grasp - 1, 1)          # 1 at t=0 -> 0 at the grasp frame
        P = a6_to_pose(rt[t][b:b + 6])
        newpos = P[:3, 3] + dpos * w
        newrot = T.quat2mat(T.quat_slerp(q_id, q_rel, w)) @ P[:3, :3]
        out[t][b:b + 6] = pose_to_a6(T.make_pose(newpos, newrot))
    return out


def blend_place(rt, actions, grasp, arm):
    """Smoothly retarget the place phase for a FIXED destination (e.g. cabinet).

    place_fixed kept the original place trajectory and bridged the gap from the
    (retargeted) new grasp pose to the original place start -- a discontinuity that
    the controller had to jump across, which made replay diverge (CAB reproduced
    only ~50%). Instead we offset the whole place by Delta = (new grasp eef) -
    (original place start eef) and DECAY that offset linearly to zero by the end:
    the place starts continuous with the new grasp and still lands the object at
    the original destination. No bridge needed. -> replay reproduces ~100%.
    """
    b = ARM_BLOCK[arm]
    Pnew = a6_to_pose(rt[grasp - 1][b:b + 6])     # end of retargeted pick (new grasp)
    Porig = a6_to_pose(actions[grasp][b:b + 6])   # original place start
    dpos = Pnew[:3, 3] - Porig[:3, 3]
    q_rel = T.mat2quat(Pnew[:3, :3] @ Porig[:3, :3].T)
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    out = rt.copy()
    Tp = len(actions) - grasp
    for t in range(grasp, len(actions)):
        w = 1.0 - (t - grasp) / max(Tp - 1, 1)    # 1 at grasp, 0 at the end
        P = a6_to_pose(actions[t][b:b + 6])
        newpos = P[:3, 3] + dpos * w
        newrot = T.quat2mat(T.quat_slerp(q_id, q_rel, w)) @ P[:3, :3]
        out[t][b:b + 6] = pose_to_a6(T.make_pose(newpos, newrot))
    return out


def shift_xy(env, objs, dx, dy):
    """Shift the given objects together by (dx, dy) in world frame (keeps the
    pick object resting on its container since both move by the same offset)."""
    for o in objs:
        jn = env.objects[o].joints[0]
        q = env.sim.data.get_joint_qpos(jn).copy()
        q[0] += dx
        q[1] += dy
        env.sim.data.set_joint_qpos(jn, q)
    env.sim.forward()


def replay_record(env, actions, capture_obs=False):
    """Step actions, recording per-frame state and (optionally) the full obs dict
    (images + low-dim) from the SAME successful rollout, so saved demos carry real
    images with no replay/reproducibility gap."""
    states = []
    obs_list = []
    obs = env._get_observations()
    for a in actions:
        states.append(env.sim.get_state().flatten())
        if capture_obs:
            obs_list.append({k: np.asarray(v) for k, v in obs.items()})
        obs, _, done, _ = env.step(a)
        if done:
            break
    return bool(env._check_success()), np.array(states), obs_list


def generate(args):
    with h5py.File(args.hdf5_path, "r") as f:
        env_args_str = f["data"].attrs["env_args"]
        eps = sorted(f["data"].keys())
        ep = f["data"][eps[args.demo_idx]]
        actions = ep["actions"][:]
        ep_meta = json.loads(ep.attrs["ep_meta"])
        src = dict(model=ep.attrs["model_file"], states0=ep["states"][0],
                   non_robot=ep.attrs["non_robot_qpos_idx"], qpos=ep.attrs["initial_qpos"],
                   ep_meta_info=ep.attrs.get("ep_meta_info", ""))
        src_name = eps[args.demo_idx]

    arm = args.active_arm
    grasp = detect_grasp_boundary(actions, arm)
    env, _ = make_env(file_name=args.hdf5_path,
                      env_args=EnvArgs(robots=args.robots, render=False, control_freq=20,
                                       controller="WHOLE_BODY_MINK_IK" if "GR1" in args.robots else None,
                                       use_camera_obs=True,
                                       camera_segmentations=None if args.no_segmentation else "instance"))
    set_meta = env.set_ep_meta if hasattr(env, "set_ep_meta") else env.set_attrs_from_ep_meta
    styles = args.styles or [f"{i:03d}_l1" for i in range(40)]

    # Match the original obs format: store the same obs keys MINUS bbox keys
    # (bbox observables don't exist in this robocasa version). 'object' is a copy
    # of 'object-state'; 'objects-joint-state' is computed from the non-robot qpos.
    with h5py.File(args.hdf5_path, "r") as f:
        orig_obs_keys = set(f["data"][eps[args.demo_idx]]["obs"].keys())
    bbox_keys = {k for k in orig_obs_keys if "bbox" in k}
    seg_keys = {k for k in orig_obs_keys if "segmentation" in k} if args.no_segmentation else set()
    store_keys = orig_obs_keys - bbox_keys - seg_keys
    computed_keys = {"object", "objects-joint-state"}  # not produced by env; we fill them
    env_keys = store_keys - computed_keys

    # source object poses
    init = {"states": src["states0"], "ep_meta": ep_meta, "model": src["model"],
            "non_robot_qpos_idx": src["non_robot"], "qpos_state": src["qpos"]}
    reset_to(env, init, replace_robot_joints=True, change_to_gr1="GR1" in args.robots)
    Hps = obj_base_pose(env, arm, args.pick_object)
    Hcs = None if args.place_fixed else obj_base_pose(env, arm, args.place_object)

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
        if args.x_jitter > 0 or args.y_jitter > 0:
            dx = float(rng.uniform(-args.x_jitter, args.x_jitter))
            dy = float(rng.uniform(-args.y_jitter, args.y_jitter))
            try:
                shift_xy(env, [args.pick_object, args.place_object], dx, dy)
                for _ in range(args.settle):
                    env.step(actions[0])
                env.sim.forward()
            except Exception:
                continue
        Hpt = obj_base_pose(env, arm, args.pick_object)
        model_xml = env.sim.model.get_xml()
        # pick segment always retargets to the (randomized) pick object
        rt = retarget_segment(actions, range(0, grasp), Hps, Hpt, arm)
        # place segment: retarget to the place object only if it moves. When the
        # destination is a FIXED fixture (e.g. cabinet in PnPSinkToCabinet), keep
        # the recorded place trajectory unchanged (--place_fixed); the bridge
        # connects the new grasp pose to the original transport path.
        if args.place_blend:
            # smooth, continuous place toward the fixed destination (no bridge)
            traj = blend_place(rt, actions, grasp, arm)
        else:
            if not args.place_fixed:
                Hct = obj_base_pose(env, arm, args.place_object)
                rt = retarget_segment(rt, range(grasp, len(actions)), Hcs, Hct, arm)
            br = slerp_bridge(rt[grasp - 1], rt[grasp], args.bridge_steps, arm,
                              actions[grasp, GRIPPER_DIM[arm]], rt[grasp].copy())
            traj = np.concatenate([rt[:grasp], br, rt[grasp:]], axis=0)
        ok, states, obs_list = replay_record(env, traj, capture_obs=True)
        if ok:
            T = len(states)
            # non-robot qpos indices for THIS (augmented) scene
            nq = env.sim.model.nq
            robot_idx = set(int(i) for i in env.robots[0]._ref_joint_pos_indexes)
            non_robot_idx = np.array(sorted(set(range(nq)) - robot_idx))
            # flattened state = [time, qpos(nq), qvel]; qpos starts at index 1
            qpos_all = np.stack([states[t][1:1 + nq] for t in range(T)])
            objects_joint_state = qpos_all[:, non_robot_idx]  # == original 'objects-joint-state'

            g = od.create_group(f"demo_{gi}")
            g.create_dataset("actions", data=traj.astype(np.float32))
            g.create_dataset("states", data=states)
            g.create_dataset("dones", data=np.zeros(T, np.int64))
            g.create_dataset("rewards", data=np.zeros(T, np.int64))
            og = g.create_group("obs")
            for key in env_keys:
                if key in obs_list[0]:
                    arr = np.stack([np.asarray(o[key]) for o in obs_list])
                    # robosuite camera obs are bottom-origin (vertically flipped);
                    # the original dataset stores them upright, so flip image obs.
                    if "image" in key:
                        arr = arr[:, ::-1]
                    og.create_dataset(key, data=arr)
            # 'object' is identical to 'object-state'
            if "object-state" in og:
                og.create_dataset("object", data=og["object-state"][()])
            og.create_dataset("objects-joint-state", data=objects_joint_state)
            g.attrs["model_file"] = model_xml
            g.attrs["ep_meta"] = json.dumps(meta)
            g.attrs["ep_meta_info"] = src["ep_meta_info"]
            g.attrs["initial_qpos"] = qpos_all[0]            # augmented scene qpos (correct)
            g.attrs["non_robot_qpos_idx"] = non_robot_idx
            g.attrs["num_samples"] = T
            g.attrs["source_demo"] = src_name
            g.attrs["style_id"] = style
            gi += 1
        if att % 25 == 0 or ok:
            print(colored(f"[{src_name}] {gi}/{args.target} ({att} att, style={style})",
                          "green" if ok else "yellow"), flush=True)
    od.attrs["total"] = sum(g["actions"].shape[0] for g in od.values())
    out.close()
    # add action_dict (rel_pos / rel_rot_6d / rel_rot_axis_angle / gripper) like the
    # original format, via the same robocasa util used by dataset_states_to_obs.
    try:
        import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
        DatasetUtils.extract_action_dict(dataset=args.out)
    except Exception as e:
        print(colored(f"[{src_name}] action_dict 추가 실패(무시): {e}", "yellow"), flush=True)
    print(colored(f"[{src_name}] DONE: {gi} demos / {att} attempts -> {args.out}", "magenta"), flush=True)


def merge(parts, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = h5py.File(out_path, "w")
    od = fout.create_group("data")
    gi = 0
    env_args = None
    total = 0
    for p in parts:
        with h5py.File(p, "r") as f:
            if env_args is None:
                env_args = f["data"].attrs["env_args"]
            for k in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1])):
                f.copy(f["data"][k], od, name=f"demo_{gi}")
                total += f["data"][k]["actions"].shape[0]
                gi += 1
    od.attrs["env_args"] = env_args
    od.attrs["total"] = total
    fout.close()
    print(colored(f"MERGED {gi} demos -> {out_path}", "magenta", attrs=["bold"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5_path")
    ap.add_argument("--demo_idx", type=int, default=0)
    ap.add_argument("--pick_object", default="vegetable")
    ap.add_argument("--place_object", default="vegetable_container")
    ap.add_argument("--place_fixed", action="store_true",
                    help="destination is a fixed fixture (cabinet): keep place trajectory unchanged, retarget pick only")
    ap.add_argument("--place_blend", action="store_true",
                    help="fixed destination, but SMOOTHLY blend the place (offset from new grasp decaying to 0) "
                         "instead of place_fixed's bridge-jump -> continuous, reproduces ~100%")
    ap.add_argument("--no_segmentation", action="store_true",
                    help="skip segmentation rendering+storage (faster/smaller; RGB+lowdim only)")
    ap.add_argument("--active_arm", default="right", choices=["right", "left"])
    ap.add_argument("--robots", default="DemoTwoHand")
    ap.add_argument("--styles", nargs="+", default=None)
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--max_attempts", type=int, default=0)
    ap.add_argument("--x_jitter", type=float, default=0.05, help="±x shift (m) of pick+container, 0=off")
    ap.add_argument("--y_jitter", type=float, default=0.04, help="±y shift (m) of pick+container, 0=off")
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--bridge_steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", nargs="+", default=None, help="merge these part files into --out")
    args = ap.parse_args()
    if args.merge:
        merge(args.merge, args.out)
    else:
        generate(args)


if __name__ == "__main__":
    main()
