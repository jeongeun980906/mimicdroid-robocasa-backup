#!/usr/bin/env python3
"""Eval-manifest GR1 refinement: v1 (frozen, Phase-1) -> v2 = D ∩ GR1-feasible.

For every scene of a frozen eval manifest, apply the GR1 gate: retarget the
tier's source demo(s) onto the scene (drawn-scene retarget, DemoTwoHand env),
then replay on GR1 (WHOLE_BODY_MINK_IK; settle + velocity-limited approach as
replay_to_gr1). A scene passes iff some source demo passes BOTH DemoTwoHand
replay (the v1 gate; its recorded cond_demo is already known-passing) AND GR1
replay. GR1-infeasible scenes are dropped and the manifest is topped back up to
the original size with NEW feasible seeds continuing the SAME eval seed stream
(seed = last logged v1 seed + 1, ...), same determinism/freeze discipline
(per-run redraw assert, state SHA-256 in the json).

v1 files are READ-ONLY. v2 written alongside as <tag>_v2.{npz,json,png}; the v2
one-shot cond_demo per scene is the demo that passed both gates.

Run (one manifest):
    MUJOCO_GL=egl .venv/bin/python robocasa/scripts/refine_eval_manifest.py \
        --env TurnOnFaucet --out_dir /mnt/ddn/dataset/gr1_eval_manifests
"""
import os
import sys
import json
import time
import argparse
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.dirname(__file__))
import augment_pnp as AP
import make_eval_manifest as MEM
import gen_unified as GU
import replay_to_gr1 as RG
from robocasa.utils.mimicdroid_utils import reset_to

MAN_DIR = "/mnt/ddn/dataset/gr1_eval_manifests"


def np_init(z, i):
    return GU.scene_init(z, i)


def rec_from_npz(z, i):
    return dict(seed=int(z["seeds"][i]), states=z["states"][i],
                initial_qpos=z["initial_qpos"][i],
                non_robot_qpos_idx=z["non_robot_qpos_idx"][i],
                object_state=z["object_state"][i],
                model_file=str(z["model_files"][i]), ep_meta=str(z["ep_metas"][i]),
                render=np.asarray(z["renders"][i]))


def retarget_on_scene(dth, spec, src, c, init, seed, settle):
    GU.seed_globals(seed)
    reset_to(dth, init, replace_robot_joints=True, change_to_gr1=False)
    GU.settle_env(dth, settle)
    return MEM.retarget_onto_drawn(dth, spec, src, c)


def replay_gr1(genv, init, traj, seed, settle, vel, rot_vel, ap_max):
    GU.seed_globals(seed)
    reset_to(genv, init, replace_robot_joints=True, change_to_gr1=True)
    obs = GU.settle_env(genv, settle)
    a_cur = RG._gr1_current_action(obs, traj[0])
    seq = np.concatenate([RG._approach_rows(a_cur, traj[0], vel, rot_vel, ap_max), traj], axis=0)
    try:
        for a in seq:
            obs, _, done, _ = genv.step(a)
            if done:
                break
        return bool(genv._check_success()), None
    except Exception as e:
        return False, f"replay_exception:{type(e).__name__}:{e}"


def gate_scene(dth, genv, spec, sources, consts, init, seed, args,
               known_dth_ok=None, order=None):
    """A scene passes iff some source demo passes DemoTwoHand replay AND GR1
    replay. Returns (passing demo key or None, per-demo log)."""
    log = []
    order = order if order is not None else list(range(len(sources)))
    for si in order:
        src, c = sources[si], consts[si]
        try:
            traj, _ = retarget_on_scene(dth, spec, src, c, init, seed, args.settle)
        except Exception as e:
            log.append(f"{src['key']}:retarget_error:{type(e).__name__}:{e}")
            continue
        if src["key"] != known_dth_ok:
            ok, states, _ = AP.replay_record(dth, traj, capture_obs=False)
            if not ok:
                log.append(f"{src['key']}:dth_success_check_false")
                continue
            # replay dirtied the scene; recompute the traj from a fresh restore
            traj, _ = retarget_on_scene(dth, spec, src, c, init, seed, args.settle)
        ok, err = replay_gr1(genv, init, traj, seed, args.settle,
                             args.approach_vel, args.approach_rot_vel, args.approach_max)
        if ok:
            log.append(f"{src['key']}:gr1_success")
            return src["key"], log
        log.append(f"{src['key']}:{err or 'gr1_success_check_false'}")
    return None, log


def refine(args):
    spec = MEM.SPECS[args.env]
    tag = f"{args.env}__{spec['tier']}"
    v1_json = json.load(open(os.path.join(MAN_DIR, f"{tag}.json")))
    z = np.load(os.path.join(MAN_DIR, f"{tag}.npz"), allow_pickle=True)
    need = args.need or len(z["seeds"])
    dth = MEM.build_env(spec)                      # DemoTwoHand (retarget + v1 gate)
    genv = GU.build_env2(spec, robots="GR1TwoHand", use_camera_obs=False,
                         seg=None, controller="WHOLE_BODY_MINK_IK")
    sources = MEM.load_sources(spec)
    consts = [MEM.src_constants(dth, spec, s) for s in sources]
    key2idx = {s["key"]: i for i, s in enumerate(sources)}
    template = v1_json["template"]
    print(colored(f"[refine {tag}] v1 scenes={len(z['seeds'])} need={need} "
                  f"template={template}", "cyan"), flush=True)

    kept, dropped, log_all = [], [], {}
    t0 = time.time()
    n_scan = len(z["seeds"]) if args.limit <= 0 else min(args.limit, len(z["seeds"]))
    for i in range(n_scan):
        seed = int(z["seeds"][i])
        init = np_init(z, i)
        cond = str(z["cond_demo_keys"][i]) if "cond_demo_keys" in z else None
        order = ([key2idx[cond]] + [j for j in range(len(sources)) if j != key2idx[cond]]) \
            if cond in key2idx else None
        demo, log = gate_scene(dth, genv, spec, sources, consts, init, seed, args,
                               known_dth_ok=cond, order=order)
        log_all[str(seed)] = log
        rec = rec_from_npz(z, i)
        if demo is not None:
            rec["cond_demo_key"] = demo
            rec["origin"] = "v1"
            kept.append(rec)
        else:
            dropped.append(seed)
        print(colored(f"[refine {tag}] v1 seed {seed}: "
                      f"{'GR1-FEASIBLE via ' + demo if demo else 'dropped'} "
                      f"({len(kept)} kept, {(time.time() - t0) / 60:.1f}m)",
                      "green" if demo else "yellow"), flush=True)
    v1_rate = f"{len(kept)}/{n_scan}"

    # ---- top-up: continue the SAME eval seed stream past the last logged seed
    last = max(int(e["seed"]) for e in v1_json["feasibility_log"])
    seed = last + 1
    topup_log, verified = [], False
    while len(kept) < need and (seed - last) <= args.max_topup_draws:
        cur = seed
        seed += 1
        try:
            with MEM.draw_deadline(600):
                _, rec = MEM.seeded_draw(dth, json.loads(json.dumps(template)), cur)
        except Exception as e:
            topup_log.append(dict(seed=cur, ok=False, reason=f"reset_error:{type(e).__name__}:{e}"))
            continue
        if not verified:            # freeze discipline: per-run determinism assert
            _, rec2 = MEM.seeded_draw(dth, json.loads(json.dumps(template)), cur)
            assert rec2["state_sha"] == rec["state_sha"], f"NON-DETERMINISTIC draw at seed {cur}"
            verified = True
            _, rec = MEM.seeded_draw(dth, json.loads(json.dumps(template)), cur)
        rec["extras"] = MEM.task_extras(dth, spec)
        init = {"states": rec["states"], "ep_meta": rec["ep_meta"], "model": rec["model_file"],
                "non_robot_qpos_idx": rec["non_robot_qpos_idx"], "qpos_state": rec["initial_qpos"]}
        demo, log = gate_scene(dth, genv, spec, sources, consts, init, cur, args)
        topup_log.append(dict(seed=cur, ok=demo is not None, log=log))
        if demo is not None:
            # settled render (discipline: never render the raw post-reset frame)
            GU.seed_globals(cur)
            reset_to(dth, init, replace_robot_joints=True, change_to_gr1=False)
            GU.settle_env(dth, args.settle)
            rec["render"] = MEM.render_center(dth)
            rec["cond_demo_key"] = demo
            rec["origin"] = "topup"
            kept.append(rec)
        print(colored(f"[refine {tag}] topup seed {cur}: "
                      f"{'GR1-FEASIBLE via ' + demo if demo else 'infeasible'} "
                      f"({len(kept)}/{need}, {(time.time() - t0) / 60:.1f}m)",
                      "green" if demo else "yellow"), flush=True)

    # ---- write v2 (alongside; v1 untouched)
    v2 = f"{tag}_v2"
    def _obj(key):
        a = np.empty(len(kept), object)
        for i, s in enumerate(kept):
            a[i] = s[key]
        return a
    np.savez_compressed(
        os.path.join(args.out_dir, f"{v2}.npz"),
        seeds=np.array([s["seed"] for s in kept], np.int64),
        states=_obj("states"), initial_qpos=_obj("initial_qpos"),
        non_robot_qpos_idx=_obj("non_robot_qpos_idx"), object_state=_obj("object_state"),
        renders=np.stack([np.asarray(s["render"]) for s in kept]) if kept else np.zeros((0,)),
        model_files=np.array([s["model_file"] for s in kept], object),
        ep_metas=np.array([s["ep_meta"] for s in kept], object),
        cond_demo_keys=np.array([s.get("cond_demo_key", "") for s in kept], object))
    sha = {str(s["seed"]): (v1_json["state_shas"].get(str(s["seed"]))
                            or __import__("hashlib").sha256(np.asarray(s["states"]).tobytes()).hexdigest())
           for s in kept}
    meta = dict(
        env=spec["env"], tier=spec["tier"], base_task=spec["base"], tag=v2,
        derived_from=f"{tag}.json (frozen v1; untouched)",
        gate="v1 DemoTwoHand feasibility AND GR1 replay success (settle + velocity-limited approach)",
        gr1_gate_params=dict(settle=args.settle, approach_vel=args.approach_vel,
                             approach_rot_vel=args.approach_rot_vel, approach_max=args.approach_max),
        template=template, n_scenes=len(kept),
        seeds=[s["seed"] for s in kept],
        origin={str(s["seed"]): s["origin"] for s in kept},
        state_shas=sha,
        cond_demo={str(s["seed"]): dict(hdf5=MEM.demo_hdf5(spec["demo_dir"]),
                                        key=s.get("cond_demo_key")) for s in kept},
        v1_gr1_feasibility_rate=v1_rate,
        v1_dropped_seeds=dropped,
        v1_gate_log=log_all,
        topup_log=topup_log,
        note=("v2 = D ∩ GR1-feasible: the GR1 gate applied identically to train "
              "generation (gen_unified) and eval scenes. cond_demo = the demo that "
              "passed BOTH gates on that exact scene (one-shot eval input)."))
    json.dump(meta, open(os.path.join(args.out_dir, f"{v2}.json"), "w"), indent=1)
    MEM.contact_sheet(os.path.join(args.out_dir, f"{v2}_contact.png"), v2, kept, sources, spec)
    print(colored(f"[refine {tag}] v1 GR1-feasibility {v1_rate}; v2 = {len(kept)} scenes "
                  f"({sum(1 for s in kept if s['origin'] == 'topup')} topup) -> "
                  f"{args.out_dir}/{v2}.npz", "magenta"), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=sorted(MEM.SPECS))
    ap.add_argument("--out_dir", default=MAN_DIR)
    ap.add_argument("--need", type=int, default=0, help="v2 size (0 = same as v1)")
    ap.add_argument("--limit", type=int, default=0, help="only gate first N v1 scenes (smoke)")
    ap.add_argument("--max_topup_draws", type=int, default=400)
    ap.add_argument("--settle", type=int, default=3)
    ap.add_argument("--approach_vel", type=float, default=0.01)
    ap.add_argument("--approach_rot_vel", type=float, default=0.06)
    ap.add_argument("--approach_max", type=int, default=60)
    args = ap.parse_args()
    try:
        refine(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
