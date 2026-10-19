#!/usr/bin/env python3
"""Phase 1 of the GR1 data/eval redesign (docs/gr1_data_eval_redesign.md):
frozen 3-tier eval manifests + layout-task feasibility matrix.

For a given task+tier this draws N seeded scenes via NATURAL ``env.reset()``
(ep_meta template = {layout_id, style_id} only; the tier env class' mixins and
``_reset_internal`` supply the rest of the scene law — cab_sink door 0.9-1.0
open, faucet off, natural placement_initializer sampling, natural object
instance sampling). NO ``reset_to``, NO artificial hinge/spawn overrides.

Determinism: before each draw we re-seed ``env.rng`` (the ONLY rng the robocasa
scene law consumes: layout/style choice, KitchenArena, SceneSampler placement,
object instance sampling, door-state draw) plus the global ``np.random`` /
``random`` (robosuite robot init noise uses the np global). seed -> bitwise
identical flattened sim state (verified per run on the first accepted seed and
re-verifiable via --mode verify).

FEASIBILITY gate (Phase-1, DemoTwoHand): the task's source teleop demo is
retargeted onto the DRAWN scene — reading the drawn hinge/handle qpos
(door/faucet; this replaces augment_door's sampled th_new0) or the drawn object
poses (pnp; reusing augment_pnp/gen_pnp_cabinet retarget fns) — and replayed;
the scene is kept iff ``env._check_success()``. A scene is tried against each
of the task's source demos in order (redrawing the seed in between — cheap and
exact thanks to determinism) and is feasible on the first success; that demo is
recorded as the scene's one-shot conditioning demo (L2/L3 eval protocol).

Modes:
  manifest  --env <ENV> --tier seen|L2|L3   -> frozen manifest npz/json + contact sheet
  matrix    --layouts 0 2 4 7 10            -> layout-task feasibility probe (json)
  coverage  --env <ENV> --n 100             -> train-side seed-stream t0 draws (npz)
  verify    --manifest <json>               -> redraw manifest seeds, compare state hashes

Run (mimicdroid-robocasa venv, EGL):
    MUJOCO_GL=egl uv run python robocasa/scripts/make_eval_manifest.py \
        --mode manifest --env TurnOnFaucet --tier seen --need 20 \
        --out_dir /mnt/ddn/dataset/gr1_eval_manifests
"""
import os
import sys
import json
import time
import random
import hashlib
import argparse
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.dirname(__file__))
import augment_pnp as AP                      # retarget helpers (validated)
from augment_door import Rax, retarget_door   # hinge retarget (validated)
import robosuite.utils.transform_utils as T

import robocasa
from robocasa.environments.kitchen.play_env.play_env import (
    PnPSinkToRightCounterPlateL2, PnPSinkToCabinetL2, TurnOnFaucetL3)
from robocasa.utils.mimicdroid_utils import make_env, reset_to, EnvArgs

TD = "/mnt/ddn/dataset/MimicDroidDataset/TaskDemos"
OUT_DEFAULT = "/mnt/ddn/dataset/gr1_eval_manifests"
EVAL_SEED0 = 1000       # frozen eval seed stream
TRAIN_SEED0 = 100000    # train-side stream (coverage proof; Phase 2 uses this side)
MATRIX_SEED0 = 7000     # layout-task probe stream


# --- L2 pnp eval classes -----------------------------------------------------
# The stock L2 classes realize "unseen objects" only via a set_ep_meta patch of a
# PRE-RECORDED ep_meta's object_cfgs. Natural draws carry no object_cfgs, so we
# apply the SAME patch (vegetable -> vegetable_set_test) in the natural
# _get_obj_cfgs path. Auto-registered with robosuite via its metaclass.
class PnPSinkToRightCounterPlateL2Eval(PnPSinkToRightCounterPlateL2):
    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        for c in cfgs:
            if c["name"] == "vegetable":
                c["obj_groups"] = "vegetable_set_test"
        return cfgs


class PnPSinkToCabinetL2Eval(PnPSinkToCabinetL2):
    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        for c in cfgs:
            if c["name"] == "vegetable":
                c["obj_groups"] = "vegetable_set_test"
        return cfgs


# LAYOUT-10 OBJECT LAW: layout 10 (= layout 7 + countertop microwave) cannot place
# the fruit/bread containers — their `left_right`-of-sink counter regions are taken
# by the microwave, so the natural draw loops forever in kitchen._load_model's
# placement retry (measured: fruit_container/bread_container RandomizationError on
# every rebuild). The env itself encodes this law in PnPSinkToMicrowaveTopL3
# (removes fruit) and in the TurnOnFaucetL3 TASKDEMOS, whose recorded scenes hold
# only [vegetable_container, packed_food, vegetable, meat] — no fruit, no bread.
# The stock TurnOnFaucetL3 class simply never got the natural-path filter; add it.
LAYOUT10_REMOVE = ("fruit", "fruit_container", "bread", "bread_container")


class TurnOnFaucetL3Eval(TurnOnFaucetL3):
    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        return [c for c in cfgs if c["name"] not in LAYOUT10_REMOVE]


for _n in ("PnPSinkToRightCounterPlateL2Eval", "PnPSinkToCabinetL2Eval",
           "TurnOnFaucetL3Eval"):
    robocasa.PLAY_TASK_NAME_TO_CLASS[_n] = _n  # make_env(task_name=...) lookup


# --- task registry -----------------------------------------------------------
# kind: "hinge" (faucet/door: rotate eef targets about the drawn joint),
#       "pnp2"  (pick+place both object-centric, slerp bridge),
#       "pnpfix"(pick object-centric, place blended toward a FIXED fixture).
# base_task: the seen-task this tier entry evaluates. demo_dir: TaskDemos subdir.
SPECS = {
    # --- seen (canonical layout+style; template read from the taskdemo) ---
    "TurnOnFaucet":         dict(tier="seen", base="TurnOnFaucet", env="TurnOnFaucet",
                                 demo_dir="TurnOnFaucet", kind="hinge", arm="right",
                                 joint_pat="handle_joint"),
    "PnPSinkToRightCounterPlate": dict(tier="seen", base="PnPSinkToRightCounterPlate",
                                 env="PnPSinkToRightCounterPlate",
                                 demo_dir="PnPSinkToRightCounterPlate", kind="pnp2",
                                 arm="right", pick="vegetable", place="fruit_container"),
    "PnPSinkToCabinet":     dict(tier="seen", base="PnPSinkToCabinet", env="PnPSinkToCabinet",
                                 demo_dir="PnPSinkToCabinet", kind="pnpfix", arm="right",
                                 pick="vegetable", dest="cab_sink"),
    "CloseLeftCabinetDoor": dict(tier="seen", base="CloseLeftCabinetDoor",
                                 env="CloseLeftCabinetDoor", demo_dir="CloseLeftCabinetDoor",
                                 kind="hinge", arm="left", joint_pat="leftdoorhinge"),
    # --- L2 (unseen style / objects; canonical layout) ---
    "PnPSinkToRightCounterPlateL2": dict(tier="L2", base="PnPSinkToRightCounterPlate",
                                 env="PnPSinkToRightCounterPlateL2Eval",
                                 demo_dir="PnPSinkToRightCounterPlateL2", kind="pnp2",
                                 arm="right", pick="vegetable", place="fruit_container"),
    "PnPSinkToCabinetL2":   dict(tier="L2", base="PnPSinkToCabinet",
                                 env="PnPSinkToCabinetL2Eval", demo_dir="PnPSinkToCabinetL2",
                                 kind="pnpfix", arm="right", pick="vegetable", dest="cab_sink"),
    "CloseLeftCabinetDoorL2": dict(tier="L2", base="CloseLeftCabinetDoor",
                                 env="CloseLeftCabinetDoorL2", demo_dir="CloseLeftCabinetDoorL2",
                                 kind="hinge", arm="left", joint_pat="leftdoorhinge"),
    # NOTE: TurnOnFaucetL2 has NO env class and NO teleop demos -> gap (see report).
    # --- L3 (held-out layout 7/10 + 014_l3 + test pools) ---
    "TurnOnFaucetL3":       dict(tier="L3", base="TurnOnFaucet", env="TurnOnFaucetL3Eval",
                                 demo_dir="TurnOnFaucetL3", kind="hinge", arm="right",
                                 joint_pat="handle_joint"),
    "PnPSinkToRightCounterPlateL3": dict(tier="L3", base="PnPSinkToRightCounterPlate",
                                 env="PnPSinkToRightCounterPlateL3",
                                 demo_dir="PnPSinkToRightCounterPlateL3", kind="pnp2",
                                 arm="right", pick="vegetable", place="fruit_container"),
    "CloseLeftCabinetDoorL3": dict(tier="L3", base="CloseLeftCabinetDoor",
                                 env="CloseLeftCabinetDoorL3", demo_dir="CloseLeftCabinetDoorL3",
                                 kind="hinge", arm="left", joint_pat="leftdoorhinge"),
    # PnPSinkToCabinet has NO L3 class; PnPSinkToMicrowaveTopL3 (layout 10, veg ->
    # microwave top, cab_sink starts CLOSED) is its pnp-to-fixed-fixture L3 analog.
    "PnPSinkToMicrowaveTopL3": dict(tier="L3", base="PnPSinkToCabinet",
                                 env="PnPSinkToMicrowaveTopL3", demo_dir="PnPSinkToMicrowaveTopL3",
                                 kind="pnpfix", arm="right", pick="vegetable",
                                 dest="microwave"),
}
SEEN_TASKS = [k for k, v in SPECS.items() if v["tier"] == "seen"]


# --- env plumbing ------------------------------------------------------------

def demo_hdf5(demo_dir):
    return f"{TD}/{demo_dir}/003/demo_im128_notp.hdf5"


def build_env(spec):
    env, _ = make_env(demo_hdf5(spec["demo_dir"]), EnvArgs(
        robots="DemoTwoHand", control_freq=20, use_camera_obs=False,
        camera_segmentations=None, task_name=spec["env"]))
    return env


def template_from_taskdemo(spec, layout_override=None, style_override=None):
    """ep_meta template = {layout_id, style_id} of the tier's own taskdemo (the
    mixins re-enforce these anyway). Everything else is drawn naturally."""
    with h5py.File(demo_hdf5(spec["demo_dir"]), "r") as f:
        k0 = sorted(f["data"].keys())[0]
        m = json.loads(f["data"][k0].attrs["ep_meta"])
    return {"layout_id": int(layout_override if layout_override is not None else m["layout_id"]),
            "style_id": str(style_override if style_override is not None else m["style_id"])}


def seeded_draw(env, template, seed):
    """NATURAL seeded reset. Returns (obs, scene_record)."""
    env.rng = np.random.default_rng(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    env.set_ep_meta(json.loads(json.dumps(template)))   # fresh copy each draw
    obs = env.reset()
    st = env.sim.get_state().flatten()
    nq = env.sim.model.nq
    robot_idx = set(int(i) for i in env.robots[0]._ref_joint_pos_indexes)
    non_robot_idx = np.array(sorted(set(range(nq)) - robot_idx))
    rec = dict(
        seed=int(seed),
        states=st,
        initial_qpos=st[1:1 + nq].copy(),
        non_robot_qpos_idx=non_robot_idx,
        model_file=env.sim.model.get_xml(),
        ep_meta=json.dumps(env.get_ep_meta()),
        object_state=np.asarray(obs["object-state"], np.float64).copy(),
        state_sha=hashlib.sha256(st.tobytes()).hexdigest(),
    )
    return obs, rec


def render_center(env, size=256):
    return np.ascontiguousarray(
        env.sim.render(height=size, width=size, camera_name="robot0_agentview_center")[::-1])


# --- source-demo constants (read ONCE per source demo) -----------------------

def load_sources(spec):
    """All teleop demos of the tier's taskdemo hdf5, with raw actions + init."""
    out = []
    with h5py.File(demo_hdf5(spec["demo_dir"]), "r") as f:
        for k in sorted(f["data"].keys()):
            ep = f["data"][k]
            out.append(dict(
                key=k, actions=ep["actions"][:], states=ep["states"][:],
                init={"states": ep["states"][0], "ep_meta": json.loads(ep.attrs["ep_meta"]),
                      "model": ep.attrs["model_file"],
                      "non_robot_qpos_idx": ep.attrs["non_robot_qpos_idx"],
                      "qpos_state": ep.attrs["initial_qpos"]},
                frame0=np.asarray(ep["obs"]["robot0_agentview_center_image"][0], np.uint8)))
    return out


def find_moving_joint(env, states, pattern):
    """The joint matching ``pattern`` whose qpos moves the most over the demo
    (robust src hinge/handle identification, layout-independent)."""
    mj = env.sim.model
    names = [mj.joint_id2name(i) for i in range(mj.njnt)]
    cand = [n for n in names if n and pattern in n and "temp" not in n]
    best, best_amp = None, -1.0
    for n in cand:
        adr = mj.get_joint_qpos_addr(n)
        if not np.isscalar(adr) and hasattr(adr, "__len__"):
            continue
        q = states[:, 1 + adr]      # flattened state = [time, qpos, qvel]
        amp = float(np.ptp(q))
        if amp > best_amp:
            best, best_amp = n, amp
    return best


def src_constants(env, spec, src):
    """Read source-side retarget constants (resets env to the source scene)."""
    arm = spec["arm"]
    reset_to(env, src["init"], replace_robot_joints=True, change_to_gr1=False)
    c = dict()
    c["Bpose_src"] = AP.controller_base_pose(env, arm)
    if spec["kind"] == "hinge":
        hinge = find_moving_joint(env, src["states"], spec["joint_pat"])
        c["src_hinge"] = hinge
        jid = env.sim.model.joint_name2id(hinge)
        c["pivot_src"] = env.sim.data.xanchor[jid].copy()
        # door-angle profile over the recorded demo states
        ang = []
        for s in src["states"]:
            env.sim.set_state_from_flattened(s)
            env.sim.forward()
            ang.append(float(env.sim.data.get_joint_qpos(hinge)))
        c["th_demo"] = np.array(ang)
        # restore source t0 (we mutated sim state reading the profile)
        reset_to(env, src["init"], replace_robot_joints=True, change_to_gr1=False)
    else:
        c["grasp"] = AP.detect_grasp_boundary(src["actions"], arm)
        c["Hps"] = AP.obj_base_pose(env, arm, spec["pick"])
        if spec["kind"] == "pnp2":
            c["Hcs"] = AP.obj_base_pose(env, arm, spec["place"])
        else:
            c["dest_src"] = fixture_pos_base(env, arm, spec["dest"])
    return c


def fixture_pos_base(env, arm, dest):
    fx = env.cab_sink if dest == "cab_sink" else env.fixtures.get("microwave_1_main_group",
                                                                  getattr(env, "microwave", None))
    p_world = np.asarray(fx.pos, np.float64)
    B = AP.controller_base_pose(env, arm)
    return (np.linalg.inv(B) @ np.append(p_world, 1.0))[:3]


# --- drawn-scene retarget (the redesign law: READ the drawn scene) -----------

def drawn_hinge_name(env, spec):
    if spec["joint_pat"] == "handle_joint":
        return env.sink.naming_prefix + "handle_joint"
    return f"{env.cab_sink.name}_leftdoorhinge"


def retarget_onto_drawn(env, spec, src, c, q0_row=None):
    """Retarget src demo actions onto the CURRENT (drawn) scene. Reads drawn
    hinge qpos / object poses; never overrides the scene. Returns traj or
    raises with a reason.

    q0_row (v6, gen-side only): the robot's SETTLED start pose as a 14-dim action
    row. When given, the PnP pre-grasp segment is blend-anchored to it
    (AP.blend_pregrasp): t0 == q0 exactly, decaying to the retargeted grasp pose
    by the demo grasp frame. None (default) = legacy behavior — the frozen eval
    manifest machinery is unchanged."""
    arm = spec["arm"]
    actions = src["actions"]
    B = AP.controller_base_pose(env, arm)
    if spec["kind"] == "hinge":
        hinge = drawn_hinge_name(env, spec)
        jid = env.sim.model.joint_name2id(hinge)
        axis = env.sim.data.xaxis[jid].copy()
        pivot = env.sim.data.xanchor[jid].copy()
        th_new0 = float(env.sim.data.get_joint_qpos(hinge))   # READ, not sampled
        th_demo, th0, thend = c["th_demo"], float(c["th_demo"][0]), float(c["th_demo"][-1])
        acts = actions
        # cross-layout probe extension: translate the arm's targets by the
        # base-frame pivot delta (≈0 for same layout+style)
        d = base_delta(B, pivot, c["Bpose_src"], c["pivot_src"])
        if np.linalg.norm(d) > 0.01:
            acts = translate_arm(actions, arm, d)
        return retarget_door(acts, arm, B, axis, pivot, th_demo, th0, thend, th_new0), \
            dict(hinge=hinge, th_new0=th_new0, pivot_shift=float(np.linalg.norm(d)))
    grasp = c["grasp"]
    Hpt = AP.obj_base_pose(env, arm, spec["pick"])            # drawn pick pose
    rt = AP.retarget_segment(actions, range(0, grasp), c["Hps"], Hpt, arm)
    if q0_row is not None:
        rt = AP.blend_pregrasp(rt, grasp, arm, q0_row)        # v6: t0 anchored at q0
    if spec["kind"] == "pnp2":
        Hct = AP.obj_base_pose(env, arm, spec["place"])       # drawn place pose
        rt = AP.retarget_segment(rt, range(grasp, len(actions)), c["Hcs"], Hct, arm)
        br = AP.slerp_bridge(rt[grasp - 1], rt[grasp], 8, arm,
                             actions[grasp, AP.GRIPPER_DIM[arm]], rt[grasp].copy())
        return np.concatenate([rt[:grasp], br, rt[grasp:]], axis=0), dict()
    # pnpfix: fixed fixture destination; cross-layout probe extension shifts the
    # original place segment by the drawn-vs-source dest delta (≈0 same layout)
    dest_drawn = fixture_pos_base(env, spec["arm"], spec["dest"])
    d = np.asarray(dest_drawn) - np.asarray(c["dest_src"])
    acts = actions
    if np.linalg.norm(d) > 0.01:
        acts = actions.copy()
        b = AP.ARM_BLOCK[arm]
        acts[grasp:, b:b + 3] += d[None, :]
    return AP.blend_place(rt, acts, grasp, arm), dict(dest_shift=float(np.linalg.norm(d)))


def base_delta(B_drawn, p_drawn_world, B_src, p_src_world):
    a = (np.linalg.inv(B_drawn) @ np.append(p_drawn_world, 1.0))[:3]
    b = (np.linalg.inv(B_src) @ np.append(p_src_world, 1.0))[:3]
    return a - b


def translate_arm(actions, arm, d):
    out = actions.copy()
    b = AP.ARM_BLOCK[arm]
    out[:, b:b + 3] += d[None, :]
    return out


def replay_success(env, traj):
    """Replay traj from the CURRENT state; success iff env._check_success()."""
    try:
        obs = env._get_observations()
        for a in traj:
            obs, _, done, _ = env.step(a)
            if done:
                break
        return bool(env._check_success()), None
    except Exception as e:
        return False, f"replay_exception:{type(e).__name__}:{e}"


# --- manifest mode ------------------------------------------------------------

def run_manifest(args):
    spec = SPECS[args.env]
    assert spec["tier"] == args.tier, f"{args.env} is tier {spec['tier']}, not {args.tier}"
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.env}__{args.tier}"
    env = build_env(spec)
    template = template_from_taskdemo(spec)
    sources = load_sources(spec)
    consts = [src_constants(env, spec, s) for s in sources]
    print(colored(f"[{tag}] template={template} sources={[s['key'] for s in sources]}", "cyan"),
          flush=True)

    scenes, log = [], []
    seed = args.seed0
    verified = False
    t_start = time.time()
    while len(scenes) < args.need and (seed - args.seed0) < args.max_draws:
        cur = seed
        seed += 1
        try:
            obs, rec = seeded_draw(env, template, cur)
        except Exception as e:
            log.append(dict(seed=cur, feasible=False, reason=f"reset_error:{type(e).__name__}:{e}"))
            print(colored(f"[{tag}] seed {cur} reset_error {e}", "red"), flush=True)
            continue
        if not verified:   # determinism gate (hard requirement)
            _, rec2 = seeded_draw(env, template, cur)
            assert rec2["state_sha"] == rec["state_sha"], \
                f"NON-DETERMINISTIC seeded reset at seed {cur}"
            print(colored(f"[{tag}] determinism verified at seed {cur}", "green"), flush=True)
            verified = True
            obs, rec = seeded_draw(env, template, cur)   # restore drawn scene
        rec["render"] = render_center(env)
        rec["extras"] = task_extras(env, spec)
        ok, why, cond_key = False, [], None
        for si, (src, c) in enumerate(zip(sources, consts)):
            if si > 0:      # restore the drawn scene (dirty from previous replay)
                seeded_draw(env, template, cur)
            try:
                traj, info = retarget_onto_drawn(env, spec, src, c)
            except Exception as e:
                why.append(f"{src['key']}:retarget_error:{type(e).__name__}:{e}")
                continue
            good, err = replay_success(env, traj)
            if good:
                ok, cond_key = True, src["key"]
                why.append(f"{src['key']}:success")
                break
            why.append(f"{src['key']}:{err or 'success_check_false'}")
        log.append(dict(seed=cur, feasible=ok, reason=";".join(why), cond_demo=cond_key))
        if ok:
            rec["cond_demo_key"] = cond_key
            scenes.append(rec)
        n = len(scenes)
        print(colored(f"[{tag}] seed {cur}: {'FEASIBLE' if ok else 'infeasible'} "
                      f"({n}/{args.need}, {seed - args.seed0} draws, "
                      f"{(time.time() - t_start) / 60:.1f}m) {why[-1] if why else ''}",
                      "green" if ok else "yellow"), flush=True)

    save_manifest(args, spec, tag, template, sources, scenes, log)


def task_extras(env, spec):
    ex = {}
    if spec["kind"] == "hinge":
        h = drawn_hinge_name(env, spec)
        ex["hinge"] = h
        ex["hinge_qpos"] = float(env.sim.data.get_joint_qpos(h))
    else:
        for o in ("vegetable", "fruit_container"):
            if o in env.objects:
                i = env.sim.model.body_name2id(env.objects[o].root_body)
                ex[f"{o}_pos_world"] = env.sim.data.body_xpos[i].tolist()
    if env.cab_sink is not None:
        ex["cab_sink"] = env.cab_sink.name
        st = env.cab_sink.get_door_state(env)
        ex["cab_sink_door_state"] = {k: float(v) for k, v in st.items()}
    return ex


def save_manifest(args, spec, tag, template, sources, scenes, log):
    npz_path = os.path.join(args.out_dir, f"{tag}.npz")
    json_path = os.path.join(args.out_dir, f"{tag}.json")
    sheet_path = os.path.join(args.out_dir, f"{tag}_contact.png")
    def _obj_arr(key):   # per-scene dims can differ (object draw varies nq) -> object array
        a = np.empty(len(scenes), object)
        for i, s in enumerate(scenes):
            a[i] = s[key]
        return a

    arr = dict(
        seeds=np.array([s["seed"] for s in scenes], np.int64),
        states=_obj_arr("states"),
        initial_qpos=_obj_arr("initial_qpos"),
        non_robot_qpos_idx=_obj_arr("non_robot_qpos_idx"),
        object_state=_obj_arr("object_state"),
        renders=np.stack([s["render"] for s in scenes]) if scenes else np.zeros((0,)),
        model_files=np.array([s["model_file"] for s in scenes], object),
        ep_metas=np.array([s["ep_meta"] for s in scenes], object),
        cond_demo_keys=np.array([s.get("cond_demo_key", "") for s in scenes], object),
    )
    np.savez_compressed(npz_path, **arr)
    meta = dict(
        env=spec["env"], tier=spec["tier"], base_task=spec["base"], tag=tag,
        demo_hdf5=demo_hdf5(spec["demo_dir"]),
        template=template, kind=spec["kind"], arm=spec["arm"],
        eval_seed0=args.seed0, n_scenes=len(scenes),
        seeds=[s["seed"] for s in scenes],
        state_shas={str(s["seed"]): s["state_sha"] for s in scenes},
        cond_demo={str(s["seed"]): dict(hdf5=demo_hdf5(spec["demo_dir"]),
                                        key=s.get("cond_demo_key")) for s in scenes},
        extras={str(s["seed"]): s["extras"] for s in scenes},
        feasibility_log=log,
        n_draws=len(log), n_feasible=sum(1 for e in log if e["feasible"]),
        legacy=legacy_block(spec) if spec["tier"] == "seen" else None,
        note=("one-shot eval: L2/L3 scenes are evaluated conditioned on cond_demo "
              "(the tier's own teleop demo that passed the feasibility replay); "
              "these demos are EVAL-ONLY, never training data."
              if spec["tier"] in ("L2", "L3") else ""),
    )
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=1)
    contact_sheet(sheet_path, tag, scenes, sources, spec)
    print(colored(f"[{tag}] saved {len(scenes)} scenes -> {npz_path}\n"
                  f"  json: {json_path}\n  sheet: {sheet_path}", "magenta"), flush=True)


def legacy_block(spec):
    """The 3 original teleop scenes (continuity tier), listed separately."""
    with h5py.File(demo_hdf5(spec["demo_dir"]), "r") as f:
        ks = sorted(f["data"].keys())
        return dict(hdf5=demo_hdf5(spec["demo_dir"]), demo_keys=ks,
                    ep_meta={k: str(f["data"][k].attrs["ep_meta"])[:400] for k in ks})


def contact_sheet(path, tag, scenes, sources, spec, cols=5):
    from PIL import Image, ImageDraw
    pair = spec["tier"] in ("L2", "L3")   # [scene | one-shot cond demo frame0]
    cw = 256 * (2 if pair else 1)
    cells = []
    src_by_key = {s["key"]: s for s in sources}
    for s in scenes:
        im = Image.fromarray(s["render"])
        if pair:
            k = s.get("cond_demo_key")
            d = src_by_key.get(k)
            dim = Image.fromarray(d["frame0"]).resize((256, 256)) if d is not None \
                else Image.new("RGB", (256, 256), (30, 30, 30))
            canvas = Image.new("RGB", (cw, 256))
            canvas.paste(im, (0, 0))
            canvas.paste(dim, (256, 0))
            im = canvas
        dr = ImageDraw.Draw(im)
        label = f"seed {s['seed']}"
        if pair:
            label += f" | demo {s.get('cond_demo_key', '?')}"
        dr.rectangle([0, 0, cw, 14], fill=(0, 0, 0))
        dr.text((3, 2), f"{label}", fill=(255, 255, 0))
        cells.append(im)
    if spec["tier"] == "seen":            # legacy teleop scenes appended, labeled
        for src in sources:
            im = Image.fromarray(src["frame0"]).resize((256, 256))
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 256, 14], fill=(0, 0, 128))
            dr.text((3, 2), f"LEGACY {src['key']}", fill=(255, 255, 0))
            cells.append(im)
    if not cells:
        return
    from PIL import Image as I
    rows = (len(cells) + cols - 1) // cols
    sheet = I.new("RGB", (cols * cw + (cols - 1) * 4, rows * 256 + (rows - 1) * 4), (60, 60, 60))
    for i, im in enumerate(cells):
        r, cc = divmod(i, cols)
        sheet.paste(im, (cc * (cw + 4), r * (256 + 4)))
    sheet.save(path)


# --- verify mode ---------------------------------------------------------------

def run_verify(args):
    meta = json.load(open(args.manifest))
    spec = SPECS[[k for k, v in SPECS.items() if v["env"] == meta["env"]][0]]
    env = build_env(spec)
    template = meta["template"]
    bad = 0
    for seed_s, sha in meta["state_shas"].items():
        _, rec = seeded_draw(env, template, int(seed_s))
        ok = rec["state_sha"] == sha
        bad += (not ok)
        print(f"seed {seed_s}: {'OK' if ok else 'MISMATCH'}", flush=True)
    print(colored(f"verify {meta['tag']}: {len(meta['state_shas']) - bad}/"
                  f"{len(meta['state_shas'])} identical", "magenta"), flush=True)
    sys.exit(1 if bad else 0)


# --- coverage mode (train-side seed stream, t0 draws only) ----------------------

def run_coverage(args):
    spec = SPECS[args.env]
    env = build_env(spec)
    template = template_from_taskdemo(spec)
    tag = f"{spec['env']}__{spec['tier']}"
    vecs, seeds = [], []
    s = TRAIN_SEED0
    while len(vecs) < args.n and s < TRAIN_SEED0 + args.n * 3:
        cur = s
        s += 1
        try:
            _, rec = seeded_draw(env, template, cur)
        except Exception as e:
            print(colored(f"[{tag}] cover seed {cur} reset_error {e}", "red"), flush=True)
            continue
        vecs.append(rec["object_state"])
        seeds.append(cur)
        if len(vecs) % 10 == 0:
            print(f"[{tag}] coverage draws {len(vecs)}/{args.n}", flush=True)
    dims = [v.shape[0] for v in vecs]
    modal = max(set(dims), key=dims.count)
    keep = [i for i, d in enumerate(dims) if d == modal]
    out = os.path.join(args.out_dir, f"{tag}_train_draws.npz")
    np.savez_compressed(out, object_state=np.stack([vecs[i] for i in keep]),
                        seeds=np.array([seeds[i] for i in keep]),
                        n_dropped_dim_mismatch=len(vecs) - len(keep))
    print(colored(f"[{tag}] coverage: {len(keep)} train draws (dim {modal}, "
                  f"{len(vecs) - len(keep)} dropped) -> {out}", "magenta"), flush=True)


# --- matrix mode (layout-task feasibility probe) --------------------------------

class draw_deadline:
    """SIGALRM guard: a natural draw that exceeds the deadline (kitchen's endless
    placement-retry loop) raises TimeoutError instead of hanging the probe."""

    def __init__(self, seconds):
        self.seconds = int(seconds)

    def __enter__(self):
        import signal
        signal.signal(signal.SIGALRM,
                      lambda *_: (_ for _ in ()).throw(TimeoutError("draw_timeout")))
        signal.alarm(self.seconds)

    def __exit__(self, *exc):
        import signal
        signal.alarm(0)
        return False

def run_matrix(args):
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for task in SEEN_TASKS:
        spec = SPECS[task]
        env = build_env(spec)
        sources = load_sources(spec)
        consts = [src_constants(env, spec, s) for s in sources]
        canon_style = template_from_taskdemo(spec)["style_id"]
        for layout in args.layouts:
            key = f"L{layout}|{task}"
            template = {"layout_id": int(layout), "style_id": canon_style}
            cell = dict(layout=int(layout), task=task, probes=[])
            # layout-10 object law (see LAYOUT10_REMOVE): without this the natural
            # draw loops forever on the unplaceable fruit/bread containers
            if int(layout) == 10:
                _orig_cfgs = type(env)._get_obj_cfgs
                env._get_obj_cfgs = lambda: [c for c in _orig_cfgs(env)
                                             if c["name"] not in LAYOUT10_REMOVE]
                cell["layout10_obj_filter"] = list(LAYOUT10_REMOVE)
            else:
                env.__dict__.pop("_get_obj_cfgs", None)
            # fixture presence gate (needs one draw to inspect references)
            try:
                with draw_deadline(600):
                    seeded_draw(env, template, MATRIX_SEED0)
            except Exception as e:
                cell["fixture_check"] = f"reset_error:{e}"
                results[key] = cell
                print(colored(f"[matrix] {key}: reset_error {e}", "red"), flush=True)
                continue
            has_cab = env.cab_sink is not None
            cell["has_cab_sink"] = bool(has_cab)
            cell["has_fruit_container"] = "fruit_container" in env.objects
            needs_cab = spec["kind"] == "hinge" and spec["joint_pat"] == "leftdoorhinge" \
                or (spec["kind"] == "pnpfix" and spec["dest"] == "cab_sink")
            if needs_cab and not has_cab:
                cell["feasible"] = False
                cell["why"] = "no cab_sink fixture in this layout"
                results[key] = cell
                print(colored(f"[matrix] {key}: no cab_sink", "yellow"), flush=True)
                continue
            if spec["kind"] == "pnp2" and not cell["has_fruit_container"]:
                cell["probes"].append(dict(seed=MATRIX_SEED0, ok=False, why="no fruit_container"))
            succ = 0
            for i in range(args.probes):
                cur = MATRIX_SEED0 + i
                probe = dict(seed=cur)
                try:
                    with draw_deadline(600):
                        seeded_draw(env, template, cur)
                    traj, info = retarget_onto_drawn(env, spec, sources[0], consts[0])
                    ok, err = replay_success(env, traj)
                    probe.update(ok=bool(ok), why=err or ("success" if ok else "success_check_false"),
                                 **{k: v for k, v in info.items()})
                except Exception as e:
                    probe.update(ok=False, why=f"{type(e).__name__}:{e}")
                succ += int(probe["ok"])
                cell["probes"].append(probe)
                print(colored(f"[matrix] {key} seed {cur}: {probe['why'][:120]}",
                              "green" if probe["ok"] else "yellow"), flush=True)
            cell["success"] = f"{succ}/{args.probes}"
            cell["feasible"] = succ > 0
            results[key] = cell
        env.close()
    out = os.path.join(args.out_dir, f"layout_task_matrix_raw{args.matrix_suffix}.json")
    json.dump(results, open(out, "w"), indent=1)
    print(colored(f"[matrix] saved {out}", "magenta"), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["manifest", "matrix", "coverage", "verify"])
    ap.add_argument("--env", default=None, help="SPECS key (manifest/coverage modes)")
    ap.add_argument("--tier", default=None, choices=["seen", "L2", "L3"])
    ap.add_argument("--need", type=int, default=20, help="feasible scenes to freeze")
    ap.add_argument("--max_draws", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--n", type=int, default=100, help="coverage draws")
    ap.add_argument("--layouts", type=int, nargs="+", default=[0, 2, 4, 7, 10])
    ap.add_argument("--probes", type=int, default=3, help="matrix probe seeds per cell")
    ap.add_argument("--matrix_suffix", default="")
    ap.add_argument("--manifest", default=None, help="manifest json (verify mode)")
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    args = ap.parse_args()
    try:
        if args.mode == "manifest":
            run_manifest(args)
        elif args.mode == "coverage":
            run_coverage(args)
        elif args.mode == "matrix":
            run_matrix(args)
        else:
            run_verify(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
