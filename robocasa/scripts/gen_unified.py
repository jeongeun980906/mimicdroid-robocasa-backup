#!/usr/bin/env python3
"""Phase 2 of the GR1 data/eval redesign (docs/gr1_data_eval_redesign.md):
UNIFIED training-data generator — replaces augment_door / gen_pnp_cabinet scene
logic with the Phase-1 law: natural seeded ``env.reset()`` scene draws +
drawn-scene retargeting (READ the drawn hinge qpos / object poses; NO artificial
angle/spawn laws), plus the multi-task same-scene counterfactual core: ONE scene
draw -> episodes for EVERY task feasible in that layout, all sharing a bitwise
identical initial scene state.

SCOPE: L1/seen cells only (canonical layouts 0/2/4/7 with their canonical *_l1
styles). L2/L3 are NEVER generated for training.

Machinery reuse: scene draw / determinism / retarget / feasibility-replay come
from make_eval_manifest.py (Phase 1, verified); episode saving mirrors
augment_door / gen_pnp_cabinet's block; GR1 replay mirrors replay_to_gr1.py
(same approach bridge + settle + bbox), with scene/task attrs preserved.

SEED NAMESPACING (hard requirement): the TRAIN generation stream starts at
1_000_000, per-cell sub-streams 100k apart. The frozen eval manifests use
seed0=1000 (all logged seeds < 1100), Phase-1 coverage draws use seed0=100000
(< 100300), matrix probes 7000. assert_seed_streams_disjoint() re-reads the
frozen manifest jsons at runtime and refuses to run on any overlap.

RENDER-SETTLE DISCIPLINE (hard requirement): robosuite's FIRST post-reset frame
misses dynamically-loaded objects (empty-sink t=0). NOTHING is recorded or
rendered here before at least one env.step(): every stage settles >=1
(default 3) unrecorded closed-loop hold steps after reset/reset_to, and every
saved episode must pass a t=0 VISIBILITY ASSERT (task anchor geoms present in a
frame-0 instance-segmentation render) or it is rejected and logged.

Stages (run in order; each is subprocess-parallel over disjoint shards):
  scenes    draw seeded scenes for one (layout,style) cell -> scenes npz/json + contact sheet
  episodes  one task x scene-range: reset_to drawn scene -> settle -> retarget -> replay
            -> DemoTwoHand hdf5 part (EXACT existing format) + feasibility json
  gr1       replay a DemoTwoHand part on GR1 (keep successes) -> GR1 hdf5 part
            + per-(scene,task) GR1 pass/fail json (defines D ∩ GR1-feasible)
  merge     concatenate part hdf5s (per cell x task)
  sibling   write sibling_tasks attrs + scene_groups.json + gr1_scene_table.json
  audit     multi-task counterfactual audit + bitwise sibling-t0 check +
            init-distribution coverage vs the frozen eval manifests + vis stats
  video     dump per-episode mp4s (validation rule: every check shown as video)
  plan      print the shard commands for a full run given budget parameters

Smoke run (layout 0, 1 scene, all 4 tasks), then the same commands with bigger
ranges regenerate the full set:
    MUJOCO_GL=egl .venv/bin/python robocasa/scripts/gen_unified.py \
        --stage scenes --layout 0 --need 1 --out_dir /mnt/ddn/tmp/gen_unified_smoke
"""
import os
import sys
import json
import time
import glob
import random
import argparse
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.dirname(__file__))
import augment_pnp as AP                    # replay_record, merge, retarget helpers
import make_eval_manifest as MEM            # Phase-1 draw/retarget/feasibility machinery
import replay_to_gr1 as RG                  # GR1 approach bridge / hold action / bbox
from robocasa.utils.mimicdroid_utils import make_env, reset_to, EnvArgs

# --- cells (L1/seen ONLY; style = the cell's canonical taskdemo style) --------
EVAL_MANIFEST_DIR = "/mnt/ddn/dataset/gr1_eval_manifests"   # frozen, READ-ONLY
TRAIN_GEN_SEED0 = 1_000_000
SEEN_TASKS = ["CloseLeftCabinetDoor", "TurnOnFaucet",
              "PnPSinkToCabinet", "PnPSinkToRightCounterPlate"]
CELLS = {   # layout -> host task (canonical there; its env draws the scene), style, seed0
    0: dict(host="CloseLeftCabinetDoor",       style="001_l1", seed0=TRAIN_GEN_SEED0 + 0),
    2: dict(host="PnPSinkToCabinet",           style="015_l1", seed0=TRAIN_GEN_SEED0 + 100_000),
    4: dict(host="TurnOnFaucet",               style="037_l1", seed0=TRAIN_GEN_SEED0 + 200_000),
    7: dict(host="PnPSinkToRightCounterPlate", style="019_l1", seed0=TRAIN_GEN_SEED0 + 300_000),
}
LAYOUT_TASKS = {    # per layout_task_matrix.md (layout 4 has no cab_sink)
    0: SEEN_TASKS, 2: SEEN_TASKS, 7: SEEN_TASKS,
    4: ["TurnOnFaucet", "PnPSinkToRightCounterPlate"],
}
VIS_CAM = "robot0_agentview_center"
# t=0 visibility is checked across ALL policy cameras: the anchor must be visible
# in >=1 of them. (Layout-2 finding: the center cam's sink view is occluded by a
# sink fixture element in EVERY natural draw — shared by the frozen eval scenes —
# while both side cams see the vegetable clearly; a center-only check mass-rejects
# scenes the eval distribution accepts.)
VIS_CAMS = ("robot0_agentview_center", "robot0_agentview_center_top",
            "robot0_agentview_sideleft", "robot0_agentview_sideright")
VIS_SIZE = 256
HOLD_TMPL = np.zeros(14)    # settle hold row template: grips 0, pose filled closed-loop

# v6 per-task GR1 recipe (user-tuned, all 4 KEPT in the 1-ep validation):
#   (approach_fixed frames @1cm/f, handoff_blend frames)
# Door/Plate: far q0 gap -> real approach, short blend; Cab/Fau: small gap ->
# no approach, blend-only over the demo's first 16 frames.
TASK_RECIPE = {
    "CloseLeftCabinetDoor":       (30, 10),
    "PnPSinkToRightCounterPlate": (20, 10),
    "PnPSinkToCabinet":           (0, 16),
    "TurnOnFaucet":               (0, 16),
}

# v6 grasp-precision recipe (validated 2026-07-13: Plate GR1 keep 1/6 -> 5/6):
#   (pre-close dwell frames, close stretch factor, close window frames)
# Pause at the pre-grasp pose so arm/object transients settle, then close the
# fingers at half speed. Grasp-object tasks only (hinge tasks don't need it).
GRASP_DWELL = {
    "PnPSinkToCabinet":           (12, 2, 6),
    "PnPSinkToRightCounterPlate": (12, 2, 6),
}

# v6 object law (REVISED 2026-07-13, pin track deleted): GR1 sources are the
# pool-track episodes with their NATIVE vegetables — the frozen seen-eval
# manifests draw random species, so species-diverse GR1 data matches eval.
# Augmentation(demo-pool) scenes draw from VEG_POOL. Door/Faucet natural scenes.
GR1_VEG = {
    "PnPSinkToCabinet":           {"primary": "tomato_1", "secondary": "lime_1"},
    "PnPSinkToRightCounterPlate": {"primary": "corn_1",   "secondary": "carrot_1"},
}
VEG_POOL_ROUND = "tomato,garlic,lime,potato"      # Cabinet family (+Door/Faucet scenes)
VEG_POOL_LONG = "corn,carrot,eggplant,artichoke"   # Plate family


def cell_tag(layout, style):
    return f"L{layout}_{style}"


def assert_seed_streams_disjoint(lo, hi):
    """Refuse to run if [lo, hi) touches any frozen Phase-1 seed (manifests,
    feasibility logs, coverage draws). The frozen files are read-only inputs."""
    frozen = set()
    for p in glob.glob(os.path.join(EVAL_MANIFEST_DIR, "*.json")):
        try:
            m = json.load(open(p))
        except Exception:
            continue
        frozen.update(int(e["seed"]) for e in m.get("feasibility_log") or [])
        frozen.update(int(s) for s in m.get("seeds") or [])
    for p in glob.glob(os.path.join(EVAL_MANIFEST_DIR, "*_train_draws.npz")):
        frozen.update(int(s) for s in np.load(p)["seeds"])
    assert frozen, f"no frozen seeds found under {EVAL_MANIFEST_DIR} (wrong dir?)"
    overlap = {s for s in frozen if lo <= s < hi}
    assert not overlap, f"train seed range [{lo},{hi}) overlaps frozen seeds: {sorted(overlap)[:5]}"
    assert max(frozen) < TRAIN_GEN_SEED0, \
        f"frozen seed {max(frozen)} >= TRAIN_GEN_SEED0 {TRAIN_GEN_SEED0}: namespace broken"


# --- settle + t0 visibility (hard requirements) --------------------------------

def settle_env(env, steps):
    """Unrecorded closed-loop hold steps after any reset/reset_to: re-read the
    CURRENT both-arm eef every step (so the robot stays put) and step, letting
    robosuite's renderer pick up dynamically-loaded objects. The hold row is a
    fixed template (grips 0) -> identical across tasks, so multi-task sibling
    episodes stay bitwise identical after settling."""
    assert steps >= 1, "settle>=1 required: the first post-reset render misses dynamic objects"
    obs = env._get_observations()
    for _ in range(steps):
        obs, _, _, _ = env.step(RG._gr1_current_action(obs, HOLD_TMPL))
    return obs


def anchor_geom_ids(env, spec):
    """Geom ids of the task-relevant anchor body subtree: the drawn hinge's body
    (door/faucet) or the pick object's root body (pnp)."""
    mj = env.sim.model
    if spec["kind"] == "hinge":
        root = int(mj.jnt_bodyid[mj.joint_name2id(MEM.drawn_hinge_name(env, spec))])
    else:
        root = int(mj.body_name2id(env.objects[spec["pick"]].root_body))
    ids = []
    for g in range(mj.ngeom):
        b = int(mj.geom_bodyid[g])
        while b > 0:
            if b == root:
                ids.append(g)
                break
            b = int(mj.body_parentid[b])
    return np.asarray(ids, np.int64)


def t0_visible_px(env, spec):
    """Per-camera pixel counts of the task anchor in fresh segmentation renders
    of ALL policy cameras (pass = max over cams >= vis_min_px). MUST be called
    only after settle_env (>=1 step). Returns (max_px, {cam: px})."""
    ids = anchor_geom_ids(env, spec)
    px = {}
    for cam in VIS_CAMS:
        seg = np.asarray(env.sim.render(width=VIS_SIZE, height=VIS_SIZE,
                                        camera_name=cam, segmentation=True))
        gid = seg[..., 1] if (seg.ndim == 3 and seg.shape[-1] == 2) else seg
        px[cam] = int(np.isin(gid, ids).sum())
    return max(px.values()), px


# --- shared episode saving (EXACT existing dataset format) ----------------------

def save_episode(od, gi, actions, states, obs_list, env, env_keys, attrs,
                 flip_seg=False, bbox_cams=()):
    """The augment_door/gen_pnp_cabinet saving block, factored: states/actions/
    dones/rewards + obs (images stored upright) + object/objects-joint-state +
    the standard attrs, plus the new scene bookkeeping attrs passed in."""
    n = len(states)
    nq = env.sim.model.nq
    robot_idx = set(int(i) for i in env.robots[0]._ref_joint_pos_indexes)
    non_robot_idx = np.array(sorted(set(range(nq)) - robot_idx))
    qpos_all = np.stack([states[t][1:1 + nq] for t in range(n)])

    g = od.create_group(f"demo_{gi}")
    g.create_dataset("actions", data=np.asarray(actions)[:n].astype(np.float32))
    g.create_dataset("states", data=np.asarray(states))
    g.create_dataset("dones", data=np.zeros(n, np.int64))
    g.create_dataset("rewards", data=np.zeros(n, np.int64))
    og = g.create_group("obs")
    for key in env_keys:
        if key in obs_list[0]:
            arr = np.stack([np.asarray(o[key]) for o in obs_list])
            if "image" in key or (flip_seg and "segmentation" in key):
                arr = arr[:, ::-1]      # bottom-origin -> upright
            og.create_dataset(key, data=arr)
    for cam in bbox_cams:               # GR1 stage: bbox from upright segmentation
        sk = cam + "_segmentation_instance"
        if sk not in og:
            continue
        seg_seq = og[sk][()]
        res = [RG.bbox_from_seg(seg_seq[t]) for t in range(n)]
        og.create_dataset(cam + "_bbox", data=np.stack([r[0] for r in res]))
        og.create_dataset(cam + "_bbox_ids", data=np.stack([r[1] for r in res]))
        og.create_dataset(cam + "_bbox_mask", data=np.stack([r[2] for r in res]))
    if "object-state" in og:
        og.create_dataset("object", data=og["object-state"][()])
    og.create_dataset("objects-joint-state", data=qpos_all[:, non_robot_idx])
    g.attrs["model_file"] = env.sim.model.get_xml()
    g.attrs["initial_qpos"] = qpos_all[0]
    g.attrs["non_robot_qpos_idx"] = non_robot_idx
    g.attrs["num_samples"] = n
    for k, v in attrs.items():
        g.attrs[k] = v
    return g


def env_obs_keys(demo_path, keep_seg):
    with h5py.File(demo_path, "r") as f:
        k0 = sorted(f["data"].keys())[0]
        orig = set(f["data"][k0]["obs"].keys())
    drop = {k for k in orig if "bbox" in k} | {"object", "objects-joint-state"}
    if not keep_seg:
        drop |= {k for k in orig if "segmentation" in k}
    return orig - drop


def build_env2(spec, robots="DemoTwoHand", use_camera_obs=True, seg=None, controller=None):
    env, _ = make_env(MEM.demo_hdf5(spec["demo_dir"]), EnvArgs(
        robots=robots, control_freq=20, use_camera_obs=use_camera_obs,
        camera_segmentations=seg, controller=controller, task_name=spec["env"]))
    return env


def scene_init(z, i):
    """reset_to() state dict from scenes-npz row i (same fields as the frozen
    eval manifests, so this stage also runs verbatim on an eval manifest npz)."""
    return {"states": z["states"][i], "ep_meta": str(z["ep_metas"][i]),
            "model": str(z["model_files"][i]), "non_robot_qpos_idx": z["non_robot_qpos_idx"][i],
            "qpos_state": z["initial_qpos"][i]}


def seed_globals(seed):
    """reset_to() keeps the robot joints of its internal env.reset(), whose robot
    init noise comes from the GLOBAL np.random (robosuite robot.py) — seed the
    globals with the scene seed before EVERY reset_to so all of a scene's
    multi-task sibling episodes (and all source-retry attempts) share a bitwise
    identical initial state, across tasks and processes."""
    np.random.seed(seed % (2**32))
    random.seed(seed)


# --- stage: scenes ---------------------------------------------------------------

def _veg_of_ep_meta(em):
    """Drawn vegetable instance (e.g. 'tomato_1') from a scene ep_meta, or None."""
    if isinstance(em, str):
        em = json.loads(em)
    for c in em.get("object_cfgs", []):
        if c.get("name") == "vegetable":
            info = c.get("info") or {}
            path = info.get("mjcf_path", "")
            if "/" in path:
                return path.rstrip("/").split("/")[-2]
            return str(c.get("obj_groups"))
    return None


def _patch_veg_groups(env, categories):
    """Restrict the host env's vegetable draw to ``categories`` (v6 demo-object law:
    GR1/DTH scenes must use THE demo object — Plate=corn_1, Cabinet=tomato_1)."""
    import types
    orig = env._get_obj_cfgs

    def patched(self):
        cfgs = orig()
        for c in cfgs:
            if c.get("name") == "vegetable":
                c["obj_groups"] = list(categories)
        return cfgs

    env._get_obj_cfgs = types.MethodType(patched, env)


def run_scenes(args):
    cell = CELLS[args.layout]
    style = args.style or cell["style"]
    tag = cell_tag(args.layout, style)
    if args.veg_set:
        tag += f"_set_{args.veg_set.split(',')[0]}"   # family dirs (round vs long)
    host = MEM.SPECS[cell["host"]]
    seed_lo = cell["seed0"] + args.start
    # veg_set rejects non-matching draws: scan extended until `need` accepted
    max_draws = args.need * (4 if args.veg_set else 1)
    seed_hi = seed_lo + max_draws
    assert_seed_streams_disjoint(seed_lo, seed_hi)
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    env = MEM.build_env(host)
    if args.veg_set:
        _patch_veg_groups(env, [c.strip() for c in args.veg_set.split(",")])
    template = MEM.template_from_taskdemo(host, layout_override=args.layout, style_override=style)
    print(colored(f"[scenes {tag}] host={cell['host']} template={template} "
                  f"seeds [{seed_lo},{seed_hi}) veg_set={args.veg_set}",
                  "cyan"), flush=True)

    scenes, log, verified = [], [], False
    t0 = time.time()
    for seed in range(seed_lo, seed_hi):
        if len(scenes) >= args.need:
            break
        t_draw = time.time()
        try:
            _, rec = MEM.seeded_draw(env, template, seed)
        except Exception as e:
            log.append(dict(seed=seed, ok=False, reason=f"reset_error:{type(e).__name__}:{e}"))
            print(colored(f"[scenes {tag}] seed {seed} reset_error {e}", "red"), flush=True)
            continue
        if not verified:    # per-job determinism gate (Phase-1 hard requirement)
            _, rec2 = MEM.seeded_draw(env, template, seed)
            assert rec2["state_sha"] == rec["state_sha"], f"NON-DETERMINISTIC draw at seed {seed}"
            print(colored(f"[scenes {tag}] determinism verified at seed {seed}", "green"), flush=True)
            verified = True
            _, rec = MEM.seeded_draw(env, template, seed)
        # contact-sheet render obeys the settle discipline (state already recorded)
        settle_env(env, args.settle)
        rec["render"] = MEM.render_center(env)
        rec["extras"] = MEM.task_extras(env, host)
        scenes.append(rec)
        log.append(dict(seed=seed, ok=True, secs=round(time.time() - t_draw, 1)))
        print(colored(f"[scenes {tag}] seed {seed} drawn ({len(scenes)}, "
                      f"{(time.time() - t0) / 60:.1f}m)", "green"), flush=True)

    stem = f"scenes_{args.start:05d}_{args.start + args.need:05d}"
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    def _obj(key):
        a = np.empty(len(scenes), object)
        for i, s in enumerate(scenes):
            a[i] = s[key]
        return a
    np.savez_compressed(
        npz_path,
        seeds=np.array([s["seed"] for s in scenes], np.int64),
        states=_obj("states"), initial_qpos=_obj("initial_qpos"),
        non_robot_qpos_idx=_obj("non_robot_qpos_idx"), object_state=_obj("object_state"),
        model_files=np.array([s["model_file"] for s in scenes], object),
        ep_metas=np.array([s["ep_meta"] for s in scenes], object),
        renders=np.stack([s["render"] for s in scenes]) if scenes else np.zeros((0,)))
    meta = dict(cell=tag, layout=args.layout, style=style, host=cell["host"],
                seed_lo=seed_lo, seed_hi=seed_hi, n_scenes=len(scenes),
                template=template, settle=args.settle,
                state_shas={str(s["seed"]): s["state_sha"] for s in scenes},
                extras={str(s["seed"]): s["extras"] for s in scenes}, log=log)
    json.dump(meta, open(os.path.join(out_dir, f"{stem}.json"), "w"), indent=1)
    contact_grid(os.path.join(out_dir, f"{stem}_contact.png"),
                 [(f"seed {s['seed']}", s["render"]) for s in scenes])
    print(colored(f"[scenes {tag}] saved {len(scenes)} scenes -> {npz_path}", "magenta"), flush=True)


def contact_grid(path, labeled_images, cols=5):
    if not labeled_images:
        return
    from PIL import Image, ImageDraw
    cells = []
    for label, arr in labeled_images:
        im = Image.fromarray(np.asarray(arr, np.uint8))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
        dr.text((3, 2), label, fill=(255, 255, 0))
        cells.append(im)
    w, h = cells[0].size
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4), (60, 60, 60))
    for i, im in enumerate(cells):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * (w + 4), r * (h + 4)))
    sheet.save(path)


def _q0_row(obs):
    """The robot's CURRENT (settled) both-arm eef pose as a 14-dim action row —
    the v6 fixed-q0 anchor passed to retarget_onto_drawn (grips 0/unused)."""
    import robosuite.utils.transform_utils as T
    row = np.zeros(14)
    for arm, b in (("right", 0), ("left", 6)):
        key = f"robot0_base_to_{arm}_eef"
        row[b:b + 3] = np.asarray(obs[key + "_pos"], float)
        row[b + 3:b + 6] = T.quat2axisangle(np.asarray(obs[key + "_quat_site"], float))
    return row


def _approach_step(obs, a_tgt, v_step, rot_step, active_b=None):
    """ONE closed-loop approach row: from the CURRENT eef pose (obs) toward the
    target row, translation capped at v_step (m/frame), rotation at rot_step
    (rad/frame). Grips copied from the target row. Returns (row, err) where err
    is the ACTIVE arm's pos error only when active_b is given (the inactive arm
    still gets stepped, but a stubborn IK residual there must not stall the
    approach — Cabinet capped out at 200f on the inactive arm)."""
    import robosuite.utils.transform_utils as T
    row = np.asarray(a_tgt, np.float64).copy()
    err = 0.0
    err_rot = 0.0
    for arm, b in (("right", 0), ("left", 6)):
        cur = np.asarray(obs[f"robot0_base_to_{arm}_eef_pos"], float)
        tgt = np.asarray(a_tgt[b:b + 3], float)
        d = tgt - cur
        n = float(np.linalg.norm(d))
        row[b:b + 3] = cur + (d if n <= v_step else d * (v_step / n))
        qc = np.asarray(obs[f"robot0_base_to_{arm}_eef_quat_site"], float)
        qt = T.axisangle2quat(np.asarray(a_tgt[b + 3:b + 6], float))
        ang = 2.0 * np.arccos(min(1.0, abs(float(np.dot(qc / np.linalg.norm(qc),
                                                        qt / np.linalg.norm(qt))))))
        if active_b is None or b == active_b:
            err = max(err, n)
            err_rot = max(err_rot, ang)   # Door's q0 gap is ROTATION-dominant (~3 rad);
                                          # a pos-only gate hands off with a rot snap
        s = 1.0 if ang <= rot_step else rot_step / ang
        row[b + 3:b + 6] = T.quat2axisangle(T.quat_slerp(qc, qt, s))
    return row, max(err, err_rot * 0.1)   # 0.1 rad rot ~ 1 cm equivalent in the gate


# --- stage: episodes (DemoTwoHand generation on drawn scenes) ---------------------

def run_episodes(args):
    spec = MEM.SPECS[args.task]
    assert spec["tier"] == "seen", "L1/seen ONLY for training; L2/L3 never"
    cell = CELLS[args.layout]
    style = args.style or cell["style"]
    tag = cell_tag(args.layout, style)
    assert args.task in LAYOUT_TASKS[args.layout], \
        f"{args.task} not feasible in layout {args.layout} (layout_task_matrix)"

    z = np.load(args.scenes, allow_pickle=True)
    lo, hi = args.scene_start, min(args.scene_end, len(z["seeds"]))
    env = build_env2(spec)
    sources = MEM.load_sources(spec)
    consts = [MEM.src_constants(env, spec, s) for s in sources]
    keys = env_obs_keys(MEM.demo_hdf5(spec["demo_dir"]), keep_seg=False)
    with h5py.File(MEM.demo_hdf5(spec["demo_dir"]), "r") as f:
        env_args_str = f["data"].attrs["env_args"]
        k0 = sorted(f["data"].keys())[0]
        ep_meta_info = f["data"][k0].attrs.get("ep_meta_info", "")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = h5py.File(args.out, "w")
    od = out.create_group("data")
    od.attrs["env_args"] = env_args_str
    gi, log = 0, []
    t_start = time.time()
    print(colored(f"[eps {tag}/{args.task}] scenes[{lo}:{hi}] of {args.scenes}", "cyan"), flush=True)

    for i in range(lo, hi):
        seed = int(z["seeds"][i])
        init = scene_init(z, i)
        t_sc = time.time()
        why, kept_src, vis_px = [], None, -1
        # rotate source try-order by seed so the kept set spreads across the 3
        # teleop bases (RCP's demo_1 transfers poorly; Phase-1 report note 6)
        order = [(seed + j) % len(sources) for j in range(len(sources))]
        for k, si in enumerate(order):
            src, c = sources[si], consts[si]
            try:
                seed_globals(seed)
                reset_to(env, init, replace_robot_joints=True, change_to_gr1=False)
                obs0 = settle_env(env, args.settle)
            except Exception as e:
                why.append(f"reset_error:{type(e).__name__}:{e}")
                break
            if k == 0:
                vis_px, vis_by_cam = t0_visible_px(env, spec)
                if vis_px < args.vis_min_px:
                    why.append(f"t0_visibility_fail:{vis_px}px:{json.dumps(vis_by_cam)}")
                    break
            try:
                # v6: DTH keeps the legacy rigid retarget (natural start variance,
                # mirroring the taskdemos' own varied starts). The fixed-q0 blend
                # is applied at the GR1 stage only (run_gr1).
                traj, info = MEM.retarget_onto_drawn(env, spec, src, c)
            except Exception as e:
                why.append(f"{src['key']}:retarget_error:{type(e).__name__}:{e}")
                continue
            ok, states, obs_list = AP.replay_record(env, traj, capture_obs=True)
            if not ok:
                why.append(f"{src['key']}:success_check_false")
                continue
            why.append(f"{src['key']}:success")
            kept_src = src["key"]
            # v6 blend hint for the GR1 stage: the frame by which the q0-anchored
            # blend offset must have decayed to zero — PnP: the demo grasp frame;
            # hinge: first frame the demo's door/handle angle moves (contact).
            if spec["kind"] == "hinge":
                th = np.asarray(c["th_demo"], float)
                moved = np.abs(th - th[0]) > 0.02
                blend_hint = int(np.argmax(moved)) if moved.any() else max(len(traj) // 3, 8)
            else:
                blend_hint = int(c["grasp"])
            save_episode(od, gi, traj, states, obs_list, env, keys, attrs=dict(
                blend_end_hint=blend_hint,
                ep_meta=init["ep_meta"], ep_meta_info=ep_meta_info,
                source_demo=kept_src, style_id=style,
                scene_seed=seed, task=args.task, cell=tag, layout_id=args.layout,
                sibling_tasks="[]",     # filled by --stage sibling
                t0_vis_px=vis_px, t0_vis_px_by_cam=json.dumps(vis_by_cam),
                retarget_info=json.dumps(
                    {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else str(vv))
                     for kk, vv in info.items()})))
            gi += 1
            break
        ok = kept_src is not None
        log.append(dict(seed=seed, task=args.task, feasible=ok, source=kept_src,
                        vis_px=vis_px, reason=";".join(why), secs=round(time.time() - t_sc, 1)))
        print(colored(f"[eps {tag}/{args.task}] seed {seed}: "
                      f"{'KEPT' if ok else 'infeasible'} ({gi} kept, "
                      f"{(time.time() - t_start) / 60:.1f}m) {why[-1] if why else ''}",
                      "green" if ok else "yellow"), flush=True)

    od.attrs["total"] = sum(g["actions"].shape[0] for g in od.values()) if gi else 0
    out.close()
    if gi:
        try:
            import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
            DatasetUtils.extract_action_dict(dataset=args.out)
        except Exception as e:
            print(colored(f"[eps {tag}/{args.task}] action_dict failed (ignored): {e}",
                          "yellow"), flush=True)
    feas = args.out.replace(".hdf5", "_feasibility.json")
    json.dump(dict(cell=tag, task=args.task, scenes=args.scenes, range=[lo, hi],
                   settle=args.settle, vis_min_px=args.vis_min_px,
                   n_kept=gi, entries=log), open(feas, "w"), indent=1)
    print(colored(f"[eps {tag}/{args.task}] DONE {gi}/{hi - lo} kept -> {args.out}\n"
                  f"  feasibility: {feas}", "magenta"), flush=True)


# --- stage: gr1 (replay DemoTwoHand part on GR1, keep successes) -------------------

def run_gr1(args):
    spec = MEM.SPECS[args.task]
    env = build_env2(spec, robots="GR1TwoHand", seg="instance", controller="WHOLE_BODY_MINK_IK")
    keys = env_obs_keys(MEM.demo_hdf5(spec["demo_dir"]), keep_seg=True)
    cams = sorted(k.replace("_segmentation_instance", "") for k in keys if "segmentation" in k)

    with h5py.File(args.aug, "r") as f:
        ea = f["data"].attrs["env_args"]
        ea = json.loads(ea) if isinstance(ea, str) else ea
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        # per-scene grouping: process in scene_seed order so a scene's multi-task
        # siblings stay adjacent in the output and the pass/fail table is per-scene
        demos.sort(key=lambda k: int(f["data"][k].attrs.get("scene_seed", -1)))
    ea.setdefault("env_kwargs", {})["robots"] = "GR1TwoHand"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = h5py.File(args.out, "w")
    od = out.create_group("data")
    od.attrs["env_args"] = json.dumps(ea)
    gi, log = 0, []
    aug = h5py.File(args.aug, "r")["data"]
    t_start = time.time()
    for name in demos[args.start:args.end]:
        g = aug[name]
        seed = int(g.attrs.get("scene_seed", -1))
        t_ep = time.time()
        actions = g["actions"][:]
        init = {"states": g["states"][0], "ep_meta": json.loads(g.attrs["ep_meta"]),
                "model": g.attrs["model_file"], "non_robot_qpos_idx": g.attrs["non_robot_qpos_idx"],
                "qpos_state": g.attrs["initial_qpos"]}
        reason, vis_px = None, -1
        try:
            seed_globals(seed)
            reset_to(env, init, replace_robot_joints=True, change_to_gr1=True)
            obs = settle_env(env, args.settle)     # common hold -> sibling t0 identity
            vis_px, vis_by_cam = t0_visible_px(env, spec)
            if vis_px < args.vis_min_px:
                reason = f"t0_visibility_fail:{vis_px}px:{json.dumps(vis_by_cam)}"
        except Exception as e:
            reason = f"reset_error:{type(e).__name__}:{e}"
        approach_len = 0
        if reason is None:
            # v6: NO approach bridge. The episode starts AT the settled q0 and the
            # DTH action sequence's early segment is BLEND-anchored to it
            # (AP.blend_pregrasp on the ACTIVE arm, offset decaying to zero by the
            # grasp frame — gripper-close detection; fixed 60f window for hinge
            # tasks with no detectable close). Recording starts at t=0 == q0, the
            # same state the eval client starts from (reset+settle, no bridge).
            # v6 FINAL LAW (user): RECORDED approach phase at ~the demo's average
            # speed, executed CLOSED-LOOP from the fixed settled q0 toward the DTH
            # trajectory's first row, until the ACHIEVED eef error is small
            # (--approach_eps); then the demo plays UNCHANGED from its start.
            # Episode = [approach (variable length)] + [full DTH traj] + [tail].
            b_act = AP.ARM_BLOCK[MEM.SPECS[args.task]["arm"]]
            # approach speed = GLOBAL all-task demo average (user: per-task averages
            # made Door's long approach insanely fast) — fixed constant, one knob.
            v_step = float(args.approach_v_step)
            seq = actions
            states, obs_list = [], []
            try:
                acts_rec = []
                # RECORDED OPEN-LOOP approach (proven v5 machinery): pre-interpolated
                # rows from the settled q0 to the DTH traj's first row at the GLOBAL
                # demo-average speed. Open loop is essential — a closed-loop "current
                # +v_step" target sits inside the IK tracking lag (~2.5 cm) and the
                # arm self-throttles (measured: err 47->50 cm over 25 f).
                a_cur = RG._gr1_current_action(obs, actions[0])
                appr_err0 = float(np.linalg.norm(
                    np.asarray(a_cur[b_act:b_act + 3]) - np.asarray(actions[0][b_act:b_act + 3])))
                appr_rows = RG._approach_rows(a_cur, actions[0], v_step,
                                              args.approach_rot_vel, args.approach_max)
                appr_fix = args.approach_fixed
                if appr_fix < 0 and args.task in TASK_RECIPE:
                    appr_fix = TASK_RECIPE[args.task][0]     # v6 default recipe
                if appr_fix >= 0:
                    # per-task recipe: truncate the approach to EXACTLY N frames
                    # (0 = no approach; residual handled by the handoff blend)
                    appr_rows = appr_rows[:appr_fix]
                for row in appr_rows:
                    states.append(env.sim.get_state().flatten())
                    obs_list.append({k: np.asarray(v) for k, v in obs.items()})
                    acts_rec.append(np.asarray(row, np.float64))
                    obs, _, _, _ = env.step(row)
                approach_len = len(appr_rows)
                arm_name = MEM.SPECS[args.task]["arm"]
                appr_errN = float(np.linalg.norm(
                    np.asarray(obs[f"robot0_base_to_{arm_name}_eef_pos"], float)
                    - np.asarray(actions[0][b_act:b_act + 3])))
                # SMOOTH HANDOFF: decay the remaining approach residual (current
                # achieved pose vs the demo's first row) over the demo's first 12
                # frames — kills the approach->demo discontinuity without touching
                # contact geometry (12 f << grasp/contact frames).
                a_now = RG._gr1_current_action(obs, actions[0])
                hb = args.handoff_blend
                if hb <= 0 and args.task in TASK_RECIPE:
                    hb = TASK_RECIPE[args.task][1]
                M = min(hb, max(len(seq) // 4, 4))
                seq = AP.blend_pregrasp(seq.copy(), M, MEM.SPECS[args.task]["arm"], a_now)
                other_arm = "left" if MEM.SPECS[args.task]["arm"] == "right" else "right"
                seq = AP.blend_pregrasp(seq, M, other_arm, a_now)
                if args.task in GRASP_DWELL:
                    Dw, St, Cw = GRASP_DWELL[args.task]
                    gI = AP.detect_grasp_boundary(seq, MEM.SPECS[args.task]["arm"])
                    if 0 < gI < len(seq) - Cw:
                        seq = np.concatenate([
                            seq[:gI],
                            np.repeat(seq[gI - 1][None], Dw, axis=0),
                            np.repeat(seq[gI:gI + Cw], St, axis=0),
                            seq[gI + Cw:]], 0)
                        print(colored(f"[gr1 {args.task}] dwell@{gI} ({Dw}f + close x{St})",
                                      "cyan"), flush=True)
                # then the demo plays (keep-or-drop by the final success check).
                for a in seq:
                    states.append(env.sim.get_state().flatten())
                    obs_list.append({k: np.asarray(v) for k, v in obs.items()})
                    acts_rec.append(np.asarray(a, np.float64))
                    obs, _, _, _ = env.step(a)
                # v5: RECORDED post-success hold tail — freeze at the current pose
                # (grips kept from the last action) so end-of-episode windows survive
                # the w0+SPAN manifest filter and "stay put after done" is in-data.
                for _ in range(args.tail_steps):
                    a = RG._gr1_current_action(obs, seq[-1])
                    states.append(env.sim.get_state().flatten())
                    obs_list.append({k: np.asarray(v) for k, v in obs.items()})
                    acts_rec.append(a)
                    obs, _, _, _ = env.step(a)
                seq = np.stack(acts_rec)   # recorded rows only (aligned with states/obs)
                if not env._check_success():
                    reason = "gr1_success_check_false"
            except Exception as e:
                reason = f"replay_exception:{type(e).__name__}:{e}"
        ok = reason is None
        if ok:
            attrs = dict(ep_meta=g.attrs["ep_meta"], ep_meta_info=g.attrs.get("ep_meta_info", ""),
                         source_demo=g.attrs.get("source_demo", ""),
                         style_id=g.attrs.get("style_id", ""),
                         scene_seed=seed, task=args.task, cell=g.attrs.get("cell", ""),
                         layout_id=int(g.attrs.get("layout_id", -1)),
                         sibling_tasks="[]", t0_vis_px=vis_px,
                         t0_vis_px_by_cam=json.dumps(vis_by_cam), aug_demo=name,
                         approach_len=approach_len, approach_recorded=0)
            save_episode(od, gi, seq, np.array(states), obs_list, env, keys, attrs,
                         flip_seg=True, bbox_cams=cams)
            gi += 1
        log.append(dict(seed=seed, task=args.task, aug_demo=name, gr1_success=ok,
                        reason=reason or "success", vis_px=vis_px,
                        secs=round(time.time() - t_ep, 1)))
        print(colored(f"[gr1 {args.task}] {name} (seed {seed}): "
                      f"{'KEPT' if ok else reason} (appr={approach_len}f "
                      f"err {locals().get('appr_err0', 0)*100:.1f}->{locals().get('appr_errN', 0)*100:.1f}cm, "
                      f"{gi} kept, {(time.time() - t_start) / 60:.1f}m)",
                      "green" if ok else "yellow"), flush=True)
    od.attrs["total"] = sum(d["actions"].shape[0] for d in od.values()) if gi else 0
    out.close()
    if gi:
        try:
            import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
            DatasetUtils.extract_action_dict(dataset=args.out)
        except Exception as e:
            print(colored(f"[gr1 {args.task}] action_dict failed (ignored): {e}", "yellow"), flush=True)
    feas = args.out.replace(".hdf5", "_gr1_feasibility.json")
    json.dump(dict(task=args.task, aug=args.aug, range=[args.start, args.end],
                   settle=args.settle, vis_min_px=args.vis_min_px, n_kept=gi,
                   entries=log), open(feas, "w"), indent=1)
    print(colored(f"[gr1 {args.task}] DONE {gi}/{len(log)} kept -> {args.out}\n"
                  f"  gr1 table: {feas}", "magenta"), flush=True)


# --- stage: sibling (post-pass: sibling_tasks attrs + scene tables) -----------------

def _scene_task_map(files):
    """{scene_seed: {task: [(file, demo_name)]}} over per-task hdf5 files.
    Unreadable files (truncated leftovers of killed jobs) are skipped loudly."""
    m = {}
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                for k in f["data"]:
                    a = f["data"][k].attrs
                    m.setdefault(int(a["scene_seed"]), {}).setdefault(str(a["task"]), []).append((p, k))
        except (OSError, KeyError, RuntimeError) as e:
            print(colored(f"[scene_task_map] SKIP unreadable {p}: {e}", "red"), flush=True)
    return m


def run_sibling(args):
    dth = sorted(glob.glob(os.path.join(args.cell_dir, "dth_*.hdf5")))
    gr1 = sorted(glob.glob(os.path.join(args.cell_dir, "gr1_*.hdf5")))
    for label, files in (("dth", dth), ("gr1", gr1)):
        m = _scene_task_map(files)
        for seed, bytask in m.items():
            tasks = sorted(bytask)
            for task, locs in bytask.items():
                sib = json.dumps([t for t in tasks if t != task])
                for p, k in locs:
                    try:
                        with h5py.File(p, "r+") as f:
                            f["data"][k].attrs["sibling_tasks"] = sib
                    except OSError as e:
                        print(colored(f"[sibling] cannot write {p}: {e}", "red"), flush=True)
        groups = {str(s): sorted(bt) for s, bt in sorted(m.items())}
        json.dump(groups, open(os.path.join(args.cell_dir, f"scene_groups_{label}.json"), "w"),
                  indent=1)
        multi = sum(1 for v in groups.values() if len(v) >= 2)
        print(colored(f"[sibling {label}] {len(groups)} scenes, {multi} with >=2 tasks "
                      f"-> scene_groups_{label}.json", "magenta"), flush=True)
    # per-(scene,task) GR1 pass/fail table (for later eval-manifest GR1 refinement)
    table = {}
    for p in sorted(glob.glob(os.path.join(args.cell_dir, "*_gr1_feasibility.json"))):
        for e in json.load(open(p))["entries"]:
            table.setdefault(str(e["seed"]), {})[e["task"]] = \
                dict(gr1_success=e["gr1_success"], reason=e["reason"])
    out = os.path.join(args.cell_dir, "gr1_scene_table.json")
    json.dump(table, open(out, "w"), indent=1)
    print(colored(f"[sibling] GR1 per-scene pass/fail table -> {out}", "magenta"), flush=True)


# --- stage: audit -------------------------------------------------------------------

def run_audit(args):
    rep = dict(cell_dir=args.cell_dir)
    dth = sorted(glob.glob(os.path.join(args.cell_dir, "dth_*.hdf5")))
    gr1 = sorted(glob.glob(os.path.join(args.cell_dir, "gr1_*.hdf5")))

    # (d) multi-task counterfactual audit + bitwise sibling t0 identity
    m = _scene_task_map(dth)
    multi = {s: bt for s, bt in m.items() if len(bt) >= 2}
    rep["dth_scenes"] = len(m)
    rep["dth_multi_task_scenes"] = len(multi)
    ident, mismatch = 0, []
    for seed, bytask in multi.items():
        t0s = []
        for task, locs in sorted(bytask.items()):
            p, k = locs[0]
            with h5py.File(p, "r") as f:
                t0s.append((task, f["data"][k]["states"][0]))
        base = t0s[0][1]
        if all(x.shape == base.shape and np.array_equal(x, base) for _, x in t0s[1:]):
            ident += 1
        else:
            mismatch.append(seed)
    rep["dth_sibling_t0_bitwise_identical"] = f"{ident}/{len(multi)}"
    rep["dth_sibling_t0_mismatch_seeds"] = mismatch[:10]
    mg = _scene_task_map(gr1)
    rep["gr1_scenes"] = len(mg)
    rep["gr1_multi_task_scenes"] = sum(1 for v in mg.values() if len(v) >= 2)
    # GR1 siblings: object part of t0 must match (robot was swapped)
    g_ident, g_tot = 0, 0
    for seed, bytask in mg.items():
        if len(bytask) < 2:
            continue
        g_tot += 1
        vecs = []
        for task, locs in sorted(bytask.items()):
            p, k = locs[0]
            with h5py.File(p, "r") as f:
                ep = f["data"][k]
                vecs.append(ep.attrs["initial_qpos"][ep.attrs["non_robot_qpos_idx"]])
        if all(np.allclose(v, vecs[0], atol=1e-10) for v in vecs[1:]):
            g_ident += 1
    rep["gr1_sibling_object_t0_identical"] = f"{g_ident}/{g_tot}"

    # (a) init-distribution coverage vs the frozen eval manifest (canonical task)
    if args.layout is not None:
        cell = CELLS[args.layout]
        man = os.path.join(EVAL_MANIFEST_DIR, f"{cell['host']}__seen.npz")
        sc = sorted(glob.glob(os.path.join(args.cell_dir, "scenes_*.npz")))
        if os.path.exists(man) and sc:
            train = [v for p in sc for v in np.load(p, allow_pickle=True)["object_state"]]
            ev = list(np.load(man, allow_pickle=True)["object_state"])
            dims = [v.shape[0] for v in train]
            modal = max(set(dims), key=dims.count)
            tr = np.stack([v for v in train if v.shape[0] == modal])
            evs = np.stack([v for v in ev if v.shape[0] == modal])
            if len(tr) >= 10:
                d_int = [np.sort(np.linalg.norm(tr - tr[i], axis=1))[1] for i in range(len(tr))]
                med = float(np.median(d_int))
                xnn = [float(np.min(np.linalg.norm(tr - e, axis=1))) / (med + 1e-12) for e in evs]
                rep["coverage"] = dict(task=cell["host"], n_train=len(tr), n_eval=len(evs),
                                       internal_nn_median=med,
                                       eval_xnn_median=float(np.median(xnn)),
                                       eval_xnn_max=float(np.max(xnn)))
            else:
                rep["coverage"] = f"skipped: only {len(tr)} scene draws (<10)"

    # (c) visibility-assert pass stats from the feasibility logs
    vis = dict()
    for p in sorted(glob.glob(os.path.join(args.cell_dir, "*_feasibility.json"))):
        j = json.load(open(p))
        ent = j["entries"]
        fails = sum(1 for e in ent if "t0_visibility_fail" in str(e.get("reason", "")))
        vis[os.path.basename(p)] = dict(n=len(ent), vis_fail=fails,
                                        vis_px_min=min((e.get("vis_px", -1) for e in ent
                                                        if e.get("vis_px", -1) >= 0), default=None))
    rep["visibility"] = vis

    out = os.path.join(args.cell_dir, "audit.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(colored(json.dumps(rep, indent=1), "cyan"), flush=True)
    print(colored(f"[audit] -> {out}", "magenta"), flush=True)


# --- stage: video (validation rule: show everything as video) -------------------------

def run_video(args):
    import imageio
    out_dir = args.video_dir or os.path.dirname(os.path.abspath(args.hdf5))
    os.makedirs(out_dir, exist_ok=True)
    try:
        h5py.File(args.hdf5, "r").close()
    except OSError as e:
        print(colored(f"[video] SKIP unreadable {args.hdf5}: {e}", "red"), flush=True)
        return
    with h5py.File(args.hdf5, "r") as f:
        names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))[:args.n_videos]
        for k in names:
            ep = f["data"][k]
            ims = ep["obs"][f"{VIS_CAM}_image"][()]
            a = ep.attrs
            stem = (f"{os.path.splitext(os.path.basename(args.hdf5))[0]}_{k}"
                    f"_seed{a.get('scene_seed', '?')}")
            path = os.path.join(out_dir, stem + ".mp4")
            with imageio.get_writer(path, fps=20, codec="libx264", quality=7) as w:
                for t in range(len(ims)):
                    w.append_data(ims[t])
            print(colored(f"[video] {path} ({len(ims)} frames, task={a.get('task', '?')}, "
                          f"src={a.get('source_demo', '?')})", "magenta"), flush=True)


# --- stage: plan (emit full-run shard commands) ----------------------------------------

def run_plan(args):
    py = ".venv/bin/python robocasa/scripts/gen_unified.py"
    print(f"# full run -> {args.out_dir} ; scenes per cell = {args.scenes_per_cell}, "
          f"{args.workers} workers over GPUs 0,1 (MUJOCO_EGL_DEVICE_ID)")
    for layout, cell in CELLS.items():
        tag = cell_tag(layout, cell["style"])
        n = args.scenes_per_cell
        chunk = (n + args.scene_jobs - 1) // args.scene_jobs
        for j in range(args.scene_jobs):
            s = j * chunk
            print(f"{py} --stage scenes --layout {layout} --start {s} "
                  f"--need {min(chunk, n - s)} --out_dir {args.out_dir}")
        for task in LAYOUT_TASKS[layout]:
            for j in range(args.ep_jobs):
                s, e = j * ((n + args.ep_jobs - 1) // args.ep_jobs), (j + 1) * ((n + args.ep_jobs - 1) // args.ep_jobs)
                print(f"{py} --stage episodes --layout {layout} --task {task} "
                      f"--scenes {args.out_dir}/{tag}/scenes_ALL.npz --scene_start {s} --scene_end {e} "
                      f"--out {args.out_dir}/{tag}/dth_{task}_p{j}.hdf5")
                print(f"{py} --stage gr1 --task {task} "
                      f"--aug {args.out_dir}/{tag}/dth_{task}_p{j}.hdf5 "
                      f"--out {args.out_dir}/{tag}/gr1_{task}_p{j}.hdf5")
        print(f"{py} --stage sibling --cell_dir {args.out_dir}/{tag}")
        print(f"{py} --stage audit --layout {layout} --cell_dir {args.out_dir}/{tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["scenes", "episodes", "gr1", "merge", "sibling", "audit",
                             "video", "plan"])
    ap.add_argument("--layout", type=int, default=None, choices=sorted(CELLS))
    ap.add_argument("--style", default=None, help="override cell style (default: canonical)")
    ap.add_argument("--task", default=None, choices=SEEN_TASKS)
    ap.add_argument("--start", type=int, default=0, help="scenes: stream offset; gr1: demo start")
    ap.add_argument("--end", type=int, default=10**9, help="gr1: demo end")
    ap.add_argument("--need", type=int, default=1, help="scenes: draws in this shard")
    ap.add_argument("--scenes", default=None, help="episodes: scenes npz (or eval manifest npz)")
    ap.add_argument("--scene_start", type=int, default=0)
    ap.add_argument("--scene_end", type=int, default=10**9)
    ap.add_argument("--aug", default=None, help="gr1: DemoTwoHand part hdf5")
    ap.add_argument("--out", default=None, help="episodes/gr1/merge: output hdf5")
    ap.add_argument("--out_dir", default="/mnt/ddn/tmp/gen_unified_smoke")
    ap.add_argument("--cell_dir", default=None, help="sibling/audit: cell directory")
    ap.add_argument("--settle", type=int, default=3,
                    help="unrecorded closed-loop hold steps after every reset (>=1 ENFORCED)")
    ap.add_argument("--vis_min_px", type=int, default=10,
                    help="t=0 visibility assert: min anchor pixels in a 256px seg render")
    ap.add_argument("--approach_eps", type=float, default=0.025,
                    help="gr1: recorded approach ends when achieved eef error < this (m)")
    ap.add_argument("--veg_set", default=None,
                    help="scenes: restrict the vegetable to a category pool (comma list, "
                         "e.g. 'tomato,corn,carrot,eggplant' — augmentation/demo-pool law)")
    ap.add_argument("--approach_fixed", type=int, default=-1,
                    help="gr1: EXACT approach frame count per task recipe (-1 = auto "
                         "velocity-capped length)")
    ap.add_argument("--handoff_blend", type=int, default=0,
                    help="gr1: handoff blend window (demo frames); 0 = per-task TASK_RECIPE default")
    ap.add_argument("--approach_v_step", type=float, default=0.010,
                    help="gr1: approach speed (m/frame @20Hz) = GLOBAL all-task demo "
                         "average (Door 0.54/Cab 0.81/Fau 0.50/Pla 0.67 cm/f -> 0.63)")
    ap.add_argument("--tail_steps", type=int, default=0,
                    help="gr1: RECORDED post-sequence hold frames (pose frozen, grips kept) — "
                         "success is re-checked after the tail")
    ap.add_argument("--approach_vel", type=float, default=0.01)
    ap.add_argument("--approach_rot_vel", type=float, default=0.06)
    ap.add_argument("--approach_max", type=int, default=60)
    ap.add_argument("--merge_parts", nargs="+", default=None)
    ap.add_argument("--hdf5", default=None, help="video: episode file")
    ap.add_argument("--n_videos", type=int, default=3)
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--scenes_per_cell", type=int, default=170)
    ap.add_argument("--scene_jobs", type=int, default=6)
    ap.add_argument("--ep_jobs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    assert args.settle >= 1, "settle>=1 required (post-reset render misses dynamic objects)"
    try:
        if args.stage == "scenes":
            run_scenes(args)
        elif args.stage == "episodes":
            run_episodes(args)
        elif args.stage == "gr1":
            run_gr1(args)
        elif args.stage == "merge":
            AP.merge(args.merge_parts, args.out)
        elif args.stage == "sibling":
            run_sibling(args)
        elif args.stage == "audit":
            run_audit(args)
        elif args.stage == "video":
            run_video(args)
        else:
            run_plan(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
