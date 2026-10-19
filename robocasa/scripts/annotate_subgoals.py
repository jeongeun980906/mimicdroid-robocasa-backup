"""Subgoal / language annotation for MimicDroid "play" training data.

For each long (~14k-step) teleop play trajectory in a SinkEnvForPlay dataset,
replay the recorded simulator states (subsampled), read named environment state
(cabinet door angles, sink faucet handle, object heights + grasp distances), and
detect subgoal boundaries via thresholded events with hysteresis. Each detected
segment is labeled with a templated language string and written to a sidecar JSON
file. The original hdf5 files are never modified.

Usage:
  cd /mnt/ddn/mimicdroid-robocasa && MUJOCO_GL=egl uv run python \
      robocasa/scripts/annotate_subgoals.py 003 004
  # or with explicit paths / output dir:
  ... annotate_subgoals.py --out /mnt/ddn/tmp/subgoal_annotations 003

Method (per file):
  1. Build env once via robocasa.utils.mimicdroid_utils.make_env + reset_to,
     keeping the original "DemoTwoHand" embodiment (state queries only).
  2. Per subsampled frame: set_state_from_flattened(states[t]) + sim.forward(),
     then record door state (cab_sink.get_door_state), faucet state
     (sink.get_handle_state), and per-object z height + min(eef<->obj) distance.
  3. Detect events with hysteresis (require N consecutive subsampled frames):
        door open/close, faucet on/off, pick (object grasped + rising),
        place (object settles + gripper far). Classify the place destination
        fixture/receptacle.
  4. Label each segment with templates using env.get_obj_lang for readable nouns.
  5. Write list of {start, end, lang, ...} per demo to <out>/<file_id>.json.
"""
import os, sys, json, glob, argparse, warnings, logging
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
warnings.filterwarnings("ignore")

# robosuite logs setup ("No private macro file found!") and per-part
# controller-config noise to its own named logger (stdout), which the
# ``warnings`` machinery does not cover. The logger is created with a hardcoded
# INFO console level at robosuite import time, and the import itself emits a few
# one-time banners, so we (a) suppress stdout during the first robosuite import
# and (b) raise the logger to ERROR, leaving only our per-demo segment summary.
import contextlib, io  # noqa: E402

import h5py
import numpy as np

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
    io.StringIO()
):
    import robosuite.utils.log_utils  # noqa: E402,F401

    logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

    import robocasa  # noqa: F401  (registers envs)
    import robocasa.utils.object_utils as OU
    from robocasa.utils.mimicdroid_utils import make_env, reset_to, EnvArgs

# Re-assert after robocasa's own imports in case anything reset the level.
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

# --------------------------------------------------------------------------- #
# Tunable detection constants
# --------------------------------------------------------------------------- #
SUBSAMPLE = 8           # query every Nth recorded frame (~14k -> ~1800 samples)
DEBOUNCE = 3            # consecutive subsampled frames required to commit an event

DOOR_OPEN_TH = 0.5      # door considered open above this (normalized 0..1)
DOOR_CLOSED_TH = 0.1    # door considered closed below this

# Grasp distance is measured from the gripper site to the object's *body origin*
# (its center), so a uniform threshold systematically misses large objects whose
# graspable surface is far from the center (e.g. the tall olive_oil_bottle whose
# center sits ~10 cm below the grip point => min eef<->center distance ~0.15 m,
# above any 0.13 m cutoff). We therefore make the near/far cutoffs *per object*,
# adding the object's effective radius (max of its horizontal radius and half its
# vertical extent) to a fixed base buffer. This recovers picks of the bottle,
# strawberry, bread, etc. that a single global threshold drops.
GRASP_BASE_TH = 0.13    # base buffer added to each object's effective radius (near)
RELEASE_MARGIN = 0.10   # extra distance beyond the near threshold => released (far)
PICK_Z_RISE = 0.08      # object must rise at least this (m) to count as a real pick
SETTLE_Z_EPS = 0.04     # |z - z_prev| below this over the debounce window => settled

# Object-motion-based pick detection (per SUBSAMPLE=8 step). A pick_place is the
# interval where the object actually translates; boundaries come straight off the
# object trajectory so they're correct regardless of the grasp/eef signal.
MOVE_STEP_TH = 0.006    # per-subsample displacement (m) above which the object is "moving"
MOVE_GAP_TOL = 4        # bridge brief pauses (subsamples) within one transport
MOVE_MIN_TOTAL = 0.06   # min net displacement (m) for a real pick_place
MOVE_MIN_DUR = 2        # min subsamples of motion

# Faucet detection from handle-joint motion (water_on flag is ~always True, unusable)
FAUCET_STEP_TH = 0.008  # per-subsample handle change above which it's "turning"
FAUCET_GAP_TOL = 3      # bridge brief pauses within one turn
FAUCET_MIN_NET = 0.05   # min net handle change for a real on/off turn

# Default training data root
DATA_ROOT = "/mnt/ddn/dataset/MimicDroidDataset/training"
DEFAULT_OUT = "/mnt/ddn/tmp/subgoal_annotations"


# --------------------------------------------------------------------------- #
# Per-frame state extraction
# --------------------------------------------------------------------------- #
def _clamp01(x: float) -> float:
    """Door normalization can return slightly <0 or >1; clamp for thresholding."""
    return float(min(1.0, max(0.0, x)))


def object_grasp_thresholds(
    env, obj_names: List[str]
) -> Dict[str, Tuple[float, float]]:
    """Per-object (near, far) eef<->center distance cutoffs.

    The grasp distance is measured to the object's body origin (center), so a
    large object's surface can be far from its center. We size the threshold by
    the object's *effective radius* = max(horizontal_radius, half vertical
    extent) so that tall (bottle) and wide (corn) objects are detected as
    grasped, while small objects keep a tight cutoff.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for name in obj_names:
        o = env.objects[name]
        half_h = float(max(abs(o.top_offset[2]), abs(o.bottom_offset[2])))
        eff_r = max(float(o.horizontal_radius), half_h)
        near = GRASP_BASE_TH + eff_r
        far = near + RELEASE_MARGIN
        out[name] = (near, far)
    return out


def extract_frame_state(env, obj_names: List[str]) -> Dict[str, Any]:
    """Read named state from the *current* (already-forwarded) sim state."""
    sim = env.sim
    eef_r = env.robots[0].eef_site_id["right"]
    eef_l = env.robots[0].eef_site_id["left"]
    r_pos = np.array(sim.data.site_xpos[eef_r])
    l_pos = np.array(sim.data.site_xpos[eef_l])

    door = {"left_door": 0.0, "right_door": 0.0}
    if env.cab_sink is not None:
        raw = env.cab_sink.get_door_state(env)
        door = {k: _clamp01(v) for k, v in raw.items()}

    handle = env.sink.get_handle_state(env)

    objs: Dict[str, Dict[str, float]] = {}
    for name in obj_names:
        op = np.array(sim.data.body_xpos[env.obj_body_id[name]])
        d = min(
            float(np.linalg.norm(r_pos - op)),
            float(np.linalg.norm(l_pos - op)),
        )
        objs[name] = {"z": float(op[2]), "dist": d, "xy": op[:2].tolist()}

    return {
        "left_door": door["left_door"],
        "right_door": door["right_door"],
        "water_on": bool(handle["water_on"]),
        "handle_joint": float(handle["handle_joint"]),
        "objs": objs,
    }


# --------------------------------------------------------------------------- #
# Source / destination classification (position-based)
# --------------------------------------------------------------------------- #
# NOTE on the previous (buggy) approach: classification used
# ``check_obj_fixture_contact`` / ``is_on_top_of``. Those rely on live MuJoCo
# contact data and the fixtures' interior "site" bbox, neither of which survives
# a replay where the state is injected via ``set_state_from_flattened`` + a
# single ``forward()``: resting objects have no active contacts and the sink /
# counter int-site bboxes are degenerate. As a result every object fell through
# to the hardcoded ``"on", "the counter"`` default -- which is exactly the
# "everything is from the counter" symptom. We instead classify purely from the
# object's world position relative to fixture regions and plate (container)
# positions, which is deterministic and replay-stable.

# Region bounds for the cabinet: an object is "in the cabinet" when it sits high
# (on/above the cabinet shelf) and within the cabinet footprint in x/y.
_CAB_XY_TOL = 0.12
_CAB_Z_MIN = 1.30            # cabinet shelf is well above sink / counter level
# Region bounds for the sink basin: low z and within the sink footprint.
_SINK_XY_TOL = 0.05
_SINK_Z_PAD = 0.02           # below the sink rim (int-site top) + small pad
# Plate (container) association tolerances.
_PLATE_XY_FRAC = 0.9         # within this fraction of the plate horizontal radius
_PLATE_Z_LO = -0.05          # object center may be slightly below plate center
_PLATE_Z_HI = 0.18           # ... up to this much above it (object sits on plate)


def _fixture_xy_z(env, attr: str) -> Optional[Tuple[float, float, float, float, float]]:
    """Return (x_min, x_max, y_min, y_max, z_top) for a fixture's interior, or None."""
    fix = getattr(env, attr, None)
    if fix is None:
        return None
    try:
        p0, px, py, pz = fix.get_int_sites(relative=False)
    except Exception:
        return None
    pts = np.array([p0, px, py, pz], dtype=float)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].min()),
        float(pts[:, 1].max()),
        float(pts[:, 2].max()),
    )


def _container_names(env) -> List[str]:
    """Object names that act as receptacles/plates (named '*_container')."""
    return [n for n in env.objects if "container" in n]


def _in_sink_region(env, p) -> bool:
    """True if world point p lies inside the sink basin footprint (low z)."""
    sink = _fixture_xy_z(env, "sink")
    if sink is None:
        return False
    x0, x1, y0, y1, ztop = sink
    return (
        p[2] <= ztop + _SINK_Z_PAD
        and x0 - _SINK_XY_TOL <= p[0] <= x1 + _SINK_XY_TOL
        and y0 - _SINK_XY_TOL <= p[1] <= y1 + _SINK_XY_TOL
    )


def classify_location(env, obj_name: str) -> Tuple[str, str]:
    """Return (preposition, readable location) using the official MimicDroid task
    vocabulary (PnPSinkTo<RightCounterPlate|Cabinet|MicrowaveTop>): the source is
    the SINK, destinations are the counter plate / cabinet / microwave top.

    A plate (container) sitting INSIDE the sink basin reads as "the sink" -- only a
    plate resting on the counter reads as "the plate". Order: cabinet (high) first,
    then sink (basin / plate-in-basin), then counter plate, else the counter.
    """
    op = np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]], dtype=float)

    # 1) up in the cabinet (high shelf within cabinet footprint)
    cab = _fixture_xy_z(env, "cab_sink")
    if cab is not None:
        x0, x1, y0, y1, _ = cab
        if (
            op[2] >= _CAB_Z_MIN
            and x0 - _CAB_XY_TOL <= op[0] <= x1 + _CAB_XY_TOL
            and y0 - _CAB_XY_TOL <= op[1] <= y1 + _CAB_XY_TOL
        ):
            return "in", "the cabinet"

    # 2) in the sink -- either the object itself or the plate it rests on sits in
    #    the sink basin. Both read as "the sink" (matches PnPSinkTo... source).
    best = None
    best_d = float("inf")
    for c in _container_names(env):
        if c == obj_name:
            continue
        rp = np.array(env.sim.data.body_xpos[env.obj_body_id[c]], dtype=float)
        radius = env.objects[c].horizontal_radius * _PLATE_XY_FRAC
        dxy = float(np.linalg.norm(op[:2] - rp[:2]))
        dz = float(op[2] - rp[2])
        if dxy < radius and _PLATE_Z_LO < dz < _PLATE_Z_HI and dxy < best_d:
            best, best_d = c, dxy
    if _in_sink_region(env, op) or (best is not None and _in_sink_region(
        env, np.array(env.sim.data.body_xpos[env.obj_body_id[best]], dtype=float)
    )):
        return "in", "the sink"

    # 3) resting on a counter plate (a container that is NOT in the sink)
    if best is not None:
        return "on", "the plate"

    # 4) default: the counter
    return "on", "the counter"


# --------------------------------------------------------------------------- #
# Event detection over the subsampled timeline (with hysteresis)
# --------------------------------------------------------------------------- #
def _debounced(condition_series: List[bool], i: int, n: int) -> bool:
    """True if condition holds for n consecutive samples ending at index i."""
    if i + 1 < n:
        return False
    return all(condition_series[i - k] for k in range(n))


def detect_events(
    env,
    frames: List[Dict[str, Any]],
    sample_t: List[int],
    obj_names: List[str],
    states: np.ndarray,
    grasp_th: Dict[str, Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """Return a chronologically ordered list of subgoal segments.

    Each segment: {start, end, lang, type, ...}. start/end are ORIGINAL frame
    indices (not subsampled indices).
    """
    events: List[Dict[str, Any]] = []  # (sample_idx, type, payload)

    # ---- doors ----
    for side in ("left", "right"):
        key = f"{side}_door"
        series = [f[key] for f in frames]
        is_open = [v > DOOR_OPEN_TH for v in series]
        is_closed = [v < DOOR_CLOSED_TH for v in series]
        state_open = is_open[0]  # initialize from first frame
        for i in range(len(series)):
            if not state_open and _debounced(is_open, i, DEBOUNCE):
                events.append((i, "door_open", {"side": side}))
                state_open = True
            elif state_open and _debounced(is_closed, i, DEBOUNCE):
                events.append((i, "door_close", {"side": side}))
                state_open = False

    # ---- faucet (handle-joint motion, not the water_on flag) ----
    # The water_on flag is a threshold on the handle that stays True for ~all of
    # the demo, so it cannot mark individual turns. We instead detect intervals
    # where the handle joint ACTUALLY moves; the net direction gives on (handle
    # opens further, +) vs off (handle closes, -). start = motion start, end =
    # motion end (mirrors the object-motion pick detection).
    hj = [f["handle_joint"] for f in frames]
    hstep = np.abs(np.diff(hj))
    hmoving = hstep > FAUCET_STEP_TH
    i, m = 0, len(hmoving)
    while i < m:
        if not hmoving[i]:
            i += 1
            continue
        j, k, gap = i, i, 0
        while k < m:
            if hmoving[k]:
                j, gap = k, 0
            else:
                gap += 1
                if gap > FAUCET_GAP_TOL:
                    break
            k += 1
        s_idx, e_idx = i, min(j + 1, len(hj) - 1)
        net = hj[e_idx] - hj[s_idx]
        if abs(net) > FAUCET_MIN_NET:
            etype = "faucet_on" if net > 0 else "faucet_off"
            events.append((s_idx, etype, {"_end": e_idx}))
        i = k

    # ---- pick / place ----
    #
    # Detection runs an independent state machine per object (robust to one
    # object's release being missed), collecting candidate pick_place events.
    # Two bugs from the original are fixed here:
    #   * the loose ``z[i] - min(z[:i+1]) > PICK_Z_RISE`` clause false-fired
    #     whenever an object merely sat above its *historical* minimum (no grasp);
    #     we now require a genuine lift relative to a short *local* pre-grasp
    #     baseline (min z over the few samples leading into the grasp), which
    #     both rejects the false positives and recovers real low lifts that a
    #     single-frame baseline misses.
    #   * independent per-object machines could emit several overlapping
    #     pick_place segments for the *same* physical motion when objects sit
    #     close together (the "two segments with the same start" symptom). A
    #     post-pass below deduplicates overlapping picks, keeping the object that
    #     actually moved the most (largest peak lift while grasped).
    # Pick/place is detected from OBJECT MOTION, not the eef grasp signal. A
    # pick_place for an object is the interval where the object itself actually
    # translates (grasp -> transport -> release). Reading the boundaries straight
    # off the object's trajectory gives correct start/end even when the grasp /
    # eef-distance signal is ambiguous (e.g. two hands running concurrent subgoals,
    # or a tall bottle whose center never gets close to the eef site). start = the
    # frame the object begins moving, end = the frame it settles at the target.
    raw_picks: List[Tuple[int, int, str, float]] = []  # (start_s, end_s, name, total_move)
    for name in obj_names:
        pos = np.array(
            [[f["objs"][name]["xy"][0], f["objs"][name]["xy"][1], f["objs"][name]["z"]]
             for f in frames]
        )
        if len(pos) < 3:
            continue
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        moving = step > MOVE_STEP_TH
        i, m = 0, len(moving)
        while i < m:
            if not moving[i]:
                i += 1
                continue
            # extend the motion interval, bridging brief pauses up to MOVE_GAP_TOL
            j, k, gap = i, i, 0
            while k < m:
                if moving[k]:
                    j, gap = k, 0
                else:
                    gap += 1
                    if gap > MOVE_GAP_TOL:
                        break
                k += 1
            s_idx, e_idx = i, min(j + 1, len(pos) - 1)
            total = float(np.linalg.norm(pos[e_idx] - pos[s_idx]))
            if total > MOVE_MIN_TOTAL and (e_idx - s_idx) >= MOVE_MIN_DUR:
                raw_picks.append((s_idx, e_idx, name, total))
            i = k

    # Deduplicate overlapping picks: keep the one with the largest lift in any
    # group of time-overlapping candidate segments.
    raw_picks.sort(key=lambda p: p[0])
    chosen: List[Tuple[int, int, str, float]] = []
    for pk in raw_picks:
        ov = [c for c in chosen if pk[0] <= c[1] and c[0] <= pk[1]]
        if not ov:
            chosen.append(pk)
            continue
        # overlaps an existing pick: keep whichever moved the most
        best = max(ov + [pk], key=lambda p: p[3])
        for c in ov:
            chosen.remove(c)
        chosen.append(best)
    chosen.sort(key=lambda p: p[0])

    for pick_sample, place_sample, name, _ in chosen:
        # classify the source at the grasp-approach frame (object still resting
        # at its origin) and the destination at the settled/released frame.
        # sample_t maps subsample idx -> original frame idx.
        env.sim.set_state_from_flattened(states[sample_t[pick_sample]])
        env.sim.forward()
        _, src = classify_location(env, name)
        env.sim.set_state_from_flattened(states[sample_t[place_sample]])
        env.sim.forward()
        prep, dst = classify_location(env, name)
        events.append(
            (
                pick_sample,
                "pick_place",
                {
                    "obj": name,
                    "place_sample": place_sample,
                    "prep": prep,
                    "dst": dst,
                    "src": src,
                },
            )
        )

    # sort by start sample, build labeled segments with original-frame ranges
    events.sort(key=lambda e: e[0])
    segments: List[Dict[str, Any]] = []
    for start_s, etype, payload in events:
        extra: Dict[str, Any] = {}
        if etype in ("door_open", "door_close"):
            verb = "open" if etype == "door_open" else "close"
            lang = f"{verb} the {payload['side']} cabinet door"
            end_s = min(start_s + DEBOUNCE, len(sample_t) - 1)
        elif etype in ("faucet_on", "faucet_off"):
            onoff = "on" if etype == "faucet_on" else "off"
            lang = f"turn {onoff} the sink faucet"
            end_s = payload.get("_end", min(start_s + DEBOUNCE, len(sample_t) - 1))
        else:  # pick_place
            obj_lang = env.get_obj_lang(payload["obj"])
            lang = (
                f"pick the {obj_lang} from {payload['src']} and place it "
                f"{payload['prep']} {payload['dst']}"
            )
            end_s = payload["place_sample"]
            extra = {
                "obj": payload["obj"],
                "obj_lang": obj_lang,
                "src": payload["src"],
                "dst": payload["dst"],
            }
        segments.append(
            {
                "start": int(sample_t[start_s]),
                "end": int(sample_t[end_s]),
                "lang": lang,
                "type": etype,
                **extra,
            }
        )
    # --- contiguity post-pass (MimicGen convention) ------------------------ #
    # Subtasks are contiguous: every frame between two consecutive events
    # belongs to the UPCOMING subgoal. The event detectors above place a
    # segment's ``start`` at the grasp/lift (or door/faucet) moment, leaving the
    # approach/transport motion that PRECEDES the event as an unlabeled gap
    # ("transition"). That approach is actually part of the upcoming subgoal, so
    # we extend each segment's ``start`` backward to the frame right after the
    # previous segment's ``end`` (or 0 for the first). The detected ``end`` (the
    # event-completion frame) is kept as-is, and all other fields (lang, type,
    # obj, src, dst, ...) are preserved. The trailing idle after the final
    # event's ``end`` is intentionally left unlabeled.
    #
    # We order the timeline by completion frame (``end``), not by ``start``.
    # Two-handed teleop can run two subgoals concurrently (e.g. one hand turns
    # the faucet while the other transports a bottle), so the raw start/end
    # spans can overlap or nest. Tiling by completion order assigns each frame to
    # the subgoal that completes next, which both removes the gaps and yields a
    # well-defined non-overlapping cover. Each segment still ends exactly at its
    # own detected event-completion frame.
    segments.sort(key=lambda s: (s["end"], s["start"]))

    # Boundary lag: hold each label ~BOUNDARY_LAG frames PAST its event-completion
    # frame before switching to the next subgoal, so transitions don't feel abrupt
    # (the raw event fires the instant a threshold is crossed). At 20 Hz, 30 frames
    # ~= 1.5 s. The lagged boundary is clamped so it never reaches the next
    # segment's own completion frame (which would collapse it).
    BOUNDARY_LAG = 30
    n = len(segments)
    lagged_end = []
    for i, s in enumerate(segments):
        raw = int(s["end"])
        if i < n - 1:
            cap = int(segments[i + 1]["end"]) - 1   # stay before next event completes
            lagged_end.append(max(raw, min(raw + BOUNDARY_LAG, cap)))
        else:
            lagged_end.append(raw + BOUNDARY_LAG)
    # keep monotonic (in case caps forced a smaller value than a prior end)
    for i in range(1, n):
        lagged_end[i] = max(lagged_end[i], lagged_end[i - 1] + 1)

    contiguous: List[Dict[str, Any]] = []
    prev_end = -1
    for s, end in zip(segments, lagged_end):
        new_start = prev_end + 1
        if new_start > end:
            new_start = end
        contiguous.append({**s, "start": int(new_start), "end": int(end)})
        prev_end = int(end)

    # Sanity: starts are non-decreasing and each tiles onto the previous end.
    for i in range(1, len(contiguous)):
        assert contiguous[i]["start"] == contiguous[i - 1]["end"] + 1, (
            f"non-contiguous segments at {i}"
        )
        assert contiguous[i]["start"] >= contiguous[i - 1]["start"], (
            "segment starts must be non-decreasing"
        )

    return contiguous


# --------------------------------------------------------------------------- #
# Per-demo / per-file drivers
# --------------------------------------------------------------------------- #
def annotate_demo(env, g, obj_names: List[str]) -> List[Dict[str, Any]]:
    states = g["states"][:]
    n = len(states)
    sample_t = list(range(0, n, SUBSAMPLE))
    if sample_t[-1] != n - 1:
        sample_t.append(n - 1)

    grasp_th = object_grasp_thresholds(env, obj_names)

    frames: List[Dict[str, Any]] = []
    for t in sample_t:
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        frames.append(extract_frame_state(env, obj_names))

    return detect_events(env, frames, sample_t, obj_names, states, grasp_th)


def annotate_file(in_path: str, out_dir: str, file_id: str) -> Dict[str, Any]:
    f = h5py.File(in_path, "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))

    ep0 = json.loads(f["data"][demos[0]].attrs["ep_meta"])
    obj_names = [c["name"] for c in ep0["object_cfgs"]]
    graspable = {
        c["name"] for c in ep0["object_cfgs"] if c.get("graspable") is True
    }
    pick_objs = [n for n in obj_names if n in graspable]

    ea = EnvArgs(robots="DemoTwoHand", render=False, control_freq=20,
                 use_camera_obs=False)
    # robocasa's env build + reset_to emit stray debug prints (e.g. the
    # ``print("arm", ...)`` in kitchen.py's object-settling loop) to stdout,
    # which the ``warnings`` / logger suppression above does not catch. Redirect
    # stdout around these calls so only our per-demo segment summary survives.
    with contextlib.redirect_stdout(io.StringIO()):
        env, _ = make_env(file_name=in_path, env_args=ea)

    result: Dict[str, Any] = {}
    for dm in demos:
        g = f["data"][dm]
        ep = json.loads(g.attrs["ep_meta"])
        with contextlib.redirect_stdout(io.StringIO()):
            reset_to(
                env,
                {
                    "states": g["states"][0],
                    "ep_meta": ep,
                    "model": g.attrs["model_file"],
                    "non_robot_qpos_idx": g.attrs["non_robot_qpos_idx"],
                    "qpos_state": g.attrs["initial_qpos"],
                },
                replace_robot_joints=False,
                change_to_gr1=False,
            )
        segs = annotate_demo(env, g, pick_objs)
        result[dm] = segs
        print(
            f"    {dm}: {g['states'].shape[0]} frames -> {len(segs)} segments",
            flush=True,
        )

    env.close()
    f.close()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{file_id}.json")
    with open(out_path, "w") as fo:
        json.dump(
            {"file": in_path, "subsample": SUBSAMPLE, "demos": result},
            fo,
            indent=2,
        )
    return {"out_path": out_path, "demos": result}


# --------------------------------------------------------------------------- #
def _resolve_inputs(tokens: List[str]) -> List[Tuple[str, str]]:
    """Map CLI tokens (ids like '003' or full paths) to (file_id, path)."""
    out: List[Tuple[str, str]] = []
    for tok in tokens:
        if os.path.isfile(tok):
            file_id = os.path.basename(os.path.dirname(tok)) or os.path.basename(tok)
            out.append((file_id, tok))
        else:
            p = os.path.join(DATA_ROOT, tok, "demo_im128_notp.hdf5")
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            out.append((tok, p))
    return out


def _print_examples(file_id: str, demos: Dict[str, Any], max_segs: int = 8) -> None:
    print(f"\n  === example timeline: {file_id} ===")
    from collections import Counter
    type_counts: Counter = Counter()
    for dm, segs in demos.items():
        for s in segs:
            type_counts[s["type"]] += 1
    print(f"  subgoal type distribution: {dict(type_counts)}")
    # show first demo's timeline
    first = next(iter(demos))
    print(f"  {first} timeline:")
    for s in demos[first][:max_segs]:
        print(f"    [{s['start']:>6} - {s['end']:>6}]  {s['lang']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", help="file ids (e.g. 003 004) or hdf5 paths")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory for JSON")
    args = ap.parse_args()

    inputs = _resolve_inputs(args.ids)
    for file_id, path in inputs:
        print(f"[FILE] {file_id}: {path}", flush=True)
        res = annotate_file(path, args.out, file_id)
        print(f"  wrote {res['out_path']}", flush=True)
        _print_examples(file_id, res["demos"])


if __name__ == "__main__":
    main()
