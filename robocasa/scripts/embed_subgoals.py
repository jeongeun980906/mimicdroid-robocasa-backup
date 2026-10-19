"""Embed per-frame subgoal / language labels INTO copies of MimicDroid play data.

This is a thin writer on top of ``annotate_subgoals.py``: it reuses that module's
(unchanged) contiguous-segment detection and, for each demo, materializes the
segment list into three per-frame datasets so every frame carries its subgoal
label. Original files are never touched -- everything is written to copies under
``/mnt/ddn/dataset/MimicDroidDataset/training_subgoal/<file_id>/demo_im128_notp.hdf5``.

Per demo (T = actions.shape[0]):
  * ``subgoal_lang``  (T,)  vlen utf-8 str   -- segment ``lang`` ("" if uncovered)
  * ``subgoal_id``    (T,)  int32            -- 0-based segment index (-1 uncovered)
  * ``subgoal_type``  (T,)  vlen utf-8 str   -- segment ``type`` ("" if uncovered)
and the raw segment list JSON is stored on ``data/<demo>.attrs['subgoals']``.

The segments come from ``annotate_subgoals.detect_events`` exactly as produced by
the current script (contiguous, lagged, all-object); detection logic is NOT
modified here. Segment ``end`` is inclusive, and the contiguous cover starts at
frame 0; frames past the final segment's ``end`` are left uncovered (id -1, "").

Usage:
  cd /mnt/ddn/mimicdroid-robocasa && MUJOCO_GL=egl uv run python \
      robocasa/scripts/embed_subgoals.py            # all 16 training files
  ... embed_subgoals.py 003 004                     # selected file ids
"""
import os, sys, json, glob, contextlib, io, logging
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import numpy as np

# Import the (unchanged) detection machinery. annotate_subgoals already silences
# robosuite's logger/stdout spam at import time; we reuse its symbols directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotate_subgoals as A  # noqa: E402

logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

DATA_ROOT = A.DATA_ROOT
OUT_ROOT = "/mnt/ddn/dataset/MimicDroidDataset/training_subgoal"
SRC_NAME = "demo_im128_notp.hdf5"

ALL_IDS = [
    "003", "004", "005", "006", "007", "008", "009", "010",
    "011", "012", "013", "015", "017", "018", "019", "020",
]

STR_DT = h5py.string_dtype(encoding="utf-8")


# --------------------------------------------------------------------------- #
def build_frame_arrays(
    segments: List[Dict[str, Any]], T: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Materialize per-frame label arrays from a contiguous segment list.

    Returns (lang[T] object-str, sid[T] int32, stype[T] object-str, covered).
    Segment ``end`` is inclusive; frames not covered by any segment get id -1
    and empty strings.
    """
    lang = np.array(["" for _ in range(T)], dtype=object)
    sid = np.full(T, -1, dtype=np.int32)
    stype = np.array(["" for _ in range(T)], dtype=object)

    covered = 0
    for idx, seg in enumerate(segments):
        s = int(seg["start"])
        e = int(seg["end"])
        # clamp to the valid frame range (lagged final end can exceed T-1)
        s = max(0, min(s, T - 1))
        e = max(0, min(e, T - 1))
        if e < s:
            continue
        sl = slice(s, e + 1)
        lang[sl] = seg.get("lang", "")
        sid[sl] = idx
        stype[sl] = seg.get("type", "")
        covered += e - s + 1
    return lang, sid, stype, covered


def add_demo_datasets(
    g: h5py.Group, segments: List[Dict[str, Any]], T: int
) -> Tuple[int, int]:
    """Write the three per-frame datasets + segment JSON attr onto a demo group.

    Returns (covered_frames, uncovered_frames).
    """
    lang, sid, stype, covered = build_frame_arrays(segments, T)

    for key in ("subgoal_lang", "subgoal_id", "subgoal_type"):
        if key in g:
            del g[key]  # idempotent re-runs
    g.create_dataset("subgoal_lang", data=lang, dtype=STR_DT)
    g.create_dataset("subgoal_id", data=sid, dtype=np.int32)
    g.create_dataset("subgoal_type", data=stype, dtype=STR_DT)
    g.attrs["subgoals"] = json.dumps(segments)
    return covered, T - covered


def copy_then_embed(
    in_path: str, out_path: str, file_id: str
) -> Dict[str, Any]:
    """Copy the source file, run detection, and embed per-frame datasets.

    Detection reuses annotate_subgoals.annotate_demo on the SOURCE file (read
    only); the labels are written to the COPY. Returns per-file stats.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Full byte-for-byte copy preserves every group/dataset/attr.
    import shutil
    print(f"  copying -> {out_path}", flush=True)
    shutil.copy2(in_path, out_path)

    # --- build env once from the source (state queries only) ---------------- #
    src = h5py.File(in_path, "r")
    demos = sorted(src["data"].keys(), key=lambda x: int(x.split("_")[-1]))

    ep0 = json.loads(src["data"][demos[0]].attrs["ep_meta"])
    obj_names = [c["name"] for c in ep0["object_cfgs"]]
    graspable = {
        c["name"] for c in ep0["object_cfgs"] if c.get("graspable") is True
    }
    pick_objs = [n for n in obj_names if n in graspable]

    ea = A.EnvArgs(robots="DemoTwoHand", render=False, control_freq=20,
                   use_camera_obs=False)
    with contextlib.redirect_stdout(io.StringIO()):
        env, _ = A.make_env(file_name=in_path, env_args=ea)

    per_demo: Dict[str, Dict[str, Any]] = {}
    total_cov = 0
    total_unc = 0
    total_frames = 0

    dst = h5py.File(out_path, "a")
    try:
        for dm in demos:
            g = src["data"][dm]
            ep = json.loads(g.attrs["ep_meta"])
            with contextlib.redirect_stdout(io.StringIO()):
                A.reset_to(
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
            segs = A.annotate_demo(env, g, pick_objs)
            T = int(g["actions"].shape[0])

            covered, uncovered = add_demo_datasets(dst["data"][dm], segs, T)
            # segment span (contiguous cover from 0..last_end) for the report
            seg_span = sum(
                max(0, min(int(s["end"]), T - 1) - max(0, int(s["start"])) + 1)
                for s in segs
            )
            per_demo[dm] = {
                "T": T,
                "segments": len(segs),
                "covered": covered,
                "uncovered": uncovered,
                "seg_span": seg_span,
            }
            total_cov += covered
            total_unc += uncovered
            total_frames += T
            print(
                f"    {dm}: T={T}  segs={len(segs)}  "
                f"covered={covered}  uncovered={uncovered}",
                flush=True,
            )
        dst.flush()
    finally:
        dst.close()
        env.close()
        src.close()

    return {
        "file_id": file_id,
        "out_path": out_path,
        "demos": len(demos),
        "total_frames": total_frames,
        "covered": total_cov,
        "uncovered": total_unc,
        "per_demo": per_demo,
    }


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_file(out_path: str, n_samples: int = 4, rng=None) -> List[Tuple]:
    """Load a written file, sample frames, confirm labels match their segment.

    Returns a list of (file_id, demo, frame, subgoal_id, lang) sample rows.
    """
    rng = rng or np.random.default_rng(0)
    file_id = os.path.basename(os.path.dirname(out_path))
    rows: List[Tuple] = []
    f = h5py.File(out_path, "r")
    try:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        for dm in demos:
            g = f["data"][dm]
            T = int(g["actions"].shape[0])
            lang = g["subgoal_lang"][:]
            sid = g["subgoal_id"][:]
            stype = g["subgoal_type"][:]
            segs = json.loads(g.attrs["subgoals"])

            # invariant checks
            assert len(lang) == T == len(sid) == len(stype), (
                f"{file_id}/{dm}: length mismatch"
            )
            covered = int((sid >= 0).sum())
            seg_span = sum(
                max(0, min(int(s["end"]), T - 1) - max(0, int(s["start"])) + 1)
                for s in segs
            )
            assert covered == seg_span, (
                f"{file_id}/{dm}: covered {covered} != seg span {seg_span}"
            )

            # per-frame: the segment that contains the frame must match labels
            for fr in rng.integers(0, T, size=n_samples):
                fr = int(fr)
                want_id, want_lang, want_type = -1, "", ""
                for idx, s in enumerate(segs):
                    s0 = max(0, min(int(s["start"]), T - 1))
                    s1 = max(0, min(int(s["end"]), T - 1))
                    if s0 <= fr <= s1:
                        want_id, want_lang = idx, s.get("lang", "")
                        want_type = s.get("type", "")
                        break
                got_lang = lang[fr]
                got_lang = got_lang.decode() if isinstance(got_lang, bytes) else got_lang
                got_type = stype[fr]
                got_type = got_type.decode() if isinstance(got_type, bytes) else got_type
                assert int(sid[fr]) == want_id, (
                    f"{file_id}/{dm} frame {fr}: id {int(sid[fr])} != {want_id}"
                )
                assert got_lang == want_lang, (
                    f"{file_id}/{dm} frame {fr}: lang mismatch"
                )
                assert got_type == want_type
                rows.append((file_id, dm, fr, int(sid[fr]), got_lang))
    finally:
        f.close()
    return rows


# --------------------------------------------------------------------------- #
def main() -> None:
    ids = sys.argv[1:] or ALL_IDS

    results: List[Dict[str, Any]] = []
    for fid in ids:
        in_path = os.path.join(DATA_ROOT, fid, SRC_NAME)
        if not os.path.isfile(in_path):
            print(f"[SKIP] {fid}: {in_path} not found", flush=True)
            continue
        out_path = os.path.join(OUT_ROOT, fid, SRC_NAME)
        print(f"[FILE] {fid}: {in_path}", flush=True)
        res = copy_then_embed(in_path, out_path, fid)
        results.append(res)

    # ----- report --------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("PER-FILE SUMMARY")
    print("=" * 70)
    print(f"{'file':>5} {'demos':>6} {'frames':>10} {'labeled':>10} "
          f"{'uncovered':>10} {'%labeled':>9}")
    g_frames = g_cov = g_unc = 0
    for r in results:
        pct = 100.0 * r["covered"] / max(1, r["total_frames"])
        print(f"{r['file_id']:>5} {r['demos']:>6} {r['total_frames']:>10} "
              f"{r['covered']:>10} {r['uncovered']:>10} {pct:>8.1f}%")
        g_frames += r["total_frames"]
        g_cov += r["covered"]
        g_unc += r["uncovered"]
    gpct = 100.0 * g_cov / max(1, g_frames)
    print("-" * 70)
    print(f"{'TOTAL':>5} {'':>6} {g_frames:>10} {g_cov:>10} "
          f"{g_unc:>10} {gpct:>8.1f}%")

    # ----- dataset dtype/shape note --------------------------------------- #
    if results:
        sample_out = results[0]["out_path"]
        f = h5py.File(sample_out, "r")
        dm = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))[0]
        g = f["data"][dm]
        print("\nADDED DATASETS (example: "
              f"{results[0]['file_id']}/{dm}):")
        for k in ("subgoal_lang", "subgoal_id", "subgoal_type"):
            print(f"  data/<demo>/{k}: shape={g[k].shape} dtype={g[k].dtype}")
        print("  data/<demo>.attrs['subgoals']: JSON str (segment list)")
        f.close()

    # ----- verification sample (>=2 files) -------------------------------- #
    print("\n" + "=" * 70)
    print("VERIFICATION SAMPLE (frame -> segment match confirmed)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    verify_ids = [r["out_path"] for r in results[:2]] if len(results) >= 2 \
        else [r["out_path"] for r in results]
    all_rows: List[Tuple] = []
    for op in verify_ids:
        all_rows.extend(verify_file(op, n_samples=4, rng=rng))
    print(f"{'file':>5} {'demo':>22} {'frame':>7} {'sid':>4}  lang")
    for row in all_rows[:5]:
        fid, dm, fr, sid, lang = row
        print(f"{fid:>5} {dm:>22} {fr:>7} {sid:>4}  {lang[:60]}")
    print(f"\nAll invariants passed for {len(verify_ids)} verified file(s).")


if __name__ == "__main__":
    main()
