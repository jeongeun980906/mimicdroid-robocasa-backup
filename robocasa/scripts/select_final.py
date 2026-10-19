#!/usr/bin/env python3
"""Final GR1 dataset selection: cap at N (default 50) episodes per task, chosen to
maximize the redesign's core properties instead of first-come:

  1. multi-task scene groups first (episodes whose scene_seed has GR1-kept
     siblings from other tasks, biggest groups first) — and a group is only taken
     if >=2 of its tasks are co-selected, so the counterfactual pairs survive;
  2. layout spread: the task's canonical cell keeps ~half (target 25), the rest
     spread over the other cells;
  3. everything GR1-kept but not selected goes to a clearly-marked _overflow
     file per task (kept on disk, excluded from training manifests).

Outputs under <root>/final/: gr1_final_<task>.hdf5, gr1_overflow_<task>.hdf5,
selection.json (full composition). Episode attrs gain final_sibling_tasks (the
co-SELECTED tasks of the same scene).

Run: .venv/bin/python robocasa/scripts/select_final.py --cap 50
"""
import os
import json
import glob
import argparse
import collections

import h5py
import numpy as np

ROOT = "/mnt/ddn/dataset/gr1_gen_v3"
TASKS = ["CloseLeftCabinetDoor", "TurnOnFaucet",
         "PnPSinkToCabinet", "PnPSinkToRightCounterPlate"]
CANON = {"CloseLeftCabinetDoor": "L0_001_l1", "TurnOnFaucet": "L4_037_l1",
         "PnPSinkToCabinet": "L2_015_l1", "PnPSinkToRightCounterPlate": "L7_019_l1"}


def catalog(root):
    """All GR1-kept episodes from COMPLETE gr1 parts (feasibility json present)."""
    eps = collections.defaultdict(list)      # task -> [entry]
    for p in sorted(glob.glob(f"{root}/L*/gr1_*_p*.hdf5")):
        if not os.path.exists(p.replace(".hdf5", "_gr1_feasibility.json")):
            continue
        cell = os.path.basename(os.path.dirname(p))
        try:
            with h5py.File(p, "r") as f:
                for k in f["data"]:
                    a = f["data"][k].attrs
                    eps[str(a["task"])].append(dict(
                        cell=cell, seed=int(a["scene_seed"]), file=p, key=k,
                        style=str(a.get("style_id", "")), n=int(a["num_samples"])))
        except (OSError, RuntimeError):
            print(f"[select] unreadable (skipped): {p}")
    # dedup (a scene-task pair could appear in overlapping parts; keep first)
    for t in eps:
        seen, uniq = set(), []
        for e in sorted(eps[t], key=lambda e: (e["seed"], e["file"])):
            if e["seed"] not in seen:
                seen.add(e["seed"])
                uniq.append(e)
        eps[t] = uniq
    return eps


def select(eps, cap):
    groups = collections.defaultdict(dict)    # seed -> task -> entry
    for t, lst in eps.items():
        for e in lst:
            groups[e["seed"]][t] = e
    cells_per_task = {t: sorted({e["cell"] for e in lst}) for t, lst in eps.items()}
    # per-task canonical minimums: PnPSinkToCabinet boosted to 30/50 (user 2026-07-10:
    # Plate's canonical L7 kitchen is right-shifted, so cabinet needs a stronger
    # central-scene (L2) share to avoid a compounding right-bias across tasks).
    CANON_TARGET = {"PnPSinkToCabinet": 40}   # user 2026-07-10: ~40 central(L2) + ~10 other
                                               # (L2 kept pool = 39, so effectively all of L2)
    def canon_t(t):
        return CANON_TARGET.get(t, cap // 2)
    def cell_cap(t, c):
        ct = canon_t(t)
        others = max(1, len(cells_per_task[t]) - 1)
        return ct + 3 if c == CANON[t] else int(np.ceil((cap - ct) / others)) + 2

    picked = {t: [] for t in eps}
    ccnt = {t: collections.Counter() for t in eps}
    chosen = set()                             # (task, seed)
    # pass 1: multi-task groups, largest first
    for seed, bytask in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(bytask) < 2:
            continue
        take = [(t, e) for t, e in sorted(bytask.items())
                if len(picked[t]) < cap and ccnt[t][e["cell"]] < cell_cap(t, e["cell"])]
        if len(take) < 2:
            continue                           # a lone pick adds no counterfactual pair
        for t, e in take:
            picked[t].append(e)
            ccnt[t][e["cell"]] += 1
            chosen.add((t, seed))
    # pass 2: fill to cap — canonical first (to ~half), then round-robin others
    for t, lst in eps.items():
        if len(picked[t]) >= cap:
            continue
        rest = [e for e in lst if (t, e["seed"]) not in chosen]
        canon = [e for e in rest if e["cell"] == CANON[t]]
        other = [e for e in rest if e["cell"] != CANON[t]]
        for e in canon:
            if len(picked[t]) >= cap or ccnt[t][e["cell"]] >= canon_t(t):
                break
            picked[t].append(e); ccnt[t][e["cell"]] += 1; chosen.add((t, e["seed"]))
        by_cell = collections.defaultdict(list)
        for e in other:
            by_cell[e["cell"]].append(e)
        rr = sorted(by_cell)
        while len(picked[t]) < cap and any(by_cell[c] for c in rr):
            for c in rr:
                if by_cell[c] and len(picked[t]) < cap:
                    e = by_cell[c].pop(0)
                    picked[t].append(e); ccnt[t][e["cell"]] += 1; chosen.add((t, e["seed"]))
        # last resort: ignore quotas entirely
        for e in rest:
            if len(picked[t]) >= cap:
                break
            if (t, e["seed"]) not in chosen:
                picked[t].append(e); ccnt[t][e["cell"]] += 1; chosen.add((t, e["seed"]))
    # final group membership = co-SELECTED tasks per scene
    sel_by_seed = collections.defaultdict(set)
    for t, lst in picked.items():
        for e in lst:
            sel_by_seed[e["seed"]].add(t)
    return picked, chosen, sel_by_seed


def write_sets(root, eps, picked, sel_by_seed, cap):
    out_dir = os.path.join(root, "final")
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for t, lst in eps.items():
        sel = picked[t]
        sel_keys = {(e["file"], e["key"]) for e in sel}
        over = [e for e in lst if (e["file"], e["key"]) not in sel_keys]
        for label, subset in (("final", sel), ("overflow", over)):
            path = os.path.join(out_dir, f"gr1_{label}_{t}.hdf5")
            with h5py.File(path, "w") as fo:
                od = fo.create_group("data")
                env_args = None
                total = 0
                for i, e in enumerate(sorted(subset, key=lambda x: x["seed"])):
                    with h5py.File(e["file"], "r") as fi:
                        if env_args is None:
                            env_args = fi["data"].attrs["env_args"]
                        fi.copy(fi["data"][e["key"]], od, name=f"demo_{i}")
                        total += fi["data"][e["key"]]["actions"].shape[0]
                    g = od[f"demo_{i}"]
                    g.attrs["final_sibling_tasks"] = json.dumps(
                        sorted(sel_by_seed[e["seed"]] - {t}) if label == "final" else [])
                od.attrs["env_args"] = env_args if env_args is not None else ""
                od.attrs["total"] = total
            print(f"[select] {t} {label}: {len(subset)} eps -> {path}")
        comp = collections.Counter(e["cell"] for e in sel)
        in_group = sum(1 for e in sel if len(sel_by_seed[e["seed"]]) >= 2)
        stats[t] = dict(candidates=len(lst), selected=len(sel), overflow=len(over),
                        by_cell=dict(comp), in_multi_task_group=in_group)
    group_sizes = collections.Counter(len(v) for v in sel_by_seed.values())
    meta = dict(cap=cap, per_task=stats,
                final_scene_groups_by_size={str(k): v for k, v in sorted(group_sizes.items())},
                note=("final_sibling_tasks attr = co-selected tasks of the same scene; "
                      "overflow files are GR1-successful episodes excluded from the "
                      "training manifest, retained as insurance"))
    json.dump(meta, open(os.path.join(out_dir, "selection.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--cap", type=int, default=50)
    args = ap.parse_args()
    eps = catalog(args.root)
    for t in TASKS:
        print(f"[select] {t}: {len(eps.get(t, []))} GR1-kept candidates")
    picked, chosen, sel_by_seed = select(eps, args.cap)
    write_sets(args.root, eps, picked, sel_by_seed, args.cap)


if __name__ == "__main__":
    main()
