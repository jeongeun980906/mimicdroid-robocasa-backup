#!/usr/bin/env python3
"""Data-level OLD (gr1_v2 / aug_obs / evenspread) vs NEW (gr1_gen_v3) verification.
Produces /mnt/ddn/dataset/gr1_gen_v3/verification/{verification.json, TABLE.md}.

Tests (docs/gr1_data_eval_redesign.md Phase-2 acceptance):
  t1  Init-distribution coverage vs the FROZEN seen eval manifests: for each of the
      20 eval scenes per task, NN distance to the train-set t0 object vectors,
      normalized by the train set's internal-NN median. OLD reference numbers from
      /mnt/ddn/tmp/eval_init_match/coverage_report.json are also cited.
  t2  GT-replay on the eval-manifest scenes (run separately via gen_unified
      --stage episodes/--stage gr1 with --scenes <manifest npz>; this script only
      summarizes the resulting feasibility jsons).
  t3  Multi-task counterfactual audit (from the driver's per-cell audit.json).
  t4  t=0 visibility: NEW = constructive assert stats from feasibility jsons;
      BOTH = stored-segmentation metric on GR1 hdf5s: count of instance-seg ids
      present at frame 10 but ABSENT at frame 0 of the center cam (the old
      empty-sink bug: dynamically-loaded objects only render after a step).
  t5  Volume: episodes per task per cell + GR1 transfer rates.

Run:  .venv/bin/python robocasa/scripts/verify_old_vs_new.py --tests t1 t3 t4 t5 report
"""
import os
import sys
import json
import glob
import argparse

import h5py
import numpy as np

NEW = "/mnt/ddn/dataset/gr1_gen_v3"
VER = os.path.join(NEW, "verification")
MAN = "/mnt/ddn/dataset/gr1_eval_manifests"          # frozen, READ-ONLY
OLD_AUG = "/mnt/ddn/tmp/augmented"                    # v2 DemoTwoHand aug (t0 vecs)
OLD_GR1 = "/mnt/ddn/tmp/augmented/gr1_v2"             # v2 GR1 replays (stored seg)
OLD_ES = "/mnt/ddn/tmp/augmented/evalstyle"           # old evenspread pass
OLD_COV_REPORT = "/mnt/ddn/tmp/eval_init_match/coverage_report.json"

TASKS = ["CloseLeftCabinetDoor", "TurnOnFaucet",
         "PnPSinkToCabinet", "PnPSinkToRightCounterPlate"]
CANON_CELL = {"CloseLeftCabinetDoor": "L0_001_l1", "TurnOnFaucet": "L4_037_l1",
              "PnPSinkToCabinet": "L2_015_l1", "PnPSinkToRightCounterPlate": "L7_019_l1"}
CAM = "robot0_agentview_center"


def _res(name):
    p = os.path.join(VER, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _save(name, obj):
    os.makedirs(VER, exist_ok=True)
    json.dump(obj, open(os.path.join(VER, f"{name}.json"), "w"), indent=1)
    print(f"[{name}] -> {VER}/{name}.json")


def _modal_stack(vecs):
    dims = [v.shape[0] for v in vecs]
    modal = max(set(dims), key=dims.count)
    return np.stack([v for v in vecs if v.shape[0] == modal]), modal, len(vecs)


def _xnn(train, evals):
    """eval NN distance to train, normalized by train internal-NN median."""
    d_int = [np.sort(np.linalg.norm(train - train[i], axis=1))[1] for i in range(len(train))]
    med = float(np.median(d_int))
    xs = [float(np.min(np.linalg.norm(train - e, axis=1))) / (med + 1e-12) for e in evals]
    return dict(internal_nn_median=med, xnn_median=float(np.median(xs)),
                xnn_max=float(np.max(xs)), n_train=len(train), n_eval=len(evals))


def _t0_vecs_hdf5(path, max_eps=None):
    out = []
    with h5py.File(path, "r") as f:
        ks = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        for k in (ks[:max_eps] if max_eps else ks):
            out.append(np.asarray(f["data"][k]["obs"]["object-state"][0], np.float64))
    return out


def t1_coverage():
    res = {}
    old_rep = json.load(open(OLD_COV_REPORT)) if os.path.exists(OLD_COV_REPORT) else {}
    for task in TASKS:
        cell = CANON_CELL[task]
        man = os.path.join(MAN, f"{task}__seen.npz")
        evals = list(np.load(man, allow_pickle=True)["object_state"])
        ev, evd, _ = _modal_stack(evals)
        row = dict(eval_dim=evd, n_eval_total=len(evals))
        # NEW: gr1_gen_v3 scene draws of the canonical cell (same DemoTwoHand law)
        vecs = [v for p in sorted(glob.glob(f"{NEW}/{cell}/scenes_*.npz"))
                for v in np.load(p, allow_pickle=True)["object_state"]]
        if vecs:
            tr, trd, n = _modal_stack(vecs)
            row["NEW"] = _xnn(tr, ev) if trd == evd else f"dim_mismatch train {trd} vs eval {evd}"
        # OLD: v2 DemoTwoHand aug t0 vectors vs the SAME frozen manifests
        for label, path in (("OLD_v2aug", f"{OLD_AUG}/{task}_aug_obs.hdf5"),
                            ("OLD_evenspread", f"{OLD_ES}/{task}_es.hdf5")):
            if os.path.exists(path):
                tr, trd, _ = _modal_stack(_t0_vecs_hdf5(path))
                row[label] = _xnn(tr, ev) if trd == evd else f"dim_mismatch train {trd} vs eval {evd}"
        # citation: the Phase-1 coverage report's OLD(V2) numbers (old 3-anchor eval)
        if task in old_rep:
            xs = [v["V2"]["nn_x_internal"] for v in old_rep[task].get("coverage", {}).values()
                  if "V2" in v]
            row["OLD_cited_phase1_report"] = dict(nn_x_internal=sorted(set(round(x, 1) for x in xs)))
        res[task] = row
        print(task, json.dumps({k: v for k, v in row.items() if k != "eval_dim"}, default=str)[:300])
    _save("t1_coverage", res)


def t2_summarize():
    """Summarize the eval-scene GT-replay outputs produced by gen_unified stages
    (episodes/gr1 run on the frozen manifest npz, outputs under VER/)."""
    res = {}
    for task in TASKS:
        row = {}
        p = os.path.join(VER, f"evalreplay_dth_{task}_feasibility.json")
        if os.path.exists(p):
            e = json.load(open(p))["entries"]
            row["dth_pass"] = f"{sum(1 for x in e if x['feasible'])}/{len(e)}"
        p = os.path.join(VER, f"evalreplay_gr1_{task}_gr1_feasibility.json")
        if os.path.exists(p):
            e = json.load(open(p))["entries"]
            row["gr1_pass"] = f"{sum(1 for x in e if x['gr1_success'])}/{len(e)}"
            row["gr1_fail_reasons"] = sorted({x["reason"] for x in e if not x["gr1_success"]})
        res[task] = row
        print(task, row)
    _save("t2_evalreplay", res)


def t3_multitask():
    res = {}
    for aud in sorted(glob.glob(f"{NEW}/L*/audit.json")):
        a = json.load(open(aud))
        cell = os.path.basename(os.path.dirname(aud))
        res[cell] = {k: a.get(k) for k in
                     ("dth_scenes", "dth_multi_task_scenes", "dth_sibling_t0_bitwise_identical",
                      "gr1_scenes", "gr1_multi_task_scenes", "gr1_sibling_object_t0_identical")}
        print(cell, res[cell])
    res["OLD"] = "0 multi-task scenes by construction (one generator+scene per task)"
    _save("t3_multitask", res)


def _seg_missing_t0(path, n_eps):
    """Per episode: #center-cam instance-seg ids present at frame 10 but absent at
    frame 0 (dynamic objects that only render after steps = the old empty-sink bug)."""
    miss = []
    with h5py.File(path, "r") as f:
        ks = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        if n_eps and len(ks) > n_eps:   # even subsample
            ks = [ks[i] for i in np.linspace(0, len(ks) - 1, n_eps).astype(int)]
        for k in ks:
            seg = f["data"][k]["obs"].get(f"{CAM}_segmentation_instance")
            if seg is None or len(seg) < 11:
                continue
            s0 = set(np.unique(seg[0])) - {0}
            s10 = set(np.unique(seg[10])) - {0}
            miss.append(len(s10 - s0))
    return miss


def t4_visibility(sample_old=50):
    res = {}
    # NEW: constructive assert stats (all saved episodes passed >= vis_min_px)
    tot, rej = 0, 0
    for p in sorted(glob.glob(f"{NEW}/L*/dth_*_feasibility.json")):
        for e in json.load(open(p))["entries"]:
            tot += 1
            rej += ("t0_visibility_fail" in str(e.get("reason", "")))
    res["NEW_dth_assert"] = dict(scenes_task_attempts=tot, visibility_rejections=rej)
    tot, rej = 0, 0
    for p in sorted(glob.glob(f"{NEW}/L*/gr1_*_gr1_feasibility.json")):
        for e in json.load(open(p))["entries"]:
            tot += 1
            rej += ("t0_visibility_fail" in str(e.get("reason", "")))
    res["NEW_gr1_assert"] = dict(episodes=tot, visibility_rejections=rej)
    # stored-seg metric on GR1 files, OLD vs NEW
    for task in TASKS:
        row = {}
        old = f"{OLD_GR1}/{task}_gr1.hdf5"
        if os.path.exists(old):
            m = _seg_missing_t0(old, sample_old)
            row["OLD_v2_gr1"] = dict(n=len(m), eps_with_missing_t0_ids=int(np.sum(np.array(m) > 0)),
                                     mean_missing=float(np.mean(m)) if m else None)
        news = sorted(glob.glob(f"{NEW}/L*/gr1_{task}_p*.hdf5"))
        m = [x for p in news for x in _seg_missing_t0(p, None)]
        if m:
            row["NEW_gr1"] = dict(n=len(m), eps_with_missing_t0_ids=int(np.sum(np.array(m) > 0)),
                                  mean_missing=float(np.mean(m)))
        res[task] = row
        print(task, row)
    _save("t4_visibility", res)


def t5_volume():
    res = {}
    for cell_dir in sorted(glob.glob(f"{NEW}/L*_*")):
        cell = os.path.basename(cell_dir)
        row = {}
        for task in TASKS:
            n_dth = n_gr1 = 0
            for p in glob.glob(f"{cell_dir}/dth_{task}_p*.hdf5"):
                with h5py.File(p, "r") as f:
                    n_dth += len(f["data"])
            for p in glob.glob(f"{cell_dir}/gr1_{task}_p*.hdf5"):
                with h5py.File(p, "r") as f:
                    n_gr1 += len(f["data"])
            if n_dth or n_gr1:
                row[task] = dict(dth=n_dth, gr1=n_gr1,
                                 gr1_transfer=round(n_gr1 / n_dth, 3) if n_dth else None)
        res[cell] = row
        print(cell, row)
    # per-task totals + canonical-cell counts
    totals = {}
    for task in TASKS:
        tot_g = sum(r[task]["gr1"] for r in res.values() if task in r)
        canon = res.get(CANON_CELL[task], {}).get(task, {}).get("gr1", 0)
        totals[task] = dict(gr1_total=tot_g, gr1_at_canonical=canon)
    res["totals"] = totals
    sel = os.path.join(NEW, "final", "selection.json")
    if os.path.exists(sel):
        res["final_selection"] = json.load(open(sel))
    _save("t5_volume", res)


def t6_refinement():
    """Summarize the *_v2 refined eval manifests (GR1 gate applied to v1)."""
    res = {}
    for p in sorted(glob.glob(os.path.join(MAN, "*_v2.json"))):
        m = json.load(open(p))
        res[m["tag"]] = dict(
            v1_gr1_feasibility=m.get("v1_gr1_feasibility_rate"),
            v2_scenes=m.get("n_scenes"),
            topup=sum(1 for v in m.get("origin", {}).values() if v == "topup"),
            v1_dropped=m.get("v1_dropped_seeds"))
        print(m["tag"], res[m["tag"]])
    _save("t6_refinement", res)


def fmt(v):
    if isinstance(v, dict) and "xnn_median" in v:
        return f"{v['xnn_median']:.2f}x med / {v['xnn_max']:.2f}x max (n={v['n_train']})"
    return str(v)


def report():
    t1, t2, t3, t4, t5, t6 = (_res(n) for n in
                              ("t1_coverage", "t2_evalreplay", "t3_multitask",
                               "t4_visibility", "t5_volume", "t6_refinement"))
    L = []
    L.append("# OLD (gr1_v2) vs NEW (gr1_gen_v3) data-level verification\n")
    L.append("| test | task/cell | OLD | NEW |")
    L.append("|---|---|---|---|")
    for task in TASKS:
        r = (t1 or {}).get(task, {})
        old = fmt(r.get("OLD_v2aug", "n/a"))
        cited = r.get("OLD_cited_phase1_report", {}).get("nn_x_internal")
        if cited:
            old += f" (phase-1 report vs old 3-anchor eval: {cited}x)"
        es = r.get("OLD_evenspread")
        if es:
            old += f"; evenspread {fmt(es)}"
        L.append(f"| 1. eval-init coverage (xNN vs frozen manifests) | {task} | {old} | {fmt(r.get('NEW', 'n/a'))} |")
    for task in TASKS:
        r = (t2 or {}).get(task, {})
        L.append(f"| 2. GT-replay on the 20 frozen eval scenes | {task} | "
                 f"FAILED all eval inits (gr1 online eval findings; old gen scenes ≠ eval scenes) | "
                 f"DemoTwoHand {r.get('dth_pass', '?')}, GR1 {r.get('gr1_pass', '?')} |")
    for cell, r in sorted((t3 or {}).items()):
        if cell == "OLD":
            continue
        L.append(f"| 3. multi-task same-scene groups | {cell} | 0 (by construction) | "
                 f"dth {r.get('dth_multi_task_scenes')}/{r.get('dth_scenes')} scenes ≥2 tasks, "
                 f"siblings bitwise {r.get('dth_sibling_t0_bitwise_identical')}; "
                 f"gr1 {r.get('gr1_multi_task_scenes')}/{r.get('gr1_scenes')}, "
                 f"object-t0 {r.get('gr1_sibling_object_t0_identical')} |")
    for task in TASKS:
        r = (t4 or {}).get(task, {})
        o, n = r.get("OLD_v2_gr1", {}), r.get("NEW_gr1", {})
        L.append(f"| 4. t=0 visibility (seg ids at frame10 missing at frame0) | {task} | "
                 f"{o.get('eps_with_missing_t0_ids', '?')}/{o.get('n', '?')} eps affected "
                 f"(mean {o.get('mean_missing', '?')}) | "
                 f"{n.get('eps_with_missing_t0_ids', '?')}/{n.get('n', '?')} eps affected "
                 f"(mean {n.get('mean_missing', '?')}); constructive assert: "
                 f"{(t4 or {}).get('NEW_gr1_assert', {}).get('visibility_rejections', '?')} rejections |")
    if t5:
        skip = {"totals", "final_selection"}
        for task in TASKS:
            tt = t5["totals"][task]
            per = "; ".join(f"{c}:{r[task]['gr1']}/{r[task]['dth']}"
                            for c, r in sorted(t5.items()) if c not in skip and task in r)
            fin = (t5.get("final_selection", {}).get("per_task", {}) or {}).get(task, {})
            fin_s = (f" — FINAL {fin.get('selected')} selected "
                     f"({fin.get('in_multi_task_group')} in >=2-task groups; "
                     f"by cell {fin.get('by_cell')}; overflow {fin.get('overflow')})"
                     if fin else "")
            L.append(f"| 5. volume (GR1/DemoTwoHand eps per cell) | {task} | "
                     f"50 GR1 eps, 1 cell, style-cycled artificial scenes | "
                     f"{per} — GR1 total {tt['gr1_total']} (canonical {tt['gr1_at_canonical']})"
                     f"{fin_s} |")
    if t6:
        for tag, r in sorted(t6.items()):
            L.append(f"| 6. eval-manifest GR1 refinement (v2) | {tag} | "
                     f"v1 had NO GR1 gate | v1 GR1-feasibility {r['v1_gr1_feasibility']}; "
                     f"v2 = {r['v2_scenes']} scenes ({r['topup']} topped up) |")
    # explicit task x cell selection matrix (user check: no single-layout skew)
    sel = (t5 or {}).get("final_selection", {}).get("per_task")
    if sel:
        cells = sorted({c for v in sel.values() for c in v.get("by_cell", {})})
        L.append("\n## Final selection: task x (layout,style) cell counts\n")
        L.append("| task | " + " | ".join(cells) + " | total | in >=2-task group |")
        L.append("|---" * (len(cells) + 3) + "|")
        for task in TASKS:
            v = sel.get(task, {})
            bc = v.get("by_cell", {})
            L.append(f"| {task} | " + " | ".join(str(bc.get(c, 0)) for c in cells) +
                     f" | {v.get('selected')} | {v.get('in_multi_task_group')} |")
    md = "\n".join(L) + "\n"
    os.makedirs(VER, exist_ok=True)
    open(os.path.join(VER, "TABLE.md"), "w").write(md)
    json.dump(dict(t1=t1, t2=t2, t3=t3, t4=t4, t5=t5, t6=t6),
              open(os.path.join(VER, "verification.json"), "w"), indent=1)
    print(md)
    print(f"-> {VER}/TABLE.md, {VER}/verification.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tests", nargs="+", required=True,
                    choices=["t1", "t2", "t3", "t4", "t5", "t6", "report"])
    ap.add_argument("--sample_old", type=int, default=50)
    args = ap.parse_args()
    for t in args.tests:
        {"t1": t1_coverage, "t2": t2_summarize, "t3": t3_multitask,
         "t4": lambda: t4_visibility(args.sample_old), "t5": t5_volume,
         "t6": t6_refinement, "report": report}[t]()


if __name__ == "__main__":
    main()
