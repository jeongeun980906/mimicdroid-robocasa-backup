"""Render labeled subgoal-review videos + montages for MimicDroid play files.

Reads per-frame subgoal labels embedded in
  /mnt/ddn/tmp/subgoal_embedded/<id>/demo_im128_notp.hdf5
and the segment JSON at
  /mnt/ddn/tmp/subgoal_annotations/<id>.json

For the FIRST demo of each file it writes:
  - <id>_labeled.mp4  : subsampled frames upscaled to ~400px, with the current
                        frame's subgoal_lang overlaid at the bottom, color-coded
                        by subgoal_type.
  - <id>_montage.png  : ~9 segment-midpoint frames, each with its label.

Pure hdf5 read + cv2 write (no MUJOCO / env). Stored agentview images are
already UPRIGHT -> do NOT flip vertically; only RGB->BGR for cv2.

Usage:
  uv run python robocasa/scripts/make_subgoal_videos.py 004 007 010 013 017 020
"""
import os
import sys
import json
from collections import Counter

import h5py
import numpy as np
import cv2

EMBED_DIR = "/mnt/ddn/tmp/subgoal_embedded"
ANNO_DIR = "/mnt/ddn/tmp/subgoal_annotations"
OUT_DIR = "/mnt/ddn/tmp/replay_videos/subgoal_samples"

UPSCALE = 400
SUBSAMPLE = 4
MAX_FRAMES = 8000
FPS = 20

# BGR colors per subgoal type
TYPE_COLORS = {
    "pick_place": (80, 200, 80),    # green
    "door_open": (60, 140, 240),    # orange
    "door_close": (180, 100, 40),   # blue
    "faucet_on": (60, 60, 230),     # red
    "faucet_off": (200, 60, 200),   # magenta
    "": (150, 150, 150),            # grey (idle / unlabeled)
}


def _dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def first_demo(f):
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    return demos[0], demos


def color_for(stype):
    return TYPE_COLORS.get(stype, (220, 220, 220))


def overlay_label(img, text, stype):
    """Draw a color-coded label bar with wrapped text at the bottom."""
    h, w = img.shape[:2]
    color = color_for(stype)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1

    # wrap text to fit width
    words = (text or "(idle)").split()
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(trial, font, scale, thick)
        if tw > w - 16 and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)

    line_h = 20
    bar_h = line_h * len(lines) + 14
    y0 = h - bar_h
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    # color swatch strip on the left edge of the bar
    cv2.rectangle(img, (0, y0), (8, h), color, -1)

    y = y0 + line_h
    for ln in lines:
        cv2.putText(img, ln, (14, y), font, scale, color, thick, cv2.LINE_AA)
        y += line_h
    # type tag top-right
    cv2.putText(img, stype or "idle", (10, y0 - 8), font, 0.45, color, 1, cv2.LINE_AA)
    return img


def make_video(imgs, langs, types, out_path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (UPSCALE, UPSCALE))
    n = min(len(imgs), MAX_FRAMES)
    count = 0
    for i in range(0, n, SUBSAMPLE):
        frame = imgs[i][..., ::-1]  # RGB->BGR, already upright
        frame = cv2.resize(frame, (UPSCALE, UPSCALE), interpolation=cv2.INTER_NEAREST)
        frame = np.ascontiguousarray(frame)
        overlay_label(frame, _dec(langs[i]), _dec(types[i]))
        vw.write(frame)
        count += 1
    vw.release()
    return count


def make_montage(imgs, segments, out_path, demo_key):
    """~9 segment-midpoint frames with labels."""
    segs = segments
    if len(segs) > 9:
        idx = np.linspace(0, len(segs) - 1, 9).round().astype(int)
        segs = [segments[i] for i in idx]
    tiles = []
    T = imgs.shape[0]
    cell = 300
    for seg in segs:
        mid = min(T - 1, (seg["start"] + seg["end"]) // 2)
        frame = imgs[mid][..., ::-1]
        frame = cv2.resize(frame, (cell, cell), interpolation=cv2.INTER_NEAREST)
        frame = np.ascontiguousarray(frame)
        overlay_label(frame, seg.get("lang", ""), seg.get("type", ""))
        tiles.append(frame)
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((cell, cell, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(out_path, grid)


def segments_from_perframe(ids, langs, types):
    """Reconstruct segment list from per-frame arrays via subgoal_id runs.

    Used when no segment JSON is available. Each contiguous run of equal
    subgoal_id becomes one segment with start/end/lang/type. obj_lang is
    parsed from the 'pick the <obj> from ...' phrasing when present.
    """
    ids = np.asarray(ids)
    T = len(ids)
    segments = []
    i = 0
    while i < T:
        j = i
        while j + 1 < T and ids[j + 1] == ids[i]:
            j += 1
        lang = _dec(langs[i])
        stype = _dec(types[i])
        if stype:  # skip idle/unlabeled runs as standalone segments
            seg = {"start": i, "end": j, "lang": lang, "type": stype}
            if stype == "pick_place" and lang.startswith("pick the "):
                rest = lang[len("pick the "):]
                seg["obj_lang"] = rest.split(" from ")[0].strip()
            segments.append(seg)
        i = j + 1
    return segments


def type_distribution(segments):
    type_ctr = Counter()
    obj_ctr = Counter()
    for s in segments:
        t = s.get("type", "")
        type_ctr[t] += 1
        if t == "pick_place":
            obj_ctr[s.get("obj_lang", "?")] += 1
    return type_ctr, obj_ctr


def process(file_id):
    h5_path = os.path.join(EMBED_DIR, file_id, "demo_im128_notp.hdf5")
    json_path = os.path.join(ANNO_DIR, f"{file_id}.json")
    os.makedirs(OUT_DIR, exist_ok=True)

    f = h5py.File(h5_path, "r")
    d0, demos = first_demo(f)
    g = f["data"][d0]
    imgs = g["obs/robot0_agentview_center_image"][:]
    langs = g["subgoal_lang"][:]
    types = g["subgoal_type"][:]
    sids = g["subgoal_id"][:]

    segments = None
    seg_src = "perframe"
    if os.path.exists(json_path):
        anno = json.load(open(json_path))
        segments = anno["demos"].get(d0) or next(iter(anno["demos"].values()))
        seg_src = "json"
    if not segments:
        segments = segments_from_perframe(sids, langs, types)
        seg_src = "perframe"

    mp4_path = os.path.join(OUT_DIR, f"{file_id}_labeled.mp4")
    png_path = os.path.join(OUT_DIR, f"{file_id}_montage.png")
    n_written = make_video(imgs, langs, types, mp4_path)
    make_montage(imgs, segments, png_path, d0)
    f.close()

    type_ctr, obj_ctr = type_distribution(segments)
    return {
        "id": file_id,
        "demo_key": d0,
        "n_demos": len(demos),
        "T": int(imgs.shape[0]),
        "frames_written": n_written,
        "seg_src": seg_src,
        "n_segments": len(segments),
        "types": dict(type_ctr),
        "objs": dict(obj_ctr),
        "mp4": mp4_path,
        "png": png_path,
    }


def main():
    ids = sys.argv[1:]
    if not ids:
        print("usage: make_subgoal_videos.py <id> [<id> ...]")
        sys.exit(1)
    for fid in ids:
        info = process(fid)
        print(json.dumps(info))


if __name__ == "__main__":
    main()
