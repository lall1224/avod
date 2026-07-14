#!/usr/bin/env python3
"""Generate a tiny synthetic dataset + split for smoke tests (no real media)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avod.data_split import build_split

SCENTS = ["coffee", "wood", "fresh", "smoke", "floral", "citrus", "earth", "spice"]
TRACK_KEYS = ["foreground_scent", "scene_scent", "emotional"]


def _profile(rng: np.random.Generator):
    return {
        "gender": "男" if rng.random() < 0.5 else "女",
        "age": int(rng.integers(18, 60)),
        "scent_sensitivity": int(rng.integers(1, 6)),
        "unpleasant_tolerance": int(rng.integers(1, 6)),
        "emotional_scent_preference": int(rng.integers(1, 6)),
        "scent_confidence": int(rng.integers(1, 6)),
    }


def _segment(rng: np.random.Generator, start: float, end: float):
    return {
        "start_time": start,
        "end_time": end,
        "scent": {
            "name": SCENTS[int(rng.integers(0, len(SCENTS)))],
            "intensity": int(rng.integers(1, 6)),
            "form": "气态",
            "temperature": "温和",
        },
    }


def make_synthetic(out_dir: Path, n_clips: int = 48, seed: int = 0, feat_dim: int = 768):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(exist_ok=True)

    samples = []
    for i in range(n_clips):
        track_id = i % 3
        video = f"synth_{i:03d}.mp4"
        start, end = 0.0, 2.0
        clip_key = f"{video}|{track_id}|{start}|{end}"
        fid = video.rsplit(".", 1)[0]
        track_abbr = {0: "fg", 1: "sc", 2: "em"}[track_id]
        st, en = int(start), int(end)

        visual = rng.normal(size=(feat_dim,)).astype(np.float32)
        acoustic = rng.normal(size=(feat_dim,)).astype(np.float32)
        np.save(feat_dir / f"{fid}_{track_abbr}_{st}_{en}_visual.npy", visual)
        np.save(feat_dir / f"{fid}_{track_abbr}_{st}_{en}_acoustic.npy", acoustic)

        row = {
            "video_file": video,
            "clip_key": clip_key,
            "user_profile": _profile(rng),
            "foreground_scent": [],
            "scene_scent": [],
            "emotional": [],
        }
        row[TRACK_KEYS[track_id]] = [_segment(rng, start, end)]
        samples.append(row)

    ann_path = out_dir / "annotations.json"
    ann_path.write_text(
        json.dumps({"samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    split_path = out_dir / "split.json"
    build_split(ann_path, split_path, seed=seed)
    print(f"Wrote {ann_path}")
    print(f"Wrote {split_path}")
    print(f"Wrote features under {feat_dir}")
    return ann_path, split_path, feat_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "examples" / "synthetic"))
    parser.add_argument("--n-clips", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    make_synthetic(Path(args.out_dir), n_clips=args.n_clips, seed=args.seed)


if __name__ == "__main__":
    main()
