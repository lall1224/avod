#!/usr/bin/env python3
"""Build stratified clip-level 8:1:1 split (Split-A)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TRACK_NAMES = {0: "Foreground", 1: "Background", 2: "Emotion"}


def build_split(samples_json: Path, output_json: Path, seed: int = 42, ratios=(0.8, 0.1, 0.1)):
    data = json.loads(samples_json.read_text(encoding="utf-8"))
    samples = data["samples"]

    clip_to_track = {}
    clip_to_rows = defaultdict(list)
    for i, s in enumerate(samples):
        ck = s["clip_key"]
        clip_to_rows[ck].append(i)
        if ck not in clip_to_track:
            # clip_key format: video|track|start|end
            clip_to_track[ck] = int(str(ck).split("|")[1])

    clip_keys = sorted(clip_to_rows.keys())
    by_track = defaultdict(list)
    for ck in clip_keys:
        by_track[clip_to_track[ck]].append(ck)

    rng = np.random.default_rng(seed)
    train_clips, val_clips, test_clips = [], [], []

    for _tid, keys in by_track.items():
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_clips.extend(keys[:n_train])
        val_clips.extend(keys[n_train : n_train + n_val])
        test_clips.extend(keys[n_train + n_val :])

    def row_indices(clips):
        idx = []
        for ck in clips:
            idx.extend(clip_to_rows[ck])
        return sorted(set(idx))

    train_idx = row_indices(train_clips)
    val_idx = row_indices(val_clips)
    test_idx = row_indices(test_clips)

    assert len(set(train_clips) & set(val_clips)) == 0
    assert len(set(train_clips) & set(test_clips)) == 0
    assert len(set(val_clips) & set(test_clips)) == 0

    split = {
        "description": (
            "Split-A: stratified clip-level 8:1:1; "
            "all profile annotations of a clip share one partition"
        ),
        "split_type": "clip_level",
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "clip_counts": {
            "train": len(train_clips),
            "val": len(val_clips),
            "test": len(test_clips),
            "total_unique_clips": len(clip_keys),
        },
        "row_counts": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
            "total_rows": len(samples),
        },
        "track_distribution_test": {
            TRACK_NAMES.get(k, str(k)): v
            for k, v in Counter(clip_to_track[ck] for ck in test_clips).items()
        },
        "train_clip_keys": train_clips,
        "val_clip_keys": val_clips,
        "test_clip_keys": test_clips,
        # annotation row indices (NOT loader expanded-sample indices)
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    return split


def main():
    parser = argparse.ArgumentParser(description="Build clip-level 8:1:1 split JSON")
    parser.add_argument("--samples", required=True, help="Annotation JSON with clip_key")
    parser.add_argument("--output", required=True, help="Output split JSON path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split = build_split(Path(args.samples), Path(args.output), seed=args.seed)
    summary = {
        k: split[k]
        for k in split
        if not k.endswith("_clip_keys") and k not in ("train_indices", "val_indices", "test_indices")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
