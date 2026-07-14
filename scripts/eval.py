#!/usr/bin/env python3
"""Evaluate a saved checkpoint on the test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avod.baselines import build_model, evaluate_model
from avod.data_split import load_clip_level_split


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model", "profile")
    num_classes = ckpt["num_classes"]

    _, _, test_loader, _ = load_clip_level_split(
        annotations_file=args.annotations,
        split_file=args.split,
        feature_root=args.features,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    results = evaluate_model(model, test_loader, model_name=model_name, device=device)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
