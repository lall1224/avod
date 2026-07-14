#!/usr/bin/env python3
"""Train a model on annotation JSON + split JSON + precomputed features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avod.baselines import build_model, evaluate_model, train_model
from avod.data_split import load_clip_level_split


def main():
    parser = argparse.ArgumentParser(description="Train avod / baselines")
    parser.add_argument("--annotations", required=True, help="Annotation JSON")
    parser.add_argument("--split", required=True, help="Split JSON (clip_key based)")
    parser.add_argument("--features", default=None, help="Feature root (optional if inline)")
    parser.add_argument(
        "--model",
        default="profile",
        choices=["profile", "av_only", "av_naive_user", "uniform", "mmclip"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, full_ds = load_clip_level_split(
        annotations_file=args.annotations,
        split_file=args.split,
        feature_root=args.features,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(full_ds.scent_name_vocab)
    model = build_model(args.model, num_classes=num_classes)

    model, history, best_val = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        model_name=args.model,
        device=device,
    )
    results = evaluate_model(model, test_loader, model_name=args.model, device=device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{args.model}_seed{args.seed}.pth"
    torch.save(
        {
            "model": args.model,
            "state_dict": model.state_dict(),
            "num_classes": num_classes,
            "scent_name_vocab": full_ds.scent_name_vocab,
            "best_val_acc": best_val,
            "test_metrics": results,
            "history": history,
            "seed": args.seed,
        },
        ckpt,
    )
    metrics_path = out / f"{args.model}_seed{args.seed}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"best_val_acc": best_val, "test": results, "history": history},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved checkpoint: {ckpt}")
    print(f"Saved metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
