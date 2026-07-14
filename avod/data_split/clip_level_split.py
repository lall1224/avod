"""
Clip-level split loader (Split-A).

IMPORTANT: split JSON may contain `train_indices` as *annotation row* indices.
Expanded loader samples must be mapped via clip_key (Split-A) or raw_row_idx
(annotator-level), never by treating annotation-row indices as loader indices.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

from torch.utils.data import DataLoader, Subset

from ..multimodal_dataset import MultimodalOdorDataset


def _indices_from_clip_keys(full_dataset, train_keys, val_keys, test_keys):
    train_indices, val_indices, test_indices = [], [], []
    for i, sample in enumerate(full_dataset.samples):
        ck = sample.get("clip_key")
        if ck in train_keys:
            train_indices.append(i)
        elif ck in val_keys:
            val_indices.append(i)
        elif ck in test_keys:
            test_indices.append(i)
    return train_indices, val_indices, test_indices


def _indices_from_raw_rows(full_dataset, train_rows, val_rows, test_rows):
    train_indices, val_indices, test_indices = [], [], []
    for i, sample in enumerate(full_dataset.samples):
        raw_idx = sample.get("raw_row_idx")
        if raw_idx is None:
            continue
        if raw_idx in train_rows:
            train_indices.append(i)
        elif raw_idx in val_rows:
            val_indices.append(i)
        elif raw_idx in test_rows:
            test_indices.append(i)
    return train_indices, val_indices, test_indices


def load_clip_level_split(
    annotations_file: Optional[str] = None,
    split_file: Optional[str] = None,
    feature_root: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    visual_dim: int = 768,
    acoustic_dim: int = 768,
) -> Tuple[DataLoader, DataLoader, DataLoader, MultimodalOdorDataset]:
    """
    Load train/val/test loaders from annotation JSON + split JSON + feature root.

    Paths may also be set via env:
      ODOR_ANNOTATIONS_FILE, ODOR_SPLIT_FILE, ODOR_FEATURE_ROOT
    """
    annotations_file = annotations_file or os.environ.get("ODOR_ANNOTATIONS_FILE")
    split_file = split_file or os.environ.get("ODOR_SPLIT_FILE")
    feature_root = feature_root or os.environ.get("ODOR_FEATURE_ROOT")

    if not annotations_file or not split_file:
        raise ValueError(
            "Need annotations_file and split_file "
            "(or ODOR_ANNOTATIONS_FILE / ODOR_SPLIT_FILE)."
        )

    print("=" * 60)
    print("Loading clip-level split")
    print("=" * 60)
    print(f"  annotations: {annotations_file}")
    print(f"  split:       {split_file}")
    print(f"  features:    {feature_root}")

    with open(split_file, "r", encoding="utf-8") as f:
        split = json.load(f)

    split_type = split.get("split_type", "clip_level")
    print(f"  split_type:  {split_type}")

    full_dataset = MultimodalOdorDataset(
        data_path=annotations_file,
        feature_root=feature_root,
        sample_mode="segment",
        visual_dim=visual_dim,
        acoustic_dim=acoustic_dim,
        speech_dim=0,
    )

    if split_type == "annotator_level" and "train_indices" in split:
        train_indices, val_indices, test_indices = _indices_from_raw_rows(
            full_dataset,
            set(split["train_indices"]),
            set(split["val_indices"]),
            set(split["test_indices"]),
        )
        split_mode = "raw_row_idx"
    elif "train_clip_keys" in split:
        train_indices, val_indices, test_indices = _indices_from_clip_keys(
            full_dataset,
            set(split["train_clip_keys"]),
            set(split["val_clip_keys"]),
            set(split["test_clip_keys"]),
        )
        split_mode = "clip_key"
    else:
        raise ValueError(
            f"Unsupported split file: need train_clip_keys or annotator-level "
            f"train_indices ({split_file})"
        )

    print(f"  split_mode:  {split_mode}")
    print(
        f"\nSamples: Train={len(train_indices)}, Val={len(val_indices)}, "
        f"Test={len(test_indices)} (loaded {len(full_dataset.samples)})"
    )
    print(f"Vocab size: {len(full_dataset.scent_name_vocab)}")
    print("=" * 60)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch_cuda_available(),
        drop_last=len(train_dataset) >= batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_cuda_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_cuda_available(),
    )
    return train_loader, val_loader, test_loader, full_dataset


def torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-load a clip-level split")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train_loader, _, _, _ = load_clip_level_split(
        annotations_file=args.annotations,
        split_file=args.split,
        feature_root=args.features,
        batch_size=args.batch_size,
    )
    batch = next(iter(train_loader))
    print("batch visual", batch["visual"].shape)
