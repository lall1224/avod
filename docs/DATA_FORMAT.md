# Data format

This repo does **not** ship the VOD / proprietary corpus. Provide your own annotations and precomputed features in the layout below.

## Annotation JSON

Top-level object:

```json
{
  "samples": [ { "...": "..." } ]
}
```

Each sample row:

| Field | Type | Meaning |
|-------|------|---------|
| `video_file` | string | Video path or id (used to locate feature files) |
| `clip_key` | string | Unique clip id: `video\|track\|start\|end` (required for Split-A) |
| `user_profile` | object | See profile fields below |
| `foreground_scent` | list | Track 0 segments |
| `scene_scent` | list | Track 1 (background / scene) segments |
| `emotional` | list | Track 2 segments |

Recommended: **one row = one clip** (only one track list non-empty). The loader can also expand a multi-track video row into multiple samples.

### `user_profile`

| Field | Type | Notes |
|-------|------|-------|
| `gender` | string | e.g. `男` / `女` (mapped internally) |
| `age` | int | years |
| `scent_sensitivity` | int | typically 1–5 |
| `unpleasant_tolerance` | int | typically 1–5 |
| `emotional_scent_preference` | int | typically 1–5 |
| `scent_confidence` | int | typically 1–5 |

### Segment object

```json
{
  "start_time": 0.0,
  "end_time": 2.0,
  "scent": {
    "name": "coffee",
    "intensity": 3,
    "form": "气态",
    "temperature": "温和"
  }
}
```

`scent.name` values are collected into a vocabulary at dataset construction time. Class count = vocab size.

Legacy 4-track rows (`object_scent` + `action_scent`) are auto-merged into `foreground_scent`.

Optional: embed `visual` / `acoustic` arrays directly in the JSON to skip a feature directory (useful for tiny demos).

## Feature files

Set `--features` / `ODOR_FEATURE_ROOT` to a directory containing NumPy arrays.

Preferred **clip-level** names (as used by the dataset):

```
{video_id}_{track}_{start}_{end}_visual.npy
{video_id}_{track}_{start}_{end}_acoustic.npy
```

where `track` ∈ `{fg, sc, em}`, and `start`/`end` are integer seconds (floor of annotation times).

Fallback **video-level** names:

```
{video_id}_visual.npy
{video_id}_acoustic.npy
```

`video_id` = basename of `video_file` without extension. Default feature dim is **768** for both visual and acoustic.

Shapes accepted: `[D]` or `[T, D]` (time is mean-pooled in the models when needed).

## Split JSON (Split-A)

Build with:

```bash
python -m avod.data_split.build_clip_level_split \
  --samples annotations.json \
  --output split.json \
  --seed 42
```

Important fields:

- `train_clip_keys` / `val_clip_keys` / `test_clip_keys` — **authoritative** partition
- `train_indices` / … — annotation **row** indices only; loaders must **not** treat these as expanded `DataLoader` indices

The loader always remaps via `clip_key` when `train_clip_keys` is present.

## Minimal example

```bash
python examples/make_synthetic_data.py --out-dir examples/synthetic
python scripts/train.py \
  --annotations examples/synthetic/annotations.json \
  --split examples/synthetic/split.json \
  --features examples/synthetic/features \
  --model profile \
  --epochs 2 \
  --batch-size 8 \
  --output-dir examples/synthetic/outputs
```
