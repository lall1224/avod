# avod

面向 GitHub 分享的**纯净研究代码包**：仅含模型、数据接口与训练脚本，**不含**数据集、预提取特征、实验日志与权重。

Profile-conditioned **video → odor** prediction: fuse audiovisual features with a structured user olfactory profile (sensitivity / unpleasant-tolerance / emotional preference) via role-aligned, track-selective modulation.

## Highlights

- Profile-conditioned backbone: scent relevance routing, reliability weighting, three-way user gating
- Baselines: AV-only, AV+naive user, uniform profile modulation, MM-CLIP-style fusion
- Clip-level split utilities (Split-A): map partitions by `clip_key` (never by raw annotation-row indices as loader indices)
- Synthetic smoke test so you can run without real media

## Install

```bash
git clone https://github.com/lall1224/avod.git
cd avod
pip install -r requirements.txt
# optional editable install
pip install -e .
```

## Quick start (synthetic)

```bash
python examples/run_smoke_test.py
```

This writes tiny fake annotations/features under `examples/synthetic/`, trains 2 epochs, and saves a checkpoint.

## Train / eval on your data

Prepare annotations + features as described in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md), then:

```bash
# optional: build an 8:1:1 clip-level split
python -m avod.data_split.build_clip_level_split \
  --samples /path/to/annotations.json \
  --output /path/to/split.json

python scripts/train.py \
  --annotations /path/to/annotations.json \
  --split /path/to/split.json \
  --features /path/to/features \
  --model profile \
  --epochs 30 \
  --output-dir outputs

python scripts/eval.py \
  --checkpoint outputs/profile_seed42.pth \
  --annotations /path/to/annotations.json \
  --split /path/to/split.json \
  --features /path/to/features
```

Environment-variable alternatives: `ODOR_ANNOTATIONS_FILE`, `ODOR_SPLIT_FILE`, `ODOR_FEATURE_ROOT`.

### Models (`--model`)

| Name | Description |
|------|-------------|
| `profile` | Full method (three-way profile gating) |
| `uniform` | Same backbone, uniform profile modulation |
| `av_only` | Audiovisual only |
| `av_naive_user` | AV + naive user concat |
| `mmclip` | Cross-attention multimodal baseline |

## Layout

```
avod/                   # Python package
  backbone.py
  modulation_form_gating.py
  multimodal_dataset.py
  baselines.py
  data_split/
scripts/                # CLI train / eval
examples/               # synthetic data + smoke test
docs/DATA_FORMAT.md
```

## License

MIT — see [LICENSE](LICENSE).
