# V10 Large Four-Source Dataset Setup

Date: 2026-07-08

## Purpose

Prepare a large four-source training setup by reusing the existing V9 120h/angle60
data infrastructure.

There are two separate uses:

1. `src/train.py` large baseline:
   train the current V9-style correct-token model on a larger four-source data
   schedule. This is kept as a strong baseline for later V10 comparison.

2. V10 static streaming-alignment probe:
   train only the learnable segment alignment MLP with oracle AABB boundaries
   and the current W5 body-anchor losses.

These are intentionally separate. The W5 losses are for the V10 alignment probe,
not for the normal V9 correct-token training criterion.

For the current oracle-boundary V10 alignment probe, AAAA clips are not used.
AAAA has no shot change, so the intended behavior is to run original Human3R
without a segment-to-global alignment transform. AAAA will become useful again
when training/evaluating the detector or no-op branch.

## Formal Large Baseline Config

Config:

- `config/train_v10_4source_angle60_large_pose_concat_human_mean_lora_bs10.yaml`

Base:

- Inherits `train_v9_4source_angle60_pose_concat_human_mean_avatarrex_weighted_lora_bs10`.
- Keeps pose correct-token pooling as `concat_mlp`.
- Keeps human latent correction pooling as `mean`.
- Starts from the original Human3R checkpoint.

Data:

- AvatarReX: 24k AABB + 8k AAAA logical samples.
- THuman: 24k AABB + 8k AAAA logical samples.
- MVHuman100: 24k AABB + 8k AAAA logical samples.
- MVHuman200: 24k AABB + 8k AAAA logical samples.

Schedule:

- `train_mixed_epoch_steps: 150`
- `epochs: 200`
- `eval_freq: 10`
- `save_freq: 10`
- `early_stopping_patience: 50`

Launcher:

- `scripts/training/run_v10_4source_large_baseline.sh`

The launcher prints the command by default and only starts with `--start`.

## V10 Alignment Probe Large Map

Manifest map:

- `config/manifests/v10_static_alignment_4source_large_angle60/manifest_map.json`

It points to the existing V9 four-source angle>=60 AABB manifests:

- AvatarReX: 8000 AABB records
- THuman: 8000 AABB records
- MVHuman100: 8000 AABB records
- MVHuman200: 8000 AABB records

No large jsonl files are copied.

W5 probe launcher:

- `scripts/training/run_v10_static_alignment_4source_large_w5.sh`

Default:

- `SAMPLES_PER_SOURCE=2000`
- `STEPS=8000`
- AABB-only records
- Oracle boundary is frame index `2`
- Per-frame shot labels are `[0, 0, 1, 0]`
- `body_frame_weight=1.0`
- `body_vector_weight=1.0`
- `body_anchor_weight=5.0`
- `body_vertical_weight=5.0`

To consume the full current AABB pool:

```bash
GPU=7 SAMPLES_PER_SOURCE=8000 STEPS=12000 bash scripts/training/run_v10_static_alignment_4source_large_w5.sh --start
```

## Notes

- Motion/speed features are not used in this setup.
- AAAA, long-sequence and pattern data are not mixed into the W5 alignment large
  probe yet.
- The detector is still assumed oracle for this probe; alignment and boundary
  detection remain separated.
