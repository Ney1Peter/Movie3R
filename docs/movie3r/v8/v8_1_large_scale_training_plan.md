# V8.1 Large-Scale Training Plan

## Goal

Scale the current working V8.1 C-version experiment from a 10-clip sanity check to a larger AvatarReX AABB training run.

Current method to keep:

```text
A_corr_t enters decoder
-> decoder refines A_corr_t with image / human / pose / state context
-> residual/gate head updates the pose token
-> fine-tuned original pose head outputs camera_pose
```

This is the current UniCon-style decoder-in baseline. Do not switch back to sidecar/post-processing correction for the main large-scale run.

## Data Audit

Only this dataset is used for the next stage:

```text
/data/wangzheng/iJCV-CODE/data/Avatarrex_output/Training
```

There is no separate `Val` or `Test` directory under `Avatarrex_output`. Therefore, validation and testing must be built as held-out splits inside `Training`.

Current `Training` contains:

| Item | Count |
| --- | ---: |
| camera / sequence folders | 16 |
| frames per camera | 1901 |
| valid frame ids | 0-1900 |
| rgb files | 30416 |
| cam files | 30416 |
| smpl files | 30416 |
| depth files | 30416 |
| mask files | 30416 |

AABB dataloader uses 4 frames:

```text
view0 = seqA, t
view1 = seqA, t+1
view2 = seqB, t+2
view3 = seqB, t+3
```

The valid start frames are `0..1897`, so each ordered camera pair has `1898` clips.

Total possible AABB clips:

```text
16 cameras * 15 ordered target cameras * 1898 start frames
= 455,520 ordered AABB clips
```

If treating `A->B` and `B->A` as the same unordered camera pair:

```text
C(16, 2) * 1898 = 227,760 unordered-pair clips
```

## Camera-Angle Distribution

Angles below are computed from raw calibration camera poses. This is important because V8.1 pose supervision must use raw calibration, not processed `cam/*.npz` as final target.

| Angle bucket | Ordered pairs | AABB clips |
| --- | ---: | ---: |
| 0-30 deg | 28 | 53,144 |
| 30-60 deg | 38 | 72,124 |
| 60-90 deg | 54 | 102,492 |
| 90-120 deg | 38 | 72,124 |
| 120-150 deg | 38 | 72,124 |
| 150-180 deg | 44 | 83,512 |

Threshold counts:

| Threshold | Ordered pairs | AABB clips |
| --- | ---: | ---: |
| all | 240 | 455,520 |
| >= 60 deg | 174 | 330,252 |
| >= 90 deg | 120 | 227,760 |
| >= 120 deg | 82 | 155,636 |
| >= 150 deg | 44 | 83,512 |
| >= 170 deg | 14 | 26,572 |

For the next training stage, `>=60 deg` is a good default because it covers many hard camera switches without only overfitting to near-180-degree cases.

## Split Strategy

Because there is no official val/test split, use fixed manifests.

Do not rely on `max_samples` directly from the dataloader for final experiments, because the current sample order is deterministic:

```text
seqA loop -> seqB loop -> all start frames
```

So `max_samples` can accidentally bias toward early camera pairs. Instead, build manifest files with explicit `(seqA, seqB, start_frame)` records.

Recommended split unit:

```text
unordered camera pair = {seqA, seqB}
```

Hold out both directions if a pair is reserved:

```text
if {A, B} is test:
  A->B and B->A are both excluded from train
```

This avoids leakage where the model trains on `B->A` and tests on `A->B`.

Recommended high-angle split from the `>=60 deg` pool:

```text
>=60 deg unordered pairs:
  87 unordered pairs
  174 ordered pairs
  330,252 clips
```

Proposed pair-level split:

| Split | Unordered pairs | Ordered pairs | Full clips |
| --- | ---: | ---: | ---: |
| train-pair pool | 61 | 122 | 231,556 |
| val-new-pair pool | 13 | 26 | 49,348 |
| test-new-pair pool | 13 | 26 | 49,348 |

Within the train-pair pool, also reserve time ranges for same-pair validation:

| Time range | Purpose | Clips if using all 122 train ordered pairs |
| --- | --- | ---: |
| start 0-1499 | train-time pool | 183,000 |
| start 1500-1699 | same-pair val pool | 24,400 |
| start 1700-1897 | same-pair test pool | 24,156 |

The actual train/eval manifests should sample from these pools, not necessarily use all clips at once.

## Training Scale Plan

### Stage A: Medium Manifest Sanity

Purpose: confirm the 10-clip result survives a meaningful data increase.

```text
train:
  5,000-10,000 clips
  sampled from train-pair pool, start 0-1499
  angle-balanced across 60-90 / 90-120 / 120-150 / 150-180

val_same:
  200 clips
  same train camera pairs, start 1500-1699

test_same:
  200 clips
  same train camera pairs, start 1700-1897

val_new:
  200 clips
  held-out camera pairs

test_new:
  200 clips
  held-out camera pairs
```

Expected runtime from current single-GPU speed:

```text
1000 steps ~= 39 min on one L20-like GPU
10k steps ~= 6.5 hours on one GPU
```

### Stage B: Large Manifest

Purpose: test whether the correction generalizes beyond a few hand-picked pairs.

```text
train:
  20,000-50,000 clips
  angle-balanced
  many camera pairs

val/test:
  500-1000 clips per split
```

Recommended first large run:

```text
20,000 train clips
1 epoch
batch size = 1 or DDP equivalent
raw calibration pose target
load_da3_depth=False
save_final_checkpoint=True
save_last_checkpoint=False
```

### Stage C: Full Pool / Longer Training

Only do this after Stage B is stable.

Options:

```text
Use all train-time high-angle clips:
  about 183,000 clips from train pairs and train time range

or use a fixed 50k-100k manifest:
  easier to repeat, cheaper to debug
```

Full one-epoch training on all 183k train-time clips is likely expensive with the current speed, so a fixed 50k manifest is the more practical next step.

## Config Recommendations

Keep:

```text
model freeze = v8_pose_prompt_pose_head
train_criterion = V81PosePromptLoss(... pose_key='raw_camera_pose')
load_da3_depth = False
raw_calibration_root = /data/wangzheng/iJCV-CODE/data/avatarrex_lbn1
```

Use manifest paths instead of long `fixed_samples` strings:

```text
train_dataset:
  AvatarReX_AABB(..., manifest_path=".../train_20k.jsonl", load_da3_depth=False, raw_calibration_root=...)

val_dataset:
  AvatarReX_AABB(..., manifest_path=".../val_same_500.jsonl", ...)

test_dataset:
  AvatarReX_AABB(..., manifest_path=".../test_new_500.jsonl", ...)
```

Important: current `AvatarReX_AABB` supports `manifest_path`, but the manifest should be generated with raw-calibration angle filtering to avoid relying on processed pose conventions.

## Metrics and Visualization

Always report:

- raw Human3R baseline on the exact same eval manifests;
- corrected mean translation error;
- corrected mean rotation error;
- B-frame-only translation error;
- B-frame-only rotation error;
- `v8_pose_prompt_gate_mean`;
- `v8_pose_prompt_delta_norm`.

B-frame-only metrics are essential because the failure is mainly at `view2/view3` after the A->B camera switch.

After each larger training run, visualize at least:

```text
1 same-pair held-out case
1 easy new-pair held-out case
1 hard new-pair held-out case
1 failure case with highest B-frame rotation error
```

Each viewer should overlay raw Human3R cameras in gray, as in the current `8115/8116` checks.

## Risks

### Gate Collapse

The 10-sample C run had `gate_mean ~= 0`. This means the current improvement may be carried mainly by the fine-tuned pose head.

For large training, monitor whether this continues. If it does:

- lower the pose-head learning rate;
- train only the last pose-head layer;
- add weak gate supervision for `view2`;
- compare against B frozen-pose-head and D pose-head-only baselines.

### Pair Leakage

Do not test on `A->B` if train contains `B->A`. Treat unordered camera pair as the split group.

### Time Leakage

For same-pair testing, use held-out time windows. Do not randomly mix adjacent start frames into train/test, because adjacent AABB clips share 3 of 4 frames.

### Disk

Each checkpoint is about 4.5G. Large training should keep:

```text
save_last_checkpoint=False
save_final_checkpoint=True
save_freq very large
```

## Immediate Next Steps

1. Add a manifest generation script:

```text
scripts/v8_1_build_avatarrex_aabb_manifests.py
```

It should:

- read raw calibration;
- compute camera-pair angles;
- group by unordered pair;
- split train / val-new / test-new;
- split train-time / same-val-time / same-test-time;
- sample angle-balanced clips;
- write JSONL manifests.

2. Generate Stage A manifests:

```text
output/v8_1_aabb_manifests/stage_a_train_10k.jsonl
output/v8_1_aabb_manifests/stage_a_val_same_200.jsonl
output/v8_1_aabb_manifests/stage_a_test_same_200.jsonl
output/v8_1_aabb_manifests/stage_a_val_new_200.jsonl
output/v8_1_aabb_manifests/stage_a_test_new_200.jsonl
```

3. Add a Stage A config copied from the current small-batch C config.

4. Run raw Human3R baseline on all eval manifests before training.

5. Train Stage A C model.

6. Evaluate and visualize the same-pair/new-pair/failure cases.

## Stage A Preparation Status

Preparation completed on 2026-05-30. Training has not been started yet.

Generated manifests:

```text
output/v8_1_aabb_manifests/stage_a_10k/
  metadata.json
  stage_a_train_10k.jsonl
  stage_a_val_same_200.jsonl
  stage_a_test_same_200.jsonl
  stage_a_val_new_200.jsonl
  stage_a_test_new_200.jsonl
```

Manifest size:

```text
1.3M total
```

Final Stage A split:

| Manifest | Records | Ordered pairs | Unordered pairs | Start frame range | Angle buckets |
| --- | ---: | ---: | ---: | --- | --- |
| train | 10000 | 122 | 61 | 0-1496 | 2500 each |
| val_same | 200 | 102 | 57 | 1501-1696 | 50 each |
| test_same | 200 | 96 | 58 | 1700-1897 | 50 each |
| val_new | 200 | 26 | 13 | 14-1897 | 50 each |
| test_new | 200 | 26 | 13 | 2-1897 | 50 each |

Leakage checks:

```text
val_same unordered pairs are subset of train pairs: true
test_same unordered pairs are subset of train pairs: true
val_new unordered pairs intersect train pairs: 0
test_new unordered pairs intersect train pairs: 0
val_new unordered pairs intersect test_new pairs: 0
train max end frame: 1499
val_same min start / max end: 1501 / 1699
test_same min start: 1700
```

Prepared config:

```text
config/train_v8_pose_prompt_posehead_stage_a_10k_nodepth_rawpose.yaml
```

The config dry-run passed with:

```text
epochs=0
save_final_checkpoint=false
```

Dry-run verified:

- manifest paths resolve correctly;
- train/val/test dataloaders build correctly;
- raw calibration target is enabled;
- DA3 depth is disabled;
- model loads from `src/human3r_896L.pth`;
- trainable parameter groups include V8.1 prompt/residual/gate and original pose head;
- no checkpoint is saved during dry-run.

Training command to run after explicit confirmation:

```text
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp/matplotlib \
.venv/bin/python src/train.py \
  --config-name train_v8_pose_prompt_posehead_stage_a_10k_nodepth_rawpose \
  exp_name=v8_1_pose_prompt_posehead_stage_a_10k_nodepth_rawpose_gpu4 \
  output_dir=/tmp/movie3r_v8_1_pose_prompt_posehead_stage_a_10k_nodepth_rawpose_gpu4 \
  logdir=/tmp/movie3r_v8_1_pose_prompt_posehead_stage_a_10k_nodepth_rawpose_gpu4/logs
```

Expected runtime:

```text
about 6-7 hours on one GPU, based on the 10-sample run speed
```

Disk note:

```text
one final checkpoint ~= 4.5G
/tmp currently also keeps the previous small-batch checkpoint
```

Before starting Stage A training, confirm whether to keep the previous small-batch checkpoint or delete it to increase `/tmp` free space.

## 11. Stage A Run Result, 2026-05-31

Stage A has been run once with the planned C-version setup.

Training setup:

```text
config:
  config/train_v8_pose_prompt_posehead_stage_a_10k_nodepth_rawpose.yaml
train manifest:
  output/v8_1_aabb_manifests/stage_a_10k/stage_a_train_10k.jsonl
checkpoint:
  /tmp/movie3r_v8_1_pose_prompt_posehead_stage_a_10k_nodepth_rawpose_gpu4/checkpoint-final.pth
GPU:
  CUDA_VISIBLE_DEVICES=4
```

Training finished cleanly:

```text
train_loss = 1.319617
train_v8_pose_prompt_trans_err = 0.523174
train_v8_pose_prompt_rot_err_deg = 13.857566
train_v8_raw_trans_err = 0.523182
train_v8_raw_rot_err_deg = 13.854701
train_v8_pose_prompt_gate_mean = 0.0002056
train_v8_pose_prompt_delta_norm = 35.6418
```

Important interpretation:

- There was no NaN, OOM, or traceback.
- Corrected metrics are almost identical to raw metrics.
- `gate_mean` collapsed close to zero.
- Therefore this run mostly validates that the C-version training loop and pose-head fine-tuning can run at scale, but it does not yet prove that the UniCon-style residual/gate branch is doing useful correction.

Evaluation results:

| Split | Clips | Mean trans err | Mean rot err | B-frame trans err | B-frame rot err |
| --- | ---: | ---: | ---: | ---: | ---: |
| `val_same_200` | 200 | 0.2657 | 8.11 deg | 0.5206 | 16.21 deg |
| `test_same_200` | 200 | 0.3113 | 10.08 deg | 0.6113 | 19.41 deg |
| `val_new_200` | 200 | 0.3974 | 13.86 deg | 0.7837 | 26.98 deg |
| `test_new_200` | 200 | 0.4218 | 14.63 deg | 0.8328 | 28.53 deg |

Evaluation JSON outputs:

```text
output/v8_1_stage_a_eval/stage_a_val_same_200_checkpoint_final.json
output/v8_1_stage_a_eval/stage_a_test_same_200_checkpoint_final.json
output/v8_1_stage_a_eval/stage_a_val_new_200_checkpoint_final.json
output/v8_1_stage_a_eval/stage_a_test_new_200_checkpoint_final.json
```

Current conclusion:

- Same-pair held-out is easier and gives usable numbers.
- New-pair held-out is significantly harder, especially on B frames after the camera switch.
- The next method change should focus on preventing gate collapse and forcing the decoder-in correction token to matter, rather than simply training longer.

Recommended next ablations:

1. Add weak gate supervision for B frames or high pose-error frames.
2. Lower pose-head learning rate so the residual/gate branch cannot be bypassed too easily.
3. Try pose-head-last-layer-only fine-tuning.
4. Compare against a frozen-pose-head version where only prompt/residual/gate is trainable.
5. Add per-view gate logging, especially `view0/view1/view2/view3`, because the target failure is mainly at the A to B boundary and the B continuation.
