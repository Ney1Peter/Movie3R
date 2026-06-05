# V8.4 Pose Benchmark

This benchmark is sampled from the existing held-out test split, not from the
training frames:

```text
output/v8_4_mixed_aabb_aaaa_manifests_no_zxc/test_aabb_no_zxc.jsonl
output/v8_4_mixed_aabb_aaaa_manifests_no_zxc/test_aaaa_no_zxc.jsonl
/data/wangzheng/iJCV-CODE/data/Test/v8_4_mixed_aabb_aaaa
```

It also includes a small train-sanity subset. Use that only to check whether the
model can fit familiar distribution samples.

## Build

```bash
.venv/bin/python scripts/v8_4_build_pose_benchmark_from_test.py
```

Outputs:

```text
output/v8_4_pose_benchmark/test_aabb.jsonl          24 clips
output/v8_4_pose_benchmark/test_aaaa.jsonl           8 clips
output/v8_4_pose_benchmark/train_sanity_aabb.jsonl  24 clips
output/v8_4_pose_benchmark/train_sanity_aaaa.jsonl   4 clips
output/v8_4_pose_benchmark/sheets/
```

The test AABB subset covers four groups and six angle buckets:

```text
lbn1, zzr, thuman00, thuman02
015_030, 030_060, 060_090, 090_120, 120_150, 150_180
```

## Evaluate

After a checkpoint exists:

```bash
.venv/bin/python scripts/v8_4_eval_pose_benchmark.py \
  --model_path output/v8_4_train_runs/<run>/checkpoint-last.pth \
  --name <run_name>
```

Outputs:

```text
output/v8_4_pose_benchmark/eval/<run_name>/summary.json
output/v8_4_pose_benchmark/eval/<run_name>/all_rows.csv
```

Main metrics:

```text
v82_trans_err / v82_rot_err_deg
v82_raw_trans_err / v82_raw_rot_err_deg
v82_trans_improvement / v82_rot_improvement_deg
v82_gate_mean
v82_drift_loss
v82_delta_norm
```

For AABB, corrected error should be lower than raw error. For AAAA, gate and
delta should stay small and corrected error should not get worse.

## Pose Dump

For viewer/debug comparisons, also dump the full camera matrices:

```bash
.venv/bin/python scripts/v8_4_eval_pose_benchmark.py \
  --model_path output/v8_4_train_runs/<run>/checkpoint-last.pth \
  --name <run_name> \
  --dump_poses
```

Additional outputs:

```text
output/v8_4_pose_benchmark/eval/<run_name>/poses_index.jsonl
output/v8_4_pose_benchmark/eval/<run_name>/poses/<subset>/*.npz
```

Each pose dump contains:

```text
gt_c2w_abs                  original GT camera matrices
gt_c2w_rel                  GT matrices relative to view 0
raw_pose_encoding           original Human3R pose-head output
corrected_pose_encoding     V8 corrected pose-head output
raw_c2w_rel                 raw 4x4 matrices in the loss coordinate system
corrected_c2w_rel           corrected 4x4 matrices in the loss coordinate system
raw_c2w_abs_gt0             raw matrices anchored by GT view 0 for visualization
corrected_c2w_abs_gt0       corrected matrices anchored by GT view 0 for visualization
gate / drift_logit / delta_norm
```

The `*_abs_gt0` matrices use GT view 0 only as an evaluation visualization
anchor. GT is not passed into model inference.

## Checkpoints

For long V8.4 runs, `checkpoint-last.pth` should be the main in-progress
evaluation target. The current bs10 long configs refresh `checkpoint-last` every
quarter epoch for future runs, keep permanent checkpoints every 4 epochs, and
write `checkpoint-final.pth` at the end.

Recommended reporting checkpoints:

```text
checkpoint-last.pth    current training state
checkpoint-best.pth    best validation-loss state if available
checkpoint-final.pth   final state
```

When comparing two training variants, evaluate the same benchmark subsets and
the same checkpoint type first. A practical first pass is:

```text
prompt-only checkpoint-last vs pose-head-unfrozen checkpoint-last
prompt-only checkpoint-best vs pose-head-unfrozen checkpoint-best
```

After evaluating several checkpoints/runs, collect the comparison table:

```bash
.venv/bin/python scripts/v8_4_collect_pose_benchmark_results.py
```

Outputs:

```text
output/v8_4_pose_benchmark/eval/run_comparison.csv
output/v8_4_pose_benchmark/eval/run_comparison.md
```
