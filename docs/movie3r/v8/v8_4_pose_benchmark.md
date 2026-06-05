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
