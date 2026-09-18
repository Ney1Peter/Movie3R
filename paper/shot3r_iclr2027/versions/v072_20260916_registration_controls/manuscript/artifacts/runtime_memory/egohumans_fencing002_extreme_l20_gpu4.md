# Standardized runtime and peak-memory benchmark

- Case: `ego_test_fencing_002_fencing_extreme_cam10_cam07_b00301` (100 frames, boundary 50)
- GPU: NVIDIA L20 (`4, NVIDIA L20, GPU-0227ecd0-4186-14bc-2c6b-6bf6377fbd7f, 550.127.08, 46068`)
- Input/precision: 512, batch size 1, FP32 with TF32 disabled
- Timing: 1 warm-up + 3 measured repetitions
- RGB decode/resize and checkpoint loading are excluded; model output decoding and locked geometry are included.

| Route | Median seconds / 100 output frames | FPS | Max torch peak allocated (GiB) |
|---|---:|---:|---:|
| Strict Human3R | 31.020 | 3.224 | 5.29 |
| Bridge3R no-cut | 31.125 | 3.213 | 5.36 |
| Bridge3R single-cut transaction | 32.191 | 3.106 | 5.25 |

The single-cut transaction adds **1.066 s** (3.42%) over the Bridge3R no-cut path, yielding 3.106 amortized output FPS.

The transaction consists of a 51-frame shadow prefix, a 50-frame clean-reset post-cut rollout, output decoding, prediction-only association, and the locked half-translation publication geometry. Detector latency is not included.

All individual repetitions and complete hashes are recorded in the adjacent JSON.
