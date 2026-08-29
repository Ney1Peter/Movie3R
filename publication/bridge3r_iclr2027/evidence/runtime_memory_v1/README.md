# Runtime/memory table provenance

`runtime_memory_table.tex` is a paper-ready rendering of the sealed
standardized runtime report. It can be inserted with:

```tex
\input{runtime_memory_table.tex}
```

The table requires the manuscript's existing `booktabs` package. It does not
modify or recompute any experimental value.

## Source bindings

- Source JSON: `egohumans_fencing002_extreme_l20_gpu4.json`
  - SHA-256: `f8a4322f3aa2673bf82894045f2873bd1ab05a85a4260b8b49670a54f5b5d736`
- Human-readable source: `egohumans_fencing002_extreme_l20_gpu4.md`
  - SHA-256: `84407432cccdc971341a750c4c166d1d520486adb5b703d2e1150274b476b732`
- Independent integrity/stability audit: `integrity.json`
  - SHA-256: `584a5985b37666e403dd79bc773bfee0169679b1d2bd8003ad40d9d8782c0660`
  - Status: all checks pass.

The source JSON additionally binds the exact benchmark script, formal
EgoHumans manifest, both checkpoints, Git revision, Python/PyTorch/CUDA stack,
GPU UUID, driver, and per-repetition measurements by path and SHA-256.

## Fixed protocol

- Case: `ego_test_fencing_002_fencing_extreme_cam10_cam07_b00301`.
- Data: preregistered EgoHumans test case, 100 frames with boundary index 50
  and a 179.63-degree camera rotation span.
- Hardware: GPU 4, NVIDIA L20, UUID
  `GPU-0227ecd0-4186-14bc-2c6b-6bf6377fbd7f`.
- Input: model size 512, batch size one.
- Precision: FP32 with autocast and TF32 disabled.
- Repetition policy: one warm-up followed by three timed runs.
- Aggregation: median time/FPS and maximum `torch.cuda.max_memory_allocated`
  across the three timed runs.
- Excluded from timing: checkpoint loading, RGB loading/decoding/resizing, and
  causal detector latency.
- Included in timing: neural reconstruction, model-output decoding, and, for
  the single-cut route, prediction-only association and locked geometry.

The single-cut path evaluates a 51-frame read-only shadow prefix and a
50-frame clean-reset post-cut rollout, producing 100 output frames. Its FPS is
therefore explicitly an amortized output FPS, not a per-forward throughput.

## Mechanical value mapping

| TeX row | Seconds source | FPS source | Peak source |
|---|---|---|---|
| Strict Human3R | `summary.strict_human3r_seconds` | `summary.strict_human3r_fps` | `summary.strict_peak_allocated_bytes / 2^30` |
| Bridge3R (no cut) | `summary.bridge3r_no_cut_seconds` | `summary.bridge3r_no_cut_fps` | `summary.bridge3r_no_cut_peak_allocated_bytes / 2^30` |
| Bridge3R (single cut) | `summary.bridge3r_single_cut_seconds` | `summary.single_cut_amortized_fps` | `summary.bridge3r_single_cut_peak_allocated_bytes / 2^30` |

All displayed values are rounded to three decimals. The unrounded values and
all individual repetitions remain in the source JSON. The stability audit
reports runtime coefficients of variation of 0.526%, 0.923%, and 0.278% for
Strict Human3R, Bridge3R no-cut, and Bridge3R single-cut, respectively.
