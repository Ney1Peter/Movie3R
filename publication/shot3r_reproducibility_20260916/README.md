# Shot3R reproducibility snapshot (2026-09-16)

This is a lightweight, Git-safe snapshot of numerical evidence used by the
current manuscript. It is not a copy of the paper source or of raw predictions.

- `results/frozen_internal/`: exact primary aggregate summaries and hashes.
- `results/v072_artifacts/`: CSV/JSON/LaTeX/Markdown evidence exported with the
  current manuscript; raster/vector figures are intentionally omitted.
- `results/registration_baselines/`: frozen protocol, paper tables, aggregate
  and per-case metrics, and integrity records for traditional registration.
- `results/runtime_h4d150/`: seven-method runtime protocol and measurements.
- `training/`: final module training curve/log records.
- `weights/`: the small causal cut detector and its audit report.

The full Shot3R checkpoint is external and is identified in
`../../docs/recovery/WEIGHTS.md`.

