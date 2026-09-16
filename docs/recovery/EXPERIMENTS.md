# Experiment and result map

This file separates authoritative numerical evidence from large, regenerable
prediction caches.

## Source of truth

| Claim/experiment | Code | Git-backed result |
|---|---|---|
| EgoBody main evaluation | `versions/v20/egobody/` | `publication/shot3r_reproducibility_20260916/results/frozen_internal/egobody_cs150_summary.json` |
| EgoHumans main evaluation | `versions/v19/egohumans/` | `publication/shot3r_reproducibility_20260916/results/frozen_internal/egohumans_cs100_summary.json` |
| Harmony4D main evaluation | `versions/v15/harmony4d/`, `versions/v17/harmony4d/` | `publication/shot3r_reproducibility_20260916/results/frozen_internal/harmony4d_cs150_summary.json` |
| Main/public baselines and supplementary protocols | dataset runners plus `publication/bridge3r_iclr2027/onlinehmr/` | `publication/shot3r_reproducibility_20260916/results/v072_artifacts/` |
| State carry-over analysis | `publication/bridge3r_iclr2027/evaluate_egohumans_state_isolation.py` | `results/v072_artifacts/egohumans_state_isolation/` |
| Runtime benchmark | `publication/runtime_benchmark_7methods_h4d150_20260908/` | `results/runtime_h4d150/` |
| Traditional registration controls | `experiments/registration_baselines/` | `results/registration_baselines/` |
| Final alignment-module training | frozen config and `src/train.py` | `training/log.txt`, `training/metrics_epoch.jsonl`, `training/train_steps.jsonl` |

All paths in the rightmost column are below
`publication/shot3r_reproducibility_20260916/`.

## Traditional-registration control

The frozen control compares per-shot Human3R reset with pelvis translation,
robust human-joint SE(3), scene registration plus ICP, and Shot3R. It uses 129
EgoBody and 90 EgoHumans test cases. Registration baselines receive the known
cut boundary, never use GT for fitting, preserve the fixed denominator, and
fall back to the identity transform on algorithmic failure. The paper-facing
conclusion follows Decision Rule B in
`results/registration_baselines/RESULT_SUMMARY_FOR_PAPER.md`.

## Results that are not Git-backed

Full `.npz` predictions, decoded RGB frames, meshes, point clouds, rendering
caches, and individual visualization payloads are regenerable and excluded.
Their former local paths and checksums are preserved where available in the
registration provenance files and in `publication/bridge3r_iclr2027/evidence/`.

## Naming note

Older paths and JSON schemas may use `Bridge3R` or `Movie3R`. The final paper
method name is Shot3R. Historical names are retained only to keep hashes,
imports, and frozen manifests reproducible.

