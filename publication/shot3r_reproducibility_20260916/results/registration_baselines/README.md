# Shot3R traditional-registration baselines

This directory is the frozen return package for the protocol in
`Shot3R_传统配准基线实验执行协议_20260915.md`.

R0 reconstructs each shot independently with the public Human3R checkpoint.
R1 adds prediction-only pelvis translation, R2 adds prediction-only robust
human-joint SE(3), and R3 performs offline FPFH/RANSAC plus point-to-plane ICP
on predicted background point clouds. Every failed fit remains in the fixed
denominator and uses an exact R0 fallback. The same transform is applied to
the post-cut cameras, joints, meshes, and cached scene geometry.

Parameters were selected once per dataset on complete Development splits,
then passed through a one-shot Holdout execution gate. Frozen Test prediction
and transform files were sealed before the independent GT evaluator was
opened. `provenance/verification.json` audits this order and verifies exact
failure fallback, unchanged pre-cut arrays, and local-pose invariance.

Paper-facing outputs are in `tables/`, `figures/`, `metrics/aggregate.json`,
and `RESULT_SUMMARY_FOR_PAPER.md`. Case-level evidence and failure diagnostics
remain available in `metrics/case_metrics.csv` and `metrics/failures.csv`.

Important aggregation note: EgoBody headline values are recording macro over
43 recordings. EgoHumans keeps the existing paper's case-macro headline so
the frozen Shot3R values remain exact; capture-macro values are retained in
the aggregate JSON and all paired confidence intervals resample captures.
All paired intervals use 100,000 bootstrap samples with seed 20260915.
