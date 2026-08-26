# BRIDGE3R Harmony4D train-only blend-sensitivity protocol

Date: 2026-08-26

## Aim and scope

This is a *sensitivity analysis*, not a new configuration-selection exercise.
The publication configuration remains fixed at camera coefficient
`alpha=1.0`, translation-only boundary registration, and translation blend
`lambda=0.5`.  No result produced by this protocol may alter that configuration
or the already reported EgoBody, EgoHumans, or Harmony4D $N=88$ tables.

The protocol quantifies how the same causal shared-translation mechanism varies
with `lambda` on previously unused Harmony4D **training** archives.  It does
not access an official Harmony4D test archive.

## Isolation from earlier Harmony4D work

The following archives/captures are excluded because they appear in retained
development, holdout, qualitative, multi-cut, or main-table artifacts:

- train/01_hugging, train/03_grappling2, train/09_karate, train/10_karate2,
  train/11_karate3, and train/15_mma4;
- all seven official-test archives;
- any capture selected by the retained four-capture multi-cut control.

The three fixed development archives are:

1. `train/02_grappling.zip`
2. `train/07_ballroom.zip`
3. `train/12_mma.zip`

For every archive, the existing stager selects the first projection-valid
capture in its frozen SHA-256 structural order.  It uses only archive
structure, calibration, and visibility; it receives no model prediction or
metric.  Each selected capture contributes the four fixed CS150 angle strata
(small, medium, large, extreme), for at most 12 development cases.

## Frozen candidates

`harmony4d/bridge3r_v016_lambda_sensitivity_candidates.json` is the complete
grid:

- `v16_0_m15_geometry` (parent);
- translation-only shared correction at `lambda` = 0.25, 0.50, 0.75, and 1.00.

All candidates use the same immutable `m3_b0_only` cache, the same
prediction-only association, no reliability gate, no root filter, and the
same evaluator.  The grid cannot be extended after any development metric is
read.  `lambda=0.50` is included only to test the robustness of the
already-fixed publication value; it is not promoted or reselected by this
study.

## Measurement and reporting

For every candidate we will report case-macro mean and median W-MPJPE,
WA-MPJPE, MPJPE, MPVPE, ATE-Sim3, IDF1, coverage, acceleration, and
seam-root error, plus the number of evaluator-unavailable cases.  The
paper-facing sensitivity plot/table will show all five values without choosing
the best one.  It will be supplementary-only unless its protocol and replay
artifacts are independently verified.

The evaluator is strictly post-materialisation.  Candidate geometry receives
only current/past predictions and the causal boundary; GT camera, identity,
and body annotations are evaluator-only.

## Fail-closed rules

- Any archive-integrity, staging, manifest-digest, inference, or evaluation
  failure is recorded and no partial aggregate is promoted.
- A candidate with a local MPJPE or MPVPE increase above 2% versus the parent,
  or a coverage decrease, is retained in the result but cannot be described as
  a safe alternative.
- The study does not update the primary method, re-run a main table, or alter
  any frozen candidate artifact.
