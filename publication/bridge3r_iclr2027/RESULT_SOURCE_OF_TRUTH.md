# Bridge3R ICLR 2027 result source of truth

This document binds every paper-facing result to one frozen source.  Paths are
relative to the `Movie3R/` repository root unless noted otherwise.  A later
paper build may copy generated tables and figures, but it must not recompute,
mix, or silently relabel these results.

| Dataset | Final internal method | Frozen result source | SHA-256 | Denominator and use |
|---|---|---|---|---|
| EgoBody-CS150 | `v19_ungated_translation_b050` | `output/v20_egobody/formal/test/aggregate/summary.json` | `91f0830b7859d678ae4d1afd3d21b053fa6db073b60416f5c7fdcdf1eea26a18` | 129 cases from 43 recordings; primary same-protocol internal and compact external comparison |
| EgoHumans-CS100 | `v19_egohumans_frozen` | `output/v19_egohumans/test/summary/summary.json` | `2515b3d9dc4f77e4358be3c99193a50cdd3d8b1e83909ea3dbfdbd6666e436fc` | 90 evaluator-available cases from 116 inferred camera pairs; transfer evidence and full external accounting belong in the supplement |
| Harmony4D-CS150 | `bridge3r_unified_half_translation` | `output/v17_harmony4d/unified_half_translation_audit/paper/summary.json` | `51ebde2f0ca2d70e54bd3ba948ab4ea47bc2821aabc6dd34b4ec34728520b7da` | 88/100 shared-internal evaluator-available cases; transparent stress test, not untouched Test evidence |

## External evidence

- EgoBody external values are bound by the retained artifact manifest at
  `../../ICLR-paper/bridge3r_iclr2027/versions/v002_20260824_harmony4d_final/manuscript/artifacts/egobody_v20/artifact_manifest.json`
  (SHA-256 `bbc04442e934727ff1a5ac7f9b8c571f06513d4eef9ab5d7e356efdd2ca37363`).
- Harmony4D five-method external accounting is bound by
  `../../data/Harmony4D_work_v17_full_test/external_predictions/harmony4d_final_artifacts/artifact_manifest.json`
  (SHA-256 `d0d5a84eac74c6239113944fc37a16e674e71d3866da92ada911a578441e1cf2`).
- PromptHMR no-SPEC is an audited non-official adapter.  It is supplementary
  sensitivity evidence only and never a main-text official PromptHMR result.
- The detector/manifest equivalence of all 116 frozen EgoHumans runtime
  reports is recorded in
  `publication/bridge3r_iclr2027/evidence/egohumans_detector_equivalence.json`
  (SHA-256 `c3923bfdbff925e1abf688ca2ee516ff7a0b5517bf2fd281704db88c83ed1927`).
- The frozen EgoHumans final adapter is reproduced by the publication
  transaction on all 116 retained Test caches in
  `publication/bridge3r_iclr2027/evidence/egohumans_publication_entry_equivalence.json`
  (SHA-256 `29c8cfb24566665f896c91f1cb06e56245863d6ffe441ea055398dbd8e9ad6f9`).
  The comparison requires exact finite values, dtype and shape, together with
  the same NaN padding mask. It does not claim a replay for EgoBody or
  Harmony4D: their raw/cache inputs were removed during disk management and
  must be restored before that dataset-specific adapter check can be performed.
- HumanMM and Multi-THuMBS remain literature context: their exact manifest,
  evaluator, code, and baseline predictions are not available, so their paper
  numbers never enter a same-manifest ranking table.

## Integrity rules

1. W/WA and other conditional errors must always state their valid-case count.
2. Coverage, IDF1, inference failures, and availability must not be collapsed
   into one number or conditioned on a method's successful predictions.
3. Harmony4D's 12 legacy prediction-dependent initial-match failures are not
   called method-independent exclusions and are not folded into external
   inference-failure rates.
4. The final manuscript may report only claims listed as supported in
   `CLAIM_EVIDENCE_LEDGER.csv`.
