# P0 experiment decisions and closure

## Completed in v030

1. **Viewpoint interaction.** Changed variable: extreme versus non-extreme
   viewpoint subset. Fixed variables: method, cases, metrics, and dataset
   aggregation. Output: cluster-bootstrap interaction CSV/JSON/TeX. Result:
   positive dataset-equal W, WA, and IDF1 interactions.
2. **Detector domain audit.** Changed input domain: EgoBody versus held-out
   weak-texture MVHuman. Fixed first-positive policy. Output: exact/early/late/
   miss and off-boundary positive rates. Result: 129/129 exact on EgoBody but
   40/50 early on MVHuman.
3. **Mechanism-table repair.** Oracle controls and deployable causal execution
   are separated; they are not presented as an additive ablation chain.
4. **Association denominator repair.** Runtime assignment, evaluator-valid
   pairs, conditional precision, complete continuation, and abstention are
   separated.
5. **Camera metric repair.** ATE-SE3 and ATE-Sim3 are named by alignment rule.
6. **Training/detail repair.** Final optimizer schedule, steps, token dimension,
   LoRA rank, loss weights, seeds, and final-checkpoint rule are documented.
7. **Compliance repair.** AI Use, Ethics, and Reproducibility statements are
   before References; body text ends on page 9.

## Scientifically valid but deferred

- Independently trained token/LoRA/auxiliary-loss ablations: training sources
  needed for a clean multi-run study are incomplete; masking is not substituted.
- Temporal-offset and motion-stratified shared-translation study: no fixed
  protocol and retained output currently isolate this variable.
- Detector-inclusive multi-clip runtime: the retained measurement excludes the
  detector and covers one clip; the paper discloses this rather than inventing
  an end-to-end number.
- Larger no-cut/negative detector calibration: requires a development protocol,
  not threshold selection on the reported MVHuman stress set.
- Full training hardware/duration and multi-seed checkpoint variance: not
  reliably recoverable from the retained final-run metadata.

## Rejected

- Changing 129/90/88 denominators, dropping adverse metrics, substituting
  oracle boundaries, or retuning on formal/stress test outputs.
- Merging literature-only HumanMM/Multi-THuMBS values into a same-input ranking.

