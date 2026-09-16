# OnlineHMR formal evidence

This directory is populated from the complete fixed-denominator run under
`data/OnlineHMR_work_v1/formal/`. Numerical LaTeX, CSV, and JSON files are
generated rather than transcribed by hand.

The `egobody/`, `egohumans/`, and `harmony4d/` subdirectories contain the
complete aggregate, case table, angle table, confidence intervals, failure
accounting, and concise summary for each protocol. `comparison/` contains the
strict three-dataset paired comparison and both generated LaTeX tables. The
comparison uses recording-macro point estimates for EgoBody and case-macro
point estimates for EgoHumans and Harmony4D.

The final table reports valid native output separately from accepted-match
availability. Process failures and structurally invalid native outputs remain
in the fixed Coverage and IDF1 denominator as zero. W-MPJPE, WA-MPJPE, and
camera ATE are conditional and always display their available case counts.
OnlineHMR is a same-input external reference with semi-online tracking and a
global camera backend, not a strict-causal same-backbone baseline.

Generation command (run from the workspace root after all three complete
aggregates exist):

```bash
Movie3R/.venv/bin/python \
  Movie3R/publication/bridge3r_iclr2027/onlinehmr/compare_onlinehmr_bridge3r.py \
  --online-root data/OnlineHMR_work_v1/formal \
  --output data/OnlineHMR_work_v1/formal/comparison \
  --datasets egobody egohumans harmony4d \
  --bootstrap-samples 10000 --seed 20260903
```

The generated evidence is copied here only after the command succeeds without
`--allow-subset`.
