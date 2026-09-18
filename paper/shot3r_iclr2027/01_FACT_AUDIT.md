# v029 Fact Audit

## Confirmed protocol facts

- Formal multi-person denominators are EgoBody 129 cases / 43 recordings, EgoHumans 90 clips / 27 captures, and Harmony4D 88 cases / 25 captures.
- The same frozen BRIDGE3R operating point, including `lambda=0.5`, is used for the three formal datasets.
- Runtime inference uses RGB-only causal event proposals; calibration, identities, annotated cuts, strata and acceptance are evaluator-only.
- The learned boundary pathway updates 34,415,333 of 1,232,459,373 parameters (2.79%).
- Shadow human output is auxiliary and discarded; reported outputs use the shadow camera only to form the coarse gauge.
- Formal paired results include adverse outcomes: Harmony4D W-MPJPE worsens; repeated-cut camera trajectory can worsen; EgoHumans coverage decreases.

## Claims requiring narrower wording

- “largest gains at extreme viewpoints” is supported as a dataset-equal interaction for W, WA and IDF1 only after cluster bootstrap; it is not uniformly significant per dataset.
- The current token masking study is inference-time sensitivity of one checkpoint, not an independent training ablation.
- Runtime is a one-clip, detector-excluded incremental measurement, not a cross-system speed ranking.
- MVHuman is a held-out weak-texture stress test, not a fourth formal multi-person benchmark.

## Sources

- `Movie3R/publication/bridge3r_iclr2027/PAPER_METHOD_LOCK.json`
- `Movie3R/publication/bridge3r_iclr2027/METHOD_TO_CODE_FACT_AUDIT_20260829.md`
- `Movie3R/publication/bridge3r_iclr2027/CLAIM_EVIDENCE_LEDGER.csv`
- `Movie3R/output/v20_egobody/formal/test/aggregate/`
- `Movie3R/output/v19_egohumans/test/summary/`
- `Movie3R/output/v17_harmony4d/unified_half_translation_audit/paper/`
- `Movie3R/output/bridge3r_mvhuman_v1/internal/formal_aggregate.json`
