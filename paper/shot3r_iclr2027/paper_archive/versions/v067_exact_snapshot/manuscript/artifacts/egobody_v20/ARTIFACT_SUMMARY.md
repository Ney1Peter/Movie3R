# EgoBody v20 paper-artifact summary

- Protocol: `Bridge3R-EgoBody-CS150-v1`.
- Frozen Test candidate: `v19_ungated_translation_b050`.
- Test recording/case count: 43 / 129 evaluable; 0 evaluator-unavailable.
- Detector exact/error: 129 / 0 over 129 Test cases.
- Strict Human3R forward FPS: 3.0 mean.
- Runtime values are component/whole-protocol wall-clock diagnostics on a shared server, not deployed single-method or hardware-normalized throughput.
- The recorded memory value is host RSS; deployed GPU peak memory is unmeasured.
- Multi-THuMBS numbers are different-protocol literature context, not a direct leaderboard comparison.
- Reliability gate: Disabled.
- Materialized detector events / detector-miss parent reuse: 129 / 0.
- Reuse array/metric exactness: N/A (0 reuse) / N/A (0 reuse).
- Materialized W harm over 5/10/20%: 7 / 5 / 5.
- Worst materialized W harm: 66.7%.
- Same-protocol external baselines: TRACE, PromptHMR (official SPEC), and
  PromptHMR (no-SPEC) are evaluated on the identical 43-recording/129-case
  manifest; the official SPEC aggregate is recorded in
  `external_baselines/bridge3r_eval/PROMPTHMR_SPEC_EGOBODY_TEST.md`.
