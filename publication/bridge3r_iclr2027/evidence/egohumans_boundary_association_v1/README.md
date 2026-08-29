# EgoHumans formal-90 boundary-association evidence

This retained snapshot summarizes the completed prediction-only direct
first-post-cut association audit on the frozen 90-case, 27-capture EgoHumans
protocol.

## Provenance

- Formal manifest SHA-256:
  `682f0b4c5eda66a602a7462a60b3842a5b28d9dc331b62eb74668a07858e056f`
- Frozen runtime-binding SHA-256:
  `11494dbf43374374f40add2843cd175245ea60732ea4de2ae8ab7d006acb0057`
- Source aggregate:
  `output/bridge3r_egohumans_association_formal90/final.json`
- Source aggregate SHA-256:
  `99ad4e8aff57f5e2b2c870f03f6702dd845e5b8b5d9303e35ea856772556a77f`
- Runtime GT access: `false`
- Future post-cut frames used by association: `0`

All 180 frozen runtime inputs (90 runtime reports and 90 prediction archives)
were re-hashed after completion and matched the immutable runtime binding.
The audit completed on exactly 90 unique cases from 27 captures, with no
partial outputs or failed capture reports.

## Result

The matcher is correct on 109 of 123 evaluator-valid proposed pairs
(88.62% pair-micro; 83.33% case macro, 95% case-bootstrap interval
[74.17%, 91.67%]). Only 60 of 90 clips contain an evaluator-valid pair.
There are 108 evaluator-excluded proposals, and correct continuation coverage
is 109/282 (38.65%). Runtime abstention is 0/231 available predicted pair
slots. The angle-resolved table retains both the favorable at-least-150-degree
result (27/28, 96.43%) and the adverse Large stratum (14/23, 60.87%).

These are conditional association-precision results, not evidence of complete
person recovery. Ground-truth identities and calibration are used only by the
evaluator. The evaluator-only identity upper bound is not a deployable
baseline.

The complete per-capture ledgers and immutable runtime binding remain under
the source output directory; this Git snapshot intentionally excludes the
large prediction archives.
