# BRIDGE3R result-source map

| Paper evidence | Denominator / unit | Source of truth | Paper destination |
|---|---|---|---|
| EgoBody paired result | 129 cases, 43 recording macro | `Movie3R/output/v20_egobody/formal/test/aggregate/summary.json` and generated aggregate tables | Main Table 1; Supplement |
| EgoHumans paired result | 90 clips, 27 capture hierarchy | `Movie3R/output/v19_egohumans/test/summary/summary.json` and paired statistics | Main Table 1; Supplement |
| Harmony4D paired result | 88 cases, 25 captures | `Movie3R/output/v17_harmony4d/unified_half_translation_audit/paper/summary.json` | Main Table 1; Supplement |
| Viewpoint all/extreme macro | Equal weight over three datasets | `manuscript/artifacts/cross_dataset_viewpoint/viewpoint_evidence.json` | Main Table 2; Supplement |
| Extreme-minus-non-extreme interaction | 20,000 recording/capture-cluster bootstrap draws, seed 20260830 | `viewpoint_interaction_statistics.csv` and `viewpoint_evidence.json` | Main Table 2; Supplement full table |
| EgoHumans association | 109/123 valid endpoint pairs; 109/282 all GT continuations | retained association audit | Main Table 3(b); Supplement |
| Harmony4D association | 142/147 valid endpoint pairs; 142/176 all GT continuations | retained association audit | Main Table 3(b); Supplement |
| EgoBody detector timing | 129 cases | retained detector summary | Main Table 3(c); Supplement |
| MVHuman detector timing | 50 fixed weak-texture stress inputs; 7,450 transitions | `manuscript/artifacts/detector_generalization/` generated from retained runtime traces | Main Table 3(c); Supplement |
| Runtime and memory | one 100-frame EgoHumans clip, 1 warm-up + 3 repetitions | `Movie3R/publication/bridge3r_iclr2027/evidence/runtime_memory_v1/` | Discussion; Supplement |
| Lambda sensitivity | 12 training-only development cases, 9 evaluator-complete | retained Harmony4D development summary | Supplement only |
| AIST++ single/repeated cuts | separate fixed 100-source protocols | retained AIST++ formal aggregates | Supplement only |
| MVHuman geometric stress | 50 fixed cases | `Movie3R/output/bridge3r_mvhuman_v1/internal/formal_aggregate.json` | Supplement failure audit |

## Metric naming lock

- EgoBody and Harmony4D primary camera column: ATE-Sim3.
- EgoHumans primary camera column: ATE-SE3; ATE-Sim3 is explicitly diagnostic.
- W/WA are reported in millimetres; camera ATE is in metres.
- Coverage is availability accounting, not a body-error metric.
- Conditional association accuracy and complete-continuation recovery use
  different denominators and must never be collapsed.

