# Bridge3R correction-module training--test overlap audit

- Checkpoint SHA-256: `de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265`
- Training config SHA-256: `67672e48db6692e35927387dd7280bfdbd87052e773e7db28139ac1bf0ae8bbb`
- Scope: the Bridge3R correction checkpoint and its five configured fine-tuning sources. The inherited Human3R pretraining corpus is a separate provenance scope and is not reconstructed by this audit.
- Conservative policy: all 192 MultiHuman train candidates are treated as potentially observed, although only 96 are sampled per epoch.

## Result

| Benchmark | Cases | Captures | Source overlap | Capture overlap | Event overlap | Frame/member overlap |
|---|---:|---:|---:|---:|---:|---:|
| EgoBody | 129 | 43 | 0 | 0 | 0 | 0 |
| EgoHumans | 90 | 27 | 0 | 0 | 0 | 0 |
| Harmony4D | 88 | 25 | 0 | 0 | 0 | 0 |

All three benchmark identities are disjoint from the conservative correction-training universe at source, capture, event, and frame/member levels. The conclusion uses dataset-qualified identities; the retained unqualified-token sanity intersections are also empty.

## Interpretation boundary

This proves zero overlap for the correction-module fine-tuning data bound by the audited configuration. It does not claim that the base Human3R model was pretrained without any benchmark-related source, because the complete inherited pretraining manifest is not part of the present evidence.
