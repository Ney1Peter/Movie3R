# V20 Explicit Metric Shot Bridge

## Pairwise upper-bound pipeline

Hard reset -> independent single-frame DA3 metric scale -> fixed shot-scale Human3R camera/pointmap/SMPL-X -> V16 torso-motion rotation (45 deg bound) -> human-root camera translation -> pairwise bounded background pointmap depth scale.

This result is retained as an offline pairwise upper bound. The final background scale `q` is fitted from the previous-shot point cloud, so it is not an independently observable absolute scale for the new shot.

## Overall 180-cut result

| Metric | Fixed mean / P95 | V20 mean / P95 |
|---|---:|---:|
| Camera translation (m) | 1.715 / 4.123 | 0.493 / 1.263 |
| Camera rotation (deg) | 24.204 / 73.612 | 16.037 / 53.555 |
| Human root motion error (m) | 0.354 / 1.241 | 0.012 / 0.033 |
| Background scene discontinuity (m) | 0.300 / 0.682 | 0.203 / 0.568 |

- Catastrophic rate: `42.2% -> 7.2%`.
- Strict success rate: `1.1% -> 31.1%`.
- Camera, human and scene all improve: `75.0%`.
- DA3 independent two-shot inference latency: mean `0.113 s`, P95 `0.196 s`.
- Harmful correction rates: camera translation `6.1%`, rotation `6.7%`, human `0.0%`, scene `2.8%`.

## Per source

| Source | Camera T | Rotation | Human | Scene | All-three improve |
|---|---:|---:|---:|---:|---:|
| avatarrex | 0.204 | 4.40 | 0.009 | 0.212 | 93.8% |
| mvhuman100 | 0.671 | 27.27 | 0.014 | 0.062 | 93.8% |
| mvhuman200 | 0.895 | 32.12 | 0.018 | 0.124 | 63.9% |
| thuman | 0.303 | 4.38 | 0.010 | 0.394 | 45.8% |

## Revised decision

The viable deployable core is the synchronized explicit metric correction: DA3 supplies per-shot depth scale, torso motion supplies rotation, and the human root supplies the camera translation equation.

The final pairwise background residual is useful for visualization and as a teacher/upper bound, but it must not be stored as the shot's absolute scale. On 38 A->B->C scale chains, the DA3 root shot scale stayed within 20% in `97.4%` of pairs, while the effective pairwise `root_scale * q` stayed within 20% in only `39.5%` of pairs. V22 replaces it with independent first-frame absolute background calibration.

Three post-cut frames did not provide a stable gain over the first frame, so the selected streaming setting remains zero-wait.

The repeated point-sampling seed changed scene mean by less than 0.001 m and kept the scene harmful rate unchanged. The 3000-point / 15-step fast setting retained the same all-three improvement and harmful rates, with a small scene-error increase.
