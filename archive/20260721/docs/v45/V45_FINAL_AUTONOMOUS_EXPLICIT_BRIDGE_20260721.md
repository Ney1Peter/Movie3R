# V45 Final Autonomous Explicit Boundary Bridge

> Superseded by the V46/V47 geometry-integrity audit. The camera-only metric
> gain below uses shot-dependent scaling that changes the Human3R human/scene
> relation. It is retained as a diagnostic upper bound, not as the selected
> deployable output.

## Final Decision

Selected deployable method:

```text
camera cut
-> hard reset Human3R
-> V22 independent DA3 metric shot scales
-> V16 torso/gravity bounded rotation
-> V32 texture-safe conditional VGGT 1+1 rotation
-> V22 explicit human-root camera translation re-solving
-> one fixed shot-level transform for camera, pointmap, and SMPL-X
```

V36 human-jump adaptive capping is rejected by H7. The V43 scene-gated background-scale replacement is also rejected by H7.

## Rotation Validation

| Set | Count | Fixed mean/P95 | Torso mean/P95 | V32 mean/P95 | V36 mean/P95 |
|---|---:|---:|---:|---:|---:|
| original180 | 180 | 24.20/73.61 | 15.67/52.21 | 12.09/37.75 | 12.09/37.75 |
| holdout1 | 120 | 22.32/65.35 | 15.02/48.46 | 12.39/34.49 | 12.37/34.49 |
| holdout2 | 120 | 22.02/72.27 | 14.85/49.13 | 12.66/40.02 | 12.43/38.20 |
| holdout3 | 120 | 19.09/58.07 | 13.78/41.77 | 11.65/32.96 | 11.65/32.96 |
| holdout4 | 120 | 22.71/66.60 | 15.44/50.64 | 12.13/33.50 | 12.15/33.50 |
| holdout5 | 120 | 22.05/68.53 | 14.44/46.82 | 12.82/45.73 | 12.53/42.94 |
| holdout6 | 120 | 25.01/67.83 | 16.99/52.94 | 15.55/47.22 | 15.55/47.22 |
| holdout7_valid | 179 | 20.66/61.57 | 13.83/40.95 | 11.78/34.22 | 11.85/34.22 |

Across `1079` paired cuts, V32 rotation is the retained rule. On H7, V36 changed `2` cases, improved none by >5 deg, and harmed one by >5 deg (`11.78 -> 11.85 deg` mean).

## Original 180 End-To-End

| Camera T mean/P95 | Rotation mean/P95 | Human motion | Scene mean/P95 | Catastrophic |
|---:|---:|---:|---:|---:|
| 0.434/1.040 m | 12.09/37.75 deg | 0.012 m | 0.288/0.683 m | 2.22% |

## Translation Search

- V38: 120 acceptable, 15 rotation-dominated, 16 metric-depth-dominated, 7 metric-transverse-dominated, and 21 mixed cases. The residual tail is concentrated in MVHuman.
- V39: pelvis/torso and one/five-frame root-scale variants differ only at millimeter-level means and do not provide a stable safety gain.
- V40: unrestricted post-cut background scale improves development mean from `0.434` to `0.406 m`, but contains harmful corrections.
- V41/V42: the 2 cm scene gate selected the same five development cases in all nine sampling runs and harmed none there.
- V44 H7: the frozen gate selected `7` cases, improved `4`, but harmed `3` by >5 cm and `2` by >10 cm. It is rejected.
- Post-hoc 7.5% bound: selected `5` H7 cases and reduced harm to `1` >5 cm / `0` >10 cm, but did not reach zero harm. It is an H8 hypothesis, not a selected rule.

## What Failed and Why

The DA3 background cue can improve metric scale on some MVHuman cuts, but Human3R scene continuity is not a reliable proxy for camera-translation correctness on an unseen set. A candidate can make the two predicted point clouds look closer while moving the real camera farther from GT. Therefore scene-only gating is not a safe deployment rule.

## Final Scope

The final method is training-free, cut-only, 1+1 streaming, and uses no token correction, learned selector, learned SE(3), GT depth, full future shot, BA, or recurrent-state edit. One H7 cut has no reset-frame human and must fall back to Fixed Explicit.
