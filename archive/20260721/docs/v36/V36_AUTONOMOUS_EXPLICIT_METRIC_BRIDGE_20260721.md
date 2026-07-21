# V36 Autonomous Explicit Metric Boundary Bridge

## Final Decision

Selected method: **V36 human-jump-adaptive explicit metric bridge**.

```text
camera cut
-> hard reset Human3R
-> independent DA3 root/background metric shot scales
-> V16 torso/gravity bounded rotation
-> texture-safe conditional VGGT full-RGB 1+1 rotation
-> if post-torso human heading jump < 30 deg, cap positive-consensus residual at 20 deg
-> explicit metric human-root camera translation re-solving
-> one fixed shot-level scale state and one final SE(3)
-> apply the same final SE(3) to camera, pointmap, and SMPL-X
```

No token correction, learned gate/selector, learned SE(3), recurrent-state edit, full-shot future, BA, GT depth, or runtime GT is used.

## Rotation Validation

| Set | Count | Fixed mean/P95 | Torso mean/P95 | V32 mean/P95 | V36 mean/P95 | V36 cat |
|---|---:|---:|---:|---:|---:|---:|
| original180 | 180 | 24.20/73.61 | 15.67/52.21 | 12.09/37.75 | 12.09/37.75 | 1.67% |
| holdout1 | 120 | 22.32/65.35 | 15.02/48.46 | 12.39/34.49 | 12.37/34.49 | 3.33% |
| holdout2 | 120 | 22.02/72.27 | 14.85/49.13 | 12.66/40.02 | 12.43/38.20 | 4.17% |
| holdout3 | 120 | 19.09/58.07 | 13.78/41.77 | 11.65/32.96 | 11.65/32.96 | 0.83% |
| holdout4 | 120 | 22.71/66.60 | 15.44/50.64 | 12.13/33.50 | 12.15/33.50 | 0.83% |
| holdout5 | 120 | 22.05/68.53 | 14.44/46.82 | 12.82/45.73 | 12.53/42.94 | 5.00% |
| holdout6 | 120 | 25.01/67.83 | 16.99/52.94 | 15.55/47.22 | 15.55/47.22 | 5.83% |

Across `900` disjoint cuts, V36 rotation mean/P95 is `12.64/38.15 deg` with `3.00%` catastrophic rate.
Relative to torso, it rescues `31` and introduces `0` catastrophes.
The frozen H6 check changed `1` cases and introduced `0` catastrophes versus V32.

## End-To-End 3D Output

| Method | Camera T mean/P95 | Rotation mean/P95 | Human motion | Scene mean/P95 | Catastrophic |
|---|---:|---:|---:|---:|---:|
| v22 | 0.490/1.218 m | 15.67/52.21 deg | 0.012 m | 0.288/0.683 m | 7.22% |
| v32 | 0.434/1.040 m | 12.09/37.75 deg | 0.012 m | 0.288/0.683 m | 2.22% |
| v36 | 0.434/1.040 m | 12.09/37.75 deg | 0.012 m | 0.288/0.683 m | 2.22% |
| gt_rotation | 0.281/0.815 m | 0.00/0.00 deg | 0.012 m | 0.289/0.698 m | 0.56% |

The synchronized metric bridge preserves the human result while rotation changes: V36 versus V32 has no >0.1 m harmful human or scene correction on the original 180 cuts.

## DA3 Role

Direct DA3 fixed-rotation SE(3) translation reaches `0.569 m` camera error but produces `1.483 m` visible Human3R root jump. `72.6%` of trusted-rotation cases improve camera by >0.1 m while harming root continuity by >0.1 m.

Therefore DA3 is retained only as an independent metric shot-scale cue inside V22/V36. Direct DA3 boundary translation is rejected.

## Rejected Extensions

- Unconditional VGGT rotation: strong tail regressions, especially AvatarReX.
- V25 background fallback: introduced catastrophes on unseen holdout.
- V29/V30 multi-window fallback: no repeatable unseen trigger.
- V31 metric-fit fallback: isolated rescue only, no stable cross-holdout activation.
- V35 scene-metric veto: can reject the only good VGGT rescue when Human3R geometry itself is inconsistent.
- Direct DA3 SE(3) translation: camera improves but Human3R human/scene geometry separates.

## Remaining Limitation

The remaining tail is concentrated in MVHuman large-view changes where torso, VGGT full RGB, background, and multi-frame estimates disagree. No additional fixed rule passed independent validation, so those cases remain unresolved instead of adding an unsafe fallback.
