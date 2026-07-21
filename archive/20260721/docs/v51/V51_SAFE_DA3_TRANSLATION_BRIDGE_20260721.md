# V51 Safe DA3 Translation Bridge

## Correct Placement Of DA3

DA3 is used after V32 rotation and before final translation. It does not scale
the Human3R camera, pointmap, SMPL-X root, or SMPL-X body.

```text
Fixed + torso + conditional VGGT rotation
-> DA3 camera-position prior
-> compare DA3 and Human3R camera positions
-> if disagreement >= 0.5 m, apply at most 0.2 m translation residual
-> one rigid SE(3) for the complete post-cut shot
```

The coordinate conversion preserves the old Human3R camera as the world origin
and transfers only DA3's estimated relative camera displacement. This avoids
the V45/V46 shot-scale and foot-contact failure.

## 180-Cut Result

| Method | T mean/P95 | Viewing error | Human motion error | Scene | Camera catastrophic | Improved/Harmed >5cm |
|---|---:|---:|---:|---:|---:|---:|
| V47 raw scale | 1.568/3.798 m | 1.162 m | 0.012 m | 0.413 m | 27.8% | - |
| V51 bounded DA3 | 1.423/3.600 m | 1.043 m | 0.154 m | 0.435 m | 22.8% | 136/0 |

V51 improves translation in all four sources:

- AvatarReX: `1.161 -> 0.963 m`;
- THuman: `0.405 -> 0.380 m`;
- MVHuman100: `3.117 -> 2.923 m`;
- MVHuman200: `1.596 -> 1.428 m`.

The method preserves zero foot/ground distortion and zero Human3R human
reprojection shift because it applies one rigid translation to the complete
post-cut reconstruction.

## Limitation

The average cross-cut human-motion error increases from `0.012` to `0.154 m`.
This is much safer than the unrestricted DA3 prior (`1.433 m` human error), but
it remains a visible tradeoff. V51 is a promising streaming candidate, not yet
the final selected bridge.

The `0.5 m` disagreement threshold lies in a broad `0.3-1.0 m` zero-harm
plateau on the 180-cut development set. Independent DA3 holdout caches are not
currently available, so threshold generalization still requires a new frozen
holdout run.
