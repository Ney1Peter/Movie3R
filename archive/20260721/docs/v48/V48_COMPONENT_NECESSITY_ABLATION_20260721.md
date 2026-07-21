# V48 Component Necessity Ablation

## Question

This experiment tests whether Fixed Explicit, torso motion, VGGT, and DA3 are
redundant. Rotation and scale are evaluated factorially with the same 180 cuts,
the same first post-cut frame, and the same explicit human-root translation
solver.

## Overall Results

| Variant | Camera T mean/P95 | Rotation mean/P95 | Scene | Foot/ground | Human reprojection | Camera catastrophic | Joint failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715/4.123 m | 24.20/73.61 deg | 0.347 m | 0 | 0 px | 39.4% | 39.4% |
| Torso, raw Human3R scale | 1.606/3.960 m | 16.04/53.56 deg | 0.409 m | 0 | 0 px | 31.1% | 31.1% |
| Torso + safe gravity, raw scale | 1.604/3.960 m | 15.67/52.21 deg | 0.408 m | 0 | 0 px | 30.6% | 30.6% |
| Pure VGGT, raw scale | 1.890/4.187 m | 37.13/166.91 deg | 0.344 m | 0 | 0 px | 50.0% | 50.0% |
| Conditional VGGT, raw scale | 1.568/3.798 m | 12.09/37.75 deg | 0.413 m | 0 | 0 px | 27.8% | 27.8% |
| Conditional VGGT + DA3 | 0.438/0.988 m | 12.09/37.75 deg | 0.292 m | 0.515 m | 29.9 px | 1.7% | 86.1% |

`Joint failure` includes camera failure, scene discontinuity, more than `0.1 m`
foot/ground distortion, or more than `25 px` Human3R human reprojection shift.

## Rotation Necessity

- Fixed is a necessary safe base. It is already accurate on AvatarReX and
  THuman, but its rotation mean is `43-45 deg` on the two MVHuman sources.
- Torso motion reduces overall rotation from `24.20` to `16.04 deg` and improves
  `139/180` cuts.
- Safe gravity adds only `0.37 deg` mean rotation gain and is optional.
- Pure VGGT is not a replacement for Fixed: rotation worsens to `37.13 deg`,
  with a `166.91 deg` P95. AvatarReX is especially damaged.
- Conditional VGGT changes only `34/180` cuts. On those cuts, rotation improves
  from `36.04` to `17.09 deg`; `30` improve by more than `5 deg`, while `1` is
  harmed by more than `5 deg`.
- The VGGT gain is concentrated in MVHuman. AvatarReX and THuman are unchanged.

Therefore VGGT is not universally required. It is a cut-only tail-rescue module
after Fixed/torso diagnostics, not the main alignment method.

## Scale Necessity

With the same V32 rotation, enabling DA3 changes:

```text
camera translation: 1.568 -> 0.438 m
scene continuity:    0.413 -> 0.292 m
rotation:            unchanged at 12.09 deg
```

DA3 improves camera translation on `166/180` cuts. The gain is large on
AvatarReX and both MVHuman sources, but small on THuman. This proves the
Human3R-native shot gauge is not reliably metric.

However, the current DA3 integration independently changes camera/root and
scene scales while retaining the original SMPL-X body dimensions. It creates
`0.515 m` average foot/ground distortion and `29.9 px` average human
reprojection shift. The camera-only gain therefore does not make the current
DA3 branch deployable.

## Decision

1. Keep Fixed Explicit as the stable initial coordinate bridge.
2. Keep torso motion as the default rotation correction.
3. Run VGGT only on the fixed, source-independent hard-cut trigger. It is not
   needed on `146/180` cuts.
4. Safe gravity is optional and contributes little.
5. Do not apply the current DA3 shot scaling to final camera, pointmap, and
   SMPL-X output.
6. A metric scale cue is still necessary for accurate camera translation, but
   it must be integrated coherently without changing Human3R reprojection,
   body size, or foot/ground contact.

The current selected visually coherent candidate remains raw-scale V32/V47.
V45 remains a camera-metric upper bound rather than a valid final geometry.
