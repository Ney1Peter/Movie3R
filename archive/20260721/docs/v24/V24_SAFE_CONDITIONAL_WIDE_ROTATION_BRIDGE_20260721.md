# V24 Safe Conditional Wide-Rotation Metric Bridge

## Goal

V22 GT-rotation partial oracle showed that 12 of 13 remaining catastrophic cuts were rotation-related. V24 tests whether the frozen V15 VGGT rotation can repair that tail while preserving the V22 metric scale, explicit human-root translation, scene scale, and one-transform-per-shot streaming contract.

## Final Pipeline

```text
camera cut
-> hard reset Human3R
-> independent DA3 metric human/background scale
-> V16 torso-motion rotation
-> diagnostic-safe gravity residual
-> if torso residual >= 10 deg: frozen VGGT full-RGB 1+1 rotation
-> fixed physical safety rules
-> explicit metric camera translation re-solving
-> one fixed shot-level SE(3) and scale state
```

No runtime GT, learned Gate, learned Selector, token correction, recurrent-state modification, full future shot, BA, or source-specific threshold is used.

## Why Pure VGGT Is Not the Method

Pure VGGT rotation plus the V22 translation equation is not safe:

- rotation mean `37.13 deg`;
- rotation P95 `166.91 deg`;
- combined catastrophic `27.8%`;
- AvatarReX is severely damaged.

The successful component is conditional wide-baseline rotation inside the already stable V22 explicit metric bridge.

## Fixed Safety Rules

1. **Large torso residual**

   - torso residual at least `30 deg`;
   - VGGT extends torso magnitude by at least `5 deg`;
   - VGGT residual at most `100 deg`;
   - VGGT internal spread at most `15 deg`;
   - accepted correction capped at `25 deg`.

2. **Torso/VGGT consensus**

   - torso residual at least `10 deg`;
   - torso and VGGT residual directions agree;
   - VGGT extends torso by at least `5 deg`;
   - VGGT internal spread at most `5 deg`;
   - VGGT residual at most `100 deg`;
   - accepted correction capped at `60 deg`.

3. **Low-texture conflict**

   - image texture score below `0.05`;
   - torso and VGGT residual directions conflict;
   - VGGT extends torso by at least `10 deg`;
   - VGGT internal spread at most `5 deg`;
   - VGGT residual at most `100 deg`;
   - second-stage correction capped at `45 deg`.

Every accepted rotation is followed by the unchanged V22 metric human-root translation equation. Camera, pointmap, and SMPL-X use the same final transform.

## Overall 180 Cuts

| Method | Camera T mean/P90/P95 | Rotation mean/P90/P95 | Scene mean/P95 | Catastrophic | Strict success |
|---|---:|---:|---:|---:|---:|
| V22 | 0.490 / 1.003 / 1.218 m | 15.67 / 38.34 / 52.21 deg | 0.288 / 0.683 m | 7.2% | 20.0% |
| V24 selected | 0.434 / 0.777 / 1.040 m | 12.09 / 30.00 / 37.75 deg | 0.288 / 0.683 m | 2.2% | 21.1% |
| Oracle best V22/VGGT rotation | 0.360 / 0.713 / 0.894 m | 8.49 / 21.58 / 27.38 deg | 0.288 / 0.694 m | 1.1% | 22.2% |
| GT rotation + V22 metric translation | 0.281 / 0.591 / 0.815 m | 0.00 / 0.00 / 0.00 deg | 0.289 / 0.698 m | 0.6% | 36.1% |

## By Source

| Source | Camera T V22 -> V24 | Rotation V22 -> V24 | Rotation P95 V22 -> V24 | Catastrophic V22 -> V24 |
|---|---:|---:|---:|---:|
| AvatarReX | 0.204 -> 0.204 m | 4.40 -> 4.40 deg | 8.92 -> 8.92 deg | 0.0% -> 0.0% |
| THuman | 0.303 -> 0.303 m | 4.38 -> 4.38 deg | 10.31 -> 10.31 deg | 2.1% -> 2.1% |
| MVHuman100 | 0.668 -> 0.570 m | 27.19 -> 20.99 deg | 56.86 -> 42.69 deg | 10.4% -> 2.1% |
| MVHuman200 | 0.882 -> 0.737 m | 30.37 -> 20.73 deg | 79.67 -> 46.11 deg | 19.4% -> 5.6% |

The rule leaves the already strong AvatarReX/THuman camera and rotation results unchanged and concentrates improvement on MVHuman.

## Capture Range

| Fixed rotation error | Count | Camera T V22 -> V24 | Rotation V22 -> V24 | Catastrophic V22 -> V24 |
|---|---:|---:|---:|---:|
| `<10 deg` | 73 | 0.241 -> 0.241 m | 4.19 -> 4.19 deg | 1.4% -> 1.4% |
| `10-30 deg` | 53 | 0.434 -> 0.432 m | 11.28 -> 10.58 deg | 0.0% -> 0.0% |
| `30-60 deg` | 34 | 0.774 -> 0.640 m | 28.49 -> 20.82 deg | 8.8% -> 0.0% |
| `>=60 deg` | 20 | 1.062 -> 0.797 m | 47.38 -> 30.05 deg | 45.0% -> 15.0% |

This is direct evidence that V24 expands the capture range instead of only refining already-correct cuts.

## Texture

- Low-texture group: camera `0.760 -> 0.641 m`, rotation `28.56 -> 20.88 deg`, catastrophic `14.3% -> 3.6%`.
- Higher-texture group: camera `0.254 -> 0.254 m`, rotation `4.39 -> 4.39 deg`, catastrophic `1.0% -> 1.0%`.

The low-texture conflict rule uses a broad stable threshold interval (`0.025-0.05` produced the same trigger set), not a source-specific cutoff.

## Safety

- Corrected cuts: `34/180`.
- Rescued catastrophic cuts: `9`.
- Introduced catastrophic cuts: `0`.
- Rotation harmful over V22 by more than `5 deg`: `0.6%`.
- Camera harmful over V22 by more than `0.1 m`: `1.1%`.
- Scene harmful over V22 by more than `0.1 m`: `0%`.
- Among 101 V22 cases below `10 deg`, only one is modified and none worsens by more than `5 deg`.

## Threshold Sensitivity

One-factor sweeps keep all other selected parameters fixed:

- extension margin `2.5-10 deg`: rotation catastrophic `1.7%`, harmful `0.6%`;
- maximum VGGT residual `90-110 deg`: rotation catastrophic `1.7%`, harmful `0.6%`;
- large-residual spread `15-20 deg`: rotation catastrophic `1.7%`, harmful `0.6%`;
- texture threshold `0.025-0.05`: identical trigger set and metrics;
- large-residual cap `25-30 deg`: rotation catastrophic `1.7%`, harmful `0.6%`;
- low-texture conflict cap `45-60 deg`: rotation catastrophic `1.7%`, harmful `0.6%`.

No neighboring setting harms any V22 case whose rotation error is below `10 deg`. The selected rule lies inside a broad plateau rather than at a single tuned threshold.

## Runtime

- V22 base cut latency on L20: `0.308 s` mean.
- VGGT 1+1 latency: `1.368 s` mean, `2.498 s` P95.
- VGGT pretrigger rate: `48.9%`.
- VGGT accepted correction rate: `18.9%`.
- Amortized cut latency: `1.014 s` mean, `2.672 s` P95.
- Mean latency on a pretriggered cut: `1.751 s`.
- Ordinary frame path is unchanged.
- V15 measured peak GPU memory at about `12.0 GB`; V22 DA3/Keypoint incremental peak is about `2.3 GB`. A final integrated memory benchmark is still needed before lightweight deployment.

## Remaining Failures

Four catastrophic cuts remain:

1. MVHuman100 with VGGT internal spread about `31.9 deg`; V24 rejects it.
2. MVHuman200 with VGGT internal spread about `23.3 deg`; V24 rejects it.
3. MVHuman200 with torso residual below `10 deg`; VGGT is not run.
4. One THuman scene-only pointmap discontinuity; rotation is already accurate.

The first three are rotation-observability/confidence failures. The THuman case is a separate scene pointmap problem.

## Decision

V24 is the strongest current training-free Boundary candidate. It demonstrates that a wide-baseline model can be useful after the scale and translation responsibilities are removed from it: VGGT supplies only a conditionally accepted coarse rotation, while V22 retains metric scale and translation.

The large VGGT cost means the preferred next step is not more heuristic expansion. Use V24 as a teacher to distill a lightweight cut-only Rotation Bridge that predicts a bounded rotation cue and confidence from the same first-frame streaming inputs.
