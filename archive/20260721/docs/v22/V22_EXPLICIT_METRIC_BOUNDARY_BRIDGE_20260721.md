# V22 Final Explicit Metric Boundary Bridge

## Selected pipeline

Hard reset -> independent DA3 metric scale -> torso-motion rotation -> diagnostic-safe gravity -> explicit human-root translation -> bounded absolute background scale -> fixed shot-level state.

## Overall 180 cuts

| Method | Camera T mean/P95 | Rotation mean/P95 | Human motion | Scene mean/P95 | Catastrophic | Strict success |
|---|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715 / 4.123 | 24.20 / 73.61 | 0.354 | 0.342 / 0.677 | 42.2% | 0.6% |
| V20 torso/root | 0.493 / 1.263 | 16.04 / 53.56 | 0.012 | 0.311 / 0.721 | 7.8% | 19.4% |
| V22 selected | 0.490 / 1.218 | 15.67 / 52.21 | 0.012 | 0.288 / 0.683 | 7.2% | 20.0% |

## Method

1. DA3 independently estimates the new shot's metric human-root and background scales from the first post-cut frame.
2. Human3R camera translation, SMPL-X root translation and pointmap geometry enter a synchronized metric gauge.
3. V16 torso motion supplies the main rotation residual.
4. Gravity is used only when the plane fit is strong and its residual is large enough to be meaningful. It triggers on `3.9%` of cuts.
5. Camera translation is solved explicitly from the old world human root and the new metric camera-frame root.
6. Background depth receives at most a 15% shot-level correction only when DA3 background scale is lower than the human-root scale by more than 5%.
7. The final transform and scale state are fixed for the whole shot.

## Robustness

- Two point-sampling seeds changed scene mean by only `0.0006 m`.
- Relative to the V20 torso/root candidate, selected gravity produced no correction worse than `+0.1 m` translation or `+5 deg` rotation.
- The absolute background correction produced no scene correction worse than `+0.1 m` relative to the root-scale candidate.
- Four leave-one-source-out threshold checks all selected the same `scene/root < 0.95` rule.
- In 38 A->B->C scale-state chains, propagated camera error was `0.479 m`, scene discontinuity was `0.263 m`, and catastrophic rate stayed at `5.3%`.
- Root and scene scale remained within 20% in `97.4%` of chains.
- 1+1 remains the selected streaming setting; 3+3 did not provide a stable gain.

## Pairwise Upper Bound

The V20 pairwise q30 correction still gives a better scene mean (`0.203 m`) and strict success (`31.1%`), but its scale depends on the particular old/new overlap. It is retained only as an offline visualization or teacher upper bound, not as a persistent streaming shot calibration.

## Remaining Limitation

The main unresolved error is the rotation tail on MVHuman. THuman camera and human alignment improve, but scene continuity is less consistent. The next useful direction is a stronger explicit wide-baseline rotation cue for difficult MVHuman cuts, not another token branch, learned scale regressor, or learned SE(3) selector.

## GT-Rotation Partial Oracle

Keeping the V22 DA3 scales, human motion prediction, and explicit translation equation unchanged, replacing only the rotation with GT gives:

| Method | Camera T mean/P95 | Catastrophic | Scene mean/P95 |
|---|---:|---:|---:|
| V22 selected | 0.490 / 1.218 | 7.2% | 0.288 / 0.683 |
| GT rotation + same metric translation | 0.281 / 0.815 | 0.6% | 0.289 / 0.698 |

The remaining `13` catastrophic cuts decompose into `11` rotation-only, `1` translation-plus-rotation, and `1` scene-only failure. All `12` rotation-related failures are MVHuman cuts and all are rescued by GT rotation. The only non-rotation failure is a THuman pointmap-continuity case, which remains after GT rotation.

Grouped analysis localizes the remaining risk:

- Fixed Explicit rotation error above `60 deg`: V22 catastrophic rate `45.0%`.
- Low-texture tertile: V22 catastrophic rate `13.3%`.
- MVHuman100 / MVHuman200: V22 catastrophic rate `10.4% / 19.4%`.
- AvatarReX: no remaining catastrophic cut.

This establishes explicit wide-baseline rotation as the next targeted research component. The DA3 metric scale and explicit human-root translation path should remain fixed while testing it.
