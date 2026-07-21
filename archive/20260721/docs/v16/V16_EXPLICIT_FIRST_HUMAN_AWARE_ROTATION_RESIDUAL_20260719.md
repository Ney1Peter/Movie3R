# V16 Explicit-First Human-Aware Rotation Residual Probe

## 1. Question

V16 tests whether human information is useful after a physical coarse Boundary candidate already exists:

```text
Explicit coarse SE(3)
-> bounded human-aware rotation residual
-> corrected rotation
-> scene-only translation re-solving
-> one fixed shot-level SE(3)
```

Human3R is frozen and hard-reset after the GT cut index. The experiment uses only pre-cut history and post-cut frames 1-3, never changes recurrent state, never uses GT depth, and never lets human root determine translation.

The same 180 real cross-camera cuts are used:

| Source | Cases |
|---|---:|
| AvatarReX | 48 |
| THuman | 48 |
| MVHuman100 | 48 |
| MVHuman200 | 36 |

The current loader uses `max_humans=1`; V16 cannot support a multi-person conclusion. Human image area is available only as a visibility proxy, not as a true occlusion label.

## 2. Stage 1: Rotation Partial Oracle

Stage 1 starts from the deployed V15 `Fixed Explicit` transform. GT constructs only partial-oracle rotations. Translation is either preserved as `t0` or re-solved by translation-only background pointmap ICP with rotation fixed. GT translation and human root are never used by the solver.

The table reports the two-post-frame rollout metric used by the V10 cache, so its Fixed mean differs slightly from the boundary-frame metric in later sections.

| Method | T mean | R mean | R P90 | Yaw | Pitch | Roll | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.7047 | 23.84 | 58.35 | 8.79 | 17.14 | 11.55 | 66.7% |
| GT Delta Yaw + resolve T | 1.6766 | 12.24 | 33.93 | 6.86 | 2.41 | 8.09 | 63.9% |
| GT Gravity + resolve T | 1.6771 | 17.76 | 41.17 | 6.45 | 16.37 | 3.12 | 66.1% |
| GT Torso Heading + resolve T | 1.6829 | 12.38 | 34.00 | 5.39 | 2.08 | 9.34 | 64.4% |
| GT Torso + Gravity + resolve T | 1.6748 | 0.25 | 0.53 | 0.00 | 0.00 | 0.00 | 63.9% |
| Full GT Rotation + resolve T | 1.6748 | 0.25 | 0.55 | 0.00 | 0.00 | 0.00 | 63.9% |

GT torso heading preserves `11.46` degrees of mean rotation improvement after Fixed Explicit and explains about `48.6%` of the full rotation-oracle gain. Gravity is complementary: heading handles the dominant torso-axis ambiguity, while gravity removes most remaining roll/pitch ambiguity.

This is a real rotation upper bound, but it does not solve world translation. Even full GT rotation plus scene translation re-solving remains at `1.675 m`, confirming that Boundary translation is still a separate bottleneck.

## 3. Translation Re-Solving

Changing rotation while preserving `t0` leaves the old translation coupled to the wrong orientation. Re-solving translation is consistently preferable:

| Rotation | Keep t0 | Resolve t | Change |
|---|---:|---:|---:|
| GT Torso Heading | 1.7069 m | 1.6829 m | -0.0241 m |
| Full GT Rotation | 1.7070 m | 1.6748 m | -0.0322 m |
| Predicted Torso Motion | 1.7151 m | 1.6790 m | -0.0362 m |

The retained cascade must therefore re-solve translation after rotation correction. The solver is scene-only and uses no human root translation.

## 4. Stage 2: Predicted Torso Geometry

Predicted SMPL-X torso frames use pelvis, hips, shoulders, and upper torso joints. The one-frame motion variant estimates angular velocity from the last five pre-cut frames, predicts the torso heading at the first post-cut time, compares it with the fresh post-cut torso after applying the coarse rotation, and applies a bounded `45 deg` heading residual.

| Method | T mean | R mean | R P90 | R P95 | Catastrophic | Harmful R | False correction `<10 deg` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715 | 24.20 | 62.30 | 73.61 | 67.2% | 0.0% | 0.0% |
| Last-frame Torso 1f + resolve T | 1.688 | 17.97 | 39.09 | 53.62 | 66.7% | 23.9% | 13.3% |
| Torso Motion 1f, keep t0 | 1.715 | 16.04 | 39.33 | 53.56 | 66.1% | 20.6% | 12.8% |
| Torso Motion 1f + resolve T | **1.679** | **16.04** | **39.33** | **53.56** | **65.6%** | 20.6% | 12.8% |
| Torso Motion 3f + resolve T | 1.677 | 17.11 | 41.35 | 54.38 | 65.6% | 26.7% | 15.6% |
| Scene Gravity 1f + resolve T | 1.686 | 23.88 | 64.52 | 72.77 | 67.8% | 30.6% | 4.4% |
| Torso Motion + Gravity 1f | 1.691 | 15.28 | 37.33 | **45.28** | 66.1% | 24.4% | 15.0% |
| Torso Motion + Root Check | 1.704 | 23.25 | 59.82 | 73.30 | 67.2% | 14.4% | 10.0% |

The main positive result is candidate quality, not a Selector effect. Always applying the bounded one-frame torso-motion residual directly improves mean, P90, P95, and catastrophic failure.

The root-motion check is too conservative. It reduces harmful corrections but rejects most useful MVHuman corrections and loses almost all mean gain. Human root should remain a diagnostic, not a hard gate.

The lower-image scene-plane gravity estimate improves the strongest rotation tail when combined with torso, but it raises harmful and false-correction rates. It is not reliable enough to be enabled unconditionally.

## 5. Source Generalization

The deployable one-frame torso-motion candidate improves rotation on all four sources:

| Source | Fixed T/R | Torso Motion 1f T/R | Rotation gain |
|---|---:|---:|---:|
| AvatarReX | 1.252 / 6.83 | 1.246 / 4.40 | 2.43 deg |
| THuman | 0.483 / 6.74 | 0.477 / 4.38 | 2.36 deg |
| MVHuman100 | 3.362 / 43.25 | 3.277 / 27.27 | 15.99 deg |
| MVHuman200 | 1.780 / 45.25 | 1.728 / 32.12 | 13.13 deg |

The gain is concentrated where it is needed:

| Fixed initial R | Cases | Fixed R | Torso Motion 1f R |
|---|---:|---:|---:|
| `<10 deg` | 73 | 4.55 | 4.19 |
| `10-30 deg` | 53 | 17.78 | 11.83 |
| `30-60 deg` | 34 | 46.33 | 29.59 |
| `>=60 deg` | 20 | 75.33 | 47.38 |

Predicted angular speed does not invalidate the cue in this dataset:

| Predicted torso speed | Cases | Fixed R | Torso Motion 1f R |
|---|---:|---:|---:|
| Slow | 75 | 13.19 | 10.05 |
| Medium | 30 | 20.15 | 12.93 |
| Fast | 75 | 36.84 | 23.27 |

Fast-turning cases remain harder, but history extrapolation still improves them. A blanket “disable on fast motion” rule is not supported.

## 6. One Frame Versus Three Frames

Three post-cut frames do not improve this rotation residual:

```text
1 frame: 16.04 deg, P90 39.33 deg
3 frames: 17.11 deg, P90 41.35 deg
```

The pre-cut motion model already predicts the first post-cut heading well enough. Extending that angular velocity for three frames adds body-motion extrapolation error, and robust median aggregation does not recover the loss.

The final V16 component should use the zero-wait one-frame setting. Three-frame geometry remains useful for other V15 tasks, but not for this torso residual.

## 7. Human Token LOSO Probe

Only after torso geometry passed, four Leave-One-Source-Out folds trained low-capacity GPU models. PCA, normalization, confidence thresholds, and all weights were fit on three sources only. The held-out source was never used for tuning.

| Method | T mean | R mean | R P90 | R P95 | Catastrophic | Harmful R | False correction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715 | 24.20 | 62.30 | 73.61 | 67.2% | 0.0% | 0.0% |
| Torso Geometry | **1.679** | **16.04** | 39.33 | 53.56 | 65.6% | 20.6% | 12.8% |
| Geometry Confidence Select | 1.692 | 17.17 | 39.96 | 55.01 | 65.6% | 12.8% | 8.9% |
| Token Confidence Select | 1.694 | 18.46 | 45.27 | 59.73 | 66.1% | 11.7% | 7.2% |
| Geometry + Token Confidence | 1.694 | 17.31 | 41.91 | 54.98 | 66.1% | 13.3% | 8.9% |
| Token-only Direct Delta R | 1.733 | **41.18** | 74.59 | 93.93 | **82.2%** | 86.1% | 40.0% |
| Torso + Token Gate | 1.681 | 16.53 | 41.51 | 54.67 | 65.6% | 16.7% | 10.0% |
| Torso + Geometry/Token Gate | 1.678 | 16.37 | **39.31** | **53.54** | **65.0%** | 15.6% | 10.0% |

Token-only direct rotation regression fails decisively and must be stopped.

Token confidence and scale gates reduce some harmful corrections, but they do not produce stable extra accuracy over explicit torso geometry. `Torso + Geometry/Token Gate` improves AvatarReX but is worse on THuman, MVHuman100, and MVHuman200.

Held-out helpfulness AUROC is inconsistent:

| Held-out source | Geometry | Token | Geometry + Token |
|---|---:|---:|---:|
| AvatarReX | 0.534 | 0.614 | 0.606 |
| THuman | 0.471 | 0.436 | 0.336 |
| MVHuman100 | 0.687 | 0.713 | 0.708 |
| MVHuman200 | 0.685 | 0.478 | 0.695 |
| Mean | 0.594 | 0.560 | 0.586 |

The token branch is therefore not retained in the final V16 route. At most, it remains an experimental confidence cue; it is not a source-general rotation estimator.

## 8. Reuse After V15 Coarse Pose

The same unmodified one-frame torso residual is applied after V15 1+1 wide-baseline coarse pose:

| Method | T mean | R mean | R P90 | Catastrophic |
|---|---:|---:|---:|---:|
| V15 Coarse | 2.120 | 37.13 | 121.21 | 66.7% |
| V15 Coarse + Torso Motion + resolve T | 2.069 | 22.72 | 78.15 | 60.6% |

Rotation improves on every source:

| Source | V15 Coarse R | + Torso Motion R |
|---|---:|---:|
| AvatarReX | 73.60 | 47.46 |
| THuman | 15.53 | 8.56 |
| MVHuman100 | 28.51 | 15.11 |
| MVHuman200 | 28.79 | 18.79 |

This proves the torso component is not tied to Fixed Explicit. It is a generic coarse-pose rotation refinement.

V15 translation remains unsafe on AvatarReX and THuman, so this does not make V15 coarse an unconditional final candidate. It only establishes modularity of the rotation residual.

## 9. Final Answers

1. **How much torso information remains after Explicit?** GT torso heading reduces rotation from `23.84` to `12.38` degrees and retains about `48.6%` of the full rotation-oracle gain.

2. **Yaw or roll/pitch?** The physical cue is a torso-up-axis heading correction. It removes the dominant heading ambiguity and also reduces reported pitch because the evaluation gauge is not a canonical gravity frame. Roll remains largely unresolved until gravity is added.

3. **Is predicted Human3R torso geometry stable enough?** Yes as an aggregate candidate: it improves all four sources and the hard initial-error groups. It is not safe on every simple sample, with `20.6%` harmful corrections and `12.8%` false corrections in the `<10 deg` group.

4. **Does motion history beat a single torso frame?** Yes. Last-frame torso gives `17.97` degrees; motion extrapolation gives `16.04` degrees.

5. **Does human token add generalizable information above torso geometry?** No stable extra accuracy. Token-only residual fails, and token confidence/gates trade mean accuracy for modest safety without consistent source-level gains.

6. **Best token role?** If retained at all, confidence/scale only. It must not predict full rotation or translation. Current evidence is not strong enough to include it in the final method.

7. **Is translation re-solving necessary?** Yes. It improves predicted torso translation from `1.715` to `1.679 m` and prevents the corrected rotation from remaining coupled to stale `t0`.

8. **Does the method improve rotation tails without depth GT or full-shot access?** Yes. Predicted one-frame torso motion reduces R-P90 from `62.30` to `39.33` degrees and R-P95 from `73.61` to `53.56` degrees under strict streaming.

9. **Does it work after V15 coarse pose?** Yes. V15 coarse rotation improves from `37.13` to `22.72` degrees and catastrophic failure from `66.7%` to `60.6%`.

## 10. Route Decision

The retained component is:

```text
Any explicit coarse Boundary candidate
-> predicted torso-motion bounded heading residual
-> fixed-rotation scene translation re-solving
-> one fixed shot-level SE(3)
```

The learned human-token residual route is stopped. Do not enlarge the token model and do not let token or human root predict world translation.

The geometry-only component is worth continuing as a modular rotation refinement, particularly for `10-60+ deg` coarse errors and MVHuman. It is not a complete Boundary solution: absolute translation remains around `1.68 m`, and catastrophic failure remains `65.6%` because the coarse translation candidate is still weak.

## 11. Implementation And Outputs

Scripts:

```text
scripts/v16_rotation_residual_partial_oracle.py
scripts/v16_human_torso_candidates.py
scripts/v16_human_torso_eval.py
scripts/v16_loso_human_token_probe.py
```

Outputs:

```text
output/v16_human_aware_rotation_residual/partial_oracle/
output/v16_human_aware_rotation_residual/candidate_cache/
output/v16_human_aware_rotation_residual/evaluation/
output/v16_human_aware_rotation_residual/token_loso/
```

All 180 cases are unique. Candidate generation used four GPU shards with about `5.4 GB` peak memory per shard. Normal no-cut Human3R inference is unchanged.
