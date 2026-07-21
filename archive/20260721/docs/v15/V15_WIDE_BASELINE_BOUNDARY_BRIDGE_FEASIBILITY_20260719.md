# V15 Wide-Baseline Boundary Bridge Feasibility Probe

## 1. Question

V15 tests whether a frozen model trained for multi-view geometry can provide the capture range missing from V14:

```text
wide-baseline visual coarse localization
+ Human3R metric pointmap metrification/refinement
-> one fixed shot-level SE(3)
```

This is a training-free feasibility probe. It does not train a Gate, Selector, Human3R module, or final Shot Bridge.

## 2. Frozen Model And Streaming Protocol

The wide-baseline model is frozen `VGGT-1B`, using its camera and track heads. The local model is the frozen Human3R checkpoint used by V14.

The same 180 real cross-camera cuts are evaluated:

| Source | Cases |
|---|---:|
| AvatarReX | 48 |
| THuman | 48 |
| MVHuman100 | 48 |
| MVHuman200 | 36 |

The protocol uses GT cut index only. It does not use GT depth, scene mesh, or correspondence. GT camera enters only evaluation and epipolar diagnostics.

Two bounded-streaming windows are tested:

- `1+1`: last pre-cut frame and first post-cut frame;
- `3+3`: last three pre-cut and first three post-cut frames.

The `3+3` candidate runs all nine pre/post frame pairs in both directions, maps every pair to a common Human3R Boundary transform, and uses robust rotation consensus. It is not a six-image average and does not access frames after post-cut offset 2.

Every final candidate is one shot-level transform shared by camera, pointmap, and SMPL-X. Human3R recurrent state is hard-reset and never modified.

## 3. Candidate Definitions

The experiment separates the information sources:

1. `Wide Coarse`: VGGT relative camera pose only;
2. `Wide Rotation + Fixed Translation`: VGGT rotation with Fixed Explicit transform translation;
3. `Correspondence Metric`: VGGT tracks, Human3R pointmaps, robust 3D fit, and VGGT rotation;
4. `Hybrid`: Correspondence Metric followed by bounded Human3R point-cloud ICP;
5. `Background`: human pixels masked before VGGT and excluded from metric fitting;
6. `Human Down-weighted`: full RGB for matching, but human matches receive 0.1 geometry weight.

`V14 World-Memory ICP` is retained as the local-refinement negative control.

## 4. Overall Boundary Results

| Method | T mean | T P90 | R mean | R P90 | Catastrophic | Success |
|---|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | **1.715** | **3.718** | **24.20** | **62.30** | 67.2% | 2.8% |
| V14 World-Memory ICP | 1.790 | 3.794 | 27.81 | 69.79 | 68.9% | 3.3% |
| Wide 1+1 Coarse | 2.120 | 4.129 | 37.13 | 121.21 | 66.7% | 0.0% |
| Wide 3+3 Coarse | 2.211 | 4.135 | 46.74 | 115.44 | 82.8% | 0.0% |
| Wide 1+1 Rotation + Fixed T | 1.715 | 3.719 | 37.13 | 121.21 | 73.3% | 3.9% |
| Wide 1+1 Correspondence Metric | 2.207 | 4.505 | 37.13 | 121.21 | 66.1% | **8.3%** |
| Wide 3+3 Correspondence Metric | 2.354 | 4.795 | 46.74 | 115.44 | 70.0% | 7.8% |
| Wide 1+1 Hybrid | 2.175 | 4.505 | 37.44 | 117.39 | **65.6%** | 5.6% |
| Wide 3+3 Hybrid | 2.297 | 4.795 | 46.80 | 108.59 | 68.9% | 6.1% |
| Background 1+1 Metric | 2.257 | 4.448 | 50.76 | 161.42 | 66.7% | 8.9% |
| Boundary Oracle | 0.000 | 0.000 | 0.02 | 0.07 | 0.0% | 100.0% |

No non-Oracle wide-baseline candidate directly dominates Fixed Explicit. The smallest catastrophic rate is `65.6%`, but its mean and tail errors are worse. This fails the main deployable-Hybrid criterion.

## 5. Source Split

The important result is strongly source-dependent.

| Source | Fixed T/R/Cat | Wide 1+1 Coarse T/R/Cat | Wide 1+1 Metric T/R/Cat | Wide 1+1 Hybrid T/R/Cat |
|---|---:|---:|---:|---:|
| AvatarReX | 1.252 / 6.83 / 77.1% | 2.510 / 73.60 / 100.0% | 2.259 / 73.60 / 77.1% | 2.137 / 73.83 / 75.0% |
| THuman | 0.483 / 6.74 / 2.1% | 4.053 / 15.53 / 100.0% | 0.907 / 15.53 / 20.8% | 0.906 / 16.72 / 20.8% |
| MVHuman100 | 3.362 / 43.25 / 100.0% | **0.680 / 28.51 / 31.2%** | 3.232 / 28.51 / 85.4% | 3.218 / 28.32 / 85.4% |
| MVHuman200 | 1.780 / 45.25 / 97.2% | **0.945 / 28.79 / 25.0%** | 2.505 / 28.79 / 86.1% | 2.527 / 28.70 / 86.1% |

VGGT Coarse is a materially new candidate on both MVHuman sources. It is not a safe replacement on AvatarReX or THuman.

The decisive negative result is that Human3R metrification erases most MVHuman gains. On MVHuman100, catastrophic failure increases from `31.2%` for Coarse to `85.4%` for Metric/Hybrid. On MVHuman200 it increases from `25.0%` to `86.1%`.

## 6. Capture Basin

Grouping by the error of the Fixed Explicit initialization directly tests capture range.

| Fixed initial R error | Cases | Fixed T/R/Cat | Wide 1+1 Coarse T/R/Cat | Wide 1+1 Metric T/R/Cat |
|---|---:|---:|---:|---:|
| `<10 deg` | 73 | 0.93 / 4.6 / 43.8% | 3.32 / 48.8 / 97.3% | 1.82 / 48.8 / 54.8% |
| `10-30 deg` | 53 | 1.81 / 17.8 / 66.0% | 1.80 / 29.7 / 62.3% | 2.24 / 29.7 / 66.0% |
| `30-60 deg` | 34 | 2.78 / 46.3 / 100.0% | **0.88 / 32.8 / 29.4%** | 2.79 / 32.8 / 85.3% |
| `>=60 deg` | 20 | 2.53 / 75.3 / 100.0% | **0.72 / 21.8 / 30.0%** | 2.55 / 21.8 / 75.0% |

This is the central positive V15 finding. VGGT breaks the local ICP basin on genuinely bad initializations. It is harmful when Fixed Explicit is already close.

The view-angle grouping is different from the Fixed-error grouping:

- `60-120 deg`: Coarse rotation improves `28.4 -> 15.5 deg` and catastrophic `65% -> 59%`;
- `>120 deg`: Coarse rotation degrades `19.4 -> 61.9 deg` and catastrophic `70% -> 75%`.

AvatarReX 150-180 degree pairs frequently produce a consistent but wrong 170-180 degree visual solution. Forward/reverse agreement alone therefore cannot certify correctness.

## 7. Rotation, Direction, Scale, And Metrification

VGGT does not provide one source-invariant metric pose.

| Source | Coarse direction error | Coarse scale log error | Interpretation |
|---|---:|---:|---|
| AvatarReX | 37.2 deg | 2.015 | rotation/direction ambiguity and severe scale mismatch |
| THuman | 7.65 deg | 1.509 | direction is useful, scale is not metric |
| MVHuman100 | 14.54 deg | 0.375 | rough pose scale is much closer |
| MVHuman200 | 14.46 deg | 0.621 | rough pose scale is partially useful |

The model mainly contributes large-range rotation and translation direction. Its raw translation scale is domain-dependent.

Human3R correspondence metrification is not reliable enough to replace that scale:

- 1+1 yields about `405` correspondences, but only `2.2%` mutual track ratio;
- 3+3 yields about `3651` correspondences, but only `2.6%` mutual ratio;
- GT epipolar median error is `20.3 px` for 1+1 and `23.6 px` for 3+3;
- Human3R 3D-fit median residual is about `0.49 m`;
- bounded ICP changes the transform by about `0.19 m / 1.91 deg` for 1+1 and `0.24 m / 2.43 deg` for 3+3, but does not recover the lost MVHuman performance.

Thus both track quality and Human3R local pointmap consistency limit metrification. The pointmap remains useful for local geometry output, but not as a robust cross-camera metric solver for these hard cuts.

## 8. One Frame Versus Three Frames

`1+1` is consistently better than the current `3+3` construction:

- Coarse catastrophic: `66.7%` versus `82.8%`;
- Metric catastrophic: `66.1%` versus `70.0%`;
- Hybrid catastrophic: `65.6%` versus `68.9%`;
- 3+3 increases correspondences by about 9x, but also increases epipolar error and rotation-consensus failures.

The additional two-frame delay is not justified. More pairwise evidence is not helpful when several pairs are confidently wrong.

## 9. Human Region Policy

Masking people before VGGT is worse than retaining full RGB:

- Full-RGB 1+1 Metric: `2.207 m / 37.13 deg`;
- Background-only 1+1 Metric: `2.257 m / 50.76 deg`.

People provide useful visual signal when the background is weak. The safer current policy is:

```text
retain human pixels for visual coarse matching
+ down-weight human pixels in the metric 3D solve
```

Human motion is useful only as a check. Torso-jump failure AUROC is `0.595` for raw Coarse, `0.822` for Correspondence Metric, and `0.876` for Background Metric. Human root jump is not informative for raw Coarse (`0.452` AUROC), but becomes informative after metric fitting (`0.855`). These checks can reject some bad metric candidates; they do not repair coarse pose or translation scale.

## 10. Candidate Complementarity

Candidate complementarity is much stronger than V14:

| Oracle candidate set | T mean | R mean | Catastrophic | Success |
|---|---:|---:|---:|---:|
| Fixed only | 1.715 | 24.20 | 67.2% | 2.8% |
| Best of Fixed / Wide 1+1 Coarse | **0.864** | **13.00** | **34.4%** | 2.8% |
| Best of all Fixed / Wide / Metric / Hybrid candidates | **0.752** | **9.72** | **29.4%** | 16.1% |

For Fixed versus Wide 1+1 Coarse, the Oracle chooses Fixed on 101 cases and Wide on 79. At the source level it chooses Wide on 44/48 MVHuman100 and 32/36 MVHuman200 cases, but only 3/48 AvatarReX and 0/48 THuman cases.

This proves that the new information source is complementary. It does not provide a deployable selector. V14 already showed that source-dependent Gate calibration is unsafe, so this Oracle result cannot be treated as a final method.

## 11. Streaming Cost

All inference ran on NVIDIA L20 GPUs.

| Component | Mean cut-time cost |
|---|---:|
| Human3R diagnostic capture | 3.76 s/case |
| VGGT full-RGB 1+1 | 1.37 s/cut |
| VGGT full-RGB 3+3 | 8.83 s/cut |
| Peak GPU memory | 11.98 GB/process |

The wide model runs only once per cut. Normal-frame FPS is unchanged. The experiment evaluates full and masked variants separately; a deployed single variant would not pay both costs.

## 12. Answers

1. **Does a frozen wide-baseline model expand capture range?** Yes, decisively for bad Fixed initializations and MVHuman. It is not universally correct.
2. **What information does it provide?** Primarily coarse rotation and translation direction. Full pose scale is source-dependent.
3. **Is Human3R pointmap suitable for metric refinement near the coarse solution?** Not reliably on hard MVHuman cuts. Metrification and ICP erase most Coarse gains.
4. **1+1 or 3+3?** Use 1+1 for this model and pairwise construction. The fixed two-frame delay is worse.
5. **How should human regions be handled?** Keep them for visual matching, then down-weight them in 3D geometry. Full exclusion is worse.
6. **Does one candidate directly beat Fixed across sources?** No. Wide Coarse wins on both MVHuman sources and loses badly on AvatarReX/THuman.
7. **What causes MVHuman failures?** Both causes exist. Remaining Coarse catastrophes are visual localization failures; when Coarse rotation succeeds, the dominant new failure is metric translation/scale and Human3R pointmap inconsistency.
8. **Should VGGT be distilled into a Shot Bridge?** Not as a full pose or bounded-SE(3) teacher yet. Its rotation/direction output is worth retaining as a teacher signal for hard cuts, but translation metrification and safe source-independent use must be solved first.

## 13. Decision

V15 does not justify the proposed complete pipeline:

```text
VGGT coarse pose
+ Human3R metric solver
+ Human3R residual ICP
```

The Hybrid is not directly better than Fixed Explicit, 3+3 is worse than 1+1, and gains do not hold on three of four sources. Therefore do not start full Shot Bridge distillation and do not distill metric translation or residual SE(3).

The positive result should be retained in a narrower form:

```text
frozen wide-baseline rotation/direction teacher
+ separate metrification research
+ no Human3R ICP assumption
```

The next feasibility target should fix wide-baseline rotation and study translation only: source-invariant scale, camera-center geometry, alternative metric depth/pointmap teachers, or multi-frame translation triangulation. A later lightweight Shot Bridge may distill coarse rotation/direction after that metric bridge has a stable, non-Oracle solution.

## 14. Code And Outputs

Code:

```text
scripts/v15_wide_baseline_boundary_bridge_candidates.py
scripts/v15_wide_baseline_boundary_bridge_eval.py
```

Outputs:

```text
output/v15_wide_baseline_boundary_bridge/candidate_cache/v15_candidates_shard_*_of_*.json
output/v15_wide_baseline_boundary_bridge/evaluation/v15_eval.json
output/v15_wide_baseline_boundary_bridge/evaluation/v15_summary.md
```
