# V14 Depth-Free Selective World-Memory Relocalization

## 1. Question

V14 tests whether the limited V13 World-Memory success can be used selectively without GT depth, GT scene coordinates, or GT correspondence:

```text
Accept reliable World-Memory refinement
Wait for three frames when one frame is uncertain
Fallback to Fixed Explicit when geometry is unobservable
```

The primary criterion is cross-source false-accept control, not training-domain average error.

## 2. Strictly Causal Candidate

The final candidate protocol uses:

- frozen Human3R;
- GT cut index only;
- eight already-observed pre-cut frames;
- 128 sampled patches per historical frame;
- 0.20 m temporal voxel aggregation in the predicted Human3R world gauge;
- a fixed 256-anchor memory;
- DINO/Multi-HMR descriptors;
- six anchor strategies;
- one-frame and fixed three-frame queries;
- raw descriptor RANSAC as a negative control;
- Fixed-Explicit-initialized World-Memory ICP as the main candidate.

The candidate code directly reads the deployable Fixed Explicit transform in the stored Human3R gauge. GT camera is used only after candidate generation to create labels and metrics. GT depth and GT correspondence are never used.

Each anchor stores predicted world XYZ, confidence, observation count, XYZ temporal variance, descriptor variance, normal stability, static-mask frequency, edge score, and source-frame presence.

All 180 candidate caches ran on four GPUs. Wall time was 13.5 minutes, or 17.95 seconds per case for the full 24-candidate diagnostic grid. Human3R capture used about 9.6 seconds per case; the remaining time evaluates all anchor, frame-count, and matcher variants. Peak process memory observed during generation was approximately 6.3 GB per GPU.

## 3. Data And LOSO

The same 180 cuts are used:

| Held-out source | Cases | Training sources |
|---|---:|---|
| AvatarReX | 48 | THuman + MVHuman100 + MVHuman200 |
| THuman | 48 | AvatarReX + MVHuman100 + MVHuman200 |
| MVHuman100 | 48 | AvatarReX + THuman + MVHuman200 |
| MVHuman200 | 36 | AvatarReX + THuman + MVHuman100 |

For every fold, descriptor/anchor configuration, model weights, and Accept/Wait/Fallback thresholds are selected only on the three training sources.

## 4. Candidate Complementarity

`Oracle Select` chooses safely between Fixed, one-frame World-Memory, and three-frame World-Memory for the LOSO-selected configuration. `Oracle All` is a diagnostic upper bound over Fixed and all 24 depth-free candidates.

| Method | T mean | T median | T P90 | T P95 | R mean | R median | R P90 | R P95 | Catastrophic | Success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.7151 | 1.4223 | 3.7199 | 4.1229 | 24.201 | 13.340 | 62.300 | 73.610 | 67.2% | 2.8% |
| World-Memory 1-frame always | 1.7734 | 1.4449 | 3.7571 | 4.1070 | 27.945 | 17.956 | 66.962 | 85.826 | 72.8% | 2.2% |
| World-Memory 3-frame always | 1.8115 | 1.5197 | 3.7750 | 4.2230 | 28.525 | 18.758 | 66.129 | 87.259 | 72.2% | 1.1% |
| Oracle Select | 1.6621 | 1.3773 | 3.7199 | 4.1203 | 22.603 | 11.970 | 59.902 | 73.108 | 66.1% | 4.4% |
| Oracle All | **1.5026** | **1.2699** | **3.3657** | **3.9240** | **21.493** | **11.065** | 61.313 | **72.947** | **61.7%** | **10.0%** |
| Boundary Oracle | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.021 | 0.000 | 0.069 | 0.080 | 0.0% | 100.0% |

Always using World-Memory is worse than Fixed Explicit. However, Oracle All lowers translation by 0.212 m, rotation by 2.71 degrees, and catastrophic failure by 5.5 percentage points. The candidate pool therefore has real but limited complementarity.

## 5. Source Results

| Source | Fixed T/R | WM3 always T/R | Oracle Select T/R | Oracle All T/R | Fixed -> Oracle All catastrophic |
|---|---:|---:|---:|---:|---:|
| AvatarReX | 1.252 / 6.83 | 1.396 / 10.71 | 1.243 / 6.15 | 1.203 / 6.09 | 77.1% -> 64.6% |
| THuman | 0.483 / 6.74 | 0.568 / 8.23 | 0.401 / 5.06 | **0.333 / 4.01** | 2.1% -> 0.0% |
| MVHuman100 | 3.362 / 43.26 | 3.361 / 48.42 | 3.312 / 41.27 | 2.929 / 41.08 | 100.0% -> 100.0% |
| MVHuman200 | 1.780 / 45.23 | 2.002 / 53.38 | 1.702 / 43.04 | 1.561 / 39.23 | 97.2% -> 88.9% |

THuman has the strongest usable upper bound. AvatarReX and MVHuman200 have some complementary candidates, but MVHuman candidates remain poor in absolute terms.

## 6. Observable Subsets

V13 offline-teacher pseudo overlap is used only as an evaluation grouping label.

| Group | Fixed T/R/Cat | WM3 T/R/Cat | Oracle All T/R/Cat |
|---|---:|---:|---:|
| High overlap | 0.65 / 11.5 / 17% | 0.76 / 13.1 / 27% | **0.47 / 7.3 / 10%** |
| Low overlap | 2.92 / 39.2 / 100% | 3.00 / 45.4 / 100% | 2.72 / 36.4 / 100% |
| High texture | 0.99 / 7.0 / 52% | 1.08 / 10.0 / 55% | **0.88 / 5.1 / 42%** |
| Low texture | 2.87 / 40.8 / 98% | 2.94 / 48.3 / 100% | 2.50 / 38.9 / 98% |
| Non-degenerate proxy | 1.76 / 19.0 / 79% | 1.88 / 20.6 / 82% | **1.60 / 16.2 / 67%** |
| Planar proxy | 1.70 / 24.0 / 52% | 1.86 / 30.1 / 65% | 1.49 / 21.2 / 50% |

The V14 hypothesis is correct only as an Oracle statement: World-Memory can help in high-overlap, high-texture, and some non-degenerate cases, but it is unsafe when used unconditionally.

## 7. Anchor Selection

All rows below use the same explicit-initialized ICP matcher; only anchor selection changes.

| Anchor strategy | T mean | R mean | Catastrophic | Helpful versus Fixed |
|---|---:|---:|---:|---:|
| Confidence only | 1.778 | 27.92 | 69.4% | 11.7% |
| Spatial only | 1.819 | 29.09 | 70.0% | 8.3% |
| Temporal only | 1.789 | 27.60 | 70.0% | 12.2% |
| Confidence + spatial | 1.790 | 27.81 | **68.9%** | **13.9%** |
| Temporal + spatial | 1.803 | 28.39 | 71.7% | 10.0% |
| Temporal + spatial + static | 1.782 | **27.01** | 70.6% | 10.0% |

Temporal stability is more useful than spatial-only selection, but it does not consistently beat confidence plus spatial coverage. Static masking improves rotation in some cases but does not reduce the overall tail. Temporal consistency is informative, not sufficient.

## 8. Reliability Results

| Reliability | T mean | R mean | Catastrophic | Coverage | False accept |
|---|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.7151 | 24.201 | 67.2% | 0% | 0% |
| Hand geometry rules | 1.7137 | 24.250 | 67.2% | 0.6% | 0.6% |
| Geometry-only | 1.7289 | 24.747 | 67.8% | 20.6% | 18.3% |
| Token-only | 1.7152 | 24.250 | 66.7% | 1.7% | 1.1% |
| Geometry + token | 1.7459 | 25.100 | 68.3% | 26.1% | 24.4% |
| Geometry + gravity | 1.7367 | 24.870 | 67.8% | 18.3% | 17.2% |
| Geometry + human | 1.7418 | 24.909 | 67.8% | 26.1% | 25.6% |
| Geometry + human + gravity | 1.7477 | 25.244 | 68.3% | 26.1% | 25.6% |
| Per-fold complete strategy | 1.7442 | 24.941 | 68.3% | 28.3% | 26.1% |

The complete strategy is worse than Fixed Explicit. It produces 33 immediate accepts and 18 three-frame accepts; 29 of the immediate accepts and all 18 three-frame accepts are false accepts.

The conservative threshold search allows zero acceptance and enforces near-zero training false-accept precision error. Even so, held-out calibration collapses:

- AvatarReX geometry accepted precision: 4.3%;
- THuman geometry accepted precision: 27.3%;
- MVHuman100 accepted precision: 0%;
- MVHuman200 accepted precision: 0%.

In contrast, the corresponding training accepted subsets were calibrated to 90-100% precision. The failure is cross-source distribution shift, not an insufficiently conservative test-time threshold.

## 9. Risk-Coverage

The held-out risk-coverage curves fail the required behavior:

- AvatarReX geometry top-10% has 80% false accept and is worse than Fixed by 1.58 normalized joint-cost units;
- THuman geometry top-10% has positive mean improvement, but still 60% false accept;
- MVHuman100 and MVHuman200 have 100% catastrophic and 100% false accept throughout the nominal high-confidence subsets;
- risk is not consistently monotonic with coverage.

Geometry can rank a small number of positives in AvatarReX (`AUROC 0.979`) and THuman (`0.685`), but the score scale and threshold do not transfer. MVHuman held-out folds contain no non-catastrophic helpful examples for the selected candidate, so reliability AUROC is undefined there.

The most influential geometry inputs are fit residual, target/source condition number, 10 cm inlier ratio, ICP correction magnitude, anchor XYZ variance, Top-K consistency, and one-frame/three-frame consistency. Their coefficient magnitude is high, but several coefficient directions change across folds, confirming source-dependent calibration.

## 10. One Frame And Three Frames

Three frames reduce catastrophic failure slightly relative to one-frame always-use (`72.2%` versus `72.8%`), but worsen mean translation and rotation. A three-frame candidate beats its one-frame counterpart in:

- 60.4% of AvatarReX;
- 45.8% of THuman;
- 39.6% of MVHuman100;
- 36.1% of MVHuman200.

Thus waiting has per-sample value, but the learned Wait decision does not generalize. In the complete strategy, every accepted three-frame wait is a false accept. V14 does not justify a deployable Wait branch.

## 11. Token, Human, And Gravity

Token-only mostly learns to fallback and gives no repeatable gain. Geometry plus token is worse than geometry-only in mean error, catastrophic failure, false accept, and cross-source AUROC. Token should not be retained.

Torso, human-root motion, dominant-normal/gravity conflict, and their combination do not lower false accepts. They are source-dependent checks and cannot repair a bad scene candidate.

The current 180-cut loader uses `max_humans=1`, so a multi-person relative-layout result is unavailable and is not inferred from single-person data.

## 12. Answers

1. **Are World-Memory and Fixed Explicit complementary?** Yes at the Oracle level, especially on THuman and high-overlap/high-texture subsets. The gain is limited and candidate-dependent.
2. **Can temporal stability select better anchors without GT depth?** It improves over spatial-only anchors and carries useful information, but confidence plus spatial coverage is at least as competitive. It is not a reliable selector by itself.
3. **Which diagnostics predict failure?** Residual/inlier statistics, geometric condition, ICP correction magnitude, anchor XYZ variance, and cross-keyframe/frame/strategy consistency. Their calibration and sometimes direction change across sources.
4. **Does Geometry Reliability generalize?** No. Training precision is high, but held-out false-accept precision collapses and risk-coverage is not reliable.
5. **Does token provide stable additional information?** No. Geometry plus token is worse than geometry-only, and token-only is effectively a fallback control.
6. **How should one-frame and three-frame be allocated?** The Oracle uses both, but the current Wait predictor cannot identify the useful three-frame cases. No deployable allocation is supported.
7. **Does a conservative selective strategy work?** No. Nontrivial coverage causes unacceptable false accepts; near-zero coverage degenerates to Fixed Explicit without repeatable improvement.

## 13. Decision

V14 triggers the stop conditions:

- high-confidence held-out subsets still fail;
- Geometry Reliability is source-specific;
- risk-coverage is not reliably monotonic;
- false accepts remain high;
- the complete strategy is worse than Fixed Explicit;
- three-frame waiting does not provide stable deployable gains.

The current depth-free Selective World-Memory mainline should stop. The safe deployed method remains:

```text
hard reset
+ Fixed Explicit shot-level SE(3)
+ no learned World-Memory acceptance
```

World-Memory candidates should remain diagnostic only. Resuming this direction requires improving the candidate itself before training another Gate, for example with better multi-frame local depth/geometry, externally supervised overlap/correspondence, or a physically grounded scene-coordinate memory. Increasing reliability-model capacity is not justified by these results.

## 14. Code And Outputs

Code:

```text
scripts/v14_depth_free_world_memory_candidates.py
scripts/v14_selective_world_memory_eval.py
scripts/v14_group_analysis.py
```

Outputs:

```text
output/v14_selective_world_memory/candidate_cache/v14_candidates_shard_*.json
output/v14_selective_world_memory/evaluation/v14_eval.json
output/v14_selective_world_memory/evaluation/v14_summary.md
output/v14_selective_world_memory/evaluation/v14_groups.json
```
