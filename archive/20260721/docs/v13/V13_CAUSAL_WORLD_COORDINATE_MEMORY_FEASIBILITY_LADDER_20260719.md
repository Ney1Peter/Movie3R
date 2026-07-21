# V13 Causal World-Coordinate Memory Feasibility Ladder

## 1. Question

V12 showed that raw Human3R persistent state is not a reliable addressable world map. V13 asks whether a separate causal memory of descriptors bound to explicit world XYZ can provide a geometric basis for cross-camera shot relocalization.

The hard first gate is intentionally independent of matcher learning:

```text
fresh reset Human3R pointmap
+ correct scene correspondence / historical world XYZ
-> robust shot-level SE(3) or Sim(3)
```

If this upper bound is not sufficiently accurate, descriptor, memory-policy, human/gravity, and reliability stages must stop.

## 2. Data Audit

The current AvatarReX, THuman, MVHuman, RICH, and BEDLAM copies were audited for calibrated static-scene depth or scene scans. No verified static-scene GT depth or mesh is available for the 180-cut protocol.

Results are therefore strictly separated into:

1. **True Scene-Coordinate Oracle:** unavailable in the current data.
2. **Same-view Teacher Pseudo Oracle:** a warmed B-camera Human3R pointmap, anchored with GT cameras, supplies same-pixel target coordinates.
3. **History-Covered Ideal XYZ:** only regions covered by the causal pre-cut memory are retained, but target XYZ still comes from the offline teacher.
4. **Actual Historical Anchors:** target XYZ is the point stored from the causal pre-cut Human3R pointmap.

The pseudo Oracle is a useful model-consistency upper bound, but it is not equivalent to GT scene coordinates.

Audit output:

```text
output/v13_world_coordinate_memory/data_audit.json
```

## 3. Protocol

The experiment uses the same 180 real cross-camera cuts:

| Source | Cases |
|---|---:|
| AvatarReX | 48 |
| THuman | 48 |
| MVHuman100 | 48 |
| MVHuman200 | 36 |

For every case:

- Human3R is frozen and hard-reset after the GT cut;
- all Human3R branches run on GPU;
- the causal memory contains eight pre-cut frames and exactly 16,384 anchors;
- only pre-cut observations enter the memory;
- one-frame and fixed three-frame relocalization are compared;
- 64, 256, 1,024, and 4,096 points per query frame are tested;
- confidence-only and spatial-coverage sampling are compared;
- all valid points and static/background-only points are compared;
- robust weighted SE(3) and Sim(3) are fitted;
- fit residual, inlier ratio, coverage, condition number, planarity, and duplicate-anchor diagnostics are recorded.

Four GPU shards completed the 180 cases in about 40.8 minutes wall time. Each case evaluated the full Stage-1 grid rather than one canonical fit.

## 4. Main Results

The canonical methods use static/background points, spatial sampling, and 1,024 points per query frame.

| Method | T mean | T median | T P90 | R mean | R median | R P90 | Fit fail | Success | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard Reset | 3.1622 m | 2.6705 | 5.5738 | 120.529 deg | 116.437 | 172.469 | 0.0% | 0.0% | 100.0% |
| Fixed Explicit | 1.7050 m | 1.4385 | 3.8447 | 23.760 deg | 13.465 | 58.267 | 0.0% | 3.3% | 66.7% |
| Same-view Pseudo Oracle SE(3), 1 frame | 0.2500 m | 0.1382 | 0.5871 | 3.553 deg | 0.991 | 8.431 | 0.0% | 67.8% | 2.8% |
| Same-view Pseudo Oracle SE(3), 3 frames | 0.2087 m | 0.0918 | 0.5571 | 2.898 deg | 0.746 | 7.740 | 0.0% | 70.0% | 1.1% |
| History-Covered Ideal XYZ SE(3), 3 frames | 0.2639 m | 0.0434 | 0.7556 | 5.516 deg | 0.430 | 16.994 | 18.3% | 57.2% | 23.3% |
| Actual Historical Anchors SE(3), 3 frames | 0.5455 m | 0.1687 | 1.5213 | 13.636 deg | 3.536 | 38.108 | 18.3% | 42.2% | 35.6% |
| Boundary Oracle | 0.0000 m | 0.0000 | 0.0000 | 0.020 deg | 0.000 | 0.069 | 0.0% | 100.0% | 0.0% |

Correct pseudo correspondence removes most of the Fixed Explicit error, but the requested 1-2 degree geometric upper bound is not reached. The median is good while the mean and P90 remain high, so the failure is concentrated in difficult data rather than universal.

## 5. Source Diagnosis

| Source | Same-view 3-frame T / R | History-Covered 3-frame T / R | Actual Anchors 3-frame T / R | Actual fit fail | Actual catastrophic |
|---|---:|---:|---:|---:|---:|
| AvatarReX | 0.0493 m / 0.494 deg | 0.0313 m / 0.381 deg | 0.3146 m / 7.744 deg | 0.0% | 8.3% |
| THuman | 0.0473 m / 0.220 deg | 0.0349 m / 0.245 deg | 0.0898 m / 1.340 deg | 0.0% | 0.0% |
| MVHuman100 | 0.3231 m / 3.124 deg | 0.5718 m / 10.135 deg | 1.4591 m / 35.873 deg | 62.5% | 79.2% |
| MVHuman200 | 0.4836 m / 9.371 deg | 0.7673 m / 18.132 deg | 1.0460 m / 27.964 deg | 8.3% | 61.1% |

AvatarReX and THuman pass the local geometric upper-bound test. The aggregate failure is driven by MVHuman, especially MVHuman200, where even same-view fresh-versus-warmed pointmaps are not rigidly consistent enough.

This rules out the interpretation that the current bottleneck is only a missing cross-view descriptor. On MVHuman, local depth/pointmap quality is already a limiting factor before matching is introduced.

## 6. SE(3), Sim(3), And Three Frames

Sim(3) is not a general solution:

| Method | SE(3) T / R | Sim(3) T / R |
|---|---:|---:|
| Same-view, 1 frame | 0.2500 m / 3.553 deg | 0.2613 m / 3.979 deg |
| Same-view, 3 frames | 0.2087 m / 2.898 deg | 0.2275 m / 3.331 deg |
| Actual anchors, 3 frames | 0.5455 m / 13.636 deg | 0.7323 m / 16.455 deg |

Sim(3) slightly improves translation on THuman and MVHuman100, but substantially hurts MVHuman200 and does not fix rotation. The dominant failure is not one global scale factor; it includes non-rigid or view-dependent pointmap/depth inconsistency.

Three frames are materially better than one frame:

- actual-anchor translation: `0.6250 -> 0.5455 m`;
- actual-anchor rotation: `15.614 -> 13.636 deg`;
- actual-anchor catastrophic rate: `38.3% -> 35.6%`;
- same-view pseudo rotation: `3.553 -> 2.898 deg`.

The fixed delay is useful, but it does not rescue the route by itself.

## 7. Coverage And Sampling

Spatial coverage is much more important than raw confidence. With 64 points:

| Mode | Confidence-only T / R | Spatial T / R |
|---|---:|---:|
| Same-view, 1 frame | 0.597 m / 12.96 deg | 0.245 m / 3.41 deg |
| Same-view, 3 frames | 0.767 m / 19.61 deg | 0.201 m / 2.71 deg |
| Actual anchors, 3 frames | 0.960 m / 30.70 deg | 0.514 m / 13.04 deg |

Increasing point count does not improve the result. Sixty-four spatially distributed points are slightly better than 1,024 or 4,096 points, indicating that biased or correlated geometry dominates over correspondence count.

Difficulty groups confirm this:

| Group | Same-view 3-frame T / R | Actual anchors 3-frame T / R | Actual catastrophic |
|---|---:|---:|---:|
| High texture | 0.04 m / 0.4 deg | 0.23 m / 5.4 deg | 5% |
| Low texture | 0.37 m / 5.0 deg | 1.25 m / 33.0 deg | 73% |
| High overlap | 0.08 m / 0.9 deg | 0.12 m / 2.1 deg | 0% |
| Low overlap | 0.38 m / 4.9 deg | 1.35 m / 37.7 deg | 83% |
| Non-degenerate 3D | 0.10 m / 1.4 deg | 0.20 m / 4.8 deg | 5% |
| Planar/coverage-failed | 0.23 m / 2.5 deg | 0.93 m / 25.6 deg | 68% |

Using all valid points gives a better pseudo upper bound than excluding the person (`0.173 m / 2.21 deg` versus `0.209 m / 2.90 deg`). This does not justify using a moving person as an absolute world anchor. It instead shows that the current static mask and static-scene coverage remove useful constraints and need improvement.

## 8. Frozen Descriptor Smoke

The Stage-2 implementation was compiled and smoke-tested before the Stage-1 gate completed. On one AvatarReX case:

- Oracle correspondence recovered approximately `0.079 m / 0.98 deg`;
- raw frozen matching had 0 physical matches within 20 cm in the best inspected branch;
- its best camera rotation remained above 35 degrees.

This is implementation validation only, not a formal Stage-2 result. The production Stage-2 job was stopped immediately after Stage-1 failed, as required by the protocol.

## 9. Conditional Stages

The following stages were not executed as full experiments because the Stage-1 hard gate failed:

- full keyframe-retrieval and frozen-correspondence comparison;
- persistent memory replacement-policy study;
- scene plus human/gravity soft constraints;
- geometry-dominant Accept/Wait/Fallback reliability training.

Consequently, V13 makes no formal claim about cross-source reliability AUROC or multi-person soft constraints. The current 180-cut loader also uses `max_humans=1`, so a multi-person subgroup cannot be estimated honestly in this round.

## 10. Answers

1. **Is fresh Human3R local geometry sufficient with correct correspondence?** Not across all 180 cuts. It is sufficient on AvatarReX and THuman, but fails the 1-2 degree upper-bound criterion on MVHuman.
2. **What is the main bottleneck?** First local pointmap/depth consistency on MVHuman, then historical scene coverage and actual world-anchor coordinate quality. Matcher and reliability are downstream bottlenecks, not the current first blocker.
3. **Is Sim(3) required?** No. Sim(3) is worse overall and especially worse on MVHuman200. A single scale correction cannot explain the error.
4. **Is one frame sufficient?** Three frames are consistently better and should be used in the next geometry study, but three frames are still not accurate enough for deployment.
5. **Do Human3R tokens become useful when bound to world XYZ?** Not established. The one-case frozen matcher smoke still fails badly, and the formal matcher stage was correctly stopped.
6. **Are human heading, motion, and gravity useful soft constraints?** Not evaluated in V13 because scene geometry failed first. Previous partial Oracles support using them only after a valid scene candidate exists.
7. **Does geometry-dominant reliability generalize?** Not tested; there is no sufficiently accurate deployable candidate for a meaningful Accept/Wait/Fallback study yet.
8. **Should a trainable World-Anchor Shot Prompt be started now?** No.

## 11. Decision

The World-Coordinate Memory concept is conditionally plausible on AvatarReX and THuman, but the current frozen Human3R pointmap is not a reliable geometric substrate across MVHuman. Training a descriptor projector or World-Anchor Matcher now would optimize a downstream component whose geometric upper bound is still inadequate.

The next experiment should focus on:

```text
hard reset
+ fixed three-frame local geometry
+ camera-frame depth/scale audit on MVHuman
+ multi-frame pointmap fusion or depth correction
+ spatially distributed static-scene anchors
```

Only after the three-frame local geometry upper bound reaches approximately 1-2 degrees with low P90/P95 and low fit-failure rate should V13 Stage-2 be resumed. At that point, train correspondence confidence or a descriptor projector, not a direct SE(3) regressor.

## 12. Code And Outputs

Code:

```text
scripts/v13_audit_world_coordinate_memory_data.py
scripts/v13_scene_coordinate_oracle.py
scripts/v13_merge_scene_coordinate_oracle.py
scripts/v13_frozen_world_memory_probe.py
scripts/v13_merge_frozen_world_memory_probe.py
```

Outputs:

```text
output/v13_world_coordinate_memory/data_audit.json
output/v13_world_coordinate_memory/stage1_scene_coordinate_oracle/stage1_shard_*.json
output/v13_world_coordinate_memory/stage1_scene_coordinate_oracle/merged/stage1_merged.json
output/v13_world_coordinate_memory/stage1_scene_coordinate_oracle/merged/stage1_cases.csv
output/v13_world_coordinate_memory/stage1_scene_coordinate_oracle/merged/stage1_summary.md
```
