# Frozen BRTC + global-orientation Kabsch on EgoHumans CPU cache

## 1. Protocol and runtime

This is the local three-chain provisional protocol, not the unpublished official Multi-THuMBS split.

Frozen policy: `max_angle=25.0°`, `fraction=0.5`, `min observable improvement=0.0`.

At every cut, frozen BRTC is executed first. Only its accepted matched people receive the frozen Kabsch rotation. The BRTC translation and one root-centred rigid rotation are then propagated to every frame in that post shot. The next cut's Kabsch estimator reads this rotated causal history, while the frozen BRTC translation branch reads its own v1 reference history so orientation cannot change roots.

Rejected/unmatched people are exact B0; camera is exact B0; native Human3R root is exact frozen BRTC v1.
No GPU/model forward, DA3, future frame, or GT-side inference is used.

## 2. Reproduced frozen MultiHuman validation

| Split | Joint v1 | Joint Kabsch | Vertex v1 | Vertex Kabsch | Applied | Safe |
|---|---:|---:|---:|---:|---:|---|
| three_offset1 | 0.274493 | 0.271315 | 0.252451 | 0.250248 | 88.0% | True |
| dance | 0.177804 | 0.168764 | 0.152914 | 0.148234 | 99.2% | True |
| box | 0.421610 | 0.418583 | 0.434528 | 0.429938 | 98.7% | True |

The current validate run reproduces the frozen GO result: root and both layout metrics remain invariant on all three splits, while joint/vertex improve.

## 3. EgoHumans provisional metrics

| Method | W | WA | pelvis MPJPE | pelvis MPVPE | Fixed root | Fixed joint | Fixed vertex | Pair dist | Pair vec | Root Accel | Joint Accel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b0 | 350.614 | 235.207 | 109.266 | 129.960 | 420.163 | 416.226 | 414.913 | 188.485 | 388.351 | 160.517 | 160.997 |
| brtc_v1 | 314.059 | 202.461 | 109.266 | 129.960 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 125.270 |
| brtc_kabsch | 312.769 | 200.029 | 101.526 | 119.928 | 380.688 | 383.933 | 383.791 | 176.559 | 333.091 | 115.698 | 123.167 |

## 4. Delta versus frozen BRTC v1

| Metric | Delta (candidate - v1) |
|---|---:|
| w_mpjpe_mm | -1.290 |
| wa_mpjpe_mm | -2.432 |
| pelvis_mpjpe_mm | -7.741 |
| pelvis_mpvpe_mm | -10.032 |
| fixed_world_root_mm | +0.034 |
| fixed_world_joint_mm | -0.796 |
| fixed_world_vertex_mm | -1.447 |
| pairwise_root_distance_mm | -0.466 |
| pairwise_root_vector_mm | -0.778 |
| world_root_accel_delta2_mm_per_frame2 | -0.315 |
| world_joint_accel_delta2_mm_per_frame2 | -2.104 |

## 5. Runtime and harm

BRTC accepted: `11/14`; Kabsch applied: `11/11` accepted boundary people; propagated person-frame rate `68.8%`.

| Error | Mean delta vs v1 | Improve | Harm >1cm | Harm >5cm | Max harm |
|---|---:|---:|---:|---:|---:|
| Fixed root | +0.052 mm | 37.5% | 0.0% | 0.0% | 4.140 mm |
| Fixed joint | -1.204 mm | 28.7% | 16.2% | 0.0% | 25.830 mm |
| Fixed vertex | -2.189 mm | 32.5% | 17.5% | 0.0% | 46.327 mm |

Rejected/unmatched exact B0 max change: `0.000e+00`.
Stored root max delta versus v1: `0.000e+00`.
Camera max delta versus B0: `0.000e+00`.
Second cut consumes inherited orientation: `True` (6/7 tracks).
Rotation SO(3) max orthogonality/determinant errors: `3.331e-16` / `4.441e-16`.

## 6. Deployable runtime versus frozen probe parity

The causal output now calls `b0_person_triangulation_orientation_kabsch.orientation_candidate` for each BRTC-accepted person. The original frozen probe is evaluated side-by-side on exactly the same corrected geometry.

EgoHumans causal accepted-person geometry max deltas (root/joint/vertex): `0.000e+00` / `0.000e+00` / `0.000e+00`; rotation max delta `0.000e+00`; parity `True`.
MultiHuman `three offset1`: `42` cuts / `125` people, geometry max delta `0.000e+00`, rotation max delta `0.000e+00`, parity `True`.

## 7. Native root versus mapped-pelvis diagnostic

The runtime root is the native Human3R `person['root']`; it is bit-exact v1. The provisional `fixed root/layout/root Accel` metrics instead regress a pelvis from SMPL-X→SMPL mapped vertices. This is a different point, so it rotates around the native root.

On v1 person frames, mapped-pelvis/native-root offset is median `18.193 mm`, mean `18.735 mm`, max `22.681 mm`.
Consequently the mapped-pelvis fixed-root mean changes by `+0.034 mm`, although the native root max delta is exactly `0.000e+00`.

## 8. Dual decision

MultiHuman frozen validation pass: `True`.
All requested Ego mean metrics non-regression: `False`.
All non-root requested means non-regression: `True`.
Root and joint Accel non-regression: `True`.
Runtime invariants pass: `True`.
Joint/vertex harm >5cm under 10%: `True`.
Strict-zero decision: **NO_GO_GLOBAL_ORIENTATION_KABSCH_EGOHUMANS**.
Secondary 0.1 mm mapped-pelvis tolerance audit: **QUALIFIED_GLOBAL_ORIENTATION_KABSCH_CANDIDATE**.

The strict-zero decision is retained unchanged: one requested diagnostic is +0.034 mm, so exact mean non-regression is false. The secondary result does not alter the frozen policy or the strict decision. It states that with an explicit 0.1 mm tolerance for this non-native mapped-pelvis proxy, the candidate qualifies: every non-root requested mean improves, both Accel metrics improve, runtime invariants pass, and joint/vertex >5 cm harm is zero.

## 9. Relation to Multi-THuMBS

Local Kabsch W/WA are `312.769` / `200.029 mm`, still `+33.769` / `+34.029 mm` above the paper's EgoHumans W/WA references.
Local pelvis MPJPE/MPVPE cannot be claimed as a paper win because the official split/evaluator is unpublished and this local metric only covers matched short-chain person frames.

## 10. Reproduction

```bash
.venv/bin/python versions/v14/probe_brtc_global_orientation_kabsch.py --phase validate
.venv/bin/python versions/v14/eval_brtc_global_orientation_kabsch_egohumans.py --self_test
.venv/bin/python versions/v14/eval_brtc_global_orientation_kabsch_egohumans.py
```
