# Timestamp-aware velocity-residual group tangent

Phase: `validate`.

> CPU cache only; no pretrained-model/GPU forward. GT is evaluator-only. Camera is unchanged; rejected/unmatched people are exact B0.
> Dataset frame timestamps—not stream-list indices—define physical `delta_t`. EgoHumans confirmation cuts all have `delta_t=0`.

## Policy

`{"fraction": 0.2, "cap_m": 0.1, "group_dispersion_gate_m": 0.2, "apply_when_dt_zero": false, "history_frames": 5, "min_history": 3, "velocity_speed_gate_m_per_frame": 0.06, "velocity_residual_gate_m_per_frame": 0.05, "extrapolation_cap_m": 0.3, "min_group_people": 2}`

## Cache and runtime audit

- Every MultiHuman case has five contiguous pre-shot dataset frames. Physical `dt` is `post_frame - pre_frames[-1]`: three-k0 is 0, three-offset1 is 1, and dance/box contain 0/1/2/4/8.
- All six Ego cuts repeat the same physical dataset timestamp across cameras; their `dt` is 0. The frozen `apply_when_dt_zero=false` bit makes them exact Kabsch fallback.
- Cache person keys came from GT mesh assignment and are never consumed by the candidate. Pre histories and cut matching are rebuilt with anonymous root+torso+joints Hungarian.
- Runtime action is one shared bounded translation over BRTC-accepted people. Camera is untouched; BRTC-rejected and unmatched people remain exact B0; no future post frame is read.

```text
velocity_i = robust_velocity(last 5 causal pre roots, dataset timestamps)
anchor_i   = pre_root_i + physical_dt * velocity_i
tangent_i  = tangent_to_post_camera_ray(anchor_i - brtc_post_root_i)
group      = coordinate_median(tangent_i)
shift      = clip(fraction * group, cap), after observable reliability/dispersion gates
```

## Split and contamination contract

The 12-policy grid was selected only on the deterministic timestamp development subset and hashed before this confirm run. Previous sequence-level reports and the earlier two-frame candidate were already known, so this is grouped-CV/exploratory—not blind validation. Confirmation results are not reused for retuning.

## MultiHuman confirmation

| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Active | Safe |
|---|---|---:|---:|---:|---:|---:|---:|---|
| three | brtc | 0.199191 | 0.250177 | 0.222681 | 0.096552 | 0.207540 | 0.0% | False |
| three | brtc_kabsch | 0.199191 | 0.250103 | 0.223251 | 0.096552 | 0.207540 | 0.0% | False |
| three | velocity_kabsch | 0.199191 | 0.250103 | 0.223251 | 0.096552 | 0.207540 | 0.0% | False |
| three | safety audit | -- | -- | -- | -- | -- | -- | vs BRTC `False`, vs Kabsch `True` |
| dance | brtc | 0.112735 | 0.170199 | 0.142649 | 0.033629 | 0.078998 | 61.3% | False |
| dance | brtc_kabsch | 0.112735 | 0.167238 | 0.139753 | 0.033629 | 0.078998 | 61.3% | False |
| dance | velocity_kabsch | 0.119232 | 0.165866 | 0.141464 | 0.033629 | 0.078998 | 61.3% | False |
| dance | safety audit | -- | -- | -- | -- | -- | -- | vs BRTC `False`, vs Kabsch `False` |
| box | brtc | 0.516311 | 0.564962 | 0.596854 | 0.051839 | 0.680089 | 44.4% | True |
| box | brtc_kabsch | 0.516311 | 0.561398 | 0.590074 | 0.051839 | 0.680089 | 44.4% | True |
| box | velocity_kabsch | 0.514029 | 0.558282 | 0.586750 | 0.051839 | 0.680089 | 44.4% | True |
| box | safety audit | -- | -- | -- | -- | -- | -- | vs BRTC `True`, vs Kabsch `True` |
| three_offset1 | brtc | 0.214908 | 0.257597 | 0.231281 | 0.089793 | 0.238882 | 92.9% | True |
| three_offset1 | brtc_kabsch | 0.214908 | 0.255917 | 0.230771 | 0.089793 | 0.238882 | 92.9% | True |
| three_offset1 | velocity_kabsch | 0.205536 | 0.246123 | 0.219988 | 0.089757 | 0.238617 | 92.9% | True |
| three_offset1 | safety audit | -- | -- | -- | -- | -- | -- | vs BRTC `True`, vs Kabsch `True` |

### Anonymous tracking/association evaluator audit

These accuracies use GT only after anonymous assignments are fixed. They are never gates.

| Split | Pre-track edge accuracy | Cut association accuracy |
|---|---:|---:|
| three | 95.6% | 64.7% |
| dance | 100.0% | 100.0% |
| box | 96.5% | 100.0% |
| three_offset1 | 97.4% | 67.5% |

## Physical timestamp-delta decomposition

The rows below are case-mean deltas of velocity+Kabsch versus Kabsch. They are evaluator-only attribution; no GT-derived value enters a gate.

| Split | dt | Cases | Active | Δroot | Δjoint | Δvertex |
|---|---:|---:|---:|---:|---:|---:|
| three | 0 | 17 | 0.0% | +0.000 mm | +0.000 mm | +0.000 mm |
| dance | 0 | 7 | 0.0% | +0.000 mm | +0.000 mm | +0.000 mm |
| dance | 1 | 6 | 83.3% | +5.574 mm | -2.179 mm | +1.978 mm |
| dance | 2 | 6 | 83.3% | +8.778 mm | -2.047 mm | +3.286 mm |
| dance | 4 | 6 | 83.3% | +10.063 mm | -1.365 mm | +2.555 mm |
| dance | 8 | 6 | 66.7% | +9.151 mm | -1.496 mm | +1.024 mm |
| box | 0 | 7 | 0.0% | +0.000 mm | +0.000 mm | +0.000 mm |
| box | 1 | 7 | 57.1% | -5.852 mm | -7.401 mm | -7.626 mm |
| box | 2 | 7 | 71.4% | -0.949 mm | -2.198 mm | -1.913 mm |
| box | 4 | 7 | 57.1% | +0.449 mm | -0.842 mm | -0.707 mm |
| box | 8 | 8 | 37.5% | -4.712 mm | -4.888 mm | -5.992 mm |
| three_offset1 | 1 | 42 | 92.9% | -8.774 mm | -9.352 mm | -10.267 mm |

## EgoHumans confirmation

| Method | W | WA | Root | Joint | Vertex | Pair dist | Pair vec | Root accel | Joint accel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| brtc | 314.059 | 202.461 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 125.270 |
| brtc_kabsch | 312.769 | 200.029 | 380.688 | 383.933 | 383.791 | 176.559 | 333.091 | 115.698 | 123.167 |
| velocity_kabsch | 312.769 | 200.029 | 380.688 | 383.933 | 383.791 | 176.559 | 333.091 | 115.698 | 123.167 |

All Ego physical timestamp deltas zero: `True`.
Ego safe versus Kabsch: `True`.
Final status: **NO_GO_VELOCITY_RESIDUAL_GROUP_TANGENT**.

## Decision analysis

- `dt=0` is an exact Kabsch fallback by frozen policy. Ego therefore validates fallback invariants and temporal metrics, not the velocity branch.
- The velocity branch improves `three_offset1` and `box`, but fails `dance`: coherent motion extrapolation and cross-shot root bias remain observationally confounded.
- `three` fails the full-stack-vs-BRTC gate only because its exact Kabsch fallback increases vertex mean under the fully anonymous, low-accuracy cut association; the velocity branch itself is inactive.
- No threshold is retuned after confirmation. Dataset-level prior reports were already known, so this is grouped-CV/exploratory confirmation, not a blind benchmark claim.

## Reproduction

```bash
.venv/bin/python versions/v14/probe_brtc_velocity_residual_group_tangent.py --phase audit
.venv/bin/python versions/v14/probe_brtc_velocity_residual_group_tangent.py --phase dev
.venv/bin/python versions/v14/probe_brtc_velocity_residual_group_tangent.py --phase freeze
.venv/bin/python versions/v14/probe_brtc_velocity_residual_group_tangent.py --phase validate
```
