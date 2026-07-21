# V11 Gauge-Neutral First-Write Complementarity Probe

## 1. Motivation

V10 showed that the first fresh-state write controls later camera, pointmap, and human rollout, but its learned Prompt was supervised toward the teacher's absolute world gauge. A fixed shot-level explicit SE(3) corrected the same gauge again, so the combination produced double correction.

V11 separates the responsibilities:

- the explicit shot-level SE(3) is the only owner of absolute world gauge;
- first-write correction may only improve shot-local motion, camera-frame geometry, and later recurrent state;
- the boundary output and shot-level SE(3) are locked;
- no absolute world loss or full teacher-state latent MSE is used by the gauge-neutral Oracle.

The experiment asks whether first-write correction still has an independent role after all absolute gauge correction is removed from its objective.

## 2. Data And Protocol

The same 180 real synchronized cross-camera cuts used by V10 were evaluated:

| Source | Cases |
|---|---:|
| AvatarReX | 48 |
| MVHuman100 | 48 |
| MVHuman200 | 36 |
| THuman | 48 |

For every case:

1. Human3R is hard-reset at the first B-camera frame.
2. The B shot contains up to nine frames, so offsets 1, 2, 4, and 8 can be evaluated.
3. A same-camera B teacher is warmed with eight earlier B frames. It supplies camera-frame pointmap targets because processed scene depth GT is unavailable.
4. Camera and human relative-motion targets use dataset GT.
5. GT Boundary SE(3) is used only for the diagnostic world upper bound.
6. The deployable Explicit comparison uses V10's fixed `human_mean_pointmap_history_standard` candidate.
7. V10's explicit transform is converted from Human3R frame-0 gauge to raw dataset world using each case's saved `gt_gauge.gauge_transform`.

All Human3R inference and Oracle optimization ran on GPUs. Human3R remained frozen.

## 3. Stage One: Residual Audit

After hard reset and perfect GT Boundary SE(3), substantial gauge-neutral tail error remains:

| Metric | Mean | Median | P90 | P95 |
|---|---:|---:|---:|---:|
| Relative camera translation | 0.077 m | 0.021 m | 0.211 m | 0.357 m |
| Relative camera rotation | 1.43 deg | 0.22 deg | 4.59 deg | 8.12 deg |
| Camera-frame pointmap | 0.157 m | 0.076 m | 0.435 m | 0.561 m |
| Human relative root | 0.102 m | 0.071 m | 0.199 m | 0.289 m |

The error grows along the shot: relative rotation rises from `0.48 deg` at offset 1 to `1.90 deg` at offset 8, and translation rises from `0.030 m` to `0.102 m`. The residual is concentrated in MVHuman; AvatarReX and THuman camera trajectories are already very accurate after reset.

Therefore a real gauge-neutral state-transition correction space exists, especially for the difficult MVHuman cases.

## 4. Stage Two: Gauge-Neutral First-Write Oracle

The boundary frame runs normally and its output is saved unchanged. The Oracle modifies only the `768 x 768` state tensor that would be committed for the next frame.

The optimized losses use only:

- relative camera translation and rotation;
- camera-frame pointmap and depth;
- human relative root trajectory;
- relative torso orientation and local pose;
- future offsets 1, 2, 4, and 8.

Losses are normalized by each hard-reset offset's error and combine mean and maximum normalized error. This prevents the optimizer from sacrificing an already-good offset 1 to repair a very bad offset 8. The state residual is bounded to `0.5` state standard deviations and regularized. Six GPU optimization steps are used per case.

The boundary frame is not part of the optimization loss and the corrected state is inserted only after boundary heads have produced their outputs.

## 5. Main Results

### 5.1 Gauge-neutral local outputs

| Method | Relative camera T | Relative camera R | Camera-frame pointmap | Human relative root |
|---|---:|---:|---:|---:|
| Hard Reset + GT Boundary | 0.0771 m | 1.433 deg | 0.1567 m | 0.1020 m |
| Full teacher `S_t` replacement + GT Boundary | 0.2083 m | 4.864 deg | 0.0081 m | 0.1486 m |
| Gauge-Neutral First-Write Oracle + GT Boundary | **0.0293 m** | **0.458 deg** | **0.1134 m** | **0.0852 m** |
| Boundary Output Correction Only | 0.0771 m | 1.433 deg | 0.1567 m | 0.1020 m |

The gauge-neutral Oracle improves:

- relative camera translation by 62.0%;
- relative camera rotation by 68.1%;
- camera-frame pointmap by 27.7%;
- human relative root by 16.5%.

Strict relative-camera success increases from `77.8%` to `96.1%`; relaxed success increases from `90.6%` to `100%`.

### 5.2 Tail metrics

| Metric | Reset P90 | Oracle P90 | Reset P95 | Oracle P95 |
|---|---:|---:|---:|---:|
| Relative camera translation | 0.212 m | 0.069 m | 0.357 m | 0.091 m |
| Relative camera rotation | 4.59 deg | 1.05 deg | 8.13 deg | 1.43 deg |
| Camera-frame pointmap | 0.435 m | 0.337 m | 0.566 m | 0.512 m |
| Human relative root | 0.200 m | 0.179 m | 0.290 m | 0.270 m |

The largest and clearest benefit is on camera trajectory tails. Pointmap and human tails improve, but less strongly.

### 5.3 Future rollout

| Offset | Reset T | Oracle T | Reset R | Oracle R | Reset pointmap | Oracle pointmap |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 m | 0.000 m | 0.002 deg | 0.002 deg | 0.251 m | 0.251 m |
| 1 | 0.030 m | 0.018 m | 0.481 deg | 0.302 deg | 0.196 m | 0.143 m |
| 2 | 0.052 m | 0.023 m | 0.881 deg | 0.321 deg | 0.177 m | 0.123 m |
| 4 | 0.075 m | 0.026 m | 1.396 deg | 0.418 deg | 0.156 m | 0.110 m |
| 8 | 0.102 m | 0.032 m | 1.901 deg | 0.505 deg | 0.133 m | 0.101 m |

The gain is a future-rollout effect rather than a boundary-output correction.

### 5.4 Dataset behavior

| Source | Reset T | Oracle T | Reset R | Oracle R | Reset pointmap | Oracle pointmap |
|---|---:|---:|---:|---:|---:|---:|
| AvatarReX | 0.0127 m | 0.0112 m | 0.129 deg | 0.162 deg | 0.0397 m | 0.0204 m |
| MVHuman100 | 0.1306 m | 0.0430 m | 2.022 deg | 0.598 deg | 0.2370 m | 0.1832 m |
| MVHuman200 | 0.1818 m | 0.0597 m | 4.153 deg | 1.041 deg | 0.3240 m | 0.2484 m |
| THuman | 0.0096 m | 0.0109 m | 0.108 deg | 0.176 deg | 0.0679 m | 0.0352 m |

The meaningful camera gain comes from MVHuman. AvatarReX and THuman start near zero and show a small rotation degradation while pointmap improves. A trained module therefore needs a reliability gate or identity fallback for easy cases.

Low-texture samples benefit most: translation drops from `0.150 m` to `0.050 m` and rotation from `2.82 deg` to `0.76 deg`. High-texture camera rotation changes from `0.123 deg` to `0.164 deg`, again showing the need for safe gating.

## 6. Explicit Complementarity

Using the correctly converted V10 fixed Explicit transform:

| Method | World camera T | World camera R | World pointmap |
|---|---:|---:|---:|
| Explicit-only | 1.7067 m | 24.231 deg | 1.6224 m |
| Gauge-Neutral Oracle + Explicit | 1.7053 m | 23.818 deg | 1.6218 m |

The local state correction survives composition with the same explicit SE(3), but the final world improvement is small: about `0.0014 m`, `0.41 deg`, and `0.0007 m` respectively. The reason is structural: the explicit boundary transform already contributes much larger absolute error, which the gauge-neutral Prompt is intentionally forbidden to correct.

Therefore V11 proves local role separation, but it does not yet prove a large practical end-to-end world-space gain. Explicit scene/world relocalization remains the dominant global bottleneck.

## 7. Negative And Causal Controls

1. Boundary outputs are exactly identical between Hard Reset and Gauge-Neutral Oracle for camera, pointmap, SMPL translation, and SMPL rotations; maximum absolute difference is exactly zero.
2. Applying a random global SE(3) changes the relative-camera diagnostic by at most `8.4e-7` and camera-frame pointmap by at most `2.62e-6 m`.
3. Boundary-output-only correction has exactly the same future rollout as Hard Reset because it does not modify the committed state.
4. Replacing the full post-update teacher `S_t` nearly reconstructs the teacher pointmap (`0.008 m`) but worsens relative camera and human motion. This is direct evidence that full-state replacement imports an incompatible gauge instead of isolating local dynamics.
5. V10's absolute first-write Oracle changed rotation from `5.24 deg` alone to `44.63 deg` when combined with Boundary output correction. V11 avoids that supervision entirely.

## 8. What Did Not Improve

- Root-centered human pose is effectively unchanged: `0.1341 m` to `0.1343 m`.
- Local body pose is unchanged: `21.64 deg` to `21.64 deg`.
- Torso relative orientation is effectively unchanged: `4.310 deg` to `4.307 deg`.
- The current depth-consistency proxy worsens from `0.162 m` to `0.180 m`, although direct camera-frame depth error improves from `0.145 m` to `0.104 m`.

The present Oracle should therefore be interpreted as a camera/local-geometry and relative-root transition probe, not as evidence that first-write state correction repairs detailed body pose.

## 9. Decision

The answers to the requested questions are:

1. **Does real cross-camera reset retain gauge-neutral state-transition error after a correct Boundary SE(3)?** Yes, especially in MVHuman and low-texture samples.
2. **Does first-write correction have an effect independent of absolute world gauge?** Yes. Relative camera, camera-frame geometry, and relative human root improve under strictly gauge-neutral losses.
3. **Can it improve future rollout without changing the boundary frame?** Yes. The boundary is exactly locked and offsets 1, 2, 4, and 8 improve.
4. **Does Prompt + Explicit clearly beat Explicit-only?** It clearly improves shot-local trajectory, but only slightly improves final world metrics because Explicit boundary alignment is still inaccurate.
5. **Is it worth training a Gauge-Neutral State-query Prompt?** Yes, as a bounded local state-transition module with reliability fallback. It is not yet sufficient for a final end-to-end claim.
6. **Should latent correction be stopped entirely?** No. The Oracle upper bound is strong enough to justify one constrained training stage. In parallel, explicit world-memory relocalization must remain the higher-priority global module.

## 10. Next Training Stage

Train only a small first-write residual/gate/reliability module:

- old `S_{t-1}` is read-only;
- only corrected fresh state is committed;
- no early-camera adapter in the first version;
- losses use future relative camera, camera-frame pointmap/depth, and relative human root;
- no absolute world loss and no full teacher `S_t` latent MSE;
- explicit SE(3) is stop-gradient and applied exactly once;
- identity fallback is encouraged on already-good AvatarReX, THuman, and high-texture cases;
- reliability or wait-three-frame decisions should target the remaining pointmap/human tails.

Do not begin a formal Multi-THuMBS or final end-to-end comparison until both conditions hold:

1. the learned Prompt retains a substantial fraction of the Oracle's local P90/P95 gain on unseen scenes;
2. the explicit relocalization module is improved enough that local gains are visible in final world-space metrics.

## 11. Code And Results

Code:

```text
scripts/v11_gauge_neutral_first_write_probe.py
scripts/v11_merge_gauge_neutral_first_write.py
scripts/v11_gauge_neutral_first_write_oracle.py
scripts/v11_merge_gauge_neutral_oracle.py
```

Results:

```text
output/v11_gauge_neutral_first_write/stage1_full/merged/
output/v11_gauge_neutral_first_write/stage2_final_full/
output/v11_gauge_neutral_first_write/stage2_final_full/merged/
```
