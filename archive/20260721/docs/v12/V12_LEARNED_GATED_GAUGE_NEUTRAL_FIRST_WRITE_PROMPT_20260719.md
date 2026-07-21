# V12 Learned Gated Gauge-Neutral First-Write Prompt

## 1. Question

V11 proved that a per-sample Gauge-Neutral First-Write Oracle can improve future shot-local camera motion, camera-frame pointmaps, and relative human motion without changing the boundary frame or the explicit Boundary SE(3).

V12 asks whether that causal Oracle effect can be converted into a small deployable module that:

- reads the unpolluted pre-cut `S_{t-1}` as read-only history;
- corrects only the first fresh-state write;
- predicts a bounded state residual and scalar gate;
- leaves boundary camera, pointmap, and SMPL-X outputs unchanged;
- uses no absolute world loss;
- safely composes with one fixed explicit shot-level SE(3).

Human3R and all original heads remain frozen. The adapter has 2,579,795 trainable parameters. No original Human3R inference path was modified.

## 2. Data And Split

Teacher-cache generation used V11's six-step GPU Gauge-Neutral Oracle:

| Split | Total | Real cuts | Pseudo-cuts | Sources |
|---|---:|---:|---:|---|
| Train | 149 | 125 | 24 | AvatarReX 48, THuman 48, MVHuman100 29, pseudo 24 |
| Validation | 31 | 19 | 12 | MVHuman100 19, pseudo 12 |
| Test | 36 | 36 | 0 | completely unseen MVHuman200 |

The full cache contains the same 180 real cross-camera cuts as V11 plus 36 pseudo-cuts. Splits are disjoint by source/capture group; MVHuman200 is never used by either training stage.

All Human3R inference, Oracle generation, Stage A distillation, Stage B rollout fine-tuning, and final evaluation ran on GPUs. Stage A trained the gated, ungated, and no-old-state variants for 1,800 steps in parallel. Stage B fine-tuned the same variants through future offsets 1, 2, 4, and 8 for 240 steps in parallel.

## 3. Model And Loss Boundary

The adapter reads:

- old state tokens and old pose-memory summary;
- raw fresh first-write state;
- current image-token and human-token summaries;
- current state-conditioned camera activation;
- eight explicit diagnostics.

It outputs a token-wise bounded residual, scalar gate, predicted gain, and wait score:

```text
corrected_state = fresh_state + gate * bounded_residual
```

The residual is bounded to 0.5 fresh-state standard deviations. The boundary output is generated before the corrected state is committed. Training uses only future relative camera, camera-frame pointmap/depth, relative human-root, and relative torso losses. Explicit SE(3) is evaluation-only and applied once.

## 4. Distillation Audit

Stage A shows a split result:

- train gate AUROC: `0.895`;
- train gate calibration error: `0.037`;
- unseen validation gate AUROC: `0.631`;
- unseen validation gate calibration error: `0.126`;
- gated latent recovery: `0.25%` on train and `-2.0%` on validation.

The difficulty label is partly readable in the training domain, but the full token-wise Oracle residual is not distilled by latent supervision. Future-rollout fine-tuning is therefore essential, but it starts from a weak residual model.

## 5. Main Unseen-Cut Results

All numbers below use the 36 completely unseen MVHuman200 cuts.

| Method | Rel camera T | Rel camera R | Camera-frame pointmap | Human rel root | Strict success |
|---|---:|---:|---:|---:|---:|
| Hard Reset | 0.1818 m | 4.153 deg | 0.3239 m | 0.1076 m | 44.4% |
| Boundary Output Correction Only | 0.1818 m | 4.153 deg | 0.3239 m | 0.1076 m | 44.4% |
| V11 Gauge-Neutral Oracle | **0.0622 m** | **1.069 deg** | **0.2491 m** | **0.0867 m** | **80.6%** |
| Learned Ungated Adapter | 0.1705 m | 3.871 deg | 0.3212 m | 0.1083 m | 44.4% |
| Learned Gated Adapter | 0.1752 m | 3.977 deg | 0.3217 m | 0.1072 m | 44.4% |
| Learned Adapter without Old State | 0.1732 m | 3.924 deg | 0.3212 m | 0.1070 m | 44.4% |

The gated model improves mean translation by only `0.0066 m` and rotation by `0.176 deg`. It retains `5.5%` of the Oracle translation gain and `5.7%` of the Oracle rotation gain, far below the 20% stop threshold and 30% feasibility target.

Strict success does not improve. Relaxed success also remains `72.2%`; only the ungated model reaches `75.0%`.

## 6. Tail And Rollout

| Method | T median | T P90 | T P95 | R median | R P90 | R P95 |
|---|---:|---:|---:|---:|---:|---:|
| Hard Reset | 0.1196 | 0.4361 | 0.4690 | 2.526 | 9.546 | 10.998 |
| Oracle | 0.0460 | 0.1231 | 0.1564 | 0.678 | 2.435 | 2.718 |
| Ungated | 0.1078 | 0.4274 | 0.4498 | 2.206 | 9.144 | 10.514 |
| Gated | 0.1070 | 0.4332 | 0.4575 | 2.133 | 9.317 | 10.688 |
| No old state | 0.1053 | 0.4278 | 0.4547 | 2.158 | 9.148 | 10.613 |

The learned adapter produces a small persistent improvement at all future offsets, but the gap to the Oracle grows with rollout length:

| Offset | Reset T | Gated T | Oracle T | Reset R | Gated R | Oracle R |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0603 | 0.0574 | 0.0272 | 1.203 | 1.134 | 0.529 |
| 2 | 0.1167 | 0.1128 | 0.0428 | 2.416 | 2.319 | 0.653 |
| 4 | 0.1852 | 0.1787 | 0.0576 | 4.182 | 4.012 | 1.058 |
| 8 | 0.2504 | 0.2380 | 0.0812 | 5.852 | 5.533 | 1.389 |

This is a real state-transition effect rather than a boundary output edit, but it is too small to meet the proposed route criteria.

## 7. Gate And History Controls

The learned gate fails to transfer to MVHuman200:

- mean gate: `0.587`;
- correction rate above 0.5: `88.9%`;
- identity fallback rate below 0.1: `0%`;
- Oracle-helpfulness AUROC: `0.284`;
- gate-target correlation: `0.257`;
- predicted-gain correlation: `0.178`;
- wait-decision AUROC: `0.479`;
- every identity-target test case is incorrectly activated.

The residual is small enough that these false-positive activations do not cause large high-texture degradation, but the gate is not performing its intended safety function.

History controls are also negative:

| History source | Rel T | Rel R |
|---|---:|---:|
| Correct old state | 0.1752 m | 3.977 deg |
| No old state | **0.1732 m** | **3.924 deg** |
| Zero old state | 0.1753 m | 3.975 deg |
| Shuffled old state | 0.1790 m | 4.078 deg |
| Wrong-video old state | 0.1751 m | 3.971 deg |

Correct history is not consistently better than the controls. The adapter has mainly learned a weak generic correction from fresh-state/current diagnostics, not a deployable query into old world context.

## 8. Difficulty Groups

The learned gain is small in every test subgroup:

| Texture | Reset T -> Gated T | Reset R -> Gated R | Oracle T / R |
|---|---:|---:|---:|
| Low | 0.1592 -> 0.1533 | 3.790 -> 3.627 | 0.0780 / 1.348 |
| Medium | 0.2151 -> 0.2043 | 5.029 -> 4.752 | 0.0529 / 0.900 |
| High | 0.1711 -> 0.1681 | 3.640 -> 3.552 | 0.0558 / 0.957 |

Across 60-90, 90-120, and 120-150 degree buckets, gated translation improves by only 0.004-0.008 m and rotation by 0.11-0.22 deg. The no-old-state model is slightly better in every texture and angle bucket.

The current manifest has no background-overlap labels and all evaluated cuts contain one detected person, so low/high-overlap and multi-person subgroups cannot be estimated in this round.

## 9. Boundary, Explicit Composition, And Runtime

Boundary locking is exact for Oracle, gated, ungated, and no-old variants: maximum camera, pointmap, SMPL translation, and SMPL rotation difference is exactly zero.

| Method | World camera T | World camera R | World pointmap |
|---|---:|---:|---:|
| Explicit-only | 1.7551 m | 45.634 deg | 1.6080 m |
| Gated + same Explicit | 1.7535 m | 45.568 deg | 1.6071 m |
| Oracle + same Explicit | 1.7179 m | 44.497 deg | 1.5978 m |

There is no double correction: Gated + Explicit is marginally better than Explicit-only. The final gain is nevertheless negligible because both the learned local correction and the explicit Boundary transform remain inaccurate.

Nine-frame reset inference averages `2.324 s`; gated inference averages `2.334 s`, an additional `10.5 ms` per cut sequence. Peak allocated memory rises from about `6035 MB` to `6039 MB`. Because the adapter exists only in independent cut-triggered experiment code, ordinary no-cut Human3R inference is unchanged by construction.

## 10. Interpretation

The failure has three separate causes:

1. **Oracle residual is not easily distillable.** Token-wise latent recovery is near zero even before cross-dataset transfer.
2. **The module does not use old state causally.** No-old and wrong-state controls match or beat correct history.
3. **The gate does not generalize.** It collapses toward always-on behavior on MVHuman200 and cannot predict Oracle benefit or wait decisions.

The small broad improvement is likely a generic fresh-state residual learned from rollout supervision. It is not evidence that the V11 read-old/write-fresh architecture has been converted into a reliable deployable State-query Prompt.

## 11. Decision

Answers to the requested questions:

1. **Can one-forward distillation learn the V11 Oracle effect?** Only about 5-6% of the camera gain, which is insufficient.
2. **Does it generalize to unseen real camera cuts?** It gives a small average improvement on MVHuman200, but fails the predefined retention, success-rate, and tail criteria.
3. **Does the gate separate difficult and simple samples?** No. AUROC is below random and identity fallback never triggers.
4. **Does it improve future rollout while locking the boundary?** Yes, slightly, and the boundary remains exactly unchanged.
5. **Is correct old state indispensable?** No. The no-old-state model is slightly better.
6. **Is gated better than hard reset, output-only, and ungated?** It is slightly better than hard reset/output-only, but worse than ungated and no-old.
7. **Does Explicit composition avoid double correction?** Yes, but the improvement is negligible.
8. **Should this become the final gauge-neutral recurrent transition module?** No.

The current first-write state-modification route should stop. The next mainline should be:

```text
hard reset local state
+ explicit scene/world relocalization
+ fixed shot-level SE(3)
+ redesigned reliability/fallback/wait prediction
```

Reliability should be geometry-dominant and explicitly regularized for cross-source calibration. Token/state summaries may remain auxiliary inputs, but the current learned gate is not deployable evidence of useful old-world state querying. Persistent world-coordinate memory and better explicit Boundary candidates remain higher-priority than further enlarging the latent correction module.

## 12. Code And Outputs

Code:

```text
src/dust3r/v12_gated_gauge_neutral_prompt.py
scripts/v12_build_gauge_neutral_teacher_cache.py
scripts/v12_merge_teacher_cache.py
scripts/v12_train_gated_first_write_prompt.py
scripts/v12_gated_first_write_runtime.py
scripts/v12_finetune_gated_prompt_rollout.py
scripts/v12_eval_gated_first_write_prompt.py
scripts/v12_merge_gated_prompt_eval.py
```

Outputs:

```text
output/v12_gated_first_write/teacher_cache_loso_mvhuman200/
output/v12_gated_first_write/training_loso_mvhuman200/
output/v12_gated_first_write/eval_loso_mvhuman200/merged/v12_eval_merged.json
output/v12_gated_first_write/eval_loso_mvhuman200/merged/v12_eval_cases.csv
output/v12_gated_first_write/eval_loso_mvhuman200/merged/v12_eval_summary.md
```
