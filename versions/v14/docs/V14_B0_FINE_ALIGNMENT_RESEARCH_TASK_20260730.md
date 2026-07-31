# V14 B0-Centered Fine Alignment Research Task

Date: 2026-07-30

Status: camera B0 frozen; BRTC-LC person root/layout refinement passed development and a new automatic-ID confirmation split; official Multi-THuMBS evaluation remains pending

## 1. Objective

Freeze the current learned `B0` as a useful identity-free coarse Boundary and find at
least one causal, shared-rigidity fine-alignment method that can be used as the main
Movie3R route.

The task is complete only when one method:

1. operates around `B0` instead of replacing it with a new unconstrained Boundary;
2. uses no GT camera, GT identity, GT depth, or future frame at deployment time;
3. outputs one shared transform for the post-cut camera, scene, and all humans;
4. improves frozen quantitative evaluation under the acceptance rules below;
5. has an implemented inference path, tests, reproducible command, and recorded failure
   boundary.

Failed experiments are part of the result. Each tested hypothesis must be recorded before
moving to the next one.

### 2026-07-31 scope correction

The original stop condition treated one shared post-shot Boundary as the complete fine-
alignment output. Subsequent GT visualization and person-level probes show that this scope
was too narrow. `B0 + da3_safe` can improve the shared camera/shot coordinate transform on
the validated MultiHuman domain, but a shared SE(3) cannot change Human3R's predicted
camera-relative person root depth, body scale, or internal structure. Therefore:

- the completed result in Sections 13--16 is retained as **camera refinement only**;
- its human-root improvement is a side effect of moving the whole shot, not proof that the
  camera-relative human reconstruction is correct;
- the original "one shared transform" constraint applies to the camera/scene Boundary,
  not to the newly required person-local refinement stage;
- the complete person fine-alignment stop condition in Section 11 has not been met.

The active task is now to freeze a qualified camera result, estimate a gated residual for
each person in that fixed camera frame, and evaluate it with world, pelvis-aligned, temporal,
camera, and identity metrics. No unvalidated person method is promoted in this document.

## 2. Frozen Starting Point

The coarse stage is frozen conceptually as:

```text
pre-cut state + first post-cut frame
-> non-committing V9-style shadow camera C_shadow

fresh reset + the same first post-cut frame
-> raw camera C_raw + raw pointmap + raw humans

B0 = C_shadow @ inverse(C_raw)
```

`B0` is applied to the raw-reset post-shot reconstruction. Shadow humans, shadow pointmaps,
and shadow recurrent state are never committed.

Current 180-cut strict-GT-ID diagnostic baseline:

| Method | Camera T | Camera R | Composite | P95 composite | Human root |
|---|---:|---:|---:|---:|---:|
| frozen `B0` | 0.277 m | 3.85 deg | 0.354 | 0.577 | 0.442 m |

The active checkpoint is a coarse baseline, not a final broadly trained model. It must not
be repeatedly tuned on the frozen evaluation while fine-alignment evidence is being
studied.

## 3. Problem Definition

Let the frozen coarse Boundary be:

\[
B_0=[R_0,t_0].
\]

Fine alignment estimates a small right-invariant residual:

\[
B^*=B_0\Delta B,
\qquad
\Delta B=[\Delta R,\Delta u].
\]

Therefore:

\[
B^*=[R_0\Delta R,\ t_0+R_0\Delta u].
\]

The residual must have a trust region and a confidence/acceptance mechanism. The exact
identity transform must always remain available as the safe fallback.

The research question is not "can a flexible solver fit GT?". It is:

> Which causal evidence consistently predicts the remaining error after `B0`, across
> camera pairs, people, motions, and datasets, without absorbing Human3R root-depth bias?

## 4. Non-Negotiable Constraints

- causal and online;
- first post-cut frame by default; a short causal post-cut window is allowed only as an
  explicitly measured latency ablation;
- fixed-size external state;
- one Boundary per shot, fixed after commit;
- one shared SE(3), or Sim(3) only if a physically shared scale is independently proven;
- no per-human world transform;
- no forcing pre/post camera centers or human roots to be equal;
- no unbounded overwrite of `B0`;
- no test-set-specific thresholds, camera-pair rules, or source-specific method selection;
- no use of GT for proposal generation, confidence, gating, or deployment-time matching.

## 5. Data and Split Discipline

### Development

- `three`: parameter selection, trust-region selection, and acceptance-rule selection;
- AvatarReX/THuman/MVHuman controlled data: mechanism diagnosis, oracle studies, and
  source-diversity checks;
- single-sequence or tiny probes: implementation validation only, never method selection.

### Frozen evaluation

- `dance`;
- `box`;
- any held-out AvatarReX/THuman/MVHuman split already disjoint from training/probe data.

Frozen results may be inspected only after a candidate, its hyperparameters, and its gate
have been fixed on development data. A failed frozen result is recorded; its parameters
must not be tuned on that frozen set.

## 6. Acceptance Rules

A main-line candidate must satisfy all safety rules:

1. overall camera composite mean is no worse than `B0` by more than 1%;
2. camera composite P95 is no worse than `B0` by more than 2%;
3. catastrophic rate does not increase;
4. no frozen sequence has camera composite mean degradation above 2%;
5. one shared transform is verified numerically for camera, pointmap, roots, joints, and
   vertices;
6. no-cut behavior is bit-exact or within the existing numerical tolerance.

It must additionally provide one material gain:

- at least 5% lower overall camera composite with human root/layout non-inferior; or
- at least 8% lower human-root error and 5% lower layout error while satisfying camera
  non-inferiority; or
- a comparably strong, predeclared geometric improvement supported on every frozen source.

The paired improve rate, mean/median/P90/P95, per-sequence result, and worst cases are all
reported. A mean-only win is insufficient.

## 7. Experimental Strategy

### Phase A: Residual observability audit

Before designing another solver, measure the GT residual around `B0` and decompose it into:

- rotation axis and magnitude;
- camera-local lateral, vertical, and depth translation;
- relation to view angle and baseline;
- relation to Human3R root-depth error;
- relation to background overlap, pointmap confidence, and scene residual;
- relation to torso/body-frame disagreement and multi-human layout disagreement.

For every causal cue, measure:

- correlation with the signed residual, not only residual magnitude;
- oracle upper bound if only that cue/component were corrected;
- calibration and tail failures;
- cross-sequence and cross-source stability.

Signals that explain only training/source identity, or improve the mean by harming tails,
are rejected before integration.

### Phase B: Candidate families

Test candidate families in increasing complexity:

1. **Component-wise bounded residuals**
   - rotation-only torso/body-frame evidence;
   - lateral/vertical translation without root-depth overwrite;
   - depth-only evidence with an explicit reliability test.
2. **Robust scene correspondence**
   - mutual feature matches rather than nearest-neighbour Chamfer alone;
   - human masking and confidence filtering;
   - repeated-plane/position-collapse hard negatives;
   - geometric verification before estimating a residual.
3. **Human/layout consensus as weak evidence**
   - torso orientation and pairwise layout;
   - robust motion prediction and identity uncertainty;
   - never a direct equation that pins each predicted root to its history anchor.
4. **Evidence fusion around `B0`**
   - independent proposal reliabilities;
   - disagreement detection;
   - constrained robust optimization or a small learned gate/weight module;
   - exact fallback to `B0` when evidence is insufficient.

Learning is allowed only after a feature has shown a stable causal relationship in the
observability audit. A learned model should predict residual weights, uncertainty, or a
small residual, not rediscover the complete cross-shot Boundary from scratch.

### Phase C: Main-line validation

Run, in order:

1. GT-ID WHERE isolation;
2. frozen automatic-ID end-to-end evaluation;
3. single- and multi-human routing;
4. multi-cut propagation;
5. no-cut parity;
6. runtime and memory measurement;
7. qualitative worst-case inspection.

## 8. Required Experiment Record

Every experiment entry must contain:

```text
ID and date
hypothesis
causal inputs
implementation and command
development/frozen split
frozen B0 artifact/checkpoint
hyperparameters fixed before evaluation
metrics and paired deltas
worst failures
decision: reject / revise / promote
next hypothesis implied by the result
```

Machine-readable results go under:

```text
output/v14/fine_alignment_research/<experiment_id>/
```

Human-readable cumulative results remain in this document's experiment ledger or a linked
result note in `versions/v14/docs/`.

## 9. Existing Negative Evidence

| Experiment family | Result | Decision |
|---|---|---|
| old Phase-2 uniform multi-human Boundary | overwrites good `B0`; camera and human means worsen | reject as final Boundary |
| direct human-root translation residual | absorbs Human3R camera-relative depth bias; worsens camera composite on 97.2% of cuts | reject |
| current full per-candidate residual | worse than `B0` on camera and human metrics | reject |
| current rotation-only mean | improves a minority but worsens aggregate and has no reliable acceptance rule | reject in current form |
| shared shot scale | modest mean human gain but inconsistent cues and unsafe tails | keep only as ablation |
| token scale readout | capture-dependent and fails cross-sequence generalization | reject as current main route |

These failures reject specific evidence equations and unconditional composition rules. They
do not reject fine alignment itself.

## 10. Initial Hypothesis Order

1. Audit the signed GT residual after `B0` and quantify per-axis headroom.
2. Test whether a reliable rotation subset exists using torso/body orientation plus a
   frozen acceptance score.
3. Test whether background feature correspondences provide signed lateral/depth residual
   after `B0`, with repeated-structure guards.
4. Test whether multiple independent weak cues agree on a useful subset of cuts.
5. Build a bounded robust residual solver for that subset, otherwise return exact `B0`.
6. Only if analytic reliability is insufficient, train a small confidence/weight model on
   controlled data and evaluate on capture-disjoint frozen sequences.

## 11. Stop Condition

Do not stop because one hypothesis fails. Continue by using its failure mode to eliminate
an assumption or isolate a missing observable.

Stop successfully only when at least one candidate passes Section 6 and has:

- reproducible code and command;
- automated tests;
- frozen evaluation artifacts;
- a clear architecture description;
- a documented domain of validity and fallback behavior.

If all accessible causal evidence is shown not to identify the remaining residual, the
task may only be declared blocked after documenting that non-identifiability with oracle
and cross-source experiments. It must not be declared complete merely because `B0` is the
strongest tested baseline.

## 12. Experiment Ledger

New experiments are appended here chronologically. Do not delete failed entries.

| ID | Hypothesis | Split | Result | Decision |
|---|---|---|---|---|
| historical-root-translation | predicted roots can directly correct `B0` translation | 180 cuts | camera composite worsens on 97.2%; root also worsens | reject |
| historical-uniform-multi | old full multi-human Boundary can replace `B0` | 180 cuts | composite `0.713` vs `0.354` | reject |
| historical-shot-scale | one shared scalar explains remaining error | 180 cuts | mean root improves, tails/cue agreement unsafe | reject as main line |
| dev-residual-observability-20260730 | signed GT residual around `B0` identifies which fine components have headroom | `three`, 41 cuts | residual is mainly local depth (`du_z` P95 `0.445 m`); oracle x improves camera and root, but oracle z/full camera improves camera while worsening Human3R root | retain decomposition; camera gauge and local-human depth bias must be handled separately |
| dev-human-torso-20260730 | torso consensus predicts a useful rotation residual | `three`, 41 cuts | camera R `7.059 deg` vs B0 `4.135 deg`; root `0.400 m` vs `0.315 m` | reject direct torso rotation |
| dev-scene-icp-20260730 | Human3R pointmaps directly register the two shots | `three`, 41 cuts | full ICP composite `0.455` on 34 cases; mutual-translation composite `0.396` on 30 cases; improve rates only `5.9%` and `3.3%` | reject raw NN/ICP on Human3R sparse scene points |
| dev-sift-unbounded-20260730 | masked SIFT + Essential pose supplies the missing explicit cross-shot transform | `three`, 37/41 valid | camera R `53.59 deg`, composite `1.318`, root `1.868 m` | reject unbounded SIFT pose |
| dev-sift-bounded-rotation-20260730 | B0 agreement plus a tiny SIFT rotation residual is safe | `three`, 37/41 valid | 1 deg cap: composite `0.3283` vs B0 `0.3326`, but root `0.3388` vs `0.3358`; gain is small and fails the main-line material-gain rule | retain only as weak rotation cue |
| dev-sift-bounded-direction-20260730 | Essential translation direction can refine B0 while retaining B0 baseline magnitude | `three`, 37/41 valid | 2 deg cap: composite `0.3384` vs `0.3326`, root `0.3467` vs `0.3358`; larger caps diverge rapidly | reject SIFT translation direction; move to learned correspondence |
| dev-da3-shared-pose-20260730 | an independent any-view model can observe the small residual around B0 without absorbing Human3R root-depth bias | `three`, 41 cuts | frozen safe candidate: camera T `0.2558 -> 0.1948 m`, R `4.135 -> 1.604 deg`, composite `0.3385 -> 0.2269`, P95 `0.5455 -> 0.4702`, root `0.3152 -> 0.2658 m`; camera composite improves on 41/41; catastrophic remains 0 | promote and freeze for untouched evaluation |
| frozen-da3-dance-box-20260730 | the development-frozen candidate generalizes without tuning | untouched `dance+box`, 139 cuts | camera T `0.2831 -> 0.1747 m`, R `3.762 -> 1.458 deg`, composite `0.3583 -> 0.2039`, P95 `0.5775 -> 0.4399`, root `0.4798 -> 0.4014 m`; composite improves on 139/139; catastrophic `0 -> 0` | pass frozen acceptance rules; promote as main line |
| four-source-da3-safe-20260730 | the frozen gate selects a reliable subset under AvatarReX/THuman/MVHuman source shift and safely rejects non-coarse B0 | existing controlled AABB source-diversity set, 180 cuts; no parameter changes | gate accepts 44/180; on accepted cases composite `0.5311 -> 0.3575` with 42/44 improved and head proxy `0.5885 -> 0.4386`; rejected 136/136 return bit-exact B0; overall catastrophic `107 -> 106`; current B0 is not coarse on MVHuman (rotation mean `86-90 deg`), so all MVHuman cases correctly reject | retain method; record that qualified B0 is a hard precondition and current single-event V14 checkpoint is not a broad four-source B0 |
| runtime-formalization-20260730 | the selected probe can be converted to a GT-free deployable callable without numerical drift | cached 180-cut MultiHuman parity + unit tests | formal runtime differs from cached probe Boundary by at most `1.11e-16`; invalid/missing/conflicting DA3 cues and runtime failure return bit-exact B0; 20 tests pass | complete |
| person-ray-anchor-20260731 | after freezing the DA3 camera, the remaining per-person error can be corrected mainly along the camera ray using a causal history anchor | `three+dance+box`, 400 person cases | current DA3 root/joint/vertex `0.3821/0.4069/0.4015`; best non-oracle history estimator `0.3750/0.3992/0.3943`, only `0.0071 m` root gain, `33.5%` improve rate, `10.8%` harmed by more than 5 cm, `43.2%` coverage; GT-ray oracle reaches `0.1600/0.2233/0.2062` with 100% improvement | retain the ray decomposition and oracle headroom; reject the current history-anchor estimator as a main line |
| person-bbox-pointmap-20260731 | DA3 range change inside a conservative person bbox core predicts the person's root-depth change | first 3 `three` cuts, 9 people | accepted 4/9 with camera bit-exact; B0 root/joint/vertex `0.3209/0.3723/0.3291`, translation proposal `0.4207/0.4691/0.4278`; accepted residual sign agreement with GT-ray is `0%`, Pearson `-0.833`; oracle is `0.1031/0.1814/0.1306` | reject bbox-core DA3 surface-change estimator; retain the negative result and oracle |
| person-mesh-depth-20260731 | an exact predicted-mesh triangle z-buffer can isolate same-surface DA3 depth and transfer it to person root depth | first 3 `three` cuts, 9 people | mesh/pixel mapping succeeds on 3/3 cuts with max bbox reprojection error `2.14e-05 px`, but only 1/9 people pass the depth gates; B0 `0.3209/0.3723/0.3291` becomes `0.3313/0.3796/0.3385`; the accepted residual is `-0.10657 m` versus GT-ray `+0.14929 m` | reject the current mesh-surface estimator; retain mapping/rasterization only as diagnostic infrastructure |
| egohumans-da3-domain-shift-20260731 | the frozen MultiHuman `da3_safe` gate generalizes to overlapping EgoHumans data without tuning | three 15-frame streams, 6 cuts, 30 post-cut observations, 80/90 matched person instances | gate accepts 6/6, yet camera T `0.3968 -> 0.4046 m`, root `0.3462 -> 0.3721 m`, joint `0.3478 -> 0.3602 m`, and vertex `0.3500 -> 0.3608 m`; camera R changes `4.143 -> 4.111 deg`; only T/R/joint improve on `4/6`, `3/6`, and `2/6` cuts | reject the current gate as domain-general; retain DA3 only as camera refinement in a validated image/domain protocol |
| multithumbs-protocol-audit-20260731 | overlapping EgoHumans data allows an immediate official comparison with Multi-THuMBS | paper audit plus three hand-selected local 15-frame raw-Human3R streams from one capture | paper EgoHumans reference is W/WA/MPJPE/MPVPE/Accel `279.0/166.0/228.3/262.2/27.3`, ATE `0.7`, IDs `0.97`; provisional local raw Human3R is `1088.3/405.1/109.3/130.0/52.49`, ATE `1.848`, IDs/stream `4.00`; split, cut list, visibility, miss/FP handling, aggregation, supplementary protocol, and official code are unavailable | retain evaluator and formulas as provisional diagnostics; no winner/loser claim is valid yet |

## 13. Historical Frozen Camera Candidate Before Holdout Access

> Scope correction (2026-07-31): this section records a successful shared **camera/shot
> Boundary** experiment. It is not a completed person fine-alignment method.

The development-selected candidate is now frozen as `da3_safe`. No threshold or cap
may be changed after inspecting `dance` or `box`.

```text
last pre-cut RGB + first post-cut RGB
-> frozen DA3-Base in forward and reverse image order
-> two independent relative camera poses in a DA3 shared space
-> map both pose estimates into the Human3R pre-shot world using the raw pre/post poses
-> forward/reverse SO(3) rotation consensus
-> forward/reverse camera-baseline direction consensus
-> discard DA3's arbitrary translation scale
-> retain the exact B0 camera-baseline magnitude
-> cap the B0-centered rotation residual at 3 degrees
-> cap the B0-centered baseline-direction change at 5 degrees
-> construct one shared post-shot Boundary for camera, pointmap, and every human
```

The deployment-only acceptance gate is:

```text
forward/reverse rotation spread <= 5 degrees
forward/reverse direction spread <= 5 degrees
DA3-vs-B0 rotation proposal <= 15 degrees
DA3-vs-B0 direction proposal <= 30 degrees
all values finite and both DA3 poses valid
```

If any check fails, the output is bit-exact frozen `B0`. The broad gate accepts all
41 development cuts; safety comes primarily from the 3/5-degree component trust regions,
while the gate rejects gross model failures.

Development metrics:

| Metric | B0 | `da3_safe` | Relative result |
|---|---:|---:|---:|
| camera translation mean | 0.2558 m | 0.1948 m | -23.8% |
| camera rotation mean | 4.135 deg | 1.604 deg | -61.2% |
| camera composite mean | 0.3385 | 0.2269 | -33.0% |
| camera composite P95 | 0.5455 | 0.4702 | -13.8% |
| human root mean | 0.3152 m | 0.2658 m | -15.7% |
| human root P95 | 0.5351 m | 0.4169 m | -22.1% |
| pairwise distance mean | 0.0845 m | 0.0845 m | exactly rigid/invariant |
| pairwise vector mean | 0.1314 m | 0.1290 m | improved |
| catastrophic count | 0/41 | 0/41 | unchanged |

Reproduction:

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python \
  versions/v14/probe_b0_da3_shared_pose.py \
  --sequences three \
  --device cuda:0 \
  --output_dir output/v14/fine_alignment_research/da3_shared_pose_three_dev

.venv/bin/python -m pytest -q \
  tests/test_v14_da3_shared_pose.py \
  tests/test_v14_b0_identity_matching.py \
  tests/test_v14_segment_boundary.py
```

The selected implementation and full per-cut artifact are:

```text
versions/v14/probe_b0_da3_shared_pose.py
output/v14/fine_alignment_research/da3_shared_pose_three_dev/
```

## 14. Historical Untouched Frozen Camera Evaluation

The following numbers remain valid for the exact MultiHuman protocol on which they were
measured. They demonstrate shared camera-boundary generalization, but they do not test or
remove camera-relative Human3R person scale/root/structure bias.

After the Section 13 configuration was frozen, it was run once on `dance` and `box`.
No cap, gate threshold, image mode, or model weight was changed after observing the result.

| Split | N | Method | Camera T | Camera R | Composite | P95 composite | Human root | Catastrophic |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dance | 61 | B0 | 0.2951 | 3.453 | 0.3641 | -- | 0.3827 | 0 |
| dance | 61 | `da3_safe` | 0.1773 | 1.239 | 0.2021 | -- | 0.3256 | 0 |
| box | 78 | B0 | 0.2736 | 4.004 | 0.3537 | -- | 0.5557 | 0 |
| box | 78 | `da3_safe` | 0.1727 | 1.630 | 0.2053 | -- | 0.4606 | 0 |
| combined | 139 | B0 | 0.2831 | 3.762 | 0.3583 | 0.5775 | 0.4798 | 0 |
| combined | 139 | `da3_safe` | 0.1747 | 1.458 | 0.2039 | 0.4399 | 0.4014 | 0 |

Frozen paired results:

- camera composite improves on `139/139` cuts;
- human root improves on `83.45%` of cuts;
- no catastrophic case is introduced;
- `dance` composite improves by `44.5%` and `box` by `42.0%`;
- pairwise distance is invariant up to floating-point noise, confirming one shared rigid
  transform rather than per-human fitting.

Across development plus frozen MultiHuman (`three+dance+box`, 180 cuts), the final result is:

| Metric | B0 | `da3_safe` | Relative result |
|---|---:|---:|---:|
| camera T mean | 0.2768 m | 0.1793 m | -35.2% |
| camera R mean | 3.847 deg | 1.491 deg | -61.2% |
| camera composite mean | 0.3538 | 0.2091 | -40.9% |
| camera composite P95 | 0.5767 | 0.4536 | -21.3% |
| human root mean | 0.4423 m | 0.3705 m | -16.2% |
| human root P95 | 1.2212 m | 1.1469 m | -6.1% |
| pairwise distance mean | 0.0728484 m | 0.0728484 m | rigid invariant |
| pairwise vector mean | 0.2684 m | 0.2498 m | -6.9% |
| catastrophic | 0/180 | 0/180 | unchanged |

Artifact and reproduction:

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python \
  versions/v14/probe_b0_da3_shared_pose.py \
  --sequences dance box \
  --image_modes full \
  --device cuda:0 \
  --output_dir output/v14/fine_alignment_research/da3_shared_pose_dance_box_frozen
```

## 15. Four-Source Source-Diversity Audit

The already selected 180 AABB records contain `48 AvatarReX`, `48 THuman`,
`48 MVHuman100`, and `36 MVHuman200` cuts. This audit uses the same frozen gate and caps.
It is a mechanism/source-shift diagnostic, not a replacement for the untouched
MultiHuman holdout: the active V14 checkpoint was fine-tuned on one AvatarReX event and
is not a broadly trained four-source B0.

| Split | N | Gate | B0 composite | `da3_safe` composite | B0 head | Fine head | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 180 | 24.4% | 2.2875 | 2.2451 | 1.4555 | 1.4189 | 107 -> 106 |
| AvatarReX | 48 | 10.4% | 2.4905 | 2.4794 | 1.1671 | 1.1640 | 27 -> 27 |
| THuman | 48 | 81.2% | 0.8461 | 0.6980 | 0.7510 | 0.6167 | 8 -> 7 |
| MVHuman100 | 48 | 0.0% | 2.9257 | 2.9257 | 2.4173 | 2.4173 | 43 -> 43 |
| MVHuman200 | 36 | 0.0% | 3.0879 | 3.0879 | 1.4972 | 1.4972 | 29 -> 29 |

The overall mean is dominated by an upstream failure: current B0 camera rotation error is
`90.5 deg` on MVHuman100 and `86.3 deg` on MVHuman200. Those are not coarse-alignment
residuals and cannot be repaired by a bounded 3/5-degree fine stage. The frozen gate
rejects every MVHuman case instead of applying a misleading small residual.

The accepted 44-case subset directly tests whether the gate identifies the valid regime:

| Metric on accepted cases | B0 | `da3_safe` | Improve rate |
|---|---:|---:|---:|
| camera T | 0.4101 m | 0.2810 m | 90.9% |
| camera R | 6.050 deg | 3.823 deg | 97.7% |
| camera composite | 0.5311 | 0.3575 | 95.5% (42/44) |
| human head proxy | 0.5885 m | 0.4386 m | 88.6% |

All `136/136` rejected cases are numerically bit-exact B0. The two accepted composite
regressions are `+0.0688` and `+0.0305`; they are recorded and were not used to retune the
frozen gate.

Reproduction:

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python \
  versions/v14/probe_b0_da3_four_source.py \
  --device cuda:0 \
  --output_dir output/v14/fine_alignment_research/da3_shared_pose_four_source_frozen
```

## 16. Historical DA3 Camera Runtime and Safety Verification

This runtime is complete for the bounded shared camera residual only. Person-local
refinement has no promoted runtime implementation yet.

The evaluator-independent implementation is:

```text
versions/v14/b0_da3_fine_alignment.py
```

It provides:

```python
DA3FineAligner.refine_images(...)
refine_b0_with_da3(...)
apply_boundary_to_pose(...)
apply_boundary_to_points(...)
```

The first callable consumes two RGB images and frozen runtime poses; the second consumes
precomputed bidirectional DA3 poses. Neither imports GT, dataset, identity, or evaluator
code. Missing DA3 output, NaN, degenerate baseline, forward/reverse disagreement, prior
disagreement, or an external DA3 runtime exception returns exact B0.

Cached parity over all 180 MultiHuman cases gives:

```text
formal-runtime Boundary vs frozen-probe Boundary:
max absolute matrix difference = 1.11e-16
accepted = 180/180
```

The final test command reports `21 passed` and verifies gauge invariance, cap bounds,
baseline-length preservation, shared pose/point application, and exact fallback.

Two DA3-Base passes cost about `0.17 s` total after warm-up on one NVIDIA L20 at
`process_res=504` (`0.086/0.083 s` forward/reverse means in the four-source run).

## 17. Revised Decision: Camera Refinement Retained, Person Refinement Ongoing

On 2026-07-30, Section 11 was considered satisfied under the narrower assumption that
fine alignment meant estimating one better shared Boundary. The 2026-07-31 evidence
revises that conclusion.

`da3_safe` is retained only as a bounded, causal **camera refinement** in the original
validated MultiHuman domain because it:

- materially improves the shared camera Boundary on development and untouched holdout;
- preserves camera/scene/human shared rigidity by construction;
- has a deployable callable, exact B0 fallback, tests, and a recorded failure boundary.

It is no longer called the complete fine-alignment solution because:

- applying one SE(3) to the camera and every human preserves their relative geometry, so
  it cannot repair Human3R's camera-local person root/depth/scale/pose bias;
- the MultiHuman human-root gain can coexist with visibly misaligned people and therefore
  is insufficient evidence of correct person structure;
- on raw fisheye EgoHumans, the frozen gate accepts `6/6` cuts while aggregate camera
  translation and root/joint/vertex errors worsen;
- all three tested non-oracle person-depth estimators remain too weak or predict the wrong
  residual sign.

The corrected system status is:

```text
qualified B0 coarse camera Boundary
-> optional da3_safe camera refinement only in a validated domain
-> freeze camera and shared scene coordinates
-> person association
-> gated person-local root/depth/orientation refinement (not solved yet)
-> exact per-person fallback when evidence is unreliable
```

The complete Section 11 stop condition remains unsatisfied for person fine alignment.
Current B0/DA3 camera work is a useful frozen foundation, not the final answer.

## 18. 2026-07-31 Person-Structure and Multi-THuMBS Update

### 18.1 What the latest experiments establish

After the camera is frozen, the dominant remaining person error has a large component
along the person's viewing ray. Across 400 person cases, current DA3-aligned Human3R has
root/joint/vertex errors of `0.3821/0.4069/0.4015 m`. If GT supplies only the correct
per-person displacement along that ray, the errors fall to
`0.1600/0.2233/0.2062 m`. This is strong evidence that person-local depth is an important
missing variable; it is not a deployable method because GT provides the oracle residual.

The causal history anchor captures too little of that headroom. Its best safe-looking
variant lowers mean root error by only `7.1 mm`, improves 33.5% of people, covers 43.2%,
and harms 10.8% by more than 5 cm. It is rejected as a main-line estimator.

Two explicit DA3 surface-depth estimators were then tested:

1. A bbox-core estimator used the DA3 pre/post surface-range change inside a conservative
   torso/pelvis box. It accepted 4/9 people but worsened root error by `99.8 mm`; its
   residual direction had 0% sign agreement and `-0.833` correlation with the oracle.
2. A triangle z-buffer estimator mapped Human3R meshes into DA3 pixels and retained only
   same-surface residuals. The coordinate bridge is correct to `2.14e-05 px`, but the
   depth cue accepted only 1/9 people and chose the opposite sign from the oracle. Most
   people fail the surface-residual dispersion gate; one has too few same-surface pixels.

Thus the geometry plumbing is available, but raw DA3 person-region depth change is not a
reliable root-depth observable in the current formulation.

### 18.2 EgoHumans domain result

EgoHumans is the only immediately runnable local dataset that overlaps the datasets named
by Multi-THuMBS. The local material is one capture represented by three manually assembled
15-frame streams, with two cuts per stream. It is useful for diagnosis but is not the
paper's unpublished benchmark split.

On 30 post-cut observations, B0 and B0+DA3 give:

| Method | Camera T | Camera R | Root | MPJPE-24 | MPVPE-6890 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.3968 m | 4.143 deg | 0.3462 m | 0.3478 m | 0.3500 m |
| B0+`da3_safe` | 0.4046 m | 4.111 deg | 0.3721 m | 0.3602 m | 0.3608 m |

The DA3 gate accepts every cut even though four aggregate metrics worsen. Raw fisheye
domain shift therefore invalidates the current acceptance rule. A future DA3 path needs a
predeclared undistortion/image protocol and a domain-aware rejection test before it can be
used on EgoHumans; that work is not complete.

### 18.3 Multi-THuMBS metrics and reference targets

The paper evaluates complementary failure modes rather than one combined score:

- **W-MPJPE:** fit one Sim(3) from the first two frames of each GT identity track, apply it
  to the whole predicted world trajectory, and average 3D joint error in millimetres.
- **WA-MPJPE:** fit one Sim(3) using the full identity trajectory, then average world joint
  error. This is more forgiving about global trajectory gauge than W-MPJPE.
- **MPJPE / MPVPE:** in camera coordinates, independently pelvis-align prediction and GT
  per person-frame, then average error over 24 joints or 6890 vertices. A shared world
  translation cancels here, so these expose person-local pose/shape rather than shot
  alignment.
- **Accel:** compare predicted and GT pelvis-centred joint second differences, scaled by
  `fps^2`; this measures temporal instability.
- **ATE:** Sim(3)-aligned camera-centre trajectory translation RMSE in metres in the local
  provisional evaluator.
- **IDs:** after evaluator-side per-frame GT association, count native predicted track-ID
  changes along a stable GT identity.

The paper's EgoHumans reference values are:

| W-MPJPE | WA-MPJPE | MPJPE | MPVPE | Accel | ATE | IDs |
|---:|---:|---:|---:|---:|---:|---:|
| 279.0 mm | 166.0 mm | 228.3 mm | 262.2 mm | 27.3 | 0.7 m | 0.97 |

A provisional evaluation of the three local **raw Human3R** streams, without B0 or DA3,
produces:

| W-MPJPE | WA-MPJPE | MPJPE | MPVPE | Accel | ATE | IDs / stream |
|---:|---:|---:|---:|---:|---:|---:|
| 1088.3 mm | 405.1 mm | 109.3 mm | 130.0 mm | 52.49 m/s^2 | 1.848 m | 4.00 |

These two rows are not an official leaderboard comparison. The local run uses three
hand-selected streams from one capture, raw shot-reset Human3R caches, repeated timestamps
at each cut, evaluator-side GT identity matching, and matched-observation-only pose
aggregation. The paper does not release its exact split, cut list, visibility/miss rules,
aggregation, ATE alignment, IDs convention, supplementary protocol, or evaluation code.
Lower local pelvis-aligned MPJPE/MPVPE therefore does not establish a win, while the much
worse W/WA/ATE/IDs values still provide useful evidence of global trajectory and identity
gaps.

The reproducible audit is recorded in:

```text
versions/v14/docs/V14_MULTITHUMBS_AUDIT_AND_EGOHUMANS_BASELINE_20260731.md
output/v14/fine_alignment_research/multithumbs_protocol/README.md
output/v14/fine_alignment_research/multithumbs_protocol/human3r_raw_egohumans_provisional.json
```

### 18.4 Next main hypothesis, not yet a result

Multi-THuMBS supports the same diagnosis: after putting shots in a shared scene, it still
optimizes each person's root translation and global orientation using 2D joints,
silhouette, and scene depth. The V14-compatible causal experiment should therefore be:

```text
last pre-cut frame + first post-cut frame
-> qualified B0 (and optional domain-qualified DA3 camera residual)
-> freeze the shared camera Boundary
-> associate each person across the cut with an explicit confidence/fallback
-> obtain a person mask, 2D joints, current Human3R mesh, and independent scene pointmap
-> optimize only bounded per-person ray depth, small tangential root residual,
   and global orientation
-> robust 2D-joint reprojection + silhouette containment + visible-surface depth losses
-> reject on cue disagreement, low visibility, high depth dispersion, or implausible scale
-> emit corrected persons in the fixed camera/world frame; otherwise emit exact B0 person
```

Unlike Multi-THuMBS's offline 500+1500-iteration joint camera/human optimization and full-
sequence smoothing, the first V14 test must keep the camera frozen and use only the causal
boundary observations. Its required success criteria are predeclared improvement in
W-MPJPE/WA-MPJPE and world root error without degrading MPJPE/MPVPE, Accel, ATE, IDs,
coverage, or camera metrics. Until that experiment passes on a development split and a
frozen cross-domain split, it remains a high-priority hypothesis rather than a solved
fine-alignment method.

## 19. 2026-07-31 Final Update: BRTC-LC Promoted for Root/Layout Fine Alignment

Section 18.4 is now superseded for the root/layout component. The successful candidate is:

```text
frozen B0 camera
-> automatic root+torso+joints association
-> last-pre / first-post five-core-joint ray triangulation
-> observable ray-gap/parallax/MAD gate
-> shared group translation + pre-layout-selected person residual
-> rigid post-person translations only
-> exact fallback for rejected/unmatched people
```

The method is named **Camera-Frozen Boundary Ray Triangulation with Layout Consensus
(BRTC-LC)**. It uses no DA3, GT, source ID, future post frame, or learned residual head.

### 19.1 Experiment record

| ID | Split | Result | Decision |
|---|---|---|---|
| exact-camera-ray-dev | four-source offset0, 200 people | root `0.8689 -> 0.0871`, 90.0% gain, 2.0% harm >5cm | mechanism pass |
| exact-camera-ray-confirm | four-source offset50, 200 people | root `0.8991 -> 0.0978`, 89.1% gain, 3.5% harm >5cm | controlled confirmation; not asset-disjoint |
| B0-independent-dev | `three` offset0, 41 cuts/122 people | root `0.3789 -> 0.2331`; pair-vector `0.3331 -> 0.2933` | retain depth evidence |
| B0-independent-frozen | `dance+box`, 139 cuts/278 people | root `0.4798 -> 0.2912`, but pair-vector `0.3088 -> 0.3894` | **reject independent correction; layout failure** |
| BRTC-LC-dev | `three` offset0, 41 cuts/122 people | root `0.3789 -> 0.2251` (40.6%); pair-vector `0.3331 -> 0.2605` (21.8%); harm 6.6% | freeze consensus rule |
| BRTC-LC-confirm | newly inferred `three` offset1, 42 cuts/125 people | root `0.3779 -> 0.2314` (38.8%); joint `0.4117 -> 0.2745`; vertex `0.3891 -> 0.2525`; pair-vector `0.3297 -> 0.2588` (21.5%); harm 7.2%; auto-ID 100%; camera delta 0 | **promote root/layout main line** |
| BRTC-LC-dance-box-posthoc | already-consumed `dance+box` | root gain 45.0%; pair-vector gain 11.2%; harm 5.0% | supporting evidence only, not pristine frozen |

### 19.2 Why layout consensus is required

Independent person triangulation predicts the signed direction correctly on about 90% of
accepted frozen people, but independent noise changes pairwise human layout. BRTC-LC writes
each shift as:

```text
s_i = group_median_shift + lambda * (individual_shift_i - group_median_shift)
```

`lambda` is chosen from the frozen set `{0,.25,.5,.75,1}` by minimizing disagreement
between predicted corrected post pairwise-root vectors and last-pre predicted pairwise-root
vectors. This uses predictions only. Rejected people remain exact B0.

### 19.3 Runtime and parity

The evaluator-independent callable is:

```text
versions/v14/b0_person_triangulation.py::refine_matched_people
```

It has no dataset, GT, evaluator, or DA3 imports. Four unit tests pass, covering ray
intersection, camera immutability, exact fallback/unmatched behavior, and gauge
equivariance. Runtime/probe final shifts agree over all 42 confirmation cuts to
`1.10e-15 m` maximum absolute difference.

### 19.4 Revised stop condition

The task has a positive answer for **person root/depth and multi-human layout refinement**:
BRTC-LC is the current main line around frozen B0. The broader research program is not
finished because rigid translations cannot fix pelvis-aligned pose/shape or global
orientation, and no official Multi-THuMBS split/evaluator is available locally.

The detailed method, failure history, metrics, limitations, Multi-THuMBS relationship, and
reproduction commands are recorded in:

```text
versions/v14/docs/V14_B0_TWO_VIEW_TRIANGULATION_FINAL_20260731.md
```

## 20. 2026-08-01 Frozen BRTC-LC v1 and EgoHumans Current-Method Result

The current root/layout method is now frozen independently of later experiments.

```text
commit: 1d77b5d
tag:    movie3r-v14-brtc-lc-v1
manifest:
  versions/v14/frozen/BRTC_LC_V1_20260801.json
checkpoint:
  checkpoints/v14_brtc_lc_v1_b0/checkpoint-best.pth
sha256:
  8379243216775adbc886d00e6f93b6492f7d8f1bd67adb4e8ad6fbdd84e47123
```

It remains the default runtime. The later experiments do not modify its B0 camera,
association, ray evidence, group/layout consensus, or checkpoint.

The same current B0+BRTC-LC method—not raw Human3R—was replayed on the available local
EgoHumans subset: `001_legoassemble`, three manually assembled 15-frame streams and six
cuts. The local Multi-THuMBS-style provisional result is:

| W | WA | Fixed root | Fixed joint | Fixed vertex | Pair distance | Pair vector | Root Accel | ATE | IDs/stream |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 314.059 | 202.461 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 0.119 | 1.00 |

The paper reports EgoHumans W/WA `279.0/166.0 mm`, so the local current method is still
approximately `+35.1/+36.5 mm` worse. This is only a provisional gap: the official split,
supplementary protocol, visibility treatment, evaluator, and complete Accel definition are
not public. The result is nevertheless useful because it tests the actual frozen method and
shows that global trajectory/person structure remains behind the paper reference.

The viewers on ports `8091/8092` were stopped. All experiments in this section and below
replay caches on CPU and write only under the `/data` Movie3R workspace.

## 21. 2026-08-01 Search Round: What Was Tried and Rejected

The search kept the hard constraints: no new pretrained vision model, zero future post
frames, B0 camera bit-exact, rejected/unmatched people safe, and complete failure records.

| Candidate | Positive evidence | Failure | Decision |
|---|---|---|---|
| Huber-IRLS BRTC | slight single-person gain | layout/coverage unstable | NO-GO |
| fixed group damping | lower harm and some spatial metrics | pair distance regresses | NO-GO |
| causal temporal ray bundle | lower correction jitter/Accel | spatial error slightly worse; below B0 Accel ceiling not reached | stabilizer only |
| linear completeness | Ego W/WA/root/harm improve | independent variable-visibility spatial errors regress | NO-GO |
| soft incomplete scale=0.9 | three/dance improve | box joint/vertex regress | NO-GO |
| joint ray-layout least squares | fits pre layout | dance/box pair distance overfits badly | NO-GO |
| FAGD-0.9 | same-visibility root gain; pair layout exact | equal-count replacements and Ego Accel/WA gate failure | NO-GO |
| acceleration-gated FAGD | MultiHuman gain retained | Ego gate acts 0/6; no improvement | NO-GO |
| geometry-only identity dustbin | dev rejects all four wrong edges | dance rejects correct IDs; box accepts confident full swap | NO-GO |
| angular-safe group damping | offset1/box improve; population changes exact v1 | dance root/joint regress; Ego exact v1/no action | NO-GO |
| learned group-alpha selector | grouped CV and offset1 root improve | pair layout, box, Ego W/WA and Accel conflict | NO-GO |
| shared/group SO(3) Kabsch | all MultiHuman joint/vertex means improve | Ego fixed joint/vertex `+0.177/+0.006 mm`; weaker than individual | NO-GO |
| person-local body scale | bone-length consistency is observable; offset1/box improve | dance vertex/Accel and Ego fixed joint/vertex regress | NO-GO |
| individual Kabsch + body scale | net better than BRTC on Ego | scale is a negative increment over Kabsch on dance/Ego fixed mesh | NO-GO as second candidate |
| two-frame group tangent translation | three-offset1/box improve; layout exact | dance root `+2.096 mm`; coherent motion is confounded with shared bias | NO-GO |
| timestamp-aware velocity group tangent | three-offset1 and box improve strongly | every nonzero dance offset moves root in the wrong direction | NO-GO |
| TORSO8 / observable orientation selector | small aggregate orientation gain | no configuration passes every timestamp and centred/raw mesh gate | NO-GO |
| person-to-scene contact from current cache | physically independent degree of freedom | Ego has no scene fields; MultiHuman deletes foot-local support | NO-GO on current cache evidence |

The repeated pattern is now explicit:

1. BRTC shared translation is useful but slightly miscalibrated in some cases.
2. Counts, action size, angular ray change, predicted acceleration, layout objective, and a
   small observable classifier cannot reliably determine a better scalar group amplitude.
3. Current root/torso/joint geometry can be confidently wrong about identity; threshold
   tuning cannot replace an independent appearance/identity cue.
4. Body scale flicker exists, but making predicted bone lengths continuous is not equivalent
   to lowering complete world-mesh error.
5. A new module should add a missing geometric degree of freedom, not repeatedly rescale the
   same BRTC translation.

The detailed negative reports are all under `versions/v14/docs/` and were committed in
logical groups. They are retained to prevent repeating the same hypotheses.

## 22. Qualified Candidate: Person-Local Global-Orientation Kabsch

The one retained candidate adds the global-orientation degree of freedom that BRTC cannot
represent:

```text
frozen B0 camera
-> frozen BRTC-LC v1 translation
-> BRTC-accepted match only
-> last-pre/current-post root-centred hips+shoulders
-> bounded per-person Kabsch SO(3)
-> rotate joints/vertices around the corrected native root
-> propagate the same rotation causally through the current shot
```

Frozen policy:

```text
maximum applied angle = 25 degrees
rotation fraction      = 0.5
minimum predicted torso-residual improvement = 0
```

It uses no RGB encoder, DA3, GT-side inference, dataset ID, future post frame, or new
pretrained model. Rejected and unmatched people are exact B0; the camera and native Human3R
root are exact frozen BRTC.

Frozen MultiHuman results:

| Split | Joint BRTC | Joint Kabsch | Vertex BRTC | Vertex Kabsch | Root/layout |
|---|---:|---:|---:|---:|---|
| three offset1 | .274493 | **.271315** | .252451 | **.250248** | exact BRTC |
| dance | .177804 | **.168764** | .152914 | **.148234** | exact BRTC |
| box | .421610 | **.418583** | .434528 | **.429938** | exact BRTC |

EgoHumans provisional result:

| Method | W | WA | pelvis MPJPE | pelvis MPVPE | Fixed joint | Fixed vertex | Root Accel | Joint Accel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 314.059 | 202.461 | 109.266 | 129.960 | 384.729 | 385.238 | 116.014 | 125.270 |
| + individual Kabsch | **312.769** | **200.029** | **101.526** | **119.928** | **383.933** | **383.791** | **115.698** | **123.167** |

Joint/vertex harm above 5 cm is zero on the Ego chains. Local W/WA remain
`+33.769/+34.029 mm` above the paper reference, so this is progress rather than a claimed
Multi-THuMBS win.

### 22.1 Strict versus practical decision

The predeclared zero-tolerance gate remains technically false because the Ego evaluator
maps SMPL-X vertices to SMPL and re-regresses a pelvis root. That proxy changes
`380.654 -> 380.688 mm`, a `+0.034 mm` regression. The stored native Human3R root has exactly
zero change, and mapped pelvis versus native root already differs by median `18.193 mm`.

Therefore both decisions are preserved:

```text
strict-zero evaluator decision:
  NO_GO_GLOBAL_ORIENTATION_KABSCH_EGOHUMANS

explicit 0.1 mm mapped-pelvis proxy tolerance:
  QUALIFIED_GLOBAL_ORIENTATION_KABSCH_CANDIDATE
```

The qualified candidate is frozen but is not yet the default runtime:

```text
tag: movie3r-v14-brtc-kabsch-v1-candidate
manifest:
  versions/v14/frozen/BRTC_PERSON_LOCAL_KABSCH_V1_20260801.json
runtime:
  versions/v14/b0_person_triangulation_orientation_kabsch.py
```

### 22.2 Necessary dual-state streaming contract

An implementation audit found a real causal-state issue. Feeding the already-rotated last
shot back into BRTC makes its next translation read rotated joints and can move the root by
up to `9.0556 mm` relative to frozen BRTC. Feeding only the translation reference prevents
orientation inheritance. The deployable interface therefore separates:

```text
BRTC translation state:
  unrotated frozen-v1 reference history

person orientation state:
  causally rotated history from the emitted previous shot
```

The full deployable runtime now consumes both states. MultiHuman and Ego runtime/probe
geometry and rotation parity are zero; first-frame propagation parity is at most
`8.88e-16`; rejected/unmatched geometry and orientation metadata are exact B0; camera and
native root remain exact. Twelve related tests pass.

## 23. Residual Error and the Next Research Direction

After BRTC+Kabsch, the available Ego post-shot fixed-root error is still `326.984 mm`.
Evaluator-only decomposition shows:

```text
shared root squared-error fraction: 71.6%
root after oracle per-frame shared removal: 148.279 mm

pelvis pose raw / oracle-SO3 / PA floor:
101.583 / 75.568 / 64.946 mm
```

This means the first Kabsch module attacks a real orientation component, but the largest
remaining world error is still a shared person/root component. Directly subtracting camera
error does not solve it, and the failed scalar damping experiments show that its magnitude
is not identifiable from BRTC action statistics alone.

A first explicit shared-translation probe was also completed. It projected each accepted
`last-pre -> current-post` root residual onto the post-camera ray tangent plane and applied
a robust bounded group median. The frozen dev policy improved `three offset1` and `box`, but
regressed native root on `dance` by `+2.096 mm`, so the decision is
`NO_GO_TWO_FRAME_GROUP_TANGENT`. The failure identifies the missing observable: with only
two frames, coherent human motion and a shared alignment bias are indistinguishable.

Timestamp-aware causal velocity compensation was then implemented and tested. It rebuilt
anonymous tracks from five causal pre-shot frames, extrapolated with the physical dataset
timestamp delta, and only then robustly aggregated the ray-tangent residual. The frozen
policy used `apply_when_dt_zero=false`, so all repeated-timestamp Ego cuts and offset-zero
cases were exact Kabsch fallback rather than using stream-list indices as fake time.

The branch improved three-offset1 by about `9.4/9.8/10.8 mm` root/joint/vertex and box by
`2.3/3.1/3.3 mm` relative to Kabsch, but failed dance: root regressed `6.497 mm` and vertex
regressed `1.712 mm`. Every nonzero dance delta (`1/2/4/8`) moved root in the wrong direction.
Ego was exactly Kabsch and therefore could validate fallback invariants but not the velocity
action. The frozen result is `NO_GO_VELOCITY_RESIDUAL_GROUP_TANGENT`; confirmation was not
retuned. This shows that causal velocity reduces the ambiguity in some scenes but does not
make shared root bias identifiable from human motion across captures.

A second orientation search also failed under stricter grouped-CV. TORSO8 correspondences,
a deterministic predicted-residual selector, and a mapped-pelvis pivot each had small mean
benefits in some groups but local regressions. A shallow observable TORSO4/TORSO8 selector
scanned 369 predeclared configurations with leave-one-timestamp-out evaluation; none passed
all raw and pelvis-centred joint/vertex gates. The closest selector improved the four means
by only `0.017--0.044 mm` and still regressed timestamp-1000 pelvis-centred vertex by
`0.052 mm`. It was not frozen and no held-out split was opened. The decision is
`NO_GO_SECOND_ORIENTATION_SELECTOR`; TORSO4 individual Kabsch remains the only retained
orientation candidate.

In parallel, the stronger long-term experiment should use an **independent explicit
observation already available from Human3R**, not another scalar gate:

```text
frozen B0+BRTC+qualified Kabsch
-> keep camera and native roots as separate audited states
-> obtain Human3R's own scene pointmap/depth and visible body/foot geometry
-> estimate a bounded shared person-to-scene contact/depth residual
-> require multi-person/contact/ray agreement
-> update accepted person roots only when this independent cue agrees
-> exact BRTC+Kabsch fallback otherwise
```

This does not introduce a second pretrained model: it reuses outputs of the existing
Human3R forward. The important test is whether scene contact/depth predicts the sign of the
remaining shared root residual across new sequences. If the cache lacks pointmap/contact
evidence, the correct next step is to extend the CPU geometry cache once, predeclare the
gate, and obtain new held-out boundaries—not to tune another function of already-failed
BRTC statistics.

The availability audit found that this extension is in fact required. The current Ego
cache has no scene/depth/confidence/foot-visibility field at all. MultiHuman retains only a
sparse cloud created after removing each complete person bbox plus an 8% margin and then
drops confidence and pixel coordinates. Foot-to-cloud coverage within 25 cm is only
`7.7%/6.3%/0.7%/29.7%` on three-dev/three-offset1/dance/box. Consequently a fallback-only
contact run would have zero meaningful coverage and is not a GO. The frozen decision is
`NO_GO_CURRENT_CACHE_FOR_PERSON_SCENE_CONTACT_RESIDUAL` (insufficient evidence, not a proof
that a correctly cached contact method cannot work). A reproducible audit and a compact
foot-local patch cache contract are recorded in
`V14_BRTC_PERSON_SCENE_CONTACT_AVAILABILITY_20260801.md`; no model or GPU was run for this
audit.

The earlier V11.2 control also prevents repeating a naive formulation: forcing predicted
foot contact to zero required a mean `0.515 m` root shift and moved the human reprojection by
`112.1 px` on average (`252.4 px` P95). Any future contact candidate must preserve an
observable past signed person-to-surface offset, keep the camera fixed, cap the initial
action at 30 mm, and fall back exactly when the local support is unobservable.

For identity replacements, a separate lightweight appearance descriptor from existing
Human3R/image tokens is required. Geometry-only confidence has been falsified and should
not be revisited without an independent identity cue.
