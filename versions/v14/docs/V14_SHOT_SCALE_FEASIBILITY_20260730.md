# V14 Shot-Scale Feasibility Audit

Date: 2026-07-30

## 1. Question

This audit tests two causal ways to make independently reset Human3R shots use a
consistent scale:

1. preserve or read scale information from frozen latent tokens;
2. estimate one explicit post-shot scalar and apply it with learned `B0`.

The required deployment contract is unchanged:

```text
causal
online
fixed-size state
one cut-time estimate
one shared transform for camera, pointmap, and all humans
```

The first question is not whether an arbitrary scale can reduce one metric. It is whether
camera, background, human roots, and body dimensions are explained by the same scalar.

## 2. Completed Experiment Checklist

- [x] Reuse the frozen 180-cut V14 protocol and strict GT-ID reassignment.
- [x] Measure body-size, camera-relative depth, multi-human layout, and background scale.
- [x] Compare their cross-shot ratios and test the shared-scale assumption.
- [x] Compute GT metric-relative and GT root-optimal scale diagnostics.
- [x] Test causal explicit body, layout, scene, and fused scale cues.
- [x] Apply every scale uniformly around the first post-cut camera center.
- [x] Combine unit, deployable explicit, and Oracle scales with rotation/translation/full
  human refinement.
- [x] Extract frozen pose, scene, human, CUT3R, Multi-HMR, and fused-prompt tokens.
- [x] Train a diagnostic linear probe on `three` and evaluate directly on `dance/box`.
- [x] Audit ridge regularization sensitivity.

## 3. Protocol

The evaluation contains the same 180 cuts used by the B0 anchor-conflict audit:

```text
three: 41
dance: 61
box: 78
```

GT is used only to measure scale and score geometry. It is not an input to the deployable
scale cues.

For a post-shot scale `s`, the complete post shot is scaled around its first camera
center. Consequently:

- the first post camera center and rotation remain those produced by `B0`;
- later camera translations relative to that first camera would be scaled;
- pointmap, roots, joints, vertices, and body offsets use the same scalar;
- RGB reprojection is unchanged by the uniform perspective scaling;
- no per-human or scene-only transform is allowed in the shared-scale methods.

## 4. Does One Clean Shot Scale Exist?

The table reports the post/pre scale needed to put each independently reconstructed
quantity into the same GT metric convention.

| Quantity | Mean | Median | P90 | Std |
|---|---:|---:|---:|---:|
| Root-centered body size | 1.004 | 1.006 | 1.037 | 0.027 |
| Camera-relative radial distance | 1.092 | 1.076 | 1.159 | 0.207 |
| Camera-axis depth | 1.099 | 1.092 | 1.171 | 0.218 |
| Multi-human pairwise layout | 0.999 | 1.007 | 1.103 | 0.099 |

Interpretation:

1. Human3R SMPL-X body size is already stable across shots. The median mismatch is below
   one percent.
2. Multi-human layout is also centered near unit scale, but is noisier because people
   move and root depth is imperfect.
3. Camera-relative human depth has a larger bias and much larger variance.
4. Therefore the dominant error is not a clean uniform enlargement of camera, scene,
   and humans. It is primarily view-dependent root/depth error.

The GT cue disagreement is substantial:

```text
median |body - radial| = 0.079
median |body - layout| = 0.065
median |radial - layout| = 0.113
```

A single shared scale is therefore only an approximation.

## 5. Explicit Causal Scale Cues

The deployable cues are:

```text
body state:   pre/post predicted root-centered body radius
layout state: predicted pre anchor/post root pairwise distances
scene state:  B0-aligned background point-cloud Chamfer scale
fused scale:  median of all available cues
```

### Overall result

| Method | Scale median | Root | Vertex | Layout distance | Scene Chamfer | Root improve rate |
|---|---:|---:|---:|---:|---:|---:|
| B0, unit scale | 1.000 | 0.442 | 0.458 | 0.073 | 1.555 | - |
| Body-state scale | 1.002 | 0.448 | 0.464 | 0.074 | 1.554 | 48.3% |
| Layout-state scale | 1.005 | 0.434 | 0.449 | 0.076 | 1.592 | 51.1% |
| Scene-Chamfer scale | 0.801 | 0.605 | 0.622 | 0.242 | 1.323 | 49.4% |
| Explicit median scale | 0.978 | **0.394** | **0.411** | **0.050** | 1.532 | 68.3% |

The fused explicit scale improves mean root and pairwise layout on every sequence:

| Sequence | B0 root | Explicit root | B0 layout | Explicit layout |
|---|---:|---:|---:|---:|
| three | 0.315 | 0.269 | 0.085 | 0.070 |
| dance | 0.383 | 0.336 | 0.082 | 0.048 |
| box | 0.556 | 0.504 | 0.060 | 0.042 |

However, it is not yet a safe frozen module:

- overall root P90 changes only from `0.630` to `0.635 m`;
- on `dance`, P95 worsens from `0.592` to `0.665 m`;
- `16.4%` of dance cuts are harmed by more than `5 cm`;
- explicit/Oracle scale correlation is only `0.224` overall and `-0.102` on dance;
- scene scale is driven toward `0.786` by low-overlap Chamfer and conflicts with human
  body/layout scale.

This is evidence of feasibility, not evidence that the current explicit estimator is
ready for the main method.

## 6. Oracle Diagnostics

| Method | Root | Vertex | Layout distance |
|---|---:|---:|---:|
| B0, unit scale | 0.442 | 0.458 | **0.073** |
| GT metric-relative scale | 0.616 | 0.615 | 0.174 |
| GT root-optimal scale | **0.222** | **0.242** | 0.138 |

The direct GT metric-relative scale makes the result much worse. The root-optimal scalar
has a large upper bound but damages pairwise layout. Its median is `0.830`, while body and
layout scales are near `1.0`.

This apparent contradiction is diagnostic: the root-optimal scalar absorbs B0 camera
error and person-specific radial-depth error. It is not a physically valid shared scene
scale. A low root error alone cannot justify a scale module.

## 7. Does Scale Rescue Current Fine Alignment?

| Method | Camera composite | Root | Vertex | Layout distance |
|---|---:|---:|---:|---:|
| Unit B0 | **0.354** | 0.442 | 0.458 | 0.073 |
| Explicit scale + B0 | **0.354** | **0.394** | **0.411** | **0.050** |
| Unit B0 + rotation | 0.396 | 0.506 | 0.520 | 0.073 |
| Explicit scale + rotation | 0.396 | 0.461 | 0.476 | 0.050 |
| Unit B0 + translation | 0.633 | 0.530 | 0.541 | 0.073 |
| Explicit scale + translation | 0.635 | 0.528 | 0.542 | 0.050 |
| Unit B0 + full multi | 0.640 | 0.516 | 0.528 | 0.073 |
| Explicit scale + full multi | 0.645 | 0.515 | 0.529 | 0.050 |

Even the GT root-optimal scale cannot rescue the current translation/full refinement:

```text
Oracle scale + B0 root:          0.222 m
Oracle scale + translation root: 0.532 m
Oracle scale + full multi root:  0.516 m
```

Camera composite rises to `0.798/0.821` for Oracle scale plus translation/full multi.

Conclusion: scale and fine translation are separate problems. The current equation still
forces each noisy predicted root onto its motion anchor and overwrites B0 translation.

## 8. Frozen Token Scale Probe

### Representations

The current model exposes all of the following without an architecture change:

```text
pose_token_out
state_summary_new
human_token_out
refined_human_tokens
fused_human_prompts
CUT3R head tokens
Multi-HMR head tokens
```

The probe uses 185 unique raw-reset frames. A fixed ridge readout is trained on `three`;
Human3R/V14 remains frozen. Input to the readout is the post-minus-pre token difference.

### Main result

For the combined pose/scene/human token:

| Test | Target | Probe MAE | Constant MAE | R2 |
|---|---|---:|---:|---:|
| dance | radial scale | 0.030 | 0.103 | 0.850 |
| dance | depth scale | 0.024 | 0.112 | 0.932 |
| box | radial scale | 0.131 | 0.168 | -0.085 |
| box | depth scale | 0.146 | 0.189 | -0.194 |
| dance+box | radial scale | 0.087 | 0.140 | 0.017 |
| dance+box | depth scale | 0.092 | 0.156 | -0.062 |

The result is stable across ridge strengths from `0.1` to `1000`:

- `three -> dance` remains strong;
- `three -> box` radial/depth R2 remains negative;
- body-size and pairwise-layout prediction usually does not beat the constant baseline;
- reversing the sequence split also causes major failures.

The tokens contain view/depth statistics, but the information is capture-dependent. They
do not currently provide a reliable shot-invariant shared scale state.

## 9. Assessment of the Two Proposed Routes

### Latent scale state

Architecturally feasible, because the relevant frozen tokens are already accessible. It
would remain causal to preserve one scalar and uncertainty in an external state while
resetting the recurrent scene state.

It is not currently justified as a no-training method. Raw token differences do not
generalize reliably to `box`, and storing a high-dimensional token would also preserve
view/content information rather than only scale. A future learned scalar readout would
need subject-, capture-, and camera-pair-disjoint supervision, but should not be added
until a physically consistent target scale is defined.

### Explicit scale after reconstruction

This route is mathematically cleaner and already satisfies streaming constraints:

```text
pre-shot fixed-size scale statistics
-> first post frame
-> one scalar plus uncertainty
-> one shared Sim(3) with B0
-> fixed propagation through the new shot
```

The current fused cue gives a useful mean improvement, but its tail and cross-cue conflict
prevent deployment. It should remain an optional ablation while `B0` rigid alignment is
the main route.

## 10. Motion and Occlusion Scope

Within-shot Human3R motion remains valid after reset. Therefore no large motion model is
needed. A short robust velocity or zero-motion prior is enough for a cut-time hypothesis.

However, a wrong cut-time motion anchor does not affect only two boundary frames. Because
one Boundary is propagated through the complete new shot, an incorrect Boundary adds a
fixed world-position error to every later frame. Internal motion remains coherent, but
the whole segment stays displaced.

Occlusion is intentionally left outside this audit.

## 11. Route Decision

1. Keep learned rigid `B0` as the current main Boundary.
2. Do not introduce a persistent latent scale token yet.
3. Keep one explicit shared scale as a causal feasibility ablation, not a frozen module.
4. Do not use root-optimal scale as evidence of physical scale correctness.
5. Do not expect scale to repair the current human-root translation equation.
6. Redesign fine alignment around independent residual evidence and a strict B0 prior.

The central finding is:

> Human3R already keeps body and multi-human layout scale near unit consistency. The
> remaining cross-shot error is dominated by view-dependent camera/root depth, which is
> not explained by one clean shared scalar. Scale correction can modestly improve human
> placement, but it is neither sufficient nor currently safe enough to replace rigid B0.

## 12. Artifacts

```text
versions/v14/probe_v14_shot_scale.py
versions/v14/probe_v14_scale_refinement.py
versions/v14/probe_v14_token_scale.py
output/v14/shot_scale_audit/
output/v14/scale_refinement_audit/
output/v14/token_scale_probe/
```
