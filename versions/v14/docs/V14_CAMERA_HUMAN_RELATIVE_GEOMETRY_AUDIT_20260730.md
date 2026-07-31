# V14 Camera-Human Relative Geometry Audit

Date: 2026-07-30

## Question

This audit tests three related design choices:

1. Whether the V9-style shadow branch changes camera-human-scene relative geometry.
2. Whether human-anchor refinement pulls the final camera away from a good learned `B0`.
3. Whether V14 should train and run camera correction only, or retain joint camera-human
   supervision.

The tested deployment contract is causal:

```text
shadow(pre-cut history + first post-cut event) -> C_shadow
raw-reset(first post-cut frame)                -> C_raw + raw scene + raw humans
B0 = C_shadow @ inverse(C_raw)
discard every shadow output except B0
apply one B0 to the complete raw-reset segment
```

## Implementation Audit

The current runtime does **not** commit shadow humans or the shadow pointmap. It consumes
only `C_shadow`, computes `B0`, and applies `B0` to raw-reset outputs.

For a camera-local human root `x_h`, the committed result is:

```text
C_final = B0 @ C_raw
x_world_final = C_final @ x_h
              = B0 @ (C_raw @ x_h)
```

Therefore the output-side operation is one shared rigid transform. Camera-local SMPL-X
parameters remain unchanged, while their world-space roots, joints, and vertices move by
the same `B0` through the transformed camera.

This behavior is covered by `tests/test_v14_segment_boundary.py`, including camera,
world pointmap, camera-local SMPL-X, and world human-root checks.

## Experiment A: Shadow Relative-Consistency

### Protocol

For the largest-view-span eligible case in each MultiHuman sequence:

```text
three: three_t1100_c1_c2_k0, 173.9 deg
dance: dance_t0200_c1_c4_k0, 139.9 deg
box:   box_t0470_c1_c4_k0, 139.9 deg
```

The same post-cut RGB frame is decoded twice:

```text
full shadow: pre-cut history + correction event
raw reset:   fresh state + event off
```

GT identity is used only to compare the same predicted person across the two outputs.

### Results

| Sequence | Shadow/raw human local-root drift | Shadow/raw local pointmap drift | Shadow pointmap internal residual | Raw pointmap internal residual |
|---|---:|---:|---:|---:|
| three | 0.418 m | 0.196 m | 0.912 m | 0.089 m |
| dance | 0.337 m | 0.193 m | 0.590 m | 0.077 m |
| box | 0.322 m | 0.096 m | 0.619 m | 0.077 m |

`Internal residual` compares the predicted world pointmap with camera-transforming its
predicted local pointmap. The shadow result is not a clean rigid replacement for the raw
local reconstruction. Committing corrected shadow humans or shadow scene would therefore
change the relative reconstruction substantially.

When human latent correction and human-head LoRA are disabled at inference:

- human local-root drift falls to `0.106 / 0.093 / 0.022 m`;
- every tested `B0` remains bit-exact (`max_abs_delta = 0`);
- scene values remain unchanged.

This inference switch is a dependency diagnostic, not a fair training comparison.

### Conclusion

The shadow branch may use all human/image/state tokens for attention, but its human and
scene outputs must remain non-committing. Only its camera estimate is a valid source for
the explicit coarse gauge `B0`.

## Experiment B: B0 Anchor-Conflict Ablation

### Protocol

The audit uses 180 existing cuts:

```text
three: 41
dance: 61
box: 78
```

All cases use strict GT-ID rebuilt by GT mesh projection. This is a diagnostic WHERE
experiment and does not claim deployable identity. Fixed Explicit, V16, and the 20-degree
bound are unchanged.

Compared methods:

```text
B0 only
old Phase-2 uniform multi-human Boundary
B0 + human rotation mean, keep B0 translation
B0 rotation + human translation, keep B0 rotation
B0 + per-candidate rotation/translation mean
B0 + shared-rotation human translation
```

### Combined Results

| Method | Camera T | Camera R | Composite | P95 composite | Human root | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| B0 only | **0.277** | **3.85** | **0.354** | **0.577** | **0.442** | 0.0% |
| Phase-2 uniform multi | 0.550 | 8.15 | 0.713 | 1.264 | 0.521 | 0.6% |
| B0 + rotation only | 0.277 | 5.94 | 0.396 | 0.682 | 0.506 | 0.0% |
| B0 + translation only | 0.556 | 3.85 | 0.633 | 0.793 | 0.530 | 0.0% |
| B0 + per-candidate full | 0.521 | 5.94 | 0.640 | 1.100 | 0.516 | 0.0% |
| B0 + shared-rotation translation | 0.552 | 5.94 | 0.671 | 1.097 | 0.530 | 0.0% |

Paired against `B0 only`:

- human translation worsens camera composite on `97.2%` of cuts;
- full per-candidate refinement worsens it on `92.8%` of cuts;
- rotation-only improves `32.8%`, but worsens `67.2%` and raises mean composite by
  `0.0419`;
- no refinement improves aggregate human-root error either.

### Interpretation

Human3R camera-relative human depth error is absorbed by the root-anchor translation
equation. That makes the seam look tighter while moving the physical post-cut camera to an
incorrect world position. Camera positions before and after a cut must not be made equal;
the objective is to place the new camera correctly in the persistent old world.

The current evidence does not justify any default human translation refinement. It also
does not justify rotation-only refinement without a reliable deployable acceptance rule.

## Experiment C: Fair Camera-Only Retraining

### Protocol

Both checkpoints use:

- the same formal V9 initialization;
- the same single AvatarReX `lbn1_1192` event;
- 80 FP32 epochs with no early stopping;
- the same correction-token architecture, full decoder attention, camera loss, pose-head
  LoRA, and pointmap-preservation losses.

The camera-only run disables:

```text
human latent correction
human-head LoRA
human translation loss
human parameter keep loss
```

It converged from validation loss `0.6960` to `0.0613` at epoch 80. Loss magnitude is not
directly comparable to the joint model because the joint objective includes human terms.

### Frozen 180-Cut Result

Values are `mean / median / P90 / P95`.

| Model | Camera T | Camera R | Composite |
|---|---:|---:|---:|
| Joint camera-human | 0.2768 / 0.2254 / 0.4849 / 0.5124 | 3.847 / 3.708 / 5.424 / 5.663 | **0.3538 / 0.3122 / 0.5458 / 0.5767** |
| Camera-only | 0.2787 / 0.2210 / 0.4904 / 0.5165 | 3.967 / 3.863 / 5.587 / 5.895 | 0.3580 / 0.3142 / 0.5498 / 0.5818 |

Camera-only minus joint composite is `+0.00423` on average. Joint is better on `93.3%`
of cuts (`paired Wilcoxon p = 2.19e-29`), although the absolute difference is only about
1.2% of the joint composite.

Identity matching is unchanged:

| Sequence | Joint B0 all-correct | Camera-only B0 all-correct |
|---|---:|---:|
| three | 100.0% | 100.0% |
| dance | 100.0% | 100.0% |
| box | 98.7% | 98.7% |

### Conclusion

Human correction is not needed as a deployed output or as a direct input to `B0`.
However, human supervision provides a small, consistent auxiliary training benefit to the
shared correction representation. The clean interpretation is:

```text
human tokens / optional human loss: training-time auxiliary evidence
shadow camera:                       the only committed estimate source
shadow human and scene outputs:      always discarded
```

## Route Decision

The main V14 path should now be:

```text
1. Detect cut.
2. Hard-reset the committed Human3R scene/camera state.
3. Run one read-only shadow transaction with old context and the first post-cut frame.
4. Read only C_shadow.
5. Independently decode the same frame from fresh raw-reset state as C_raw + raw geometry.
6. Compute B0 = C_shadow @ inverse(C_raw).
7. Discard all shadow state, humans, and pointmaps.
8. Apply one shared B0 to the raw-reset camera, world pointmap, and all humans.
9. Use B0-aligned humans for cross-shot WHO association and track continuity.
10. Propagate the fixed B0 through the new shot's raw-reset stream.
```

For the current main method:

- keep the joint checkpoint as the stronger `B0` baseline;
- describe human correction only as auxiliary training supervision;
- remove human-root translation refinement from the default path;
- do not enable rotation-only refinement without a new frozen, deployable rule that beats
  `B0` on aggregate;
- retain Uniform Multi-Human Consensus as an oracle/diagnostic, not as an automatic
  post-`B0` replacement under the current evidence.

This route preserves the central claim:

> Decouple state continuity from world continuity: commit clean raw-reset state, recover
> the world gauge through a non-committing shadow transaction, and express that correction
> as one explicit shared Boundary.

## Artifacts

```text
versions/v14/probe_b0_anchor_conflict.py
versions/v14/probe_v14_shadow_relative_consistency.py
config/train_v14_1_cut_event_single_v9_event_only_geometry_camera_only.yaml
output/v14/b0_anchor_conflict/v14_b0_anchor_conflict.json
output/v14/shadow_relative_consistency/v14_shadow_relative_consistency.json
output/v14/b0_identity_matching_camera_only/{three,dance,box}/
```
