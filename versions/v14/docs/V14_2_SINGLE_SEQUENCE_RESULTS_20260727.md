# V14.2 Single-Sequence Shadow Boundary Validation

## Protocol

Development case:

```text
AvatarReX lbn1
A: 22053926 frames 1189, 1190, 1191, 1192
B: 22010716 frames 1192, 1193, 1194, 1195, 1196, 1197
cut index: 4
view span: 132.85 degrees
```

The validation uses four causal rollouts:

1. `continue`: all event labels are off; this is only a comparison baseline.
2. `shadow`: four pre-cut frames plus the first post-cut frame; only the cut frame has the event label.
3. `raw first-only`: fresh-state inference of B1192 only.
4. `raw full`: fresh-state inference of B1192-B1197; this is the only post-cut state that is retained.

The explicit Boundary is computed without future frames:

```text
B0 = C_shadow(B1192) @ inverse(C_raw-first-only(B1192))
```

The same fixed `B0` is left-multiplied into every raw post-cut camera and
world pointmap. Camera-local SMPL-X parameters are not modified. The shadow
state is discarded.

## Results

Post-cut summary over six frames:

| Method | Translation mean (m) | Rotation mean (deg) | Composite mean | Human head mean (m) | Catastrophic |
|---|---:|---:|---:|---:|---:|
| Continue | 3.2790 | 134.848 | 5.9759 | 0.8864 | 100.0% |
| Raw reset | 3.3029 | 133.261 | 5.9681 | 0.8071 | 100.0% |
| Shadow fixed B0 | **0.1313** | **3.776** | **0.2068** | 0.7459 | **0.0%** |
| GT-camera-only Boundary | 0.0042 | 0.032 | 0.0048 | 0.7626 | 0.0% |

Here:

```text
composite = translation_m + 0.02 * rotation_deg
catastrophic = translation > 1 m or rotation > 30 degrees
```

Boundary diagnostics:

```text
predicted B0 error:                  0.1315 m / 3.792 deg
raw first-only vs raw-full frame 0: 0.0000 m / 0.000 deg
shadow first-post camera error:      0.1315 m / 3.792 deg
```

The fixed-B0 post-frame ranges are stable:

```text
translation: 0.1298 - 0.1333 m
rotation:    3.758  - 3.805 deg
```

## Interpretation

The state split and fixed-Boundary propagation are valid for this case:

- the raw first-only and raw-full first predictions are identical;
- `B0` therefore does not read future frames;
- one fixed `B0` removes the camera-cut catastrophe for all six post frames;
- the post-cut raw rollout remains the only committed recurrent state;
- camera, pointmap, and humans share one world transform.

This is not yet a solved V14 model:

- the expanded four-frame pre-cut context is worse than the earlier
  training-length V14.1 capacity result of about `0.017 m / 0.105 deg`;
- V14.1 is therefore sensitive to the amount of pre-cut recurrent context;
- the human head error remains around `0.76 m` even with the GT-camera-only Boundary,
  so this component is caused by camera-local human reconstruction rather
  than the propagated world Boundary;
- the next training experiment should vary pre-cut context length and enforce
  shared-SE(3) consistency instead of treating camera and human corrections as
  independent final outputs.

## V12/V13 Post-Processing Ladder

The frozen V12 post-processing was also applied after the learned V14 `B0`,
with every intermediate fixed Boundary saved separately:

```text
raw reset
-> learned V14 B0
-> B0 rotation + V16 torso residual
-> V16 rotation + V12 explicit root translation
```

| Stage | Translation mean (m) | Rotation mean (deg) | Composite | Human head mean (m) | Catastrophic |
|---|---:|---:|---:|---:|---:|
| V14 learned B0 | **0.1313** | **3.776** | **0.2068** | **0.7459** | **0.0%** |
| B0 + V16 rotation | 0.1311 | 13.099 | 0.3931 | 0.7700 | 0.0% |
| B0 + V12 Lite translation | 1.6354 | 13.099 | 1.8974 | 0.7942 | 100.0% |

V16 accepted an unclipped `-9.355 deg` torso residual. That residual was useful
when V16 refined the much weaker Fixed Explicit initializer, but it is the
wrong correction for this already accurate learned `B0`: Boundary rotation
error rises from `3.792 deg` to `13.115 deg`. Replacing the learned translation
with the old single-human root anchor then raises Boundary translation error
from `0.132 m` to `1.637 m`.

This case therefore rejects unconditional composition of the old geometry
stack after V14. V12/V16 should only be retained as a guarded fallback or
independent comparison, not as a mandatory residual. V13 adds no new geometry
on this one-person AvatarReX sample: uniform multi-human consensus with `N=1`
is exactly the V12 single-human fallback. A distinct V13 visualization requires
a multi-human cut with at least two correctly associated identities.

## GT Boundary Audit

The result previously labelled `GT-Boundary oracle` is only a camera oracle:

```text
A = predicted_continue_camera_0 @ inverse(GT_camera_0)
target_camera_cut = A @ GT_camera_cut
B_gt_camera = target_camera_cut @ inverse(raw_reset_camera_cut)
```

It guarantees that the first post-cut **camera pose** matches the selected GT
camera gauge. It does not use GT depth, GT pointmaps, or GT human geometry, and
therefore is not a full 3D visualization upper bound. Its near-zero camera
metric is expected because the Boundary is constructed directly from that same
camera target.

The AvatarReX calibration convention itself is correct. Projecting the complete
GT SMPL-X mesh with `X_cam = R @ X_world + T` gives mask-bbox IoU `0.954` for
camera `22053926` and `0.914` for camera `22010716`; about `93%-95%` of projected
vertices land inside the person mask. Interpreting the same `R,T` as c2w puts
one mesh entirely behind the camera and the other entirely outside the image.

The visual failure comes from camera-local reconstruction bias:

| View | GT head depth (m) | Human3R head depth (m) | Depth excess |
|---|---:|---:|---:|
| `22053926` | 1.75 | 2.43 | 39% |
| `22010716` | 1.72 | 2.48 | 45% |

After the metric GT camera Boundary, that biased post-cut human is about
`1.40 m` from the last pre-cut predicted human. V12 Lite instead anchors the
post-cut predicted root to the pre-cut predicted root, reducing this visual
jump to about `0.02 m`. This explains why V12 Lite looks better even though its
metric camera translation is worse.

The earlier `human_head_m` values in this document were also invalid: the
AvatarReX annotations do not contain precomputed `smplx_head_world`, and the
runner had read that optional zero-filled field directly. The corrected runner
decodes the head from the GT SMPL-X parameters whenever
`smplx_has_precomputed_keypoints=0`. The tables above contain the corrected
human errors.

Cut-level visual consistency separates the two objectives directly:

| Boundary | Camera translation error (m) | Predicted-human cut jump (m) | Pointcloud NN median (m) |
|---|---:|---:|---:|
| GT-camera-only | 0.00 | 1.401 | 2.534 |
| V14 B0 + V12 Lite | 1.64 | 0.023 | 1.453 |
| GT rotation + predicted-human anchor | 1.41 | 0.018 | 1.344 |

The pointcloud nearest-neighbor number is only a seam diagnostic, not a GT
scene metric. It nevertheless agrees with the visible discontinuity.

The runner now reports two separate diagnostics:

```text
gt_camera_only_boundary
gt_rotation_human_anchor_oracle
```

The first evaluates camera correctness. The second uses GT relative rotation
but a predicted-human continuity translation, and diagnoses the best visual
continuity available under the local Human3R depth bias. There is no unique
full "GT Boundary" for distorted local predictions unless the evaluation
objective (camera, human, or pointmap) is specified.

## Artifacts

Reproducible runner:

```text
versions/v14/run_v14_2_single_sequence.py
```

Manifest:

```text
config/manifests/v14_2_segment/single/lbn1_1192.jsonl
```

Runtime report and viewer payloads:

```text
/dev/shm/movie3r_v14_2/lbn1_1192_gt_audit_v2
```
