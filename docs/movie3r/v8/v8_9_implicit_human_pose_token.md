# V8.9 Implicit Human-Pose Correction Token

Date: 2026-06-11

V8.9 is the current best-performing UniCon-style branch for joint camera and
human correction. It keeps the streaming constraint: inference uses only the
current frame, recurrent state, pose memory, and previous correction state. GT
camera and SMPL are used only for loss, metrics, and visualization overlays.

## Model

The correction token is built before the decoder:

```text
A_corr_t =
  current image / pose / human tokens
  + recurrent state memory
  + pose memory
  + previous corr token / delta / gate
```

`A_corr_t` enters the decoder together with the normal Human3R tokens. The
decoder then outputs refined image, pose, human, and correction tokens.

Camera correction is latent:

```text
refined A_corr_t
  -> pose residual head
  -> delta pose token + shared gate

raw pose token + gated delta pose token
  -> original Human3R pose head
  -> corrected camera pose
```

Human correction is also latent in this version:

```text
refined A_corr_t + refined human token + corrected pose token
  -> human latent residual head
  -> delta human token + shared gate

raw decoder human token + gated delta human token
  -> original Human3R human head
  -> corrected SMPL
```

This replaces the earlier diagnostic branch that directly edited
`smpl_transl` after the human head. The explicit `smpl_transl` branch remains
available as an A/B baseline, but it is not the current target method.

## Coordinate Rules

All AvatarReX pose and human losses must use the same raw camera gauge:

```text
camera GT: raw calibration c2w
SMPL GT: same raw/world gauge
T_w2c for SMPL projection: inverse(raw camera c2w)
```

Viewer comparisons must use the verified convention:

```text
gray  = raw Human3R camera / scene in raw frame-0 world
yellow = corrected camera / scene / SMPL aligned to raw frame-0 world
red   = GT camera / GT SMPL overlay, aligned by GT frame-0 camera
```

The corrected output stores both poses for analysis:

```text
v8_raw_camera_pose = Human3R pose before correction
camera_pose        = final corrected pose
```

This is a debug/metric output convention only. It does not leak GT or future
frames into streaming inference.

## Verified Single-Clip Result

Config:

```text
config/train_v8_9_avatarrex_lbn1_single_human_latent_from_human3r_coordfix.yaml
```

Checkpoint:

```text
output/v8_9_avatarrex_single_clip_no_crop_latent/
  v8_9_avatarrex_lbn1_single_human_latent_from_human3r_coordfix_gpu7/
  checkpoint-best.pth
```

Data:

```text
AvatarReX lbn1 AABB
seqA = lbn1/22070935
seqB = lbn1/22053926
start_frame = 1671
view_angle_deg = 143.418318
resize_mode = resize_only_16
```

Single-clip benchmark after 120 epochs:

```text
raw human trans err       0.708 m
corrected human trans err 0.0037 m
raw camera trans err      0.592 m
corrected camera trans err 0.016 m
raw rot err               24.04 deg
corrected rot err          0.119 deg
gate mean                  0.339
```

Visual check:

```text
corrected viewer: http://127.0.0.1:8171
original Human3R top-1 viewer: http://127.0.0.1:8173
```

The original Human3R viewer needs top-1 SMPL filtering for this case because
the Multi-HMR detector can emit multiple heatmap peaks for one real person. The
scene still contains one person; the extra SMPL meshes are duplicate candidates,
not real people.

