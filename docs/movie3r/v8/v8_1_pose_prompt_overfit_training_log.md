# V8.1 Pose Prompt Overfit Training Log

## Purpose

This run verifies that the UniCon-style decoder-in pose correction prompt can be trained end-to-end on a very small AvatarReX AABB subset.

The goal is not final performance yet. The goal is to check:

1. the new correction token can enter the decoder;
2. the residual head can update the pose token before the original pose head;
3. the loss can supervise the corrected camera pose;
4. training loss and pose error can decrease on a tiny fixed split.

## Setup

- Data: `/data/wangzheng/iJCV-CODE/data/Avatarrex_output`
- Config: `config/train_v8_pose_prompt_overfit.yaml`
- Base checkpoint: `src/human3r_896L.pth`
- Output: `checkpoints/v8_1_pose_prompt_overfit_2samples_gpu7`
- GPU: `CUDA_VISIBLE_DEVICES=7`
- Reason for GPU choice: GPU 7 was idle when the run started.

Command:

```bash
cd /data/wangzheng/iJCV-CODE/Movie3R
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=7
export HYDRA_FULL_ERROR=1
.venv/bin/python src/train.py --config-name train_v8_pose_prompt_overfit
```

## Saved Files

- `checkpoint-last.pth`
- `checkpoint-1.pth`
- `checkpoint-best.pth`
- `checkpoint-final.pth`
- `train_steps.jsonl`
- `metrics_epoch.jsonl`
- `log.txt`
- `train.log`
- TensorBoard event file

## Key Metrics

### Initial Evaluation

| Split | Loss | Corrected Trans Err | Raw Trans Err | Corrected Rot Err | Raw Rot Err | Gate | Delta Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.8931 | 2.0023 | 2.0022 | 88.32 deg | 88.33 deg | 0.0179 | 0.0 |

At initialization, corrected and raw pose are almost the same. This is expected because the residual branch is initialized to produce near-zero correction.

### Final Training Step

| Step | Loss | Pose Loss | Corrected Trans Err | Raw Trans Err | Corrected Rot Err | Raw Rot Err | Gate | Delta Norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 19 | 0.2178 | 0.1726 | 0.3356 | 0.4071 | 80.27 deg | 80.62 deg | 0.0286 | 7.0182 |

The final training step improves translation error compared with the raw pose. Rotation changes are still small.

### Final Evaluation

| Split | Loss | Corrected Trans Err | Raw Trans Err | Corrected Rot Err | Raw Rot Err | Gate | Delta Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train/test eval | 0.1206 | 0.1746 | 0.2118 | 62.19 deg | 62.69 deg | 0.0286 | 6.2795 |
| val | 0.1660 | 0.3026 | 0.3759 | 80.28 deg | 80.63 deg | 0.0287 | 7.0478 |

## Interpretation

This small overfit sanity check passed.

The new V8.1 branch is trainable, the loss is connected, and the corrected pose improves translation error on the tiny held validation split. The current version does not yet strongly improve rotation, and the gate stays low while the latent residual norm becomes relatively large. These two points should be monitored in the next experiment.

## Next Checks

1. Increase the fixed AABB samples from 2 to 20-50.
2. Keep the view-angle gap large, preferably at least 60 degrees.
3. Track raw vs corrected translation and rotation error separately.
4. Add stronger gate diagnostics or adjust gate supervision.
5. Watch `v8_pose_prompt_delta_norm`; if it keeps growing, increase latent regularization or reduce learning rate.

## 2026-05-30 Update: Raw-Camera Pose Overfit Success

### What Passed

The V8.1 UniCon-style decoder-in pose prompt was successfully overfit on one fixed AvatarReX AABB sample after fixing the camera-pose target coordinate system.

Sample:

```text
seq_a = 22010710
seq_b = 22053923
start_frame = 0
views = A_t, A_t+1, B_t+2, B_t+3
```

Training command:

```bash
cd /data/wangzheng/iJCV-CODE/Movie3R
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=4
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR=/tmp/matplotlib
.venv/bin/python src/train.py \
  --config-name train_v8_pose_prompt_overfit_1sample_start0_nodepth_rawpose \
  exp_name=v8_1_pose_prompt_overfit_1sample_start0_nodepth_rawpose_gpu4 \
  logdir=/tmp/movie3r_v8_1_pose_prompt_overfit_1sample_start0_nodepth_rawpose_gpu4/logs \
  output_dir=/tmp/movie3r_v8_1_pose_prompt_overfit_1sample_start0_nodepth_rawpose_gpu4/
```

Output:

```text
checkpoint:
  /tmp/movie3r_v8_1_pose_prompt_overfit_1sample_start0_nodepth_rawpose_gpu4/checkpoint-last.pth

saved inference:
  /tmp/movie3r_v8_1_pose_prompt_train_start0_nodepth_rawpose/v8_dataset

viewer:
  http://127.0.0.1:8112
```

Final evaluation:

| Target | Corrected Trans Err | Corrected Rot Err | Raw Trans Err | Raw Rot Err |
| --- | ---: | ---: | ---: | ---: |
| `raw_camera_pose` | 0.0075 | 0.0617 deg | 0.1154 | 4.7876 deg |

Visual result:

- The previous B-frame upside-down failure is fixed.
- The corrected B-frame camera has the expected orientation:

```text
frame 2/3 corrected:
  y-axis ~= +1
  z-axis ~= -1
```

This is the expected 180-degree view change without the erroneous roll/up-axis flip.

### What This Proves

This run is the intended V8.1 training style:

```text
A_corr_t token enters decoder
  -> decoder refines correction token with image / human / pose / state tokens
  -> residual head predicts pose-token residual
  -> corrected pose token goes through the original pose head
  -> corrected camera_pose is supervised by pose loss
```

It is not a post-processing pose smoother, not BA, not pose graph optimization, and not a sidecar head applied after inference.

The inference path remains recurrent / online. During inference it does not read future frames or GT pose. GT pose is used only as the training loss target.

### Critical Coordinate-System Lesson

The first successful-looking run was actually wrong. It used `Avatarrex_output/Training/<seq>/cam/*.npz` as `camera_pose` supervision. That processed pose is not the correct target for this viewer/SMPL coordinate convention on the AABB test.

Old wrong target:

```text
source:
  Avatarrex_output/Training/<seq>/cam/*.npz
field:
  camera_pose = cam["pose"]

problem on B frames:
  z-axis ~= -1
  y-axis ~= -1
```

The `z-axis ~= -1` part is expected for the opposite camera view, but `y-axis ~= -1` means the camera up direction is flipped. Training against this target made the loss very low while the viewer still showed the last two frames upside down. The model was learning the wrong target correctly.

Correct target:

```text
source:
  /data/wangzheng/iJCV-CODE/data/avatarrex_lbn1/calibration_full.json

calibration convention:
  X_cam = R_w2c @ X_world + T_w2c

c2w conversion:
  R_c2w = R_w2c.T
  t_c2w = -R_w2c.T @ T_w2c

loss target:
  T_rel_i = inv(raw_camera_pose_0) @ raw_camera_pose_i
```

Correct B-frame orientation:

```text
raw calibration target:
  z-axis ~= -1
  y-axis ~= +1
```

Implementation state:

- `AvatarReX_AABB` now can emit `raw_camera_pose` from raw calibration.
- `V81PosePromptLoss` now supports `pose_key`.
- The rawpose config uses:

```text
V81PosePromptLoss(..., pose_key='raw_camera_pose')
AvatarReX_AABB(..., load_da3_depth=False, raw_calibration_root="/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1")
```

### DA3 Depth Rule

Do not use `Avatarrex_output/depth/*.npy` as metric GT depth for V8.1 pose correction.

Reason:

- These depth files are DA3 / monocular pseudo-depth.
- They are not guaranteed to be in the same metric scale as raw calibration and SMPL.
- Using them as camera/world geometry can create false conclusions about alignment.

Allowed:

- Keep `depthmap` zero-filled for pose-only V8.1 training:

```text
load_da3_depth=False
```

- Use Human3R's own predicted pointmap/depth for visualization or as a model output/cue.
- Use raw calibration + SMPL without DA3 depth for coordinate sanity checks.

Not allowed:

- Do not use DA3 depth as cross-camera metric GT.
- Do not use DA3 depth to validate whether camera pose is correct.
- Do not mix raw calibration camera + raw SMPL + DA3 pointcloud and treat the scene as GT.

### Debug Protocol Before Future Training

Before any new AvatarReX pose-prompt training run:

1. Print the relative pose target axes for all four frames.
2. B frames should have `z-axis ~= -1` and `y-axis ~= +1`.
3. If B frames show `y-axis ~= -1`, the target is wrong.
4. Set `load_da3_depth=False` unless the experiment explicitly studies DA3 as a weak monocular cue.
5. Use no-depth raw calibration + SMPL viewer as the coordinate sanity check.
6. Only after this check passes, evaluate model-corrected pointmap / SMPL in the Human3R viewer.
