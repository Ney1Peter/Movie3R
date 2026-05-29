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
