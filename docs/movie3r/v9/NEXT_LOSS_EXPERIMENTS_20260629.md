# V9 Next Loss Experiments

## Goal

The current best 60h variant is `h3_c2_human_delta_weak`. It improves camera translation on AABB and keeps AAAA gate low, but some subjective examples still show incomplete human/scene alignment. The next step is to test whether extra coupling losses can reduce this residual mismatch.

All three experiments use the same 60h AvatarReX + THuman data, the same original Human3R initialization, and the same V9 pose+human LoRA architecture. Only the loss is changed.

## Experiment A: H3 Improvement 0.075

Config: `config/train_v9_60h_h3_imp075_pose_human_lora_bs10.yaml`

This is the smallest change. It keeps H3's deadzone gate target and weak human delta regularization, but increases `improvement_weight` from `0.05` to `0.075`.

Expected behavior:

- More pressure for corrected pose to beat raw Human3R pose.
- Still protected against over-correction by H3's gate deadzone.
- Useful as a low-risk check before adding new loss terms.

## Experiment B: H3 Human-Camera Reference Loss

Config: `config/train_v9_60h_h3_hcam_ref_pose_human_lora_bs10.yaml`

This adds `human_cam_ref_weight=5.0`.

For each view, the predicted SMPL translation is treated as camera-space human position and transformed by the predicted relative camera pose into the view-0 reference frame. The same transform is applied to GT SMPL translation using the GT relative camera pose. The loss supervises these two reference-frame positions.

Intuition:

- Camera pose and human translation should not be correct independently; their relative placement should be correct together.
- This directly targets the issue where the camera looks right but the human appears offset in depth or height.

## Experiment C: H3 Human-Camera Reference + Pairwise Motion

Config: `config/train_v9_60h_h3_hcam_ref_pairwise_pose_human_lora_bs10.yaml`

This adds:

- `human_cam_ref_weight=5.0`
- `human_pairwise_ref_weight=5.0`

The pairwise term compares human motion between every pair of views after transforming each human translation into the view-0 reference frame.

Intuition:

- If a person truly moves between frames, the model should preserve that motion.
- This is meant to reduce the failure mode where the correction branch pulls a fast-moving person back toward the earlier frames.

## Training Arrangement

Prepared launch script:

`scripts/training/run_v9_60h_loss_followup.sh`

Default GPU assignment:

| Experiment | Config | GPU |
| --- | --- | ---: |
| A | `train_v9_60h_h3_imp075_pose_human_lora_bs10` | 5 |
| B | `train_v9_60h_h3_hcam_ref_pose_human_lora_bs10` | 6 |
| C | `train_v9_60h_h3_hcam_ref_pairwise_pose_human_lora_bs10` | 7 |

The script prints commands by default and starts training only with `--start`.

## Evaluation Priority

After training:

1. Run the same zxc held-out AABB/AAAA benchmark as the previous 60h sweep.
2. Compare camera translation/rotation error, gate, MPJPE, PA-MPJPE, PVE, root error, and human translation error.
3. Visualize the same two selected zxc clips from `EXPERIMENT_60H_SWEEP_RESULTS_20260629.md`.
4. Test the fast-motion H36M clips again to see whether pairwise motion loss helps avoid pulling the moving human back.
