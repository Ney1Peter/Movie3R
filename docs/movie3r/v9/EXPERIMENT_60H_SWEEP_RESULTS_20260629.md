# V9 60h Loss Sweep Results

## Purpose

This note records the V9 60h follow-up experiment after the pose+human latent correction design was working. The goal was to test whether stronger gate/loss supervision can improve correction quality without changing the model architecture.

The two formal variants were trained on the same 60h AvatarReX + THuman mixed setting:

- Same base checkpoint: `src/human3r_896L.pth`
- Same model: V9 implicit pose+human correction with pose-head LoRA and human-head LoRA
- Same dataloader: resolution-aware mixed AvatarReX/THuman training with `resize_only_16`
- Same batch/schedule: inherited from `train_v9_mixed_avatarrex_thuman_60h_pose_human_lora_bs10`
- Different only in `V82PoseRelationLoss` parameters

## Original Human3R Behavior

Original Human3R is strong on continuous clips, especially when the background has enough texture. On normal `AAAA` clips, the raw camera pose error is already very small, so correction should stay nearly off.

On discontinuous `AABB` clips, Human3R can produce camera-pose jumps. The V9 correction task is to detect when the raw pose is unreliable and apply a small learned correction to camera pose and human translation while preserving normal clips.

For the zxc held-out AABB test set, raw Human3R averaged:

| Metric | Raw Human3R |
| --- | ---: |
| Camera translation error | 0.256m to 0.268m |
| Camera rotation error | 5.28 deg to 8.36 deg |
| Human translation error | 0.104m to 0.106m |

The range above comes from the two separately evaluated variants because each run dumps its own raw values from the same evaluation script.

## Variant C1: Stronger Drift And Improvement

Config: `config/train_v9_60h_c1_drift_x2_improve_x2_pose_human_lora_bs10.yaml`

Idea: make the model more aggressive by strengthening the two losses that push correction to happen.

Changes from the normal V9 loss:

- `drift_weight=0.1`: double the gate/drift supervision strength.
- `improvement_weight=0.1`: double the loss that asks corrected pose to beat raw Human3R pose.
- `drift_target_deadzone=0.0`: no safe zone for small raw errors.
- `drift_target_scale=1.0`: direct raw-error-to-gate target scaling.
- `improvement_margin=0.0`: corrected only needs to be better than raw, with no extra required margin.
- `human_trans_delta_weight=1e-5`: keep the original human delta regularization.

Interpretation: C1 says "correct more actively." It increases correction pressure, but it does not protect small-error clips as much.

## Variant H3: Deadzone Gate Plus Weaker Human Delta Regularization

Config: `config/train_v9_60h_h3_c2_human_delta_weak_pose_human_lora_bs10.yaml`

Idea: keep gate behavior stable on normal clips, but allow the human branch to move more when correction is really needed.

Changes from the normal V9 loss:

- `drift_weight=0.1`: double gate/drift supervision strength.
- `drift_target_deadzone=0.05`: raw errors below a small threshold should not strongly activate the gate.
- `drift_target_scale=0.45`: make the gate target saturate more deliberately after the deadzone.
- `improvement_margin=0.05`: corrected pose should be meaningfully better than raw pose, not just slightly better.
- `human_trans_delta_weight=1e-6`: make human latent correction less constrained, so human translation can move when needed.
- `improvement_weight=0.05`: keep improvement weight moderate.

Interpretation: H3 says "only correct when there is enough evidence, but correct more freely once needed." This matched the task better than C1.

## Full Test Metrics

All values are means. Lower error is better. Gate should be high on AABB drift cases and low on AAAA normal cases.

### AvatarReX In-Domain

| Variant | Subset | Camera trans | Camera rot | Gate | Human trans |
| --- | --- | ---: | ---: | ---: | ---: |
| H3 | AABB | 0.3136m -> 0.1400m | 5.20 deg -> 4.27 deg | 0.499 | 0.0640m -> 0.0333m |
| C1 | AABB | 0.3155m -> 0.2010m | 8.53 deg -> 6.11 deg | 0.388 | 0.0598m -> 0.0322m |
| H3 | AAAA | 0.0043m -> 0.0040m | near zero | 0.009 | stable |
| C1 | AAAA | 0.0078m -> 0.0069m | near zero | 0.035 | stable |

### THuman In-Domain

| Variant | Subset | Camera trans | Gate | Human trans |
| --- | --- | ---: | ---: | ---: |
| H3 | AABB | 0.1688m -> 0.0464m | 0.445 | 0.0655m -> 0.0581m |
| C1 | AABB | 0.0758m -> 0.0513m | 0.163 | 0.0722m -> 0.0575m |
| H3 | AAAA | 0.0022m -> 0.0022m | 0.001 | stable |
| C1 | AAAA | 0.0097m -> 0.0082m | 0.023 | stable |

### zxc Held-Out

| Variant | Subset | Camera trans | Camera rot | Gate | Human trans |
| --- | --- | ---: | ---: | ---: | ---: |
| H3 | AABB | 0.2559m -> 0.1617m | 5.28 deg -> 4.81 deg | 0.498 | 0.1048m -> 0.1247m |
| C1 | AABB | 0.2680m -> 0.2480m | 8.36 deg -> 8.08 deg | 0.385 | 0.1064m -> 0.1348m |
| H3 | AAAA | 0.0037m -> 0.0036m | near zero | 0.009 | stable |
| C1 | AAAA | 0.0075m -> 0.0066m | near zero | 0.036 | stable |

## Selected zxc Visualization Cases

Two large-angle held-out AABB clips were selected for subjective comparison:

| Case | Sequence | Start | Angle |
| --- | --- | ---: | ---: |
| 1 | `zxc/22070932 -> zxc/22053912` | 1663 | 174.45 deg |
| 2 | `zxc/22053925 -> zxc/22053917` | 1545 | 173.81 deg |

Selected-case metrics:

| Variant | Raw trans | Corrected trans | Gate |
| --- | ---: | ---: | ---: |
| H3 | 0.2991m | 0.1140m | 0.492 |
| C1 | 0.2653m | 0.2119m | 0.451 |

H3 gave much stronger camera translation correction on these held-out large-angle cases.

## Conclusion

H3 is the better current direction:

- Better camera translation correction on AvatarReX, THuman, and zxc held-out.
- Lower AAAA gate, so it is less likely to over-correct normal continuous clips.
- More useful behavior on selected subjective cases.

C1 is useful as a control because it shows that simply increasing correction pressure is not enough. The deadzone/margin design in H3 is important.

## Next Experiments

The next three experiments should use the same 60h data and original Human3R initialization:

1. `h3_improve075_or_010`: keep H3 gate/deadzone, increase `improvement_weight` moderately.
2. `h3_human_cam_relative`: add a human-camera relative-position loss to constrain the relative placement between camera and SMPL translation.
3. `h3_human_cam_pairwise`: add both human-camera relative loss and pairwise temporal relative-motion loss.

The main thing to test is whether these losses reduce the residual mismatch where the camera looks correct but the human/scene relative depth is still not fully aligned.
