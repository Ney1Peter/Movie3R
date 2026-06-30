# V9 Experiment Recap - 2026-06-30

## Purpose

This note summarizes the V9 experiments so far: model changes, token ablations, loss ablations, subjective findings, and the current long-running follow-up trainings.

The main project goal is to make Human3R robust to discontinuous clips, multi-view jumps, and camera/world gauge drift while keeping the original forward, streaming behavior.

## Core Problem

The first assumption was:

```text
If camera pose is corrected, the whole scene and human should become aligned.
```

Experiments showed this was incomplete. Human3R has a separate human branch, and the SMPL output has its own translation / latent placement. Therefore camera pose can look correct while the human mesh is still shifted in depth or height.

Current interpretation:

- Camera pose correction is necessary.
- Human placement correction is also necessary.
- We should not try to improve detailed human reconstruction quality in this stage; the target is world/camera alignment.
- The correction should stay feed-forward and streaming, using only input frames and model internal state at inference time.

## Model Evolution

### 1. Pose-only Correction

The first working version added a pose correction token and a pose residual head.

```text
correct token -> pose residual head -> delta pose token + gate
raw pose token + gate * delta -> original pose head -> corrected camera pose
```

This improved camera translation, but it did not fully fix human/scene misalignment because SMPL translation could still be wrong.

### 2. Explicit Human Translation Correction

The diagnostic version added a direct human translation correction head:

```text
correct token -> human translation head -> delta SMPL transl
```

Single-clip overfit showed that the method could move the human back into the right place. This proved the problem was solvable, but this branch was too explicit and less aligned with the UniCon3R-style latent correction design.

### 3. Implicit Human Latent Correction

The current V9 design corrects the human latent token before the original Human3R human head:

```text
refined correct token + refined human token + corrected pose token
  -> human latent residual head
  -> delta human token + shared gate
  -> corrected human token
  -> original Human3R human head
  -> corrected SMPL
```

This keeps the correction branch closer to UniCon3R: the new branch predicts a residual in latent space, then lets the original head decode the final output.

### 4. LoRA Head Tuning

The better V9 setting uses correction branch training plus LoRA on Human3R heads:

- pose head LoRA
- human head LoRA
- original Human3R weights as initialization

LoRA is useful because it gives the original heads a small adaptation path without fully overwriting Human3R's learned capability.

## Correct Token Design

The current correct token is a relation prompt, not a hand-coded body-part tracker.

It is built from three information streams:

| Token | Meaning | Why it helps |
|---|---|---|
| semantic token | current image / pose / human tokens plus state memory | tells the model what the current frame looks like and what Human3R currently believes |
| alignment token | current pose token, previous pose token, pose-memory difference | tells the model whether camera motion looks like normal continuous motion or a jump |
| momentum token | previous refined correct token, previous delta, previous gate | gives temporal continuity to correction behavior |

These tokens enter the decoder together with image, pose, human, and state tokens. After decoder interaction, the refined correct tokens feed the pose correction head and human latent correction head.

This was inspired by UniCon3R's contact-token pattern, but adapted to our task. UniCon3R uses a contact relation token to refine human-scene contact; V9 uses relation correction tokens to refine camera-human alignment.

## Token Ablations

Small10 token ablations were trained on 7 AvatarReX + 3 THuman training clips. The purpose was not final performance, but to understand which correct-token structure is useful.

Tested directions:

| Variant | What changed | Interpretation |
|---|---|---|
| all_mean | semantic + alignment + momentum, mean pooled | original V9 baseline |
| single_token | compress relation info into one token | checks whether multiple tokens are necessary |
| no_semantic | remove semantic stream | tests whether current image/human context matters |
| no_alignment | remove pose-alignment stream | tests whether camera-motion mismatch matters |
| no_momentum | remove temporal correction memory | tests whether previous correction state matters |
| learned_pooling | learned attention pooling over relation tokens | lets model choose token importance |
| global_weighted | learn global token weights | simpler learned weighting |
| all_concat / contact-style | keep all relation tokens separately and fuse by concat MLP | closest to "use all contact-like relation evidence" |

### Latest Pooling Result

The latest objective benchmark compared `global_weighted` and `all_concat` on `benchmark_mixed_small18`.

| Variant | AABB camera trans | AABB improve | AABB human trans | AAAA gate | AABB loss |
|---|---:|---:|---:|---:|---:|
| global_weighted | 0.580 -> 0.403 m | 0.177 m | 0.271 -> 0.120 m | 0.147 | 1.412 |
| all_concat | 0.487 -> 0.274 m | 0.213 m | 0.297 -> 0.116 m | 0.055 | 0.892 |

Subjective note from the training-sequence visualization:

```text
The contact-style / all_concat behavior looks visually just right.
```

Current interpretation:

- `all_concat` preserves more relation information than global averaging.
- It corrects AABB more strongly while keeping AAAA gate lower.
- It is the best current correct-token pooling candidate.
- There is a tooling issue: full 4-frame demo-style saved-output inference with `all_concat` was abnormally slow, while benchmark eval and 1/2/3-frame saved-output inference worked. This should be checked before using it heavily for visualization.

## Loss Ablations

### Baseline V9 Loss

The current V9 loss family includes:

| Loss | Purpose |
|---|---|
| pose loss | corrected camera pose should match GT |
| drift / gate loss | gate should reflect raw Human3R drift |
| improvement margin loss | corrected pose should be better than raw pose |
| residual small loss | pose latent residual should not become unbounded |
| human translation loss | corrected SMPL translation should match GT |
| human delta small loss | human latent correction should stay residual-like |
| LoRA norm loss | LoRA should not destroy original head behavior |

GT is used only for training losses and benchmark metrics. Inference remains forward-only and does not see GT.

### C1 vs H3

The 60h sweep compared stronger correction pressure against a more selective gate design.

| Variant | Idea | Result |
|---|---|---|
| C1 | stronger drift + stronger improvement loss | more aggressive, but less stable |
| H3 | deadzone gate + weaker human delta regularization | best current loss direction |

H3 worked better because it says:

```text
Do not correct tiny normal errors, but allow stronger human/camera correction once drift is real.
```

This gave better AABB correction and lower AAAA over-correction risk.

### Current Three Long Follow-up Runs

The current three long trainings are based on the H3 loss direction:

| Session | Output dir | Main change | Current status at 2026-06-30 11:22 |
|---|---|---|---|
| `v9_60h_h3_imp075_gpu5` | `output/v9_60h_loss_followup/v9_60h_h3_imp075_pose_human_lora_bs10` | increase improvement pressure to 0.075 | around epoch 66, still running |
| `v9_60h_h3_hcam_ref_gpu6` | `output/v9_60h_loss_followup/v9_60h_h3_hcam_ref_pose_human_lora_bs10` | add human-camera reference-frame loss | around epoch 64, still running |
| `v9_60h_h3_hcam_pair_gpu7` | `output/v9_60h_loss_followup/v9_60h_h3_hcam_ref_pairwise_pose_human_lora_bs10` | add human-camera reference + pairwise motion loss | around epoch 56, still running |

These runs should answer whether extra human-camera coupling reduces the remaining failure case where camera pose looks right but human/scene relative depth is not fully aligned.

## Data And Training Design

Important training decisions:

- Use original Human3R checkpoint as initialization for every new formal run.
- Use `resize_only_16` to avoid crop-induced coordinate mistakes.
- Do not mix AvatarReX and THuman with forced square padding when their aspect ratios differ.
- Use dataset-aware / resolution-aware mixed training: AvatarReX batches and THuman batches can be forwarded separately, then gradients accumulated before optimizer step.
- Include both AABB drift clips and AAAA stable clips so gate learns when not to correct.

## Evaluation Design

Important objective metrics:

| Metric | Meaning | Direction |
|---|---|---|
| camera translation error | corrected camera center vs GT | lower is better |
| camera rotation error | corrected camera rotation vs GT | lower is better |
| translation improvement | raw error minus corrected error | higher is better |
| gate mean | correction activation | high on AABB, low on AAAA |
| human translation error | corrected SMPL translation vs GT | lower is better |
| MPJPE / PVE | absolute human joint/vertex error | lower is better |
| PA-MPJPE / PA-PVE | Procrustes-aligned human shape/pose error | lower is better |

Subjective checks remain necessary because metrics can improve while point cloud, SMPL, and camera frame are visually inconsistent.

## Current Conclusions

1. Pose-only correction is not enough; camera and human placement must both be corrected.
2. Explicit SMPL translation correction proved the capability exists, but implicit human latent correction is the better model design.
3. Pose + human head LoRA is useful as a small adaptation path and should remain in the current formal setting.
4. H3-style gate/loss is better than simply increasing correction pressure.
5. The all-concat / contact-style correct-token pooling is the best current token candidate based on both metrics and subjective viewing.
6. The remaining important risk is generalization to fast-moving humans and longer sequences; pairwise human-motion loss is being tested for that.

## Immediate Next Steps

1. Let the three long follow-up trainings finish or reach a clear plateau.
2. Run the same held-out zxc / AvatarReX / THuman benchmarks for all three.
3. Visualize the same two or three fixed subjective clips for all three.
4. Compare against the current best H3 baseline and the new all-concat/contact-style token candidate.
5. If all-concat remains best, port it into the next larger formal training setup after fixing or bypassing the slow 4-frame saved-output visualization path.
