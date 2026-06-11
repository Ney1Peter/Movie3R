# V8.1 Experiment Log and Memory Plan

## 1. Current Task

We want to correct Human3R / CUT3R camera pose drift in online reconstruction, especially around camera/view changes in an AABB sequence:

```text
A_t, A_t+1, B_t+2, B_t+3
```

The scene and time are continuous, but the camera changes from view A to view B. The main observed failure is that Human3R predicts the camera pose after the view switch with an offset, so the person and scene are not placed consistently in one world coordinate system.

The V8 idea follows UniCon3R's prompt-based correction style, but changes the target:

```text
UniCon3R:
human + scene + contact + temporal cues
-> contact prompt
-> human/contact refinement

Movie3R V8:
human + local geometry + history/memory
-> pose correction prompt
-> camera pose residual correction
```

At this stage we are not training a new module. We are validating which information can act as a reliable correction anchor.

## 2. Human3R Flow Assumption

The working mental model of Human3R is:

1. Input frames go through encoder/tokenizers to obtain image tokens, camera/pose token, human tokens, and related features.
2. These tokens interact with recurrent state tokens in the decoder through attention.
3. The decoder outputs refined image, pose, human tokens, then downstream heads predict pointmap/depth, camera pose, SMPL-X body, mask, confidence, etc.

The V8 correction module should be a UniCon-style decoder-in prompt/refinement branch around existing tokens, not a new reconstruction model.

## 3. What UniCon3R Adds

From UniCon3R, the most relevant components are:

| UniCon3R component | Meaning | V8 analogue |
|---|---|---|
| `Ht` | human prompt / human token | pelvis, torso, feet, human token |
| `Ft` | current image tokens | current frame body/scene tokens |
| `St-1` | recurrent scene state | Human3R/CUT3R latent memory |
| `Ucurr = CA(Ht, Ft)` | current-frame scene context read by human query | current human-centered token context |
| `Umem = CA(Ht, St-1)` | historical scene/memory context | recurrent latent memory cue |
| `gamma_t` | gate between current and memory context | current-vs-memory reliability gate |
| `Gt` | local metric geometry token | local floor/foot/human geometry token |
| `Mt` | previous refined contact token | previous pose correction token / correction momentum |
| latent refinement | contact token updates human latent | correction token updates pose residual |

Important point: UniCon3R does not only use local scene. It also uses recurrent state memory, temporal momentum, and a learned gate.

## 4. Data Case

Current single AABB test case:

```text
view0: 22070932 @ frame 820
view1: 22070932 @ frame 821
view2: 22070935 @ frame 822  (A -> B boundary)
view3: 22070935 @ frame 823
```

Raw data:

```text
/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1
```

Processed data:

```text
/data/wangzheng/iJCV-CODE/data/Avatarrex_output
```

Important convention fixed during debugging:

```text
AvatarReX calibration stores world-to-camera:
X_cam = R @ X_world + T

c2w[:3,:3] = R.T
c2w[:3, 3] = -R.T @ T
```

With this convention, original SMPL projection aligns with masks well.

## 5. Completed Tests

### 5.1 Explicit SMPL / Mask Projection

We fixed the previous camera/SMPL convention bug and verified that original SMPL projects correctly onto the masks.

Observed SMPL-mask bbox IoU:

```text
22070932 @ 820/821: about 0.88 - 0.89
22070935 @ 822/823: about 0.93 - 0.94
```

This validates that explicit SMPL anchors can be used as a sanity-check proxy.

### 5.2 Token Correspondence

We ran frozen encoder token probes and checked token similarity heatmaps for:

```text
pelvis
torso
left_foot
right_foot
near_foot
near_human
```

Results:

- `pelvis`, `torso`, `left_foot`, `right_foot` are strong and stable.
- `near_foot` is usable but weaker than human body parts.
- `near_human` is weaker and should remain optional/ablation-only.

This means explicit correction experiments should only use anchors that token probes can also locate.

### 5.3 GT Camera Oracle

We built a corrected viewer by replacing Human3R raw camera poses with AvatarReX GT-relative cameras:

```text
T_corr_i = T_raw_0 @ inverse(T_gt_0) @ T_gt_i
```

This used only GT camera extrinsics, not human or scene tokens.

Observation:

- Camera direction/view relation improves.
- But the person/scene can still show residual mismatch because Human3R predicted local geometry and SMPL may have its own scale/shape/depth inconsistency.

Conclusion:

- GT camera oracle is useful as a visual sanity check.
- It does not validate the human-centric correction hypothesis.

### 5.4 Full-Body Human-Only Oracle

We then built a human-only explicit oracle:

```text
Human3R predicted SMPL camera-space joints
align to AvatarReX GT SMPL world joints
-> estimate camera pose
```

This did not use GT camera, scene, pointmap, or tokens directly.

Using many stable body joints worked very well.

### 5.5 Token-Aligned Human-Only Oracle

To make explicit proxy match token feasibility, we restricted the anchors to only:

```text
pelvis
torso
left_foot
right_foot
```

These correspond to the strongest token heatmap results.

Result:

```text
raw human anchor RMSE, frame2/frame3: 0.667 / 0.455
token-aligned human-only RMSE:       0.029 / 0.035
```

Observation:

- The A -> B boundary correction is largely solved by these four human anchors.
- B -> B still has a slight residual drift, but the result is already strong.

Current best viewer:

```text
output/v8_1_human3r_aabb_compare/token_aligned_human_only
```

### 5.6 Human + Local Background

We tested local scene/background additions using token-aligned regions:

```text
near_foot
near_human
near_foot + near_human
```

Variants:

- direct joint rigid fitting with human + scene points
- low scene weight
- human-first pose, then small scene translation residual

Result:

Adding explicit scene points did not improve the current case. In several variants it pulled the strong human solution away.

Representative numbers:

```text
token-human only:
  frame2 human RMSE = 0.029
  frame3 human RMSE = 0.035

nearfoot residual:
  frame2 human RMSE = 0.085
  frame3 human RMSE = 0.087
```

Conclusion:

- Local background feature/point alignment is not reliable in this low-texture case.
- This matches the original motivation: low texture makes background matching weak.
- Scene cues should be weak reliability/geometry cues, not strong pose anchors for now.

### 5.7 Near-Foot Floor Normal Probe

We tested a low-texture-specific idea: use the near-foot area as a floor candidate and estimate a local plane normal.

Procedure:

1. Use frozen encoder near-foot token heatmap.
2. Gate explicit near-foot background region by high token similarity.
3. Fit local plane normal from GT depth and Human3R predicted depth.
4. Draw normal arrows on heatmap overlays.

Outputs:

```text
output/v8_1_floor_normal_token_probe/floor_normal_heatmap_grid.png
output/v8_1_floor_normal_token_probe/normal_heatmaps/
```

Key result:

```text
GT depth normal:
A-A:  0.35 deg
A->B: 1.74 deg
B-B:  1.16 deg

Human3R depth normal in raw world:
A-A:  0.22 deg
A->B: 15.54 deg
B-B:  10.50 deg
```

Interpretation:

- GT geometry confirms that token-gated near-foot floor is a stable plane.
- Human3R depth can produce a normal, but it is noisy and sometimes wrong, especially on the second B frame.
- Floor normal is not yet safe as a strong correction anchor.
- It may still be useful as a geometry inconsistency/reliability feature.

## 6. Concept Clarification: What Counts as Memory?

The current token-aligned human-only oracle uses cross-frame human alignment, but it is not recurrent latent memory in the UniCon3R sense.

Four levels should be separated:

| Level | Meaning | Current status |
|---|---|---|
| single-frame human anchor | current pelvis/torso/feet | validated |
| explicit temporal human anchor | previous corrected human world position / human trajectory | partially validated through explicit oracle |
| recurrent latent memory | Human3R/CUT3R internal `state_feat` and `pose_retriever mem` | not yet tested |
| correction momentum | previous correction token / delta pose / gate | not yet implemented |

Current human-only oracle is best described as:

```text
explicit human trajectory alignment oracle
```

It proves that human anchors can correct camera pose, but it does not prove that internal recurrent state tokens carry the same information.

## 7. How to Test Recurrent Latent Memory

Human3R has two relevant memory structures:

### 7.1 Global Recurrent State

In `model_human3r.py`:

```text
state_feat: [B, state_size, dec_dim]
```

It is initialized from learned state tokens and updated by the state decoder.

It is the closest match to UniCon3R's `St-1`.

### 7.2 Pose Retriever Memory

Also in `model_human3r.py`:

```text
mem: [B, local_mem_size, 2 * dec_dim]
```

It stores pairs of global image feature and refined pose feature:

```text
new_mem = pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
pose_feat_i = pose_retriever.inquire(global_img_feat_i, mem)
```

This is highly relevant for pose correction, because it is already designed as pose memory.

### 7.3 Engineering Detail

`model.forward(..., ret_state=True)` can return:

```text
(state_feat, state_pos, init_state_feat, mem, init_mem)
```

However, the `demo.py` path uses `forward_recurrent_lighter`. That function has `ret_state=True` in the interface, but currently does not append per-frame state into `all_state_args`. Therefore:

- direct demo saved outputs do not include state memory;
- to dump state, either use non-lighter forward, or add a read-only dump hook to `forward_recurrent_lighter`.

## 8. Can State / Momentum / Gate Be Visualized?

### 8.1 Recurrent Latent Memory

It cannot be visualized as directly as SMPL joints or masks.

Reasons:

- `state_feat` is high-dimensional latent tokens.
- State tokens are not guaranteed to correspond to image pixels.
- `pose_retriever mem` is an abstract memory bank, not a 2D map.

Possible visualizations:

- token norm over state index / pseudo-grid;
- cosine similarity between current human/body token and state tokens;
- cross-attention weights from human query to state tokens, if hooked;
- PCA/UMAP over state tokens across frames;
- correlation between memory features and known drift/error proxies.

These are diagnostic visualizations, not direct semantic proof.

### 8.2 Temporal Momentum

Temporal momentum is not a native explicit output unless we define it.

Possible V8 version:

```text
M_pose_t =
MLP(previous A_corr,
    previous delta_xi,
    previous gate,
    previous corrected pelvis/feet world anchors,
    previous pose velocity)
```

This can be partially visualized through curves:

- previous/current pelvis world displacement
- support foot displacement
- correction magnitude
- pose jump score

But the latent token itself is not directly human-interpretable.

### 8.3 Current-vs-Memory Gate

The gate is also not explicit unless we implement it.

Possible inputs:

```text
gate_t = sigmoid(MLP(
    current human token reliability,
    current pose jump score,
    memory-human consistency,
    previous correction confidence,
    local geometry reliability
))
```

This is visualizable as a scalar curve over time, not as a spatial heatmap.

## 9. Recommended Next Experiments

### Experiment A: Recurrent Latent Memory Dump

Goal:

Check whether `state_feat` / `mem` changes meaningfully at A -> B and B -> B.

Outputs:

- per-frame state/mem tensor shapes
- norms and cosine changes
- similarity between current human anchor token and previous memory
- simple plots over the 4 frames

No correction yet.

### Experiment B: Memory Retrieval Probe

Goal:

Test whether a current human query can retrieve useful history from `state_feat` or `mem`.

Possible probes:

```text
query = pooled current pelvis/torso/feet tokens
memory = previous state_feat or mem
score = attention(query, memory)
```

Validate against:

- raw pose error proxy
- human anchor drift
- A -> B boundary
- B -> B residual instability

### Experiment C: Explicit Temporal Human Anchor

Goal:

Separate "human history" from "latent memory".

Use previous corrected human anchors:

```text
previous pelvis/torso/feet world positions
current Human3R pelvis/torso/feet camera positions
-> estimate residual pose correction
```

This is more interpretable than state memory and may already be enough.

### Experiment D: Temporal Momentum Token

Goal:

After `A_corr_t` exists, carry previous correction token forward:

```text
M_t = MLP(A_corr_{t-1}, delta_pose_{t-1}, gate_{t-1})
```

Start with scalar diagnostics before training:

- correction magnitude curve
- residual human anchor error
- pose velocity / acceleration

### Experiment E: Current-vs-Memory Gate

Goal:

Avoid over-correcting stable frames and decide when to trust history.

Candidate gate cues:

- human token confidence
- body-anchor consistency
- pose jump score
- current-vs-previous human anchor mismatch
- floor normal reliability, if stable
- pointmap confidence near human

First version can be a heuristic gate before becoming learnable.

## 10. Current Best Direction

Based on V8.1 evidence, the strongest path is:

```text
pelvis / torso / left_foot / right_foot
+ explicit temporal human history
+ recurrent latent memory probe
+ small residual correction
+ gate
```

Scene/background should remain optional and weak until a more reliable cue is found.

Near-foot floor normal is interesting, but in this case it is not stable enough on Human3R predicted depth to be a strong correction anchor.

## 11. Recurrent Memory / Momentum / Gate Probe

Added script:

```text
scripts/v8_1_probe_memory_momentum_gate.py
```

Run command used for the current AABB case:

```bash
export PYTHONPATH=src:. && export MPLCONFIGDIR=/tmp/matplotlib && CUDA_VISIBLE_DEVICES=7 \
.venv/bin/python scripts/v8_1_probe_memory_momentum_gate.py \
  --output_dir output/v8_1_memory_momentum_gate_probe \
  --device cuda
```

Main outputs:

```text
output/v8_1_memory_momentum_gate_probe/anchor_overlays/
output/v8_1_memory_momentum_gate_probe/state_memory_heatmaps/state_anchor_similarity_panel.png
output/v8_1_memory_momentum_gate_probe/state_memory_heatmaps/memory_query_similarity_panel.png
output/v8_1_memory_momentum_gate_probe/momentum_gate_curves/temporal_momentum_curves.png
output/v8_1_memory_momentum_gate_probe/momentum_gate_curves/state_memory_curves.png
output/v8_1_memory_momentum_gate_probe/momentum_gate_curves/gate_proxy_curves.png
output/v8_1_memory_momentum_gate_probe/memory_momentum_gate_metrics.csv
```

The probe only uses token-accessible sources:

- body anchor image tokens: pelvis, torso, left foot, right foot
- Human3R predicted SMPL anchors from saved output
- raw Human3R camera pose
- recurrent `state_feat`, `state_pos`, and `mem` returned by `ret_state=True`
- global image token projected through `pose_retriever.proj_q`

It does not use GT camera pose, GT pointmap, background matching, or body anchors outside the four token-validated parts for the gate proxy.

### 11.1 Ret-State Dump Sanity Check

The lighter recurrent path now returns 5 state snapshots for 4 input frames:

```text
snapshot 0: initial state before frame 0
snapshot 1: state after frame 0
snapshot 2: state after frame 1
snapshot 3: state after frame 2
snapshot 4: state after frame 3
```

Observed shapes:

```text
state_feat: [768, 768]
mem:        [256, 1536]
image token grid: [32, 23]
```

This confirms recurrent memory can be dumped without changing the model architecture.

### 11.2 Temporal Human Momentum Result

The token-aligned human momentum cues are strongly useful on this case.

At the A -> B boundary frame (`view2_B_t2_boundary`):

```text
raw camera center step:      1.132
raw human anchor step mean:  0.633
prev-human fit delta t:      4.382
prev-human fit delta rot:    106.8 deg
```

On the normal A -> A step (`view1_A_t1`), the same cues are much smaller:

```text
raw camera center step:      0.009
raw human anchor step mean:  0.038
prev-human fit delta t:      0.201
prev-human fit delta rot:    4.26 deg
```

This supports using explicit temporal human anchors as a first-class prompt source.

Candidate token for V8:

```text
A_history_human =
  pelvis/torso/left_foot/right_foot token at t
  + previous corrected/world anchor positions
  + previous support/human motion state
```

### 11.3 Recurrent Latent Memory Result

The recurrent state is dumpable and can be visualized by comparing body anchor decoder tokens against `state_feat`.

The state-anchor heatmaps are structured, so the state is not random noise. However, the current direct memory score:

```text
global image token -> LocalMemory key top-k cosine
```

is not yet a reliable gate by itself. It does not peak cleanly at the A -> B boundary; in this 4-frame case, the raw memory disagreement is also high at `view1_A_t1`.

Interpretation:

- `state_feat` may still be useful as a latent context token.
- Direct cosine against `mem` keys is too crude.
- If memory is used later, it should probably enter through a small adapter/cross-attention block, not as a hand-written scalar gate.

Candidate token for V8:

```text
A_state =
  cross-attention(query=body/pose tokens, key_value=state_feat or mem)
```

but not:

```text
gate = raw_cosine(global_img_token, mem_key)
```

### 11.4 Gate Proxy Result

Two heuristic gates were plotted:

```text
token_aligned_gate_no_memory
token_aligned_gate_with_memory
```

The no-memory gate is cleaner for this case:

```text
view0_A_t:             0.00
view1_A_t1:            0.08
view2_B_t2_boundary:   1.00
view3_B_t3:            0.13
```

The with-memory gate is still high at the boundary, but it is less clean because the simple memory disagreement cue is noisy:

```text
view0_A_t:             0.00
view1_A_t1:            0.27
view2_B_t2_boundary:   0.83
view3_B_t3:            0.11
```

Current conclusion:

```text
Use temporal human momentum + pose jump as the first gate baseline.
Keep recurrent memory as an optional latent prompt, not as a scalar gate yet.
```

### 11.5 Updated Prompt Priority

For the next pose-correction token ablation, prioritize:

```text
1. A_human_parts:
   pelvis, torso, left foot, right foot tokens

2. A_history_human:
   previous token-aligned human anchor state and motion

3. A_camera_motion:
   raw camera relative pose, velocity, acceleration, jump score

4. A_gate:
   camera jump + human anchor jump + previous-human fit delta

5. A_state_memory:
   optional recurrent latent context through adapter attention
```

Do not prioritize background / near-foot floor normal / raw memory cosine for the first correction token.

## 12. Online Human-Motion Correction Baseline

Added script:

```text
scripts/v8_1_build_online_human_motion_correction.py
```

Run command used:

```bash
export PYTHONPATH=src:. && export MPLCONFIGDIR=/tmp/matplotlib && \
.venv/bin/python scripts/v8_1_build_online_human_motion_correction.py --device cpu
```

Viewer command used for the gated output:

```bash
export PYTHONPATH=src:. && export MPLCONFIGDIR=/tmp/matplotlib && \
.venv/bin/python scripts/view_human3r_saved_output.py \
  --output_dir output/v8_1_human3r_aabb_compare/online_human_motion_gated \
  --raw_output_dir output/v8_1_human3r_aabb_compare/raw \
  --viewer_port 8096 \
  --device cpu \
  --vis_threshold 2 \
  --downsample_factor 1 \
  --smpl_downsample 1 \
  --camera_downsample 1
```

Open:

```text
http://127.0.0.1:8096
```

### 12.1 What This Baseline Tests

This is the missing online validation step.

Previous `token_aligned_human_only` showed:

```text
If four token-aligned body anchors are aligned to the correct human trajectory,
camera pose can be pulled back.
```

But that target trajectory was effectively an oracle, because it came from AvatarReX GT SMPL world anchors.

The new online baseline removes that oracle.  It only uses:

- current Human3R predicted SMPL camera-space anchors
- previous corrected world anchors
- raw Human3R camera pose for the residual/gate

It does not use:

- GT camera pose
- GT SMPL world trajectory
- future frames
- background matching
- body anchors outside pelvis / torso / left foot / right foot

### 12.2 Online Algorithm

For frame 0:

```text
T_corr_0 = T_raw_0
save corrected world anchors:
  pelvis_0, torso_0, left_foot_0, right_foot_0
```

For each later frame:

```text
current anchors in camera frame:
  pelvis_cam_t, torso_cam_t, left_foot_cam_t, right_foot_cam_t

history target:
  previous corrected world anchors

fit a rigid camera pose:
  T_fit_t @ current_anchors_cam_t
  ~= previous_corrected_anchors_world
```

Two variants are written:

```text
online_human_motion_always:
  gate = 1 for every frame after frame 0

online_human_motion_gated:
  gate = clip((raw_anchor_history_rmse - 0.08) / (0.35 - 0.08), 0, 1)
```

The gated version is closer to the intended V8 gating behavior because normal frames should not be over-corrected.

Important clarification:

```text
The "corrected history" for frame 1 comes from frame 0.
Frame 0 has no previous frame, so it keeps T_raw_0 as T_corr_0.
The four body anchors under T_corr_0 are then saved as the first history.
```

Therefore, the online chain is:

```text
frame 0:
  raw pose is used as the initial world frame
  save corrected anchors from raw frame 0

frame 1:
  compare current raw anchors with frame-0 corrected anchors
  if the residual is small, keep raw pose
  save frame-1 final anchors

frame 2 and later:
  compare current raw anchors with the previous final anchors
  if the residual is large, fit a camera correction
```

This test is not yet a UniCon-style decoder-in correction prompt. It is a token-aligned explicit-anchor baseline:

```text
current implementation:
  Human3R predicted SMPL joints -> four explicit body-anchor coordinates

why this is still useful:
  these four parts were already validated as token-accessible regions
  and the online test proves their history can correct camera pose

next step:
  replace explicit anchor coordinates / hand-written residuals
  with A_corr_t tokens that enter the decoder
  and a lightweight residual latent head after the decoder
```

### 12.3 Outputs

Saved-output directories:

```text
output/v8_1_human3r_aabb_compare/online_human_motion_always/
output/v8_1_human3r_aabb_compare/online_human_motion_gated/
```

Diagnostics:

```text
output/v8_1_human3r_aabb_compare/online_human_motion_diagnostics/online_human_motion_metrics.png
output/v8_1_human3r_aabb_compare/online_human_motion_diagnostics/online_human_motion_anchor_trajectory_xz.png
output/v8_1_human3r_aabb_compare/online_human_motion_summary.json
```

### 12.4 Result

The gated online version behaves as intended on the current AABB case:

```text
view0_A_t:
  gate = 0

view1_A_t1:
  raw history anchor RMSE = 0.040
  gate = 0
  no correction

view2_B_t2_boundary:
  raw history anchor RMSE = 0.637
  gate = 1
  corrected history anchor RMSE = 0.019

view3_B_t3:
  raw history anchor RMSE = 0.426
  gate = 1
  corrected history anchor RMSE = 0.029
```

This means the token-aligned human history cue does not merely detect drift; it can explicitly pull the pose back in a causal setting.

### 12.5 Updated Interpretation

The strongest validated V8 cue is now:

```text
pelvis / torso / left_foot / right_foot current tokens
+ previous corrected human-anchor memory
+ raw camera jump / anchor history residual gate
```

In prompt-token language:

```text
A_body_part_t:
  current pelvis, torso, left foot, right foot tokens

A_history_human_t:
  previous corrected world anchors or their embedded history state

A_gate_t:
  raw pose jump + current-vs-history human anchor residual
```

This is stronger and more interpretable than the current recurrent-memory cosine probe.

### 12.6 Additional Constant-Velocity History Run

Also ran:

```bash
export PYTHONPATH=src:. && export MPLCONFIGDIR=/tmp/matplotlib && \
.venv/bin/python scripts/v8_1_build_online_human_motion_correction.py \
  --device cpu \
  --history_mode constant_velocity \
  --output_prefix online_human_motion_cv
```

Outputs:

```text
output/v8_1_human3r_aabb_compare/online_human_motion_cv_always/
output/v8_1_human3r_aabb_compare/online_human_motion_cv_gated/
output/v8_1_human3r_aabb_compare/online_human_motion_cv_diagnostics/
output/v8_1_human3r_aabb_compare/online_human_motion_cv_summary.json
```

The constant-velocity version is a useful control, but on this 4-frame AABB case it is not clearly better than the simpler previous-anchor target.  The default recommendation remains:

```text
history_mode = previous
variant = gated
```

Reason:

- `previous + gated` does not correct the normal A -> A step.
- It strongly corrects A -> B.
- It also keeps B -> B close to the corrected human-anchor history.
- The method is simpler and less sensitive to short-term human articulation noise.

## 13. V8.1 Decoder-In Pose Prompt Overfit Success and Coordinate Fix

### 13.1 Result

On 2026-05-30, the V8.1 UniCon-style decoder-in pose prompt successfully overfit one AvatarReX AABB sample using the correct raw calibration camera pose target.

The successful run used:

```text
sample:
  seq_a = 22010710
  seq_b = 22053923
  start_frame = 0

config:
  config/train_v8_pose_prompt_overfit_1sample_start0_nodepth_rawpose.yaml

checkpoint:
  /tmp/movie3r_v8_1_pose_prompt_overfit_1sample_start0_nodepth_rawpose_gpu4/checkpoint-last.pth

viewer:
  http://127.0.0.1:8112
```

Final eval:

```text
corrected trans err = 0.0075
corrected rot err   = 0.0617 deg
raw trans err       = 0.1154
raw rot err         = 4.7876 deg
```

The visual result matches expectation: the B-camera frames are no longer upside down.

### 13.2 Why the Earlier Low-Loss Result Was Wrong

The earlier overfit run used the processed `camera_pose` stored in:

```text
/data/wangzheng/iJCV-CODE/data/Avatarrex_output/Training/<seq>/cam/*.npz
```

That target can make the loss decrease, but for the AABB B camera it contains a roll/up-axis flip in the viewer convention:

```text
processed camera_pose B frame:
  z-axis ~= -1
  y-axis ~= -1   # wrong: upside-down camera
```

Therefore the model did learn the target, but the target itself was wrong for the Human3R saved-output viewer and raw SMPL/camera sanity check.

### 13.3 Correct Camera Convention

The correct AvatarReX camera target for this experiment comes from:

```text
/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1/calibration_full.json
```

Raw calibration convention:

```text
X_cam = R_w2c @ X_world + T_w2c
```

Convert to viewer/training c2w:

```text
R_c2w = R_w2c.T
t_c2w = -R_w2c.T @ T_w2c
```

Use the relative pose to the first frame as the loss target:

```text
T_target_i = inv(raw_camera_pose_0) @ raw_camera_pose_i
```

Correct B-frame axes:

```text
raw calibration B frame:
  z-axis ~= -1
  y-axis ~= +1
```

This keeps the 180-degree viewpoint change but preserves the up direction.

### 13.4 DA3 Depth Boundary

AvatarReX `depth/*.npy` in `Avatarrex_output` should not be treated as metric GT depth for V8.1 pose correction. It is DA3 / monocular pseudo-depth and can have arbitrary or inconsistent scale.

For V8.1 pose prompt training:

```text
load_da3_depth = False
```

Allowed:

- Use zero-filled `depthmap` in dataloader for pose-only training compatibility.
- Use Human3R predicted pointmap/depth as model output or visualization cue.
- Use raw calibration camera + raw SMPL in no-depth viewer for coordinate sanity.

Not allowed:

- Do not validate pose with raw calibration camera + DA3 pointcloud.
- Do not supervise camera/world geometry with DA3 depth.
- Do not use DA3 depth to decide whether A/B camera alignment is correct.

### 13.5 Practical Rule

Before trusting any V8.1 AvatarReX result, print the target axes:

```text
expected:
  frame 0/1: y-axis ~= +1, z-axis ~= +1
  frame 2/3: y-axis ~= +1, z-axis ~= -1

wrong:
  frame 2/3: y-axis ~= -1
```

If the B-frame `y-axis` is negative, the target is the old processed pose and the result will look upside down even if the training loss is excellent.
