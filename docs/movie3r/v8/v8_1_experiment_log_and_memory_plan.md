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

The V8 correction module should be a plugin around these outputs/tokens, not a new reconstruction model.

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
