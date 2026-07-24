# Body-Part Guided Correct Token Plan

本文档记录下一阶段模型改动计划：在当前 v9 correct-token 框架上，仿照 UniCon3R 的 "explicit geo cue + token internalization + latent refinement" 思路，引入人体部位显式 token 和对应辅助监督。

## 1. 背景和目标

当前 v9 baseline 已经证明：

- 只修 camera pose 不够，human translation 也需要修。
- `semantic + alignment + momentum` correct-token group 可以帮助模型修正 AABB 镜头跳变。
- pose-human LoRA 通常比 pose-only LoRA 更稳定。
- 单纯在输出端加 pelvis / hip / feet anchor loss，主观和指标上没有稳定超过 baseline。

最近的问题是：

- 后两帧的人体仍然有轻微回偏。
- 对没见过的视频或运动更明显的输入，泛化不稳定。
- 只在输出端加人体 anchor loss，人体显式信息没有进入 decoder，不像 UniCon3R 的 geo cue 那样参与中间推理。

下一步目标：

- 保持 feed-forward / streaming / UniCon-style。
- 不在推理时使用 GT。
- 让人体部位信息作为显式 token 进入 decoder。
- 让这个 token 通过 auxiliary head/loss 学到 raw human 应该如何修正。
- 最终仍然通过 latent residual 去修 pose head 和 human head，而不是测试时优化。

## 2. UniCon3R 可借鉴点

UniCon3R 的 explicit geometry cue 不是一个单独输出端 loss，而是 contact prompt 的输入组成部分。

它的 contact prompt 由几类信息构成：

- human prompt `H_t`
- semantic scene context `U_scene`
- explicit metric geometry token `G_t`
- temporal momentum token `M_t`

其中 explicit metric geometry 做法是：

- 使用上一帧 world pointmap `X^world_{t-1}`
- 以当前人的 2D anchor `u_t` 为中心做 RoIAlign
- 取局部 3D pointmap 坐标，平均成 `R^3` 几何描述
- 过 MLP 映射到 decoder hidden dimension

然后 contact token 和 image / pose / human token 一起进入 decoder：

```text
[F'_t, z'_t, H'_t, C'_t], S_t = Decoder([F_t, z_t, H_t, C_t], S_{t-1})
```

decoder 输出的 refined contact token 有两个用途：

- `C'_t -> contact head -> dense contact logits`
- `C'_t -> residual head -> delta H_t -> human head`

关键结论：

- contact 不是 detached auxiliary output。
- contact token 会参与 decoder attention。
- contact-conditioned latent residual 会直接改变最终 human output。

我们的设计应该借鉴这个思想：人体部位 cue 不能只作为最终 joint loss，而应该作为 token 进入 decoder，并通过 residual/confidence auxiliary head 学会如何纠正 human latent。

## 3. 新 correct-token 结构

当前 baseline correct-token group：

```text
A_corr,t = [A_sem,t, A_align,t, A_mom,t]
```

下一版改为：

```text
A_corr,t = [A_sem,t, A_align,t, A_mom,t, A_body,t]
```

四个 token 分工如下。

### 3.1 Semantic Token

作用：

- 提供当前画面中的人、场景、语义上下文。
- 类似 UniCon3R 的 semantic scene context。

输入：

- 当前 image tokens `F_t`
- 当前 human token / HMR token `H_t`
- recurrent scene state memory `S_{t-1}`

逻辑：

```text
U_curr = CrossAttn(H_t, F_t)
U_mem  = CrossAttn(H_t, S_{t-1})
gamma  = sigmoid(MLP(H_t, U_curr, U_mem))
A_sem  = gamma * U_curr + (1 - gamma) * U_mem
```

说明：

- `U_curr` 负责当前帧图像证据。
- `U_mem` 负责历史 scene memory。
- `gamma` 是 learned gate，不是人工指定比例。

### 3.2 Alignment Token

作用：

- 提供当前帧和历史帧之间的相对偏移信息。
- 帮助模型判断哪里发生了不连续。

输入：

- 当前 pose token
- 当前 human token
- 当前 raw camera pose
- 当前 raw human translation / raw joints
- previous corrected camera memory
- previous corrected human memory
- previous pose delta / human delta / gate

注意：

- 这里不能使用 GT。
- 它表达的是当前预测和历史预测之间的 relative mismatch。
- 它不是直接等于最终 pose delta，只是给 decoder 的对齐线索。

### 3.3 Momentum Token

作用：

- 记录上一帧模型如何修正。
- 提高长序列稳定性。
- 避免每帧独立乱修。

输入：

- previous refined correct token
- previous pose delta
- previous human delta
- previous gate

说明：

- 它类似 UniCon3R 的 temporal momentum。
- 对 streaming 场景很重要。

### 3.4 Body-Part Token

这是新增重点。

作用：

- 显式告诉模型人体哪些稳定部位可以作为局部 anchor。
- 让 human correction 不只依赖抽象 token，而是有可解释的人体部位几何 cue。

第一版建议只做一个整体 `A_body,t`，不要每个部位一个 token。这样改动更小，也方便和 baseline 对比。

推荐部位：

- pelvis
- left hip
- right hip
- left foot
- right foot
- spine / chest

head 暂时不作为强 anchor，后续可消融。

输入信息必须全部来自推理时可获得的预测：

- current raw SMPL joints
- current raw SMPL translation
- previous corrected SMPL joints memory
- part-wise displacement: `current_raw_joint - previous_corrected_joint`
- part visibility / mask confidence
- part 2D projected location
- optional local image token feature
- optional local pointmap 3D feature

第一版轻量实现：

```text
part_feat_k = [
  raw_joint_3d,
  prev_corr_joint_3d,
  raw_joint_3d - prev_corr_joint_3d,
  visibility_or_confidence
]

A_body = MLP(Pool(part_feat_1 ... part_feat_K))
```

其中 `Pool` 第一版可用 mean 或 concat+MLP。建议先用 concat+MLP，如果显存或实现复杂度有问题再退回 mean。

后续增强版本可以加入类似 UniCon3R 的局部几何 cue：

```text
local_img_feat_k = RoIAlign(image_tokens, projected_part_2d)
local_geo_feat_k = RoIAlign(pointmap_world_or_self, projected_part_2d)
part_feat_k = concat(raw_joint, prev_joint, diff, visibility, local_img_feat, local_geo_feat)
```

## 4. Decoder 输入方式

不要提前 mean pooling 成一个 token。

推荐 decoder 输入：

```text
[F_t, z_t, H_t, A_sem,t, A_align,t, A_mom,t, A_body,t]
```

设计理由：

- 四类 token 职责不同，提前平均会丢信息。
- decoder attention 可以自己学习 pose branch / human branch 应该看哪个 token。
- 这更接近 UniCon3R 的 contact token internalization。

如果为了兼容现有代码，也可以把 correct tokens 看成一个 token group：

```text
correct_tokens = stack([A_sem, A_align, A_mom, A_body])
decoder_input = concat(original_tokens, correct_tokens)
```

## 5. Decoder 输出后的分支

decoder 输出 refined correct tokens：

```text
A'_sem,t
A'_align,t
A'_mom,t
A'_body,t
```

后面分三条路径。

### 5.1 Pose Correction Branch

目标：修 camera latent。

第一版保持和 baseline 尽量一致，避免同时改太多变量：

```text
pose_corr_feat = PoolPose(A'_sem, A'_align, A'_mom, A'_body)
delta_z_pose_raw, pose_gate = PoseCorrectHead(pose_corr_feat)
delta_z_pose = pose_gate * delta_z_pose_raw
z_hat = z_raw + delta_z_pose
camera_hat = original_pose_head(z_hat)
```

建议：

- 第一版 `PoolPose` 继续用 mean pooling。
- 后续可测 all-concat/contact-style pooling。
- `A_body` 可以参与 pose branch，但不应让它主导 pose correction。

### 5.2 Human Correction Branch

目标：修 human latent / SMPL translation。

```text
human_corr_feat = PoolHuman(A'_sem, A'_align, A'_mom, A'_body)
delta_human_raw, human_gate = HumanCorrectHead(human_corr_feat)
delta_human = human_gate * delta_human_raw
H_hat = H_raw + delta_human
SMPL_hat = original_human_head(H_hat)
```

建议：

- 第一版 `PoolHuman` 也用 mean pooling，保持变量单一。
- 后续可以测试 human branch 对 `A'_body` 加权更高。
- human gate 第一版继续用 shared learned gate，后续再加 part confidence gate。

### 5.3 Body-Part Auxiliary Branch

这是新增分支，类似 UniCon3R 的 contact head。

输入：

```text
A'_body,t
```

输出：

```text
pred_part_delta = PartResidualHead(A'_body)
pred_part_conf  = PartConfidenceHead(A'_body)
```

它不直接输出最终 SMPL，而是让 `A_body` 学会：

- raw body parts 哪里错了
- 每个部位当前是否可信
- 当前是否应该依赖人体 anchor 进行纠正

## 6. Loss 设计

总 loss 设计：

```text
L_total =
  L_pose
+ L_human_trans
+ L_improvement
+ L_gate
+ L_part_residual
+ L_part_conf
+ L_part_anchor
+ L_temporal_pairwise
+ L_reg
```

不要一开始全部打开。要分阶段加。

### 6.1 Existing Pose Loss

已有。

目标：

```text
camera_hat vs camera_gt
```

包含：

- translation error
- rotation error

作用：

- 保证 camera pose 修正正确。
- 不能因为新增 body token 导致 pose error 明显变差。

### 6.2 Existing Human Translation Loss

已有。

目标：

```text
smpl_transl_hat vs smpl_transl_gt
```

作用：

- 监督 human translation。
- 这是当前 human correction 的主监督。

### 6.3 Part Residual Loss

新增，优先级最高。

定义：

```text
raw_part_joints = raw SMPL selected joints
gt_part_joints  = GT SMPL selected joints
target_delta    = gt_part_joints - raw_part_joints
pred_delta      = PartResidualHead(A'_body)
```

loss：

```text
L_part_residual = SmoothL1(pred_delta, target_delta)
```

意义：

- 让 `A_body` 明确学到 raw 人体部位应该怎么修。
- 比单纯 final joint loss 更接近 UniCon3R 的 contact head。
- 这个 loss 只在训练中使用 GT，推理时不需要 GT。

### 6.4 Part Confidence Loss

新增，第二阶段再加。

目标：

- 让模型学会哪些部位可以相信，什么时候不要强行用人体 anchor。

target 可由 raw error 自动构造：

```text
raw_part_err = ||raw_part_joints - gt_part_joints||
conf_target = clamp(raw_part_err / threshold, 0, 1)
```

loss：

```text
L_part_conf = BCEWithLogits(pred_part_conf, conf_target)
```

也可以使用 soft target：

```text
conf_target = sigmoid((raw_part_err - deadzone) / scale)
```

意义：

- AABB 中 raw 偏差大，模型应该更愿意修。
- AAAA 或正常连续帧 raw 偏差小，模型不应该乱修。
- 面对真实运动时，如果部位 displacement 不是镜头跳变导致，模型应该降低强制 anchor 的力度。

### 6.5 Part Anchor Loss

可选，权重不要太大。

定义：

```text
selected_joints_hat vs selected_joints_gt
```

loss：

```text
L_part_anchor = SmoothL1(pred_selected_joints, gt_selected_joints)
```

建议：

- 作为辅助 loss，而不是主线。
- 权重需要小心，过大容易让模型只适合“人原地”场景。
- 当前 pelvis/hip/feet anchor loss 效果不明显，说明它不能替代 token-level body cue。

### 6.6 Temporal Pairwise Loss

用于防止模型把所有运动都拉回原地。

定义：

```text
pred_delta_t = pred_joint_t - pred_joint_{t-1}
gt_delta_t   = gt_joint_t - gt_joint_{t-1}
```

loss：

```text
L_temporal_pairwise = SmoothL1(pred_delta_t, gt_delta_t)
```

意义：

- 如果人真的在走，GT 相邻帧 delta 不为 0，模型不应该强行拉回。
- 如果人原地但镜头跳变，GT 相邻帧人体位移接近稳定，模型会学习对齐。

### 6.7 Gate / Improvement / Regularization

沿用当前已有设计：

- learned gate supervision
- improvement margin loss
- residual norm regularization

新增 regularization 可包括：

```text
||delta_z_pose||
||delta_human||
||pred_part_delta||
```

防止过修。

## 7. 实验阶段

不要一次加完。按阶段控制变量。

### Stage 0: Baseline

固定当前标准版本：

```text
semantic + alignment + momentum
mean pooling
pose-human LoRA
human latent correction
shared learned gate
```

目的：

- 作为所有 body-token 实验的对照。
- 使用同一个 4-source 小数据集。
- 指标和可视化都保存。

### Stage 1: Body Token Only

改动：

```text
+ A_body
```

不加新 loss。

目的：

- 确认 decoder 可以稳定处理 4-token correct group。
- 确认 loss 不发散。
- 看结构本身是否带来改善。

预期：

- 可能改善不明显。
- 但不能明显变差。

### Stage 2: Body Token + Part Residual

改动：

```text
+ A_body
+ PartResidualHead
+ L_part_residual
```

这是主实验。

判断标准：

- human translation error 是否下降。
- selected body-part joint error 是否下降。
- camera pose error 不能明显变差。
- AAAA gate 不能明显异常变大。

### Stage 3: Body Token + Part Residual + Part Confidence

改动：

```text
+ PartConfidenceHead
+ L_part_conf
```

判断标准：

- AABB 上 part confidence / gate 更高。
- AAAA 上 part confidence / gate 更低。
- 正常连续帧不乱修。

### Stage 4: Add Temporal Pairwise

改动：

```text
+ L_temporal_pairwise
```

重点测试：

- h36
- keling
- owill
- 其他人有明显运动的视频

判断标准：

- 减少“把运动的人拉回原地”的问题。
- 不牺牲原地镜头切换场景的修正能力。

### Stage 5: Add Weak Part Anchor

改动：

```text
+ weak L_part_anchor
```

判断标准：

- 如果只提升训练集，不提升 test 或外部视频，则不作为主线。
- 如果对人体贴合有稳定帮助，再考虑进入大训练。

## 8. 推荐实验组合

先跑以下几组：

| Version | Tokens | New Heads | New Loss | Purpose |
| --- | --- | --- | --- | --- |
| baseline | sem + align + mom | none | current losses | 对照 |
| body_token_only | sem + align + mom + body | none | current losses | 测结构稳定性 |
| body_residual | sem + align + mom + body | part residual | part residual | 主实验 |
| body_residual_conf | sem + align + mom + body | part residual + conf | part residual + conf | 学是否该修 |
| body_residual_conf_pairwise | sem + align + mom + body | part residual + conf | residual + conf + pairwise | 防止运动人被拉回 |
| body_residual_conf_pairwise_anchor | same | same | + weak part anchor | 测输出端部位约束是否有额外收益 |

## 9. 实现位置

主要修改：

- `src/dust3r/v8_pose_prompt.py`
  - 新增 body-part token 构造。
  - 新增 body memory。
  - decoder token concat 增加第四类 correct token。
  - 新增 `PartResidualHead` / `PartConfidenceHead` 输出字段。

- `src/dust3r/losses.py`
  - `V82PoseRelationLoss` 增加：
    - `part_residual_weight`
    - `part_conf_weight`
    - `part_anchor_weight`
    - `part_pairwise_weight`
    - `part_joint_indices`
  - 计算 selected joints 的 raw / corrected / GT losses。

- config 新增：
  - `v9_body_token_enabled`
  - `v9_body_part_indices`
  - `v9_body_part_pooling`
  - `v9_body_part_residual_weight`
  - `v9_body_part_conf_weight`
  - `v9_body_part_anchor_weight`
  - `v9_body_part_pairwise_weight`

## 10. 需要记录的指标

每组实验都记录：

- camera translation error
- camera rotation error
- human translation error
- selected body-part joint error
- part residual error
- part confidence mean
- AABB gate/conf mean
- AAAA gate/conf mean
- total loss
- pose loss
- human loss
- part residual loss
- part confidence loss
- temporal pairwise loss

主观可视化固定：

- 一个训练序列
- 一个 4-source probe test 序列
- 一个外部视频片段，例如 h36 / keling / owill

## 11. 判断标准

一个版本可以进入下一阶段，需要满足：

- human trans error 下降。
- selected body-part joint error 下降。
- camera trans/rot error 不明显变差。
- AAAA gate 不乱开。
- 外部视频中不明显把运动的人强行拉回原地。
- 主观可视化至少不比 baseline 更差。

如果只在单序列过拟合上变好，但 test 和外部视频变差，则该设计只记录为 overfit-friendly，不作为主线。

## 12. 当前推荐主线

下一版优先实现：

```text
semantic + alignment + momentum + body token
mean pooling
pose-human LoRA
shared learned gate
PartResidualHead
L_part_residual
```

暂时不要同时加 confidence、pairwise、anchor。

这个版本最能回答核心问题：

> 把人体部位显式 cue 作为 token 送进 decoder，并用 residual head 监督它，是否比单纯输出端 anchor loss 更有效？

如果有效，再按顺序加：

```text
part confidence -> temporal pairwise -> weak part anchor
```

## 13. 风险和注意事项

- 不能在 token 构造中使用 GT。
- GT 只能用于 loss。
- MVHuman / AvatarReX / THuman 坐标系必须保持训练坐标一致，不能混用可视化坐标。
- selected joints 需要确认在 SMPL-X / Human3R 输出中的 index 一致。
- 不能只用 feet 作为强 anchor，否则对走路场景容易错。
- part confidence/gate 对未来泛化很关键。
- 每次实验都必须和同一 baseline、同一数据划分、同一评估脚本比较。

