# ShotToken V5 规划

本文档记录 V5 的设计方向。V5 的目标是在不破坏 Human3R 原有重建质量的前提下，让 ShotToken 更早参与 camera pose 修正，重点解决 AABB 中 `A2 -> B1` 跳变后的相机位姿和世界坐标接回问题。

## 背景

当前已验证的主要结论：

```text
V2: q_t 作为普通 decoder token，权限过大，直接污染 image/human/pointmap 分支。
V3: q_t 不进 decoder，只修 camera translation，重建安全，但 translation-only 不够。
V4: q_t 不进 decoder，decoder 后只和 pose token 做 alignment，再修 camera pose。
```

V4 的优点是安全：ShotToken 没有直接进入 image tokens，因此 pointmap 和人物重建基本不再崩坏。

V4 的问题是：

```text
1. q_t 只在 decoder 全部结束后修 camera，过于接近后处理。
2. translation residual 容易在 y/z 方向也打满，造成额外竖直/深度错位。
3. 当前 loss 缺少对 A2 -> B1 跳变边界和 B 段 anchor 的显式监督。
```

因此 V5 分为两个阶段：

```text
V5.1: 保守方案，在每层 decoder 后插入 pose-only shot attention。
V5.2: 如果 V5.1 不够，再改 decoder attention mask，让 ShotToken 真正作为受控 token 进入 decoder。
```

## 核心目标

V5 主要服务两个目标：

```text
1. 跳变帧本身的位姿要算对，尤其是 AABB 中的 B1/view2。
2. 整个序列的 pose 要一致，AABB 中 A1/A2/B1/B2 都要各自落在同一个 GT world coordinate 下。
```

对应监督不能理解成让 `A2` 和 `B1` 位姿相同。`A2 -> B1` 是真实镜头跳变，GT 里也有真实的相对变换。需要监督的是：

```text
relative(T_pred_A2, T_pred_B1) ≈ relative(T_gt_A2, T_gt_B1)
```

也就是预测出的跳变相对位姿要等于 GT 的跳变相对位姿。

## V5.1: Interleaved Pose-Only Shot Attention

### 设计动机

V5.1 不把 `q_t` 作为普通 decoder token append 到 `[pose, image, human]` 序列里，而是在 decoder 每层之后单独更新 pose token：

```text
decoder layer k 正常运行：
    [pose, image, human] -> [pose, image, human]

pose-only shot attention：
    pose = pose + Adapter(query=pose, key/value=[pose, q_t])

进入下一层 decoder：
    [updated pose, image, human]
```

这样 `q_t` 不直接暴露给 image tokens / human tokens / pointmap head，但 pose token 可以在 decoder 中间逐层读取 shot 信息。

### 与 V4 的区别

V4：

```text
12 层 decoder 全部完成
-> final pose token 读取 q_t
-> 修 camera_pose
```

V5.1：

```text
每一层 decoder 完成后
-> 当前层 pose token 读取 q_t
-> 修正后的 pose token 进入下一层 decoder
```

所以 V5.1 不是最终输出后的单次后处理，而是在 pose token 形成过程中持续注入 shot cue。

### 权限边界

V5.1 的权限设计：

```text
q_t -> pose token: allowed
q_t -> image tokens: forbidden directly
q_t -> human tokens: forbidden directly
q_t -> pointmap head: forbidden directly
```

需要注意：更新后的 pose token 会进入下一层 decoder self-attention，image tokens 理论上可以 attend 到这个已经被 q_t 修过的 pose token。因此 V5.1 不是绝对隔离，但比 V2 安全得多，因为 image/human token 不会直接读取 `q_t`。

### 插入层数策略

V5.1 首版先使用每层都插入：

```text
shot_pose_layers = all 12 decoder layers
```

如果发现对 decoder 扰动太强，可以逐步减少：

```text
方案 A: 每 2 层插入一次，例如 layers = [1, 3, 5, 7, 9, 11]
方案 B: 只在后半段插入，例如 layers = [6, 7, 8, 9, 10, 11]
方案 C: 只在最后几层插入，例如 layers = [8, 9, 10, 11]
方案 D: 使用 learnable gate / scale，从接近 0 开始训练
```

判断依据：

```text
1. 如果 camera jump 指标改善，但 pointmap/no-harm 变差，减少插入层数或降低 gate。
2. 如果 pointmap 安全但 B 段 pose 没改善，增加插入层数或提高 jump/anchor loss 权重。
3. 如果 y/z correction 继续打满，考虑 axis-wise residual limit 或降低 pose update scale。
```

### 模块建议

建议新增一个 decoder-loop 内使用的模块，例如：

```text
LayerwisePoseShotAdapter
```

输入：

```text
pose_token_k: 当前 decoder layer 后的 pose token
q_t: ShotTokenGenerator 输出的 shot token
shot_prob: 当前帧是否为 jump 的 gate
```

输出：

```text
pose_token_k_refined
```

首版只更新 pose token，不直接输出 camera residual。最终 camera pose 仍由 downstream head 基于 decoder token 输出。

### 训练参数范围

`freeze='shot_adaptation'` 仍然保持 base Human3R 冻结：

```text
encoder: frozen
decoder: frozen
downstream head: frozen
MHM/R head: frozen
```

V5.1 只训练：

```text
ShotTokenGenerator
LayerwisePoseShotAdapter
```

如果保留 V4 的最终 `PoseAlignmentAdapter` 作为 ablation，默认应先关闭，避免难以判断改善来自 decoder-loop 还是最终后处理。

## V5.1 Loss 修改

### 当前已有 loss

当前训练已经有：

```text
pose_loss: 全序列 absolute camera pose 监督
pose_loss_view2_AABB: AABB 中 B1/view2 的额外 absolute pose 监督
shot_bce: jump gate / shot boundary 分类监督
shot_q0_loss: 连续帧压低 q_t 能量
shot_noop_loss: 连续帧 shot-on 输出接近 shot-off 输出
shot_pointmap_keep_loss: pointmap no-harm
shot_pose_residual_loss: V4 residual 大小正则
```

其中 `shot_pose_residual_loss` 不是 supervised residual GT，只是限制 `delta_t/delta_q` 不要过大。V5.1 如果不再输出最终 pose residual，这个 loss 可以关闭或仅用于兼容旧 adapter。

### 新增核心 loss

V5.1 需要新增三类 camera 监督。

#### 1. Boundary Absolute Pose Loss

加强跳变边界两帧本身的 absolute pose：

```text
L_boundary_abs =
    d(T_pred_A2, T_gt_A2)
  + d(T_pred_B1, T_gt_B1)
```

目的：

```text
A2 是 jump 前 anchor，B1 是 jump 后第一帧。
只监督 B1 不够，A2 本身也必须对。
```

当前 `pose_loss_view2_AABB` 可以升级为 `shot_boundary_abs_loss`，从只监督 `view2` 改成监督 `view1 + view2`。

#### 2. Jump Relative Pose Loss

显式监督 AABB 中 `A2 -> B1` 的真实跳变关系：

```text
L_jump_rel = d(
    relative(T_pred_A2, T_pred_B1),
    relative(T_gt_A2, T_gt_B1)
)
```

目的：

```text
不是让 A2 和 B1 变得相同，而是让预测出的 jump 相对变换等于 GT jump 相对变换。
```

#### 3. Post-Jump Anchor Loss

把 B 段整体挂回 A2 的世界坐标：

```text
L_anchor =
    d(relative(T_pred_A2, T_pred_B1), relative(T_gt_A2, T_gt_B1))
  + d(relative(T_pred_A2, T_pred_B2), relative(T_gt_A2, T_gt_B2))
```

目的：

```text
B1/B2 不能在另一个局部坐标系里自洽，而必须接回 A 段的 world frame。
```

### 暂不优先新增的 loss

V5.1 首版不建议一次加太多监督。

```text
L_rel_all: 暂不加全相邻帧 relative loss，避免和 L_jump_rel/L_anchor 重复。
supervised L_residual: 暂不加，除非 V5.1 仍显式输出 DeltaT。
L_bg: 暂不加训练项，先作为后续 metric 或 V5.1 稳定后再加。
```

### No-Harm loss 调整

`shot_noop_loss` 和 `shot_pointmap_keep_loss` 需要避免和 AABB 的 B 段 correction 打架。

建议：

```text
1. no-op 主要作用在 AAAA/Video 样本，或至少只作用于 shot_label=0 且 is_video=True 的帧。
2. pointmap keep 可以保留，但优先监控它是否阻止 B 段接回 A 段。
3. 如果 V5.1 对 pointmap 影响明显，再提高 keep/no-op；如果 pose 修不动，再降低 keep/no-op。
```

### 推荐总 loss

V5.1 首版推荐：

```text
L_total =
    L_task
  + lambda_boundary * L_boundary_abs
  + lambda_jump     * L_jump_rel
  + lambda_anchor   * L_anchor
  + lambda_gate     * shot_bce
  + lambda_q0       * shot_q0_loss
  + lambda_noop     * noop_on_video
  + lambda_keep     * pointmap_keep
```

初始权重建议保守：

```text
lambda_boundary = 2.0
lambda_jump     = 2.0 或 3.0
lambda_anchor   = 1.0
```

如果 B 段仍然接不回 A 段，再逐步提高 `lambda_jump` 和 `lambda_anchor`。不建议第一版直接使用过大的权重，避免又出现 V4 中 y/z residual 打满的问题。

## V5.1 Metrics

必须新增专门指标，不能只看 `pose_loss`。

推荐记录：

```text
shot_boundary_abs_t_err
shot_boundary_abs_q_err
shot_jump_t_err
shot_jump_q_err
shot_anchor_t_err
shot_anchor_q_err
shot_anchor_view2_t_err
shot_anchor_view3_t_err
```

同时继续记录 no-harm：

```text
shot_noop_loss
shot_pointmap_keep_loss
regr_self_pts3d_avg
regr_cross_pts3d_avg
SMPLLoss_transl
SMPLLoss_j3d / SMPLLoss_v3d
```

Demo / eval 对比至少包含：

```text
1. V5.1 full
2. disable_shot_adaptation
3. disable_layerwise_pose_shot_adapter
4. 不同 shot_pose_layers 配置
```

## V5.2: Masked Decoder ShotToken

如果 V5.1 仍然不能修复跳变，V5.2 采用更正确但工程量更大的方案：让 ShotToken 真正进入 decoder token 序列，同时使用 attention mask 控制权限。

### 设计目标

Token 序列：

```text
[pose token, image tokens, human tokens, shot token]
```

权限：

```text
pose token -> shot token: allowed
image tokens -> shot token: forbidden
human tokens -> shot token: forbidden initially
shot token -> image/human tokens: forbidden initially
```

这样 ShotToken 真正处在 decoder 内，但不会像 V2 一样成为所有 token 都可读写的普通 token。

### 当前工程难点

当前 `src/dust3r/blocks.py` 的 `Attention` / `CrossAttention` / `DecoderBlock` 没有暴露 `attn_mask` 参数。它们直接调用：

```python
scaled_dot_product_attention(query=q, key=k, value=v, dropout_p=..., scale=...)
```

但没有传入：

```python
attn_mask=...
```

所以 V5.2 需要改底层 decoder block：

```text
Attention.forward(..., attn_mask=None)
CrossAttention.forward(..., attn_mask=None)
DecoderBlock.forward(..., self_attn_mask=None, cross_attn_mask=None)
model._decoder(...) 构造 token-level mask
```

这会影响 frozen base decoder 的执行路径，因此工程风险比 V5.1 高。

### V5.2 触发条件

只有当 V5.1 出现以下情况时，再进入 V5.2：

```text
1. pose token 中间注入仍无法修复 A2 -> B1 jump。
2. B 段 anchor 指标无明显改善。
3. 减少层数 / 调权重 / no-harm 调整后仍无效。
4. 主观 demo 仍显示 B 段明显另开坐标系。
```

### V5.2 风险

```text
1. attention mask 改错会影响所有 decoder layer。
2. RoPE position 和 shot token dummy position 需要小心处理。
3. gradient checkpointing 下需要同步传 mask。
4. 如果 mask 方向定义错误，可能重新出现 V2 的 reconstruction pollution。
```

因此 V5.2 需要更完整的单元/结构测试，先验证 mask 可视化和 token permission，再训练。

## 实验顺序

建议按以下顺序推进：

```text
1. 实现 V5.1 layerwise pose-only shot attention，默认每层插入。
2. 同步新增 L_boundary_abs / L_jump_rel / L_anchor 和对应 metrics。
3. 跑 5/10 epoch debug，观察 jump/anchor 指标和 no-harm 指标。
4. 若 pointmap 或 camera 过度扰动，减少插入层数或降低 adapter scale。
5. 若 pose 修不动，增加插入层数、提高 lambda_jump/lambda_anchor，或恢复最终 pose adapter 做 ablation。
6. V5.1 多组 ablation 均失败后，再进入 V5.2 masked decoder。
```

## 决策摘要

V5.1 是当前首选，因为它满足：

```text
比 V4 更早介入 decoder 过程。
比 V2 更安全，不让 image/human 直接读取 q_t。
不需要改底层 Attention mask，工程风险低。
可以通过插入层数和 gate 控制干预强度。
```

V5.2 是后备方案，因为它更接近理论上的正确结构，但需要改 decoder attention mask，工程量和回归风险都更高。
