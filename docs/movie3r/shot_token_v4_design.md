# ShotToken V4 设计草案

本文档记录 V4 的设计方向。当前阶段只做设计，不修改代码。

## 背景

V2 的核心思路是把 `q_t` 作为 shot token append 到 decoder token 序列中：

```text
[pose token, image tokens, human tokens, q_t]
```

然后让它参与 decoder attention。这个设计的初衷是让模型在 shot change 时有机会全局调整 camera / pointmap / human / state。

但实验结果显示，`q_t` 作为普通 decoder token 后权限太大：

```text
1. q_t-only 就足以破坏背景 pointmap 和 camera token。
2. full-on 会导致背景重建错误、相机固定在第一帧附近、尺度错误。
3. no decoder q_t + no WorldLoRA 后，背景和人物恢复正常。
4. PoseLoRA 同时修 translation 和 rotation 时，会引入额外前倾/竖直方向偏移。
```

因此，V2 的主要问题不是 ShotTokenGenerator 的语义完全错误，而是 ShotToken 的使用场景过宽。

## 当前 ShotToken 语义判断

当前 `ShotTokenGenerator` 输入为相邻两帧的 decoder image token：

```text
F_dec[t]
F_dec[t-1]
```

内部使用：

```text
g_curr = mean(F_dec[t])
g_prev = mean(F_dec[t-1])
diff = g_curr - g_prev
sim = cosine(g_curr, g_prev)
x = concat(g_curr, g_prev, diff, sim)
```

输出：

```text
q_t
shot_logit
shot_prob
```

从 generator 角度看，语义基本正确：

```text
1. 当前帧和上一帧是否属于同一个 shot。
2. 如果不是同一个 shot，提供一个对齐 cue。
```

问题在于，V2 把 `q_t` 作为普通 token 放进 decoder 后，它实际可以影响：

```text
pose token
image tokens
human tokens
state token
camera head
pointmap head
SMPL head
```

这导致“设计语义窄，但实际权限宽”。

## V4 核心原则

V4 的核心原则是：

```text
ShotToken 是受限的 alignment token，不是 reconstruction token。
```

它应该服务于：

```text
shot boundary detection
cross-shot camera / state alignment
```

它不应该直接服务于：

```text
pointmap reconstruction
background geometry
RGB reconstruction
human shape
human rotation
texture or local details
```

V4 的目标不是让 ShotToken 管所有东西，而是让它专注解决目前最明确的问题：

```text
shot change 后 camera pose / world alignment 偏移。
```

## 推荐结构

### 总体方向

不再使用 V2 的方式：

```text
f_img = concat([pose, image, human, q_t])
```

改为：

```text
q_t -> alignment token a_t
a_t 只和 pose/state 相关路径交互
image tokens 不 attend a_t
pointmap head 不使用被 a_t 污染的 image tokens
human tokens 默认不 attend a_t
```

### V4 首选方案：Pose Alignment Cross-Attention Block

V4 首版建议新增一个受限的 pose alignment block：

```text
base decoder 正常运行
z_out = base decoder 输出的 pose token
q_t = ShotTokenGenerator(F_dec[t], F_dec[t-1])
a_t = AlignmentToken(q_t, optional state summary)
z_align = PoseAlignmentCrossAttention(query=z_out, key=a_t, value=a_t)
z_out_final = z_out + gate * residual(z_align)
camera_pose = camera head / pose adapter 使用 z_out_final
```

权限边界：

```text
a_t -> pose token: allowed
a_t -> state token: optional, later version
a_t -> image tokens: forbidden
a_t -> pointmap head: forbidden
a_t -> human tokens: forbidden in first version
```

这不是把 `q_t` 作为普通 token 送进原 decoder，而是在 decoder 后增加一个受限 cross-attention block。

它比 V3 的 translation-only adapter 有更强交互能力，但仍然避免污染重建 token。

### 后续增强方案：Decoder 内 Attention Mask

如果 V4 首版有效，再考虑更激进方案：

```text
tokens = [state tokens, pose token, image tokens, human tokens, alignment token]
```

但必须使用 attention mask 控制权限：

```text
pose token 可以 attend alignment token
state token 可以选择性 attend alignment token
image tokens 不能 attend alignment token
human tokens 默认不能 attend alignment token
alignment token 可以 attend pose/state/global image summary
```

这个方案最接近“ShotToken 进入 decoder”，但需要确认当前 `DecoderBlock` 是否支持 attention mask。若不支持，需要改 attention 实现，风险更高。

## Alignment Token 输入

`q_t` 只描述相邻帧视觉差异，不一定足够表达“如何对齐”。V4 的 alignment token 可以参考更多上下文：

```text
q_t: 相邻帧差异和 shot boundary 信息
z_out: 当前帧 pose token
state_summary: 前一 shot / 当前 recurrent state 的全局摘要
camera_pose_base: base camera head 的初始预测
shot_prob: 是否允许 correction 的 gate
```

首版可以先用：

```text
a_t = MLP([q_t, z_out])
```

后续再加：

```text
state_summary
previous shot anchor pose
current shot local pose estimate
```

## Loss 设计

V4 不应只依赖原始重建 loss。需要明确约束 ShotToken 的职责。

### Shot Boundary Loss

继续保留：

```text
shot_bce = BCE(shot_logit, shot_label)
```

作用：让 `shot_prob` 明确学习是否发生 shot change。

### Continuous No-Op Loss

非 shot boundary 时，shot-on 输出必须接近 shot-off 输出：

```text
L_noop_cont = ||pred_on - stopgrad(pred_off)||, only where shot_label = 0
```

关注对象：

```text
camera_pose
pts3d_in_self_view
pts3d_in_other_view
smpl_transl
```

其中 pointmap no-op 权重要更高，因为重建本身是 base model 已经做好的能力。

### Pointmap No-Change Loss

即使在 shot boundary，也不希望 alignment token 改 pointmap：

```text
L_pointmap_keep = ||pointmap_on - stopgrad(pointmap_off)||
```

作用：允许 ShotToken 参与 pose alignment，但禁止它通过 image tokens / pointmap 路径改重建。

这是防止 V2 背景崩坏的关键 loss。

### Pose Alignment Loss

在 shot boundary 帧上加强 camera pose 监督：

```text
L_pose_boundary = pose_loss(camera_pose_pred, camera_pose_gt), only where shot_label = 1
```

作用：让 alignment token 学习“怎么对齐”，而不是只学习“这里变了”。

如果现有 pose loss 已经包含所有帧，可以额外给 shot boundary 帧加权。

### Residual Magnitude Loss

限制 correction 幅度：

```text
L_residual = ||delta_pose||
```

首版建议只允许 translation residual，rotation residual 后续再加。

### 推荐总 Loss

```text
L = L_base
  + lambda_shot * L_shot_bce
  + lambda_noop * L_noop_cont
  + lambda_pointmap * L_pointmap_keep
  + lambda_pose_boundary * L_pose_boundary
  + lambda_residual * L_residual
```

初始建议：

```text
lambda_pointmap 较高，确保不破坏重建。
lambda_pose_boundary 中等，确保 shot boundary 有学习信号。
lambda_residual 较小但必须存在，防止 correction 爆。
```

## 数据与 Label 语义

AABB 构造仍然有用，但 label 必须清晰。

对于 10 帧：

```text
A0 A1 A2 A3 A4 B5 B6 B7 B8 B9
```

shot label 应该是：

```text
0 0 0 0 0 1 0 0 0 0
```

真正 boundary 只有 `B5`。

对于当前 4-view 训练 sample，可以是：

```text
A(t), A(t+1), B(t+2), B(t+3)
```

label：

```text
0, 0, 1, 0
```

训练时要区分：

```text
continuous frame: 必须 no-op
shot boundary frame: 允许 alignment correction
post-boundary same-shot frame: 不应该持续强 correction
```

## 与 Reset 的关系

`reset_interval` 的实验说明，reset state/memory 可以减轻 B 段偏移。

reset 的语义是：

```text
不要让前一个 shot 的 recurrent state 和 pose memory 污染后一个 shot。
```

但 reset 不会学习：

```text
新 shot 如何对齐到前一个 shot 的世界坐标系。
```

V4 可以把 reset 作为辅助机制，而不是最终能力：

```text
shot boundary detected -> optional state/memory reset or gated update
alignment token -> 学习 camera/world alignment correction
```

建议首版 V4 不强行修改 state reset 逻辑，先让 alignment block 只服务 pose。

## 实施顺序

### 阶段 1：文档和接口设计

当前阶段只写设计文档，不改代码。

需要确认：

```text
DecoderBlock 是否支持 attention mask
pose head 输入是否可以局部替换为 z_out_final
当前 loss 中 camera pose 的 shape 和 GT 来源
shot_label 在 Video/AABB 数据里的语义是否始终正确
```

### 阶段 2：V4-A，受限 Pose Alignment Block

实现最小可控版本：

```text
q_t 不作为普通 decoder token
新增 PoseAlignmentCrossAttention
只更新 pose token 或 camera pose adapter 输入
不修改 image tokens / human tokens / pointmap
```

验证目标：

```text
背景重建必须接近 baseline
camera 在 shot boundary 后偏移应小于 shot-off
continuous frame 不应变差
```

### 阶段 3：V4-B，加入更强 alignment 上下文

如果 V4-A 安全但能力不足，再加入：

```text
state_summary
previous shot anchor pose
current shot local pose estimate
```

### 阶段 4：V4-C，decoder 内 masked alignment token

如果需要让 alignment token 更深地进入 decoder，再考虑 attention mask 方案。

前提：

```text
V4-A / V4-B 已证明受限 alignment 有收益
pointmap no-change loss 能稳定防止重建崩坏
当前 DecoderBlock 支持或可安全扩展 attention mask
```

## 验证实验

每次 V4 修改后至少跑以下组：

```text
1. shot-off baseline
2. V2 full-on 作为失败参照
3. V3 translation-only adapter
4. V4 pose alignment block
5. V4 with pointmap no-change loss ablation
```

重点观察：

```text
background / pointmap 是否保持正常
camera trajectory 是否在 shot boundary 后更对齐
scale 是否稳定
continuous segment 是否 no-op
shot_prob 是否只在 boundary 处升高
```

## 成功标准

V4 首版成功标准：

```text
1. 背景和人物重建不比 shot-off baseline 差。
2. A 段连续帧 camera 正常。
3. B 段开始后 camera 偏移小于 shot-off baseline。
4. 不再出现 full-on 那种相机固定在第一帧、尺度错误、背景崩坏。
5. continuous no-op 指标稳定。
6. shot_prob 在 boundary 帧明显高于连续帧。
```

## 当前结论

ShotToken 的定义方向是正确的：

```text
是否同一个 shot
如何对齐到前一个 shot
```

V2 的问题是权限过大，不是 generator 语义必然错误。

V4 应该让 ShotToken 重新进入“decoder-style interaction”，但必须受限：

```text
只服务 pose/state alignment
不服务 reconstruction
不影响 image tokens
不修改 pointmap
```
