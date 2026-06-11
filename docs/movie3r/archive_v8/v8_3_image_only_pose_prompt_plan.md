# Movie3R V8.3 Plan: Image-Only Human-History Pose Prompt

本文档记录 V8.3 的设计。V8.3 的核心目标是修正 V8.2 暴露出来的问题：

```text
V8.2 在 AvatarReX dataloader 测试上看起来有效，
但在严格 image-only 推理时泛化明显变差。
```

这说明当前 correction branch 可能没有真正学会：

```text
根据当前人体和历史人体/场景关系去纠正 camera pose。
```

而是部分依赖了训练数据管线里的额外提示，例如 `ray_map`、`ray_mask`、`shot_label`，或者学到了固定的 AABB 帧位置模式。

V8.3 要把训练输入改得更接近真实推理：**模型 forward 只允许使用 RGB 图像生成的 token 和模型内部 memory，GT 只允许用于 loss / evaluation，不允许作为模型输入。**

## 1. V8.2 的失败经验

V8.2 Stage-B 训练设置是：

```text
train: lbn1 + zxc
test:  zzr
clips: 40k AABB clips
model: freeze Human3R backbone / pose head
train: v8_pose_prompt + v8_pose_residual_head
```

在 dataloader eval 里，指标看起来不错：

```text
zzr held-out:
  raw rot error       ~= 42 deg
  corrected rot error ~= 20 deg
```

但是当我们严格改成普通图片文件夹推理，也就是只输入四张 RGB 图像时，效果明显下降：

```text
60 deg case:
  raw 1->2 rotation       ~= 25.8 deg
  corrected 1->2 rotation ~= 101.0 deg
  GT 1->2 rotation        ~= 60.4 deg

176 deg case:
  raw 1->2 rotation       ~= 0.7 deg
  corrected 1->2 rotation ~= 7.5 deg
  GT 1->2 rotation        ~= 177.0 deg
```

同时 gate 也暴露出问题：

```text
60 deg case:
  frame0 gate = 0.03
  frame1 gate = 0.11
  frame2 gate = 0.95
  frame3 gate = 0.90

176 deg case:
  frame0 gate = 0.03
  frame1 gate = 0.10
  frame2 gate = 0.97
  frame3 gate = 0.81
```

也就是说，gate 学到的更像是：

```text
前两帧少修，后两帧强修。
```

它还没有学到：

```text
当前帧到底需要修多少？
该往哪个方向修？
60 度和 180 度应该怎么区分？
```

## 2. 为什么 V8.2 会这样

### 2.1 `shot_label` 不应该作为模型输入

AABB dataloader 里有：

```text
shot_label = [0, 0, 1, 0]
```

这等于显式告诉模型：

```text
第 2 帧是跳变帧。
```

如果训练时保留 `shot_label`，模型很容易学成：

```text
看到第 2 帧/shot_label=1 -> 强修。
```

这会破坏我们真正想要的能力：

```text
模型应该从 image tokens / human tokens / memory 里自己判断是否需要修。
```

### 2.2 `ray_map / ray_mask` 不应该作为 V8.3 输入

检查 AvatarReX dataloader 后发现，AABB 样本中 view1 存在：

```text
ray_mask = True
```

这意味着训练/eval 时模型可能会使用 `ray_map` 作为输入 token。`ray_map` 来自相机几何，是很强的几何先验。

但真实视频 image-only 推理时没有这个信息。

所以 V8.3 必须去掉：

```text
ray_map
ray_mask=True
camera_pose as input prior
raw_camera_pose as input prior
```

否则 dataloader 结果和真实推理结果不等价。

### 2.3 现在的 residual 没有被强制“按人对齐”

V8.2 的 correction branch 确实能影响 pose token，但当前 loss 主要还是最终 pose loss。这样模型可能找到一个捷径：

```text
学一个常见旋转偏移，
而不是根据人体和历史帧关系计算偏移。
```

所以 V8.3 不只是去掉泄漏输入，还要加强 human-history alignment 监督。

## 3. V8.3 的任务边界

V8.3 暂时仍然只做 4 帧训练，不考虑长序列。

原因：

1. Human3R 原训练本身也是 4 帧视频式训练。
2. 当前最重要的是先把短序列 image-only correction 跑通。
3. 如果 4 帧都不能稳定学到 human-history relation，直接扩长序列只会让问题更难定位。

因此 V8.3 输入仍然是：

```text
frame0, frame1, frame2, frame3
```

AABB 构造仍然可以用：

```text
A_t, A_{t+1}, B_{t+2}, B_{t+3}
```

但模型 forward 只能看到这四张 RGB 图像。

## 4. V8.3 的输入约束

### 4.1 模型 forward 可以使用

```text
RGB image
image encoder tokens
Human3R / Multi-HMR branch 从 RGB 预测出的 human tokens
pose token
recurrent state token
pose memory
previous refined correction token
previous correction delta / gate
```

这些都是 image-only 推理时真实可用的信息。

### 4.2 模型 forward 不允许使用

```text
GT camera_pose
raw_camera_pose from AvatarReX calibration
GT SMPL params
GT joints / GT mesh
GT mask
GT depth / DA3 depth
GT pointmap
ray_map
ray_mask=True
shot_label
anchor cache from GT projection
```

这些只能作为 loss / evaluation 的监督信息，不能进入模型。

## 5. V8.3 的数据组织

需要新增一个 image-only AABB 数据路径。可以叫：

```text
AvatarReX_AABB_ImageOnly
```

它应该返回两部分：

```text
model_views:
  只包含模型 forward 需要的 image-only 字段

gt_views / gt_meta:
  只给 loss 和 metric 使用
```

通俗理解：

```text
模型看到的是普通视频帧。
训练器手里另外拿着答案，用来算 loss。
```

### 5.1 model_views 保留

```text
img
img_mhmr
K_mhmr / pseudo K
true_shape
img_mask
ray_mask = False
reset
update
idx
instance
```

其中 `ray_mask` 必须强制为 `False`。

### 5.2 model_views 删除

```text
camera_pose
camera_intrinsics
raw_camera_pose
depthmap
pts3d
valid_mask
msk
sky_mask
smpl_mask
smplx_*
shot_label
anchor_*
aabb_view_angle_deg
```

### 5.3 gt_meta 保留

```text
GT camera pose from raw AvatarReX calibration
GT relative pose
optional GT SMPL / joints / masks
view angle bucket
seqA / seqB / start_frame
```

注意：`gt_meta` 不传给模型，只传给 loss / metric。

## 6. V8.3 的模型结构

整体仍然保持 UniCon-style：

```text
RGB frames
  -> Human3R encoder / human branch
  -> image / pose / human tokens

image / pose / human tokens + recurrent state / pose memory
  -> [新增] A_corr_t pose correction prompt

image / pose / human tokens + A_corr_t + recurrent state
  -> decoder
  -> refined pose token + refined correction token

refined correction token
  -> residual head
  -> delta pose-token latent

refined pose token + delta
  -> pose head
  -> corrected camera pose
```

V8.3 不改变大 backbone，不重新做重建模型。

## 7. V8.3 的 correction token 设计

V8.3 的 `A_corr_t` 仍然是 pose relation prompt，但要明确它依赖 image-only 可用信息。

建议包含三类 token：

### 7.1 Current Human-Image Token

来源：

```text
当前帧 image tokens
当前帧 predicted human tokens
当前帧 pose token
```

作用：

```text
看当前帧的人在哪里、朝向如何、身体结构是否稳定。
```

注意：这里的人体信息必须来自模型 RGB 分支预测，而不是 GT SMPL。

### 7.2 History Human-Pose Memory Token

来源：

```text
上一帧 recurrent state
上一帧 pose memory
上一帧 refined correction token
上一帧 corrected pose token
上一帧 predicted human token
```

作用：

```text
提供上一帧世界状态，让当前帧知道自己是否和历史对齐。
```

### 7.3 Current-History Relation Token

来源：

```text
current human/image token
history human/pose memory token
current pose token
previous corrected pose token
```

作用：

```text
显式表达“当前帧和历史帧是否在同一个世界坐标中一致”。
```

这一类是 V8.3 最重要的部分。它不能只告诉模型“第几帧要修”，而要帮助模型判断：

```text
应该修多少？
往哪个方向修？
修完后人体和历史关系是否更合理？
```

## 8. V8.3 的 loss 设计

V8.3 需要保留 pose loss，但不能只依赖 pose loss。

### 8.1 Pose Supervision Loss

使用 raw AvatarReX calibration 得到 GT camera pose：

```text
L_pose = L_trans + L_rot
```

这是主监督，保证 corrected pose 接近 GT。

### 8.2 Human-History Alignment Loss

目标不是假设人原地不动，而是约束短时间内人体运动合理：

```text
predicted human relation after correction
  should be closer to GT / short-term motion relation
```

第一版可以先做简单版本：

```text
修正后 frame1 -> frame2 的人体中心 / torso / pelvis 相对关系
应该比 raw pose 下更接近 GT camera relation。
```

如果使用 GT SMPL / joints，它们只能用于 loss：

```text
GT joints + predicted corrected camera pose -> world anchor relation
```

不能作为模型输入。

### 8.3 Residual Direction Loss

V8.2 的问题之一是只会“转一下”，但方向不稳定。

V8.3 可以监督 residual 的方向：

```text
delta pose latent / corrected pose
应该朝着 raw pose -> GT pose 的方向改善。
```

也就是：

```text
L_direction = 1 - cosine(delta_pred, delta_teacher)
```

其中 teacher delta 来自：

```text
T_teacher_delta = T_gt @ inverse(T_raw)
```

这里的 `T_raw` 是原 Human3R 输出，不是输入 GT。

### 8.4 Improvement Margin Loss

要求修正结果必须比 raw 更好：

```text
error(corrected, GT) + margin < error(raw, GT)
```

这个 loss 可以避免模型只是输出一个大旋转，但没有真正改善。

### 8.5 Gate Calibration Loss

gate 不应该只学 frame index。

gate target 应该来自 raw pose error：

```text
raw pose error 大 -> gate 高
raw pose error 小 -> gate 低
```

例如：

```text
gate_target = sigmoid(
  rot_error_raw / rot_scale
  + trans_error_raw / trans_scale
)
```

并且训练数据要包含：

```text
no-change clips
small-angle clips
medium-angle clips
large-angle clips
```

这样 gate 才能学会“该不该修”，而不是学“第 2 帧要修”。

### 8.6 Small Residual Loss

继续保留：

```text
L_small = ||delta||^2
```

它的作用是防止过度修正。

但权重不能太大，否则模型不敢修大角度。

## 9. V8.3 的训练数据设计

仍然用 4 帧，但要改变采样分布。

### 9.1 AABB camera-switch clips

用于学习明显 pose correction：

```text
A_t, A_{t+1}, B_{t+2}, B_{t+3}
```

角度覆盖：

```text
60-90
90-120
120-150
150-180
```

### 9.2 AAAA same-camera clips

用于训练 gate 不要乱修：

```text
A_t, A_{t+1}, A_{t+2}, A_{t+3}
```

这类样本非常重要，因为它告诉模型：

```text
不是所有第 2/3 帧都要强修。
```

### 9.3 Small-angle AABB clips

除了 60 度以上，还应该加入：

```text
0-15
15-30
30-60
```

这类样本训练 gate 和 residual 幅度：

```text
小错误小修，大错误大修。
```

## 10. 第一版 V8.3 实验配置

先不要做长序列，不要复杂 global alignment。

第一版只做：

```text
4-frame image-only AABB/AAAA training
freeze Human3R backbone
freeze original pose head
train:
  v8_pose_prompt
  v8_pose_residual_head
```

数据建议：

```text
train:
  lbn1 + zxc

test:
  zzr image-only
```

训练样本组成：

```text
25% AAAA no-change
25% small-angle AABB
25% medium-angle AABB
25% large-angle AABB
```

评估必须同时报告：

```text
raw pose error
corrected pose error
improvement ratio
gate per frame
delta norm per frame
small-angle over-correction rate
large-angle under-correction rate
```

## 11. V8.3 成功标准

V8.3 不以 dataloader eval 单独作为成功标准。

必须满足：

### 11.1 Image-only 推理成功

输入普通 RGB 图片文件夹，不使用 GT：

```text
run_human3r_save_output.py
```

结果应当在 `zzr` 上明显优于 raw Human3R。

### 11.2 Gate 行为合理

期望：

```text
AAAA:
  gate 低

small-angle AABB:
  gate 中低，delta 小

large-angle AABB:
  gate 高，delta 大
```

### 11.3 Correction 不只是旋转

修正后应该表现为：

```text
人体和历史帧更对齐
相机和场景关系更合理
rotation / translation 都朝 GT 改善
```

不能只是：

```text
后两帧统一旋转一个常见角度。
```

## 12. 一句话总结

V8.3 的重点是：

```text
把训练改成真正 image-only 的 4 帧 pose correction。

去掉 ray_map / shot_label / GT-like 输入，
让 correction prompt 只能依靠 RGB token、human token 和 recurrent memory，
并用 human-history alignment + pose improvement loss
强制它学习“为什么修、往哪修、修多少”。
```

