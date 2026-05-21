# V7 Plan: Online Human-Scene Pose Correction

## 1. 计划定位

本文记录 V7 当前候选思路：在冻结或基本冻结 Human3R 主干的前提下，在每帧 forward 末端增加一个轻量在线 pose correction head，用来修正 Human3R 原始预测的 camera pose。

该方向不是 offline post-processing，也不是 chunk alignment / pose graph / bundle adjustment。目标是在每一帧推理结束时，立即得到 corrected camera pose，并让后续 visualization / saving 使用 corrected pose。

## 2. 背景动机

当前调研发现，Human3R 在 RICH / AvatarReX 等纹理丰富数据上通常较稳定；明显偏移更多出现在低纹理、弱背景特征、简单场景的 shot boundary 附近。

这说明 V2-V6 的 background AnchorToken 思路存在一个矛盾：

```text
Human3R 容易失败的低纹理场景，往往也是背景特征最难可靠匹配的场景。
```

因此，V7 不应只依赖背景特征匹配。更合理的方向是利用 Human3R 已有的人体检测、human token、SMPL-X 输出、scene geometry 和 causal memory，在线预测一个小的 SE(3) pose residual。

## 3. 核心流程

Human3R 原始输出：

```text
RGB frame
  -> Human3R encoder / decoder / recurrent state
  -> camera token z_t
  -> refined human token H_t
  -> pose head: original camera pose T_hat_t
  -> scene head: camera-frame pointmap / depth / confidence
  -> human head: SMPL-X params / mesh / joints
```

V7 correction head：

```text
T_hat_t, z_t, H_t, SMPL-X camera-frame body anchors,
scene geometry cues, causal memory, reliability gates
  -> HumanAnchorPoseCorrectionHead
  -> delta_xi_t
  -> T_corr_t = exp(delta_xi_t) @ T_hat_t
```

最终推理优先使用 `T_corr_t`，而不是 `T_hat_t`。

## 4. delta_xi_t 应该由什么预测

关键问题是：`delta_xi_t` 不能只由单帧人体预测。单帧人体自身无法判断当前相机是否漂移，尤其是当人体和相机 pose 一起偏移时，当前帧内部仍可能看起来自洽。

因此，V7 correction head 需要融合五类信息。

### 4.1 原始相机位姿 prior

输入 Human3R 原始预测：

```text
T_hat_t
```

correction head 不从零预测相机位姿，只预测小残差：

```text
T_corr_t = exp(delta_xi_t) @ T_hat_t
```

`T_hat_t` 是稳定 prior，可以显著降低新 head 的学习难度和失控风险。

### 4.2 人体 dynamic anchor

人体不是静态 landmark，但人体可以提供 dynamic anchor。

可用信息包括：

```text
human token H_t
SMPL-X parameters
camera-frame body joints J_cam_t
pelvis / root / torso / shoulders / hips
feet / ankles / knees as optional cues
human confidence / visibility / mask
track id and short motion history
```

低纹理场景中，背景 RGB 特征可能很弱，但人通常仍然存在，并且 Human3R 对单人检测较稳定。人体可以提供：

```text
body scale prior
skeletal proportion prior
motion continuity
pelvis / torso continuity
foot-ground contact when available
multi-person relative layout when available
```

第一版优先使用：

```text
pelvis
torso / spine
left shoulder
right shoulder
left hip
right hip
```

`feet / ankles / knees` 保留接口，但不作为强依赖。手和头暂不作为第一版强 anchor。

### 4.3 场景几何 cue

低纹理不代表场景几何完全无效。Human3R / CUT3R 仍会输出：

```text
pointmap
depth
confidence
high-confidence background points
possible ground / wall plane cues
human-ground contact relationship
camera height consistency
```

V7 不再把背景特征匹配作为唯一依据，但也不应该完全丢掉 scene branch。scene geometry 应作为可选 cue，由 reliability gate 动态决定权重。

### 4.4 Causal memory

V7 必须是 online / causal，但可以维护小型历史 memory。

memory 可以包含：

```text
previous corrected camera pose
previous corrected human anchors
previous pelvis / torso / shoulder / hip trajectory
previous foot contact candidates
previous high-confidence scene anchors
human token memory
motion velocity
```

该 memory 只来自过去帧，不使用未来帧，不做全局优化，不做 chunk stitching。每一帧只读取当前 memory，前馈输出 correction，再更新 memory。

这类 memory 很关键，因为单帧人体无法判断相机漂移，必须通过跨帧一致性提供修正参考。

### 4.5 Reliability / gating

需要让模型知道当前帧应该更相信人体，还是更相信场景。

一个自然形式是：

```text
delta_xi_t = alpha_t * (g_t * delta_xi_t_human + (1 - g_t) * delta_xi_t_scene)
```

其中：

```text
alpha_t: correction gate，表示这一帧是否需要修正
g_t: human-scene gate，表示更相信人体还是场景
```

当背景低纹理、pointmap confidence 低、图像特征弱时：

```text
g_t -> human side
```

当人体遮挡严重、只有半个人、脚不可见、人体运动剧烈时：

```text
g_t -> scene / pose prior side
```

当 Human3R 本身稳定或所有 cue 都不可靠时：

```text
alpha_t -> 0
```

即尽量 no-op，避免把正常帧改坏。

## 5. 坐标系和输出使用

不要直接使用 raw world-frame human joints 作为 anchor，因为它们已经依赖 `T_hat_t`。如果 `T_hat_t` 漂了，world-frame joints 也会跟着漂。

第一版应优先使用：

```text
camera-frame SMPL-X joints / mesh / body anchors
```

得到 `T_corr_t` 后，用 corrected pose 重新摆放 Human3R 的 camera-frame 输出：

```text
M_world_corr_t = T_corr_t @ M_cam_t
X_world_corr_t = T_corr_t @ X_cam_t
```

这与当前 demo 的输出逻辑一致：Human3R 的 self-view pointmap 和 SMPL mesh 主要在 camera frame 下生成，最后通过 camera pose 变换到 world frame。因此只要替换最终 `camera_pose`，后续 world-frame 点云和人体 mesh 可以一起跟着修正。

## 6. 第一版最小实现范围

第一版目标是最小可行验证，不做复杂重构。

建议模块名：

```text
HumanAnchorPoseCorrectionHead
```

第一版输入：

```text
T_hat_t
camera token z_t if easy to expose
refined human token H_t
camera-frame stable body joints
basic human confidence
optional previous corrected human anchors memory
```

第一版输出：

```text
delta_xi_t: [..., 6]
alpha_t: correction gate
g_t: human-scene reliability gate, optional in first smoke
```

第一版约束：

```text
Human3R main backbone frozen
only train correction head
bounded residual
strong no-op behavior when confidence is low
no future frames
no offline optimization
```

## 7. 主要风险

### 7.1 人体不是静态锚点

人体有真实运动，不能强行把当前人体和上一帧人体完全对齐。correction head 应学习抑制不合理的 camera/world jump，而不是抹平人体运动。

### 7.2 Human3R 的 SMPL 也可能随相机一起错

如果失败帧中 camera-frame SMPL joints 本身也严重错误，那么 human cue 会不可靠。因此需要先做诊断，确认低纹理失败样本中哪些输出仍相对可信。

### 7.3 直接线性混合 SE(3) residual 有近似性

`delta_xi_t_human` 和 `delta_xi_t_scene` 在线性空间混合只适用于 small residual。第一版应限制 residual 范围，避免大幅修正导致不稳定。

## 8. 当前结论

V7 候选方向可以概括为：

```text
Pose prior
+ human dynamic anchor
+ scene geometry cue
+ causal memory
+ reliability gating
-> online SE(3) residual correction
```

一句话说明：

```text
V7 不再只依赖低纹理场景中不可靠的背景特征，而是在 Human3R 原始 pose 基础上，融合人体动态先验、场景几何、历史 memory 和可靠性 gate，在线预测一个小的相机 pose residual，让人和背景通过 corrected camera pose 一起回到更一致的世界坐标。
```
