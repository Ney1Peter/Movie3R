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
g_t: human-scene reliability gate, optional in first version
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

## 9. 新增候选：Boundary-Pair Human Consistency

2026-05-21 新增一个更适合主线化的 human-anchor 思路：不假设人在整段视频中静止，而是假设视频在时间上连续，只有镜头发生切换。因此，shot boundary 前后两帧的人体真实状态应该仍然连续。

核心先验：

```text
即使人正在运动，t-1 -> t 的一帧间隔内，人体不会发生瞬移。
人体朝向、大体姿态、躯干结构、骨盆/肩膀/头部位置变化都应该较小。
```

这比“整段人体基本不动”的假设更弱，也更通用。

### 9.1 关键公式

设 shot boundary 发生在：

```text
t-1 -> t
```

Human3R 给出当前帧原始 camera pose 和 camera-frame joints：

```text
T_hat_t
J_cam_t
```

上一帧已有 corrected pose：

```text
T_corr_{t-1}
```

则上一帧 corrected world human anchors 为：

```text
J_world_{t-1} = T_corr_{t-1} @ J_cam_{t-1}
```

当前帧 correction head 输出：

```text
T_corr_t = exp(delta_xi_t) @ T_hat_t
J_world_t = T_corr_t @ J_cam_t
```

训练/推理希望满足：

```text
J_world_t ≈ J_world_{t-1}
```

但这个一致性只应是 robust / soft constraint，不能把真实人体动作抹掉。

### 9.2 推荐监督

无 GT camera / 无人工 label 时，可以使用 boundary-pair self-supervision：

```text
L = L_boundary_human_pair
  + L_torso_orientation_pair
  + L_foot_support_pair
  + L_noop_inside_shot
  + L_delta_smooth_inside_shot
  + L_delta_prior
```

其中最核心的是：

```text
L_boundary_human_pair = robust( T_corr_t @ J_cam_t - T_corr_{t-1} @ J_cam_{t-1} )
```

优先使用的稳定人体 anchors：

```text
pelvis
left hip / right hip
spine joints
left shoulder / right shoulder
head as weak cue
```

脚部作为 support cue：

```text
ankle / foot / heel / toe
```

脚不能强行左右一一对应，因为走路、交叉脚、抬脚都会带来真实运动。第一版应使用 set matching / Chamfer / robust loss，并根据脚部运动幅度降低移动脚的权重。

### 9.3 Online Memory 形式

该方法可以保持 online / causal。当前帧只需要读取上一帧 corrected memory：

```text
M_{t-1} = {
  T_corr_{t-1},
  J_world_{t-1},
  torso_frame_{t-1},
  optional foot support candidates,
  optional human token memory
}
```

然后当前帧前馈预测：

```text
delta_xi_t = f(T_hat_t, J_cam_t, H_t, M_{t-1})
```

并立即输出：

```text
T_corr_t
```

之后更新 memory：

```text
M_t <- corrected current human anchors
```

这不是 chunk alignment，也不是 offline post-processing，因为没有使用未来帧，也没有全局优化整段轨迹。

### 9.4 与 long-reference 版本的区别

Long-reference baseline 使用的是更强的参考假设：

```text
前半段人体基本站在原地，因此用前半段人体作为整段 reference。
```

Boundary-pair 版本使用的是更弱的时间连续假设：

```text
人可以运动，但 boundary 前后相邻两帧的人体变化应该很小。
```

因此，Boundary-pair 更适合作为 V7 后续主线。Long-reference 适合诊断“人体是否能当 anchor”，Boundary-pair 更接近真实可部署的 online correction。

### 9.5 设计定位

Boundary-pair 应作为人体连续性约束的基础形式：

```text
只要求相邻帧人体不能瞬移；
不要求整段人体保持静止；
不要求 post-shot segment 长期贴住同一个 reference。
```

它适合和 transient gate / scene geometry guard 组合，而不是作为独立最终方案。

### 9.6 Boundary-Pair + Causal State Propagation

Boundary-pair 只约束 `t-1 -> t` 的瞬时跳变时，容易出现一个问题：

```text
boundary 后第一帧被拉回来了，但后续帧如果继续沿用原 Human3R 的错误 state / memory，仍可能慢慢漂。
```

因此需要把 corrected boundary frame 写回 causal memory，让后续帧参考新的 corrected state，而不是继续参考 shot 前旧 state。

在线流程可以写成：

```text
for each frame t:
  read Human3R raw outputs: T_hat_t, J_cam_t, H_t
  read causal memory M_{t-1}
  delta_xi_t = f(T_hat_t, J_cam_t, H_t, M_{t-1})
  T_corr_t = exp(delta_xi_t) @ T_hat_t
  J_world_corr_t = T_corr_t @ J_cam_t
  output T_corr_t immediately
  update M_t from J_world_corr_t
```

在 shot boundary 处：

```text
M_{t-1} = corrected anchors from the last frame of previous shot
```

当前帧被纠正后，后续帧不再一直拉向 shot 前 reference，而是进入第二段自己的 corrected state：

```text
M_t = update(M_{t-1}, corrected anchors at frame t)
```

这相当于：

```text
先用人体连续性修正第二段起点，
再让第二段后续帧以这个 corrected 起点作为新的 state 缓存继续推理。
```

这样比纯 boundary-pair 更适合解决后续漂移，也比 long-reference 更少依赖“整段人静止”。

第一版 memory update 可以是 EMA：

```text
M_t = (1 - beta) * M_{t-1} + beta * J_world_corr_t
```

其中：

```text
beta 小：更稳定，更像锁定第二段 corrected state
beta 大：更能跟随真实人体运动
```

该版本仍然是 online / causal，因为只读取过去 corrected memory，不读取未来帧。

## 10. 新增候选：Learnable Transient Gate Correction

2026-05-22 新增 transient gate 方向。当前观察显示，Human3R 的错误不一定是第二段整体都存在同一个长期偏移，而更像 shot 后短暂 settling error：

```text
90: shot 前最后一帧，通常可信
91: shot 后第一帧，可能偏移最严重
92: shot 后第二帧，Human3R 可能已经基本恢复
93+: 第二段内部相对轨迹通常比 91 稳定
```

因此，直接把同一个 `Delta_T` 应用到整个 post-shot segment 可能会把后续本来较准的相机轨迹带偏；但固定只修第 91 帧又过于死板，不利于泛化到其他样本。

### 10.1 核心建模

将 correction 拆成 persistent 和 transient 两部分：

```text
delta_xi_t = delta_xi_persistent + alpha_t * delta_xi_transient
T_corr_t = exp(delta_xi_t) @ T_hat_t
```

其中：

```text
delta_xi_persistent: 如果 post-shot 整段确实有轻微 gauge offset，则长期保留
delta_xi_transient: shot 后短暂坏帧需要的额外修正
alpha_t: learnable transient gate，范围 [0, 1]
```

期望模型自己学出类似：

```text
91: alpha_t 高，强修正
92: alpha_t 较低，弱修正
93+: alpha_t 接近 0，基本保持 Human3R 原始第二段轨迹
```

但这不是手写固定规则，而是由网络根据当前帧特征、raw pose jump、human anchor jump、confidence 和距离 boundary 的时间自动预测。

### 10.2 Gate 输入特征

第一版 gate head 可以使用 causal / online 可得特征：

```text
raw camera translation step: ||t_hat_t - t_hat_{t-1}||
raw camera rotation step
stable human anchors jump
foot/support-point set jump
torso orientation jump
Human3R confidence / visibility, if available
boundary flag / post-shot flag
```

这些特征表达的是：当前帧是否仍像 shot transition 中的不稳定帧。如果 raw Human3R 在 92 之后已经恢复，gate 应该学会快速降低，而不是继续强行对齐 long-reference。

### 10.3 推荐损失

该方向的关键不是让后续帧全部贴住 shot 前 reference，而是同时满足：

```text
1. abnormal boundary pair 的 human anchors 不应瞬移
2. reliable post-shot pair 的 raw 相对轨迹应尽量保留
3. transient gate 应稀疏、平滑、尽快衰减
4. persistent correction 应默认较小，除非确有长期偏移
```

因此第一版 self-supervision 可以写成：

```text
L = L_anomaly_pair_human_continuity
  + L_reliable_pair_relative_preservation
  + L_camera_relative_preservation
  + L_gate_sparsity
  + L_gate_smoothness
  + L_persistent_prior
  + L_delta_prior
```

其中最重要的是 relative preservation：

```text
T_corr_t^{-1} @ T_corr_{t+1} ≈ T_hat_t^{-1} @ T_hat_{t+1}
```

它表达：当第二段 Human3R 内部相对运动可信时，不要为了修 boundary 把后续轨迹改坏。

### 10.4 与其他候选方案的关系

前三个版本可以统一理解为不同的 correction freedom：

```text
long-reference: post-shot 全段都被强 reference 约束，容易压制真实人体运动
boundary-pair: 主要修 boundary 两帧，后续传播弱
causal memory: 逐帧可更新 correction，但容易把后续本来正确的轨迹也改坏
transient gate: 学习哪些 post-shot 帧仍需要 correction，默认尽快回到 Human3R 原始轨迹
```

Transient gate 不是固定“只修第 91 帧”，而是一个可学习的软选择机制。它保留了 boundary-pair 的局部性，也避免了 long-reference 对整段人体静止的强假设。

### 10.5 设计定位

Transient gate 的作用不是替代 correction loss，而是控制 correction 何时生效：

```text
boundary / settling frames: alpha_t 高，允许修正；
稳定帧: alpha_t 接近 0，尽量 no-op；
不可靠 cue: alpha_t 降低，避免错误修正扩散。
```

因此，transient gate 应与 human anchor 和 scene geometry guard 合并使用，形成：

```text
delta_xi_t = alpha_t * f_human_scene(T_hat_t, J_cam_t, scene_t, M_{t-1})
T_corr_t = exp(delta_xi_t) @ T_hat_t
```

## 11. 当前主线计划：Human-Guided, Geometry-Guarded Pose Correction

详细实验记录、指标和可视化结论见：

```text
docs/movie3r/v7/human_scene_pose_correction_experiment_log.md
```

V7 当前主线从 human-only correction 调整为：

```text
Human-Guided, Geometry-Guarded Pose Correction
```

核心思想：

```text
人体负责发现和监督 shot boundary 处的人体瞬移；
场景几何负责保护地面、墙面、世界方向和背景结构；
gate 负责判断当前帧是否需要修正；
camera prior 负责防止过度修正。
```

### 11.1 为什么不是 human-only

人体 anchor 是强 cue，但不能单独决定完整 camera pose。

human-only full SE(3) 容易出现：

```text
人对齐了，但场景倾斜 / 上下漂移。
```

planar yaw + horizontal 又容易出现：

```text
场景不倾斜，但人体对齐不足。
```

因此，人体应作为 correction 的监督信号之一，而不是唯一锚点。

### 11.2 几何 guard 的职责

场景几何 cue 应承担 human anchor 不擅长的部分：

```text
single dominant normal:
  约束 pitch / roll / vertical drift，防止地面或墙面转歪。

top-K planes:
  额外约束 yaw 和平面间相对关系。

background point guard:
  弱约束平面内左右滑动，避免只沿着地板或墙面滑走。
```

场景几何必须带 reliability / confidence：

```text
plane fit 可靠时强使用；
plane fit 不可靠时降低权重，退回 human + camera prior。
```

### 11.3 Online / Causal 约束

后续所有版本必须保持：

```text
只使用当前帧和过去 memory；
不使用未来帧；
不做全局 BA / chunk alignment / offline post-processing；
每帧 forward 末端立即输出 corrected pose。
```

机制验证阶段的单帧 geometry 版本也应保持：

```text
reference frame = t - 1
corrected frame = t
future frame t + 1 只用于可视化 sanity check，不参与训练或优化
```

### 11.4 目标模型结构

未来的 correction head 应接在 Human3R forward 末端：

```text
Human3R raw outputs:
  T_hat_t
  pointmap / depth / confidence
  SMPL-X joints / human token
  human mask / visibility
  recurrent memory

online geometry cues:
  dominant plane normals
  top-K plane confidence
  background geometry descriptors
  previous corrected scene geometry memory

correction head outputs:
  alpha_t correction gate
  bounded delta_xi_t
  human / scene reliability weights

T_corr_t = exp(alpha_t * delta_xi_t) @ T_hat_t
```

### 11.5 推荐训练路线

第一阶段：继续使用 saved-output 机制验证 cue 和 loss。

```text
human anchor + multi-plane geometry guard
single boundary frame
causal t-1 / t only
```

第二阶段：加入 learnable transient gate。

```text
异常 boundary 帧 gate 打开；
恢复后的正常帧 gate 关闭；
后续帧尽量保留 Human3R raw relative trajectory。
```

第三阶段：训练轻量 correction head。

```text
input: Human3R tokens / joints / pointmap / confidence / geometry cues
output: alpha_t + delta_xi_t
loss: human continuity + scene geometry guard + relative trajectory preservation + prior
```

第四阶段：将 deterministic geometry extraction 替换或蒸馏为 neural scene geometry head。

```text
pointmap/conf/features -> plane normals / scene geometry confidence
```

### 11.6 当前下一步

下一步最值得实现的是：

```text
Offline Teacher -> Online Student
+ Transient Gate
+ Multi-Plane Scene Geometry Guard
+ Post-Shot Local Gauge Memory
```

目标：

```text
只在 shot boundary / settling frames 上打开 correction；
用 human anchor 提供人体连续性监督；
用 multi-plane / background geometry 防止倾斜和左右滑动；
92 之后等稳定帧尽量 no-op。
```

训练和推理必须严格区分：

```text
teacher / analysis:
  可以使用 post-shot stable frames 估计 cam2 local gauge，生成 pseudo target。

student / inference:
  只能使用当前帧和过去 memory，不能读取未来帧。
```

因此，未来帧只允许作为训练 pseudo label 或 oracle upper bound 的来源，不能进入最终前馈模型输入。

## 12. 隐式 Human-Scene Token Adapter 路线

当前显式 human-scene geometry 方法主要用于生成 teacher / pseudo label，而不是最终部署形式。

当前 teacher 使用的是 Human3R 已经输出后的显式结果：

```text
decoded SMPL-X / joints -> human anchor
decoded depth / pointmap / confidence -> background point cloud
background planes / normals / offsets -> scene geometry guard
```

这一路线已经验证了两个关键事实：

```text
human cue 能发现 shot boundary 处的人体瞬移；
scene geometry guard 能防止 human-only correction 把场景修歪。
```

但最终模型不应依赖 decoded SMPL-X body 做 post-hoc anchor，也不应在 inference 时做慢速 RANSAC plane fitting。更合理的目标是把显式 teacher 蒸馏进 Human3R forward 内部的隐式 token adapter。

### 12.1 Teacher 和 Student 的分工

Teacher 阶段可以复杂一些，用来生成较可靠的 pseudo target：

```text
Human3R saved output
  -> decoded SMPL human anchors
  -> background point cloud / top-K planes
  -> post-shot local gauge teacher
  -> delta_xi_teacher, alpha_teacher, r_human_teacher, r_scene_teacher
```

Student / final inference 阶段必须保持 causal feed-forward：

```text
RGB frame I_t
  -> frozen / mostly frozen Human3R backbone
  -> pose token z_t
  -> image / scene tokens F_t
  -> pre-SMPL human tokens H_t
  -> previous corrected memory M_{t-1}
  -> Human-Scene Token Adapter
  -> alpha_t, delta_xi_t, r_human_t, r_scene_t
  -> T_corr_t = exp(alpha_t * delta_xi_t) @ T_hat_t
```

Inference 时不使用：

```text
decoded SMPL-X bodies as explicit anchors
future frames
global BA / pose graph / chunk alignment
post-shot stable window to correct current frame
```

### 12.2 为什么背景也可以隐式化

显式 teacher 中的 background plane / normal 来自 decoded depth / pointmap，但 depth / pointmap 本身也是由 Human3R tokens 解码得到的。因此，scene / image tokens 理论上已经包含了背景几何、纹理置信度和相机运动线索。

可行的蒸馏目标不是让 student 显式输出完整平面，而是让它学会：

```text
当前 scene cue 是否可靠；
背景是否支持 correction；
human cue 和 scene cue 冲突时应该信谁；
应该输出多大的 bounded camera residual。
```

因此最终 claim 应是：

```text
We do not use decoded SMPL-X bodies as explicit post-hoc anchors at inference.
Instead, we distill an offline human-scene geometry teacher into a causal
pre-SMPL token adapter, which uses implicit human tokens and scene tokens to
predict bounded camera pose corrections under shot changes and low-texture
backgrounds.
```

### 12.3 推荐验证顺序

不要一开始就全量训练隐式模型。更稳的验证顺序是：

```text
Step 1: 单 clip / 少量 clip 显式 teacher label
  先跑 Human3R saved output；
  用当前 SMPL + background geometry teacher 生成 delta_xi_teacher。

Step 2: 单 clip implicit student overfit
  输入只给 pre-SMPL human tokens / scene tokens / pose token / memory；
  不给 decoded SMPL 和显式 planes；
  目标是拟合同一 clip 的 teacher correction。

Step 3: 小规模 train / val
  train: 20-50 个 shot-change clips；
  val: 5-10 个 clips；
  验证 student 是否能泛化到未见 clip。

Step 4: 接入 Human3R forward
  把 token adapter 放到 downstream head 前后合适位置；
  frozen / mostly frozen backbone；
  只训练 correction head / reliability gates。
```

单 clip overfit 的意义是先回答一个最关键问题：

```text
Human3R 内部 token 中是否已经包含足够的 human + scene 信息，
可以在不看 decoded SMPL / planes 的情况下复现 teacher correction？
```

如果单 clip 都 overfit 不起来，说明 token input / adapter 设计不对，没必要马上做大规模 teacher dataset。若单 clip 能 overfit，再做小规模 train / val，才是在验证泛化。

### 12.4 当前数据入口

当前已有第一批裁剪后的 shot-change clips：

```text
/data/wangzheng/iJCV-CODE/data/data-V7-shot-change-clips/ms-aist
```

该目录包含：

```text
299 source videos
599 shot-change clips
manifest.json
detections.csv
```

推荐先从其中挑选少量样本：

```text
1-2 个 clip: 做 implicit student overfit sanity check
5-10 个 clip: 检查 teacher label 稳定性
20-50 个 clip: 做第一版 feature/token student 小训练
```

### 12.5 主要风险

这个方向的主要风险不在于 adapter 结构，而在于监督质量和可靠性判断：

```text
teacher 可能被错误 SMPL / 错误 plane match 拉偏；
human tokens 可能受遮挡、多人、动作剧烈变化影响；
scene tokens 在低纹理背景下可能本身不稳定；
human cue 和 scene cue 冲突时需要 reliability gate；
stable frames 必须学会 no-op，不能到处乱修。
```

因此 student 必须输出并学习：

```text
alpha_t: 是否修正以及修正强度
r_human_t: human cue 可信度
r_scene_t: scene cue 可信度
bounded delta_xi_t: 有上限的 pose residual
```

### 12.6 当前验证结果

2026-05-25 已完成两段 H36M shot-change clip 的 implicit token adapter overfit sanity check。

详细记录见：

```text
docs/movie3r/v7/implicit_token_adapter_validation.md
```

核心结果：

```text
h36m_test_boundary63:
  best input mode: human_scene
  best loss: 0.00000936
  target_err_t: 0.00027
  target_err_r: 0.0045 deg

h36m_18s_boundary91:
  best input mode: human_scene
  best loss: 0.00001627
  target_err_t: 0.00332
  target_err_r: 0.0533 deg
```

阶段性判断：

```text
Human3R internal tokens 中有足够信号，
可以让一个小 adapter 预测 teacher camera correction。
```

但该结论只说明单 clip 可拟合，不说明泛化。下一步必须进入 multi-clip held-out validation。

### 12.7 下一步计划：MS-AIST Shot2 Multi-Clip

下一步使用：

```text
/data/wangzheng/iJCV-CODE/data/data-V7-shot-change-clips/ms-aist/videos/shot2
```

该目录当前有 99 个 shot-change clips，总视频大小约 115 MB。建议按 staged pilot 扩大：

```text
Stage A: 5 clips smoke test
  检查 Human3R raw output、teacher label、token dump 是否稳定。

Stage B: 20 train + 5 val
  第一次验证未见 clip 泛化。

Stage C: 80 train + 19 val
  覆盖完整 shot2，评估不同动作和背景。
```

评估必须包含：

```text
human_scene / human / scene / pose / all ablation
target correction error
normal no-op error
alpha_t 是否只在 boundary / settling frames 打开
held-out viewer visual audit
```

存储策略：

```text
训练主数据只保留 tokens + pseudo labels + metrics。
完整 Human3R saved-output 只保留少量 debug / viewer 样本。
corrected viewer 输出必须 hardlink / symlink 大文件，不复制 depth/color/smpl。
```
