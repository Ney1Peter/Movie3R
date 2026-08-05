# Adaptive Detector / Policy 与 V9 专项化研究计划

日期：2026-08-05

本文件只定义后续研究计划。本轮不启动训练、不运行大批量实验、不修改冻结 baseline。

## 0. 总体目标

在冻结 B0+BRTC+C1 的基础上，验证并实现一个真正流式、可学习、可解释的自适应跨 shot 修正系统，同时重新判断 V9 是否有不可替代的贡献。

最终要回答两个问题：

1. shot detector 和 correction policy 是否值得从手工规则升级为学习模块？
2. V9 是否能在原版 Human3R 做不到的低纹理人体锚点场景中提供额外能力？如果不能，是否应重构为专门的低纹理人体锚点模块？

最终目标不是简单提高某一个样例，而是得到一条有明确因果约束、可泛化、可复现的主线：

```text
RGB 流 → 因果事件检测 → 场景/人体可观测性估计
      → 自适应选择 baseline、背景锚点或人体锚点
      → 相机-人体联合修正 → 后续帧稳定输出
```

## 1. 当前证据与待验证假设

### 1.1 Detector 当前状态

当前 detector 是 RGB 相邻帧特征 + logistic regression：RGB/灰度差异、颜色直方图、光流、ORB 匹配等。它只看 `I[t-1], I[t]`，符合因果约束，但本身不是核心创新，也没有显式建模“低纹理/单人/多人/相机跳变类型”。

已有探针中，全图像特征 logistic 约为 accuracy 0.983、F1 0.984、FPR 0.021，但这是已有探针划分上的 pair 分类结果，不能直接等同于完整流式系统的性能。

待验证假设：

- detector 可以学习到跨 shot 与 within-shot motion 的稳定差异；
- detector 不仅能判断“是否跳变”，还能预测跳变类型和可用证据；
- detector 的输出可以直接作为后续 policy 的因果状态，而不是单独的二分类器。

### 1.2 Adaptive policy 当前状态

当前 gate 主要使用固定阈值：人体旋转、顶点 RMS、尺度归一化 RMS、多人排列 margin。它在已测单人低纹理和多人案例上行为正确，但仍属于规则策略。

待验证假设：

- 人体 residual、背景 residual、相机 residual、纹理质量和人数共同决定“应该信任谁”；
- 可以训练一个小型 policy 预测：`keep baseline / shared SE(3) / human-anchor joint / scene-anchor`；
- 学习 policy 必须保留 abstention/fallback，不能因为置信度低而强行修改世界坐标。

### 1.3 V9 当前状态

当前观察是：

- 多人高纹理时，原版 Human3R 的相机已经相对可靠，V9 的额外作用不明显；
- 单人低纹理时，主要问题是背景相机缺少可观测性，简单 B0/V9 并不能自动解决；
- 人体锚点几何修正可以改善结果，但这并不能证明 V9 本身提供了额外能力；
- 因此必须做严格的原版 Human3R、B0/V9、B0+显式几何修正消融。

待验证假设：

- V9 可能没有学到稳定的跨 shot 相机能力，而主要复用了原模型已有能力；
- V9 的 latent correction 在低纹理人体区域仍包含可利用信息，只是当前训练目标过于混合；
- 将 V9 改为“低纹理人体锚点专用模块”后，才能体现不可替代的增益。

## 2. 数据组织与标签

### 2.1 数据来源

优先使用 `/data/wangzheng/iJCV-CODE/data` 中已有数据：

- AvatarReX：单人、低纹理、相机视角跳变；
- THuman：单人、相对干净的几何与视角变化；
- MVHuman100 / MVHuman200：多人、多视角、遮挡和纹理变化；
- MultiHuman/EgoHuman：更接近最终多人真实场景；
- 已有 v9/v14 pattern manifests 和历史 payload：用于复用边界定义与重建结果。

### 2.2 必须按 sequence/camera/group 划分

不能随机按 pair 划分，否则同一个人、同一个背景和同一相机的相邻 pair 会泄漏到 train/validation。

建议：

- train：若干 group 和 camera 组合；
- validation：未见过的 group/camera；
- held-out：未见过的数据源组合，例如 AvatarReX 训练、THuman+MVHuman 测试；
- final test：完全冻结，只在模型和阈值确定后运行。

### 2.3 Detector 标签

每个相邻 pair 至少包含：

- `is_cut`：是否跨 shot；
- `angle_bucket`：0–30°、30–90°、90–150°、150°以上；
- `people_count`：单人/多人；
- `texture_level`：高纹理/低纹理；
- `camera_reliable`：Human3R/B0 相机是否可靠；
- `human_anchor_reliable`：人体几何是否足以作为锚点；
- `visibility_level`：人体可见比例和遮挡比例。

其中后四类标签只用于训练 policy/分析，不允许把 GT 直接作为推理输入。

### 2.4 Policy 标签

根据 GT 评估结果离线生成 oracle action：

```text
KEEP_BASELINE
SHARED_SE3
HUMAN_ANCHOR_JOINT
SCENE_ANCHOR / CAMERA_ONLY
ABSTAIN
```

oracle action 的选择必须同时考虑相机误差、人体误差、seam jump 和 ID continuity，不能只看单一 MPVPE。

## 3. 阶段一：严格验证当前 V9 是否有额外作用

### 3.1 固定推理协议

在完全相同输入、相同 reset、相同输出坐标系下比较：

1. 原版 Human3R；
2. V9/B0，不加显式修正；
3. 原版 Human3R + 纯显式 SE(3)；
4. V9/B0 + 纯显式 SE(3)；
5. V9/B0 + BRTC+C1；
6. V9/B0 + 当前 adaptive joint。

必须保证每一项只改变一个组件，不能把不同 checkpoint、不同 reset 或不同可视化坐标混在一起。

### 3.2 分层测试矩阵

至少包含：

- AvatarReX 单人低纹理：60°、90°、150°、180°；
- THuman 单人：低/高纹理对照；
- MultiHuman 三人：纹理丰富、多人 ID 容易交换的案例；
- MVHuman：多人遮挡和大视角案例。

### 3.3 V9 有效性的判断标准

只有在原版 Human3R 做不到、而 V9 在不依赖 GT 的情况下稳定改善的场景，才能把 V9 作为有效模块保留。

建议的 go/no-go：

- 如果 V9 相比原版在低纹理 camera/human 指标上没有稳定增益，则不再把 V9 宣传为通用跨 shot 相机校正器；
- 如果 V9 仅在人 ID 或人体 token 稳定性上有增益，则将其定位为 identity/geometry prior；
- 如果 V9 只在特定数据源有效，则必须重训或降级为数据源特定 ablation，不能作为主线。

## 4. 阶段二：Detector 学习化

### 4.1 先做无模型、纯流式版本

建立三个基线：

1. 当前 handcrafted feature + logistic regression；
2. handcrafted feature + temporal MLP/GRU；
3. 轻量 RGB encoder + causal temporal head。

输入只允许包含当前帧及历史缓存，不能使用未来帧和 Human3R 输出。

### 4.2 Detector 输出不只二分类

建议输出：

```text
p_cut
p_low_texture
p_single_person
p_camera_reliable
p_human_anchor_reliable
event_type embedding
```

其中 `p_cut` 决定是否进入 shot 分支，其他输出作为后续 adaptive policy 的先验。

### 4.3 训练目标

```text
L = L_cut + λ1 L_type + λ2 L_texture + λ3 L_calibration
```

要求概率可校准。低置信度时必须输出 abstain，而不是强行产生 cut。

### 4.4 Detector 成功标准

- held-out sequence 上 cut recall 高；
- within-shot false positive 低；
- 小角度真实 cut 不被全部过滤；
- 连续流式运行不依赖未来帧；
- 误报经过后续 geometry gate 后不能造成明显 3D 恶化。

## 5. 阶段三：Adaptive policy 学习化

### 5.1 输入特征

只使用推理时可获得的量：

- detector 输出概率和事件类型；
- 人数、ID 匹配 margin；
- B0 前后人体 Kabsch rotation/RMS；
- body scale-normalized residual；
- B0/raw shadow camera 差异；
- root-ray agreement；
- 背景纹理/ORB/光流质量；
- 可见比例和置信度统计。

### 5.2 模型形式

优先顺序：

1. calibrated decision tree / monotonic logistic policy；
2. 小型 causal MLP；
3. GRU policy，维护 shot 内状态。

不要一开始使用大型 transformer。第一目标是证明“学习 policy 比固定阈值更稳定”，而不是增加模型规模。

### 5.3 Action 设计

```text
KEEP_BASELINE
APPLY_SHARED_SE3
APPLY_HUMAN_ANCHOR_JOINT
APPLY_SCENE_CAMERA_UPDATE
ABSTAIN
```

policy 只决定动作和强度；实际几何更新仍由可解释的 Kabsch/root-ray solver 完成。这样可以避免学习模块直接输出不可解释的世界坐标。

### 5.4 策略训练和评估

- 训练阶段用 GT 只生成 oracle action；
- 推理阶段只用预测的几何和图像统计；
- 用代价函数惩罚错误更新，且错误更新的代价高于不更新；
- 评估 action accuracy 之外，还要评估最终 3D 误差和 seam jump。

建议代价：

```text
错误触发大范围修正 > 错过一次修正 > 保守 fallback
```

## 6. 阶段四：V9 专项化方向

### 6.1 低纹理人体锚点模块

如果阶段一证明 V9 通用作用有限，建议把 V9 改成明确的专用模块：

```text
输入：跨 shot 前后帧的图像 token + 人体 token + B0 粗相机
输出：人体方向 residual、root/camera ray residual、可靠性分数
```

训练目标不再混合“所有相机和所有人体误差”，而是聚焦：

- 低纹理背景下的人体朝向；
- 人体 anchor 的跨 shot 对齐；
- camera-human relative pose；
- 是否应该 abstain。

### 6.2 训练数据构造

用 AvatarReX/THuman 构造低纹理训练集：

- 随机跨 camera pair；
- 随机背景纹理弱化/遮挡；
- 随机角度和尺度；
- 输入 B0/raw 预测作为 noisy coarse state；
- GT 只用于 residual supervision。

必须加入 hard negative：

- 人体真实运动但没有 shot；
- 多人交换但不应整体旋转；
- 低纹理但人体不可见；
- B0 已经正确但残差看起来较大的姿态变化。

### 6.3 V9 训练目标

```text
L_v9 = L_rotation_residual
     + λt L_root_ray / translation
     + λh L_human_relative_pose
     + λg L_reliability_gate
     + λs L_temporal_stream_consistency
```

其中可靠性 gate 和 temporal consistency 是关键，否则模块可能在低纹理失败时继续输出错误大修正。

### 6.4 V9 go/no-go

- 若专项 V9 在 held-out AvatarReX 低纹理上显著优于原版 + 显式 solver，保留为主线学习模块；
- 若只提供很小增益，则作为 learned prior/ablation，不承担最终 camera correction；
- 若仍无增益，则删除 V9 的“通用校正”叙事，主线改为 B0 + adaptive geometric policy。

## 7. 大规模实验顺序

必须按以下顺序，避免一次性混合变量：

1. 冻结当前 B0+BRTC+C1，完成原版/V9/显式修正六组严格消融；
2. 在固定 gate 下扩展 AvatarReX、THuman、MVHuman、MultiHuman；
3. 训练并验证 detector，不改变几何 solver；
4. 训练并验证 adaptive policy，不改变 detector；
5. 单独重训 V9 专项低纹理人体模块；
6. 最后才做 detector + policy + V9 的组合实验；
7. 组合版本冻结后，再跑 Multi-THUMBS 和最终论文表格。

## 8. 论文主线与边界

建议主线：

> 面向在线人体-场景重建的事件条件自适应 gauge correction：系统根据当前 shot 的纹理和人体可观测性，在保持 baseline 安全 fallback 的前提下，选择背景或人体证据，因果地联合更新相机与人体。

应该强调：

- online/causal 是问题设定和系统价值；
- adaptive policy 是核心方法；
- camera-human joint solver 是实现机制；
- low-texture human anchor 是关键泛化能力。

不应声称：

- 专门解决严重遮挡；
- 完整的人体 Re-ID；
- 离线全局优化；
- 任意无人体可见场景都能恢复相机。

## 9. 最终停止标准

只有同时满足以下条件，才停止探索并进入论文大规模实验：

- 原版、V9、B0、显式 solver 的作用边界已经清楚；
- detector 在 held-out sequence 上稳定且可校准；
- policy 的错误触发不会显著恶化 baseline；
- 单人低纹理和多人高纹理都不需要人工选择模式；
- V9 是否保留有明确 go/no-go 结论；
- 运行时无 GT、无未来帧，延迟和额外计算量已报告；
- 最终主线、fallback 和失败边界都可以用一张方法图解释清楚。

