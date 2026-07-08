# V10 Streaming Global Alignment

## 1. 背景和问题定义

V9 的核心探索集中在 Human3R decoder 内部的 correct token、correction head、gate/loss 等局部结构上。大量消融说明：这些 token 可以在训练分布内带来一定改善，但很难稳定解决分镜跳变后的全局对齐问题。尤其在 AABB 这类输入中，原版 Human3R 在每个连续镜头段内部通常是稳定的，但 A 段和 B 段会落在不同的 local coordinate system 里，导致 camera、human、point cloud 在全局坐标下无法正确拼接。

因此 V10 的问题定义从“修正某一帧的 token”转为：

> 在单目、前馈、流式的条件下，维护一个跨分镜的 global state，使每个新镜头段的 local Human3R reconstruction 能被接入同一个全局坐标系。

这条路线的重点不是替代 Human3R，而是补上 Human3R/CUT3R 在 shot-discontinuous sequence 上缺少的 global state alignment 机制。

## 2. 关键观察

1. 原版 Human3R 对连续镜头段内部重建较强。  
   对于 AA 或 BB 这种同镜头连续帧，原版输出的 camera 和 human 往往已经足够稳定。

2. 跳变后的主要问题是 gauge 不一致。  
   B 段可以被 Human3R 重建成一个合理的局部结构，但它没有自动对齐到 A 段的历史全局坐标系。

3. 直接把所有帧都交给 V9 correction 会过度修正。  
   稳定帧也被修，导致 AA 内部本来正确的结果反而漂移。

4. 人体是强锚点，但不能简单等价为“人体静止”。  
   静止人物场景中，可以把人体作为近似固定锚点；运动人物场景中，人体 anchor 应该由历史 state 预测，而不是简单把当前人体贴回上一帧。

## 3. V10 整体框架

V10 使用分层流式框架：

### 3.1 Local Reconstruction

每一帧仍然先经过原版 Human3R 的 recurrent reconstruction。

对于连续镜头段：

- 继续沿用 Human3R 的 local recurrent state；
- 不做额外 correction；
- 保留原版模型在连续视频上的稳定性。

对于检测到的新镜头段：

- 重置或新建一个 local segment state；
- 让 Human3R 对新段先完成局部重建；
- 再由 V10 alignment module 将这个 local segment 接入 global state。

### 3.2 Global State

V10 需要显式维护一个 global state。它不是 Human3R 原始 recurrent state 的简单复用，而是跨镜头段累计的全局参考。

global state 至少包含：

- 历史相机轨迹的全局坐标；
- 历史人体 anchor 的全局位置和运动趋势；
- 最近一段的 segment-to-global transform；
- 可选的 scene/point-cloud anchor；
- 可选的 confidence 或 reliability 统计。

这个 state 的作用是回答一个问题：

> 当前新段应该以什么姿态、什么位置接到历史全局世界里？

### 3.3 Segment Alignment Module

当出现新镜头段时，alignment module 预测一个从当前 local segment 到 global state 的刚体变换：

- 输入：历史 global state + 当前 local Human3R reconstruction；
- 输出：一个 SE(3) transform；
- 应用对象：当前 segment 的 camera、human、point cloud。

当前 probe 版本已经实现了最小形式：

- 冻结 strict original Human3R；
- 使用 A 段历史人体 anchor 和 B 段第一帧人体 anchor；
- MLP 输出 3D rotation vector + 3D translation；
- 将该 transform 应用到 B 段所有帧。

这证明了“local segment to global state alignment”这件事可以作为独立模块学习。

## 4. 当前 Probe 使用的信息

当前 learned alignment probe 只使用 Human3R 预测的人体关节几何，不使用 image token、decoder token 或背景点云。

输入特征包括：

- `hist`: A 段 selected human anchors 的平均位置；
- `cur`: B 段第一帧 selected human anchors；
- `hist_shape`: A 段去中心化后的人体形状；
- `cur_shape`: B 段去中心化后的人体形状；
- `cur - hist`: 当前人体 anchor 与历史人体 anchor 的差；
- `hist_center`: 历史人体中心；
- `cur_center`: 当前人体中心；
- `cur_center - hist_center`: 人体中心位移。

selected anchors 当前来自 `STABLE_JOINTS + FOOT_JOINTS` 的并集，主要覆盖 pelvis/hip/torso/feet 等相对稳定的人体部位。

模块输出：

- `rotvec`: 3D rotation vector；
- `trans`: 3D translation；
- 组合成一个 SE(3) transform；
- 对 B 段 camera pose、SMPL/human joints、point cloud 统一变换。

## 5. 当前 Probe 的监督方式

训练时使用 GT camera 和 GT SMPLX joints，但不是直接在原始 GT 坐标系里监督。

原因是 Human3R 的输出坐标系与数据集 GT world coordinate 不一定一致。为了监督 segment alignment，先做一个 target bridge：

1. 取 A 段，也就是跳变前历史帧；
2. 用 A 段 GT joints 和 Human3R 预测 joints 估计一个 `GT world -> Human3R A-gauge` 的刚体变换；
3. 将所有 GT camera 和 GT joints 变换到 Human3R A 段坐标系；
4. 用变换后的 GT 作为 B 段监督目标。

loss 只作用在 B 段：

- human anchor loss：aligned B 段 joints 对齐 bridged GT joints；
- camera translation loss：aligned camera translation 对齐 bridged GT camera；
- camera rotation loss：aligned camera rotation 对齐 bridged GT camera；
- transform prior：约束输出的 rotation/translation 不要无意义发散。

当前权重：

- human: 5.0；
- camera translation: 2.0；
- camera rotation: 1.0；
- prior: 1e-4。

## 6. 为什么不是简单 SMPLX 对齐

直接 SMPLX 对齐可以作为 strong heuristic baseline，但它不是完整答案。

直接对齐的隐含假设是：

> 当前人体应该贴回历史人体位置。

这个假设只适合“人物原地不动”的窄场景。对于走动、跑动、跳跃的场景，真实人体位置本来就应该变化，直接把 B 段人体贴回 A 段会把真实运动抹掉。

V10 的目标是 state-aware human anchoring：

- 静止人物时，历史 state 预测的人体位置接近固定 anchor；
- 运动人物时，历史 state 应该编码运动趋势，预测“如果没有镜头跳变，当前人体应该在哪里”；
- alignment 目标不是上一帧人体位置，而是 global state 对当前时刻的预测位置；
- 因此人体是可随时间演化的 anchor，而不是固定模板。

这也是 V10 相比后处理规则的关键意义：它要学习如何使用人体、相机、场景和历史状态，而不是手工规定“把人贴一起”。

## 7. 创新点表述

### 7.1 Shot-Discontinuous Monocular Reconstruction

现有 Human3R/CUT3R 类方法主要假设输入是连续视频，缺少针对分镜跳变的 streaming global state 维护。V10 明确面向 shot-discontinuous monocular reconstruction。

### 7.2 Segment-Local to Global-State Alignment

V10 不强行让一个 recurrent state 横跨所有镜头，而是允许每个镜头段先形成稳定 local reconstruction，再预测 segment-to-global transform 接入历史全局坐标。

### 7.3 Human-Anchored but Motion-Aware State

人体作为强语义锚点，但不是静态锚点。V10 希望通过 state 表示人体运动趋势，使方法能从“静止人物对齐”扩展到“运动人物对齐”。

### 7.4 Unified Camera/Human/Scene Transform

alignment 不是只修 SMPLX，而是将同一个 transform 作用到 camera、human 和 point cloud，使整个 segment 在一个全局坐标系中一致。

### 7.5 Streaming and Feed-Forward

V10 仍然保持流式约束：

- 新帧到来时只使用历史和当前帧；
- 不做全局 BA；
- 不需要离线优化整段序列；
- 后续帧复用当前 segment 的 transform 或根据 state 增量更新。

## 8. 当前实验结论

单序列 overfit 说明该模块有能力学习 segment alignment：

- camera rotation 可以从约 139 deg 降到约 0.4 deg；
- human anchor error 可以从约 0.34 m 降到约 0.07 m。

4source、每个 source 两个训练序列的小训练也能在训练样本上明显收敛：

- overall camera rotation: 118.13 deg -> 4.29 deg；
- overall camera translation: 3.77 m -> 0.85 m；
- overall human post error: 0.323 m -> 0.180 m。

但 held-out test 暴露了当前最小 MLP 泛化不足：

- camera 有一定改善；
- human anchor 在测试样本上会被拉坏；
- AIST/H36M 定性测试也出现过度错误变换。

结论：

> 当前 probe 证明了方向可行，但仅用人体 joints 的小 MLP 不足以泛化。V10 正式版本需要引入更强的 state 表示、motion-aware target、reliability/gating，以及更系统的数据训练。

## 9. 下一步计划

1. 建立 V10 baseline。
   使用显式 human-anchor SE(3) 对齐作为 rule-based upper/strong baseline，量化“仅后处理”能到哪里。

2. 设计 state-aware alignment module。
   输入不再只用当前人体 joints，而要包含历史 global state、local recurrent state、camera trajectory、human motion features 和 confidence。

3. 训练 motion-aware anchor。
   静止人物数据监督固定 anchor；运动人物数据监督历史 motion-predicted anchor，避免简单拉回。

4. 分离 detector 和 alignment。
   暂时可以继续使用 oracle boundary 验证 alignment 本身。shot detector 后续单独实现。

5. 统一评估。
   评估项包括 camera pose、human anchor、segment consistency、stable-frame no-op、long-sequence drift。

6. 保留 V9 作为对照。
   V9 correct-token 路线不归档，后续作为 decoder-token correction baseline 与 V10 state alignment 对照。

