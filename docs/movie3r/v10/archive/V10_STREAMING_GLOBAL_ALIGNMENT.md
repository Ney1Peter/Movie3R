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

但 2026-07-09 的 AIST/H36M 和 held-out test 泛化测试说明：直接让 MLP 从人体 joints 里回归完整 SE(3) 不够稳定。这个问题本质上是强几何约束问题，网络不应该从零猜一个 6DoF 变换，而应该在几何约束给出的稳定 proposal 上学习如何使用、修正和信任这个 proposal。

因此 V10 正式路线调整为：

> geometry-constrained streaming alignment：显式人体几何给出粗配准 `T_geo`，学习模块只预测 anchor reliability、alignment gate 和小 residual，最终得到 `T_final`。

新的 boundary-frame 流式流程是：

1. 当前帧先由原版 Human3R 输出 local camera、SMPLX/human、point cloud；
2. shot detector 判断当前帧是否是新 segment 起点；
3. 如果不是新 segment，继续使用当前 segment transform；
4. 如果是新 segment，用历史 global state 里的人体 anchor 和当前帧 local 人体 anchor 解一个显式粗配准 `T_geo`；
5. learned alignment head 基于 `T_geo` 后的残差特征，输出小的 `delta_R/delta_t`、anchor weights 和 gate；
6. 组合得到 `T_final = delta_T * T_geo`，并把 current segment 的 camera/human/point cloud 统一变换到 global state；
7. 后续同 segment 帧复用或平滑更新该 segment transform，并持续更新 global state。

这个流程仍然是严格流式的：在 boundary 帧只使用历史 state 和当前帧 Human3R 输出，不使用未来帧，也不做全局 BA。

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

这是旧 probe 的最小设计。它的问题是输出空间太自由：输入分布稍变，MLP 可能预测 30 deg、150 deg 甚至更大的错误旋转，且没有结构性保证输出后人体 anchor 一定更接近。

正式 V10 的输入信息应改为围绕 `T_geo` 的残差特征：

- `hist_anchor_global`: 历史 global state 预测的当前时刻人体 anchor；
- `cur_anchor_local`: 当前帧 local Human3R 人体 anchor；
- `T_geo`: 由显式人体几何解出的 local-to-global 粗配准；
- `cur_anchor_after_geo`: 当前 anchor 经 `T_geo` 变换后的结果；
- `geo_residual`: `cur_anchor_after_geo - hist_anchor_global`；
- `body_frame_delta`: 躯干/髋/头/脚构成的人体朝向差；
- `anchor_confidence`: Human3R/SMPLX 检测置信度、mask 面积、可见性、深度合理性等；
- optional state features: 历史 camera 速度、人体中心速度、segment transform 变化、detector confidence。

正式 V10 的模块输出也应收窄：

- `anchor_weight_logits`: 每类 anchor 的可学习权重，例如 pelvis、hip、torso、head、feet；
- `gate`: 当前帧是否采用 alignment，以及采用多强；
- `delta_rotvec`: 小旋转 residual，范围建议限制在 5-15 deg；
- `delta_trans`: 小平移 residual，范围建议限制在人体高度的一小部分或固定 0.2-0.5 m；
- optional `state_update_gate`: 当前 segment 对 global state 的更新强度。

也就是说，网络不再预测完整 SE(3)，而是在一个几何上合理的 `T_geo` 周围做小范围可学习修正。

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

正式 V10 的 loss 应在旧 loss 基础上拆成三层。

### 5.1 Final Alignment Loss

这些 loss 监督最终 `T_final` 后的输出：

- `human_anchor_loss`: aligned B 段人体 anchor 对齐 bridged GT 或 state-predicted anchor；
- `camera_t_loss`: aligned camera translation 对齐 bridged GT camera；
- `camera_r_loss`: aligned camera rotation 对齐 bridged GT camera；
- `body_frame_loss`: pelvis/head/hip/shoulder 构成的人体朝向对齐；
- `body_vertical_loss`: 防止整体人体上下方向或高度系统性漂移；
- `segment_consistency_loss`: 同一 segment 内相邻帧的相对结构不被 alignment 破坏。

### 5.2 Geometry Proposal and Residual Loss

这些 loss 约束网络只做小修正，而不是推翻几何 proposal：

- `geo_diagnostic_loss` 或 metric：记录 `T_geo` 单独能达到的 human/camera error，作为 strong rule baseline；
- `residual_pose_loss`: 如果有 GT，可计算 `T_res_gt = T_target * inv(T_geo)`，监督网络输出的小 residual；
- `residual_prior_loss`: 约束 `delta_rotvec` 和 `delta_trans` 足够小；
- `proposal_improvement_loss`: 鼓励 `T_final` 不比 `T_geo` 更差，例如 anchor distance after final <= after geo；
- `anchor_weight_regularization`: 防止权重塌缩到单个不稳定关节，同时允许低置信 anchor 被降权。

### 5.3 Gate and No-op Loss

这些 loss 保证稳定帧不被误修：

- `shot_gate_bce_loss`: 用 oracle boundary 或 detector label 监督 gate；
- `stable_noop_residual_loss`: 非跳变帧要求 `delta_T` 接近 identity；
- `stable_raw_output_consistency_loss`: 非跳变帧最终输出应接近原版 Human3R 输出；
- `segment_transform_smoothness_loss`: 同一个 segment 内 transform 不应逐帧乱跳。

这组 loss 的重点是：跳变帧打开 alignment，稳定帧保留 Human3R 原始连续重建能力。

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

更准确地说，简单 SMPLX 对齐应该作为 strong baseline 或 geometry proposal，而不是最终方法本身。

纯显式方法的优点是稳定、可解释、天然满足人体 anchor 约束；缺点是规则固定，难以处理以下情况：

- 人体在真实运动，不能简单贴回上一帧；
- 脚、头、手、躯干的局部预测质量不同；
- 遮挡、截断、mask 错误会让某些 anchor 不可信；
- 不同数据源的尺度、裁剪、Human3R 输出误差分布不同；
- 相机和人体都需要统一变换，不能只让 SMPLX 看起来对齐。

因此 V10 的学习部分应该回答：

- 哪些人体 anchor 可靠；
- `T_geo` 应该被信任多少；
- 是否需要小幅 residual 修正；
- global state 应该如何吸收当前 segment；
- 运动场景下当前人体应该对齐到历史 state 的预测位置，而不是上一帧位置。

这使得方法不再是简单后处理，而是一个 streaming global gauge alignment layer。

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

2026-07-09 进一步使用 medium V10 head 做真正在线泛化测试：

| Case | local reset A->B0 | learned V10 A->B0 | hand/root-yaw reference | 观察 |
|---|---:|---:|---:|---|
| AIST 69-72 | 1.267 m | 2.364 m | 0.171 m | learned 明显变差 |
| H36M 61-64 | 1.338 m | 4.846 m | 0.310 m | learned 明显变差 |
| AvatarReX lbn1 held-out | 0.516 m | 0.286 m | 0.106 m | learned 有改善但不如显式几何 |
| AvatarReX zxc held-out | 0.383 m | 0.442 m | 0.285 m | learned 略变差 |

这组结果说明：

- 流式 reset + segment transform cache 机制本身是可行的；
- 显式人体几何对齐在这些样本上更稳定；
- 纯 MLP 直接回归完整 SE(3) 泛化不足；
- 下一版应从 black-box SE(3) regression 改成 geometry proposal + learned residual。

结论：

> 当前 probe 证明了方向可行，但仅用人体 joints 的小 MLP 不足以泛化。V10 正式版本需要引入更强的 state 表示、motion-aware target、reliability/gating，以及更系统的数据训练。

更新后的结论：

> 强约束问题不应该交给网络从零猜完整 SE(3)。V10 应把人体几何约束写进结构里，用显式 `T_geo` 提供稳定粗配准，再让网络学习可靠性、权重、gate 和小 residual。

## 9. 下一步计划

1. 建立 V10 geometry baseline。
   使用显式 human-anchor SE(3) 对齐作为 rule-based strong baseline，同时记录 `T_geo` 的 human/camera 指标。

2. 改造 alignment head。
   不再直接输出完整 `rotvec/trans`，而是输出 `anchor weights + gate + small residual`。最终 transform 为 `T_final = delta_T * T_geo`。

3. 改造 loss。
   保留 final human/camera/body losses，新增 residual prior、residual target、proposal improvement、stable no-op 和 gate BCE。

4. 训练 motion-aware anchor。
   静止人物数据监督固定 anchor；运动人物数据监督历史 motion-predicted anchor，避免简单拉回。

5. 分离 detector 和 alignment。
   暂时可以继续使用 oracle boundary 验证 alignment 本身。shot detector 后续单独实现。

6. 统一评估。
   评估项包括 camera pose、human anchor、segment consistency、stable-frame no-op、long-sequence drift。

7. 保留 V9 作为对照。
   V9 correct-token 路线不归档，后续作为 decoder-token correction baseline 与 V10 state alignment 对照。

## 10. 代码改动建议

当前最小实现可以先不动 Human3R 主模型，只改 V10 alignment probe 和 online eval 脚本。

### 10.1 训练脚本

主要文件：

- `scripts/v10_static_alignment_4source_probe.py`
- `scripts/v9_learned_stream_alignment_overfit.py`
- `scripts/v9_learned_stream_alignment_4source_probe.py`

建议新增：

- `solve_weighted_rigid_transform_batch(src, dst, weights)`: batched weighted Procrustes；
- `compute_geo_proposal(pred_joints, state_or_target_anchor, joint_ids, weights)`: 计算 `T_geo`；
- `GeometryResidualAlignmentMLP`: 输入 residual/canonical features，输出 `delta_rotvec/delta_trans/gate/anchor_weight_logits`；
- `compose_transform(delta_T, T_geo)`: 组合得到 `T_final`；
- `apply_transform_batch(..., T_final)`: 复用现有 transform 应用逻辑。

训练循环从：

```text
features -> MLP -> rotvec/trans -> apply -> losses
```

改成：

```text
pred anchors + history/state anchors -> T_geo
T_geo-aligned residual features -> residual head
delta_T, gate, anchor weights -> T_final
T_final -> apply -> losses
```

第一版可以先固定 anchor weights，用 pelvis/hip/torso/head/feet 的手工权重计算 `T_geo`，只训练 gate 和 residual。等验证稳定后，再把 anchor weights 也交给网络预测。

### 10.2 Online Eval 脚本

主要文件：

- `scripts/v9_online_stream_human3r_segment_align.py`

当前新增的 `online_human3r_learned_aligned` 是严格流式的，但仍加载旧的 direct-SE(3) checkpoint。下一步应支持 geometry-residual checkpoint：

1. boundary 前输出 identity；
2. boundary 帧从历史 aligned anchors 和当前 local anchors 计算 `T_geo`；
3. residual head 输出 `delta_T/gate`；
4. 缓存 `T_final` 给后续同 segment 帧；
5. summary 中同时保存 `T_geo`、`delta_T`、`T_final`、gate 和 anchor weights，方便诊断。

### 10.3 推荐实验顺序

1. `geo_only`: 只用显式 `T_geo`，不训练 residual。作为 strong baseline。
2. `geo_plus_residual_fixed_weights`: 固定 anchor weights，训练小 residual 和 gate。
3. `geo_plus_residual_learned_weights`: 让网络预测 anchor weights。
4. `geo_plus_residual_state_features`: 加入历史速度/camera/state features。
5. `detector_integrated`: 接入 image-only detector，替代 oracle boundary。

判断标准：

- `geo_plus_residual` 不应比 `geo_only` 更差；
- stable frames 的输出应接近原版 Human3R；
- held-out 和 AIST/H36M 上不应出现大角度错误旋转；
- learned residual 的平均幅度应该小，说明网络是在修正几何 proposal，而不是重新发明完整对齐。

## 11. 并行分支：Token-Level Segment Alignment Probe

ReCal3R 的启发是：如果要修正 streaming reconstruction，不能只在最终 camera/human 输出上暴力补一个大变换，而应该回到底层 state/token 更新机制，判断哪些信息可靠、哪些信息应该被写入或用于对齐。

因此 V10 同时保留一条并行探索路线：

> 检测到新 segment 后，不先做显式 SMPLX 刚性对齐，而是查看 A 段和 B 段的 pose/human/state token 是否已经包含足够的坐标系信息，让一个小模块从 token 层面预测 segment-to-global alignment。

这条路线要回答一个核心问题：

> Human3R/CUT3R 的中间 token 里，是否已经编码了可以泛化的相机/人体坐标信息？

### 11.1 为什么要单独验证

直接使用显式人体几何 `T_geo` 很稳定，但容易被质疑是后处理规则。token-level probe 的意义是验证模型内部表示本身是否已经拥有对齐线索：

- 如果 pose/human/state token 能预测 A->B 对齐，说明 V10 可以设计成真正的 learned global gauge alignment layer；
- 如果只有人体显式几何能稳定工作，说明正式方法应该把显式几何作为结构约束；
- 如果 token 信息和几何 proposal 互补，后续可以做 `T_geo + token residual/reliability`。

### 11.2 候选 token

优先验证以下几类中间表示：

- `pose token`: 最可能包含 camera pose / coordinate gauge 信息，因为最终 camera pose 由 pose head 解出；
- `human token`: 最可能包含人体位置、朝向、身体语义 anchor 信息；
- `state token`: 包含历史场景/几何记忆，但信息较混合，需要 pooling 或 attention summary；
- `pose + human`: 先验证最直接的相机和人体组合；
- `pose + human + state`: 再验证加入 recurrent state 是否能提升段间对齐。

不建议第一版直接使用 raw image token 做主输入。image token 更偏外观和局部语义，可能有几何线索，但坐标系信息不如 pose/human latent 明确。

### 11.3 最小实验设计

第一版不改 Human3R 主模型，只做 probe：

1. 冻结原版 Human3R；
2. 对小 4source 数据做 local reset streaming inference；
3. 保存 A 段最后一帧和 B 段第一帧的候选 token；
4. 训练一个小 alignment head，从 token pair 预测 B 段 local 到 A 段 global 的 SE(3)；
5. 对比不同输入组合：
   - `pose_only`;
   - `human_only`;
   - `state_only`;
   - `pose_human`;
   - `pose_human_state`;
6. 使用同一套 bridged GT camera/human 监督，指标与 V10 geometry probe 保持一致。

监督目标仍然是 segment alignment，而不是 token reconstruction：

- camera rotation / translation loss；
- human anchor loss；
- body frame / body vertical loss；
- residual prior，防止输出过大；
- 可选 no-op loss，用于稳定帧不应被修。

### 11.4 判断标准

这条分支的结论不只看训练集 loss，还要看泛化：

- 如果 `pose_human` 在小训练集和 held-out/AIST/H36M 上都稳定优于 local reset，说明 token 本身有可靠坐标信息；
- 如果训练集能过拟合但测试崩，说明 token 有信息但 probe 泛化不足，需要更强约束；
- 如果 `state` 加入后明显更好，说明 global state/memory 对 alignment 有价值；
- 如果 `geo_only` 始终强于 token probe，说明显式几何应该作为主干，token 只适合做 residual/gate/reliability。

### 11.5 与 Geometry-Residual 路线的关系

token-level probe 不是替代 `T_geo + residual`，而是并行验证两种可能：

```text
Route A: explicit geometry proposal -> small learned residual
Route B: pose/human/state token pair -> learned segment alignment
Route C: explicit geometry proposal + token residual/reliability
```

如果 Route B 成立，V10 可以减少显式规则依赖。
如果 Route B 不稳定但 token 能提供部分改善，则 Route C 更合理：显式 `T_geo` 负责强几何约束，token 负责判断可靠性、补小 residual、控制 state update。

### 11.6 小 4source token probe 初步结果

2026-07-09 在小 4source 数据上跑了第一版 token-level probe：

- 数据：4 个 source，每个 source 2 个 AABB 样本，共 8 个训练样本；
- 模型：冻结 strict original Human3R；
- 输入：local-reset streaming inference 中导出的 compact token summary；
- 训练：每个 feature set 单独训练一个小 SE(3) alignment head，1500 steps；
- 输出：`output/v10_token_alignment_probe/4source_s2_alltokens_20260709`。

整体指标：

| Feature set | Cam Rot ↓ | Cam Trans ↓ | Human ↓ | Amean-B0 ↓ | 观察 |
|---|---:|---:|---:|---:|---|
| raw local reset | 118.13° | 3.769m | 0.323m | 0.296m | A/B gauge 未对齐 |
| pose_only | 11.44° | 0.832m | 0.244m | 0.214m | pose token 有相机坐标信息 |
| human_only | 4.60° | 0.842m | 0.181m | 0.132m | 当前 token 分支里最好 |
| state_only | 11.53° | 0.814m | 0.253m | 0.212m | state 有信息，但不如 human token |
| pose_human | 28.45° | 0.888m | 0.359m | 0.328m | 直接拼接反而变差 |
| pose_human_state | 36.91° | 1.090m | 0.400m | 0.377m | 高维拼接不稳定，伤害人体对齐 |

这组结果说明：

1. token 里确实有段间坐标信息。
   单独使用 `pose_token_out`、`human_token_out` 或 state summary，都能显著降低 camera rotation/translation error。

2. `human_token_out` 是当前最有效的单 token 来源。
   它不仅改善 camera，也改善 human anchor，说明 human decoder latent 里包含人体位置/朝向和局部 gauge 信息。

3. 简单 concat 多类 token 不一定更好。
   `pose_human` 和 `pose_human_state` 反而让 human 和 A/B anchor 变差，说明“信息更多”不等于“对齐更稳”。后续如果融合 token，需要 attention/gate/reliability，而不是裸 concat。

4. token-only 仍然不如显式几何在人形锚点上稳定。
   对比同一组小 4source 的 `geo_only`，token `human_only` 的 camera rotation 更低，但 Amean-B0 和 human anchor 仍更差。当前判断是：token 适合做 residual/gate/reliability，显式几何仍适合做强约束 proposal。

因此更推荐的下一步不是纯 token-only，而是：

```text
T_geo from explicit human anchors
+ human_token_out / pose_token_out reliability
+ small residual/gate
```

也就是 Route C：geometry proposal 负责几何正确性，token 负责判断哪些锚点可靠、是否需要修，以及小范围 residual。

## 12. Route C：Geometry Proposal + Human Token Residual

2026-07-09 开始验证 Route C，也就是：

```text
显式人体锚点先算一个 T_geo
+ Human3R decoder 后的 human_token_out
+ 一个小 residual head
-> 输出一个受限的小 delta_T
```

这一版的动机是避免让网络直接从 token 里预测完整 SE(3)。完整 SE(3) 太自由，容易在小数据上过拟合，也容易变成黑盒。更合理的方式是：

1. 先用 pelvis / hip / torso / feet / head 等稳定人体锚点算一个几何 proposal；
2. 这个 proposal 已经给出大致正确的 segment-to-global 对齐；
3. token 只负责补一个很小的 residual，或者判断这个 proposal 是否可靠；
4. residual 被限制在小范围内，默认最大 `10 deg` 旋转、`0.5 m` 平移。

代码：

- `scripts/v10_geometry_token_residual_probe.py`

输出：

- `output/v10_geometry_token_residual_probe/4source_s2_geo_human_token_20260709`

实验设置：

- 数据：4 个 source，每个 source 2 个 AABB 样本；
- Human3R：strict original Human3R，local reset streaming；
- boundary：oracle AABB boundary，第三帧作为新 segment 第一帧；
- token 输入：`human_token_out`；
- 训练步数：1500 steps；
- loss：camera rotation / camera translation / human anchor / body frame / body vector / body anchor / vertical offset / residual target / residual prior / proposal improvement。

整体指标：

| Variant | Cam Rot ↓ | Cam Trans ↓ | Human ↓ | Amean-B0 ↓ | Amean-B1 ↓ |
|---|---:|---:|---:|---:|---:|
| geo_only | 12.26° | 1.026m | 0.152m | 0.054m | 0.083m |
| geo + human_token residual | 6.56° | 0.903m | 0.150m | 0.080m | 0.100m |

对比结论：

1. `geo + human_token residual` 明显改善 camera 指标。
   相比 `geo_only`，camera rotation 从 `12.26°` 降到 `6.56°`，camera translation 从 `1.026m` 降到 `0.903m`。

2. human post error 略有改善，但 A/B 人体锚点一致性略差。
   `human_post_m` 从 `0.152m` 到 `0.150m`，变化很小；但 `Amean-B0/B1` 从 `0.054/0.083m` 变成 `0.080/0.100m`，说明 residual 更偏向修 camera，不一定保护人体锚点。

3. 这版几乎复现了之前的 `geo_residual_guarded`。
   也就是说，在当前小 4source 训练集上，加入 `human_token_out` 后没有明显超过纯 geometry residual。当前判断是：human token 有信息，但直接拼进 residual MLP 还没有转化成稳定增益。

下一步更值得尝试的不是简单继续加 token，而是让 token 明确承担一个更具体的职责：

- 预测 `T_geo` 的可靠性；
- 预测 anchor 权重，而不是直接参与 residual；
- 控制 residual gate；
- 或者在 detector / state update 中使用 token，而不是在 final residual head 里裸拼接。

### 12.1 Held-out 泛化测试

同一天又用同一个 `geo + human_token residual` checkpoint 做了 held-out 测试：

- 训练 checkpoint：`output/v10_geometry_token_residual_probe/4source_s2_geo_human_token_20260709/alignment_head_geo_token_residual.pth`
- 测试数据：同一份 4source angle>=60 AABB manifest，但每个 source 跳过前 2 条训练样本，取第 3-4 条；
- 输出：`output/v10_geometry_token_residual_probe/4source_s2_geo_human_token_heldout_offset2_20260709`
- 模式：eval-only，不更新权重。

整体指标：

| Variant | Cam Rot ↓ | Cam Trans ↓ | Human ↓ | Amean-B0 ↓ | Amean-B1 ↓ |
|---|---:|---:|---:|---:|---:|
| geo_only | 6.85° | 1.170m | 0.153m | 0.044m | 0.081m |
| geo + human_token residual | 10.37° | 1.169m | 0.177m | 0.100m | 0.131m |

结论：

1. `geo_only` 在 held-out 上反而更稳。
   它的 camera rotation、human error、A/B anchor consistency 都优于当前 token residual 版本。

2. 当前 `human_token residual` 有明显过拟合迹象。
   在 8 条训练样本上 residual 能把 camera rotation 从 `12.26°` 降到 `6.56°`，但在 held-out 上从 `6.85°` 变差到 `10.37°`。

3. 这说明裸拼接 `human_token_out` 预测 residual 不是当前最可靠路线。
   token 里有信息，但这个小 MLP 学到的 residual 不够泛化，容易破坏已经很强的几何 proposal。

下一步更合理的是把 `T_geo` 作为主输出，把学习模块从“修正 SE(3)”改成更保守的角色：

- residual 默认接近 0；
- 只有在 geometry proposal 明显不可靠时才开 gate；
- token 预测 `confidence/gate/anchor weights`，而不是直接输出大 residual；
- 或者先做中等规模训练，看更多数据能否让 residual 稳定，但不建议直接做大规模长训。

## 13. Route C2：Learned Anchor Weight Instead of SE(3) Residual

根据 `geo + human_token residual` 的泛化结果，2026-07-09 又做了一版更保守的学习模块：

```text
不是让 token 预测 SE(3) residual
而是让 token/geometry feature 预测人体 anchor 权重
然后仍然用 weighted Procrustes 显式求 T_geo
```

代码：

- `scripts/v10_geometry_anchor_weight_probe.py`

这版的核心变化：

1. 显式几何仍然是主干。
   最终 transform 仍由人体 anchor 的加权刚体配准求出，不是 MLP 直接输出 6DoF。

2. 学习模块只决定 anchor 权重。
   它可以让 pelvis / hip / torso / head / feet / hand 等不同 anchor 获得不同权重。

3. 第一版带 gate 的设计有两个极端：
   - gate 开到 1，权重容易塌缩，训练集相机更好但泛化略差；
   - gate 被 prior 压到 0，则完全退化成 fixed_geo。

4. 后续更稳定的是 `no-gate bounded anchor weight`：
   - 直接预测小范围权重偏移；
   - `max_logit_delta=1.0` 限制权重变化幅度；
   - `weight_prior=5.0` 防止权重塌缩；
   - 不再用 gate 混合 base weights 和 learned weights。

### 13.1 小 4source 训练集指标

输出：

- `output/v10_geometry_anchor_weight_probe/4source_s2_human_token_nogate_20260709`

| Variant | Cam Rot ↓ | Cam Trans ↓ | Human ↓ | Amean-B0 ↓ | Amean-B1 ↓ |
|---|---:|---:|---:|---:|---:|
| fixed_geo | 12.26° | 1.026m | 0.152m | 0.054m | 0.083m |
| learned anchor weight, no gate | 6.46° | 0.906m | 0.148m | 0.054m | 0.086m |

训练集上，这版能明显改善 camera rotation / translation，同时基本不破坏 human anchor。相比直接 residual，它更符合“几何 proposal 为主，学习模块只做可靠性/权重选择”的设计目标。

### 13.2 Held-out 泛化指标

输出：

- `output/v10_geometry_anchor_weight_probe/4source_s2_human_token_nogate_heldout_offset2_20260709`

| Variant | Cam Rot ↓ | Cam Trans ↓ | Human ↓ | Amean-B0 ↓ | Amean-B1 ↓ |
|---|---:|---:|---:|---:|---:|
| fixed_geo | 6.85° | 1.170m | 0.153m | 0.044m | 0.081m |
| learned anchor weight, no gate | 6.81° | 1.157m | 0.153m | 0.047m | 0.083m |

泛化测试说明：

1. no-gate anchor weight 没有像 direct residual 那样崩。
   held-out 上 camera rotation 和 translation 略优于 fixed_geo，human 基本持平。

2. A/B anchor consistency 略差一点，但幅度很小。
   `Amean-B0` 从 `0.044m` 到 `0.047m`，`Amean-B1` 从 `0.081m` 到 `0.083m`。

3. 当前增益还很小，不足以直接做大规模长训。
   但它证明了方向比 direct residual 更合理：学习模块应控制 anchor 权重/可靠性，而不是直接修 SE(3)。

建议下一步：

- 用 no-gate anchor-weight 作为中等规模 probe；
- 每个 source 先取几十到一百条，验证增益是否随数据量变稳定；
- 暂时不要直接做 50h/60h 大训练；
- 如果中等规模仍只带来极小增益，则 fixed_geo 可以作为非常强的 V10 baseline，学习模块主要放到 detector/state update 上。
