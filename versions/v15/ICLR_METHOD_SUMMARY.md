# Movie3R-v15：面向 ICLR 的完整方法总结

本文档把目前已经完成的工程、受控实验和论文故事放在同一条主线上。它是研究总结，不把尚未在正式大规模 split 上验证的结果写成最终结论。

## Abstract（工作摘要）

流式三维重建模型通常在单一连续镜头内表现良好，但在镜头切换时会出现坐标系跳变、相机估计失真和人物身份置换。我们提出 Movie3R，一个不依赖额外预训练模型的因果 camera-human gauge correction framework。对于每个检测到的 shot boundary，Movie3R 首先用 Human3R/V9 shadow branch 给出 learned coarse gauge proposal；随后在该坐标系中进行匿名跨镜头身份匹配，并利用五个稳定人体关节的两视线三角化完成相机冻结的人体 root/depth 精对齐。对于 shot 内静止人物，因果 EMA 稳定器抑制残余漂移；对于低纹理场景，系统从同一模型的 raw shadow branch 提取人体 root rays，与 B0 人体残差共同构成一个置信度门控的 camera-human 联合更新。证据不足时系统明确 abstain 并保留上一可信状态。该设计把“相机错”和“人体错”统一为一个在线 gauge transaction，同时保持 pre-cut 帧不变、只使用 boundary 之前的信息。

当前受控实验显示，B0 是必要的粗初始化和人物身份预条件：在 AvatarReX 单人低纹理案例中，no-V9 raw SE(3) 的相机误差为 2.107 m / 64.53°，加入 V9 后的 adaptive joint 为 0.054 m / 0.44°；在约 174° 的三人跨镜头案例中，no-V9 为 4.265 m / 151.89°，B0+BRTC+C1 为 0.054 m / 1.82°。在 41 个多人 cuts 上，原版 Human3R 的直接跨镜头匹配准确率为 41.5%--46.3%，B0 后为 100%。这些结果说明方法方向成立，但正式 ICLR 结论仍需在统一 checkpoint 和公开协议上完成大规模评估。

## 1. Introduction

### 1.1 问题

输入是一段连续 RGB 流，底层 Human3R 对每帧输出相机、背景和 SMPL-X 人体。镜头切换时，post shot 的模型状态和坐标 gauge 可能与 pre shot 不一致。典型的两类失败是：

- 多人且背景纹理丰富：相机可能已经接近正确，但 detection index 在两个 shot 中置换，导致人物互换或 180° 错配；
- 单人且背景低纹理：人体的局部结构仍有信息，但背景不足以确定相机的绝对移动，直接使用相机 SE(3) 会把相对正确的人体带到错误的世界位置。

这说明“只对相机”或“只对人体”都不是通用答案。相机和人体必须共享一个可验证的更新；同时，流式系统不能等待未来帧，也不能在不确定时强行修正。

### 1.2 核心假设

我们不假设背景永远可观测，也不假设 Human3R 已经提供 persistent identity。我们只依赖三种边界证据：

1. V9 shadow/raw 两次同 checkpoint 推理给出的粗 camera/body gauge；
2. 人体五个核心关节在两个相机中的投影射线；
3. B0 后人体形状、root、torso 和 centered joints 的跨 shot 一致性。

如果这些证据互相矛盾，最优的在线动作不是继续优化，而是 abstain。

### 1.3 贡献

1. 提出一种把 shot cut 视为 **causal gauge transaction** 的 streaming 设计：proposal、association、human refinement、joint commit 和 fallback 都是独立且可审计的状态。
2. 提出 **identity-preserving B0+BRTC-LC**：学习粗 gauge 只负责把问题带入可匹配区域，五关节射线与布局共识负责显式修正人体 root/depth，而不是把整个人体和相机盲目刚体搬动。
3. 提出 **adaptive shared camera-human correction**：在背景低纹理时利用 raw shadow 人体 root rays 作为相机平移辅助，在几何 gate 通过后用同一更新同时改变 camera 与 human，保持其相对关系。
4. 给出 causal/GT-free/fallback contract，使所有修正可以在 CPU 上复现，并把 gate acceptance、ID margin、seam jump 和运行时间纳入评估。

## 2. Related Work 的写法

相关工作应分为四组，而不是把 Movie3R 描述成另一个 pose network：

1. **Streaming feed-forward 3D reconstruction**：Human3R、DUSt3R/CUT3R 等模型提供相机、点图和人体，但通常没有跨 shot 的 persistent identity transaction。
2. **Camera/scene alignment and gauge fixing**：传统 SE(3)/Sim(3) 对齐通常假设可用的场景锚点；Movie3R 研究的是场景锚点不可靠时如何切换到人体证据并保持相对几何。
3. **Multi-person association / tracking**：检测 index、姿态匹配和 Re-ID 方法解决的是身份线索；Movie3R 的重点是身份匹配如何成为跨 shot gauge correction 的前置条件，并且允许 unmatched/abstain。
4. **Online filtering and robust estimation**：EMA、hysteresis、RANSAC/triangulation 等提供局部工具；Movie3R 把它们组织成一个 causally gated camera-human transaction，并报告拒绝率与错误代价。

论文中要明确：我们不声称解决遮挡、完整 Re-ID 或所有 articulated pose error；方法聚焦于 shot boundary 的坐标一致性、人物身份保持和低纹理 camera-human 联合校正。

## 3. Method

### 3.1 记号与输入

令 boundary 为 `b`，最后 pre 帧为 `b-1`，第一 post 帧为 `b`。模型对每帧输出：

- 相机到世界矩阵 `C_t ∈ SE(3)`；
- 背景 color/depth/confidence；
- 第 `i` 个人的世界 mesh `V_t^i`、root `r_t^i` 和 joints `J_t^i`；
- 当前帧 native detection index `s_t^i`。

`pre` 状态是已提交的可信状态；post 第一帧只读取 `b`，后续帧复用已接受的 boundary update。

### 3.2 Shadow B0：learned coarse gauge proposal

对 pre 序列和第一 post 帧运行带 shot boundary 的 shadow branch，对 post 序列 clean reset 得到 raw branch。两者相机矩阵给出粗变换：

\[
T_{B0}=C_{shadow,b}\,C_{raw,b}^{-1}.
\]

`T_B0` 同时作用于 raw post camera、背景和人体，得到统一的 `B0` world gauge。这里的 V9 并不被声称为最终对齐器；它的作用是把相机和人体带到可比较的粗坐标系，并显著减少后续 permutation search 的错误区域。

### 3.3 Anonymous identity association

对每个 post detection 与 pre track 计算三类不依赖全局绝对位置的描述：root 距离、torso 方向/尺寸、centered joints 或 centered mesh 的 shape residual。对有限人数枚举一对一 permutation，保留最佳代价和第二佳 margin：

\[
\pi^*=\arg\min_\pi \frac{1}{N}\sum_i d_{shape}(P_{post}^{\pi(i)},P_{pre}^i).
\]

margin 不足时不创建新的 persistent ID，也不把错配结果强行送入几何修正。`smpl_id` 只被当作帧内索引，不能当作跨 shot ID。

### 3.4 BRTC-LC：camera-frozen human refinement

选取 pelvis、left/right hip、left/right shoulder 五个关节。对于 pre 相机中心 `o_0`、post B0 相机中心 `o_1` 和两帧对应的归一化射线 `u_0,u_1`，求两条直线的最近点 `x_k` 和 ray gap `g_k`。只有 gap、parallax sine 和跨关节 MAD 通过门控的关节才参与估计。

有效关节的 group median 给出稳健 root shift；再用 pre layout 选择 individual residual。最终只更新：

\[
V_{post}^{i,new}=V_{post}^{i}+\Delta r_i,\quad
J_{post}^{i,new}=J_{post}^{i}+\Delta r_i.
\]

相机 `C_t`、人体朝向、pose 和 shape 不被 BRTC 改动。这样可以在相机已经可信的多人场景中修正人物 ID/深度，而不破坏背景。

### 3.5 C1-EMA25：within-shot stabilization

在一个 shot 内，以 camera-local root/body step 为输入，使用 `alpha=.25` 的 causal EMA。root/body enter-exit 阈值和 3 帧 moving hold 防止把真实运动滤掉。稳定 track 的修正被一致地写入 root、joints 和 vertices；所有其他 track 使用 B0+BRTC 值。相机保持不变。

### 3.6 Adaptive joint gate

当背景低纹理时，B0 camera translation 可能不可靠。令 `R_B0` 是 B0 post body 到 pre body 的共享刚体残差。其旋转由 body Kabsch 给出；raw shadow 分支提供相同人物的 root rays。对当前 root `r_i` 和候选相机旋转 `R`，相机平移由 B0/raw 两类 ray 的平均约束：

\[
t^*=\operatorname{mean}_i\left(r_i-Rq_i^{mean}\right).
\]

人体绕当前 BRTC root 施加 `R`，相机使用同一个 `R,t*`。提交前检查：共享 rotation 至少 20°、vertex RMS ≤ .20 m、归一化 RMS ≤ .20、最佳 permutation margin ≥ .01 m。任一条件不满足就保留 baseline。这是“相机和人体联合修正”而不是先固定一个错误相机再强行贴人体。

### 3.7 Causal state machine

每个 track 保存最近 root、body step、EMA 状态、track age 和 moving-until frame。每个 boundary 保存 detector probability、B0 transform、association permutation/margin、BRTC diagnostics、joint gate decision 和 fallback reason。这样可以在 demo、离线评估和线上系统中得到完全一致的行为。

## 4. 实验设计

### 4.1 数据与场景分层

正式实验应覆盖：

- AvatarReX：单人、低纹理、相机不可靠，重点考察 joint gate；
- MultiHuman/MVHuman：多人、纹理较强、重点考察 ID continuity 和 BRTC；
- EgoHuman：真实 ego/shot 变化，重点考察 detector、camera error 和 within-shot drift；
- Multi-THuMBS：按照官方协议报告 camera、人和多人的指标，不能用自定义 split 冒充官方结果。

每个数据集都要按 cut angle、人数、纹理强弱、运动/静止状态分层，避免单一好看的 demo 代表全部性能。

### 4.2 Baselines 和消融

所有 end-to-end 表格固定一个 checkpoint。必须有 strict Human3R、no-V9 raw SE(3)、B0、B0+BRTC、B0+BRTC+C1、完整 adaptive joint，以及 oracle boundary/causal detector 两组。每一项都要保存预测 payload 和 diagnostics，便于重新计算指标。

### 4.3 指标

- camera：translation/rotation error；
- human：MPVPE、root、world joint、pairwise distance/vector；
- temporal：cut seam jump、within-shot drift、静止人物速度；
- identity：跨 cut ID continuity、permutation accuracy、unmatched/entry/exit；
- adaptive behavior：gate acceptance、abstention、误接受/误拒绝；
- systems：CPU latency、memory、是否读取未来帧。

## 5. 当前证据与应如何表述

### 已有证据

- V9/no-V9 两个受控案例显示 V9 对粗 gauge 和相机初始化有决定性作用；
- 41 个多人 cuts 显示原版 native index 不是 persistent ID，而 B0 后 shape matcher 可达到 100%；
- 42 cuts/125 people 的 BRTC-LC 确认集显示 root/joint/vertex 都有明显下降且相机不变；
- causal GRU detector 的审计指标优于 static logistic 和 causal MLP。

### 尚不能写成最终 claim 的内容

- 目前没有完整官方 Multi-THuMBS leaderboard 复现；
- adaptive joint 尚应在大规模统一 split 上报告接受率和失败代价；
- severe occlusion、新人物进入/离开和复杂 Re-ID 尚未解决；
- 两个 checkpoint 的结果必须分表，不能混合成一条“最终提升”。

## 6. 论文落地顺序

1. 用 `FINAL_RUNTIME_SPEC.json` 和统一 batch manifest 跑完各数据集的所有消融。
2. 先锁定 camera/human/ID/temporal 四张主表，再做按纹理、人数、cut angle 的分层分析。
3. 复现 Multi-THuMBS 官方指标和 protocol，明确标注数据重合与不可比之处。
4. 用失败案例验证 abstain 是否比错误更新更安全，补充定性 demo 和 runtime 分析。
5. 最后再整理摘要、图表和 limitations；不要用单个 demo 代替泛化结论。
