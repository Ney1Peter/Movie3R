# Movie3R ICLR 最终对比实验计划（2026-08-07）

本文档回答三个问题：最后应该在哪些数据上比较、比较哪些方法和指标、如何把 Movie3R 的真实优势证明出来。

结论先说：Movie3R 的主任务不是普通单帧 HMR，也不是单纯的相机 ATE，而是 **causal multi-shot 3D human-scene reconstruction**。因此主表必须同时覆盖 camera、world human、cross-shot identity、temporal continuity 和 online cost；只报告 MPJPE 或只报告 ATE 都无法证明项目价值。

## 1. 相关工作的任务边界

### 1.1 Multi-THuMBS：最直接的外部对标

`Multi-THuMBS` 的论文目标与 Movie3R 最接近：多人、多 shot、跨 shot 的全局人体轨迹、相机和身份一致性。它在 boundary 两帧构造 shared 3D space，逐人优化 root/orientation/camera，再做 geometry+appearance+pose Re-ID 和全序列 smoothing。

论文主文提到的数据为：

- **EgoHumans**；
- **EgoBody**；
- **Harmony4D**；
- 没有完整 3D GT 的自然编辑视频：**AVA、Friends、The Big Bang Theory**。

论文公开的直接 baseline 为 `Multishot`、`GVHMR`、`PromptHMR` 和适配多人 multi-shot 的 `HSfM†`；Re-ID 额外比较 `KPR`、`Pose2ID`。论文没有公开完整 evaluator、cut manifest 和 supplementary，因此当前本地 `EgoHumans 001_legoassemble` 只能称为同源/协议近似测试，不能称为官方 split。详见 [Multi-THuMBS 审计](/data/wangzheng/iJCV-CODE/Movie3R/versions/v14/docs/V14_MULTITHUMBS_PUBLIC_PROTOCOL_DATA_OVERLAP_AUDIT_20260801.md)。

### 1.2 HumanMM：单人 multi-shot 的强参考

HumanMM 主要验证多 shot 世界坐标人体运动恢复，构建了 `ms-Motion`：

- `ms-AIST`：AIST 多摄像机数据；
- `ms-H3.6M`：Human3.6M 多摄像机数据；
- 共 600 个视频、42.7K 帧、约 237 分钟，2/3/4-shot。

它比较 SLAHMR、WHAM、GVHMR，指标包括 PA-MPJPE、WA-MPJPE、W-MPJPE、PVE、Accel、RTE、ROE、Jitter、Foot Sliding，以及 shot detector 的 Precision/Recall/F1 和 camera 的 ATE/RPE。HumanMM 是单人或单主体运动连续性的参考，不应作为 Movie3R 多人主表的唯一对手，但非常适合验证 AvatarReX 单人低纹理场景。

### 1.3 Human3R / CUT3R / TTT3R：底层在线基线

Human3R 是 Movie3R 的基础模型，也是必须严格复现的 baseline。它本身是在线、单次前向、同时输出多人体、相机和场景的模型，但没有为 shot boundary 提供 Movie3R 这种显式的跨 shot transaction。原版 Human3R 的公开实验为：

- 局部人体：3DPW、EMDB-1，PA-MPJPE、MPJPE、PVE；
- 全局人体：EMDB-2、RICH，WA-MPJPE、W-MPJPE、RTE；
- 通用相机：TUM-D，Sim(3)-aligned ATE；
- 视频深度：Bonn，Abs Rel、δ<1.25；
- 效率：FPS、显存。

CUT3R、TTT3R、ReCal3R 和 VGGT 是通用场景/相机 reconstruction 参照：它们可以放在 TUM-D/Bonn 或 camera-only ablation 中，但不能直接当作完整多人跨 shot 方法与 Movie3R 一行比较。

### 1.4 其他相关方法

- **Multishot / Recovering Human Mesh from Multiple Shots**：单人 multi-shot 的早期直接方法，若代码可运行，应放在单人 multi-shot 表。
- **GVHMR、WHAM、TRAM、SLAHMR**：强 global human motion baseline；需标明它们是否需要预计算 camera、mask、depth、tracking，不能隐藏额外输入。
- **ShowMak3r**：多人 multi-shot 的 scene/rendering 方向，可做定性或 scene consistency 参考，但论文重点不是 motion/ID accuracy。
- **KPR、Pose2ID**：appearance/pose Re-ID 参考，适合 ID-only 表，不应被当作完整 camera-human reconstruction baseline。
- **UniCon3R**：contact-aware single-shot human-scene baseline，可用于说明 Movie3R 不依赖接触监督；不是跨 shot 直接竞争对手。

## 2. 最终数据集分层

### P0：论文主表，必须完成

| 数据集/协议 | 场景 | 为什么必须用 | 主要指标 |
|---|---|---|---|
| EgoHumans | 真实多人、ego/exo、多相机、相机和人体 GT | Multi-THuMBS 的主数据之一，也是我们已有数据和多人 ID 的核心 | W/WA-MPJPE、MPJPE、MPVPE/PVE、Accel、ATE、IDs、ROE、seam jump |
| EgoBody | 真实多人/人体-相机世界坐标 | 与 Multi-THuMBS 对齐，验证跨人群和视角泛化 | 同上 |
| Harmony4D | 真实近距离多人交互、遮挡、相机运动 | 检验多人交互和可见人数变化；不能只在简单多人上得结论 | 同上，特别是 ID、coverage、occlusion 分层 |
| Multi-THuMBS protocol | 上述三类数据按照论文 cut/clip 构造 | 论文最终外部胜负线 | 完全复现论文表 1/2/4 |

重要说明：当前本地 `data/EgoBody/001_legoassemble` 根据官方配置实际属于 EgoHumans，不应在论文里错误标成 EgoBody。官方 Multi-THuMBS 的具体 sequence/cut/aggregation 尚未公开，拿不到时应把本地协议命名为 `Movie3R-CrossShot-v1`，并单独报告“same-source protocol-matched”，不能写成“官方 Multi-THuMBS 结果”。

### P1：证明 Movie3R 独特优势，必须完成

| 数据集/协议 | 目标 failure mode | 为什么对 Movie3R 重要 |
|---|---|---|
| AvatarReX held-out single-person | 低纹理背景、相机天然不可靠、人体结构相对稳定 | 证明 raw/B0 camera 错时，adaptive camera-human joint gate 仍能把人和相机一起修正 |
| MultiHuman / MVHuman held-out multi-person | 多人、高纹理、180° 或大跨度、人数变化 | 证明背景相机已经较准时，B0+ID+BRTC 不会破坏相机，并能修正人物置换 |
| EgoHuman/EgoHumans multi-cut chains | 真实 shot 链、多人、连续两个以上 cut | 证明方法不是只对单 boundary 有效，而是 causal state 可以累积传播 |
| ms-AIST / ms-H3.6M（若可取得 HumanMM benchmark） | 单人/主体 2、3、4-shot | 与 HumanMM 用同一指标和同一 shot 数量比较 orientation、trajectory、jitter、foot sliding |

AvatarReX、THuman、MVHuman 如果曾用于 V9/Human3R 训练，正式测试必须按 identity/sequence 做 disjoint split；训练来源不能直接当泛化测试集。THuman/MVHuman 更适合作为训练域和 held-out identity 测试，不能把训练样本结果放进主表。

### P2：防止方法退化的继承能力测试

| 数据集 | 作用 | 指标 |
|---|---|---|
| 3DPW、EMDB-1 | 局部人体质量不应因跨 shot 模块而退化 | PA-MPJPE、MPJPE、PVE/MPVPE |
| EMDB-2、RICH | 全局人体和长时轨迹能力不应退化 | W/WA-MPJPE、RTE、Accel |
| TUM-D | 证明相机和场景分支没有被人体修正破坏 | Sim(3) ATE、RPE trans/rot |
| Bonn | 证明深度/metric scene 输出没有回归 | Abs Rel、δ<1.25 |
| AVA、Friends、Big Bang Theory | 无 GT 的真实编辑视频定性和弱监督 motion quality | PCK*、Jitter、Foot Sliding |

P2 不一定都放进主表，但至少要有一张 regression table 或 appendix，说明 Movie3R 的跨 shot 修正没有损害 Human3R 原本的单 shot 能力。

## 3. 必须比较的方法矩阵

### 3.1 Movie3R 内部主消融

所有方法必须使用相同 RGB、相同 checkpoint、相同 frame window、相同 GT evaluator。不能把不同 checkpoint 的结果混在一张 end-to-end 表。

| 方法 | 作用 |
|---|---|
| Strict original Human3R | 原始 state 跨 cut 的真实 baseline；不传 cut 事件、不做 B0 |
| Current Human3R raw/reset | 每个 post shot clean reset，但不做跨 shot 对齐；区分 state reset 影响 |
| No-V9 raw SE(3) | 直接用 raw 相机算 SE(3)；验证 V9/B0 是否必要 |
| B0 only | learned coarse gauge 的单独贡献 |
| B0 + identity | 只加入 persistent cross-shot ID association |
| B0 + BRTC-LC | 相机冻结的人体 root/depth/layout 精对齐 |
| B0 + BRTC-LC + C1-EMA25 | 加 shot 内静态人体稳定 |
| B0 + BRTC-LC + adaptive joint | 完整最终方法 |
| Full + oracle boundary | 给定真实 cut 的上界；不能替代 causal detector 结果 |
| Full + causal GRU detector | 论文默认部署版本 |
| Full + static/logistic detector | detector 消融，证明可学习 detector 的作用 |

此外要单独报告 `B0+BRTC+C1` 与 `Full adaptive` 的 gate acceptance/fallback；如果 adaptive 在某场景拒绝，最终结果必须保留 baseline，而不是静默删除该 case。

### 3.2 外部 baseline

主表按可公平复现程度分三层：

1. **强直接对手**：Multi-THuMBS、HumanMM、Multishot。它们都明确处理 shot transition，但部分是离线/单人或需要额外优化。必须报告额外输入、是否使用未来帧、是否迭代优化。
2. **底层重建对手**：严格 Human3R、CUT3R/TTT3R、VGGT；用于 camera/scene/general 3D regression 和说明 Movie3R 的上游基础。
3. **人体运动/身份对手**：GVHMR、WHAM、TRAM、SLAHMR、PromptHMR、HSfM†、KPR、Pose2ID；只在其能覆盖的任务表中比较，不把缺少多人、相机或 shot 能力的方法伪装成完整同任务 baseline。

每个外部 baseline 要记录：输入是否需要 2D detector、mask、depth、SLAM、GT intrinsics、GT cut、future frames；是否有 per-scene optimization；运行时间和 GPU。Multi-THuMBS 报告 150 帧约 10 分钟 RTX 3090，Human3R 是在线单次前向，Movie3R 需要把额外 CPU geometry cost 单独列出。

## 4. 指标设计：主表和补充表

### 4.1 Shot detector

对 RGB-only causal detector 报：

- Precision、Recall、F1；
- false positive / 100 frames；
- boundary onset delay（帧）；
- tolerance `±1` 和 `±2` 帧两种口径；
- causal latency 和每帧额外时间。

主运行可用 manifest 给出的 oracle boundary，但论文必须同时给出 `oracle boundary` 与 `causal detector` 两列，否则无法证明在线性。

### 4.2 相机与场景

必须同时报告两种 camera 误差，避免 scale/gauge 争议：

1. **Metric boundary error**：第一 post 帧 camera center 的 translation error（m）和 rotation error（deg），直接在 GT metric world 中计算；
2. **Trajectory metrics**：ATE、RPE translation、RPE rotation；明确使用 SE(3) 还是 Sim(3)，alignment 只允许使用规定的 train/eval segment，不能用整段 GT 偷调。

对每个 cut 额外报告：

- camera seam jump：post-first 与 pre-last 的相对相机变化误差；
- background reprojection/depth error（有 GT 的数据）；
- camera change max/mean after BRTC（理论上 BRTC 为 0）；
- adaptive camera update acceptance 和 translation/rotation magnitude。

Human3R 官方 TUM-D 使用 Sim(3)-aligned ATE；Multi-THuMBS 只公开 ATE 名称，具体 alignment 未公开，所以论文中必须同时给出 protocol 和实现。

### 4.3 人体空间质量

主指标：

- `W-MPJPE`：前两帧/初始 boundary alignment 后的世界轨迹误差；
- `WA-MPJPE`：整段 trajectory alignment 后的误差；
- `MPJPE`、`MPVPE/PVE`：局部 pose/mesh 质量；
- `PA-MPJPE`、`PA-MPVPE`：去除 global gauge 后的局部形状/姿态质量；
- `root translation error`；
- `ROE`：root/global orientation error；
- `world joint/vertex error`：不做 pelvis 对齐，专门衡量 Movie3R 的 world placement。

要明确：如果只报告 pelvis-aligned MPJPE，BRTC 的刚性 root shift 会被抵消，无法反映跨 shot 对齐效果。因此主表必须同时有 world root/joint/vertex 和 PA/local 指标。

### 4.4 人物 ID 与可见性

这是 Movie3R 相对于原版 Human3R 的关键优势，不能只写一个视觉案例。建议报告：

- boundary identity accuracy；
- identity switches per 100 boundaries；
- track continuity / IDF1（若能统一 MOT matching）；
- matched precision、matched recall、unmatched/entry/exit coverage；
- permutation margin 分布；
- 误匹配后的 human error 与安全 fallback 比例。

Multi-THuMBS 的 `IDs` 是必须复现的论文命名指标；同时保留 Movie3R 自己的 `association accuracy` 和 `coverage`，因为 Multi-THuMBS 主文没有公开漏检、进入/退出和 aggregation 规则。

### 4.5 时间连续性

- `Accel`：与 Multi-THuMBS 同名指标；明确二阶差分、坐标系、fps 和单位；
- `Jitter`：逐帧轨迹抖动，按 HumanMM/Multishot 的公开口径复现；
- `Foot Sliding`：接触状态下脚部顶点平均位移；
- `cut seam jump`：root、joint、orientation 在最后 pre/第一 post 的跳变；
- `within-shot drift`：静止人物在 shot 内的 root/world vertex 漂移；
- moving-person distortion：运动人物不应被 C1 过度平滑。

Movie3R 的自适应 gate 目标不是让所有人都静止，而是静止人物稳定、运动人物 fallback。因此必须按 static/moving 分层，不能只报整体平均。

### 4.6 在线效率和安全行为

至少报告：

- FPS / latency per frame；
- boundary extra latency；
- CPU/GPU memory；
- 额外预训练模型数；
- 是否使用未来帧；
- detector proposal rate、geometry gate acceptance、abstention rate；
- accepted update 的 improve rate、harm rate（例如 root error 增加 >5 cm 的比例）；
- 从 boundary 到输出稳定结果的延迟。

这组指标直接对应 Movie3R 的论文卖点：online、causal、无额外预训练模型、可信才修正。

## 5. Multi-THuMBS 的参考胜负线

论文 Table 1/2 中公开的 Multi-THuMBS 参考值如下。只有在同一官方 split、同一公式和同一 aggregation 下才能使用这些数值宣称胜负。

| 数据集 | W-MPJPE | WA-MPJPE | MPJPE | MPVPE | Accel | ATE | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|
| EgoHumans | 279.0 | 166.0 | 228.3 | 262.2 | 27.3 | 0.7 | 0.97 |
| EgoBody | 99.2 | 72.8 | 72.0 | 94.9 | 6.0 | 0.1 | 0.00 |
| Harmony4D | 221.0 | 116.9 | 215.9 | 278.3 | 17.4 | 0.7 | 0.46 |

Harmony4D 的 MPVPE 还要同时关注 `HSfM†=257.6`，因为该项低于 Multi-THuMBS 的 278.3；不能只以 Multi-THuMBS 自己的数值作为所有指标的 SOTA 线。

这些值目前应标记为 `literature reference`，不是我们已经复现的数字。Multi-THuMBS 主文没有公开完整 evaluator，当前本地 provisional evaluator 也明确不能冒充官方实现。

## 6. 针对 Movie3R 优势的必做分析

### A. 低纹理单人：证明 adaptive joint 真正解决核心问题

在 AvatarReX 上按背景纹理、camera angle、单人运动速度分层，比较：

```text
strict Human3R
→ raw/reset
→ no-V9 raw SE(3)
→ B0+BRTC
→ B0+BRTC+C1
→ Full adaptive joint
```

重点看 camera boundary translation/rotation、world root/joint/vertex、seam jump 和 gate harm。预期论文叙事是：背景相机不可靠时，单独 camera SE(3) 会失败；人体 root rays + B0 body residual 的联合 gate 才能同时恢复 camera-human relative placement。

### B. 多人高纹理：证明 B0/ID/BRTC 的互补性

在 MultiHuman/EgoHumans/Harmony4D 上选择 2/3/4 人、人数变化、180° 大跨度和遮挡分组，比较：

- original Human3R native index；
- B0 后 anonymous matching；
- B0+BRTC；
- Full adaptive 是否安全 fallback。

重点指标为 ID switches、permutation accuracy、world root/joint/vertex、pairwise distance/vector、camera non-regression。要明确展示：多人场景相机可能本来就对，但人物 ID/布局仍然错；Movie3R 不是只靠修相机得到收益。

### C. 在线性：证明不是离线优化换来的收益

把 `oracle boundary` 和 `causal GRU detector` 分开。对每个 boundary 只允许最后 pre/第一 post 和历史状态；禁止使用完整 shot smoothing、future post frame 或 GT identity。额外报告 detector delay、gate delay、CPU latency 和内存。

### D. 安全性：证明拒绝比错误修正更好

画出 gate acceptance 与 error/harm 的 reliability curve：

- accepted cases 的 improve/harm；
- rejected cases 的 baseline error；
- gate threshold sweep 只在 train/val 调，test 只用冻结阈值；
- static、moving、unmatched、visibility-change 分层。

这部分是 Movie3R 和“所有边界都强行优化”的根本区别。

## 7. 最终论文表格建议

### 主表 1：Multi-THuMBS-style cross-shot reconstruction

行：Multishot、GVHMR、PromptHMR、HSfM†、strict Human3R、B0、B0+BRTC、Full Movie3R。列：W/WA-MPJPE、MPJPE、MPVPE、Accel、ATE、IDs。数据集分 EgoHumans/EgoBody/Harmony4D 三组。

### 主表 2：Movie3R boundary 和 identity

行：strict、no-V9、B0、B0+ID、B0+BRTC、Full；列：camera boundary trans/rot、world root/joint/vertex、seam jump、ID accuracy、IDs/100 cuts、coverage、harm。

### 主表 3：HumanMM-style multi-shot motion

在 ms-AIST/ms-H3.6M 或等价可公开协议上，列 PA、WA、W、RTE、ROE、Jitter、FS、Accel。此表专门证明单人/主体 multi-shot，不与多人 Multi-THuMBS 表混合。

### 主表 4：在线效率和 detector

列 shot F1/latency、FPS、boundary overhead、memory、future frames、extra pretrained models、gate acceptance。把 Human3R、HumanMM、Multi-THuMBS 的离线/在线差异显式标出来。

### Appendix：回归、失败和定性

3DPW/EMDB/RICH/TUM-D/Bonn；AVA/Friends/BBT；低纹理、180°、人数变化、遮挡、entry/exit 的 demo。每个失败例说明是 detector、ID、camera、human geometry 还是 fallback 触发。

## 8. 公平性规则

1. 同一数据集用同一 RGB、GT、intrinsics convention、frame rate 和 camera coordinate convention。
2. 训练身份、验证身份、测试身份严格分离；AvatarReX/THuman/MVHuman 参与训练的来源不能直接当 test。
3. 外部方法需要的 detection、mask、depth、SLAM、GT intrinsics 和 future frame 必须逐项列出。
4. 不把 GT cut 结果当成线上主结果；oracle 只作上界。
5. 不把原版 Human3R 的 local pose 数字、B0 的 camera 数字和 Full Movie3R 的 world 数字混成一条结果。
6. 每个结果至少报告 3 个随机/序列聚合统计或 bootstrap 置信区间；单个漂亮 demo 只能放 qualitative。
7. 对每个方法保存逐帧 payload、identity mapping、camera pose、gate diagnostics，允许第三方重新计算指标。
8. 官方 Multi-THuMBS evaluator 未公开前，表格标题写 `literature reference` 或 `protocol-matched local evaluation`，不能写 `official reproduced`。

## 9. 建议执行顺序

### 第 1 阶段：协议闭环

冻结 `Movie3R-CrossShot-v1` manifest、GT ID association、missing/entry/exit 规则、单位和 aggregation；先跑 strict Human3R、B0、BRTC、Full 四个核心版本。

### 第 2 阶段：P0 主数据

扩展 EgoHumans、EgoBody、Harmony4D 的多 sequence、多 camera pair、多 cut；生成 Multi-THuMBS-style 全指标表，单独保存官方 protocol 缺口。

### 第 3 阶段：P1 优势分析

跑 AvatarReX 低纹理和 MultiHuman/EgoHumans 多人分层；重点完成 no-V9、camera-only、human-only、joint gate、ID 和 harm 曲线。

### 第 4 阶段：外部方法和 HumanMM

优先接入 HumanMM/ms-Motion；再接 Multishot/GVHMR/PromptHMR/HSfM†。如果某方法无法公平运行，放 literature-only 表，并写明原因。

### 第 5 阶段：效率与回归

跑 3DPW/EMDB/RICH/TUM-D/Bonn，统计 FPS、CPU/GPU、内存、detector latency 和未来帧约束，确认跨 shot 模块没有损害 Human3R 原有能力。

## 10. ICLR 最终 claim 的最低成立条件

在正式投稿前，至少要达到：

- 在 P0 的同一协议下，Full Movie3R 稳定优于 strict Human3R、no-V9 和 B0-only；
- 在 camera、world human、ID、seam/Accel 至少三类指标上相对最强可复现 baseline 有一致提升，而不是只改善一个指标；
- AvatarReX 低纹理和多人高纹理两类 failure mode 都有统计结果，不能只展示多人；
- detector、gate、B0、BRTC、C1 每个模块都有消融和拒绝/伤害审计；
- 至少一个公开 multi-shot 方法（优先 Multi-THuMBS 或 HumanMM）在完全明确的 protocol 下可复现比较；
- 结果表同时报告 online cost、extra model、future-frame usage；
- 所有未解决的问题（严重遮挡、新人进入/退出、官方 Multi-THuMBS split 不可得）明确写入 limitation。

## 11. 参考材料

- [Multi-THuMBS PDF](/data/wangzheng/iJCV-CODE/paper/Multi-THuMBS.pdf)
- [Human3R 原论文 PDF](/data/wangzheng/iJCV-CODE/paper/Human3R-ori.pdf)
- [HumanMM PDF](/data/wangzheng/iJCV-CODE/paper/HumanMM.pdf)
- [CUT3R PDF](/data/wangzheng/iJCV-CODE/paper/CUT3R.pdf)
- [TTT3R PDF](/data/wangzheng/iJCV-CODE/paper/TTT3R.pdf)
- [v14 Multi-THuMBS 审计](/data/wangzheng/iJCV-CODE/Movie3R/versions/v14/docs/V14_MULTITHUMBS_AUDIT_AND_EGOHUMANS_BASELINE_20260731.md)
- [v14 Multi-THuMBS 数据重合审计](/data/wangzheng/iJCV-CODE/Movie3R/versions/v14/docs/V14_MULTITHUMBS_PUBLIC_PROTOCOL_DATA_OVERLAP_AUDIT_20260801.md)
- [v15 冻结运行规格](/data/wangzheng/iJCV-CODE/Movie3R/versions/v15/FINAL_RUNTIME_SPEC.json)
