# Shot3R ICLR 2027 第一版重构总结

## 1. 本轮目标与边界

本轮工作的目标是把此前从标题、摘要、相关工作、方法、实验到结论的讨论整理为一套连贯的论文方案，并将其写入一个独立、可编译的 ICLR 初稿版本。

执行边界如下：

- 不修改 v031、v032 或更早版本；
- 不重新运行方法，不改变模型结构、推理流程、数据集协议或评价指标；
- 不覆盖、删改或重新挑选已经确定的实验结果；
- 只调整论文定位、叙事结构、术语、公式组织、图表布局和中英文表达；
- 论文中面向读者的方法名统一为 `Shot3R`，内部实验路径和证据文件仍可保留历史 `bridge3r` 标识以维持可追溯性；
- 主文英文正常参与编译，对应中文以 `% 中文：...` 注释保存在 LaTeX 源码中，导入 Overleaf 后可见，但不会显示在 PDF 中。

本轮独立版本：

`versions/v033_20260905_shot3r_bilingual_first_draft`

论文入口：

`versions/v033_20260905_shot3r_bilingual_first_draft/manuscript/main.tex`

## 2. 从旧稿到 v033 的整体变化

| 部分 | 旧稿的主要问题 | v033 初稿方案 |
|---|---|---|
| 标题与名称 | `BRIDGE3R` 和 `Read--Reset--Register` 信息不足，任务范围不直观 | 改为 `Shot3R`，标题同时说明流式、多人体、4D、多镜头输入和核心解耦思想 |
| 摘要 | 以三个工程步骤组织，缺少“在线重建为何需要状态、镜头切换为何破坏状态”的完整动机 | 采用“任务困难—现有缺口—核心思想—方法流程—总体结果”的短摘要 |
| Introduction | 容易被读成基于 Human3R 的局部改进 | 从在线重建与离线全局优化的差别出发，把问题表述为有状态流式重建中的一般冲突 |
| Related Work | 四部分互相重合，早期工作和多人匹配的主次不清 | 合并为世界坐标人体重建、有状态在线重建、多镜头人体重建三部分 |
| Method | 标题和术语偏工程化，模块堆叠感强，旧公式与新故事不完全一致 | 以“对齐信息与传播状态具有不同生命周期”为统一原则，保留实际流程与公式 |
| Experiments | 协议、数据集、指标和实现分散；表格信息密度不均；缺少 OnlineHMR | 合并为 4.1--4.3，加入 OnlineHMR，重排主表、视角表、机制表和边界分析表 |
| Discussion | 与实验解释、协议和局限重复 | 删除独立 Discussion，把有用内容归入 4.1--4.3 |
| Conclusion | 过短且沿用 `gauge`、`Read--Reset--Register` 等旧术语 | 改为“工作总结 + 高层次局限”两段，实验后直接结束主文 |
| 图与 caption | 方法图不完全对应新叙事，图注过长，旧名称残留 | 重画方法图；保持定性结果像素和实验图数据不变；缩短 caption 并统一 Shot3R/Human3R |
| 双语源码 | 英文正文缺少逐段中文对照 | 所有新写主文段落、标题、主文图注和表注均加入中文 LaTeX 注释 |

## 3. 标题与方法名

### 3.1 最终英文标题

> **Shot3R: Streaming Multi-Person 4D Reconstruction from Multi-Shot Videos by Decoupling Alignment and Recurrent State Propagation**

### 3.2 对应中文标题

> **Shot3R：基于对齐与递归状态传播解耦的多镜头视频流式多人体四维重建**

### 3.3 标题表达的逻辑

标题的主干是“以对齐与递归状态传播解耦为方法，实现多镜头视频中的流式多人体四维重建”。

- `Streaming` 表明输入按时间到达，不等待完整序列，也不依赖逐视频全局优化；
- `Multi-Person 4D Reconstruction` 明确任务对象是多个随时间运动的人体；
- `from Multi-Shot Videos` 描述输入包含依次出现的镜头片段；
- `by Decoupling Alignment and Recurrent State Propagation` 概括方法原则；
- 摘要和正文进一步明确输入是单目视频流，不是同步多视角重建；
- `3R` 不再强制展开成三个以 R 开头的操作，而是延续 3D reconstruction 系列命名。

以下名称和表达不再作为面向读者的正式叙事：`BRIDGE3R`、`Read--Reset--Register`、`gauge`、`coarse-to-fine`、`Strict Human3R`。

## 4. 摘要

### 4.1 v033 当前采用：概括结果版

英文：

> Reconstructing human motion in a common world coordinate system from a monocular video with shot transitions is challenging, as abrupt cuts can cause drastic changes in viewpoint, visible scene content, and human appearance. Existing reconstruction methods are typically designed for continuous videos or full-sequence processing, and often struggle to preserve spatial and identity continuity across shots. To fill this gap, we present Shot3R, an online feed-forward framework for streaming multi-person 4D reconstruction from multi-shot videos. Shot3R decouples inter-shot alignment from recurrent state propagation. At each shot boundary, a correction-token branch reads the previous state together with the first new-shot frame to estimate inter-shot alignment, while an independently reinitialized branch provides the only state propagated through the new shot. Prediction-based person association and a shared camera--human transform then register new-shot predictions to the preceding world frame, without accessing future frames or performing per-video global optimization. Extensive experiments on several multi-person datasets demonstrate that Shot3R achieves strong reconstruction accuracy and cross-shot consistency while retaining efficient streaming inference.

中文：

> 从包含镜头切换的单目视频中重建统一世界坐标系下的人体运动是一项具有挑战性的任务，因为突然发生的镜头切换会带来大幅度的视角变化，并显著改变画面中可见的场景内容和人体外观。现有重建方法通常面向连续视频或完整序列处理而设计，因此往往难以在不同镜头之间维持空间连续性和人物身份连续性。为弥补这一空缺，我们提出 Shot3R，一个面向多镜头视频流式多人体四维重建的在线前馈框架。Shot3R 将镜头间对齐与递归状态传播解耦。在每个镜头边界，校正 token 分支联合读取先前状态和新镜头的第一帧，以估计镜头间对齐关系；与此同时，一个独立重置的分支生成在新镜头中唯一继续传播的状态。随后，仅依赖预测结果的人物关联与相机—人体共享变换，将新镜头的预测结果配准到此前的世界坐标系中，整个过程既不访问未来帧，也不需要针对每段视频执行全局优化。多个多人体数据集上的大量实验表明，Shot3R 在保持高效流式推理的同时，实现了出色的重建精度与跨镜头一致性。

### 4.2 保留候选：具体指标与效率版

英文：

> Reconstructing human motion in a common world coordinate system from a monocular video with shot transitions is challenging, as abrupt cuts can cause drastic changes in viewpoint, visible scene content, and human appearance. Existing reconstruction methods are typically designed for continuous videos or full-sequence processing, and often struggle to preserve spatial and identity continuity across shots. To fill this gap, we present Shot3R, an online feed-forward framework for streaming multi-person 4D reconstruction from multi-shot videos. Shot3R decouples inter-shot alignment from recurrent state propagation. At each shot boundary, a correction-token branch reads the previous state together with the first new-shot frame to estimate inter-shot alignment, while an independently reinitialized branch provides the only state propagated through the new shot. Prediction-based person association and a shared camera--human transform then register new-shot predictions to the preceding world frame, without accessing future frames or performing per-video global optimization. Extensive experiments on several multi-person datasets demonstrate strong reconstruction accuracy and cross-shot consistency. In particular, on the predefined extreme-viewpoint subset of EgoBody, Shot3R reduces W-MPJPE by 29.9% (from 505.1 to 354.3 mm) and increases IDF1 from 0.758 to 0.983 over continuous-state recurrence, demonstrating particularly strong gains under large viewpoint changes. Shot3R further runs at [X] FPS with only [Y]% end-to-end runtime overhead over continuous-state recurrence and achieves a [Z]× speedup over [offline method] under the same evaluation setup.

中文：

> 从包含镜头切换的单目视频中重建统一世界坐标系下的人体运动是一项具有挑战性的任务，因为突然发生的镜头切换会带来大幅度的视角变化，并显著改变画面中可见的场景内容和人体外观。现有重建方法通常面向连续视频或完整序列处理而设计，因此往往难以在不同镜头之间维持空间连续性和人物身份连续性。为弥补这一空缺，我们提出 Shot3R，一个面向多镜头视频流式多人体四维重建的在线前馈框架。Shot3R 将镜头间对齐与递归状态传播解耦。在每个镜头边界，校正 token 分支联合读取先前状态和新镜头的第一帧，以估计镜头间对齐关系；与此同时，一个独立重置的分支生成在新镜头中唯一继续传播的状态。随后，仅依赖预测结果的人物关联与相机—人体共享变换，将新镜头的预测结果配准到此前的世界坐标系中，整个过程既不访问未来帧，也不需要针对每段视频执行全局优化。多个多人体数据集上的大量实验表明，Shot3R 取得了出色的重建精度与跨镜头一致性。特别是在 EgoBody 预先定义的极端视角子集上，相比连续状态递归方法，Shot3R 将 W-MPJPE 从 505.1 mm 降低至 354.3 mm，降幅达到 29.9%，并将 IDF1 从 0.758 提升至 0.983，体现出其在大幅视角变化下尤为显著的优势。此外，Shot3R 的运行速度达到 [X] FPS，相比连续状态递归方法仅增加 [Y]% 的端到端运行开销；在相同评测设置下，其速度是离线方法 [offline method] 的 [Z] 倍。

候选版暂未写入论文。当前 3.106 FPS 是排除了镜头检测器、RGB 读取和缩放的受控重建计时，不能直接填入端到端占位符。只有完成同硬件、同输入、同计时范围的端到端实验后，才决定是否用本版替换概括结果版。

## 5. Introduction 的统一故事

Introduction 现在按以下顺序展开：

1. 世界坐标系人体运动重建的重要性；
2. 完整序列处理和逐视频全局优化能够利用全局信息，但必须等待完整输入且计算开销较大；
3. 在线重建随图像到达逐帧处理，因此自然使用递归状态以有限内存持续积累历史；
4. 镜头切换会同时改变视角、可见场景和人体外观，使原本有益的状态产生冲突；
5. 延续旧状态会把旧镜头视觉信息带入新镜头，直接重置又会丢失世界坐标参照和人物身份；
6. 核心观察是历史信息在边界处承担两种生命周期不同的作用：临时对齐证据与长期递归记忆；
7. Shot3R 通过历史条件对齐路径、独立重置的传播路径、预测人物关联和共享相机—人体变换实现该原则；
8. 最后用总体结果、极端视角结果和三项贡献闭合叙事。

Human3R 不在摘要和 Introduction 中被写成“我们基于它做了补丁”。它只在 Related Work、Method 的基础重建器说明和实验受控基线中透明出现。

## 6. Related Work

Related Work 从四部分合并为三部分：

### 6.1 World-Coordinate Human Reconstruction from Monocular Video

中文：单目视频中的世界坐标系人体重建。

主旨：从 SLAHMR、TRAM、WHAM、GVHMR 到 OnlineHMR，说明全局人体与相机运动恢复的发展，并指出它们主要面向时间连续输入。

### 6.2 Stateful Online Reconstruction

中文：有状态在线重建。

主旨：用 CUT3R、Human3R、UniCon3R、GUSH3R、TTT3R、TTSA3R 和 ReCal3R 说明在线递归状态如何积累历史，再引出镜头切换造成的状态不兼容。在线本身不是 Shot3R 首创，本文贡献在镜头边界处的信息生命周期设计。

### 6.3 Multi-Shot Human Reconstruction

中文：多镜头人体重建。

主旨：明确 multi-shot 是时间上依次出现的单目镜头，而不是同步 multi-view；介绍 HumanMM、Multi-THuMBS 和 ShowMak3r，并把跨镜头多人身份关联归入本部分。区别句不笼统宣称这些方法“没有解决镜头边界”，而是准确说明其重点不在流式递归状态转换。

## 7. Method

方法大标题使用最直接的 `Method`，不再使用偏工程化的系统名。当前结构为：

1. `3.1 Overview`：定义输入、输出、递归状态和镜头边界处的核心冲突；
2. `3.2 Stateful Online Human Reconstruction`：给出基础在线映射以及相机、场景和多人体输出；
3. `3.3 Decoupling Alignment from State Propagation`：给出双路径和明确的信息流约束；
4. `3.4 Camera--Human Consistency across Shots`：说明人物对应、相机—人体共享变换与身份延续；
5. `3.5 Learning Alignment at Shot Transitions`：说明同步相机训练构造、边界损失和可训练参数。

方法的理论主线不是三个工程步骤，而是：

> 镜头边界处，用于估计空间关系的历史信息与负责后续重建的递归状态，应具有不同的信息生命周期。

对应实现保持不变：

- 同一个新镜头首帧沿历史条件路径和重置路径处理；
- 历史条件路径产生的状态只用于边界对齐，随后丢弃；
- 只有重置路径产生的状态进入新镜头后续帧；
- 镜头间相机变换由两条路径的相机预测建立；
- 基于预测的骨盆位置、躯干朝向和根中心关节结构完成人物匹配；
- 人物引导平移和相机变换共同作用于相机、关节与网格；
- 不修改已输出历史结果，不访问未来帧，也不进行逐视频全局优化。

现有公式按照这条信息流重新组织，但公式含义、实际模块、损失和参数量均未改变。完整方法在主文第 6 页内结束。

## 8. Experiments

实验合并为三个小节：

### 8.1 Experimental Setup

按 `Baselines`、`Datasets and metrics`、`Implementation details` 三段组织。

- Human3R 是相同输入和评测协议下的主要受控基线；
- OnlineHMR 是最接近任务的公开在线参考，但明确说明其半在线跟踪和独立全局相机后端带来不同时间访问；
- TRACE 与 PromptHMR-SPEC 作为同 RGB 输入上的公开可执行参考；
- 明确三套协议在推理时均是依次到达的单目 RGB，不是同步多视角输入；
- W、WA、Camera、IDF1 和 Coverage 的含义以及条件指标的有效支持被集中说明；
- 主文只保留关键训练信息，完整损失和参数配置留在补充材料。

HumanMM 与 Multi-THuMBS 未直接数值比较的英文说明：

> HumanMM and Multi-THuMBS address closely related multi-shot settings, but at the time of our evaluation, their executable implementations and complete evaluation configurations required for reproduction on our frozen inputs were not publicly available. We therefore discuss them as related task and protocol context, rather than mixing source-paper results obtained under different data and protocols with our matched-input comparisons.

对应中文：

> HumanMM 和 Multi-THuMBS 研究了与本文密切相关的多镜头设置，但截至本文评测时，其可执行实现以及在本文冻结输入上复现所需的完整评测配置尚未公开。因此，我们仅将它们作为相关任务和协议背景进行讨论，而不把它们在不同数据和协议下报告的原文结果混入本文的同输入比较。

### 8.2 Multi-Shot Multi-Person Reconstruction

按 `Overall results`、`Large viewpoint changes`、`Comparison with executable methods` 组织。

- 先总结三个数据集上的人体、相机和身份结果；
- 明确保留 EgoHumans Coverage 轻微下降、EgoHumans WA 区间跨零和 Harmony4D W 回退；
- 报告三个数据集等权汇总和预定义极端/最远视角汇总；
- 使用 EgoBody 极端视角的 29.9% W 降幅与 IDF1 0.758→0.983 作为最直观例子；
- 不把外部方法的条件误差与内部配对结果混成统一 SOTA 排名。

### 8.3 Analysis and Ablation Studies

按 `Alignment and state propagation`、`Multi-person association`、`Shot-transition detection and efficiency` 组织。

- 核心机制表分离持续传播、只重置、相机对齐加重置和完整 Shot3R；
- 真实边界行明确标为诊断控制，不冒充完整在线变体；
- 人物关联同时报告已接受匹配的准确率和全部跨镜头身份的延续率；
- 镜头检测报告部署实际使用的首次触发，而不是只报告真实边界上的响应；
- 运行时间只称为受控重建开销，不扩大为端到端速度。

## 9. Discussion 与 Conclusion

独立 Discussion 已删除。原有内容分别并入实验设置、结果解释、机制分析和补充材料，避免实验后再次重复。

当前 Conclusion 英文：

> We presented Shot3R, an online framework for streaming multi-person 4D reconstruction from monocular video with shot transitions. Its key idea is to decouple inter-shot alignment from recurrent state propagation: history is used only to establish the spatial relation between shots, while an independently reinitialized state handles recurrence in the new shot and all subsequent frames. Prediction-based person association and a shared camera--human transform further preserve coordinate and identity consistency across shots. Experiments on several multi-person datasets show that Shot3R improves world-coordinate human reconstruction, camera trajectories, and identity continuity overall, with more pronounced gains under large viewpoint changes, while retaining sequential streaming inference without per-video global optimization.
>
> **Limitations and future work.** Maintaining cross-shot spatial and identity consistency under complex edits, repeated entrances and exits, and rapid visual changes remains an important open problem for streaming multi-person 4D reconstruction.

对应中文：

> 本文提出 Shot3R，一个从包含镜头切换的单目视频中进行流式多人体四维重建的在线框架。其核心思想是将镜头间对齐与递归状态传播解耦：历史状态仅用于建立新旧镜头之间的空间关系，而独立重置的状态负责新镜头及后续帧的递归推理；在此基础上，仅依赖预测结果的人物关联与相机—人体共享变换进一步维持跨镜头的坐标和身份一致性。多个多人体数据集上的实验表明，Shot3R 总体改善了世界坐标系下的人体重建、相机轨迹和人物身份连续性，并在大幅视角变化下表现出更明显的优势，同时保留了按时间顺序逐帧处理且无需逐视频全局优化的流式推理方式。
>
> **局限与未来工作。** 如何在复杂剪辑、人物反复进出画面以及快速画面变化下持续保持跨镜头的空间与身份一致性，仍是流式多人体四维重建中值得进一步研究的问题。

## 10. 主文图

### Figure 1：大视角定性示例

- 使用 EgoHumans 中相机旋转跨度为 176.7° 的固定样本；
- 保留原始 RGB、Human3R、TRACE、PromptHMR-SPEC 和 Shot3R 的已有重建结果，不重新生成或修改实验内容；
- 只统一方法显示名称、图内术语和短 caption；
- 明确 TRACE 与 PromptHMR 通过公开接口只提供人体网格输出，避免暗示其具有相同相机输出。

英文 caption：

> An EgoHumans shot transition with a 176.7° camera rotation. Shot3R preserves coherent camera placement and person identities under the large viewpoint change; TRACE and PromptHMR provide mesh-only outputs through their released interfaces.

中文：

> EgoHumans 上相机旋转达到 176.7° 的镜头切换示例。Shot3R 在大幅视角变化下保持一致的相机位置与人物身份；TRACE 和 PromptHMR 通过其公开接口仅提供人体网格输出。

### Figure 2：方法图

- 重新绘制为从左到右的 ICLR 风格信息流；
- 左侧输入是依次到达的单目视频和在线镜头检测；
- 中间突出“Alignment--state decoupling”；
- 上分支是只在边界临时使用的历史条件对齐路径；
- 下分支是从初始状态开始并持续传播的重建路径；
- 右侧通过相机对齐、人物对应和共享世界变换输出对齐后的相机、人体和身份；
- 图底部明确区分“temporary alignment information”和“propagated recurrent state”。

英文 caption：

> Overview of Shot3R. A history-conditioned path estimates cross-shot alignment, while a reinitialized path alone propagates recurrent state. Prediction-based person association and a shared camera--human transform place new-shot outputs in the existing world frame.

中文：

> Shot3R 方法概览。历史条件路径估计镜头间对齐，只有重新初始化的路径传播递归状态。基于预测的人物关联与共享相机—人体变换将新镜头输出置于已有世界坐标系中。

### Figure 3：视角变化分层

- 使用原 90 个 EgoHumans 固定样本重新渲染；
- 数据、均值和 bootstrap 区间不变；
- 图例统一为 Human3R 与 Shot3R；
- caption 只说明视角分层和 95% 区间，统计定义留在正文与补充材料。

英文 caption：

> Reconstruction across EgoHumans viewpoint-change strata. Curves show case means with 95% capture-cluster bootstrap intervals; the largest gains occur in the two widest camera-rotation strata.

中文：

> EgoHumans 不同视角变化区间下的重建结果。曲线表示样本均值及以采集序列为聚类单位的 95% bootstrap 区间；最大的两个相机旋转区间取得了最明显的改进。

补充材料中的既有实验图和数据证据保持不变，只清理会直接显示给读者的旧方法名。内部 provenance 文件名和路径不机械改名，以免破坏来源追踪。

## 11. 主文表

### Table 1：主结果

英文 caption：

> Multi-shot multi-person reconstruction on three benchmarks with same-input executable references.

中文：

> 三个基准上的多镜头多人体重建结果及同输入可执行方法参考。

布局变化：Human3R--Shot3R 配对结果与外部可执行参考分区；保留 W、WA、Camera、IDF1 和 Coverage；外部条件指标标注有效样本数，不与内部配对结果共同加粗排名。

### Table 2：大幅视角变化

英文 caption：

> Gains over Human3R for all views and predefined extreme/farthest-view subsets.

中文：

> 全部视角及预定义极端/最远视角子集上相对于 Human3R 的改进。

布局变化：只保留直接支撑核心结论的 W 降幅、WA 降幅和 IDF1 提升。

### Table 3：机制分析

英文 caption：

> Effects of inter-shot alignment and recurrent-state propagation on EgoBody. Annotated-boundary rows are diagnostic controls.

中文：

> EgoBody 上镜头间对齐与递归状态传播的作用。使用真实边界的行仅作为诊断控制。

布局变化：删除四行完全相同的 Coverage；用直观名称替代 `clean reset`、`read-only gauge` 等内部术语。

### Table 4：边界分析与效率

英文 caption：

> Boundary association, shot-transition detection, and controlled reconstruction cost.

中文：

> 镜头边界处的人物关联、镜头切换检测与受控重建开销。

布局变化：三个并列面板分别呈现人物关联、首次触发检测和受控重建开销；各面板已按栏宽安全缩放，不再发生横向覆盖。

## 12. 术语与写作规范

- 正式方法名：`Shot3R`；
- 基线名：`Human3R`，不再使用 `Strict Human3R`；
- 输入首次出现时明确为含镜头切换的 `monocular video stream`；
- 用 `multi-shot` 描述输入，用 `cross-shot` 修饰跨镜头关系、对齐或一致性；
- 用按时间顺序处理、只读取当前与历史 RGB、不访问未来帧等直观表述解释在线约束；
- 不使用未解释的 `gauge`，不使用工程化的 `coarse-to-fine`；
- 不把 Shot3R 写成 Human3R 的插件、后处理或局部补丁；
- 不声称已验证 backbone-agnostic；
- 不把不同输入、输出和时间访问协议下的结果组成无条件统一 SOTA 排名；
- 受控重建计时不写成端到端运行时间。

## 13. 排版与编译状态

v033 当前排版结果：

- 主科学正文共 9 页；
- Method 在第 6 页内结束；
- Conclusion 在第 9 页结束；
- AI Use、Ethics 和 Reproducibility statements 从第 10 页开始；
- References 从第 11 页开始；
- 补充材料继续与主文使用同一个编译入口；
- 主文四张表已消除横向越界；
- 未发现未定义引用、未定义文献或致命编译错误；
- 当前使用 Tectonic/XeTeX 完整编译通过；提交前仍建议在 Overleaf 的 pdfLaTeX 环境做最终验证。

## 14. 后续仍需作者确认或补充的事项

1. 摘要最终选择概括结果版，还是在完成端到端时间实验后改用具体指标与效率版；
2. 投稿前重新核验 HumanMM 和 Multi-THuMBS 的官方代码发布状态；
3. 在 Overleaf/pdfLaTeX 中检查字体、浮动体位置和最终会议模板页限；
4. 后续逐段精修时继续沿用本版核心故事，不重新引入旧三步式或 gauge 叙事；
5. 任何新增结果都应新建后续版本，不回写或覆盖 v033 及更早实验版本。
