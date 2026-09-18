# Shot3R ICLR 2027 论文修改意见与定稿决策

## 1. 文档用途

本文档用于逐项记录基于当前 ICLR 论文进行的写作修改，并将每一项“需要修改的问题”与“作者最终确认的方案”一一对应。后续关于标题、摘要、引言、相关工作、方法表述、实验呈现、图表、结论和补充材料的讨论，均按相同格式追加到本文档。

本轮改写的历史基线为：

`ICLR-paper/bridge3r_iclr2027/versions/v032_20260904_onlinehmr_formal`

已将确认方案写入独立初稿：

`ICLR-paper/bridge3r_iclr2027/versions/v033_20260905_shot3r_bilingual_first_draft`

摘要与训练细节的后续确认版为：

`ICLR-paper/bridge3r_iclr2027/versions/v035_20260905_metric_abstract_and_training_details`

最终模型训练与曲线修正版为：

`ICLR-paper/bridge3r_iclr2027/versions/v036_20260906_final_training_curve`

本轮工作的基本边界：

- 不改变已经确定的方法和实际执行流程；
- 不修改、覆盖或重新解释既有实验结果；
- 重点完善论文的定位、标题、叙事、术语和逐段写法；
- 只有经过作者明确确认的方案才标记为“已定稿”；
- 尚未讨论或尚未确认的内容不提前写成最终决定。
- 本文档以中文记录修改对象、存在问题、修改原则、执行方案和检查事项，不为尚未正式写入论文的内容提前生成英文版本。
- 只有必须预先锁定并原样写入论文的英文内容，例如正式标题、指定句子、关键定义或固定术语，才同时记录英文定稿及其对应中文翻译；不得只给英文而不提供中文。
- 文件路径、LaTeX 命令、方法名、数据集名、指标缩写和代码标识符可保留原文，不要求逐字翻译。

## 2. 修改决策总表

| 编号 | 修改对象 | 当前版本 | 最终方案 | 状态 | 是否已写入论文源文件 |
|---|---|---|---|---|---|
| R01 | 论文标题与方法名称 | `BRIDGE3R: Read--Reset--Register for Causal Multi-Shot Reconstruction`<br>BRIDGE3R：面向因果多镜头重建的读取—重置—配准 | `Shot3R: Streaming Multi-Person 4D Reconstruction from Multi-Shot Videos by Decoupling Alignment and Recurrent State Propagation`<br>Shot3R：基于对齐与递归状态传播解耦的多镜头视频流式多人体四维重建 | 已定稿 | 是（v033） |
| R02 | 未开源相关方法的比较说明 | 仅笼统说明 HumanMM 和 Multi-THuMBS 缺少共享实现与评测器 | 用一至两句话明确说明其可执行实现和足以进行同输入复现的完整评测配置在评测时未公开，因此只作相关工作与协议背景，不混入同输入数值表 | 已定稿 | 是（v033） |
| R03 | Human3R 基线命名 | 在正文和表格中反复使用 `Strict Human3R`（严格版 Human3R） | 所有面向读者的基线名称统一恢复为 `Human3R`；用实验协议句说明相同输入与评测条件，不再另造 `Strict Human3R` 名称 | 已定稿 | 是（v033） |
| R04 | 摘要及全文核心叙事 | 从递归模型在同场景视角切换处的脆弱性直接起笔，并以 `Read--Reset--Register` 组织方法，缺少“为什么首先需要在线递归重建”的完整动机 | 从包含镜头切换的单目视频重建统一世界坐标系下的人体运动起笔；将问题归结为镜头间对齐与递归状态传播的冲突；v035 选用具体指标版，报告 EgoBody 极端视角的 W-MPJPE 与 IDF1，未添加尚未完成的端到端效率数字 | 已定稿 | 是（v035） |
| R05 | Related Work 的主题划分与段落写法 | 当前分为四部分，内容存在重合；状态、联合重建、多镜头和多人关联之间的主次关系不够清楚 | 合并为“单目视频中的世界坐标系人体重建”“有状态在线重建”“多镜头人体重建”三部分；采用“任务发展—代表方法—共同局限—本文区别”的写法，多人身份关联并入多镜头人体重建 | 已定稿 | 是（v033） |
| R06 | Experiments 的结构、基线与图表呈现 | 当前四部分存在重合；对比方法与实现细节不够集中；标题、术语和图注偏工程化；部分内容按内部证据文件而非研究问题组织；早期版本尚未纳入 OnlineHMR | 按 4.1 Experimental Setup、4.2 Multi-Shot Multi-Person Reconstruction、4.3 Analysis and Ablation Studies 三部分组织；加入 OnlineHMR；主表围绕人体、相机、身份和可用性呈现，并按信息价值精简 Coverage；分析围绕对齐与状态传播、人物关联、镜头检测和效率展开 | 初稿方案已落实 | 是（v033） |
| R07 | Discussion 与 Conclusion | 当前在 Experiments 与 Conclusion 之间设置独立 Discussion，主要重复实验解释、协议说明与局限；Conclusion 过短且沿用多项旧术语 | 删除独立 Discussion，将实验解释归入 4.1--4.3；实验后直接进入两段式 Conclusion，第一段总结方法与主要发现，第二段仅从复杂剪辑、人物反复进出画面和快速画面变化概括未来问题 | 初稿方案已落实 | 是（v033） |
| R08 | 最终模型的训练目标、实现细节与收敛曲线 | v035 将一次使用重复数据的 6 轮操作误写为有效训练阶段，并据此报告五个数据源、2,880 步和八项 event-local loss | 不报告该 6 轮操作；将 `v9_mixed_60h_pose_human_lora_bs10` 的 72 轮训练直接作为 Shot3R 最终训练，恢复其真实关系校正目标、AvatarReX/THuman 数据组成、7,200 步优化设置与 19.62M 可训练参数；从保留日志恢复训练/验证曲线并放入补充材料 | 已定稿 | 是（v036） |
| R09 | 补充材料与实验表格重构 | 补充材料包含较多重复、全常数或内部审计式表格，章节顺序和正文表格的信息层级不够清楚 | 将补充材料重组为 A--E 五部分并精简为 24 张有效表格；优化正文 Table 1/Table 3；参考 Human3R 增加只列附录 A--E 与 D.1--D.7、带自动页码的独立目录 | 初稿方案已落实 | 是（v037） |
| R10 | 运行效率数字及表述边界 | 正文已有单段视频上的受控重建 FPS，但容易被误解为包含读图与镜头检测的完整端到端速度，其他方法的历史 wall-clock 记录也不具备统一协议 | 保留同硬件受控结果：Human3R 3.224 FPS、Shot3R 无切换 3.213 FPS、单切换 3.106 摊销输出 FPS；只声称单次切换相对无切换路径增加 3.42% 重建开销，不声称 real-time 或相对离线方法的统一加速倍数；后续另做包含检测和预处理的多样例端到端测速 | 已确认，后续实验待完成 | 是（v042 保留现有数值并补齐原始报告） |
| R11 | 核心贡献的包装 | 第一条使用 `general principle` 和 `information lifetimes`，表述抽象且超出当前单一递归框架的直接验证范围；前两条重复，第三条偏实验配置清单 | 将贡献组织为“状态冲突—Shot3R 实现—实验证据”：明确旧状态只用于对齐而不进入新镜头递归，随后说明双路径、预测人物关联与共享变换，最后报告三数据集、大视角和 3.4% 受控切换开销 | 已定稿 | 是（v043） |
| R12 | 方法术语统一 | 同一可学习组件混用 `correction-token branch`、`alignment representation`、`relation tokens` 和 `relation-correction objective`，容易被误解为多个独立模块 | 路径统一为 `history-conditioned alignment path`（历史条件对齐路径），可学习组件统一为 `shot-alignment module`（镜头对齐模块），只在内部结构中使用三个 `alignment tokens`（对齐 token）；目标记号同步为 alignment objective | 已定稿 | 是（v044） |

## 3. 逐项修改意见与最终方案

### R01. 论文标题与方法名称

#### 3.1 当前写法

英文标题：

> **BRIDGE3R: Read--Reset--Register for Causal Multi-Shot Reconstruction**

对应中文：

> **BRIDGE3R：面向因果多镜头重建的读取—重置—配准**

当前方法显示名称为 `BRIDGE3R`。

#### 3.2 需要修改的问题

1. `BRIDGE3R` 的含义过于宽泛，不能让读者立即判断论文关注流式、多镜头和多人四维重建。
2. `Read--Reset--Register` 虽然对应实际流程，但标题没有说明读取什么、重置什么、注册什么，必须阅读正文后才能理解。
3. 当前标题只写 `Causal Multi-Shot Reconstruction`，没有明确表达 `streaming`、`multi-person` 和 `4D reconstruction` 三项任务属性。
4. 标题不需要强调 `same-scene`。它是当前任务成立的前提和正文中的适用范围，不是标题中最需要突出的方法贡献。
5. 不采用 `Movie3R`。目前没有真实影视数据上的正式实验，使用该名称可能让读者期待电影或影视分镜数据验证。
6. 不要求把 `3R` 展开为三个以 R 开头的操作；此处的 `3R` 延续 3D reconstruction 方法系列的命名方式。
7. 不在标题中使用 `gauge`，避免引入生僻术语以及 SE(3)、Sim(3) 或 translation gauge 范围上的额外歧义。
8. 不使用 `coarse-to-fine`，避免把核心贡献表述成偏工程化的阶段式流水线。
9. 标题需要同时回答两件事：论文解决什么任务，以及采用什么核心思想。任务由 `Streaming Multi-Person 4D Reconstruction from Multi-Shot Videos`（从多镜头视频进行流式多人体四维重建）表达；方法由 `Decoupling Alignment and Recurrent State Propagation`（解耦对齐与递归状态传播）表达。

#### 3.3 作者最终确认方案

英文标题：

> **Shot3R: Streaming Multi-Person 4D Reconstruction from Multi-Shot Videos by Decoupling Alignment and Recurrent State Propagation**

中文标题：

> **Shot3R：基于对齐与递归状态传播解耦的多镜头视频流式多人体四维重建**

#### 3.4 标题的准确含义

标题中的逻辑主干是“以对齐与递归状态传播解耦为方法，实现多镜头视频的流式多人体四维重建”，而不是把“对齐与状态传播解耦”本身作为最终任务。

- `Streaming`：强调按输入流在线处理，而不是读取完整视频后进行离线全局优化。
- `Multi-Person 4D Reconstruction`：明确输出涉及多个人体及其随时间变化的世界空间重建。
- `from Multi-Shot Videos`：明确输入由包含镜头切换的多镜头视频构成。
- `by Decoupling Alignment and Recurrent State Propagation`：概括核心方法思想，即历史状态可以为跨镜头对齐提供信息，但新镜头从干净状态开始，且只有该状态参与后续递归传播。

#### 3.5 落实到论文时的对应修改

后续正式修改论文源文件时，需要同步完成以下一致性修改：

1. 将论文标题替换为最终英文标题。
2. 将正文中的正式方法显示名称由 `BRIDGE3R` 统一为 `Shot3R`。
3. 摘要和 Introduction 第一次解释标题中的 `decoupling` 时，明确写出：历史状态用于估计跨镜头对齐，而干净状态独占新镜头的后续递归传播。
4. 标题和正文中不再将 `Shot3R` 强制展开为 `Read--Reset--Register`；三个操作仍可在方法部分作为实际过程解释，但不再承担方法名称的定义。
5. 不因标题修改而改变现有模型、模块、推理协议、实验数据或数值结果。
6. 项目内部目录、历史版本和证据文件可继续保留原有 `bridge3r` 路径，以保证实验与版本追踪不被破坏；面向读者的论文文本统一使用 `Shot3R`。

#### 3.6 与历史决策的关系

本条决定取代 `REVISION_DECISIONS_V6.md` 中的旧标题决策 D16，以及 v031 中的当前标题。该替换只涉及论文标题、正式方法名称和相应文字一致性，不自动推翻其他已经确认的方法、实验、范围或证据决策。

#### 3.7 状态

**已定稿并写入 v033；历史版本和内部证据路径保持不变。**

### R02. HumanMM 和 Multi-THuMBS 未进入直接数值比较的说明

#### 3.8 当前写法

当前 v031 已在 Related Work 和 Experiments 中笼统说明不同方法的输入结构、时间访问方式和输出接口不同，并写有：

> HumanMM and Multi-THuMBS are compared at the protocol level because a shared implementation and evaluator are unavailable.

对应中文：

> 由于缺少可共同使用的实现和评测器，HumanMM 和 Multi-THuMBS 仅在协议层面进行比较。

这句话方向正确，但没有充分说明为什么不能在本文冻结输入上运行这两个方法，也没有把“代码不可执行”和“原文数值来自不同实验协议”两个原因明确分开。

#### 3.9 需要修改的问题

1. HumanMM 和 Multi-THuMBS 都是与本文高度相关的 multi-shot 工作。若只在 Related Work 中引用而不进入数值表，审稿人可能会追问缺失原因。
2. 不能笼统写成“其实验设置完全未知”。两篇论文都披露了部分方法和实验信息，但缺少足以在本文冻结 RGB 输入和统一 evaluator 下无歧义复现的完整执行链路。
3. HumanMM 的数据集资源已有部分公开，但其 processing/baseline executable、版本化 evaluator 和完整聚合配置在评测时不可用；因此不能表述为“HumanMM 什么都没有公开”。
4. Multi-THuMBS 在核验时没有作者链接的可执行代码，其论文构造数据、修改后的基线和完整评测链路也不足以支持本文的同输入复现。
5. 两篇论文报告的原文数值来自各自构造的数据和协议，不能直接填入本文的 matched-input 表并作为公平排名。
6. 表述必须采用 `at the time of our evaluation`（截至本文评测时），避免把可能随时间变化的代码发布状态写成永久事实。

#### 3.10 作者最终确认方案

论文中用一至两句话说明没有进行直接数值比较。建议采用以下两句英文作为正式写法：

> **HumanMM and Multi-THuMBS address closely related multi-shot settings, but at the time of our evaluation, their executable implementations and complete evaluation configurations required for reproduction on our frozen inputs were not publicly available. We therefore discuss them as related task and protocol context, rather than mixing source-paper results obtained under different data and protocols with our matched-input comparisons.**

对应中文：

> **HumanMM 和 Multi-THuMBS 研究了与本文密切相关的多镜头场景，但截至本文评测时，其可执行实现以及在本文冻结输入上进行复现所需的完整评测配置尚未公开。因此，我们仅将它们作为相关任务和协议背景进行讨论，而不把它们在不同数据和协议下报告的原文结果混入本文的同输入比较。**

#### 3.11 修改后的准确含义

这两句话只说明为什么无法进行公平的直接重运行和数值比较，不表达以下含义：

- 不声称 HumanMM 或 Multi-THuMBS 的方法无效或弱于 Shot3R；
- 不声称两篇论文没有提供任何实验信息；
- 不声称 HumanMM 的数据资源完全没有公开；
- 不把“未公开完整复现链路”偷换成“论文不可复现”的笼统评价；
- 不将不同数据、输入、时间访问方式和 evaluator 下的原文数字组成伪统一排行榜。

#### 3.12 落实到论文时的对应修改

1. 在 Experiments 的 baseline/complementary-method 说明中放置上述两句，直接回答为什么主数值表没有 HumanMM 和 Multi-THuMBS。
2. Related Work 继续说明两项工作的任务相关性和与本文的主要差别，不需要重复完整代码状态说明。
3. Supplement 的 protocol-context 表保留逐方法、带核验日期的具体状态：HumanMM 标明 processing/baseline executable 和完整 evaluator 不可用；Multi-THuMBS 标明未找到作者链接的可执行实现。
4. 最终投稿前再次核验两项工作的官方项目页和作者链接仓库；如代码状态发生变化，必须更新本条措辞及相应比较计划。
5. 该说明不影响现有内部实验、公开可执行 baseline 结果或任何已冻结数值。

#### 3.13 状态

**已定稿并写入 v033；最终投稿前仍需复核代码发布状态。**

### R03. 删除 `Strict Human3R` 这一工程化基线名称

#### 3.14 当前写法

当前 v031 在正文、主结果表、补充表、图注和运行效率表中反复使用 `Strict Human3R`，即“严格版 Human3R”这一名称。该名称原本用于强调基线遵循原始 Human3R 的递归推理流程，没有使用 Shot3R 的边界处理方法。

#### 3.15 需要修改的问题

1. “严格版 Human3R”不是原论文中的正式方法名称，而是项目实验与事实审计阶段使用的内部标签。
2. 学术论文通常默认 `Human3R` 指被引用的原方法；除非确实修改了模型并形成新变体，否则不需要额外增加 `Strict` 前缀。
3. “严格版”容易让读者误以为还存在一个正式的“非严格版 Human3R”，反而增加理解成本。
4. 该标签带有明显的内部实现、审计和工程管理色彩，不适合成为面向审稿人的方法名称。
5. 是否遵循相同输入、推理条件和评测器，应在实验协议中直接说明，而不应编码进方法名。

#### 3.16 作者最终确认方案

所有面向读者的论文内容统一使用原方法的正式名称 `Human3R`，不再使用“严格版 Human3R”这一名称。

为了保留公平比较所需的信息，在实验协议中直接说明：在每个数据集内，Human3R 与 Shot3R 接收顺序相同的 RGB 输入，并采用相同的拓扑转换、匹配规则和聚合协议进行评测。该句目前只记录中文含义；正式修改论文时再给出英文定稿及其对应中文翻译。

#### 3.17 修改后的准确含义

- `Human3R` 直接表示作为比较基线运行的原始方法，不再为其另造论文名称。
- 删除 `Strict` 只改变论文呈现方式，不改变已经运行的 Human3R 推理流程、输入、输出或任何实验数值。
- 公平性由相同的有序 RGB 输入、拓扑转换、匹配规则、聚合协议和评测器保证，而不是由方法名前缀保证。
- Shot3R 的状态重置、对齐、人物关联和共享平移修正仍只属于 Shot3R，不会因为基线改名而被错误归入 Human3R。

#### 3.18 落实到论文时的对应修改

1. 将主文、补充材料、表格、图例和图注中所有作为基线名称使用的“严格版 Human3R”统一改为 `Human3R`。
2. 将“严格版减 Shot3R”之类的统计描述改为“Human3R 减 Shot3R”。
3. 将运行效率表中的“严格版帧率”改为“Human3R 帧率”，或根据表格上下文简写为“基线帧率”。
4. 将“严格比较”改为“配对比较”或“同输入比较”，具体取决于该句强调配对统计还是输入一致性。
5. 将“严格流式片段”改为“Human3R 片段”，或直接按照有效样本条件描述，不保留内部路线名称。
6. 一般意义上的“严格因果约束”若确实描述信息访问规则，可以保留；本条删除的是作为方法别名使用的“严格版 Human3R”，不是机械删除全文所有 `strict`。
7. 内部实验产物文件名、缓存目录、脚本参数和历史报告中的 `strict` 可以保留，以维护实验来源和路径稳定性；但这些内部名称不得出现在最终渲染的论文正文、表格或图中。
8. 该修改不触碰任何实验结果，也不需要重新运行 Human3R 或 Shot3R。

#### 3.19 状态

**已定稿并写入 v033；面向读者的正文、表格和图中均使用 `Human3R`。**

### R04. 摘要及全文核心叙事

#### 3.20 当前写法

当前 v031 摘要直接从递归人体重建在同场景视角切换处的脆弱性起笔，随后以 `Read--Reset--Register` 三个操作逐项介绍方法。该版本准确描述了现有执行流程，但没有先说明为什么在线重建自然需要递归状态，因此镜头切换处的状态问题显得缺少上层动机。同时，`Read--Reset--Register` 占据了摘要的主要叙事位置，容易让论文呈现为围绕三个工程步骤组织的系统，而不是从一个一般性的在线状态冲突出发提出的方法。

#### 3.21 需要修改的问题

1. 摘要首先需要定义任务困难：输入是一条包含镜头切换的单目视频流，而不是同一时刻同时输入多个相机视角。突然的镜头切换会导致视角、可见场景内容和人体外观发生显著变化。
2. 需要建立完整的在线动机。现有离线方法通常依赖完整序列处理或逐视频全局优化；在线方法则需要在帧到达时进行因果、增量式推理，递归状态由此成为自然选择。
3. 核心问题不应写成某个既有模型的局部缺陷，而应表述为在线递归重建中的一般状态冲突：继续传播旧状态会把前一镜头的视角相关信息带入新镜头，直接重置又会丢失维持跨镜头空间与身份连续性所需的历史参照。
4. 核心方法思想统一为“解耦镜头间对齐与递归状态传播”：旧状态只作为估计镜头间对齐的临时证据，独立重置产生的干净状态才是新镜头后续传播的唯一状态。
5. 摘要不提 Human3R。Human3R 在方法部分作为具体递归重建器透明说明，在实验部分作为相同输入和评测协议下的受控基线；摘要首先建立 Shot3R 的独立问题、思想和贡献。
6. 不照搬 GUSH3R 的 `single forward pass`。Shot3R 在镜头边界实际包含只读评估与干净重置评估，写成单次前向传播不符合执行事实；统一使用在线前馈、增量式和因果流式推理描述。
7. 不将任务写成动态人体—场景重建。GUSH3R 明确训练和评测了静态场景高斯解码器，而 Shot3R 当前贡献与正式指标集中在世界坐标系下的人体重建、相机一致性和身份连续性，不提出稠密背景重建改进。
8. `multi-shot` 在标题中继续保留；摘要首次定义输入时使用“包含镜头切换的单目视频”，明确推理时每个时刻只有一个顺序 RGB 观测，避免被理解为同步多视角联合重建。
9. 不使用相对生僻的 `world-grounded human motion` 作为摘要首句。采用 `human motion in a common world coordinate system`，与 HumanMM 常用表述一致且更容易直接理解。
10. 结果句不笼统宣称全面优于在线和离线 SOTA，也不把不同有效支持、输出接口和评测协议下的方法组成统一排名。候选版本 A 采用“多个多人体数据集上的大量实验表明”概括实证范围；候选版本 B 在相同概括之后报告 EgoBody 预定义极端视角子集的突出结果，并为符合可比性要求的端到端时间数据预留位置。

#### 3.22 作者确认保留的两个候选版本

两个版本共享完全相同的问题定义、方法思想和流程表述，仅结果段的具体程度不同。候选版本 A 使用概括性实验结论；候选版本 B 报告表现突出的预定义极端视角子集，并为后续端到端时间实验预留数字。作者将在完成时间评测后确认最终采用哪一个版本。

##### 候选版本 A：概括结果版

英文摘要：

> **Reconstructing human motion in a common world coordinate system from a monocular video with shot transitions is challenging, as abrupt cuts can cause drastic changes in viewpoint, visible scene content, and human appearance. Existing reconstruction methods are typically designed for continuous videos or full-sequence processing, and often struggle to preserve spatial and identity continuity across shots. To fill this gap, we present Shot3R, an online feed-forward framework for streaming multi-person 4D reconstruction from multi-shot videos. Shot3R decouples inter-shot alignment from recurrent state propagation. At each shot boundary, a correction-token branch reads the previous state together with the first new-shot frame to estimate inter-shot alignment, while an independently reinitialized branch provides the only state propagated through the new shot. Prediction-based person association and a shared camera--human transform then register new-shot predictions to the preceding world frame, without accessing future frames or performing per-video global optimization. Extensive experiments on several multi-person datasets demonstrate that Shot3R achieves strong reconstruction accuracy and cross-shot consistency while retaining efficient streaming inference.**

对应中文：

> **从包含镜头切换的单目视频中重建统一世界坐标系下的人体运动是一项具有挑战性的任务，因为突然发生的镜头切换会带来大幅度的视角变化，并显著改变画面中可见的场景内容和人体外观。现有重建方法通常面向连续视频或完整序列处理而设计，因此往往难以在不同镜头之间维持空间连续性和人物身份连续性。为弥补这一空缺，我们提出 Shot3R，一个面向多镜头视频流式多人体四维重建的在线前馈框架。Shot3R 将镜头间对齐与递归状态传播解耦。在每个镜头边界，校正 token 分支联合读取先前状态和新镜头的第一帧，以估计镜头间对齐关系；与此同时，一个独立重置的分支生成在新镜头中唯一继续传播的状态。随后，仅依赖预测结果的人物关联与相机—人体共享变换，将新镜头的预测结果配准到此前的世界坐标系中，整个过程既不访问未来帧，也不需要针对每段视频执行全局优化。多个多人体数据集上的大量实验表明，Shot3R 在保持高效流式推理的同时，实现了出色的重建精度与跨镜头一致性。**

##### 候选版本 B：具体指标与效率版

英文摘要：

> **Reconstructing human motion in a common world coordinate system from a monocular video with shot transitions is challenging, as abrupt cuts can cause drastic changes in viewpoint, visible scene content, and human appearance. Existing reconstruction methods are typically designed for continuous videos or full-sequence processing, and often struggle to preserve spatial and identity continuity across shots. To fill this gap, we present Shot3R, an online feed-forward framework for streaming multi-person 4D reconstruction from multi-shot videos. Shot3R decouples inter-shot alignment from recurrent state propagation. At each shot boundary, a correction-token branch reads the previous state together with the first new-shot frame to estimate inter-shot alignment, while an independently reinitialized branch provides the only state propagated through the new shot. Prediction-based person association and a shared camera--human transform then register new-shot predictions to the preceding world frame, without accessing future frames or performing per-video global optimization. Extensive experiments on several multi-person datasets demonstrate strong reconstruction accuracy and cross-shot consistency. In particular, on the predefined extreme-viewpoint subset of EgoBody, Shot3R reduces W-MPJPE by 29.9% (from 505.1 to 354.3 mm) and increases IDF1 from 0.758 to 0.983 over continuous-state recurrence, demonstrating particularly strong gains under large viewpoint changes. Shot3R further runs at [X] FPS with only [Y]% end-to-end runtime overhead over continuous-state recurrence and achieves a [Z]× speedup over [offline method] under the same evaluation setup.**

对应中文：

> **从包含镜头切换的单目视频中重建统一世界坐标系下的人体运动是一项具有挑战性的任务，因为突然发生的镜头切换会带来大幅度的视角变化，并显著改变画面中可见的场景内容和人体外观。现有重建方法通常面向连续视频或完整序列处理而设计，因此往往难以在不同镜头之间维持空间连续性和人物身份连续性。为弥补这一空缺，我们提出 Shot3R，一个面向多镜头视频流式多人体四维重建的在线前馈框架。Shot3R 将镜头间对齐与递归状态传播解耦。在每个镜头边界，校正 token 分支联合读取先前状态和新镜头的第一帧，以估计镜头间对齐关系；与此同时，一个独立重置的分支生成在新镜头中唯一继续传播的状态。随后，仅依赖预测结果的人物关联与相机—人体共享变换，将新镜头的预测结果配准到此前的世界坐标系中，整个过程既不访问未来帧，也不需要针对每段视频执行全局优化。多个多人体数据集上的大量实验表明，Shot3R 取得了出色的重建精度与跨镜头一致性。特别是在 EgoBody 预先定义的极端视角子集上，相比连续状态递归方法，Shot3R 将 W-MPJPE 从 505.1 mm 降低至 354.3 mm，降幅达到 29.9%，并将 IDF1 从 0.758 提升至 0.983，体现出其在大幅视角变化下尤为显著的优势。此外，Shot3R 的运行速度达到 [X] FPS，相比连续状态递归方法仅增加 [Y]% 的端到端运行开销；在相同评测设置下，其速度是离线方法 [offline method] 的 [Z] 倍。**

候选版本 B 中占位内容的使用条件：

1. `[X] FPS` 必须来自包含镜头检测、模型推理、人物关联和坐标变换的端到端计时，不得直接使用当前排除检测器的受控重建耗时。
2. `[Y]%` 必须由相同硬件、输入、分辨率、数值精度和计时范围下的 Human3R 与 Shot3R 计算得到，并明确切换频率或采用数据集平均。
3. `[Z]` 和 `[offline method]` 只有在可执行离线方法能够于相同输入、硬件和计时范围下完成足够接近的任务时才保留；若输出范围或执行协议不可比，则从摘要删除这一分句，只在正文中作带限制的运行模式参考。
4. 当前已有的 L20、100 帧、FP32 受控实验表明，Human3R 为 3.224 FPS，Shot3R 单切换为 3.106 FPS，单切换相对 Shot3R 无切换路径增加 3.42% 的重建时间；但该计时排除了检测器，不能直接填入上述端到端占位符。

#### 3.22.1 最终选择（v035）

作者确认使用候选版本 B 中已有实验支持的具体指标部分，并删除所有端到端速度占位符。最终结果段为：

> **Extensive experiments on several multi-person datasets demonstrate strong reconstruction accuracy and cross-shot consistency. In particular, on the predefined extreme-viewpoint subset of EgoBody, Shot3R reduces W-MPJPE by 29.9% (from 505.1 to 354.3 mm) and increases IDF1 from 0.758 to 0.983 over continuous-state recurrence, demonstrating particularly strong gains under large viewpoint changes.**

对应中文：

> **多个多人体数据集上的大量实验表明，Shot3R 取得了出色的重建精度与跨镜头一致性。特别是在 EgoBody 预先定义的极端视角子集上，相比连续状态递归方法，Shot3R 将 W-MPJPE 从 505.1 mm 降低至 354.3 mm，降幅达到 29.9%，并将 IDF1 从 0.758 提升至 0.983，体现出其在大幅视角变化下尤为显著的优势。**

#### 3.23 修改后的核心叙事

后续全文统一沿用以下故事线：

1. 实际视频通常以单目 RGB 流的形式逐帧到达，在线重建无需等待完整序列，也不依赖逐视频全局优化。
2. 为了在有限的计算和记忆开销下持续整合历史信息，在线重建自然采用带状态的递归框架。
3. 连续镜头内有益的状态，在镜头突然切换时会携带不再兼容的视角相关信息；但完全丢弃历史又会切断共同世界坐标和人物身份。
4. Shot3R 的核心原则是区分信息的两种用途和生命周期：历史状态可以被读取以提供镜头间对齐证据，但不得成为被作为新镜头状态继续传播；新镜头只传播独立重置得到的干净状态。
5. 校正 token、预测人物关联和相机—人体共享变换是实现该原则的具体机制，而不是彼此孤立的工程组件。
6. 实验围绕该核心原则验证世界坐标人体重建、相机一致性、跨镜头身份连续性和因果执行效率；局限性与失败案例不得被概括性结果句掩盖。

#### 3.24 落实到论文时的对应修改

1. Introduction 的前两段沿用摘要的因果链条，先解释在线重建相对完整序列处理和逐视频全局优化的意义，再引出递归状态在镜头切换处的矛盾。
2. Related Work 按离线完整序列重建、在线递归重建和多镜头人体重建三条线组织，说明 Shot3R 解决的是它们交叉处的状态边界问题。
3. Method 首先给出“临时对齐证据”和“持续传播状态”具有不同生命周期的原则，再介绍校正 token、独立重置、预测人物关联和共享变换。
4. Experiments 的主结果与消融围绕摘要声称的重建精度、空间连续性、身份连续性和因果流式执行组织；不得在协议不匹配时扩写成统一 SOTA 排名。
5. 实验解释与 Conclusion 回到“对齐证据与递归记忆分离”这一一般性原则，并如实保留 Harmony4D、重复切换、检测器和稠密场景范围等限制。
6. 后续各部分不得重新把 Shot3R 展开为 `Read--Reset--Register` 方法名，不得将其写成 Human3R 的后处理、插件或补丁，也不得声称已经验证 backbone-agnostic。
7. v033 初稿使用候选版本 A；v035 已改用候选版本 B 的具体指标部分。由于端到端速度尚未完成，效率占位符和推测数字均未写入论文源文件。

#### 3.25 状态

**两个中英文候选版本均已保留；v033 初稿暂时写入候选版本 A，候选版本 B 待端到端时间实验完成后再决定是否替换。**

### R05. Related Work 的主题划分与段落写法

#### 3.26 当前写法

当前 v031 的 Related Work 分为四部分：

1. 有状态视觉重建；
2. 跨相机与镜头的全局人体重建；
3. 人体、相机和场景的联合重建；
4. 多人体预测与边界关联。

四部分涵盖了相关方法，但主题之间存在明显交叉：世界坐标系人体重建与人体—相机—场景联合重建有所重合，多人体身份关联本质上又是多镜头重建在镜头切换处产生的具体问题。当前写法还在多个段落中重复介绍 Shot3R 的具体流程，使 Related Work 部分接近 Introduction 或 Method 的叙述方式。

#### 3.27 需要修改的问题

1. 将四部分压缩为三个彼此分工明确的主题，分别回答“重建什么”“采用什么范式”和“核心场景是什么”。
2. 第一部分聚焦统一世界坐标系下的人体运动重建，不承担多人身份匹配的讨论。
3. 第二部分说明有状态在线重建及其对连续观测的依赖。在线范式不是 Shot3R 首创，因此不将“在线”本身写成方法创新。
4. 第三部分作为全文最重要的相关工作主题，集中讨论多镜头输入、镜头切换、坐标关系和多人身份连续性。
5. 多人体身份关联不再单独成节，而是作为多镜头人体重建中的一个核心问题展开。
6. 不为追求历史完整性而加入大量 SfM、MVS、NeRF 或早期人体网格恢复工作；优先讨论近期且与本文任务或方法直接相关的研究。
7. 参考 UniCon3R 的段落组织，每部分依次说明任务或研究方向、代表性方法、尚存问题以及 Shot3R 的区别。每节结尾只用简洁表述定位本文，不提前重复完整方法流程。
8. 不使用抽象的“因果地处理输入”解释在线设置，改为“按照时间顺序处理视频”，必要时进一步说明不能依赖未来帧或完整序列全局优化。
9. 不笼统声称现有多镜头方法“没有解决镜头边界问题”，而应准确说明其研究重点不在流式递归重建中的状态转换。

#### 3.28 作者最终确认方案

Related Work 调整为以下三个部分，并采用下列中文版作为正式英文改写的内容依据。

##### 第一部分：单目视频中的世界坐标系人体重建

单目人体重建已经从逐帧恢复相机坐标系下的人体网格，逐步发展到估计统一世界坐标系下的全局人体运动。SLAHMR 和 TRAM 等方法结合人体运动先验、相机运动估计与序列级优化，以恢复具有全局一致性的人体轨迹。WHAM 和 GVHMR 进一步通过学习式的相机—人体运动建模，提高了运动相机条件下全局人体恢复的准确性与效率。OnlineHMR 则将世界坐标系人体网格恢复拓展到在线推理场景。这些方法显著推进了单目视频中的全局人体重建，但大多建立在时间连续的输入之上。当镜头切换引起相机视角、画面内容和可见人物的突变时，原有的运动连续性和视觉对应关系将不再可靠。本文研究如何在此类不连续观测下，保持人体运动与相机坐标之间的一致性。

##### 第二部分：有状态在线重建

有状态在线重建按照时间顺序处理输入，并利用持续更新的递归状态积累历史几何与运动信息。CUT3R 将三维场景重建表述为由持久状态驱动的在线预测过程，Human3R 在此基础上将相机、场景与多个人体纳入统一的前馈重建框架。UniCon3R 和 GUSH3R 分别引入人体—场景接触信息和高斯场景表达，进一步扩展了有状态人体—场景重建。TTT3R、TTSA3R 和 ReCal3R 则从测试时适应、时空状态更新与可靠性校准等角度，改善长视频中的状态维护。现有方法主要研究如何在连续观测中保留和更新历史信息，却较少考虑镜头切换使历史状态突然失效的情况。与之不同，Shot3R 关注镜头边界处的状态转换，并将用于建立镜头间关系的对齐信息与负责后续在线重建的递归状态分离。

##### 第三部分：多镜头人体重建

多镜头人体重建旨在从时间上依次出现的单目镜头片段中恢复跨镜头一致的人体运动，而不同于依赖同步相机观测的多视角人体重建。HumanMM 从多镜头视频中恢复全局人体运动；Multi-THuMBS 研究跨越镜头边界的多人体三维网格跟踪与身份关联；ShowMak3r 则通过组合式辐射场和离线优化重建电视节目中的演员与场景。这些工作分别从全局运动恢复、跨镜头人体跟踪和影视内容重建等角度探索了多镜头输入，但其重点并不在流式递归重建中的状态转换。当模型按照时间顺序处理视频时，直接延续上一镜头的状态会将其中的视觉信息带入新镜头，而完全重置状态又会丢失跨镜头坐标对齐与人物身份关联所需的历史参照。为此，Shot3R 将跨镜头对齐与后续的递归状态传播解耦：历史状态仅用于建立新旧镜头之间的坐标和人物对应关系，重置后的状态则负责处理新镜头及其后续帧，从而保持相机姿态、人体运动和人物身份的跨镜头一致性。

#### 3.29 三部分的逻辑分工

1. “单目视频中的世界坐标系人体重建”说明本文要恢复什么，即统一世界坐标系下的人体运动。
2. “有状态在线重建”说明本文采用的计算范式，以及递归状态为何会在镜头切换处产生冲突。
3. “多镜头人体重建”说明本文最核心的任务场景，并集中讨论镜头间坐标对齐、状态转换和多人身份连续性。

#### 3.30 落实到论文时的对应修改

1. 将当前 Related Work 的四个段落重新组织为上述三个部分。
2. 保留与三个主题直接相关的近期工作，不机械保留旧版本中的每一项引用；正式写英文稿时逐条核验方法描述与引用对应关系。
3. 第一部分不展开多人身份匹配，相关内容统一移入第三部分。
4. 第二部分不把 Human3R 写成 Shot3R 的主体，也不把 Shot3R 描述成 Human3R 的补丁；Human3R 仅作为有状态联合重建路线中的代表性工作介绍。
5. 第二、三部分各自只保留与本节主题直接相关的一句 Shot3R 定位，避免重复摘要和方法部分的完整流程。
6. 正式英文稿应清楚区分 `multi-shot` 与同步 `multi-view`，并使用直接可理解的顺序处理表述替代未经解释的抽象 `causal processing`。
7. 本条只调整 Related Work 的组织和写法，不改变方法、实验协议、引用事实或任何已有实验结果。

#### 3.31 状态

**中英文版本已写入 v033，中文以 LaTeX 注释保存，不进入编译后的 PDF。**

### R06. Experiments 的结构、基线与图表呈现

#### 3.32 当前写法

当前 v031 的实验部分分为四个主要部分，协议、数据集、指标、因果执行说明和实现细节分散在不同位置。部分标题沿用内部实验与事实审计阶段的命名，结果叙述也较多按照证据文件和实验路线展开。表格包含若干信息价值有限或在各行完全相同的列，caption 承担了过多协议解释。此外，v031 尚未纳入后续在 v032 中完成的 OnlineHMR 正式实验。

#### 3.33 需要修改的问题

1. 将相互重合的四部分压缩为三个层次清楚的部分，使实验设置、主要结果以及分析消融各自承担明确功能。
2. 将 Baselines、Datasets and Metrics、Implementation Details 集中放入实验设置，避免协议说明在结果段重复出现。
3. 结果应围绕论文提出的研究问题组织，而不是按照内部证据文件、运行批次或工程模块罗列。
4. 标题、表格行名和正文术语应与摘要和方法的核心思想一致，减少 `strict`、`gauge`、`clean reset`、`read-only gauge` 和 `evidence block` 等内部工程化表达。
5. 加入 OnlineHMR 作为最直接的公开在线参考，同时明确其半在线人物跟踪与独立全局相机后端和 Shot3R 的逐帧流式协议并不完全相同。
6. 主表需要同时呈现人体重建、相机轨迹、人物身份和预测可用性，但外部方法的条件几何结果不得与内部同输入配对实验组成无条件的统一排名。
7. Coverage 应按实际信息价值保留：三数据集总体结果中存在变化，因而保留；在各行完全相同的消融表中删除。
8. caption 应只概括表格回答的问题，必要的有效样本、时间访问方式、对齐规则和统计限制移至表注、正文或补充材料。

#### 3.34 作者阶段性确认方案

实验部分暂按以下结构组织：

1. `4.1 Experimental Setup`：包含 Baselines、Datasets and Metrics、Implementation Details；
2. `4.2 Multi-Shot Multi-Person Reconstruction`：包含 Overall Results、Large Viewpoint Changes、Comparison with Executable Methods；
3. `4.3 Analysis and Ablation Studies`：包含 Alignment and State Propagation、Multi-Person Association、Shot-Transition Detection and Efficiency。

中文版完整候选稿、候选表格以及中英文 caption 以 `EXPERIMENTS_RESTRUCTURE_DRAFT_20260905.md` 为准。本版暂时保留，后续再进行正式英文改写、LaTeX 表格制作和逐句确认。

具体呈现原则如下：

1. Human3R 作为相同有状态在线范式下的主要受控基线，面向读者的名称统一写为 `Human3R`。
2. OnlineHMR 作为三个数据集上的同输入在线参考，报告正式结果、有效输出数量和协议差异；TRACE 与 PromptHMR-SPEC 作为输出接口和有效支持不同的可执行参考紧凑呈现。
3. HumanMM 与 Multi-THuMBS 只作为相关任务和协议背景讨论，不将不同数据与协议下的原文结果混入同输入数值表。
4. 表 1 围绕人体重建、相机、身份和可用性组织；外部可执行方法不与 Human3R—Shot3R 配对结果共同加粗排名，并在数值旁标明有效样本数。
5. 表 2 聚焦大幅视角变化，只保留 W、WA 和 IDF1，完整 Coverage 与统计结果移至正文或补充材料。
6. 表 3 验证镜头间对齐与递归状态传播的作用；由于各行 Coverage 均为 0.981，删除该列。
7. 表 4 采用人物关联、首次触发检测和受控重建开销三个紧凑面板。
8. 4.3 不按零散模块堆叠，而是围绕解耦对齐与状态传播这一核心思想及其必要机制组织。

#### 3.35 指标与比较协议的准确含义

1. W-MPJPE 使用序列最早的有效观测确定 Sim(3) 对齐，并将该变换固定到后续帧，因此对跨镜头全局轨迹偏移更敏感。
2. WA-MPJPE 使用全部有效观测确定 Sim(3) 对齐，衡量完整序列对齐后的人体重建误差。
3. EgoBody 在完整序列和所有人物之间共享一个对齐；EgoHumans 与 Harmony4D 按人物轨迹对齐。因此绝对指标只在同一数据集内比较，跨数据集只汇总相对变化。
4. OnlineHMR 在 EgoBody、EgoHumans 和 Harmony4D 上的正式有效原生输出分别为 129/129、87/90 和 88/88；所有外部方法均需如实报告实际有效支持。
5. “在线”“流式”和“因果执行”的结论必须与各方法真实的时间访问协议对应，不能因输入相同就把不同后端和跟踪设置描述为完全等价。

#### 3.36 落实到论文时的对应修改

1. 后续正式执行时，将当前实验部分重组为 4.1–4.3，并基于候选稿生成英文正文和对应中文翻译。
2. 重新制作主文表格和简短 caption，但不得更改任何已有方法、输入样本、评测协议、聚合方式或数值结果。
3. 将详细置信区间、sign-flip 检验、辅助指标、完整有效输出与失败记录、重复切换实验、检测器审计和详细计时协议移入补充材料。
4. 在完成包含镜头检测、模型推理、人物关联和坐标变换的端到端计时前，受控重建耗时不得被写成完整系统速度，也不得直接填入摘要效率占位符。
5. 正式写入前继续核对 OnlineHMR 等外部方法的公开实现、时间访问方式和有效输出，避免把条件结果误写成统一 SOTA 排名。
6. 本次记录不修改 v031、v032 或其他现有 LaTeX 论文源文件。

#### 3.37 状态

**已按阶段性确认方案写入 v033；主文保留关键协议、数值和负面结果，完整细节继续置于补充材料。**

### R07. 删除独立 Discussion 并重写 Conclusion

#### 3.38 当前写法

当前 v031 在 Experiments 与 Conclusion 之间单独设置 Discussion。其第一段重新解释 WA-MPJPE、相机 ATE、IDF1、局部人体误差和大视角结果，并补充弱纹理条件下的镜头检测失败；第二段说明受控比较协议、可训练参数和计时口径，随后集中列出 Harmony4D、AIST++、重复切换、共享平移与稠密场景范围等局限。

紧随其后的 Conclusion 只有一个很短的段落，既再次概括相同实验与失败现象，又沿用 `same-scene cuts`、`boundary gauge`、`read-only history`、`clean-reset recurrence` 和 `universal editing` 等已决定删除或弱化的旧术语。因此，两节之间存在明显重复，且 Conclusion 没有以当前核心叙事完整总结工作。

#### 3.39 需要修改的问题

1. Discussion 中的多数内容分别属于实验结果解释、实验协议、效率分析或局限性，没有形成必须单独成节的进一步理论讨论。
2. 实验结果的含义应在对应结果出现时立即解释，避免读者在下一节再次阅读同一结论。
3. 比较协议、参数量与计时范围应分别回到 4.1 和 4.3，而不是在实验结束后补充说明。
4. 失败案例和适用范围不能删除，但主文只需保留简洁、具有代表性的局限；完整数字和诊断继续放在补充材料。
5. Conclusion 应围绕“多镜头单目视频中的在线流式多人体四维重建”和“解耦镜头间对齐与递归状态传播”总结全文，不再使用旧的三步式或 gauge 叙事。

#### 3.40 作者确认的结构方案

删除独立的 Discussion。第四节 Experiments 结束后直接进入第五节 Conclusion。原 Discussion 中有用的内容按以下方式归并：

1. 主要结果及其含义并入 4.2；
2. 对比协议与参数设置并入 4.1；
3. 检测失败和受控开销并入 4.3；
4. 详细失败实验放入补充材料；
5. 总体局限和未来方向压缩为 Conclusion 的第二段。

Conclusion 采用两段式结构：第一段总结问题、核心思想与实验发现；第二段不罗列具体模块故障，只将复杂剪辑、人物反复进出画面和快速画面变化概括为值得进一步研究的问题。当前保留的中文全文以 `CONCLUSION_RESTRUCTURE_DRAFT_20260905.md` 为准。

#### 3.41 落实到论文时的对应修改

1. 后续正式执行时，从主文入口中删除独立 Discussion 的引用，使 Experiments 后直接进入 Conclusion。
2. 在确认候选文字后，将 Conclusion 重写为英文，并提供逐段对应中文；当前不提前写入 LaTeX 源文件。
3. 不因删除 Discussion 而删除负面结果、失败案例或适用范围；只调整它们在主文和补充材料中的位置。
4. 不重复已经写入 4.1--4.3 的实验数字，也不将受控重建计时扩大解释为端到端效率。
5. 本项只修改论文结构与写法，不改变任何方法、实验协议或数值结果。
6. Conclusion 的局限段不逐项罗列检测器、人物关联、共享平移、重复切换或单项指标回退等具体问题；这些事实继续在对应实验和补充材料中如实保留。主文结尾只将复杂剪辑、人物反复进出画面和快速画面变化概括为仍待进一步研究的复杂条件。

#### 3.42 状态

**已写入 v033：删除独立 Discussion，实验后直接进入带中文注释的两段式 Conclusion。**

### R08. 最终模型的训练目标、实现细节与收敛曲线

#### 3.43 当前写法与问题

v035 将 `v14_cut_first_cross_source_multihuman_p0_e6` 的 6 轮操作当作最终有效训练阶段，据此写入五个数据源、每轮 480 个事件、总计 2,880 步和八项 event-local objective。但作者确认该操作使用了重复数据，属于此前的失误操作，本质上没有形成需要单独报告的有效训练阶段。因此，论文不应继续区分“60h 基础训练”和“6 轮边界训练”，也不应把后者的 loss、数据或曲线写成 Shot3R 的正式训练配置。

#### 3.44 最终修改方案

正式训练直接对应：

- checkpoint：`Movie3R/checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth`；
- 配置：`Movie3R/config/train_v9_mixed_avatarrex_thuman_60h_pose_human_lora_bs10.yaml`；
- 完整日志：`Movie3R/checkpoints/v9_mixed_60h_pose_human_lora_bs10/train.log`。

论文将该 checkpoint 统一称为 Shot3R 最终模型，不再使用“基础阶段”“边界阶段”或“两阶段训练”的区分。配置名中的 `60h` 是预计预算名称；日志记录的实际运行时间为 1 天 5:28:44。论文训练设置以 72 轮和 7,200 次更新为准，不把 `60h` 当作实际训练时长报告。

Method 写入该最终模型实际使用的关系校正目标：

\[
\begin{aligned}
\mathcal L_{\mathrm{rel}}={}&
\mathcal L_t+5\mathcal L_R
+0.05\mathcal L_g+0.05\mathcal L_{\mathrm{imp}}\\
&+10^{-5}\mathcal R_{\mathrm{cam}}
+10\mathcal L_h+10^{-5}\mathcal R_{\mathrm{human}}.
\end{aligned}
\]

各项的准确含义为：

1. \(\mathcal L_t\) 是相机平移的 Smooth-L1 loss；
2. \(\mathcal L_R\) 是相机四元数的测地线旋转 loss；
3. \(\mathcal L_g\) 用以 0.5 米和 45 度归一化并截断的原始相机误差监督校正门控；
4. \(\mathcal L_{\mathrm{imp}}\) 要求校正后的归一化相机误差小于未经校正的误差；
5. \(\mathcal L_h\) 是人体 SMPL 平移的 Smooth-L1 loss；
6. \(\mathcal R_{\mathrm{cam}}\) 和 \(\mathcal R_{\mathrm{human}}\) 分别约束相机与人体潜变量残差大小。

4.1 Implementation Details 补充以下信息：

1. 训练源为 AvatarReX 和 THuman；每个来源包含 8,000 个 AABB 跳变序列与 2,000 个 AAAA 连续序列；
2. 两个来源按 0.6/0.4 的概率采样，batch size 为 10；
3. 共训练 72 轮，每轮 100 次更新，总计 7,200 次优化器更新，输入分辨率为 512；
4. 使用 AdamW，\(\beta=(0.9,0.95)\)，weight decay 为 0.05，并启用自动混合精度和梯度检查点；
5. 学习率经 4 轮线性预热到 \(2\times10^{-4}\)，随后余弦衰减至 \(10^{-6}\)；
6. 冻结图像编码器、递归解码器、点图头、基础相机/人体预测头和 Multi-HMR 主干，只训练关系 token、相机/人体潜变量残差分支与秩 8 LoRA，共 19.62M 参数，占完整模型 1.63%；
7. 每 6 轮在 AvatarReX-AABB、AvatarReX-AAAA、THuman-AABB 和 THuman-AAAA 四个固定划分上验证；第 72 轮 final checkpoint 作为正式模型。

#### 3.45 loss 曲线与补充材料边界

原始 TensorBoard event 在空间清理时未被保留，但完整文本日志保留了 72 个逐轮训练汇总点，以及从第 0 轮到第 72 轮、每 6 轮一次的 13 组验证结果。已按日志打印精度恢复：

- 训练目标由 2.2939 降至 0.0711；
- 四个验证划分中位数 loss 的平均值由 1.9567 降至 0.0849；
- 星号直接标记正式使用的第 72 轮 final checkpoint，不区分额外边界训练阶段。

曲线 PDF/SVG/PNG、训练/验证/test CSV、恢复脚本与 SHA-256 来源说明均保存在 v036。曲线放入补充材料；主文 4.1 只简要报告起止值并引用补充材料，以保持主文篇幅。

#### 3.46 状态

**作者已推翻 v035 的 6 轮训练叙事；新方案已写入 v036，并恢复最终模型曲线。**

### R09. 补充材料与实验表格重构

#### 3.47 当前写法与问题

v036 的补充材料包含 11 个章节和 36 张表，部分章节实质上是内部证据记录而不是论文叙事；EgoBody 主结果、公开方法结果和检测器结果存在重复。一些表格包含整列相同数值、全 0/1 行或单行结果，例如原检测器表，以及 $\lambda$ 敏感性表中不随 $\lambda$ 改变的 MPJPE、MPVPE、IDF1 和 Coverage。正文实验表格虽然数值完整，但方法属性和消融关系不够直观。

#### 3.48 最终修改方案

1. 补充材料重组为五部分：评测协议、训练与实现细节、补充方法细节、补充实验、局限性；补充实验按照“大视角变化—公开方法—多次镜头切换—三个主要数据集—弱纹理场景”的顺序展开。
2. 删除与正文重复的 EgoBody 总表和公开方法表；将全 1/0 的检测结果改为一句正文说明；删除 Coverage 等完全相同的列。
3. 将 Harmony4D 的均值/中位数 $\lambda$ 敏感性合并为一张表；合并 AIST++ 多次切换表和 MVHuman 公开方法结果；单人实验不重复展示不起作用的人物关联行。
4. 将“audit、manifest、ledger、deployable、evidence boundary”等内部记录式表述改为直接说明实验问题、协议和结论的论文语言。
5. 参考 Human3R Table 2 的信息组织方式，在正文 Table 1 中用彩色对勾/叉号标明 Online、Camera trajectory 和 Shot-aware 属性，在 Table 3 中标明 Alignment、State reinitialization 和 Association 组件。
6. 本轮不修改任何实验输出、表格数值或已有图片内容；定性图与更多可视化留到后续单独修改。
7. 参考 Human3R 的 appendix，在补充材料开头设置独立 `Table of Contents`：列出 A--E 五个主章节和 D.1--D.7 七个补充实验子节，以点线连接标题和自动页码；目录不重复正文条目，并保留对应中文 LaTeX 注释。

#### 3.49 状态

**已写入 v037：补充材料由 36 张表精简为 24 张，正文 Table 1/Table 3 完成彩色属性与组件重排，并增加带自动页码的独立附录目录。**

### R10. 运行效率数字、计算方式与后续测试

#### 3.50 当前证据与计算方式

当前正式速度结果来自一段固定的 100 帧 EgoHumans 视频，镜头边界位于第 50 帧。测试使用单张 NVIDIA L20、输入分辨率 512、batch size 1 和 FP32，并关闭 TF32。模型与输入在计时前完成加载和预处理；每条路径先运行一次 warm-up，再正式重复三次，表中时间与 FPS 使用正式重复的中位数。

FPS 统一按“最终输出帧数除以总耗时”计算：

- Human3R：$100/31.020=3.224$ FPS；
- Shot3R 无切换路径：$100/31.125=3.213$ FPS；
- Shot3R 单切换路径：$100/32.191=3.106$ FPS。

单切换路径实际执行 101 次神经网络帧计算：历史条件路径处理切换前 50 帧与新镜头首帧，共 51 帧；重新初始化路径处理新镜头 50 帧。两条路径最终产生 100 帧输出，因此 3.106 应理解为摊销后的输出 FPS。单次切换相对 Shot3R 无切换路径增加 1.066 秒，即 3.42% 的受控重建开销。

#### 3.51 计时包含与排除范围

当前计时包含模型推理、模型输出解码、仅依赖预测的人物关联和共享空间变换；不包含 checkpoint 加载、RGB 解码、图像缩放和镜头切换检测。因此，这组结果用于衡量重建核心路径及切换操作的增量成本，不等同于完整系统的端到端运行速度。

#### 3.52 其他方法与历史时间记录

仓库中保留了 EgoBody 多方法流程、OnlineHMR 和 JOSH 的逐样例 wall-clock time，Multi-THuMBS 原文也报告了 150 帧 1080p 视频在 RTX 3090 上约 10 分钟的完整优化时间。但这些记录在硬件、输入长度、分辨率、并发环境和计时范围上与当前受控表不同，不能直接换算为公平的 FPS 排名或速度倍数。它们只能作为内部诊断或带明确来源限制的运行模式背景。

#### 3.53 已确认的论文表述边界

1. 可以写明 Shot3R 在受控重建测试中达到 3.106 摊销输出 FPS，并且单次切换仅增加 3.42% 的重建开销。
2. 不将 3.106 FPS 写成包含检测器和输入处理的端到端 FPS，也不使用 `real-time`。
3. 在完成统一端到端实验前，不声称 Shot3R 相对 Multi-THuMBS、JOSH 或 OnlineHMR 快具体多少倍。
4. 后续测试应在同一硬件、分辨率、精度与输入集合上分别记录预处理、镜头检测、重建、边界处理和总时间，并覆盖多个样例与不同切换次数。
5. 若最终仍无法在同一协议下执行离线方法，则正文只报告 Shot3R/Human3R 的同协议增量成本；离线方法仅引用其原论文报告的时间，并明确不可直接比较。

#### 3.54 状态

**已确认并记录。v042 保留现有论文数值，同时将包含逐次计时、硬件、软件版本与脚本哈希的原始 JSON/MD 报告纳入版本化论文证据；其他方法的统一端到端测速后续完成。**

### R11. 将核心贡献改写为“洞察—方法—证据”

#### 3.55 需要修改的问题

Introduction 原贡献第一条将方法概括为一个 `general principle`，并使用 `different information lifetimes` 解释两类状态。该写法较抽象，也容易让审稿人要求跨架构或更一般任务上的验证。第一、二条都在重复“解耦对齐与状态传播”，第三条则主要罗列同输入比较、公开方法和消融设置，没有直接概括论文建立的能力与主要发现。

#### 3.56 作者确认的核心表述

> **At a shot boundary, the previous state should serve as a reference for inter-shot alignment, rather than the recurrent state propagated through the new shot.**

对应中文：

> **镜头切换时，先前状态应作为镜头间对齐的参照，而不应成为在新镜头中继续传播的递归状态。**

#### 3.57 作者确认的三条贡献

1. **We identify a state conflict in streaming multi-shot reconstruction: preserving the previous state is necessary for connecting shots, yet propagating it interferes with reconstruction in the new view. We resolve this conflict by using history for alignment without carrying it into subsequent recurrence.**

   **我们指出了流式多镜头重建中的状态冲突：连接不同镜头需要保留先前状态提供的参照，但继续传播该状态又会干扰新视角中的重建。我们的解决思路是让历史信息参与镜头间对齐，但不进入新镜头的后续递归过程。**

2. **We introduce Shot3R, which realizes this separation through a temporary history-conditioned alignment path and an independently reinitialized recurrent path. Prediction-based person association and a shared camera--human transform register the new-shot reconstruction to the existing world frame without future frames or per-video optimization.**

   **我们提出 Shot3R，通过临时的历史条件对齐路径和独立重置的递归路径实现上述分离。基于预测的人物关联和相机—人体共享变换，将新镜头的重建结果配准到已有世界坐标系中，整个过程不访问未来帧，也不需要逐视频优化。**

3. **Across three multi-person benchmarks, Shot3R improves world-space motion reconstruction and identity continuity over continuous-state recurrence, with larger gains under substantial viewpoint changes. A controlled runtime study further shows that one shot transition adds only 3.4% reconstruction overhead.**

   **在三个多人体基准上，相比连续状态递归，Shot3R 提高了世界空间人体运动重建精度和身份连续性，并且在大幅视角变化下取得更明显的提升。受控运行时间实验进一步表明，一次镜头切换仅增加 3.4% 的重建开销。**

#### 3.58 落实范围与状态

贡献列表按上述中英文定稿写入 v043。Method 中 `two paths with different information lifetimes` 同步改为 `two paths with separate roles in alignment and recurrence`，不再把当前结论扩写为未经验证的跨架构一般理论。本项只修改贡献包装和对应术语，不改变方法、公式、实验或数值。

### R12. 统一历史条件对齐路径与镜头对齐模块的名称

#### 3.59 需要修改的问题

此前摘要、方法和补充材料使用 `correction-token branch`、`history-conditioned path`、`alignment representations`、`relation representation`、`relation tokens` 和 `relation-correction objective` 指代同一条路径或其中的可学习组件。读者难以判断这些名称是同义词还是相互独立的模块，也会削弱方法主线。

#### 3.60 作者确认的命名层级

1. 完整的临时路径统一称为 **`history-conditioned alignment path`（历史条件对齐路径）**。
2. 路径中的可学习组件统一称为 **`shot-alignment module`（镜头对齐模块）**。
3. 只有介绍该模块内部结构时使用 **`alignment tokens`（对齐 token）**，并具体区分 semantic token、camera-alignment token 和 temporal-continuity token。
4. 对齐门控称为 `alignment gate`，训练目标称为 `shot-alignment objective`，公式记号使用 $\mathcal L_{\mathrm{align}}$。

#### 3.61 摘要中的定稿表述

> **At each shot boundary, a history-conditioned alignment path combines the previous state with the first new-shot frame to estimate the inter-shot transform, while an independently reinitialized path provides the only state propagated through the new shot.**

对应中文：

> **在每个镜头边界，历史条件对齐路径结合先前状态与新镜头第一帧估计镜头间变换，而独立重置的路径提供在新镜头中唯一继续传播的状态。**

#### 3.62 落实范围与状态

上述命名已同步写入 v044 的摘要、Method、补充实现细节、补充方法细节、图注和方法图。方法图顶部的抽象 `information lifetimes` 改为更直接的 `Previous state aligns; reinitialized state propagates`（先前状态负责对齐，重置状态负责后续传播）。公式的数学内容、训练参数、实验协议和所有数值保持不变。

### R13. 在摘要中区分在线连续视频方法与离线多镜头方法

#### 3.63 需要修改的问题

此前摘要将“面向连续视频的方法”和“完整序列处理方法”合并为一类，并笼统称其难以维持跨镜头空间与身份连续性。该表述没有区分两类工作的不同限制，也可能不准确地弱化 Multi-THuMBS 等已有多镜头方法所解决的问题。

#### 3.64 作者确认的定稿表述

> **Existing online reconstruction methods assume temporally continuous input and can be unreliable at shot boundaries. Prior multi-shot approaches instead rely on cross-shot or full-sequence processing rather than online inference.**

对应中文：

> **现有在线重建方法假设输入在时间上连续，因此在镜头边界处可能变得不可靠。已有多镜头方法则依赖跨镜头处理或完整序列优化，而非在线推理。**

#### 3.65 修改后的准确含义

该表述将已有方法分为两类：在线有状态方法能够逐帧处理连续视频，但连续性假设在镜头切换处受到破坏；已有多镜头方法能够利用不同镜头的信息，但依赖跨镜头联合处理或完整序列优化。Shot3R 所填补的空缺是随着输入到达进行在线多镜头重建，而不是声称已有多镜头方法无法建立跨镜头一致性。

#### 3.66 落实范围与状态

上述中英文已写入 v045 摘要。其余方法、实验和数值保持不变。

### R14. 将镜头对齐模块改写为可解释的三类证据

#### 3.67 需要修改的问题

此前 Eq. (5) 使用单个抽象函数 $\Phi_\phi$ 一次性接收 $u_b,\bar u_b,p_b,p_{b-1},m_b,h_{b-1}$ 等多个高层符号。虽然形式紧凑，但读者无法从公式判断三个对齐 token 分别携带什么信息，这既削弱了方法动机，也没有提供足够的实现对应关系。

#### 3.68 作者确认的修改原则

1. 保留三 token 结构，但不再用未对应实际模块的统一黑盒函数表示。
2. 将三个 token 明确写为语义证据、相机变化和历史校正上下文：语义 token 自适应融合当前观测与旧镜头记忆；相机对齐 token 显式编码相机 token 及其潜变量差；时间连续性 token 汇总此前的对齐表示、相机校正量和门控值。
3. 正文解释三个 token 的互补作用，补充材料记录注意力汇聚、语义门控、类型嵌入和归一化等实现细节。
4. 所有公式与最终权重对应的 `relation_v8_2`、三 token `all` 配置保持一致，不引入代码中不存在的信息或操作。

#### 3.69 落实范围与状态

新的三 token 公式及其中英文说明已写入 v046 Method，补充材料同步补全其实际构造。方法行为、模型权重、训练过程、实验协议和数值均未改变。

### R15. 明确人物关联的实际匹配规则

#### 3.70 需要修改的问题

Method 和补充材料此前使用 `reliable person match`，但正式发布路径没有人工置信度阈值或匹配拒绝模块。实际实现对边界两侧的预测人物执行 Hungarian assignment，返回 $\min(N_b^-,N_b^+)$ 个匹配；人数不一致时，新镜头未匹配检测结果获得新匿名身份，空匹配时跳过人物引导的平移并保留相机对齐。

#### 3.71 作者确认的定稿表述

> **Given $N_b^-$ and $N_b^+$ detected people on the two sides of the boundary, Hungarian assignment returns $\min(N_b^-,N_b^+)$ one-to-one pairs, which form $\mathcal M_b$. All assigned pairs are retained without a hand-tuned confidence threshold. Matching uses only model predictions, without identity labels, camera calibration, or future frames. Unmatched new-shot detections are initialized with new anonymous identities. If $\mathcal M_b=\emptyset$, the person-guided translation is omitted and the camera alignment is retained.**

对应中文：

> **设镜头边界前后分别检测到 $N_b^-$ 和 $N_b^+$ 个人，Hungarian assignment 返回 $\min(N_b^-,N_b^+)$ 个一对一匹配，并由此构成 $\mathcal M_b$。所有匹配结果均被保留，不使用人工设定的置信度阈值。匹配仅依赖模型预测，不使用人物身份标签、相机标定或未来帧。新镜头中未匹配的检测结果将被初始化为新的匿名身份。当 $\mathcal M_b=\emptyset$ 时，跳过人物引导的平移修正，仅保留相机对齐。**

#### 3.72 落实范围与状态

上述规则已同步写入 v047 的 Method、补充材料和中英文注释。该项只澄清正式实现，不改变匹配代价、模型权重、运行结果或实验协议。

### R16. 清理面向读者的旧 `relation` 模块术语

#### 3.73 需要修改的问题

实验设置和 MVHuman 弱纹理补充实验仍使用 `relation parameters` 与 `relation module`，与正文已经统一的 `shot-alignment module` 不一致。全文中表示普通空间含义的 `spatial relation` 和 `inter-shot camera relation` 不属于模块别名，不做机械替换。

#### 3.74 作者确认的定稿表述

> **The trainable parameters comprise the shot-alignment module, camera/human residual branches, and low-rank prediction-head adapters (19.62M in total, 1.63\% of the model). We optimize these parameters for 7,200 updates...**

> **可训练参数包括镜头对齐模块、相机/人体残差分支和预测头低秩适配器，共 19.62M 个参数，占完整模型的 1.63\%。我们使用 7,200 次更新对这些参数进行优化……**

以及：

> **We further evaluate on ten MVHuman captures that are not used to train the shot-alignment module.**

> **我们进一步在十个未参与镜头对齐模块训练的 MVHuman 采集序列上进行测试。**

#### 3.75 落实范围与状态

上述术语已写入 v048 的实验设置、补充材料和中英文注释。该项只统一名称和句法，不改变参数数量、训练过程或任何实验结果。

### R17. 收窄运行时间贡献的证据范围

#### 3.76 需要修改的问题

贡献列表此前写成一次镜头切换“only adds 3.4% reconstruction overhead”，但该数值来自一项受控的 100 帧测试，计时包含模型前向、输出解码、人物关联和世界坐标对齐，不包含图像读取、缩放和镜头检测。因此不应将其表述为完整端到端或所有序列上的普遍效率结论。

#### 3.77 作者确认的定稿表述

> **In a controlled 100-frame benchmark, one shot transition adds 3.4\% overhead to the measured reconstruction path.**

对应中文：

> **在一项受控的 100 帧测试中，一次镜头切换使测得的重建路径耗时增加 3.4%。**

#### 3.78 落实范围与状态

上述表述已写入 v049 Introduction 贡献列表。补充材料和实验段中的详细计时范围保持不变；后续若完成多序列统一端到端测速，再重新评估是否扩大效率结论。

### R18. 简化摘要中的 `online feed-forward framework`

#### 3.79 需要修改的问题

`feed-forward` 在三维重建文献中可以表示不进行逐视频全局优化，但与 Shot3R 同时维护 recurrent state 的表述放在摘要中容易造成不必要的阅读停顿。摘要后文已经明确说明不访问未来帧且不执行逐视频全局优化，因此无需在框架名称中重复强调 `feed-forward`。

#### 3.80 作者确认的定稿表述

> **To fill this gap, we present Shot3R, an online framework for streaming multi-person 4D reconstruction from multi-shot videos.**

对应中文：

> **为弥补这一空缺，我们提出 Shot3R，一个面向多镜头视频流式多人体四维重建的在线框架。**

#### 3.81 落实范围与状态

上述中英文已写入 v050 摘要。Related Work 中用于方法类别描述的 `feed-forward` 保留；方法、实验和结果不变。

### R19. 将实验结果解释改为直接的指标观察

#### 3.82 需要修改的问题

实验结果段此前使用 `locating the main benefit in world-frame consistency`，容易把指标变化解释为已经证明了方法机制来源。

#### 3.83 作者确认的定稿表述

> **Local pose error is nearly unchanged, while the largest measured gains are in world-frame placement and identity continuity.**

对应中文：

> **局部姿态误差基本不变，而测得的最大幅度提升体现在世界坐标定位和人物身份连续性上。**

#### 3.84 落实范围与状态

上述表述已直接修改 v050 的实验结果段。该项只收窄结论强度，不改变表格、指标或数值。

### R20. 限定状态重置消融的因果表述

#### 3.85 需要修改的问题

消融段此前使用 `confirming that reset alone severs the world-frame relation`，容易将一个使用固定边界的受控消融结果写成关于所有状态重置的普遍机制证明。

#### 3.86 作者确认的定稿表述

> **State reinitialization alone worsens W/WA to 752.6/589.4\,mm and camera error from 0.124 to 1.141\,m, showing that, in this controlled setting, reset alone does not preserve the world-frame relation.**

对应中文：

> **仅重置状态会将 W/WA 恶化至 752.6/589.4 mm，并将相机误差从 0.124 m 增至 1.141 m，表明在这一受控设置下，单独重置状态无法保持世界坐标联系。**

#### 3.87 落实范围与状态

上述表述已直接修改 v050 的消融实验段和中文注释。表格、消融协议及数值保持不变。

### R21. 修正匹配平移与消融实验的措辞边界

#### 3.88 需要修改的问题

人物引导平移此前写成由 `valid matches` 确定，可能暗示存在未说明的匹配筛选阈值；Table 3 的中间设置此前使用 `isolate`，容易被理解为独立重训练的严格组件因果消融。

#### 3.89 作者确认的定稿表述

> **All matched pairs jointly determine a person-guided translation.**

> **所有匹配对共同确定人物引导的平移修正。**

以及：

> **The two intermediate rows use annotated transition timing to examine the effects of state reinitialization and camera alignment under a controlled boundary setting.**

> **两个中间设置使用真实镜头切换位置，在受控边界条件下考察状态重置和相机对齐的影响。**

#### 3.90 落实范围与状态

上述两处已直接修改 v050 的 Method、实验分析段和中文注释。所有表格、checkpoint、消融协议及结果数值保持不变。

## 4. 后续条目格式

后续每项修改继续采用以下结构：

1. 当前写法；
2. 需要修改的问题；
3. 作者最终确认方案；若包含必须原样写入论文的英文，则同时给出英文定稿和对应中文，否则只用中文记录；
4. 修改后的准确含义；
5. 落实到论文时的对应位置与一致性要求；
6. 与历史决策或既有实验的关系；
7. 定稿和执行状态。
