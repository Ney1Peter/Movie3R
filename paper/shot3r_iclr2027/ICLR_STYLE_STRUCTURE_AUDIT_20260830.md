# BRIDGE3R ICLR 2027 正文结构与写作风格审计

日期：2026-08-30  
对象：`Human3R.pdf`、`JOSH.pdf`、`StreamVGGT.pdf`、`TTT3R.pdf`、`WinT3R.pdf` 与 BRIDGE3R v024  
用途：为 v025 提供可直接执行的九页正文预算、叙事顺序与图表安排；本文不修改论文源文件。

## 1. 审计口径

- 使用 `pdfinfo` 核对页数和页面尺寸，使用 `pdftotext -layout` 核对章节起始页、正文/参考文献分界及图表所在页。
- 五篇参考论文来自不同 ICLR 版本或预印本状态，正文允许页数并不完全一致，因此不能机械复制其总页数；本审计只总结其结构规律。
- BRIDGE3R v024 采用 ICLR 2027 letter-size 模板。其正文结束于第 9 页，参考文献从第 10 页开始，补充材料从第 11 页开始，满足当前项目中记录的“初投稿正文最多九页、参考文献不计入九页”的要求。
- 以下“篇幅”按章节实际占据页面的大致比例估计，跨页浮动图表会造成少量误差。

## 2. 五篇参考论文的结构

| 论文 | 正文主线 | Related Work | Method | Experiments | 主要图表节奏 |
|---|---|---:|---:|---:|---|
| Human3R | 统一在线人体、相机与场景重建 | 约 1 页，p3–p4 | 约 2.5 页，p4–p6 | 约 3.5 页，p6–p9 | p1 teaser；p5 方法总览；p7 两张主表；p8–p9 定量/定性分析 |
| JOSH | 从分离估计转向人、相机、场景联合优化 | 主体 Related Work 移入附录；正文以 Preliminaries 交代依赖 | 约 3.5 页，p3–p6 | 约 4 页，p6–p10 | p1 teaser；p4 方法总览；p7 主结果与定性图；p9 消融；p10 效率与结果图 |
| StreamVGGT | 用因果架构和缓存记忆实现流式几何重建 | 约 1 页，p2–p3 | 约 4 页，p3–p7 | 约 3 页，p7–p10 | p1 总览；p4 框架图；p5 推理机制图；p7–p9 主表；p10 三张消融表 |
| TTT3R | 将循环状态解释为测试时更新的快速权重 | 约 1 页，p3–p4 | 约 3 页，p4–p7 | 约 3 页，p7–p10 | p1 核心机制 teaser；p4 序列建模对照图；p6 方法图；p8–p10 以曲线和定性图为主 |
| WinT3R | 滑动窗口与相机 token 池兼顾质量和在线速度 | 约 1 页，p2–p3 | 约 3 页，p3–p6 | 约 4 页，p6–p10 | p1 总览；p4 pipeline；p5 attention mask；p7 主表；p8 定性图；p9 消融表 |

### 2.1 Human3R

- 第 1 页用一句明确能力描述配合 teaser：输入 RGB 流，在线输出多人、相机和场景。
- Introduction 先说明“人体不能脱离场景理解”，再归纳既有方法的多阶段、重依赖与低效率问题，最后逐层引出统一模型、参数高效训练和人体 prompt。
- Related Work 不是文献罗列，而是围绕三个技术缺口组织；每一段最后都回到 Human3R 与该方向的区别。
- Method 先给总览，再解释 prompt、人体读出、跟踪及训练。主方法图位于方法展开早期，而不是方法结束后。
- Experiments 先按任务说明验证顺序，再分 local human、global human、generic 3D reconstruction 和 analysis。主表集中在一页，定性图与通用 3D 结果随后出现。
- 写作特点：核心能力反复用相同但非工程化的概念表达；结果段明确指出“哪一个模块改变哪一类能力”。

### 2.2 JOSH

- Introduction 用“分别优化无法利用人—场景相互约束”建立唯一主矛盾，全文都围绕 joint optimization 展开。
- 为节省正文，详细 Related Work 放在附录；正文 Preliminaries 只保留理解目标函数所需的上游表示。
- Method 从问题定义、初始化、联合目标到端到端蒸馏版本，呈现严格的输入—变量—目标—输出链路。
- Experiments 先用一个主数据集同时验证联合任务和消融，再分别扩展到人体运动和场景重建；避免每个小数据集重复一整套叙述。
- 写作特点：方法名不带实验配置后缀；变体只在消融表中出现；每张表前给问题，每张表后给结论。

### 2.3 StreamVGGT

- Introduction 对比 offline 全局注意力与 online memory 两类范式，随后给出因果注意力、历史 token 缓存和蒸馏三个对应贡献。
- Method 篇幅约四页，方法图之外另有一张“如何缓存和增量推理”的机制图。关键工程成本被转化为模型机制和复杂度论述，而不是运行脚本描述。
- Experiments 首先给实现和训练集，然后按 3D reconstruction、depth、camera pose、analysis 排列；相邻主表共享一页，减少解释重复。
- 写作特点：表格列中明确区分 `Pair-wise`、`Dense-view`、`Streaming`，先声明执行范式再比较数值。这一点对 BRIDGE3R 区分 causal 与 offline baseline 很有参考价值。

### 2.4 TTT3R

- Introduction 篇幅较长，用实测的长序列退化和显存曲线建立问题，再提出两个明确研究问题。
- Method 先重写基础状态更新，再推导闭式更新率，最后解释置信度控制；公式服务于一个中心观点，不为形式完整而堆叠。
- Experiments 使用多张曲线展示输入长度变化下的精度、速度和显存，而非只给单点结果；附录承接更多长度和 reset 分析。
- 写作特点：定量主张与图表直接绑定，例如“2×”只在对应定义和实验支撑明确时使用；limitations 不混进每个结果段。

### 2.5 WinT3R

- Introduction 在一页内完成任务、两类既有方法的权衡、两个观察、两个模块和贡献清单。
- Method 按 window mechanism、point-map/camera prediction、training objective 组织，结构与贡献清单一一对应。
- Experiments 将主 benchmark 表集中在 p7，将定性图集中在 p8，将深度与消融集中在 p9，读者可以快速找到主张对应的证据。
- 写作特点：使用稳定的论文概念名，避免诸如 `adapt`、`v1`、`test run` 等实现过程名称。

## 3. 五篇论文的共同规律

1. **只有一个中心矛盾。** Introduction 不平均介绍所有能力，而是反复强化一个失败模式，并让每个方法模块与该失败模式一一对应。
2. **第一页必须能独立讲清任务。** teaser 不是装饰图，而是同时表达输入、原方法失败、核心机制或最终输出。
3. **方法通常占约三页。** 方法总览图出现在方法前半部分；正文只保留理解贡献必需的训练与公式，超参数和执行细节放补充材料。
4. **实验从“总体”到“原因”。** 先主结果，再压力条件或规模变化，最后消融；不会先用大量 protocol 细节阻断主结论。
5. **图表以问题为单位。** 主表回答“是否更好”，曲线回答“何时更有优势”，消融回答“为什么有效”，定性图展示“误差表现在哪里”。
6. **执行范式必须显式。** online、offline、streaming、pair-wise 等标签与数值同表出现，避免将不同信息条件下的结果解释为严格同条件排名。
7. **正文结论短而集中。** 失败案例、完整协议、额外数据集、所有可用性账本和大部分实现细节进入 supplement。
8. **学术措辞以可证实动词为主。** 常用 `formulate`、`estimate`、`preserve`、`improve`、`evaluate`、`observe`；较少使用 `pipeline version`、`final config`、`rerun`、`successful job` 等工程过程词。

## 4. BRIDGE3R v024 现状核对

### 4.1 当前页面分布

| 页码 | v024 内容 |
|---:|---|
| 1 | Abstract；Introduction 开始 |
| 2 | teaser（Figure 1）；Introduction 结束；Related Work 开始 |
| 3 | Related Work 结束；Method/problem formulation 开始 |
| 4 | 方法总览（Figure 2）；coarse gauge |
| 5 | gauge learning；causal execution |
| 6 | association/translation 结束；Experiments/protocol 开始 |
| 7 | 主结果表（Table 1）；消融表（Table 2） |
| 8 | boundary、multi-cut、AIST、angle analysis；角度图（Figure 3） |
| 9 | Discussion；Conclusion；Reproducibility/Ethics/AI statements |
| 10 | References 开始 |

正文源文件约含：Abstract 186 词、Introduction 449 词、Related Work 334 词、Method 1,445 词、Experiments 1,188 词、Discussion 290 词、Conclusion 92 词、statements 145 词。这个比例说明 **v024 的 Method 并不少**：它从 p3 延续到 p6，约占正文三页以上，与五篇参考论文相当。真正的问题是实验层级较拥挤，而不是简单增加 Method 字数。

### 4.2 已符合参考论文习惯的部分

- “历史证据与未来状态所有权冲突”已经构成清楚的单一主线。
- coarse gauge、clean state、prediction-only association、shared translation 形成了相对完整的模块链路。
- 方法总览图位于 Method 前半段，位置正确。
- 主实验、消融、角度压力测试、multi-cut 已形成从总体到机制的基本证据链。
- 正文九页、参考文献第十页开始；补充材料与正文保持同一 PDF。
- causal 与 offline 的区别、coverage 与 conditional geometry 的区别已经明确提出。

### 4.3 v025 应优先修正的结构问题

1. **Abstract 过度承担审计说明。** v024 在摘要中同时写“supported comparisons”“保留恶化指标”“区分 public offline baseline”等限制，压缩了核心机制与主结论。摘要应保留任务、方法、因果约束和跨三个多人基准的核心结论；详细的 metric-specific trade-off 放 Experiments/Discussion。
2. **Introduction 的结果钩子不足。** 当前贡献清单可靠，但在进入 Related Work 前缺少一段简洁、定量或强定性的核心发现，例如总体视角与极端视角的收益差异。
3. **主表承担了过多异构职责。** Table 1 同时放严格配对、多种外部方法、coverage、geometry support 和不同 ATE 口径，表注很长。v025 应让正文主表只回答“三个多人数据集上，同输入 Human3R 与 Bridge3R 的变化”，外部方法的完整 availability-aware 比较移入 supplement。
4. **极端视角优势尚未成为独立主证据。** 当前只在段落和单一 EgoHumans 图中呈现。根据已确认的论文主线，正文应增加三个多人数据集的等权总体汇总和 extreme-view 专项表/图。
5. **Experiments 第 8 页主题过多。** boundary association、multi-cut、AIST、角度趋势和定性入口在同一页连续出现，读者难以分辨主证据与补充证据。AIST 完整表、multi-cut 明细、external availability 和完整 CI 应下沉 supplement。
6. **Figure 3 只覆盖 EgoHumans。** 若正文主张是“三个多人数据集在极端视角收益最明显”，图或配套表必须覆盖三个数据集，不能让单数据集曲线替代跨数据集证据。
7. **Discussion 偏防御性。** 需要保留 Harmony4D W 与 AIST 平移 trade-off，但应先解释稳定优势对应的方法机制，再用一个独立段落界定剩余误差；不要在主结果段反复插入审计语言。
8. **图像任务需要分工。** teaser 负责概念，method figure 负责机制，qualitative figure 负责具体失败模式；不能让角度统计图同时代替真正的定性对比。

## 5. v025 的九页可执行预算

下面的预算以“参考文献从第 10 页开始”为硬约束。LaTeX 浮动会造成约 0.2–0.4 页波动，因此建议正文实际内容控制在约 8.7–8.9 页，不通过缩字号、负间距或超长 `resizebox` 强塞。

| 页码 | 目标内容 | 预算与目的 |
|---:|---|---|
| 1 | Title、Abstract、Introduction 前半、Figure 1 teaser | Abstract 150–180 词；首段建立 hard viewpoint cut；teaser 直接显示同场景两镜头、Human3R 失败与 Bridge3R 保持一致性的结果 |
| 2 | Introduction 后半、贡献、Related Work 前半 | 用一段核心结果完成动机闭环；贡献保持三点；Related Work 从 streaming state 与 global human reconstruction 两条线展开 |
| 3 | Related Work 结束；Problem formulation；state/evidence distinction | 只保留与“为何继续状态和清空状态都不够”直接相关的文献；给出因果输入、输出和不可修改历史的定义 |
| 4 | Figure 2 method overview；learned correction-token coarse gauge | 全宽可编辑矢量图；图内明确 shadow/read-only 与 clean/committed 两条分支，避免代码模块名 |
| 5 | Event-conditioned learning；causal invariants；为什么小参数分支有效 | 保留训练目标和因果 transition 两个核心公式；参数高效性的解释应落在“只估计低维跨镜头 gauge residual，而非重学人体/场景先验” |
| 6 | Association/shared translation；Experiments setup | 方法收尾约半页；实验设置只保留数据集规模、公平输入、核心指标与运行范式，完整 evaluator/denominator 进 supplement |
| 7 | Table 1 三个多人数据集严格配对主结果；Table 2 总体与极端视角汇总；解释 | Table 1 只含 Human3R/Bridge3R；Table 2 展示三数据集等权汇总和 extreme-view 增益；正文首先回答“整体是否有效、何时最有效” |
| 8 | Figure 3 大视角主观对比；Figure 4 角度趋势或紧凑曲线；机制消融摘要 | 主观图选择有真实 RGB、相机、背景和多人重建的相同序列，标注错误匹配/方向漂移/相机 gauge；角度图不得写成单调收益，只写 extreme 最大 |
| 9 | Table 3 核心消融；boundary/multi-cut 一段；Discussion、Conclusion、statements | 消融只保留 strict、clean reset、coarse gauge、association、full；multi-cut 给一句核心量化并指向 supplement；Discussion 先机制解释，再界定 trade-off |

### 正文建议保留

- 三个多人数据集上的严格同输入 Human3R–Bridge3R 结果。
- 三数据集等权总体汇总与极端视角专项结果。
- Coverage，避免收益被解释为丢弃困难人物。
- 一个主方法图、一个大视角定性图、一个跨数据集角度趋势图。
- 核心模块消融，以及一段简洁的 repeated-event/multi-cut 证据。
- Harmony4D 的 W-MPJPE trade-off，但只在结果和 Discussion 各解释一次。

### 建议移入 Supplement

- TRACE、PromptHMR、GVHMR 的完整 availability-aware 表和准入说明。
- AIST++ 的完整 CS150、MC150-3、MC150-4 表；正文至多保留一句弱纹理单人压力测试结论。
- 完整置信区间、denominator ledger、topology conversion、acceptance rules。
- ATE-Sim3/ATE-SE3 双口径完整表、所有 boundary pair 明细和 failure taxonomy。
- 全部超参数、数据构造、训练细节和额外敏感性实验。

## 6. v025 推荐主线叙事

### 一句话主张

> A hard same-scene viewpoint cut requires historical evidence to recover the old coordinate gauge, but requires a clean recurrent state to prevent view-specific contamination; Bridge3R separates these two roles and causally reconnects cameras, people, and identities across the cut.

### 叙事顺序

1. **失败模式：** Human3R 在连续、纹理充分、小视角变化时可以维持稳定状态；大视角硬切使旧视角状态与新观测不兼容，容易产生相机 gauge、人体全局位置和身份错误。
2. **关键矛盾：** 继续旧状态会污染新 shot；直接 reset 会丢失与旧世界坐标系的关系。
3. **核心洞见：** 历史可以作为一次性的 gauge evidence，但不能成为新 shot 的 persistent state owner。
4. **方法对应：** correction-token shadow 只估计 coarse gauge；clean branch 独占未来状态；预测量 association 和 shared translation 完成精对齐。
5. **为何参数小仍有效：** 分支不重新学习人体、场景或相机的通用表征，只学习低维、事件局部的坐标残差；冻结 backbone 保留原有空间与人体先验。
6. **主实验证据：** 三个多人数据集的同输入比较显示整体收益；极端视角收益最大，直接对应方法动机。
7. **机制证据：** clean reset 单独不足，coarse gauge 恢复相机关系，association/translation 进一步恢复人物与身份连续性。
8. **边界与诚实结论：** 方法改善的是跨镜头 camera–human coherence，而不是替换局部人体估计器或完成 dense scene reconstruction；Harmony4D W 和 AIST 平移仍存在 metric-specific trade-off。

## 7. 图表安排与制作要求

### Figure 1：任务与效果 teaser（p1–p2）

- 左侧：同一场景两个相机的 RGB 画面，标注较大相机夹角。
- 中间：Human3R 在切镜后出现相机方向/人体全局位置/身份不连续；只标 2–3 个最显著错误。
- 右侧：Bridge3R 保持相机和人物在旧 gauge 中一致。
- 图像必须来自固定评估序列和实际方法输出；不得由生成模型伪造实验结果。生成工具只可用于箭头、布局、背景框和图例，不能生成看似真实的定量重建结果。
- 保留 SVG/PDF 矢量源；RGB 和渲染帧作为嵌入的真实 raster 面板。

### Figure 2：方法总览（p4）

- 上路：historical state → read-only shadow → correction tokens → coarse gauge $T_0$，末端明确“discard state”。
- 下路：empty state → clean reset → committed recurrent state。
- 合流：prediction-only association → shared translation → aligned camera and people。
- 用颜色区分 evidence、state、output-space transform；图中直接标注“no future frames”“no history rewrite”。

### Table 1：三数据集主结果（p7）

- 每个数据集只放 Strict Human3R 与 Bridge3R 两行。
- 列保留 W、WA、dataset-specific Camera ATE、IDF1、Coverage；在方法名或表注中清楚说明 ATE 口径。
- 不在该表加入 TRACE/PromptHMR 的稀疏 support 数值，避免将不同输出接口误读为共同排行榜。

### Table 2 / Figure 4：总体与极端视角（p7–p8）

- 表格给三数据集等权汇总：全视角 W/WA 相对降低 9.5%/8.8%，极端视角降低 19.1%/18.7%，IDF1 增益从 +0.099 增至 +0.195；不汇总 ATE。
- 图形给每个数据集的 all/extreme 配对差异或相对变化，避免暗示收益随角度单调增加。
- Harmony4D extreme 的 W 近似持平（约 −0.4%），仍须显示；其 WA 和 IDF1 改善是更可靠的证据。

### Figure 3：真实大视角主观对比（p8）

- 对每个被选方法使用同一序列、同一帧和同一观察视角。
- 至少显示 input RGB、camera trajectory/frusta、带背景的 reconstruction、局部放大框。
- 标注应对应可见事实，例如 wrong identity transport、orientation flip、global offset 或 camera drift；不要用主观的 “bad/good”。
- caption 必须写清 sequence selection rule，避免只凭肉眼挑一个有利案例却不披露选择标准。

### Table 3：核心模块消融（p9）

- Oracle timing 只能标为 diagnostic/control，不进入 baseline 排名。
- 推荐行：strict streaming、clean reset、learned coarse gauge、+ association、full Bridge3R。
- 若 correction-token 微组件没有独立重训结果，不在正文声称每个 token 的独立增益；完整变体以后放 supplement。

## 8. 学术写作执行清单

- 用 `same-scene viewpoint cut`、`read-only shadow branch`、`clean future-state ownership`、`coarse gauge`、`prediction-only association` 和 `shared translation` 作为稳定术语。
- 不在正文使用版本号、实验目录名、脚本名、`adapt`、`final config`、`rerun`、`test output`、`method-v1` 等工程语言。
- 每个 subsection 首句说明它回答的问题；末句说明与主张的关系。
- 每个实验段遵循“比较条件 → 数值 → 解释 → 范围”顺序；不要在数值前堆叠大量免责声明。
- 只有严格配对的结果使用 `outperforms` 或 `improves over`；异构接口使用 `executable reference`、`offline reference` 或 `reported separately`。
- `extreme viewpoint` 只写“收益最大/更显著”，不写“随视角单调增加”。
- `low texture` 在没有定量纹理分层前写成 `weakly textured studio setting` 或 `visually weak-texture stress test`，不要称为正式 low-texture benchmark。
- Abstract 中不必堆准确数值；正文 Introduction/Experiments 给出数字，并确保与表格完全一致。
- caption 可以自包含，但避免把 protocol ledger 全部写入 caption；复杂口径用正文一句和 supplement 交叉引用解决。
- Discussion 只保留对机制有解释价值的 trade-off，完整失败类型放 supplement。

## 9. v025 编译验收标准

1. 正文页码 1–9；第 10 页首个编号章节不得仍为正文，References 应从第 10 页开始。
2. 单一 PDF 包含正文、References 和 Supplement；figure/table 编号跨 Supplement 连续。
3. `main.log` 无 undefined citation/reference、fatal error、overfull box。
4. 所有 `TODO`、占位框、虚构图像和未核验数字在正式投稿版清除；草稿若暂留 TODO，应在 README/CHANGELOG 明示。
5. 所有图提供可编辑 SVG 或等价矢量源和 PDF；真实输入/重建面板保留来源清单。
6. 主表由事实账本或脚本生成，Harmony4D 只保留正式 88-case 口径，旧冲突表不进入 Overleaf ZIP。
7. 主结果、极端视角表、Abstract、Introduction 和 Conclusion 中的文字主张逐项与相同数值口径核对。

## 10. 总结

v024 的九页结构已经接近五篇 ICLR 参考论文的成熟形态，Method 篇幅并不短。v025 不应通过继续堆字来“补方法”，而应重排证据层级：让三数据集严格配对主表和极端视角结果成为正文中心，让真实大视角定性图直接展示问题，把 AIST、外部方法可用性、完整置信区间和协议账本移入 Supplement。这样既能保持事实边界，也能把 BRIDGE3R 的核心贡献从“复杂实验工程”转化为清楚的学术论点：在大视角同场景硬切中，将历史 gauge evidence 与未来 recurrent-state ownership 分离。
