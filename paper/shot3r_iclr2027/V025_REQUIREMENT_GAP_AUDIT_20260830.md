# BRIDGE3R v025 需求差距审计（2026-08-30）

## 1. 审计范围与结论

本报告只读对照了以下材料：

- `BRIDGE3R_v021_ICLR2027_事实核验版_系统性修改主提示词.md`；
- `BRIDGE3R_ICLR2027_后续实验执行计划_20260829.md`（含 2026-08-30 新增的第 8--22 节）；
- `versions/v024_20260829_evidence_closure/manuscript/` 中的正文、Supplement、全部表格源文件、图源、编译日志、辅助文件和现有 PDF；
- v024 Overleaf ZIP 的实际文件清单。

审计未修改 v024，也未创建论文新版本。

总体判断：v024 已经完成了任务范围、因果合同、三套多人主协议、90/88 分母、标准 runtime、correction-stage overlap audit、关联审计和单 PDF 编译闭环；但它尚未实现最新计划中最重要的论文重排。最关键的缺口不是再扩写一般性文字，而是：

1. 三数据集总体与极端视角证据尚未进入正文；
2. Harmony4D 旧的冲突表仍留在 v024 源树中；
3. AIST++ 在正文占比仍过高；
4. 现有 Figure 1/2 是抽象示意，Figure 3 是单数据集曲线，没有真实 RGB、人体、相机和场景的主观比较；
5. Method 图没有画出 correction tokens、residual heads、LoRA、$T_0$ 和共享平移的实际结构；
6. 引用、外部代码状态、Ethics 与 AI-use 披露仍不完整；
7. 公开 Supplement 仍包含只能称为 inference-time masking 的细粒度 token/head 表，与“独立重训练前只保留内部证据账本”的最新计划存在冲突。

因此，v025 可以立即开始，但不能把 v024 直接改名为新版。应从 v024 复制后，按机器生成表格、真实图像、用词清理和新 ZIP 白名单重新构建。

---

## 2. v024 已满足且应保留的内容

### 2.1 科学范围与因果合同

- 标题准确限定为 same-scene viewpoint cuts，没有扩展到 arbitrary editing 或 dense scene reconstruction。
- 正文明确只允许读取 pre-cut 历史和第一张 post-cut 图像，不允许未来帧。
- shadow branch 是 read-only evidence，clean-reset branch 是后续 recurrent state 的唯一来源。
- 已说明 no-cut fallback、no-association fallback、prefix immutability 和 repeated-cut composition。
- 主文没有声称 real-time、backbone-agnostic、universal 或 all-metric dominance。

### 2.2 三个多人主协议

- EgoBody：129 cases / 43 recordings；
- EgoHumans：90 clips / 27 captures；
- Harmony4D：88 cases / 25 captures / 7 archive groups；
- 主表中的 Harmony4D 正式结果是正确的统一口径：`496.6/259.6/0.1420 -> 519.6/247.0/0.0168`（W/WA/ATE-Sim3）。
- EgoHumans 正文和公开方法表已经使用同一 90-clip 集合，不再混用 90 与 116 的主表分母。
- TRACE / PromptHMR 的 conditional geometry 与 full-denominator Coverage 已分开解释。

### 2.3 已完成的证据闭环

- 标准 reconstruction runtime / peak VRAM 已实测并清楚披露 detector inference 未计入；
- correction-module fine-tuning 与三套多人评估集的 source/capture/event/frame-member overlap 为 0，并且没有把该结论外推到 inherited Human3R pretraining；
- EgoHumans 与 Harmony4D 的 direct association audit 同时报告 conditional accuracy、evaluator-excluded pairs 和 continuation coverage；
- Harmony4D multi-cut 明确保留 ATE-Sim3 的负结果；
- AIST++ 完整保留 causal/offline 与 anchored/orientation trade-off；
- 当前使用的 PDF 无 undefined reference、fatal error 或 overfull box。

### 2.4 当前 PDF 的版面事实

- 当前已编译 PDF 共 24 页；
- 正文占第 1--9 页；
- References 从第 10 页开始；
- Supplementary Material 从第 11 页开始；
- Figure 编号为 1--3，Table 编号为 1--30，正文和 Supplement 编号连续；
- PDF metadata 的 Author、Title、Subject 均为空，没有直接身份泄露；
- 页面为 US Letter，使用 `iclr2027_conference.sty`，匿名行号存在。

这些是 v025 的最低回归标准。任何新改动都必须在 pdfLaTeX/Overleaf 目标引擎下重新核验，不能只依赖另一 TeX 引擎，因为字体与分页可能不同。

---

## 3. 可立即修改：v025 的必做项

## 3.1 正文主证据重排（最高优先级）

### 当前差距

v024 正文只有三数据集原始主表，并且正文角度图只展示 EgoHumans。最新计划已经锁定的以下证据均未进入正文：

- 三数据集等权 all-view macro；
- 三数据集 dataset-specific extreme/farthest macro；
- 极端视角逐数据集原始 W、WA、dataset-specific ATE、IDF1、Coverage；
- all-view 与 extreme/farthest 的紧凑比较图；
- 极端视角下收益最明显、但并非随角度单调增长的准确结论。

### 必须加入

1. 保留现有三数据集原始主表；
2. 新增 extreme/farthest-viewpoint 主表，严格使用：
   - EgoBody：每个 recording 的 farthest camera pair；
   - EgoHumans：`>=150°`，22 captures；
   - Harmony4D：预定义 extreme 档，25 captures；
3. 新增三数据集等权 macro 行：
   - all-view W reduction：`9.5%`；
   - extreme/farthest W reduction：`19.1%`；
   - all-view WA reduction：`8.8%`；
   - extreme/farthest WA reduction：`18.7%`；
   - IDF1 gain：`+0.099 -> +0.195`；
   - 不汇总 ATE；
4. Coverage 必须保留，不能只展示成功样本上的几何误差；
5. Harmony4D extreme W 只能写为基本持平，不能标显著改善；
6. 正文写“most pronounced under extreme viewpoint changes”，不能写“monotonically improves with viewpoint angle”。

### 数据产生要求

新表必须由当前 case-level 结果和固定 manifest 重聚合，不允许手工复制本计划中的汇总数字。Supplement 需给出 paired gain、95% CI、win/tie/loss 和外部方法的 finite support / full-denominator Coverage。

## 3.2 Abstract、Introduction 与 Conclusion

### Abstract

当前 Abstract 的 `supported comparisons` 是审计式且含混的措辞；`post-shot` 也与全文锁定的 `post-cut` 不一致。应：

- 加入已锁定句子：`Across three multi-person benchmarks, the gains are most pronounced under extreme viewpoint changes.`；
- 不在 Abstract 写 `roughly doubles` 或宏平均的具体百分比；
- 保留 Harmony4D W 的 trade-off，但用直接指标名而不是 `complementary global error`；
- 将 `post-shot` 全部统一为 `post-cut`；
- 不写 `supported comparisons`、`retained result` 等内部审计语言。

### Introduction

应加入：

- 三数据集等权 all-view 与 extreme/farthest 的主要数字和宏平均定义；
- 明确四个实验问题：coarse gauge、identity、shared translation、causal invariants；
- 参数效率事实：只更新 `34,415,333 / 1,232,459,373 = 2.79%` 参数；
- 避免把 correction tokens 写成已经被独立重训练消融验证的贡献。

### Conclusion

当前结论只写 EgoBody/EgoHumans 与 Harmony4D trade-off，尚未总结三数据集的极端视角规律。应加入定性结论，但仍保留 Harmony4D W 和 AIST 受限结论，不扩大到 arbitrary edits。

## 3.3 AIST++ 从正文降级为一句迁移结论

最新计划已锁定：AIST++ 完整表只放 Supplement，正文只保留一句受限结论。v024 仍有一个完整的 `Single-person cross-view stress test` 小节，占用了正文第 8 页的大量篇幅。

v025 应：

- 删除正文中的完整 AIST 数字段、MC150-3/4 段和 GVHMR pilot 细节；
- 在 Protocol 或 Discussion 中保留一句：弱纹理棚拍条件下 orientation / relative-camera rotation 改善，但 anchored/root geometry 并非全面领先；
- AIST CS150、MC150-3、MC150-4、PromptHMR offline 区别和 GVHMR pilot 全部移至 Supplement；
- 在尚无客观纹理分数前，不称其为正式 `low-texture benchmark`。

## 3.4 Method 的学术化与细节闭环

### 需要立即补充

- 将 `A single recurrent forward pass cannot enforce both properties` 改为更严谨的“conventional single-state update does not explicitly separate ...”；
- 在正文 Method 报告 2.79% 可训练参数及其机制解释：基础模型已经提供 RGB-to-3D 表征，新分支只学习 event-conditioned low-dimensional camera--human gauge/residual correction；
- 用 robust shrinkage 形式解释共享平移：
  \[
  \Delta^*=\arg\min_\Delta \sum_i\rho(\Delta-d_i)+\beta\|\Delta\|_2^2,
  \]
  并说明固定 `lambda=0.5` 是 observation 与 no-update prior 的折中，而非所有指标上的唯一最优值；
- 补清 token 的插入位置、attention flow、camera residual、auxiliary human residual、LoRA 和 shadow discard；
- 明确 camera-to-world 左乘约定、camera centre 更新、scene pointmap 不进入定量 correction；
- 明确 unequal person counts、unmatched tracks、ambiguity/admissibility 和 multi-cut transform composition；
- 把 `post-shot` 统一为 `post-cut`。

### 必须先与代码核对的公式冲突

事实核验文档把 association 代价锁定为 `joint + alpha * torso`，而 v024 Eq. (5) 写成等权的 `root + torso + joint`，同时 Experiments 又报告 `alpha=1`。这三种表述不能同时作为正式事实。v025 写作前必须对照运行代码确定：

- 是否真的有 root term；
- `alpha` 乘在哪一项；
- 三项如何归一化；
- 是否有 admissibility threshold；
- assignment 后的 unmatched 行为。

这是方法--代码一致性问题，不能仅靠润色解决。

## 3.5 主文加入紧凑效率证据

最新写作决定要求正文展示 Strict Human3R 与 BRIDGE3R 的同硬件增量成本。v024 只在 Supplement 有 runtime 表。

正文至少应写：

- 同一 100-frame EgoHumans clip、L20、512、FP32、batch 1；
- Strict：31.020 s；BRIDGE3R single cut：32.191 s；
- 相对 no-cut BRIDGE3R 的单次切换增量为 1.066 s / 100 output frames，即 3.42%；
- detector inference 未包含，因此不得称 end-to-end real-time；
- peak VRAM 可称 comparable，不称更低。

详细 timing boundary、warm-up 和 I/O 排除仍放 Supplement。

## 3.6 Related Work 与 protocol context

v024 Related Work 只有三个段落，且缺少多个已经要求的相关方向。应重构为：

1. stateful/streaming reconstruction；
2. global human motion across shots；
3. multi-person tracking across shots；
4. joint humans--scenes--cameras；
5. shot-boundary detection and association；
6. protocol positioning。

至少补入 ShowMak3r、TROPHIES、HSfM、UniCon3R，并准确区分：

- HumanMM：官方仓库存在，但可执行 processing/baseline 未发布；
- Multi-THuMBS：检查到的官方项目页无代码链接；
- ShowMak3r：有代码，但为 offline compositional reconstruction；
- TROPHIES：检查的官方来源未发现作者链接的可执行代码；
- HSfM：有 demo code，但为不同的多阶段/多视图协议。

Supplement 的 protocol table 目前只有 4 列和 3 个工作，必须增加 code status、checked date、causal/future-frame、multi-person、camera、identity、global optimization 等字段。删除 `manufacture a paired ranking` 这种防御性措辞。

## 3.7 引用审计

v024 虽然 BibTeX 中已有 EgoBody、EgoHumans、Harmony4D 等条目，但正文没有实际引用这些数据集论文；AIST++、AvatarReX、THuman、MVHuman、Real-World-Capture、LoRA、IDF1、Sim(3)/Umeyama、bootstrap/sign-flip、shot-boundary detection 等条目或正文引用仍缺失。

必须完成：

- 每个数据集第一次出现时引用原始论文；
- GRU、LoRA、Hungarian/assignment、alignment、IDF1 和统计方法有出处；
- 外部方法的 venue/year/project/code 状态逐项复核；
- 删除未使用或无法核验的 BibTeX；
- 不用二手页面支持 code availability。

## 3.8 Ethics、隐私与 AI Use

v024 Ethics 只列出 EgoBody、EgoHumans、Harmony4D，遗漏：

- AIST++；
- correction-module fine-tuning 使用的 AvatarReX、THuman、MVHuman100、MVHuman200、Real-World-Capture/MultiHuman；
- 新增 MVHuman evaluation；
- 内部采集或授权数据的使用条件。

v025 应对每个来源说明 public/licensed/internal、research-only 条款、可识别人物、同意/原数据协议、存储和发布边界。

AI Use 当前披露 editing、planning、analysis 和 code scaffolding。由于新需求明确使用 AI 辅助图形设计，最终版本还应披露 scientific-figure assistance，并强调实验 RGB、重建结果、数字和标注均由真实结果产生并由作者核验。

---

## 4. 图像与可视化：当前不满足新需求

## 4.1 Figure 1

当前 teaser 是人工绘制的三栏抽象流程图，没有真实 RGB、模型重建、相机或场景。它不能作为用户要求的主观比较图，也不符合事实核验文档提出的“真实 teaser”。图中文字仍有 `Causal commit`、`future-state owner`、`validated state` 等工程化表达。

## 4.2 Figure 2

当前 Method 图是通用的四步流程图，缺失核心 learned contribution：

- semantic/alignment/temporal correction tokens；
- frozen backbone；
- camera/human residual heads；
- LoRA；
- shadow state discard；
- clean future recurrence；
- $T_0$；
- association；
- shared `Delta`；
- camera/human/ID 的输出空间更新。

图中还存在 `Causal commit`、`validated update`、`Commit to clean state` 等工程化用语。渲染图中部分长句靠近或越过面板边缘，可读性不足。

## 4.3 Figure 3 与未使用 qualitative 文件

- 正文 Figure 3 是 EgoHumans 的 W/IDF1 角度曲线，不是 qualitative comparison；
- `qualitative.svg/.png/.pdf` 没有被论文引用；
- 该文件自称 `Protocol schematic, not a rendered reconstruction`，并含 `locked transaction`、`causal commit`；
- 另一个未使用的轨迹图同样没有 RGB、mesh、GT 或背景。

## 4.4 v025 图像要求

建议按以下方式完成，且不得让生成式工具虚构实验效果：

1. Figure 1 使用真实 EgoHumans 或 Harmony4D 大视角成功案例：pre-cut RGB、post-cut RGB、Strict Human3R、BRIDGE3R、camera frusta、human meshes、persistent-ID colours、场景/点云背景和关键指标；
2. Figure 2 用 SVG 重新绘制正式方法结构；
3. Figure 3 改为 all-view 与 extreme/farthest 的三指标宏比较，或真实主观比较；完整逐角度曲线放 Supplement；
4. 至少一张 Supplement failure case 保留 Harmony4D absolute-placement trade-off；
5. RGB 和实验 render 可以作为 raster panel 嵌入 SVG，但布局、箭头、文字、标注和 camera frusta 必须可编辑；同时交付 `.svg` 源与用于 LaTeX 的 `.pdf`；
6. 图像生成工具只能协助构图、视觉样式和非证据性图标，不能生成或补画作为实验结果的 mesh、camera、GT 或 background；
7. 新 Overleaf ZIP 当前只包含 PDF 图，不包含 SVG 源。v025 必须把可编辑 SVG 一并放入包内。

---

## 5. 事实冲突与高风险旧文件

## 5.1 Harmony4D 旧数值冲突

v024 当前 PDF 使用正确的 88-case 主结果，但源树中仍存在未引用的旧文件：

- `artifacts/harmony4d_final/harmony4d_five_method_table.tex`：BRIDGE3R 为 `584.0/266.9/0.019`；
- `artifacts/harmony4d_final/harmony4d_angle_table.tex`：extreme W 为 `550.2 -> 689.3`；
- `artifacts/harmony4d_final/harmony4d_action_table.tex`；
- 对应的旧报告使用 100-case 外壳和 conditional 88-case 口径。

它们与正式主表 `519.6/247.0/0.0168` 以及最新计划锁定的 extreme/farthest 结果不一致。虽然 v024 PDF 和 v024 Overleaf ZIP 没有引用这些旧文件，但 v025 源目录、ZIP 和公开证据包必须使用白名单，不得复制它们。

新版五方法总体/角度表需从统一 88-case manifest 和 case-level 输出脚本生成，并同时显示 full denominator、inference failure、finite W/WA support、IDF1 和 Coverage。

## 5.2 训练数据来源命名不一致

Supplement Training configuration 写第五个来源为 `Real-World-Capture multi-person data`，但 overlap audit 随后称其为 `MultiHuman candidates`。必须确认二者是否为同一数据源/内部别名，并在论文中只使用一个可公开、可解释、匿名且与许可一致的名称。

## 5.3 Association 公式冲突

如第 3.4 节所述，事实文档、v024 公式和 `alpha=1` 的含义不一致。需由实际运行代码裁决。

## 5.4 Camera metric 叙述风险

主表对 EgoHumans 报 ATE-SE3（`1.673 -> 1.423`），但正文 paired-statistics 段落转而报告 ATE-Sim3 gain `0.515`。v024 已称其为 scale-normalized diagnostic，但两者相邻出现仍容易被误读。

v025 应：

- 主表和主叙述坚持 EgoHumans ATE-SE3；
- Supplement 同时给三数据集 ATE-Sim3 / ATE-SE3、单位、support 和对齐自由度；
- 若引用 paired ATE-Sim3 CI，明确它不是主 camera score；
- 不把 Sim(3) 的大幅改善写成 metric-scale camera accuracy 的同等改善。

## 5.5 Detector “first positive” 与 repeated cuts

正文一处写 detector 只调用 `its first positive proposal`，另一处写每个后续 proposal 都重复操作。应改为“每个尚未处理的 boundary event 的首个正例触发一次”，避免被理解为整段视频只能处理第一个 cut。

---

## 6. 工程化措辞清理

### 6.1 当前公开正文/Supplement 中需要替换

- `supported comparisons`：直接写具体数据集和指标；
- `manufacture a paired geometric ranking`：改为 `we report them as availability-aware references rather than paired same-input comparisons`；
- `post-shot`：统一为 `post-cut`；
- `state owner` / `commit`：改为 `sole source of subsequent recurrent states` / `propagate only the clean-reset state`；
- `retained`, `fixed`, `pre-registered`, `audit`：仅在科学上确需说明预先固定或追溯时保留，避免每段重复；
- `Test consumption`：改为 `before final test-set evaluation`；
- Plain `Bridge3R`：统一使用 `\method{}`，显示为 BRIDGE3R。

### 6.2 必须从 v025 包中排除的未引用工程文件

- `egobody_v20/recording_macro_main.tex`：含 `B0`、`runtime parent`、`geometry parent`；
- `harmony4d_final/*` 旧表：含 `ours; sealed`、`no-SPEC adapter` 和冲突数值；
- 旧 `qualitative.*`：含 `locked transaction`；
- 内部 README 中的 Movie3R 路径、V14/V19/V20、checkpoint 内部命名；
- 任何绝对路径、用户名、服务器路径和内部 hash。

当前 v024 Overleaf ZIP 已使用文件白名单，未包含大部分旧文件；v025 应继续使用白名单构包，而不是压缩整个 manuscript 目录。

---

## 7. 依赖 MVHuman 新结果的项目

以下内容不能在看到新数据和正式结果前写成事实，但可以先搭建协议、表格和图位置：

1. `Mvhuman-test1.zi` 与 `Mvhuman-test2.zip` 的完整性、压缩格式、目录结构、RGB、相机内外参、人体 GT、身份和同步性核验；
2. correction-module fine-tuning 与新序列的 subject/capture/event/frame-member overlap audit；
3. 人体区域外背景纹理分数：梯度/Laplacian 的预定义、单位和阈值；
4. `<60°`、`60--90°`、`90--120°`、`120--150°`、`>=150°` 的固定角度分层；
5. 相同 RGB、cut、视角对、evaluator 和 denominator 下的 Strict Human3R、BRIDGE3R 与可执行 single-/multi-person references；
6. 整体、纹理分层、角度分层和 texture x viewpoint interaction；
7. case-level outputs、failure ledger、support、coverage、paired CI 和 machine-generated TeX；
8. 若 MVHuman 与 AIST 趋势一致，正文可升级低纹理证据；若不一致，只放 Supplement 并保留 mixed result；
9. 任何“held-out subject/capture/event”措辞必须由 overlap audit 决定，不能预写。

新 MVHuman 结果不应阻塞 v025 的三数据集 extreme/farthest 主线和写作清理；建议为它预留 Supplement 小节和一张可删除的正文短段，不预填数值。

---

## 8. 明确延后项

以下任务已由作者确认可延后，不阻塞当前 v025：

1. correction-token 变体的独立重训练；
2. 多 seed 或 full checkpoint 的独立重现；
3. 更大规模 natural edited-video / temporal-offset benchmark；
4. EgoBody direct association audit（原始数据恢复后再做）；
5. 第二个 recurrent backbone；
6. 更大规模 3/5/10-cut 长流实验；
7. detector 的完整困难负样本和 calibration benchmark；
8. 完整 inherited Human3R pretraining corpus provenance；
9. 最终匿名代码包和投稿前 OpenReview/author/reviewer 检查。

这些项目应在内部待办和 Limitations 中准确记录，但不应以醒目的 `TODO` 字样进入拟提交 PDF。用户允许工作稿保留 TODO，不等于正式 v025 交付可以把未完成结果写成表格事实。

---

## 9. 细粒度 token/head 表的决策冲突

最新计划第 12 节说明，当前 EgoHumans token/head 结果均来自同一 checkpoint 的 inference-time masking，不能称为独立训练的架构消融，并倾向在重新训练前只保留内部证据账本。v024 Supplement 仍公开展示：

- semantic-only / alignment-only / semantic+alignment；
- LoRA-off / camera-residual-off；
- complete-vs-mask paired sensitivity。

虽然 v024 已诚实称其为 sensitivity，并保留 full 不是最佳的负结果，但严格遵循最新计划时，v025 应二选一：

- **推荐**：从公开 PDF 移除三张细粒度 masking 表，只保留一句“inference-time sensitivity is retained internally; independently retrained variants are pending”，正文继续使用 EgoBody 系统级五行消融；
- 或保留在 Supplement，但标题必须明确 `frozen-checkpoint inference-time component masking`，不能使用 architecture ablation、necessity、independent contribution 等措辞。

这项不依赖 MVHuman 数值，但需要作者/主执行 agent 在 v025 集成时做一致选择。当前 v024 的“表格公开”与计划中的“只保留内部账本”不能同时视为最终状态。

---

## 10. ICLR 格式和发布风险

### 已通过

- 当前 pdfLaTeX 产物：正文 9 页，References 第 10 页，Supplement 第 11 页；
- letter paper、匿名页眉、连续编号；
- 无 undefined/fatal/overfull；
- 当前 PDF metadata 无作者身份。

### v025 必须重新检查

1. 新增 extreme 表、runtime 证据和真实图后，正文必须仍在 9 页内；
2. AIST 正文降级所释放的空间优先给极端视角表、主观图和 Method 细节；
3. References 不计 9 页，但必须从正文结束后的新页开始；
4. Supplement 与正文保持同一 PDF，figure/table counter 不重置；
5. 宽表不要依赖不可读的极小字体；当前 Supplement 多张 `scriptsize + resizebox` 表可进一步拆分；
6. Method 图当前存在小字和边界拥挤，必须按单栏/双栏最终尺寸检查；
7. Overleaf ZIP 必须在空目录、官方目标引擎下重编译；本地 Tectonic/XeTeX 的分页与 pdfLaTeX 不一致，不能作为唯一页数验收；
8. 提交前重新下载/核对 ICLR 2027 官方 style 和 Author Guidelines，不能只依赖 2026-08-28 的历史网页核验；
9. 重新扫绝对路径、用户名、Git remote、内部版本、checkpoint hash、图 metadata 和 PDF metadata；
10. ZIP 应包含可编辑 SVG 图源，但不包含 raw licensed RGB、非匿名 provenance 或内部运行路径。

---

## 11. 推荐的 v025 执行顺序

1. 复制 v024 到新的 v025 目录，不覆盖 v024；
2. 建立 v025 文件白名单，先排除旧 Harmony4D 表、旧 qualitative、工程化内部文件；
3. 用脚本生成三数据集 all-view/extreme 表、统计附表和 all-vs-extreme SVG；
4. 重排 Experiments：多人主表 -> 极端视角 -> EgoBody 系统消融 -> causality/efficiency -> association/multi-cut -> 一句 AIST；
5. 重写 Method 图和 Method 文字，先解决 association 公式与训练源名称冲突；
6. 用真实实验输出制作可编辑 SVG 主观图；
7. 扩充 Related Work、protocol status table 和引用；
8. 更新 Abstract、Introduction、Discussion、Conclusion；
9. 更新 Ethics、AI Use、Limitations；
10. 为 MVHuman 建立预定义协议和 Supplement 插槽；只有正式结果通过后再写数字；
11. 运行工程化用词、匿名、引用、label、浮动体、页数和数值一致性审计；
12. 在空目录使用目标 pdfLaTeX 流程编译，生成单 PDF 和新 Overleaf ZIP。

---

## 12. v025 接收门槛

- [ ] 三数据集原始主表与 extreme/farthest 主表均由脚本生成；
- [ ] all-view / extreme macro 定义和数值一致；
- [ ] Harmony4D 旧 `584.0/266.9/0.019` 文件不在 v025/ZIP；
- [ ] Abstract 含锁定的极端视角定性句，不含宏平均数字；
- [ ] AIST 正文仅一句，完整结果在 Supplement；
- [ ] 主文有参数效率和增量 runtime；
- [ ] association 公式与真实代码一致；
- [ ] Real-World-Capture/MultiHuman 名称一致；
- [ ] Figure 1 有真实 RGB/人/相机/背景；Figure 2 画出 learned modules；
- [ ] 所有图均提供 SVG 源和 PDF；生成式工具未虚构实验结果；
- [ ] Related Work 和 code-status table 完整且带核验日期；
- [ ] 数据集、模块、指标和统计方法引用完整；
- [ ] Ethics 覆盖训练、多人测试、AIST 和 MVHuman；AI Use 包含 figure assistance；
- [ ] 无 `supported comparisons`、`manufacture`、`locked transaction`、`sealed`、`parent`、`B0`、`Test consumption`；
- [ ] 正文 1--9 页，References 从第 10 页或更后开始，Supplement 位于 References 后；
- [ ] figure/table 编号连续，无 undefined/fatal/overfull；
- [ ] 匿名审计、PDF metadata 和空目录 Overleaf 重编译通过。

