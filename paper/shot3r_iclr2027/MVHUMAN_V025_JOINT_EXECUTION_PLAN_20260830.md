# BRIDGE3R：Held-out MVHuman 低纹理实验与 ICLR v025 联合执行计划

**日期：** 2026-08-30  
**状态：** 已完成并停止；PromptHMR 50/50 通过正式审计，GVHMR pilot 未通过扩展门槛，MVHuman 未达到 BRIDGE3R 正文正向证据门槛。最终结论见 `MVHUMAN_FINAL_EXPERIMENT_AND_PAPER_DECISION_20260830.md`。  
**方法约束：** 不改变已冻结的 BRIDGE3R checkpoint、因果 detector、共享平移系数或推理合同；不依据新 Test 结果重新选参。

## 0. 实际执行状态（2026-08-30）

- 数据准入已通过：10 个 capture/motion 与 correction-module 训练清单在 capture、event 和 frame/member 层面无重叠，但共享 MVHuman 数据域与两套相机 rig；论文只使用 ``held-out captures/motions, same rigs'' 的限定表述。
- 正式协议已在任何正式结果产生前冻结为 50 cases：10 captures × 5 个 SO(3) camera-rotation strata；每例 150 帧，由 75 个切换前帧和 75 个切换后帧组成。
- Strict Human3R 与 BRIDGE3R 已完成同一 50-case 分母的全部推理和 capture-macro 聚合。因自动 detector 在 40/50 例提前首次触发，该结果只进入 Supplement 的受控弱纹理端到端压力/失败审计，不作为正文正向优势证据，也不以 GT boundary 替换自动 detector。
- GVHMR 的预注册 12-case availability pilot 完成 10 例，2 例在官方 SimpleVO 产生原生重建前失败；83.3% completion 未通过扩展门槛，禁止扩展到 50 例或把条件精度写成排行榜。
- PromptHMR 采用官方 offline full-video 推理，已完成固定 50-case 全量协议并通过 native output、SPEC camera、conversion、metric、failure-record 和固定分母的完整性审计。其 Anchor/Seam-root 优于两个 recurrent routes，而 BRIDGE3R 的相对 camera-rotation error 更低；结合自动 detector 的提前触发，该结果只作为 Supplement 的 availability/failure stress audit，不作 BRIDGE3R 领先结论。
- 论文唯一后续来源为 `versions/v027_20260830_iclr_compliance_and_mvhuman`；最终从该版本创建 v029。并行存在的 v028 仅作历史草稿，不再合并。

## 1. 最终交付

本轮同时完成两条主线：

1. 将 `Mvhuman-test1.zip` 与 `Mvhuman-test2.zip` 构造成一个训练身份隔离、包含不同物理相机跨度的单人同场景硬切换评测，并完成 BRIDGE3R、Strict Human3R 与可执行公开方法的统一输入实验；
2. 依据事实核验文档与已确认的后续决策创建不覆盖 v024 的新版 ICLR 稿件，补齐三多人数据集综合证据、极端视角分析、学术化表述、真实定性图和可编辑矢量图，最终交付单 PDF 与可在 Overleaf 重编译的 ZIP。

THuman01 不进入本轮实验。MVHuman 只能在重叠审计通过后称为 held-out subject/capture evaluation，不能称为从未出现在任何预训练语料中的全新数据集。

## 2. 数据接收与完整性门槛

### 2.1 归档与解包

1. 保留两个上传 ZIP 原件，只读记录大小、修改时间和 SHA-256；任何修复均输出到新文件，不覆盖原件。
2. 分别验证外层 ZIP、每个内层 `tar.gz`、归档成员路径和解包后文件数；拒绝绝对路径、`..` 路径和符号链接逃逸。
3. 解包到新的 MVHuman Test 根目录，不与既有 `Training/mvhuman` 合并，防止训练和测试目录混淆。
4. 对每个 subject 核验 RGB、mask、相机内外参、SMPL/SMPL-X、2-D/3-D 标注、帧号和相机 ID 是否一一对应。

### 2.2 数据隔离

正式推理前，对最终 correction-module fine-tuning 的五份源 manifest 做四层交集审计：

- subject；
- capture / camera；
- event / frame；
- archive member。

当前已知训练 MVHuman subject 为 `100001--100005` 与 `200001--200005`；上传包候选 subject 为 `100021--100025` 与 `200021--200025`。该观察只能作为初筛，最终结论由机器生成的 overlap ledger 与哈希给出。

### 2.3 Test 接收条件

每个保留 subject 必须同时满足：

- 至少两台同步物理相机拥有连续 RGB；
- camera extrinsics 可转换到统一 world convention；
- cut 两侧人物 GT、mesh/joints 和可见性可评；
- 相机旋转跨度能覆盖至少三个预定义档位；
- 人体在连续时间窗中持续存在，不构造跨场景或任意时间跳跃。

不满足条件的 subject 在冻结 manifest 前排除，并记录数据原因；不得根据任何模型成绩排除。

## 3. 冻结的任务构造

### 3.1 输入形式

每个案例为一个 150-frame RGB 流：前 75 帧来自相机 A，后 75 帧来自同一物理场景中的相机 B，时间连续。模型只看到 RGB 和因果 detector；相机 ID、GT cut、标定、人体 GT、纹理分数与角度档位仅供 evaluator 使用。

### 3.2 角度档位

按相机相对旋转角预定义五档：

1. small：`<60 deg`；
2. medium：`60--90 deg`；
3. large：`90--120 deg`；
4. very large：`120--150 deg`；
5. extreme：`>=150 deg`。

每个 subject 在各可用档中选固定相机对与边界时刻。选择依据只允许是 GT 完整性、连续可见性、角度档和预定义随机种子，不允许参考任何方法输出。目标是每个 subject 每档至少一个案例；最终样本数由数据可用性审计决定，并在推理前冻结 JSONL、schema、seed 与 SHA-256。

### 3.3 低纹理定义

“低纹理”不只依赖视觉印象。冻结 manifest 时同时计算 evaluator-only 背景纹理指标：

- 使用 GT/可靠人体 mask 排除人体与扩张边界区域；
- 在统一缩放与灰度化后计算背景梯度幅值和 Laplacian 方差；
- 对有效背景像素做稳健归一化，并报告每个 subject/相机的分布；
- 阈值、mask 膨胀半径和图像缩放在模型结果产生前固定。

若没有足够的高纹理对照，不将 MVHuman 宣称为官方 low-texture benchmark；论文使用“held-out MVHuman captures with weakly textured backgrounds”并给出可复核的纹理统计。

## 4. 方法范围与公平性

### 4.1 必跑的严格配对方法

1. **Strict Human3R：** 原始递归状态跨 cut 继续传播；
2. **BRIDGE3R：** 冻结的因果 detector、learned coarse gauge、clean recurrence、association 与 shared translation 全部启用。

二者使用相同 RGB、case 顺序、图像预处理、输出帧、GT evaluator 和完整分母，是正文低纹理/角度结论的主要因果对照。

### 4.2 公开方法

公开方法先运行不接触正式结果的 subject-stratified pilot，并按原生输出能力分组：

- **PromptHMR：** 官方不改动的离线整段视频推理；通过 pilot 后运行完整 Test，明确标注可访问未来帧；
- **GVHMR：** 官方 world-grounded 单人路线；必须通过 raw tracker 在 cut 双侧的身份连续性门槛，不能用插值后的满帧输出掩盖单侧原始跟踪缺失；
- **WHAM / SLAHMR / MultiShot：** 仅在官方 checkpoint、依赖和全局坐标语义均可验证时进入 pilot；离线优化方法单独标注，不与因果延迟混排；
- **TRACE：** 若只能提供人体轨迹而没有可验证物理相机轨迹，只进入 local-pose/coverage 或定性参考，不进入 camera leaderboard。

任何方法若未通过预注册 pilot，只报告 availability audit 和失败原因，不从 Test 中手工修复个别案例，也不虚构 `N/A` 为数值结果。

## 5. 正式指标

### 5.1 主要全局指标

- First-shot Anchor MPJPE：仅在第一 shot 拟合一次 Sim(3)，切镜后沿用；
- W-MPJPE / WA-MPJPE：在统一 world/anchor 口径下衡量人体全局位置与对齐后结构；
- Camera relative rotation / translation error；
- ATE-Sim3 与 ATE-SE3：同时报告，正文根据 MVHuman 标定尺度审计选择主指标；
- Seam-root 与 seam-orientation：衡量切镜边界的不合理位置和朝向突变。

### 5.2 人体与可用性指标

- PA-MPJPE、MPJPE 与可用时的 MPVPE；
- completion、valid-frame coverage；
- detector precision / recall / F1 与 boundary offset；
- 每个几何均值对应的 finite support。

单人数据不把 IDF1 作为核心指标；若外部 tracker 产生多条身份，则身份连续性只作为 availability/track-integrity 审计。

### 5.3 统计

- subject-macro 为主，避免长序列或相机数多的 subject 支配均值；
- Strict Human3R 与 BRIDGE3R 使用 subject-cluster paired bootstrap 95% CI；
- 报告整体、五个角度档、弱纹理子集，以及纹理与视角跨度的交互；
- 不把不同支持数的外部方法条件均值包装成统一 SOTA 排名。

## 6. 执行顺序与 GPU 调度

1. 完成归档、格式、坐标、纹理与训练重叠审计；
2. 生成候选 case，做 GT 投影和角度 sanity check；
3. 冻结 pilot/Test manifest 和 evaluator；
4. 在少量 small/extreme 案例上运行 Strict Human3R 与 BRIDGE3R，验证单位、相机方向、GT 对齐和 detector；
5. 并行运行内部两方法全量 Test；
6. 依次完成 PromptHMR 与其他公开方法 pilot，通过者进入全量；
7. 聚合 case-level、angle-level、subject-level 与 texture-level 结果并做完整性审计；
8. 选择定性案例时先按预定义指标规则筛选，再保留完整 provenance，避免只凭视觉任意挑选。

独立前向最多使用五张空闲 GPU；同卡增加进程前先测总吞吐和 IO wait。评估、解包和统计不为占满 GPU 而并发。所有输出写入新目录，不覆盖历史实验。

## 7. ICLR 新版本重构

### 7.1 版本策略

- v024 保持只读；
- v025 先整合当前已锁定的事实核验与三多人数据集证据；
- 若 MVHuman 正式结果在 v025 编译后完成，则创建 v026 合入，不回写旧版本；
- 最终只向用户交付最新通过验收的 PDF 与 Overleaf ZIP，同时保留版本日志。

### 7.2 九页正文预算（References 不计）

1. Abstract + Introduction：约 1.5 页；
2. Related Work：约 0.8--1.0 页；
3. Method：约 2.6--2.9 页；
4. Experiments：约 3.2--3.5 页；
5. Discussion + Conclusion：约 0.5--0.7 页。

正文优先保留：问题与因果合同、方法主图、三个多人数据集主表、extreme/farthest 表、EgoBody 系统级消融、角度趋势和真实定性图。完整外部 availability、AIST++、MVHuman 全指标、multi-cut、CI、失败与实现细节进入同一 PDF 的 Supplement。

### 7.3 本轮必须落实的写作决定

- 正式名称统一为 BRIDGE3R；不出现内部版本号、`adapt`、`v1`、运行目录名或工程分支名；
- 核心任务固定为 same-scene hard viewpoint cuts，不外推任意跨场景电影剪辑；
- Camera--Human 为主，不声称定量 scene reconstruction；用相机、人体和背景上下文的真实定性结果呈现整体空间一致性；
- 三多人数据集增加 dataset-equal all-view 与 extreme/farthest 证据，保留 Harmony4D W 基本持平的事实；
- Coverage 与 conditional geometry 同时出现；Camera ATE 的 Sim(3)/SE(3) 自由度在正文解释；
- AIST++ 暂定位为弱纹理棚拍单人方向/相机连续性补充，不声称全面领先；
- MVHuman 只有在正式结果支持时才升级低纹理结论，且不得隐藏平移或 coverage 退化。

## 8. 图像与可编辑性

### 8.1 方法与趋势图

- 方法主图、任务示意、角度趋势和纹理--角度交互图使用 SVG/TikZ/矢量 PDF；
- 保留可编辑源文件、字体、颜色、数据 CSV 和生成脚本；
- 不使用生成式位图替代科学流程、坐标关系或定量曲线。

### 8.2 真实定性图

从相同序列和相同帧提取 RGB、Strict Human3R、BRIDGE3R 和具备可比输出的公开方法。每个案例至少显示：

- cut 前 RGB 与 cut 后 RGB；
- 相同尺度下的相机 frustum；
- 世界坐标人体 mesh/skeleton；
- 方法自身可恢复的点云/背景上下文；
- 用简洁箭头标注错误身份、错误相机朝向、人体漂移或边界跳跃。

定性图的排版、箭头和文字为可编辑 SVG/PDF；RGB、mesh 和点云渲染本身是由真实实验输出生成的栅格面板。图注列出 case ID、角度、选择规则和输出 provenance，不生成或补画不存在的实验结果。

## 9. 验收与停止条件

实验验收要求：不可变 manifest、checkpoint/代码哈希、完整命令、case-level 结果、失败账本、finite support、机器生成表格与可从逐例结果重建的 aggregate。

论文验收要求：

- ICLR 2027 模板；正文恰好不超过 9 页，References 从下一页开始；
- Supplement 与正文为一个 PDF，图表编号和引用连续；
- 无 TODO、占位图、undefined reference、overfull box、内部绝对路径、作者身份或工程版本名；
- PDF 与 Overleaf ZIP 在空目录完整重编译一致；
- 方法、代码、数值、分母和文字主张逐条通过 claim--evidence audit。

只有两个上传归档、正式实验、真实图、论文集成与独立编译验收全部完成后，本轮任务才结束。若外部 checkpoint/许可证或数据损坏构成真实阻塞，必须保留可复核证据并明确报告，不以猜测数值代替实验。
