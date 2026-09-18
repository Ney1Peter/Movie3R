# BRIDGE3R v029 审阅裁决与执行计划（2026-09-01）

## 1. 目标与固定基线

- 审阅输入：`BRIDGE3R_v029_ICLR2027_全面审阅与修改主提示词_事实核验版.md`。
- 唯一论文基线：`versions/v029_20260830_mvhuman_audit_and_submission_package/manuscript`。
- 唯一原始交付包：`BRIDGE3R_ICLR2027_v029_20260830_Overleaf.zip`；不以临时 `_FIXED` 包为基线。
- 新版本：`versions/v030_20260901_fact_audit_and_evidence_revision`。
- 原则：不改变正式协议分母，不隐藏负面结果，不把推理时遮挡冒充独立训练消融，不把 oracle 控制写成可部署基线。

## 2. 审阅意见裁决

### 2.1 立即执行：事实与写作修正

1. 收敛主线为“因果的同场景相机—人体跨镜头坐标衔接”，明确输入为同步或时间相邻的同场景视角切换。
2. 弱化 `owns/state ownership/exclusively owns` 等拟人化措辞，改用“唯一被传播的重置状态”“只读边界分支”等可验证定义。
3. Abstract 删除 `most pronounced` 和 `establish` 等超出当前逐数据集区间证据的表述，并加入一组可追溯的 EgoBody 数值。
4. Method 增加严格因果定义、temporal token 公式、内部融合系数定义、粗变换的坐标组成，以及 camera/human/scene/state 的变换与保留规则。
5. 把 association 的两个分母写清：可评估 boundary pair 的条件正确率，以及相对于全部 continuation 的恢复率；同时报告运行时未主动弃权。
6. 把 `pre-registered/preregistered` 统一改为 `pre-specified`，因为当前没有公开时间戳注册记录。
7. 保留 Harmony4D 的 W-MPJPE 和重复切换 camera-trajectory 负面结果，避免全指标优势暗示。
8. 在 references 前加入独立的 AI Use、Ethics 和 Reproducibility statements；Supplement 仍与正文共用一个 PDF 且编号连续。

### 2.2 立即执行：由保留输出支持的新证据

1. **视角交互统计。** 从三个多人数据集的保留 case-level CSV 计算
   `extreme gain - non-extreme gain`，使用 recording/capture 聚类 bootstrap，输出 CSV、JSON 和 TeX。主张仅限数据集等权宏观交互，不将区间跨零的单数据集结果写成显著。
2. **检测器域外审计。** 对 EgoBody 与 MVHuman 低纹理压力集统一报告 first-trigger exact/early/late/miss、原始 off-boundary positive 和每 1000 个可判定转移的误报率。MVHuman 只作压力诊断，不能与正式三数据集主结果合并。
3. **关联审计。** 主文报告 EgoHumans 与 Harmony4D 的条件 pair accuracy、continuation recovery 和 runtime abstention；完整分母和 oracle upper bound 留在 Supplement。
4. **机制表重构。** oracle timing 与 causal timing 分 block 展示；不同 timing 的行不得被描述为单变量逐步消融。增加 IDF1 与 Coverage，仅使用可追溯的同协议数字。
5. **运行时间与系数说明。** 现有单 clip 运行时间明确为 detector-excluded incremental measurement；`lambda=0.5` 仅表述为跨数据集固定的预设折中，并保留 development-only sensitivity。

### 2.3 延后但保留为公开风险

1. correction-token 组成的独立重训练消融：正式训练约 29.5 小时，所需 AvatarReX/THuman 训练主体当前不完整；现有同 checkpoint masking 只能作为 sensitivity，不能证明每个 token 的训练必要性。
2. 第二重建骨干、自然编辑视频、盲测用户研究和更大规模 multi-cut：缺少冻结协议或对应数据/实现，不在本轮生成结果。
3. detector 的专门低纹理重训练：当前只如实披露域外早触发，不在测试集上调阈值或重选模型。

### 2.4 不执行

1. 删除不利指标、改变 90/88/129 正式分母或只展示成功案例。
2. 将 HumanMM/Multi-THuMBS 的论文数字与本地协议合并为伪统一排行榜。
3. 将 oracle boundary、oracle identity 或 evaluator-only acceptance 写作部署时输入。
4. 用生成模型伪造实验可视化或科学曲线；生成式工具只能用于不承载结果的装饰性素材。

## 3. 执行顺序与验收条件

### 阶段 A：证据审计

- 生成事实来源、方法—代码、结果来源、不一致、claim ledger 与 overclaim 报告。
- 每项 claim 必须指向源文件或自动生成 artifact。

### 阶段 B：补充统计

- 完成 viewpoint interaction bootstrap 与 detector generalization audit。
- 重构 mechanism、association、runtime 和 lambda 证据边界。
- 所有生成表均由脚本产生，论文中不手填新实验数字。

### 阶段 C：v030 写作

- 从 v029 原目录复制，不覆盖 v029。
- 修改 Abstract、Introduction、Method、Experiments、Discussion、Conclusion 和 Supplement。
- 外部方法完整表移至 Supplement；正文以严格同骨干配对比较为主。

### 阶段 D：提交级验收

- 正文（references 前、模板不计声明除外）不超过 9 页；References 后为连续编号 Supplement。
- PDF 中无 TODO、绝对路径、版本/hash、作者身份泄露。
- 编译无 undefined references、Type-3 字体和 overfull box。
- 交付 `main.pdf`、可直接导入 Overleaf 的干净 ZIP、`CHANGES_FROM_V029.md`、`REMAINING_RISKS.md` 和最终审稿式检查报告。

## 4. 状态约定

- `completed`：已生成证据、写入论文并通过引用核验。
- `in_progress`：正在执行且输入完整。
- `deferred`：科学上有价值，但缺失训练数据、算力时间或冻结协议。
- `rejected`：与事实、合规或公平比较原则冲突。

逐项状态记录在 `09_EXPERIMENT_STATUS_MATRIX.csv`。
