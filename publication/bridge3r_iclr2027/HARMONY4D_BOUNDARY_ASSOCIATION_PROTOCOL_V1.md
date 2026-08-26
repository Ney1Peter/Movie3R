# Harmony4D 边界身份关联评测协议 v1（执行中）

## 目的

这是 `REVISION_DECISIONS_V6.md` 中 D09 的 Harmony4D 部分：直接验证
BRIDGE3R 在首个 post-cut frame 所输出的预测驱动一对一关联，而不是只通过
长序列 IDF1 间接推断其是否有效。GT identity、标定和人体标注仅在
**evaluator** 打开；runtime 只读取已经冻结的 RGB 推理 cache 与其中记录的
`geometry.association.pairs`。

本协议不修改方法、不重新前向推理、不改变论文的主结果或 λ=0.5 配置。

## 冻结对象与执行阶段

| 阶段 | 输入 | 用途 | 论文状态 |
|---|---|---|---|
| Pilot（当前） | `test/05_sword2.zip` 中的正式 cache 对应 4 个 test case | 核验 cache、GT evaluator 和 runtime pair 的索引语义是否完全一致 | 不报告数值 |
| Full Harmony4D | 统一 audit 的共同可评 88 cases | 产生 Harmony4D 的直接关联证据 | 通过审计后进入补充材料；主文仅保留紧凑行 |
| Cross-dataset | EgoBody/EgoHumans 恢复后，以同一定义重放 | 形成不依赖单一数据集的组件证据 | D09 完成条件 |

Pilot 的 staging 位于 `data/Bridge3R_harmony4d_association_pilot/`，它是可
删除的临时工作区；不属于已验证的 multi-cut retention subset，也不得替代
原始 `Harmony4D.zip`。

## 固定评测定义

设切点为 `B`，runtime 在 frame `B-1` 的预测人索引 `i` 与 frame `B` 的
预测人索引 `j` 间输出一个 pair `(i,j)`。evaluator 在两个 frame 分别使用
既有 Harmony4D 的 camera--joint Hungarian assignment 和既有可接受代价阈值，
将预测人映射到可见 GT identity：`g_pre(i)` 和 `g_post(j)`。

一个 runtime pair 仅当两个端点都有唯一且可接受的 evaluator-only GT 匹配时
进入 **evaluable-pair** 分母。该 pair 的 correspondence 正确，当且仅当
`g_pre(i) = g_post(j)`。报告：

1. **First-post-cut correspondence accuracy**：正确 evaluable-pair 数除以
   evaluable-pair 数。它直接评测 boundary 的 permutation，而非整段轨迹。
2. **Evaluable pair count / continuation coverage**：同时报告分母大小，以及
   可评 pair 相对于 evaluator-only 可见 GT continuation 的覆盖，避免把检测
   漏失误写成关联错误。
3. **Runtime abstention rate**：runtime 明确不输出 pair 的可关联预测人比例。
   该指标只计关联模块的实际 abstain；因 GT 不可见或 evaluator assignment
   失败造成的样本排除必须单列，不能伪装成 runtime abstention。
4. **Oracle-identity upper bound（evaluator only）**：在完全相同的预测几何与
   detection 上，以两个端点的 evaluator GT assignment 强制正确的 post-cut
   persistent label，再按同一 ID evaluator 计算 IDF1。它只量化“若 WHO 完全
   正确，现有 detection/geometry 最多还能达到什么”，绝不进入 runtime。

若一个 case 的 `B-1` 或 `B` 没有有效 shared evaluator fit，必须与主结果一
样明确记为 evaluator unavailable；不能换用 case-specific GT 对齐来补数。

## 必要输出与审计

正式 full run 必须输出下列文件，且只在它们完整后才改论文表格：

- `case_rows.jsonl`：case id、pair 数、evaluable 数、正确数、abstain 数、
  evaluator exclusion reason、并保留 runtime pair 原值；
- `summary.json`：case-macro 与 pair-micro 两种聚合、95% bootstrap 区间和
  exact manifest hash；
- `strata.json`：人数变化、遮挡/不可见、相似姿态（若这些可由冻结
  evaluator metadata 无歧义定义）分层结果；
- `oracle_upper_bound.json`：与 runtime 结果分开保存的 evaluator-only
  label upper bound；
- 可复核脚本及其 SHA-256、输入 cache/runtime/manifest 的哈希清单。

任何只完成 cache 读取、但尚未经过上述 endpoint identity 审计的数字都只能叫
pilot diagnostic，不能写作 BRIDGE3R 的正式 association accuracy。
