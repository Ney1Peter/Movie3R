# BRIDGE3R 实验指标分母与有效样本私有总账

> **内部审计文件，不得进入论文、Supplement、Overleaf ZIP、投稿 PDF、公开代码仓库或匿名审稿材料。**  
> 建立日期：2026-08-28  
> 目的：防止后续把不同正式样本数、聚合层级、有效样本支持数和条件指标误写成同一分母。  
> 本文件记录的是“数据如何被计数”，不是论文表格排版稿；正式表格必须由本总账重新生成并再次核验。

## 1. 最重要的使用规则

1. **正式样本数 (N) 与指标支持数不是一回事。** 例如某方法在 90 个案例上运行，但只有 17 个案例能计算 W-MPJPE，则必须写成“正式协议 (N=90)，W-MPJPE support (n=17)”；不能把 17 写成整个测试集大小。
2. **Coverage、IDF1、Precision 衡量完整协议上的可用性。** 推理失败、没有人、没有成功匹配的案例仍留在正式分母中，并按零贡献处理，不能从平均中删除。
3. **几何误差是条件均值。** W-MPJPE、WA-MPJPE、MPJPE、PA-MPJPE、MPVPE、ATE 等只在该指标有限且可评的案例上平均，必须同时显示 support。
4. **不同方法不得共享一个虚构的几何分母。** 外部方法分别与 BRIDGE3R 在该外部方法的逐对共同有效案例上比较；不强制取 TRACE、PromptHMR、Human3R、BRIDGE3R 四者的极小全局交集。
5. **case macro 与 recording/capture/source macro 不得混用。** EgoBody 先在每个 recording 内平均三个角度，再对 43 个 recording 等权；EgoHumans/Harmony4D 当前主结果是 case macro；AIST++ 是 source macro。
6. **推理完成不等于几何可评。** 例如 TRACE 可以完成 129/129 次推理，但可能没有被匹配到的人体，因而 W/WA support 很小。
7. **N/A 不等于 0，也不等于失败。** TRACE 不输出独立物理相机轨迹，因此相机 ATE 为 N/A；不能填 0，也不能把它计为 evaluator failure。
8. **空白表示证据没有保留，不得猜测。** 尤其 EgoBody 原始逐 case 外部结果已删除；某些 case-level support 只能等待恢复 sealed artifact 后填写。
9. **所有百分比必须说明分母层级。** 例如“142/147 endpoint pairs”不能改写成“84/88 cases”，二者回答的是不同问题。
10. **旧表中的数值不得直接复制。** Harmony4D 旧 100-case 五方法表与当前统一 88-case BRIDGE3R 结果绑定冲突，已列为禁止使用项。

## 2. 术语与计数口径

| 字段 | 含义 | 常见误用 |
|---|---|---|
| Formal (N) | 预先定义、对所有方法一致的正式案例清单大小 | 把成功案例数当成 Formal (N) |
| Inference attempt | 实际尝试运行的方法输入数 | 把推理成功当成匹配成功 |
| Inference failure | 程序未产生可消费预测的案例 | 把零匹配误写成推理失败 |
| Coverage denominator | 计算完整协议 Coverage 时保留的案例清单 | 删除零预测案例导致 Coverage 虚高 |
| IDF1 denominator | 计算完整协议 IDF1 时保留的案例清单 | 只在成功跟踪案例上算 IDF1 |
| Metric support (n) | 某个几何指标实际为有限值的案例数 | 不标 support，制造全量平均的错觉 |
| Cluster (N) | bootstrap、统计检验或宏平均的独立单元数 | 把同一 recording 的三个角度当三个独立样本 |
| Pairwise support | 外部方法和 BRIDGE3R 对同一指标均可评的共同案例数 | 用两种方法各自条件均值直接宣称逐例胜出 |

Coverage/IDF1 的“分子”不是简单的成功案例数，而是由逐帧可见 GT 人体、预测人体和身份匹配累计得到的统计量。本总账主要锁定它们的**正式案例分母**；逐帧分子只在已封存报告明确给出时记录。

## 3. 当前正式协议总览

| 数据集/协议 | 正式案例数 | 独立聚合单元 | 聚合方式 | 当前用途 |
|---|---:|---:|---|---|
| EgoBody-CS150 | 129 cases | 43 recordings | 每个 recording 先平均 small/medium/extreme 三个角度，再对 43 recordings 等权 | 多人主结果与外部方法 |
| EgoHumans-CS100 | 90 cases | 27 captures（paired statistics） | 主指标 case macro；配对统计按 capture 聚类 | 条件式跨镜头 bridging 主结果 |
| Harmony4D-CS150 | 88 cases | 25 captures（位于 7 个 archive/action groups） | 主指标 case macro；配对统计按 capture 聚类 | 多人主结果与外部方法 |
| AIST++-CS150 | 100 official `pose_test` sources | 100 sources | source macro | 单人域外诊断，宜放补充材料 |
| AIST++-MC150-3 | 100 official `pose_test` sources | 100 sources | source macro | 三段/两次切换的事件缩放诊断 |
| AIST++-MC150-4 | 100 official `pose_test` sources | 100 sources | source macro | 四段/三次切换的事件缩放诊断 |

EgoHumans 的 90 例是当前确定的正式条件式协议：每例均满足 frozen Human3R 在切换前已经建立有效人物锚点。论文若使用该协议，应只描述这一 eligibility 条件，不写成“随机 90 例”，也不把未纳入的历史候选描述成 GT 无效。

## 4. EgoHumans-CS100：正式 90 例

### 4.1 全正式协议支持数与结果

所有方法均以同一 90-case manifest 作为输入。Coverage、IDF1、Precision 的分母始终是 90；几何均值只在 support 内计算。

| 方法 | 推理失败 | W support | WA/local support | ATE support | W-MPJPE | WA-MPJPE | ATE-Sim3 | IDF1 | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 0/90 | 90 | 90 | 90 | 1005.1 mm | 365.7 mm | 0.547 m | 0.471 | 0.639 |
| BRIDGE3R | 0/90 | 90 | 90 | 90 | 866.8 mm | 336.9 mm | 0.032 m | 0.571 | 0.616 |
| TRACE | 10/90 | 43 | 59 | 0（N/A） | 2464.6 mm | 749.5 mm | N/A | 0.070 | 0.072 |
| PromptHMR-SPEC | 51/90 | 17 | 25 | 39 | 1193.3 mm | 487.7 mm | 0.704 m | 0.061 | 0.062 |
| PromptHMR no-SPEC | 51/90 | 17 | 25 | 39 | 1148.4 mm | 486.9 mm | 0.734 m | 0.061 | 0.061 |

这里 TRACE 的 `10/90` 和 PromptHMR 的 `51/90` 是外部运行账本中的失败状态；其余成功推理案例仍可能因没有成功人体匹配而缺少 W 或 WA。BRIDGE3R 的 Coverage 略低于 Strict Human3R，而 W、WA、ATE 和 IDF1 更好；后续正文不得写成“所有指标全面提升”。

### 4.2 外部方法与 BRIDGE3R 的逐对共同案例

| 外部方法 | 指标 | 共同 support | 外部方法 | BRIDGE3R（同一批案例） |
|---|---|---:|---:|---:|
| TRACE | W-MPJPE | 43 | 2464.6 mm | 758.2 mm |
| TRACE | WA-MPJPE | 59 | 749.5 mm | 340.2 mm |
| PromptHMR-SPEC | W-MPJPE | 17 | 1193.3 mm | 526.4 mm |
| PromptHMR-SPEC | WA-MPJPE | 25 | 487.7 mm | 301.2 mm |
| PromptHMR-SPEC | ATE-Sim3 | 39 | 0.704 m | 0.019 m |
| PromptHMR no-SPEC | W-MPJPE | 17 | 1148.4 mm | 526.4 mm |
| PromptHMR no-SPEC | WA-MPJPE | 25 | 486.9 mm | 301.2 mm |
| PromptHMR no-SPEC | ATE-Sim3 | 39 | 0.734 m | 0.019 m |

逐对表只用于回答：“在外部方法本身能够计算该指标的案例上，BRIDGE3R 如何？”它不能替代完整 90 例上的 Coverage/IDF1，也不能把 support 写成正式测试集大小。

来源：

- 内部逐 case：`Movie3R/output/v19_egohumans/test/summary/case_metrics.csv`
- 外部逐 case：`Movie3R/output/v19_egohumans/final/external_baseline_case_metrics.csv`
- 正式 case IDs：`egohumans_cs100_formal_case_ids.txt`
- 自动聚合：`FORMAL_METHOD_SUPPORT.csv`、`PAIRWISE_EXTERNAL_GEOMETRY.csv`

## 5. Harmony4D-CS150：统一正式 88 例

### 5.1 全正式协议支持数与结果

| 方法 | 推理失败 | W support | WA/local support | ATE support | W-MPJPE | WA-MPJPE | ATE-Sim3 | IDF1 | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 0/88 | 88 | 88 | 88 | 496.6 mm | 259.6 mm | 0.142 m | 0.553 | 0.915 |
| BRIDGE3R | 0/88 | 88 | 88 | 88 | 519.6 mm | 247.0 mm | 0.0168 m | 0.636 | 0.913 |
| TRACE | 0/88 | 4 | 26 | 0（N/A） | 677.1 mm | 270.2 mm | N/A | 0.016 | 0.016 |
| PromptHMR-SPEC | 12/88 | 5 | 26 | 76 | 963.2 mm | 306.9 mm | 0.478 m | 0.030 | 0.034 |
| PromptHMR no-SPEC | 12/88 | 5 | 25 | 76 | 957.7 mm | 297.2 mm | 0.471 m | 0.029 | 0.031 |

关键事实：BRIDGE3R 的 W-MPJPE 比 Strict Human3R 差 23.0 mm；WA、ATE-Sim3 和 IDF1 更好，Coverage 基本持平但略低。这个 mixed result 必须保留，不能通过换回旧表隐藏。

按 25 个真实 capture 聚类的回顾性配对统计进一步给出：W 的有利增益为
`-23.0 mm`（95% CI `[-67.9, 19.0]`），WA 为 `12.6 mm`
（`[-12.4, 35.8]`），ATE-Sim3 为 `0.1252 m`
（`[0.1005, 0.1574]`），IDF1 为 `0.0834`
（`[0.0400, 0.1287]`），Coverage 为 `-0.0028`
（`[-0.0054, -0.0009]`）。因此正文只可将 W/WA 写成点估计及权衡；
ATE-Sim3 和 IDF1 有稳定的正向证据，Coverage 是幅度很小但可检测的下降。

### 5.2 外部方法与 BRIDGE3R 的逐对共同案例

| 外部方法 | 指标 | 共同 support | 外部方法 | BRIDGE3R（同一批案例） |
|---|---|---:|---:|---:|
| TRACE | W-MPJPE | 4 | 677.1 mm | 254.2 mm |
| TRACE | WA-MPJPE | 26 | 270.2 mm | 216.7 mm |
| PromptHMR-SPEC | W-MPJPE | 5 | 963.2 mm | 359.7 mm |
| PromptHMR-SPEC | WA-MPJPE | 26 | 306.9 mm | 233.2 mm |
| PromptHMR-SPEC | ATE-Sim3 | 76 | 0.478 m | 0.017 m |
| PromptHMR no-SPEC | W-MPJPE | 5 | 957.7 mm | 359.7 mm |
| PromptHMR no-SPEC | WA-MPJPE | 25 | 297.2 mm | 233.8 mm |
| PromptHMR no-SPEC | ATE-Sim3 | 76 | 0.471 m | 0.017 m |

TRACE 的 W 只有 4 例、PromptHMR 的 W 只有 5 例，这些条件均值尤其不能单独放进不带 support 的主排行榜。

来源：

- 内部统一方法逐 case：`Movie3R/output/v17_harmony4d/unified_half_translation_audit/paper/case_metrics.csv`
- 外部逐 case：`data/Harmony4D_work_v17_full_test/external_predictions/harmony4d_external_aggregate/external_baseline_case_metrics.csv`
- 正式 case IDs：`harmony4d_cs150_formal_case_ids.txt`
- 自动聚合：`FORMAL_METHOD_SUPPORT.csv`、`PAIRWISE_EXTERNAL_GEOMETRY.csv`

### 5.3 已确认的历史表冲突

旧文件：

`versions/v021_20260828_nine_page_expansion/manuscript/artifacts/harmony4d_final/harmony4d_five_method_table.tex`

| 结果绑定 | BRIDGE3R W | WA | ATE-Sim3 | 状态 |
|---|---:|---:|---:|---|
| 旧 100-case 五方法表中的 88-support 行 | 584.0 | 266.9 | 0.019 | **禁止继续使用** |
| 当前统一冻结方法、正式 88 例 | 519.6 | 247.0 | 0.0168 | 当前事实来源 |

旧表还把外部方法的 availability、failure 与 100-case 口径绑定，而当前确定的正式协议是统一 88 例。后续修改论文时必须从当前 88-case case IDs 重新生成整个五方法表，不能只手改 BRIDGE3R 三个数字。

## 6. EgoBody-CS150：43 recordings / 129 cases

### 6.1 聚合方式

每个 recording 有 small、medium、extreme 三种相机角度案例。正式聚合先在 recording 内平均三个角度，再对 43 个 recordings 等权。因此：

- 表格的 recording-macro 均值不是简单的 129-case micro mean；
- bootstrap 与配对统计的独立单元是 43 个 recordings；
- 三个角度只能用于 recording 内诊断，不能当作 129 个完全独立实验单元做显著性检验。

### 6.2 支持数

| 方法 | 推理完成 | Coverage/IDF1 分母 | W support | WA/local support | ATE support |
|---|---:|---:|---:|---:|---:|
| Strict Human3R | 129/129 | 129 cases / 43 rec. | 129 / 43 | 129 / 43 | 129 / 43 |
| BRIDGE3R | 129/129 | 129 / 43 | 129 / 43 | 129 / 43 | 129 / 43 |
| TRACE | 129/129 | 129 / 43 | 22 cases / 15 rec. | 35 / 17 | N/A |
| PromptHMR-SPEC | 129/129 | 129 / 43 | 64 / 34 | 87 / 36 | 129 / 43 |
| PromptHMR no-SPEC | 129/129 | 129 / 43 | case 数未保留 / 33 rec. | case 数未保留 / 36 rec. | 129 / 43 |

补充计数：

- TRACE：94/129 为零匹配案例，26/43 recordings 全部零匹配；W 只有 22 cases，WA/local 只有 35 cases。
- PromptHMR-SPEC：42/129 零匹配，7/43 recordings 零匹配；相机 ATE 可在 129/129 计算。
- PromptHMR no-SPEC：38/129 零匹配，7/43 recordings 零匹配；保留报告只写了 recording-level W/WA support，没有足够证据恢复 case-level support。
- EgoBody 原始 per-case 外部 artifact 已为释放磁盘而删除；没有重新上传/恢复 sealed artifact 前，不能生成外部方法与 BRIDGE3R 的逐对共同案例几何表。

来源：

- `external_baselines/bridge3r_eval/TRACE_EGOBODY_TEST.md`
- `external_baselines/bridge3r_eval/PROMPTHMR_SPEC_EGOBODY_TEST.md`
- `external_baselines/bridge3r_eval/PROMPTHMR_EGOBODY_TEST.md`
- `versions/v021_20260828_nine_page_expansion/manuscript/artifacts/egobody_v20/EGOBODY_FINAL_RESULTS.md`

## 7. AIST++ 单人诊断

### 7.1 CS150

- 正式集合：100 个不同的官方 `pose_test` sources；不是从 train/val 混入的 100 个窗口。
- Strict Human3R、BRIDGE3R、PromptHMR 都是 100/100 完成，所有已报告指标使用 source-macro (N=100)。
- PromptHMR 是 offline full-video 路线，不能与因果流式方法写成相同延迟条件下的排名。
- GVHMR 只做了预注册的 12-case availability pilot；仅 2/12 有单侧 raw tracker support，因此没有进入正式 100-source 表。它不是“正式结果为 2/12”，而是“未通过扩展到 Test 的可用性门槛”。

### 7.2 MC150-3 与 MC150-4

- 两个协议各有独立冻结的 100-source `pose_test` manifest；每张表的每个组件均为 (N=100)。
- MC150-3 和 MC150-4 不能合并平均，也不能与 CS150 合并成 (N=300)，因为它们是同一来源池上的不同事件结构压力测试。
- Clean reset、coarse alignment only、coarse alignment + identity、完整 BRIDGE3R 属于组件/事件缩放诊断，不是外部 baseline。

来源：

- `versions/v021_20260828_nine_page_expansion/manuscript/artifacts/aist_cs150_formal/AIST_CS150_FORMAL_REPORT.md`
- `versions/v021_20260828_nine_page_expansion/manuscript/artifacts/aist_multicut_formal/AIST_MULTICUT_FORMAL_REPORT.md`

## 8. 辅助实验分母

| 实验 | 原始集合 | 可评支持 | 聚合/统计单元 | 允许的解释 |
|---|---:|---:|---|---|
| EgoBody paired statistics | 129 cases | 43 recordings | recording-paired | 三角度先合并，43 个独立 recording |
| EgoHumans paired statistics | 90 cases | 27 captures | capture-clustered | 90 case 的配对效应，置信区间按 capture 相关性处理 |
| Harmony4D paired statistics | 88 cases | 25 captures（7 groups） | capture-clustered | 25 个物理 capture 为相关性单元；7 个 archive/action groups 不是 capture |
| Harmony4D association | 88 cases | 84/88 至少一个可评 pair；142/147 endpoint pairs | pair micro + case macro | 直接验证 boundary association；分母不能互换 |
| Harmony4D multi-cut | 4 captures | 8 boundaries | capture macro | 有限 repeated-transition evidence，不是大规模 benchmark |
| Harmony4D lambda sensitivity | 12 train-only cases | 9 evaluator-complete；seam 8 | case macro | 训练集合上的描述性灵敏度，不用于 Test 选参 |
| EgoBody cut detector | 129 Test cases | 129 | case macro | detector 的完整 Test 账本 |
| EgoBody existing runtime | 129 Test cases | 只有组件 wall time 与 host RSS | case mean | 不能宣称标准化 end-to-end FPS 或 GPU peak memory |

Harmony4D association 的细分分母必须原样保留：

- 84/88 cases 至少有一个 evaluator-valid endpoint pair；
- 142/147 evaluator-valid endpoint pairs 正确；
- correct-pair continuation coverage 为 142/176；
- runtime abstention 为 0/256；
- evaluator-excluded runtime pairs 为 109；
- IDF1 的 case-level 汇总仍对应 88-case manifest。

## 9. 当前仍缺少或不可恢复的分母证据

1. **EgoBody 外部逐对共同集。** 当前只保留 aggregate report 与哈希，per-case external artifact 不在工作区。恢复 EgoBody 数据和 sealed per-case results 后，才能像 EgoHumans/Harmony4D 一样计算“TRACE vs BRIDGE3R 同一可评 cases”和“PromptHMR vs BRIDGE3R 同一可评 cases”。
2. **EgoBody PromptHMR no-SPEC 的 case-level W/WA support。** 现有报告只明确 33/43 与 36/43 recording support；不得从零匹配数反推 W support。
3. **标准化 runtime。** 现有 Test 129-case runtime 只有组件 wall-time、共享服务器上的诊断性 FPS 与 host RSS；没有单一方法端到端 FPS，也没有 GPU peak memory。
4. **Harmony4D 旧表重建。** 论文执行阶段需要统一在 88-case manifest 上重聚合并重写，当前私有审计只标记冲突，不修改论文。

## 10. 自动生成文件与复现

本目录文件分工：

| 文件 | 用途 |
|---|---|
| `build_denominator_ledger.py` | 从仍存在的 EgoHumans/Harmony4D 逐 case CSV 自动重聚合，并生成总账 |
| `FORMAL_METHOD_SUPPORT.csv` | 两个数据集、10 个方法行的全量支持数和条件均值 |
| `PAIRWISE_EXTERNAL_GEOMETRY.csv` | 外部方法与 BRIDGE3R 的逐指标共同 support 和同案例均值 |
| `DENOMINATOR_LEDGER.csv` | 所有正式/辅助/冲突行的机器可读总账 |
| `egohumans_cs100_formal_case_ids.txt` | EgoHumans 正式 90-case manifest |
| `harmony4d_cs150_formal_case_ids.txt` | Harmony4D 正式 88-case manifest |

复现命令：

```bash
python3 ICLR-paper/bridge3r_iclr2027/private_audit/build_denominator_ledger.py
```

脚本会检查：

- 正式 ID 数是否恰为 90/88；
- 是否存在重复 case ID；
- 每个方法在正式 manifest 上是否都有一行状态记录；
- 输入逐 case CSV 的 SHA-256；
- full-protocol 可用性与 conditional geometry 是否使用不同的聚合规则。

## 11. 后续制作论文表格前的检查清单

每张新表生成前必须逐项回答：

- [ ] 表标题中的 (N) 指 formal cases、recordings、captures、sources 还是 pairs？
- [ ] 所有方法是否使用同一个正式 case-ID manifest？
- [ ] Coverage/IDF1 是否保留失败和零匹配案例？
- [ ] 每个几何指标是否显示自己的 support，而不是只显示一个模糊的 (N)？
- [ ] 外部方法的几何均值是否与 BRIDGE3R 在同一批可评案例上成对重聚合？
- [ ] TRACE 的相机指标是否保持 N/A？
- [ ] EgoBody 是否使用 recording macro，而不是误写为 129-case macro？
- [ ] AIST++ 是否使用 source macro，且没有把 CS150、MC150-3、MC150-4 合并？
- [ ] Harmony4D 是否彻底移除了旧 584.0/266.9/0.019 的 BRIDGE3R 结果绑定？
- [ ] mixed result 是否如实保留，特别是 Harmony4D 的 W-MPJPE 负结果？
- [ ] 表格来源能否追溯到本目录 CSV、case ID manifest 和源文件哈希？

## 12. 投稿隔离规则

本目录应始终位于所有 `versions/*/manuscript` 和 `releases/*.zip` 之外。构建新 Overleaf 包时只复制对应版本的 `manuscript/` 及正式所需证据，不递归复制 `bridge3r_iclr2027/` 根目录。任何发布脚本若发现以下字符串，应视为失败：

- `private_audit`
- `DENOMINATOR_LEDGER_PRIVATE`
- `FORMAL_METHOD_SUPPORT.csv`
- `PAIRWISE_EXTERNAL_GEOMETRY.csv`
- `DO NOT USE; must regenerate`

内部总账允许记录路径、哈希、历史冲突和被排除的错误写法；正式论文只保留经过核验后必要的协议定义、support 与学术化表述。
