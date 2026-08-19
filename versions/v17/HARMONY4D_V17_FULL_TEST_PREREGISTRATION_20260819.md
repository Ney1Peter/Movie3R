# Movie3R-v17 Harmony4D 全 test-capture 扩展预注册

日期：2026-08-19  
状态：在任何新增 capture 的 Movie3R 推理或指标读取前冻结

## 1. 目的与方法冻结

前一轮 `H4D-CS150` 已覆盖 Harmony4D 官方 test split 的全部 7 个动作包，但每个动作包只按结构哈希顺序选择了第一个坐标有效 capture，共 28 个 camera-pair case。该结果是“动作全覆盖”，不是“全部可用 capture 覆盖”。

本扩展保持 `Movie3R-v17 MultiCue-Safe` 的代码、checkpoint 和全部阈值不变，评测官方 test split 中所有满足 150 帧结构条件且能通过统一相机/坐标审计的 capture。Harmony4D train split 不并入测试主表。

冻结候选：`versions/v17/harmony4d/frozen_multicue_candidate.json`

```text
gate_min_matches                 = 2
gate_max_boundary_residual_m     = 0.25
gate_max_translation_m           = 1.6
boundary_blend                   = 1.0
root_alpha / root_beta           = 0.5 / 0.02
failure policy                   = bit-exact parent fallback
```

不允许根据本扩展的任何 GT 指标改变上述参数。若全量结果暴露缺陷，只能把该结果作为 v18 的诊断集，并在新的数据集或预先隔离的 Harmony4D train-development split 上改进。

## 2. 冻结评测总体

外层归档：`/data/wangzheng/iJCV-CODE/data/Harmony4D.zip`

官方 test split：

```text
test/01_hugging.zip
test/03_grappling2.zip
test/05_sword2.zip
test/06_sword3.zip
test/08_ballroom2.zip
test/15_mma4.zip
test/16_mma5.zip
```

归档结构索引在任何新增模型结果产生前已经给出：39 个 capture，其中 28 个满足“至少 150 个同步、连续帧”的结构条件。每个通过坐标审计的 capture 选择 small/medium/large/extreme 四个相机旋转跨度 strata 各一个确定性 camera pair，因此名义上限为 112 个 case。

结构合格不等于坐标可评测。对 PnP/标定失败、没有共同初始人体匹配等情况：

- 使用对所有方法完全相同的检查；
- 不读取某一方法的好坏决定是否保留；
- 记录 capture、错误类型和受影响 case；
- 仅作为 method-independent evaluator-unavailable 排除；
- 不用更短序列或更容易镜头替换进 150 帧主表。

## 3. 协议

- 每例 150 帧：75 pre + 75 post；
- 一个真实同步时刻、两个不同外部相机组成一个人工 cross-shot cut；
- known-boundary/oracle-boundary 几何主表与 detector 结果分开报告；
- 在线推理，边界决策不读取未来帧；
- 推理不读取 Harmony4D GT、标定或最终指标；
- GT 只在预测缓存写完后进入 evaluator；
- 每个 capture 的 camera pair 只依赖 GT 标定/可见性和固定角度分层，不依赖模型结果；
- 主聚合同时报告 clip macro、action/sequence macro、capture-aware分层和 95% bootstrap CI；
- 显著性检验以 7 个 action sequence 为最高层独立单元，不把同一 capture 的4个镜头对伪装成独立序列。

## 4. 必报方法与指标

同一缓存、同一 manifest 下比较：

1. Strict Human3R；
2. Movie3R-v15 / causal parent；
3. v17 parent（B0 + boundary ID）；
4. Movie3R-v17 MultiCue-Safe。

正文核心指标：

- W-MPJPE、WA-MPJPE；
- MPJPE、MPVPE；
- Accel；
- RTE-H3R；
- ATE-Sim3、ATE-SE3；
- IDF1、IDs、Coverage；
- Seam-root 与 camera-human relative seam。

同时报告 gate accept/fallback、accepted harm、按动作/镜头跨度/capture 的结果，以及所有 evaluator-unavailable case。

## 5. 决策规则与论文表述

v17 可作为 Harmony4D 正文主方法需要满足：

1. 相对 v17 parent，至少在 W/WA、Accel、ATE、IDF1/IDs、Seam 中形成多个稳定优势指标；
2. accepted 子集没有系统性灾难退化，fallback 保持 exact parent；
3. MPJPE/MPVPE 等局部人体指标退化不超过可解释的小幅范围；
4. 优势不能只来自单一动作或单一镜头跨度；
5. 所有结论使用同协议内部比较。

无需在每个指标全面超过 Strict Human3R 才能进入论文。若 v17 在相机、身份、时序连续性等与 multi-shot 在线重建目标直接相关的指标上明显领先，可把论文主张写为“跨镜头联合一致性提升”，而非“所有单帧人体指标全面 SOTA”。

Multi-THuMBS 的公开数字只作为协议不同的文献背景；在没有相同 manifest/evaluator 前，不宣称官方 leaderboard 胜出。

## 6. 运行与磁盘约束

- 使用独立 tmux 会话执行；
- 每次只展开一个 test 动作包；
- 先审计和冻结该包全部 capture manifest，再进行 GPU 推理；
- 预测缓存和指标验证完成后删除该包展开数据；
- 永久保留原始 `Harmony4D.zip`；
- 所有临时文件仅写入 `/data/wangzheng/iJCV-CODE/data/Harmony4D_work_v17_full_test`；
- 不向系统根目录写入任何数据。

