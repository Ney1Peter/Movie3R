# Movie3R-v17 Harmony4D 最终批量实验报告

日期：2026-08-19  
最终候选：`Movie3R-v17 MultiCue-Safe`  
主协议：150 帧（75 pre + 75 post），在线、无未来帧、无 GT 推理、无额外外部预训练模型。

## 1. 最终结论

v16 的第一版 residual-only 安全门控在官方 7-sequence batch 中出现一个强接触错误接受，导致 W-MPJPE 从父模型的 544.8 mm 恶化到 577.0 mm。该版本不能作为论文主方法。

v17 在不改动有效几何模块的前提下，加入由旧开发集支持的 1.6 m prediction-only translation trust region。它在已见官方 test regression 中消除了唯一灾难例，并在两个候选冻结后才读取的新 Harmony4D 序列上独立获得一致收益：

- 新留出 8-case：W-MPJPE **477.7 → 441.8 mm（-7.5%）**；
- WA-MPJPE **294.3 → 276.0 mm（-6.2%）**；
- Accel **106.9 → 94.4 mm/frame²（-11.7%）**；
- RTE-H3R **9.05% → 7.59%（-16.2%）**；
- ATE-Sim3 **0.0254 → 0.0226 m（-10.8%）**；
- ATE-SE3 **0.2198 → 0.1426 m（-35.1%）**；
- Seam-root **0.985 → 0.648 m（-34.2%）**；
- MPJPE/MPVPE 仅约 +0.25%，IDF1 -0.00038，Coverage 略升。

因此 v17 已满足“可冻结、可批量扩展、可作为 Harmony4D 主线消融”的标准。它还不是完整 ICLR 最终 SOTA 证据：新留出目前只有 2 个动作序列，且 Strict Human3R 的 W/WA 仍更低；后续必须扩充跨动作/跨数据集测试并加入自动 detector 端到端结果。

## 2. 方法冻结

输入为一个流式 RGB 两-shot 序列。每个时刻仍由 Human3R/Movie3R 主干输出相机、人体 joints/vertices 与 recurrent state。边界处依次执行：

1. **B0 与 boundary permutation ID**：提供共同粗坐标和跨 shot 人物槽位对应；
2. **Human-Anchored Coupled Boundary Registration**：利用已匹配 torso/pelvis 估计一个共同平移，同时作用于整个 post shot 的相机和所有人体，严格保持相机—人体相对几何；
3. **Causal root stabilization**：对持久身份做因果 alpha-beta 平移滤波，不改 body pose 和相机；
4. **Multi-cue trust gate**：至少 2 个匹配、平均 torso residual 不超过 0.25 m、共同平移不超过 1.6 m；
5. 任一条件失败时返回 **bit-exact parent fallback**。

阈值只读取当前/历史预测。1.6 m 来自 v16 已冻结开发报告中“有效接受最大 1.526 m、相邻失败 1.577 m”的圆整安全边界，不由官方 test 数值拟合。候选配置见 `versions/v17/harmony4d/frozen_multicue_candidate.json`。

## 3. 官方 7-sequence batch：失败发现与 seen-test regression

### 3.1 冻结 v16 原始结果

```text
7 sequences
28 preregistered cases
25 evaluable cases
3 method-independent evaluator-unavailable cases (Ballroom2)
10 accept / 15 fallback
```

| Method | W | WA | Accel | RTE-H3R | ATE-Sim3 | IDF1 |
|---|---:|---:|---:|---:|---:|---:|
| Strict Human3R | **466.1** | 262.7 | 93.6 | 6.8 | 0.1387 | 0.535 |
| Movie3R-v15 / v16 parent | 544.8 | 265.8 | 98.0 | 7.1 | 0.0197 | **0.612** |
| v16 Harmony-Safe | 577.0 | **249.9** | **91.9** | **6.6** | **0.0189** | **0.612** |

MMA4 medium 的 2 个错误人体锚给出很低 residual（0.043 m），却请求 1.702 m 共同平移；v16 错误接受后该 case W 达到 2640.4 mm，使全量 W 反而恶化。该 case 没有被删除。

### 3.2 v17 精确回归

v17 仅改变 gate 决策：接受时与 v16 几何 bit-identical，拒绝时与 parent bit-identical。因此无需重跑 GT evaluator 即可从冻结行构造精确 regression：

| Method | W | WA | Accel | RTE-H3R | ATE-Sim3 | IDF1 |
|---|---:|---:|---:|---:|---:|---:|
| v17 parent | 544.8 | 265.8 | 98.0 | 7.12 | 0.0197 | 0.6117 |
| v17 MultiCue-Safe | **513.0** | **247.7** | **91.7** | **6.49** | **0.0191** | 0.6115 |

9 accept / 16 fallback；灾难性 MMA4 medium 被 1.6 m trust region 精确回退。由于官方 test 已用于发现 v16 问题，本表只能作为 regression diagnostic，不能作为 v17 的首次无偏验证。

## 4. 新未见留出集

冻结顺序：先预注册 train10，读取并写死 GO 决策后，才运行此前未读的 train11。两者都取结构 SHA 顺序第一个 coordinate-valid capture，四个 camera-angle strata 全部保留。

### 4.1 分序列结果

| Sequence | Cases | Accept / fallback | W parent | W v17 | WA parent | WA v17 | Accel parent | Accel v17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train10 karate2 | 4/4 | 2 / 2 | 483.8 | **448.9** | 264.6 | **248.2** | 86.4 | **73.3** |
| train11 karate3 | 4/4 | 1 / 3 | 471.6 | **434.7** | 324.0 | **303.8** | 127.4 | **115.5** |

三个 accepted case 的 W 变化为 -26.8%、-5.2%、-29.7%；没有 accepted harm。其余五例 exact fallback。

### 4.2 合并 8-case 主表

| Method | W ↓ | WA ↓ | MPJPE ↓ | MPVPE ↓ | Accel ↓ | RTE-H3R ↓ | ATE-Sim3 ↓ | IDF1 ↑ | IDs ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict Human3R | **390.0** | **245.0** | **101.4** | **115.2** | 106.7 | **5.23** | 0.1123 | 0.763 | 2.8 |
| Movie3R-v15 | 473.5 | 293.2 | 104.1 | 118.2 | 106.4 | 9.02 | 0.0452 | 0.841 | 2.8 |
| v17 parent | 477.7 | 294.3 | 104.1 | 118.2 | 106.9 | 9.05 | 0.0254 | **0.843** | 2.8 |
| **Movie3R-v17** | **441.8** | **276.0** | 104.3 | 118.6 | **94.4** | **7.59** | **0.0226** | **0.843** | 2.8 |

相对 Movie3R-v15，v17 的 W/WA/Accel 分别改善 6.7%/5.9%/11.3%，ATE-Sim3 改善 49.9%，IDF1 增加 0.0022，Coverage 增加 0.0029。

Strict Human3R 在这两个序列的 W/WA 仍领先，但它的 ATE-Sim3 为 0.1123 m、IDF1 为 0.763；v17 分别为 0.0226 m 和 0.843。正确论文表述是“Movie3R 改善自己完整跨-shot/身份主线，并在相机和身份上显著优于 strict backbone”，不能声称所有指标全面击败 Human3R。

## 5. 统计与可引用范围

10,000 次 hierarchical sequence-then-clip bootstrap：

```text
v17 W-MPJPE  441.8 mm, 95% CI [348.5, 551.3]
v17 WA-MPJPE 276.0 mm, 95% CI [220.7, 342.4]
v17 Accel      94.4,    95% CI [63.2, 132.8]
v17 ATE-Sim3    0.0226 m, 95% CI [0.0179, 0.0267]
```

两个新序列上，v17 相对 parent 的 W、WA、Accel、RTE、ATE、Seam-root 均为 2/2 序列改善。由于序列数只有 2，exact sign randomization 的最低双侧 p 值只能为 0.5；不能写“统计显著”。

Multi-THuMBS 文献数值（如其 Harmony4D W/WA/Accel 221.0/116.9/17.4）使用不同且未公开完全 manifest/evaluator，只能作为背景，不能与本表宣称同协议胜负。目前也不能说已经“打过 Multi-THuMBS”。

## 6. 可复现产物

- v16 原始全量表：`output/v16_harmony4d/test_batch/paper/`
- v17 seen-test regression：`output/v17_harmony4d/seen_test_regression/paper/`
- train10 结果：`output/v17_harmony4d/new_holdout/per_sequence/10_karate2.json`
- train11 结果：`output/v17_harmony4d/new_holdout/per_sequence/11_karate3.json`
- 新留出最终表：`output/v17_harmony4d/new_holdout/paper/`
- 每 case GPU cache/runtime：`output/v17_harmony4d/new_holdout/predictions/`
- 预注册：`versions/v17/HARMONY4D_V17_HOLDOUT_PREREGISTRATION_20260819.md`
- Train10 GO 决策：`versions/v17/HARMONY4D_V17_TRAIN10_GO_NO_GO_20260819.md`

代码提交：

```text
c5932e4 eval(v16): batch frozen Harmony4D test protocol
88b014d eval(v16): record method-independent unavailable cases
aacffbd feat(v17): add prediction-only multicue trust gate
1b15f22 docs(v17): freeze train10 holdout go decision
```

## 7. 下一步优先级

1. 冻结 v17，不再用 Harmony test/train10/train11 调参数；
2. 增加至少 4 个未见动作序列，使序列级统计可检验；
3. 在 EgoHumans、EgoBody 复用同一 CS150 协议，形成跨数据集主表；
4. 把当前 oracle/known boundary 评测补成自动 detector 端到端表；
5. 补运行时/FPS/峰值显存和 gate risk-coverage；
6. Multi-THuMBS 只有拿到相同 manifest/evaluator 或官方可执行输出后才能做直接结论。
