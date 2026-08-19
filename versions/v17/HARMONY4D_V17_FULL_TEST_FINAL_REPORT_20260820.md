# Movie3R-v17：Harmony4D 官方 test 全 capture 最终报告

日期：2026-08-20  
状态：**全量运行完成；v17 冻结；v18 候选否决；结果可直接用于论文表格，但必须保留协议差异和 test-selection 披露。**

## 1. 最终结论

冻结的 `Movie3R-v17 MultiCue-Safe` 已完成 Harmony4D 官方 test split 的 7 个动作包、全部可用 capture 和固定四档镜头跨度测试。正式 runner 返回：

```text
FULL_TEST_EXIT_CODE=0
7 action sequences
28 structurally eligible captures
25 coordinate-valid captures
100 preregistered cases
88 evaluable cases
12 method-independent evaluator-unavailable cases
0 inference failures
28 gate accepts / 60 exact parent fallbacks
```

相对 `v17 parent`，v17 在 88 个同 manifest、同 evaluator 的 case 上取得：

- W-MPJPE：618.45 → **584.00 mm**，改善 **5.57%**；
- WA-MPJPE：287.06 → **266.86 mm**，改善 **7.04%**；
- Accel：97.96 → **90.14 mm/frame²**，改善 **7.98%**；
- RTE-H3R：7.180% → **6.584%**，改善 **8.29%**；
- ATE-Sim3：0.01964 → **0.01895 m**，改善 **3.52%**；
- ATE-SE3：0.31401 → **0.26005 m**，改善 **17.18%**；
- Boundary-root：1.0314 → **0.9069 m**，改善 **12.07%**；
- Post-root：0.8842 → **0.7947 m**，改善 **10.13%**；
- Seam-root：0.8761 → **0.7135 m**，改善 **18.56%**；
- Jitter：483.59 → **445.29**，改善 **7.92%**；
- Foot sliding：6.380 → **6.109 cm**，改善 **4.25%**。

局部人体质量几乎不变：MPJPE/MPVPE 仅恶化 0.03%/0.03%，PA-MPJPE 恶化 0.006%。IDF1 从 0.636105 变为 0.635975，绝对变化 -0.000130；Coverage 不变。该结果支持的准确论文主张是：

> Movie3R-v17 在不牺牲局部人体重建质量和覆盖率的情况下，稳定改善跨 shot 的世界坐标人体、相机、边界连续性和时序稳定性。

它不支持“所有指标全面击败 Strict Human3R”或“已经同协议击败 Multi-THuMBS”。

## 2. 冻结方法

最终方法文件：

```text
versions/v17/harmony4d/frozen_multicue_candidate.json
SHA256 ae2bc503ca5abdb0735abea231b6b953c0463d2e33f96741469191ad71adbafe
```

冻结配置：

```text
boundary correction             shared translation
boundary_blend                  1.0
root_alpha / root_beta          0.50 / 0.02
gate_min_matches                2
gate_max_boundary_residual_m    0.25
gate_max_translation_m          1.60
failure policy                  exact parent fallback
```

每个输入为 150 帧在线两-shot 序列：75 帧 pre + 75 帧 post。Human3R 主干先输出相机、人体和 recurrent state；B0 与 boundary permutation 提供共同粗坐标和人物槽位；人体锚定的共同边界平移同时作用于 post 的相机和人体，从而保持相机—人体相对几何；因果 root stabilization 抑制 shot 内漂移；prediction-only trust gate 决定采用修正或回退 parent。推理不读取 Harmony4D GT、标定或最终指标。

## 3. 数据与排除审计

### 3.1 动作覆盖

| Action | 结构合格 capture | 坐标有效 capture | Manifest case | 可评测 | 统一排除 |
|---|---:|---:|---:|---:|---:|
| 01_hugging | 1 | 1 | 4 | 4 | 0 |
| 03_grappling2 | 7 | 7 | 28 | 28 | 0 |
| 05_sword2 | 2 | 2 | 8 | 8 | 0 |
| 06_sword3 | 6 | 6 | 24 | 24 | 0 |
| 08_ballroom2 | 4 | 4 | 16 | 4 | 12 |
| 15_mma4 | 1 | 1 | 4 | 4 | 0 |
| 16_mma5 | 7 | 4 | 16 | 16 | 0 |
| **合计** | **28** | **25** | **100** | **88** | **12** |

`16_mma5` 的 `009/011/013` 三个 capture 在任何 Movie3R 推理前即因公开标定的 `cam15` PnP 重投影审计失败而排除。`08_ballroom2` 的 12 个 case 在统一 GT evaluator 中均报 `No initial matched people for shared world fit`；四种方法同时不可评测。所有排除由方法无关规则产生，没有查看某一方法的好坏后删例。

### 3.2 固定协议

- 每例固定 150 帧，不根据动作或结果改成 60/90/120 帧；
- 每个 capture 固定 small/medium/large/extreme 四个相机旋转跨度；
- known/oracle shot boundary 与自动 detector 结果分开；
- 主表使用 clip macro，同时报告 action macro；
- 95% CI 使用 10,000 次 action-then-clip hierarchical bootstrap；
- 显著性检验的最高独立单元是 7 个 action，不把同一 capture 的四个镜头对当作独立样本；
- 全 fallback action 按方法契约将 v17-parent 差值严格置零，避免 evaluator 的约 `1e-5` mm 浮点噪声制造虚假显著性。

## 4. 论文主表

以下均为 88-case clip macro。W/WA/MPJPE/MPVPE/Accel 的单位为 mm（Accel 为 mm/frame²），RTE 为百分比，ATE 为 m。

| Method | W ↓ | WA ↓ | MPJPE ↓ | MPVPE ↓ | Accel ↓ | RTE ↓ | ATE-Sim3 ↓ | ATE-SE3 ↓ | IDF1 ↑ | IDs ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict Human3R | **496.6** | **259.6** | 98.3 | 116.7 | 96.2 | **5.90** | 0.1420 | 0.4445 | 0.553 | 3.82 |
| Movie3R-v15 | 618.5 | 287.3 | **97.9** | **116.1** | 98.0 | 7.18 | 0.0204 | 0.3200 | **0.636** | **3.07** |
| v17 parent | 618.5 | 287.1 | **97.9** | **116.1** | 98.0 | 7.18 | 0.0196 | 0.3140 | **0.636** | **3.07** |
| **Movie3R-v17** | **584.0** | **266.9** | **98.0** | **116.1** | **90.1** | **6.58** | **0.0189** | **0.2601** | **0.636** | **3.09** |

相对 Strict Human3R，v17 的 W/WA/RTE 仍分别差 17.6%/2.8%/11.5%；但 MPJPE、MPVPE、Accel、ATE-Sim3、ATE-SE3、IDF1 和 IDs 分别改善 0.34%、0.51%、6.26%、86.66%、41.50%、15.07% 和 19.05%。因此论文应强调“联合跨镜头一致性与身份/相机优势”，而不是把单一 W 指标包装成全面 SOTA。

## 5. 动作、镜头跨度与安全性

### 5.1 分动作泛化

| Action | Accept / cases | W 改善 | WA 改善 | Accel 改善 | ATE-SE3 改善 | Seam-root 改善 |
|---|---:|---:|---:|---:|---:|---:|
| hugging | 3 / 4 | 13.53% | 17.68% | 18.08% | 19.85% | 24.06% |
| grappling2 | 7 / 28 | 2.95% | 4.79% | 5.39% | 9.12% | 12.75% |
| sword2 | 3 / 8 | 18.96% | 14.74% | 12.54% | 30.28% | 35.85% |
| sword3 | 14 / 24 | 10.22% | 14.25% | 20.28% | 48.40% | 52.44% |
| ballroom2 | 0 / 4 | exact fallback | exact fallback | exact fallback | exact fallback | N/A |
| mma4 | 0 / 4 | exact fallback | exact fallback | exact fallback | exact fallback | exact fallback |
| mma5 | 1 / 16 | 1.05% | 1.24% | 1.14% | 3.01% | 1.38% |

五个实际触发修正的 action 在 W、WA、Accel、RTE、ATE-SE3 和 Seam-root 的 action macro 上全部改善；其余两个 action 完整回退。结果不是由单一动作独占。

### 5.2 分镜头跨度

| Stratum | Cases | Accept | W 改善 | WA 改善 | Accel 改善 | ATE-SE3 改善 | Seam-root 改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 21 | 11 | 2.09% | 1.30% | 12.40% | 22.95% | 27.22% |
| medium | 21 | 4 | 2.90% | 4.51% | 5.34% | 5.45% | 13.27% |
| large | 21 | 6 | 7.26% | 12.36% | 6.42% | 16.04% | 18.41% |
| extreme | 25 | 7 | 7.55% | 7.37% | 8.26% | 23.32% | 20.96% |

四个跨度层均有收益，且 large/extreme 并未破坏稳定性。这是方法适合 multi-shot 大视角切换的关键证据。

### 5.3 Gate 风险

28 个 accepted case 相对 parent：

- W：22 改善 / 6 恶化，平均改善 108.27 mm，最差恶化 73.35 mm；
- WA：25 改善 / 3 恶化，平均改善 63.47 mm；
- Accel 与 Jitter：28/28 改善；
- RTE：23 改善 / 5 恶化；
- ATE-SE3 与 Seam-root：均为 27 改善 / 1 恶化；
- IDF1 总体只下降 0.00013，但 accepted 中有 4 个轻微下降；IDs 有 2 个恶化、1 个改善、25 个不变。

这说明 gate 的平均风险收益明显为正，但不是“每个 accepted case 都改善”。论文应报告 22/28 的 W success rate 和最坏 harm，不能只报均值。60 个 fallback 与 parent 按方法契约相同。

## 6. 不确定性与显著性

主表是 clip macro；下列中心值是 action macro，与 hierarchical bootstrap 的估计单位一致：

| Metric | v17 action macro | 95% CI |
|---|---:|---:|
| W-MPJPE | 577.68 mm | [404.53, 744.08] |
| WA-MPJPE | 256.12 mm | [200.21, 309.87] |
| Accel | 90.44 mm/frame² | [62.19, 119.52] |
| ATE-Sim3 | 0.02053 m | [0.01608, 0.02511] |

相对 parent，五个真正非 fallback action 的主要跨镜头指标方向一致；把两个全 fallback action 严格视为零差异后，7-action 双侧 exact randomization 的最小可信 p 值为 0.0625。因此当前可写“跨五个触发动作方向一致、具有稳定总体收益”，不应写 `p<0.05`。扩大到 EgoHumans/EgoBody 的独立动作后再做跨数据集显著性检验。

## 7. 超参与长度实验的最终判定

### 7.1 固定减小边界平移不泛化

v18 在开发集把 `boundary_blend` 从 1.0 改为 0.75，仅得到约 0.08% 综合改善。候选冻结后在三个独立 Harmony4D holdout 动作上相对 v17：

- W：改善 0.17%；
- WA：恶化 2.56%；
- Accel：恶化 2.74%；
- ATE-SE3：恶化 9.70%；
- Seam-root：恶化 16.85%。

收益只来自 MMA，Sword 明显恶化，因此 v18 已正式否决。最终保留 `boundary_blend=1.0`，不根据 action 选择 blend。

### 7.2 60/90/120 帧只能作为长度消融

短窗口会天然缩短误差累积时间，不能与 150 帧主表混用。开发集上 60/90/120 帧虽降低部分 W，但 60/90 帧的 Accel 明显变差，90 帧还降低 IDF1；120 帧较均衡却未达到预注册的 5% 综合提升门槛。因此：

- 正文固定 150 帧，体现真正的长时在线稳定性；
- 60/90/120 帧放附录，表述为 context length–quality trade-off；
- 不允许按 test case 挑最有利长度。

## 8. 与 Multi-THuMBS 的关系

Multi-THuMBS 论文在其 Harmony4D 协议下报告 W/WA/Accel 为 221.0/116.9/17.4，当前 Movie3R 同名数值为 584.0/266.9/90.1，不能声称在这些数值上打过它。Movie3R 的 MPJPE/MPVPE/ATE 原始数值更低，但 manifest、轨迹对齐、身份处理和 evaluator 不同，同样不能作为正式胜负。

当前可发表的差异化优势是：在线流式、跨 shot、同时评估相机—人体—身份—边界连续性、具有 prediction-only adaptive gate 和 exact fallback。若要正式比较 Multi-THuMBS，必须拿到其可执行输出或在完全相同 manifest/evaluator 下重跑。

## 9. ICLR 证据状态与下一步

Harmony4D 已从“小规模主观样例”提升为可复现的 7-action、88-case 定量证据，足以进入论文的主要消融/数据集表。但它仍不是整篇 ICLR 论文的终点：

1. 本官方 test 的早期单-capture 子集曾用于发现 v16 失败，因此完整 88-case 表必须标注 `test_used_for_parameter_selection=true`，应称为 exhaustive frozen regression；
2. 两个 candidate-freeze-held-out Harmony4D action 的 8-case 结果仍作为无偏支持证据保留；
3. 下一优先级是在未触碰的 EgoHumans/EgoBody 上复用同一 CS150 协议，形成真正跨数据集主表；
4. 补 automatic detector 端到端表，并与 known-boundary 表分开；
5. 补运行时间/FPS/峰值显存、长多-shot 流和 scene non-regression；
6. 在论文中同时报告 accepted harm、fallback 和 Coverage，避免只展示成功 case。

## 10. 可复现产物

```text
output/v17_harmony4d/full_test/paper/summary.json
output/v17_harmony4d/full_test/paper/SUMMARY.md
output/v17_harmony4d/full_test/paper/main_table.tex
output/v17_harmony4d/full_test/paper/case_metrics.csv
output/v17_harmony4d/full_test/per_sequence/*.json
output/v17_harmony4d/full_test/staging/*/full_capture_manifest.jsonl
output/v17_harmony4d/full_test/logs/tmux_master.log
```

原始 `Harmony4D.zip` 保留不变。逐动作展开数据在结果验证后已删除；临时工作目录仅剩约 116 KB 审计日志。预测与指标缓存约 43 GB，位于 `/data` 下，用于复现、可视化和后续审计。
