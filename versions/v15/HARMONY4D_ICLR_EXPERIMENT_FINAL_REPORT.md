# Movie3R-v15 × Harmony4D：ICLR 专项实验最终报告

日期：2026-08-19
协议：`Movie3R-Harmony4D-CrossShot-v1`
冻结主方法：`M15 safe boundary permutation + causal GRU`
正式 test runtime 提交：`3d022495bd9f3e870ef4de924e7a939042f33887`
全局 test manifest SHA256：`9c5cacfadb7a50d2618415b119c286cf23582c2d185e58b768323302e86638d2`

---

## 1. 结论先行

Harmony4D 专项实验已经完成冻结 test 的全部 GPU 推理和论文级统计：

- 7 个 test sequences；
- 28 个预注册 cross-shot cases；
- 每例 75 pre + 75 post，共 150 帧；
- small / medium / large / extreme 各 7 例；
- 28/28 runtime 成功；
- 28/28 cache SHA256 复核通过；
- 25/28 完整 evaluator report；
- 3/28 为同一种明确 evaluator-unavailable 特例；
- 10,000 次 sequence bootstrap；
- 20,000 次 paired permutation test；
- 17 个同-forward 方法行全部保留。

最终答案不是“当前完整 Movie3R 已经全面打败 Human3R/Multi-THuMBS”，而是一个更具体、也更重要的结论：

1. **B0 的 camera gauge 作用真实且跨序列稳定。** 相对 clean reset，W、WA、ATE、边界相机误差和 seam 均显著改善；相对 strict Human3R，Sim(3)-aligned ATE 从 `0.1387 m` 降到 `0.0197 m`。
2. **Boundary-Permutation ID 是当前最确定的新贡献。** 相对 B0 native ID，IDs/clip 从 `4.60` 降到 `3.84`，IDF1 从 `0.501` 升到 `0.612`，coverage 完全不变，且差异显著。
3. **冻结的 BRTC/C1/adaptive fine alignment 没有在 Harmony4D test 上形成有效主结果。** 安全 gate 在 28/28 cases 上全部 abstain，所以 M15 的几何结果实际等价于 `B0 + Boundary-Permutation ID`。
4. **相对 strict Human3R，当前 M15 的人体世界轨迹和 seam 反而更差。** W-MPJPE 为 `544.8 vs 466.1 mm`，Seam-root 为 `0.869 vs 0.375 m`。因此当前结果不能支撑“完整 camera-human fine alignment 已解决”的论文主张。
5. **Multi-THuMBS 只能做文献量级参考。** 其 exact Harmony4D manifest/evaluator 未公开，不能把两张表直接当 leaderboard 判胜负。

这批数据可以直接用于论文中的正式、可复现实验表和 failure analysis；但如果要把 Harmony4D 放进 ICLR 主表并主张完整方法优于 Human3R，仍需开发一个真正改善 W/seam、且能在一部分 case 安全接受的 fine-alignment 模块。当前 test 已读，后续不能继续用这 28 例调参。

---

## 2. 实验合同

### 2.1 数据

正式 test 使用 Harmony4D 官方 test 包中的：

```text
01_hugging
03_grappling2
05_sword2
06_sword3
08_ballroom2
15_mma4
16_mma5
```

每个 sequence 固定一组 small / medium / large / extreme camera pair。所有 capture、cut、camera pair 只由 GT calibration、visibility、投影有效性和固定 seed 决定，在任何 test forward 之前冻结。

协议文件：

```text
versions/v15/harmony4d/protocols/h4d_cs150_test.jsonl
```

### 2.2 因果与无泄漏合同

28 个 runtime 全部满足：

```text
tracked_worktree_dirty = false
gt_in_runtime = false
future_frames_at_boundary = 0
pre_cut_frames_mutated = false
same_forward_ablations = true
methods per cache = 17
```

所有正式 runtime 均来自同一提交：

```text
3d022495bd9f3e870ef4de924e7a939042f33887
```

### 2.3 方法行

同一个 compact cache 同时保存：

| ID | 方法 | 作用 |
|---|---|---|
| M0 | Strict Human3R | 原始 recurrent carry baseline |
| M1 | Clean reset | 每个 shot 独立重建 |
| M2 | No-V9 raw SE(3) | 不使用 learned shadow gauge |
| M3 | B0 only | learned coarse gauge |
| M4 | B0 + frozen ID | 原匿名关联策略 |
| M5 | B0 + ID + BRTC | 无条件人体 root/depth 修正 |
| M6 | M5 + C1 | 增加 shot 内稳定 |
| M7 | Full v15 oracle | 原 full 方法，上界 detector |
| M10 | OS-BRTC oracle | 可观测性安全分支 |
| M13 | B0 + Boundary-Permutation ID | 新冻结 ID 策略 |
| M14 | M13 + safe fine alignment | oracle boundary |
| M15 | M14 + causal GRU | 默认部署/论文候选 |
| M16 | M14 + static detector | detector 消融 |

---

## 3. 完整性和 evaluator 特例

### 3.1 完整性

```text
expected cases        = 28
runtime caches        = 28
valid metric reports  = 25
cache hashes verified = 28
provenance errors     = 0
hard errors           = 0
```

机器可读审计：

```text
output/v15_harmony4d/paper/test/test_audit.json
```

### 3.2 三个透明跳过的 case

以下 3 个 `ballroom2` case 的 GPU 推理与 cache 都完整：

```text
h4d_test_08_ballroom2_009_ballroom2_large_cam07_cam08_b00359
h4d_test_08_ballroom2_009_ballroom2_medium_cam03_cam05_b00359
h4d_test_08_ballroom2_009_ballroom2_small_cam04_cam17_b00359
```

它们在 evaluator 的前两帧都没有通过冻结的 `2.0 m` camera-coordinate assignment gate，因而无法建立 shared initial world fit：

```text
ValueError: No initial matched people for shared world fit
```

处理原则：

- 不重跑 GPU；
- 不替换 camera pair；
- 不删除 runtime/cache；
- 不在看过 test 后放宽匹配阈值；
- 主表固定使用 25 个可评估 case；
- 单独报告 evaluator-unavailable rate：`3/28 = 10.7%`；
- `ballroom2` 的 sequence mean 仅来自 extreme case，必须在正文或附录标注。

这比事后修改 evaluator 得到更漂亮的数值更符合 test freeze 原则。

---

## 4. 主结果

以下均为 25 个可评估 case 的 clip macro。它们属于我们的公开协议，不是 Multi-THuMBS 官方复现。

| 方法 | W ↓ | WA ↓ | MPJPE ↓ | MPVPE ↓ | Accel ↓ | Coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Strict Human3R | **466.1** | **262.7** | 93.9 | 111.8 | **93.59** | **90.8%** |
| Clean reset | 720.7 | 408.5 | 93.3 | 111.0 | 103.03 | 90.6% |
| No-V9 raw SE(3) | 720.5 | 407.8 | 93.3 | 111.0 | 102.97 | 90.6% |
| B0 only | 544.8 | 265.8 | **93.3** | **111.0** | 97.99 | 90.6% |
| B0 + frozen ID | 537.7 | 266.5 | 94.4 | 112.4 | 98.23 | 87.4% |
| B0 + ID + BRTC | 611.7 | 311.6 | 102.0 | 122.0 | 116.97 | 84.8% |
| B0 + ID + BRTC + C1 | 611.6 | 311.5 | 102.0 | 122.0 | 116.94 | 84.8% |
| Full v15 oracle | 611.6 | 311.5 | 102.0 | 122.0 | 116.94 | 84.8% |
| B0 + Boundary-Permutation ID | 544.8 | 265.8 | **93.3** | **111.0** | 97.99 | 90.6% |
| **M15 Movie3R causal** | 544.8 | 265.8 | **93.3** | **111.0** | 97.98 | 90.6% |

M15 的 95% sequence-bootstrap CI：

| 指标 | Mean | 95% CI |
|---|---:|---:|
| W-MPJPE | 544.8 mm | [421.1, 706.9] |
| WA-MPJPE | 265.8 mm | [223.7, 311.3] |
| MPJPE | 93.35 mm | [89.77, 105.87] |
| MPVPE | 111.02 mm | [106.33, 127.55] |
| Accel | 97.98 mm/frame² | [69.66, 134.09] |
| ATE Sim(3) | 0.0197 m | [0.0167, 0.0231] |
| IDs/clip | 3.84 | [2.11, 5.64] |
| IDF1 | 0.612 | [0.423, 0.713] |
| Coverage | 0.906 | [0.640, 0.953] |

---

## 5. 相机和身份

| 方法 | ATE-Sim3 ↓ | ATE-SE3 ↓ | Boundary-t ↓ | Boundary-R ↓ | IDs/clip ↓ | IDF1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 0.139 | 0.460 | **0.882** | 9.95° | 4.40 | 0.535 |
| Clean reset | 1.601 | 2.156 | 4.477 | 79.27° | 4.60 | 0.501 |
| No-V9 raw SE(3) | 1.359 | 2.150 | 4.489 | 79.28° | 4.60 | 0.501 |
| B0 only | **0.020** | **0.371** | 1.115 | **9.43°** | 4.60 | 0.501 |
| B0 + Boundary-Permutation ID | **0.020** | **0.371** | 1.115 | **9.43°** | **3.84** | **0.612** |
| **M15 Movie3R causal** | **0.020** | **0.371** | 1.115 | **9.43°** | **3.84** | **0.612** |

### 5.1 相对 strict Human3R

- ATE-Sim3：`0.1387 → 0.0197 m`，25/25 case 改善，permutation `p < 5e-5`；
- ATE-SE3：`0.4599 → 0.3705 m`，差异不显著；
- Boundary-R：`9.95° → 9.43°`，差异不显著；
- Boundary-t：`0.882 → 1.115 m`，反而更差，permutation `p=0.0103`。

因此准确表述应是：

> B0 显著恢复跨 shot camera trajectory 的 Sim(3) gauge consistency，但尚未改善 first-post 的 metric translation。

不能简化成“相机所有指标都更准”。

### 5.2 相对 B0 native ID

- IDs/clip：`4.60 → 3.84`；
- IDF1：`0.501 → 0.612`；
- IDs paired permutation：`p=0.00040`；
- IDF1 paired permutation：`p=0.00050`；
- coverage：`0.9056 → 0.9056`，完全不变；
- camera、W、WA、seam：与 B0 等价。

这是当前 Harmony4D 上最干净、最适合作为论文贡献的结果。

---

## 6. 人体世界轨迹和 seam

| 方法 | Boundary root ↓ | Boundary CHRGE ↓ | Pair vector ↓ | Seam-t ↓ | Seam-R ↓ | Seam-root ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Strict Human3R | **0.439** | **0.498** | **0.291** | **0.767** | 4.13° | **0.375** |
| Clean reset | 1.293 | 0.666 | 1.755 | 4.325 | 78.01° | 1.275 |
| B0 only | 0.954 | 0.666 | 0.302 | 0.945 | **3.80°** | 0.869 |
| Unconditional BRTC | 0.968 | 0.671 | 0.537 | 0.945 | **3.80°** | 0.840 |
| **M15 Movie3R causal** | 0.954 | 0.666 | 0.302 | 0.945 | **3.80°** | 0.869 |

M15 相对 strict Human3R：

- W-MPJPE：`+78.7 mm`，permutation `p=0.0327`，显著变差；
- WA-MPJPE：`+3.1 mm`，不显著；
- Seam-root：`+0.509 m`，permutation `p=0.00035`，显著变差；
- Coverage：`-0.2` percentage point，不显著。

这说明 B0 解决的是 shot gauge 的一部分，不等于人体在 fixed world 中已经精确连续。当前最关键的未解决问题仍是：

> 在保持正确 camera-human relative geometry 的同时，如何修正 first-post metric translation、人体 root/world trajectory 和 seam。

---

## 7. BRTC、C1 和 adaptive gate 的最终判定

### 7.1 无条件 BRTC 是 No-Go

相对 B0 + ID：

```text
W       537.7 → 611.7 mm
WA      266.5 → 311.6 mm
MPJPE    94.4 → 102.0 mm
MPVPE   112.4 → 122.0 mm
Coverage 87.4% → 84.8%
```

Harmony4D 的多人接触、检测数量变化和 root-ray 不稳定使无条件 BRTC 产生明显 harm。

### 7.2 C1 没有形成可测贡献

M5、M6、M7 的结果几乎相同。C1 在这些变量可见性/身份条件下大多 disabled 或 fallback，不能据此宣称 Harmony4D shot 内稳定已被解决。

### 7.3 安全 gate 的行为

```text
OS-BRTC accepted                    = 0 / 28
Boundary-permutation adaptive accept = 0 / 28
evaluable harmful accept            = 0 / 25
catastrophic harmful accept          = 0 / 25
```

它成功避免了无条件 BRTC 的灾难更新，但代价是没有任何实际 fine-alignment gain。论文中可以写“precision-first abstention is safe”，不能写“adaptive joint correction improves Harmony4D”。

---

## 8. Detector

28 个 runtime 上，两个 RGB-only 流式 detector 都得到：

| Detector | TP | FP | FN | Precision | Recall | F1 | First-positive=boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| Causal GRU | 28 | 0 | 0 | 1.000 | 1.000 | 1.000 | 28/28 |
| Static logistic | 28 | 0 | 0 | 1.000 | 1.000 | 1.000 | 28/28 |

平均 Brier：

```text
causal GRU       3.07e-5
static logistic  2.74e-7
```

限制：这些 case 是预注册的人工 cross-camera transition，不能把 28/28 直接外推成任意真实电影剪辑的 detector 泛化能力。真实电影上的 natural editing benchmark 仍需另测。

---

## 9. 角度和序列规律

### 9.1 角度

| Angle | N | Human3R ATE | M15 ATE | Human3R IDF1 | M15 IDF1 | Human3R W | M15 W |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 6 | 0.072 | 0.016 | 0.682 | 0.683 | 441.1 | **419.9** |
| medium | 6 | 0.086 | 0.017 | **0.618** | 0.584 | **394.9** | 474.9 |
| large | 6 | 0.103 | 0.024 | 0.459 | **0.571** | **430.5** | 587.1 |
| extreme | 7 | 0.271 | 0.021 | 0.404 | **0.609** | **579.1** | 675.6 |

规律：

- M15 的 ATE-Sim3 对视角跨度稳定；
- ID 优势在 large/extreme 更明显，符合 Boundary-Permutation ID 的设计目标；
- 人体 W 在 medium/large/extreme 没有随 camera gauge 改善，说明 camera gauge 与 human world trajectory 仍然解耦。

### 9.2 序列

M15 的 W 只在 `hugging`、`ballroom2-extreme` 和 `mma4` 平均有改善；在 `grappling2`、`sword2`、`sword3`、`mma5` 上变差。IDF1 在多数序列改善，但 `grappling2` 下降、`ballroom2` 持平。

这进一步说明当前方法最稳的是 camera gauge 和 boundary ID，不是通用人体精对齐。

---

## 10. Multi-THuMBS 文献参考

| 方法 | W | WA | MPJPE | MPVPE | Accel | ATE | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multi-THuMBS（文献） | 221.0 | 116.9 | 215.9 | 278.3 | 17.4 | 0.7 | 0.46 |
| HSfM†（文献） | -- | -- | 225.6 | **257.6** | 28.3 | 3.2 | 1.58 |
| M15（我们的协议） | 544.8 | 265.8 | 93.3 | 111.0 | 97.98 | 0.0197 | 3.84 |

只能得到量级判断：

- 我们的 W、WA、Accel、IDs 明显没有达到 Multi-THuMBS 文献线；
- 我们的 pelvis-aligned MPJPE/MPVPE 和 Sim3 ATE 数值更低；
- 由于 clip、camera pair、visibility、matching、topology、W/WA/ATE/IDs 公式都未官方对齐，不能声称后四项正式超越。

论文建议把该表标为：

> Literature reference; protocol matched only as far as public information.

而不是 leaderboard 表。

---

## 11. 运行效率

硬件：`NVIDIA L20`，FP32。

28-case mean：

| Component | Mean |
|---|---:|
| Causal GRU RGB detector | 100.72 s / 150 frames |
| Strict Human3R forward | 41.97 s / 150 frames |
| Shadow proposal | 21.00 s / 76 frames |
| Clean post forward | 21.34 s / 75 frames |
| Explicit geometry | 10.56 s / boundary |
| Full experiment process | 517.89 s / case |
| Peak VRAM | 5.33 GiB |
| Peak process RAM | 12.62 GiB |

`total_process_s` 包括两个 detector、M0、shadow/raw forward、17 个方法的 common-SMPL packing、压缩 cache、SHA256 和 Python 开销，不是单个 M15 的部署延迟。

Multi-THuMBS 公开运行参考为 150 帧、RTX 3090、约 10 分钟。我们的完整实验进程均值约 8.63 分钟，但硬件、分辨率、保存开销和任务模块不同，不能据此正式宣称速度领先。可写的系统事实是：

- peak VRAM 约 5.33 GiB；
- boundary geometry 无未来帧；
- 所有 case 可断点续跑；
- detector 和显式几何仍有明显 CPU 优化空间。

---

## 12. 定性结果

定性选择固定覆盖：

1. `hugging-extreme`：大视角、ID 完全修正、成功案例；
2. `sword3-extreme`：ID 改善但 W 变差的 stress case；
3. `ballroom2-extreme`：低 coverage、safe abstention、真实失败；
4. `ballroom2-large`：evaluator-unavailable failure。

每例导出 5 pre + 25 post，共 30 帧，分别包含 strict Human3R 和 M15。payload 使用正式 cache 中的 camera、mesh、persistent ID；RGB/depth/confidence 只由冻结 checkpoint 因果重算以恢复 demo.py 背景，不读取 GT、不改变指标。

最终导出 4 cases × 2 methods = 8 个 payload、共 240 帧；camera、depth、confidence、RGB、SMPL 文件数和有限值检查全部通过，总大小约 682 MiB。

```text
output/v15_harmony4d/qualitative/selection.json
output/v15_harmony4d/qualitative/demo_payloads/
```

正式 test 中 adaptive accept 为 0/28，因此不存在可诚实展示的“adaptive accept success”；报告明确标为 unavailable。

---

## 13. ICLR claim ledger

### 13.1 当前数据支持

| Claim | 状态 | 证据 |
|---|---|---|
| 方法是 causal / online boundary transaction | 支持 | 28/28 runtime contract |
| runtime 不使用 GT、不看未来帧 | 支持 | provenance audit 0 error |
| B0 比 clean reset 恢复 camera gauge | 强支持 | W/WA/ATE/boundary 全部显著改善 |
| B0 的 Sim3 camera trajectory 优于 strict Human3R | 强支持 | 0.1387→0.0197 m，25/25 改善 |
| Boundary-Permutation ID 优于 B0 native ID | 强支持 | IDs、IDF1 显著且 coverage 不变 |
| 安全 gate 能避免 BRTC 灾难更新 | 支持 | 0 harmful accept；exact fallback |
| 视角越大，ID 模块越有价值 | 支持 | large/extreme IDF1 分层结果 |

### 13.2 当前数据不支持

| Claim | 状态 | 原因 |
|---|---|---|
| Full Movie3R 全面优于 Human3R | 不支持 | W、seam-root 显著更差 |
| Harmony4D 上人体 fine alignment 已解决 | 不支持 | safe gate 0/28 accept |
| BRTC/C1 在 Harmony4D 有稳定收益 | 不支持 | BRTC harm，C1 近似无变化 |
| first-post metric camera translation 更准 | 不支持 | Boundary-t 反而更差 |
| 正式超越 Multi-THuMBS | 不支持 | 协议不公开且 W/WA/IDs 有明显差距 |
| adaptive low-texture correction 已在 Harmony 验证 | 不支持 | 0 accepted case |

---

## 14. 对 ICLR 的最终判断

### 14.1 这批 Harmony4D 结果能怎么用

可以直接用于：

- protocol/data setup；
- causal detector 表；
- B0/no-V9/reset ablation；
- Boundary-Permutation ID 主消融；
- camera gauge 与 human gauge 解耦的 failure analysis；
- safe abstention 结果；
- angle/sequence 分层；
- runtime/resource 表；
- appendix literature reference。

### 14.2 不能怎么用

不能把 M15 行加粗后写成：

> Movie3R solves online camera-human alignment and outperforms Human3R/Multi-THuMBS on Harmony4D.

当前数据会直接反驳这个表述。

### 14.3 论文故事需要收缩或继续做方法

如果现在投稿，最可信的故事是：

> Camera cuts expose two separable failures in streaming 3D reconstruction: camera gauge discontinuity and identity permutation. A causal learned gauge bridge plus one-shot boundary permutation reliably fixes these two axes, while geometry-gated abstention exposes the remaining human-world alignment bottleneck.

这是一个扎实但比原计划更窄的贡献。若仍希望保留“完整 camera-human joint correction”作为论文中心，下一版本必须满足：

1. 在新的 train/dev 上改善 W 和 seam-root；
2. 相对 M13/B0 有非零、可重复的 accepted gain；
3. accepted case 不牺牲 coverage/MPJPE；
4. rejected case exact fallback；
5. 使用新的、从未读过的 held-out test，不再用本报告 28 例调参。

---

## 15. 可复现产物

### 15.1 聚合

```text
output/v15_harmony4d/paper/test/aggregate.json
output/v15_harmony4d/paper/test/case_metrics.csv
output/v15_harmony4d/paper/test/main_table.tex
```

### 15.2 论文表和图

```text
output/v15_harmony4d/paper/test/artifacts/main_human_table.tex
output/v15_harmony4d/paper/test/artifacts/camera_identity_table.tex
output/v15_harmony4d/paper/test/artifacts/boundary_table.tex
output/v15_harmony4d/paper/test/artifacts/detector_table.tex
output/v15_harmony4d/paper/test/artifacts/gate_table.tex
output/v15_harmony4d/paper/test/artifacts/efficiency_table.tex
output/v15_harmony4d/paper/test/artifacts/literature_reference_table.tex
output/v15_harmony4d/paper/test/artifacts/angle_sensitivity.pdf
output/v15_harmony4d/paper/test/artifacts/sequence_generalization.pdf
output/v15_harmony4d/paper/test/artifacts/paper_results.json
```

### 15.3 代码

```text
versions/v15/harmony4d/audit_test_results.py
versions/v15/harmony4d/aggregate_harmony.py
versions/v15/harmony4d/build_paper_artifacts.py
versions/v15/harmony4d/export_harmony_qualitative.py
```

---

## 16. 最终冻结决定

Harmony4D-v1 实验本身冻结为完成：

- 不再调整这 28 个 test case；
- 不再用它们改 gate/threshold；
- 保留所有成功和失败 cache；
- M15 在该协议上的真实解释固定为：

```text
B0 learned coarse camera gauge
+ Boundary-Permutation persistent ID
+ safe abstention
```

而不是已经生效的 fine camera-human correction。

Harmony4D-v1 的最重要科学产出，是把项目从“看起来多人 demo 很好”推进到一个可量化结论：**相机 gauge 和人物 ID 已有明确进展；人体 fixed-world 精对齐仍是 ICLR 主线的最后关键缺口。**

---

## 17. 数据清理与保留状态

正式推理、审计和定性导出完成后，已于 2026-08-19 清理所有可由原始压缩包重建的临时数据：

```text
removed  /data/wangzheng/iJCV-CODE/data/Harmony4D_work/staging  (106G)
removed  /data/wangzheng/iJCV-CODE/data/Harmony4D_work/tmp       (28G)
removed  Harmony4D_work/logs/arxiv_2607.01626_source.tar.gz
kept     /data/wangzheng/iJCV-CODE/data/Harmony4D.zip           (328G)
kept     /data/wangzheng/iJCV-CODE/data/Harmony4D_work/logs
kept     output/v15_harmony4d/                                  (30G)
```

清理前 `/data` 可用约 `190 GiB`，清理后约 `323 GiB`。该操作没有删除 prediction cache、metrics、论文表图或 demo payload；若需重建原始帧，可从保留的外层 ZIP 按冻结 staging 脚本恢复。
