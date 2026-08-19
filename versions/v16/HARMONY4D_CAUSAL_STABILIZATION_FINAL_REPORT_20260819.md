# Movie3R-v16 × Harmony4D：因果联合稳定专项最终报告

日期：2026-08-19  
冻结方法：`Movie3R-v16-Harmony-Safe`  
方法提交：`298a789bbb0dae4c3090e35de62b520b3eb07317`  
评测工具提交：`bdee6b1`  
协议：`Movie3R-Harmony4D-CrossShot-v1`，每例 75 pre + 75 post

## 1. 最终结论

本轮任务已得到一个能够在未见 Harmony4D 序列上稳定提升 M15 的可用版本，而不是只在开发序列上有效的后处理。

在预注册、此前完全未用于 v16 调参的 `train/09_karate / capture015` 上，4 个 small/medium/large/extreme case 全部完成 150 帧 GPU 推理和评测：

| 指标 | M15 | v16 Harmony-Safe | 变化 |
|---|---:|---:|---:|
| W-MPJPE ↓ | 593.6 mm | **499.2 mm** | **−15.9%** |
| WA-MPJPE ↓ | 295.2 mm | **255.0 mm** | **−13.6%** |
| Accel ↓ | 116.12 | **86.14** | **−25.8%** |
| Seam-root ↓ | 0.864 m | **0.433 m** | **−49.9%** |
| Boundary-root ↓ | 1.074 m | **0.732 m** | **−31.9%** |
| Post-root ↓ | 0.980 m | **0.818 m** | **−16.5%** |
| MPJPE ↓ | 107.0 mm | **106.4 mm** | −0.5% |
| MPVPE ↓ | 122.4 mm | **121.8 mm** | −0.5% |
| ATE-Sim3 ↓ | 0.0180 m | **0.0159 m** | −11.9% |
| IDF1 ↑ | 0.806 | **0.809** | +0.0035 |
| Coverage ↑ | 89.17% | **89.17%** | 完全不变 |

最终 4/4 case 的 W-MPJPE 均不变差；3/4 case 接受精对齐，1/4 因边界匹配不足自动、逐数组精确回退 M15。没有删除普通难例，也没有根据 GT 或最终误差选择是否更新。

因此本轮成功标准全部达到：

```text
W-MPJPE       改善 >= 10%    实际 15.9%
Accel         改善 >= 15%    实际 25.8%
seam-root     改善 >= 15%    实际 49.9%
MPJPE/MPVPE   变差 <= 5%     实际均改善约 0.5%
ATE-Sim3      恶化 <= .005 m 实际改善 .0021 m
IDF1          下降 <= .01    实际上升 .0035
coverage      不下降         实际完全相同
```

## 2. 最终方法

最终方法可概括为：

> 在 M15/B0 已完成相机粗 gauge 和跨 shot 人物 ID 匹配后，用人体边界一致性估计一个相机—全体人体共享的世界平移；只有预测残差足够小才执行，再用逐人、逐帧的因果 root 滤波消除 shot 内抖动。

### 2.1 输入

运行时只使用：

1. 当前 RGB 帧和已经到达的历史帧；
2. M15/B0 输出的相机 `camera-to-world`；
3. 每个人的 joints、vertices、native slot；
4. 边界人物匹配对。

不读取 Harmony4D 标定、人体 GT、未来帧或最终评测误差。

### 2.2 Boundary-Permutation ID

沿用 v15 已冻结的边界一次置换：利用 B0 后可建立的 pre/post 对应关系，将 post 的 native slots 映射到 pre 的 persistent IDs；后续帧继续按 native slot 传播。该步骤不改变任何相机或人体几何，并保持所有检测。

### 2.3 Human-Anchored Coupled Boundary Registration（HCBR）

在 cut 到来时：

1. 取最后一帧 pre 与第一帧 post 中已匹配人物；
2. 用 pelvis 和 torso joints 计算每对人物的 root 偏移；
3. 对多人的偏移取稳健中位数，得到一个共同平移；
4. 同一个平移同时作用于 post 相机和所有 post 人体。

因此 HCBR 对齐的是整个 post 世界 gauge，不会单独把某个人从相机中“拉走”，相机—人体相对位置在这一步严格不变。pre 不修改；之后到来的 post 帧使用同一个已确定变换，不需要回看未来帧。

### 2.4 Prediction-Only Reliability Gate

无条件 HCBR 在 MMA 强接触场景会产生灾难更新，因此最终门控要求：

```text
有效边界人物匹配数 >= 2
共同平移后 torso mean residual <= 0.25 m
```

两项都只由预测计算。任一条件不满足，就返回与 M15 几何逐数组一致的结果；不是删除该 case，也不是用 GT 选较好的分支。

### 2.5 Causal Alpha-Beta Root Stabilization

可靠 HCBR 被接受后，对每个 persistent ID 独立维护 root 位置和速度：

```text
prediction = previous_root + velocity
residual   = observed_root - prediction
root       = prediction + 0.5 * residual
velocity   = velocity + 0.02 * residual
```

只平移当前人体 joints/vertices，不改变身体姿态，不修改已经输出的历史帧。它负责降低 Accel 和 shot 内不应出现的人体漂移。

### 2.6 输出

每帧输出：

```text
稳定后的 camera-to-world
全体人体 world joints / vertices
跨 shot persistent person ID
gate accepted / fallback 诊断
```

## 3. 为什么这是在线方法

边界更新只需要 `pre[-1]` 与 `post[0]`；第一帧 post 到来即可决定是否接受。共同平移一旦确定，只作用于当前与以后到来的 post 帧。root 滤波每帧只维护每个 ID 的位置和速度状态。整个 v16 合同为：

```text
GT used at runtime                 = false
future frames used at boundary    = 0
pre frames rewritten              = false
already emitted frames rewritten  = false
```

150 帧设置沿用 Multi-THuMBS 公开的 150-frame 量级，并让 pre/post 对称为 75/75，既有足够 shot 内轨迹计算 Accel，也避免只展示边界附近几帧造成偶然性。它是我们的冻结 cross-shot 协议，不表示 Multi-THuMBS 未公开的 camera/cut manifest 已完全复现。

## 4. 数据拆分与无泄漏过程

| 阶段 | 数据 | 用途 | 是否用于参数选择 |
|---|---|---|---|
| 探索 | train/01_hugging capture001 | 有限候选和参数网格 | 是 |
| 验证 A | train/03_grappling2 capture008 | 跨动作验证 | 是 |
| 验证 B | train/15_mma4 capture005 | 强接触失败域与 gate | 是 |
| 开发留出 | train/15_mma4 capture014 | gate 冻结后的首次检查 | 否 |
| 最终未见 | train/09_karate capture015 | 最终四 case GPU 推理 | 否 |

最终 `train09` capture 由外层 ZIP 的冻结结构 SHA 顺序选择第一个投影有效 capture；选择发生在任何 Movie3R 预测或指标产生之前。manifest SHA256：

```text
4a9ef5c8433744d8de49b8b00f2a73a8ecc1996f50c26e2e056fb132027e439e
```

4 个正式 runtime 全部来自提交 `298a789`，`tracked_worktree_dirty=false`。

## 5. 探索与验证结果

### 5.1 train01 探索

有限网格先分别验证共同 boundary 校正和 root 滤波，再组合：

| 指标 | M15 | 最终 gated v16 | 变化 |
|---|---:|---:|---:|
| W-MPJPE | 206.3 | 173.7 | −15.8% |
| Accel | 25.84 | 18.55 | −28.2% |
| Seam-root | 0.165 | 0.030 | −82.0% |
| MPJPE | 83.48 | 83.48 | 0 |
| MPVPE | 100.13 | 100.13 | 0 |
| Coverage | 1.0 | 1.0 | 0 |

4/4 case 接受，4/4 W 不变差。

### 5.2 train03 验证

| 指标 | M15 | gated v16 | 变化 |
|---|---:|---:|---:|
| W-MPJPE | 605.9 | 397.6 | −34.4% |
| Accel | 78.82 | 58.02 | −26.4% |
| Seam-root | 0.991 | 0.414 | −58.3% |
| ATE-Sim3 | 0.0173 | 0.0153 | −11.1% |

3/4 case 接受；原先无门控会使 medium 的 W 恶化 16.3%，冻结门控识别其校正后 torso residual 为 0.402 m 并回退。门控后 4/4 W 不变差。

### 5.3 MMA 验证与开发留出

无门控 HCBR 在 `train15/capture005` 会使 W 从 915.7 增到 1385.1 mm（+51.3%），虽然 Accel 和 seam 看起来变好。这证明“接缝更小”不能单独代表世界位置正确。

这些 case 的校正后 torso residual 全部高于阈值：

```text
capture005: 0.264 / 0.329 / 1.027 / 2.922 m
capture014: 0.306 / 0.389 / 0.980 / 2.941 m
```

因此冻结 gate 在 capture005 与此前未读的 capture014 上均 0/4 接受、4/4 exact fallback；所有指标与 M15 相同。它确认了安全性，但也明确 MMA 强接触仍不是当前模块的有效域。

## 6. 最终 train09 逐 case 结果

| Angle | Gate | HCBR residual | W：M15→v16 | Accel：M15→v16 | Seam：M15→v16 |
|---|---|---:|---:|---:|---:|
| extreme | fallback（仅 1 match） | 0.075 m | 869.4→869.4 | 109.1→109.1 | 1.576→1.576 |
| large | accept | 0.121 m | 477.8→379.7 | 121.0→78.9 | 1.067→0.051 |
| medium | accept | 0.076 m | 580.5→395.8 | 103.1→60.9 | 0.486→0.037 |
| small | accept | 0.086 m | 446.7→351.7 | 131.3→95.7 | 0.326→0.068 |

安全回退的 extreme 被完整纳入均值，未作为“难例”跳过。

## 7. 与 Human3R 和 Multi-THuMBS 的关系

### 7.1 同一 train09 协议

| 方法 | W ↓ | WA ↓ | MPJPE ↓ | MPVPE ↓ | Accel ↓ | ATE-Sim3 ↓ | Seam-root ↓ | IDF1 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 524.5 | 365.4 | **106.1** | **121.3** | 133.4 | 0.1701 | 1.587 | 0.713 |
| M15 | 593.6 | 295.2 | 107.0 | 122.4 | 116.1 | 0.0180 | 0.864 | 0.806 |
| **v16 Harmony-Safe** | **499.2** | **255.0** | 106.4 | 121.8 | **86.1** | **0.0159** | **0.433** | **0.809** |

在该未见序列上，v16 首次同时把 M15 的 camera/ID 优势和 Human3R 原本更好的 W 轨迹统一起来：W 比 strict Human3R 低约 4.8%，而 WA、Accel、ATE、seam 和 IDF1 明显更好。局部 MPJPE/MPVPE 与 Human3R 基本相当，但没有足够样本声称普遍显著超越。

### 7.2 Multi-THuMBS 文献量级参考

| 方法 | W | WA | MPJPE | MPVPE | Accel | ATE | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multi-THuMBS（论文公开值） | 221.0 | 116.9 | 215.9 | 278.3 | 17.4 | 0.7 | 0.46 |
| v16（我们的 train09 协议） | 499.2 | 255.0 | 106.4 | 121.8 | 86.1 | 0.0159 | 1.50 |

仍然只能做量级参考：Multi-THuMBS 没有公开完全相同的 sequence/camera/cut/visibility/evaluator manifest。当前 W、WA、Accel 和 IDs 与其公开数值仍有明显差距；MPJPE、MPVPE、ATE 的数值更低，但协议与对齐公式不完全一致，不能写成正式 leaderboard 胜出。

## 8. 统计解释

train09 只有一个预注册 sequence、4 个 camera-pair case。case bootstrap 10,000 次得到：

| 指标差值（v16−M15） | 95% case-bootstrap CI | exact paired sign p |
|---|---:|---:|
| W-MPJPE | [−162.3, −24.5] mm | 0.125 |
| WA-MPJPE | [−58.8, −14.3] mm | 0.125 |
| Accel | [−42.1, −10.5] | 0.125 |
| Seam-root | [−0.827, −0.112] m | 0.125 |

四个 clip 的方向一致，bootstrap CI 不跨零；但 exact permutation 因 `n=4` 最小只能到 0.125，且四例来自同一动作序列。因此本结果证明“冻结候选在预注册未见序列上有效”，尚不能替代多序列正式显著性实验。

## 9. 候选消融结论

1. **HCBR-T 有效。** 它直接负责 W、WA、boundary-root 和 seam 的主体改善。
2. **root alpha-beta 有效。** 单独使用时 W 基本不变，但 Accel 明显下降；与 HCBR 组合后保持正交收益。
3. **无条件 HCBR 不安全。** MMA 上 W +51.3%，必须有可观测性 gate。
4. **0.25 m residual gate 有效。** train03 自动拒绝唯一退化例、MMA 全部回退、train09 接受 3 个受益例。
5. **CSGS-freeze 不进入最终主线。** 它在 Harmony 固定机位上能得到更低 W，但会把 shot 内相机轨迹压成静态，ATE-Sim3 接近数值零，难以作为通用电影移动相机方法的公平主张。
6. **V9/B0 仍作为粗 gauge 与 ID 可匹配性的前置基础。** v16 不宣称 HCBR 可以替代 B0；本轮只解决 B0 后的可信精对齐和 shot 内 root 稳定。

## 10. 已知限制

1. ATE-SE3 从 0.1616 增到 0.1751 m（+0.0135 m），尽管 ATE-Sim3 改善；共同人体锚定平移与 metric camera translation 仍存在权衡。
2. MMA 强接触域全部 fallback，说明错误 ID/遮挡下共同人体锚仍不可观测。
3. gate 是开发集冻结的显式阈值，不是当前版本的可学习置信度头。
4. 最终未见结果只有一个 sequence；下一轮论文主表需要更多未参与 v16 设计的动作序列。
5. 当前专项使用人工构造的 synchronized cross-camera cut，不等价于所有真实电影编辑形式。

## 11. 论文可支持的主张

目前可以诚实写：

> Movie3R-v16 introduces prediction-only, confidence-gated human-anchored coupled registration and causal root stabilization. On a preregistered unseen Harmony4D sequence, it improves M15 world-space trajectory, acceleration, and cross-shot seam by 15.9%, 25.8%, and 49.9%, respectively, while preserving coverage and safely reverting uncertain interactions.

当前不能写：

```text
全面超越 Multi-THuMBS
在所有 Harmony4D 动作上都执行并改善精对齐
解决遮挡或强接触下的通用多人重识别
所有相机 metric translation 指标都改善
```

## 12. 主要产物

```text
方法与 gate
versions/v16/harmony4d/causal_stabilization.py
versions/v16/harmony4d/frozen_harmony_candidate.json

正式 train09 指标
output/v16_harmony4d/final/train09.json
output/v16_harmony4d/final/train09.csv

论文表与统计
output/v16_harmony4d/paper/train09/summary.json
output/v16_harmony4d/paper/train09/paired_case_metrics.csv
output/v16_harmony4d/paper/train09/main_table.tex

标准 demo.py payload
output/v16_harmony4d/qualitative/train09_best/

正式 GPU runtime/cache
output/v16_harmony4d/predictions/train09_line1..4/
```

## 13. 下一步

该 Harmony 专项方法已经完成，下一阶段不应继续在 train09 上调阈值。论文级后续应是：

1. 冻结 v16，在更多未参与设计的 Harmony4D sequences 上批量评测，报告 accept/fallback 分层和 paired significance；
2. 将 v16 作为 M15、Human3R、Multi-THuMBS 文献参考之间的主消融行；
3. 单独研究强接触域的 match-confidence 或多人相对布局一致性，不能在当前 test 上继续手调；
4. 把可学习 gate 作为后续增强，而不是替换已经验证有效的显式安全版本；
5. 在论文中同时报告成功域、fallback 域和 ATE-SE3 权衡。

本轮任务的最终状态：**达到预注册成功门槛，得到可用于后续大规模 Harmony4D 论文实验的冻结方法。**

## 14. 数据清理与恢复

指标、预测缓存和可视化全部完成后，已删除以下可重建 staging：

```text
Harmony4D_work_v16/staging/train_01_hugging   2.4G
Harmony4D_work_v16/staging/train_03_grappling2 20G
Harmony4D_work_v16/staging/train_15_mma4       12G
Harmony4D_work_v16/staging/train_09_karate     14G
Harmony4D_work_v16/tmp
```

约释放 47GB；它们可由保留的 `/data/wangzheng/iJCV-CODE/data/Harmony4D.zip`（328GB）和 `output/v16_harmony4d/staging/*` 中的 ledger/manifest 重新恢复。原始 ZIP、全部正式 runtime/cache、指标、统计和 demo payload 均未删除。
