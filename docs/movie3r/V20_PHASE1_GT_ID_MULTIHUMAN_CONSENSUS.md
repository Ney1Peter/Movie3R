# V20 Phase 1: GT-ID Multi-Human Consensus Alignment Feasibility Study

> Historical invalidated report. Do not use its old geometry conclusion.
> The corrected current result is
> `docs/movie3r/V20_PHASE1_GT_ID_MULTIHUMAN_CONSENSUS_V2.md`, released as
> **Movie3R-Multi V20.0** under `versions/v20-multi/`.

## 0. 2026-07-24 Identity Audit Addendum

**原 `Phase 1 geometry gate: FAIL` 结论已撤回，等待修复 GT-ID assignment 后重跑。**

在案例 `three_t0900_c0_c3_k0` 中发现，原评测并没有获得严格的 GT-ID。它使用每帧独立的 GT 投影 bbox 与 Human3R bbox Hungarian assignment；遮挡、bbox 重叠和 Human3R root/depth 偏移会导致身份交换。该案例的明确证据是：

```text
pre predicted left order:  person1, person0, person2
pre GT left order:         person1, person0, person2
post GT left order:        person1, person0, person2
post old assignment:       person0, person1, person2
```

交换 post-cut `person0/person1` 后：

| Assignment | Candidate T dispersion | Candidate R dispersion | Camera T | Camera R | Composite |
|---|---:|---:|---:|---:|---:|
| Old `P0/P1/P2 <- D0/D1/D2` | 3.458 m | 116.86 deg | 4.947 m | 165.34 deg | 8.254 |
| Corrected `P0/P1/P2 <- D1/D0/D2` | 0.458 m | 11.71 deg | 0.695 m | 7.67 deg | 0.848 |

六种 post-cut permutation 的无 GT camera 几何一致性评分也选择了 corrected permutation（`1.657`，旧 permutation 为 `2.619`）。因此，原报告中的大范围 180 度冲突至少有一部分是 identity-assignment confound，不能继续解释为“GT-ID 下多人几何本身失败”。

后续必须：

1. 将 bbox-only assignment 替换为经过审计的 GT-ID association；
2. 对全部 315 cuts 重新生成身份映射和指标；
3. 在重跑完成前，不启动 token Re-ID，也不宣称多人 consensus 成立或失败；
4. 本文以下旧结果只保留为历史诊断，不再作为路线决策依据。

## 1. 原结论（已撤回）

**历史结论：Phase 1 geometry gate: FAIL。当前无效，原因见上面的 identity audit。**

在已经知道 `person0/person1/person2` 身份的理想条件下，当前多人几何共识仍没有超过单人上界 `Oracle Best Single`，鲁棒共识也没有超过简单平均。同步 cut 已经是最有利条件，因此当前没有足够证据继续把 Human3R token Re-ID 接入多人 Boundary alignment。

这不是说“多人没有信息”，而是说当前各人的 Boundary candidate 不是独立、零均值的小噪声。它们会形成相关的错误分支：常见情况是两个人同时给出接近 180 度的错误 rotation，只有一个人正确。mean、Huber、geometric median 和 layout majority 都会相信错误多数，甚至把正确人物删除为 outlier。

目前更准确的研究问题是：

> 如何从多个候选中识别正确 anchor，而不是如何平均多个 anchor。

在没有可靠的、与当前人体 root/orientation 错误相独立的选择信号之前，不建议把多人 consensus 合入 Movie3R 主方法。

## 2. 实验范围

### 2.1 数据

- 数据集：`MultiHuman/Real-World-Capture/extracted`
- 序列：`three`
- 人数：3，稳定 GT identity 为 `person0/person1/person2`
- 相机：6 个同步视角
- 原始视频：2048 x 2048
- Human3R 输入：完整画面缩放为 512 x 512，不裁剪
- 历史：cut 前同一 source camera 的 5 帧
- cut 后：target camera 的 1 帧，解码前 fresh hard reset

采样了 7 个时间点：

```text
500, 700, 900, 1000, 1100, 1300, 1500
```

每个时间点使用 9 个 camera pairs：

```text
0->1, 1->2, 2->3, 3->4, 4->5, 5->0,
0->3, 1->4, 2->5
```

共完成：

- same timestamp `k=0`：63 cuts；
- temporal `k=1/2/4/8`：各 63 cuts；
- 总计：315 cuts。

### 2.2 Lite 约束

开启：

- frozen Human3R；
- post-cut pre-decode hard reset；
- Fixed Explicit coarse alignment；
- V16 torso-motion yaw refinement；
- 固定 20 度 V16 bound；
- explicit root translation；
- 一个 shared shot-level Boundary。

关闭：

- DA3；
- Keypoint R-CNN；
- V11.4 shared scale；
- VGGT；
- continuity memory；
- token Re-ID；
- learned identity adapter；
- 额外 scene refinement。

因此，本实验隔离的是“增加人体数量及其鲁棒融合”这一项因素。

## 3. GT 使用边界

GT identity 通过 GT SMPL-X mesh 投影框与 Human3R mesh bbox 的 Hungarian assignment 获得。它只回答检测结果对应 `person0/person1/person2` 中的哪一个。

| 变量 | Candidate generation | Evaluation |
|---|---:|---:|
| GT identity | 是，Oracle association | 是 |
| GT SMPL-X mesh projection | 仅用于 identity assignment | 是 |
| GT camera | 否 | 是 |
| GT world root/joints/vertices | 否 | 是 |
| GT Boundary | 否 | 仅 Oracle/evaluation |
| Source/camera ID 作为 learned cue | 否 | 仅定义输入 pair |

所有可部署 candidate 的 `R_i,t_i` 都只读取 Human3R 预测、历史 root/torso、当前 fresh root/torso、置信度和 pointmap。

`Oracle Best Single` 会在每个 cut 后根据 GT evaluator 选择误差最小的人，只是理论上界，不可部署。

## 4. 几何实现

### 4.1 每人候选

对匹配人物 `i`：

1. 在 cut 前 Human3R world gauge 中，用最近 5 帧 root 的鲁棒速度预测目标 anchor：

```text
a_i = root_i(last) + delta_frame * robust_velocity_i
```

2. 从历史 torso frame 预测 post 时刻 torso orientation。

3. post-cut 图像在 hard reset 后 fresh 解码，得到 camera/shot-local root `r_i`、root rotation 和 torso frame。

4. 用历史 root orientation 与当前 root orientation 构造 Fixed Explicit initial transform，并用背景 pointmap 做局部 coarse refinement。

5. 用 V16 torso residual 只修正 bounded yaw，最大 20 度，得到 `R_i`。

6. 显式求 translation：

```text
t_i = a_i - R_i * r_i
```

### 4.2 Shared Boundary

所有多人方法最终只能输出一组：

```text
B = [R, t]
```

同一组 `B` 作用于 post-cut camera 和所有人物。没有 per-person world transform，也没有 BA、未来帧或逐帧重估。

### 4.3 比较方法

- `single_first`
- `single_largest`
- `single_highest_confidence`
- `oracle_best_single`
- `naive_mean`：分别平均 `R_i,t_i`
- shared-rotation mean
- confidence weighted mean
- SO(3) geometric median + translation median
- SO(3) geometric median + translation geometric median
- coordinate trimmed mean
- Huber rotation/translation
- layout candidate selection
- layout selection + at most one outlier rejection

最后一种方法按 predicted pairwise root layout、torso consistency 和 translation consistency 给各真实 candidate rotation 打分，选择内部最一致分支；最多删除一个异常人物并重新求解一次。

## 5. 评价定义

Evaluator 先用 pre-cut camera 定义合法公共 gauge：

```text
G = C_pred_pre * inverse(C_gt_pre)
C_target = G * C_gt_post
C_final = B * C_pred_post
```

GT human root、joints、vertices 同样乘 `G`，再与 `B` 变换后的 Human3R 输出比较。

Camera composite 仅用于排序与 gate：

```text
composite = translation_error_m + 0.02 * rotation_error_deg
```

Catastrophic 定义：

```text
translation > 2 m OR rotation > 45 deg
```

## 6. Same-Timestamp 主结果

`k=0` 共 63 cuts，其中 62 个至少有 2 个可用人物 candidate。

| Method | N | Camera T mean/P90 | Camera R mean/P90 | Composite mean/P90 | Catastrophic |
|---|---:|---:|---:|---:|---:|
| Single first | 63 | 1.407 / 3.980 | 40.88 / 168.60 | 2.225 / 7.541 | 23.8% |
| Single highest confidence | 63 | 1.280 / 3.425 | 34.52 / 117.04 | 1.970 / 5.740 | 25.4% |
| Oracle Best Single | 63 | **0.555 / 0.791** | **8.62 / 15.80** | **0.727 / 1.076** | **1.6%** |
| Naive mean | 62 | 1.042 / 2.497 | 36.48 / 155.78 | 1.772 / 5.654 | 21.0% |
| Rotation/translation median | 62 | 1.291 / 3.797 | 37.35 / 165.27 | 2.038 / 7.176 | 19.4% |
| Robust Huber | 62 | 1.294 / 3.756 | 36.06 / 161.71 | 2.016 / 7.232 | 22.6% |
| Layout + one reject | 62 | 1.339 / 3.770 | 38.67 / 165.27 | 2.113 / 7.222 | 25.8% |

关键 paired 结果：

- Layout primary 相比 Oracle Best Single：composite 平均增加 1.385；只有 16.1% 改善，83.9% 退化，Wilcoxon `p=1.22e-8`。
- Layout primary 相比 naive mean：composite 平均增加 0.341；27.4% 改善，72.6% 退化，`p=0.00187`。
- Huber 相比 naive mean：composite 平均增加 0.244；29.0% 改善，71.0% 退化，`p=0.00095`。

因此 robust/layout 方法不是“尚未显著优于”，而是在当前协议下显著更差。

## 7. Human 与多人布局

Same-timestamp human absolute error：

| Method | Root | Joints | Vertices | Pairwise vector |
|---|---:|---:|---:|---:|
| Highest confidence | 0.472 | 0.494 | 0.483 | 0.429 |
| Oracle Best Single | **0.407** | **0.428** | **0.418** | **0.314** |
| Naive mean | 0.736 | 0.754 | 0.744 | 0.467 |
| Robust Huber | 0.470 | 0.487 | 0.478 | 0.454 |
| Layout + one reject | 0.495 | 0.519 | 0.507 | 0.456 |

Pairwise distance error 约为 0.118 m，并且对各 Boundary 方法几乎相同。这是正常现象：刚性变换不会改变两人距离，因此 pairwise distance 本身不能选择 world Boundary。Pairwise vector 可以约束 rotation，但在多个错误候选形成一致错误分支时仍无法判断绝对方向。

Naive mean 的 camera translation 均值较好，但 human root/joint/vertex 明显变差，原因是分别平均 `R_i,t_i` 后不再严格满足每个人的 root anchor equation。这也说明不能仅凭 camera 指标选择多人融合方式。

## 8. 人数消融

为避免不同有效样本集合混淆，以下只使用同一批 43 个三人均可用的 `k=0` cuts，并枚举该人数的所有子集：

| 人数 | Evaluations | Composite mean/P90 | Catastrophic |
|---:|---:|---:|---:|
| 1 | 129 | 2.400 / 7.368 | 31.8% |
| 2 | 129 | 2.403 / 7.725 | 29.5% |
| 3 | 43 | 2.462 / 7.640 | 30.2% |

同步条件下，人数从 1 增加到 2 或 3 没有降低 mean 或 P90。

在全部 offsets 的同一批 222 个三人 cases 上：

| 人数 | Composite mean/P90 | Catastrophic |
|---:|---:|---:|
| 1 | 2.316 / 7.247 | 27.9% |
| 2 | 2.242 / 7.393 | 24.9% |
| 3 | 2.260 / 7.353 | 26.1% |

两人使平均值和 catastrophic 略降，但 P90 反而上升，而且仍远差于 Oracle Best Single。这不满足“多人特别改善 tail/catastrophic”的成功标准。

## 9. Leave-One-Out 与人物质量

全部 offsets、222 个三人 cases 的两人 Huber leave-one-out：

| 设置 | Composite mean/P90 | Catastrophic |
|---|---:|---:|
| minus person0 | 1.979 / 5.719 | 22.1% |
| minus person1 | 2.235 / 7.201 | 24.3% |
| minus person2 | 2.513 / 7.726 | 28.4% |

`person2` 在该序列中总体更有价值，但这是数据集和人物特定现象，不能转化为可部署的固定 ID 规则。在多个严重失败 case 中，`person2` 恰好是唯一正确候选，却被错误多数判成 outlier。

候选质量分析：

- 高于全局 median quality 的 candidate composite 为 1.942，低质量为 2.325；
- catastrophic 为 21.3% 对 27.7%；
- quality 与误差的 Spearman 相关只有 -0.089；
- head score 的相关为 -0.176；
- completeness 的相关接近 0（-0.002）；
- highest-confidence 选中 Oracle Best identity 的比例只有 39.3%（全部 offsets）。

所以质量筛选有弱收益，但不足以识别 rotation flip。

## 10. Layout 与 Outlier Rejection

Same-timestamp：

- translation candidate pairwise dispersion mean：1.629 m；
- rotation candidate pairwise dispersion mean：49.15 度；
- 28 个 cases 触发一次 reject；
- 被删除者确实是 GT-evaluated worst single 的比例只有 46.4%；
- layout selector 命中 Oracle Best identity：41.9%；
- first/highest-confidence 的命中率也为 41.9%。

全部 offsets：

- layout selector 命中 Oracle Best identity：39.3%；
- first：41.9%；
- largest：36.7%；
- highest confidence：39.3%。

因此当前 layout verification 没有提供超过简单启发式的 self-verification 能力。

## 11. 失败样本解剖

最稳定的失败簇出现在：

```text
t0900 c3->c4
t0900 c0->c3
t1000 c3->c4
t1100 c2->c3
t1100 c3->c4
t1100 c0->c3
t1300 c4->c5
t1300 c5->c0
t1300 c2->c5
```

典型模式以 `t0900 c0->c3` 为例：

- person0：rotation error 165.3 度；
- person1：rotation error 174.1 度；
- person2：rotation error 12.9 度；
- Oracle Best Single 选择 person2；
- layout majority 认为 person0/person1 更一致，删除 person2；
- final primary composite 为 8.254。

对这些 case 的阶段诊断显示：

- 错误通常已经存在于 per-human root-orientation initial；
- Fixed pointmap refinement 很少改变这个 180 度分支；
- V16 只允许最多 20 度 yaw，不可能修复 160-180 度错误；
- translation 通过 `t_i=a_i-R_i r_i` 依赖 rotation，因此 rotation flip 同时放大 translation error。

Same-timestamp 168 个 per-human candidates 中：

- 141 个完成 Fixed pointmap refinement；
- 27 个背景点不足，显式退化为 human-only initial；
- 只有 3 个 V16 residual 触发 20 度 clipping。

全部 845 个 per-human candidates 中：

- 692 个完成 pointmap refinement；
- 153 个背景点不足；
- 61 个 V16 residual 被 clipping。

这说明主失败不是 V16 bound 经常截断了正确小修正，而是更早的 coarse/root orientation 存在离散错误分支。

## 12. Temporal Cut 诊断

下表中 `N` 是至少有两人、可运行 consensus 的数量：

| Offset | N | Oracle Best C mean/P90 | Naive mean C mean/P90 | Huber C mean/P90 | Layout C mean/P90 |
|---:|---:|---:|---:|---:|---:|
| 0 | 62 | 0.728 / 1.076 | 1.772 / 5.654 | 2.016 / 7.232 | 2.113 / 7.222 |
| 1 | 62 | 0.771 / 1.107 | 1.746 / 5.680 | 1.988 / 7.288 | 2.100 / 7.216 |
| 2 | 62 | 0.997 / 1.219 | 1.806 / 5.680 | 2.052 / 7.277 | 2.153 / 7.242 |
| 4 | 61 | 0.716 / 1.087 | 1.694 / 5.715 | 1.833 / 6.632 | 2.019 / 7.189 |
| 8 | 61 | 0.777 / 1.107 | 1.751 / 5.710 | 1.818 / 6.262 | 1.961 / 7.093 |

结论：

- `k=1/2/4/8` 没有让多人 consensus 超过单人 Oracle；
- 结果没有随 offset 单调恶化，说明 8 帧内 motion extrapolation 不是当前第一瓶颈；
- 同一 rotation-flip case 会在多个 offset 稳定出现；
- `k=4/8` 各有 2 个 cuts 只剩 1 个共享人物，需要退回 single-human fallback。

因为 `k=0` 已经失败，temporal 结果只作为 motion 敏感性诊断，不用于挽救 geometry gate。

## 13. GT-ID Assignment 限制与敏感性

post-cut GT projection assignment 的 bbox IoU：

- mean：0.287；
- median：0.293；
- 低于 0.1：7.6%；
- 低于 0.2：22.9%。

遮挡和重叠使 bbox association 并非完美。为检查结论是否完全由身份赋值质量造成，又在高 IoU 子集上复核：

- 同步、所有 post assignment IoU >= 0.25 的 10 cases 中，Huber composite 0.784，naive mean 0.904，Oracle 0.738；样本很少，且 Huber 仍未超过 Oracle。
- 全 offsets、所有 post assignment IoU >= 0.25 的 61 cases 中，layout mean 0.999，naive mean 1.004，但 P90 为 1.583 对 1.241，仍不满足 mean + tail gate，也远未建立稳定优势。

所以低 IoU 是正式 benchmark 前必须改进的限制，但不能解释本轮主要负结果。

## 14. 对核心问题的回答

### 1. 已知人物身份时，多人是否比单人更适合作为 Boundary anchor？

当前否。多人比固定 first-person 偶尔更好，但明显不如 Oracle Best Single；人数增加也不稳定改善 mean 和 P90。

### 2. 多人收益来自更多约束、rotation、translation，还是异常检测？

当前只观察到相对固定 first-person 的小幅平均收益，没有可靠 tail 收益。真正潜力主要在异常检测/anchor selection，而现有 layout 和 confidence 还不能实现它。主要失败首先是 rotation branch，随后通过显式方程污染 translation。

### 3. 简单平均是否足够？

不够。Naive mean 在 camera composite 上反而优于当前 Huber/layout，但 catastrophic 仍约 21%，P90 rotation 约 156 度，并且 human root/joints/vertices 明显变差。更复杂的 robust consensus 也没有解决错误多数。

### 4. 多少个人开始产生收益？

同步公平子集上，1/2/3 人 composite mean 为 2.400/2.403/2.462，没有人数收益。全 temporal 子集两人平均值略好，但 P90 更差，三人也没有继续改善。

### 5. 多人 alignment 是否值得进入最终 Movie3R？

当前不值得。按预先定义的停止规则，应停止“多人 consensus 改善 Boundary”以及后续 token-Re-ID-to-alignment 集成。单人 Movie3R 主路线保持不变。

如果以后重启该方向，前置硬条件应改为：先找到 capture-disjoint、无需 GT、能可靠识别 180 度错误 branch 的 anchor selector 或独立几何 cue，再重新运行 GT-ID gate。不能继续只更换 mean/median/Huber 形式。

## 15. 可复现实验产物

实现：

```text
scripts/v20_phase1_gt_id_multihuman_consensus.py
```

Same-timestamp 报告：

```text
output/v20_phase1_gt_id_multihuman_consensus/v20_phase1_offsets_0.json
output/v20_phase1_gt_id_multihuman_consensus/v20_phase1_offsets_0.md
```

全部 offsets：

```text
output/v20_phase1_gt_id_multihuman_consensus/v20_phase1_offsets_0_1_2_4_8.json
output/v20_phase1_gt_id_multihuman_consensus/v20_phase1_offsets_0_1_2_4_8.md
```

每个 cut 的 Human3R cache：

```text
output/v20_phase1_gt_id_multihuman_consensus/case_cache/
```

重评命令：

```bash
.venv/bin/python scripts/v20_phase1_gt_id_multihuman_consensus.py \
  --evaluation_only \
  --timestamps 500 700 900 1000 1100 1300 1500 \
  --camera_pairs 0-1 1-2 2-3 3-4 4-5 5-0 0-3 1-4 2-5 \
  --offsets 0 1 2 4 8
```

## 16. 路线冻结建议

本阶段对应预设的失败情况：

> GT-ID 多人没有超过单人上界，且 Re-ID 尚未进入实验。

因此冻结以下决策：

1. 不把 current multi-human consensus 加入 Lite 或 Full；
2. 不因本实验启动 token Re-ID alignment integration；
3. 保留该实验作为多人几何负结果和 anchor-selection 诊断；
4. 原 single-human short-shot alignment 继续作为当前可用方法；
5. 多人输入仍可由 Human3R 重建和显示，但不代表多人已用于可靠 Boundary 求解。
