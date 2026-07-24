# V13 Phase 2: Multi-Human Boundary Fusion Optimization

历史任务名：`V20 Phase 2`

实验日期：2026-07-24

## 1. 目标

Phase 1 已证明，在 strict GT-ID 下，多个人共同生成一个 shared Boundary，明显优于
first/largest/highest-confidence 等单人选择器。Phase 2 只研究一个问题：是否存在
无需训练、严格流式、固定预算，而且稳定优于 naive mean 的 fusion。

本阶段固定：

```text
frozen Human3R
+ pre-decode hard reset
+ Fixed Explicit
+ V16 torso rotation, 20 deg bound
+ s = 1
```

关闭 DA3、Keypoint R-CNN、V11.4 scale、VGGT、continuity、token Re-ID 和 scene
refinement。GT identity 只用于严格关联，GT camera/SMPL-X 只进入 evaluator。

## 2. 实现

主入口：

```text
versions/v13/experiments/fusion_optimization.py
```

脚本直接复用 Phase 1 的 raw Human3R cache 和 strict V2 identity assignment，不重新
运行 Human3R。每个人仍先独立产生：

```text
rotation candidate R_i
translation candidate t_i
quality / visibility / motion / dispersion features
```

所有方法最终只输出一个 `R,t`，并共同应用到 camera、pointmap 和所有人体。

比较内容包括：

1. highest-confidence single；
2. Oracle Best Single；
3. naive `SO(3) mean(R_i) + mean(t_i)`；
4. 只融合 translation；
5. 只融合 rotation；
6. 在统一 rotation 下重算 translation；
7. 原 confidence weighting；
8. quality、visibility、motion、candidate dispersion 和 layout 的连续 soft weighting；
9. leave-one-out、人数消融和 candidate feature correlation。

没有 hard reject，也没有训练 selector。soft rule 只在 development timestamps 上选择，
然后固定到其余 timestamp-held-out cases。

## 3. `three` 完整结果

数据为 Phase 1 的 315 cuts；308 个 cuts 至少有两人可做融合。Development timestamps
为 500/700/900，held-out timestamps 为 1000/1100/1300/1500。

| Method | N | Camera T | Rotation | Composite | Composite P90 |
|---|---:|---:|---:|---:|---:|
| Highest-confidence single | 315 | 0.616 | 9.90 | 0.814 | 1.314 |
| Oracle Best Single | 315 | 0.493 | 7.01 | 0.633 | 0.935 |
| Naive multi mean | 308 | 0.517 | 7.01 | **0.657** | **0.977** |
| Translation-only consensus | 308 | 0.615 | 9.90 | 0.814 | 1.380 |
| Rotation-only consensus | 308 | 0.588 | 7.01 | 0.728 | 1.120 |
| Shared-R recomputed translation | 308 | 0.569 | 7.01 | 0.710 | 1.061 |
| Existing confidence weighted | 308 | 0.569 | 7.39 | 0.717 | 1.077 |
| Dev-selected soft rule | 308 | 0.518 | 7.02 | 0.658 | 0.977 |

所有方法 catastrophic rate 均为 0%。

### 3.1 多人收益来自哪里

在 308-case common support 上，以 highest-confidence single 为基线：

- translation-only 的 composite 为 `0.814`，没有改善，paired `p=0.985`；
- rotation-only 降到 `0.728`，改善率 62.0%，`p=4.36e-6`；
- rotation + translation 的 naive mean 降到 `0.657`，改善率 74.0%，
  `p=1.20e-16`。

因此，多人最清楚的独立贡献是 rotation ambiguity reduction。Translation 候选不能在
保持单人 rotation 时直接平均；它只有与多人 rotation 一起融合时才产生额外收益。

### 3.2 Soft uncertainty 是否有效

Development 选择出：

```text
soft_shared_translation_consensus_tau50_raw_t
```

但在 180 个 held-out cases 上：

| Method | Composite mean | P90 |
|---|---:|---:|
| Naive mean | **0.647** | 0.948 |
| Dev-selected soft rule | 0.650 | **0.943** |

mean 退化 `+0.0027`，paired `p=0.589`；rotation 退化 `+0.091 deg`。虽然 P90 略好，
但没有同时满足 mean 和 P90 的成功标准，因此不能替换 naive mean。

### 3.3 为什么不能按 residual 删除人

单人误差与 rotation/translation deviation 有中等相关性，Spearman rho 分别为
`0.349/0.342`；但是它们与 leave-one-out removal gain 的相关性为 `-0.462/-0.441`。
也就是说，离群候选有时单独误差较大，但删除它并不一定让融合变好。这正好解释了
Phase 1 中 hard reject 明显退化。

Quality、score、completeness 和 motion 对“谁是 best single”的命中率只有约
37%-49%，不足以作为稳定 selector。Candidate dispersion 与 naive mean 相对单人的
收益反而正相关：translation/rotation mean dispersion 的 rho 为 `0.443/0.375`。
冲突越大时，多人平均往往越有价值，而不是越应该删人。

### 3.4 人数

所有三人可用的 common support 上：

| 人数 | Evaluations | Composite mean | P90 |
|---:|---:|---:|---:|
| 1 | 828 | 0.843 | 1.315 |
| 2 | 732 | 0.681 | 1.024 |
| 3 | 212 | **0.611** | **0.920** |

人数增加仍然呈单调改善。

## 4. `dance` 两人跨序列 pilot

Phase 1 runner 已扩展为 `--sequence three|dance|box`。`dance` 使用独立的六相机视频、
calibration、两个人的 SMPL-X 和 parameter。测试 6 个时间点、3 个 camera pair、
offset 0/4，共 36 cuts：

- 25 cuts 同时检测到两人，可做多人 fusion；
- 11 cuts 只有一人有效，自动退化为单人；
- 无 catastrophic failure。

在 25-case multi-human support 上：

| Method | Composite mean | P90 |
|---|---:|---:|
| Highest-confidence single | 0.809 | 约 1.30 |
| Naive two-human mean | **0.745** | **1.187** |
| Dev-selected quality soft rule | 0.762 | 1.252 |

held-out 部分 naive mean 为 `0.844`，dev-selected soft rule 退化到 `0.883`。样本量较小，
naive 对单人的 paired 差异尚不显著，但方向与 `three` 一致；soft weighting 再次未通过。

静态 `two_closely_inter/two_naturally_interactive` 只有离散 OBJ/SMPL-X 扫描，没有同步
连续 RGB、相机标定和逐帧外参，因此不能用于本阶段 streaming camera-cut Boundary
评测。它们可用于离线遮挡/mesh 几何诊断，但不能代替 `dance`。

## 5. 最终决策

1. 多人 fusion 稳定优于可部署单人 anchor，`three` 和 `dance` 方向一致。
2. 主要收益来自 rotation；joint fusion 才能进一步改善 translation/human placement。
3. Naive mean 有效是因为误差在多个人之间近似抵消，并且保留所有独立约束。
4. 当前 quality、visibility、motion、dispersion 和 layout 都不是可靠的 uncertainty cue。
5. 没有训练-free soft rule 在 held-out 上同时改善 mean、P90 和 catastrophic rate。
6. **V13 默认 fusion 继续保留 naive mean。**
7. 下一步不应继续堆叠手工 residual；应先做跨 shot WHO/Re-ID，同时收集更多独立
   sequence，再重新评估 uncertainty calibration。

完整结果：

```text
output/v13/phase2_fusion/v13_phase2_fusion.json
output/v13/phase2_fusion/v13_phase2_fusion.md
output/v13/dance_phase2/v13_gtid_offsets_0_4.json
output/v13/dance_phase2/fusion/v13_phase2_fusion.json
```
