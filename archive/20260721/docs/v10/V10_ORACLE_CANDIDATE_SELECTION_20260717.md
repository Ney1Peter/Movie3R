# V10 显式候选 Oracle Selection 实验

日期：2026-07-17

## 1. 实验目的

RICH 当前没有可用相机 GT，因此本实验改用 AvatarReX、THuman 和 MVHuman 的 GT camera pose，验证：

1. 多种显式 SE(3) 候选是否具有互补性；
2. 固定 Explicit 方法失败时，正确结果是否已经存在于候选集合中；
3. 是否已经值得训练 Token + Geometry Candidate Selector。

Human3R 完全冻结。AABB 的边界由 GT 直接给出，cut 后 reset recurrent state。每个候选只在边界处计算一次 SE(3)，后续 B 段固定复用。GT 不参与候选生成，只用于最后计算误差和 Oracle 选择。

## 2. 数据

共评测 180 个 AABB case：

| 数据源 | 数量 | 角度范围 |
|---|---:|---|
| AvatarReX | 48 | 60-180 度四档 |
| MVHuman100 | 48 | 60-180 度四档 |
| MVHuman200 | 36 | 60-150 度三档 |
| THuman | 48 | 60-180 度四档 |

每个角度桶确定性抽取 12 个 case。没有样本因数据缺失、相机参数异常或无人检测而失败。

使用的 held-out manifest 映射：

```text
config/manifests/v10_oracle_candidate_selection_gt_sources/manifest_map.json
```

## 3. 固定候选集合

每个 case 生成 12 个固定模板候选：

```text
identity fallback
上一帧人体完整朝向 + root 对齐
历史人体均值完整朝向 + root 对齐
上一帧/历史均值的仅平移对齐
identity + 单帧 pointmap refinement
上一帧人体初值 + 单帧 pointmap refinement
历史人体均值初值 + standard/strict/loose pointmap refinement
等待 B2 后的两帧 pointmap refinement
```

候选集合和计算参数对所有数据源保持一致，没有针对单个样本手工调整。

原先指定的固定 Explicit baseline 是：

```text
human_mean_pointmap_history_standard
```

事后统计发现，所有样本上最强的单一固定候选其实是：

```text
human_last_full_no_refine
```

这说明当前 pointmap refinement 经常发生过度修正，不能默认认为 refinement 一定优于人体粗配准。

## 4. 总体结果

| 方法 | T mean | T median | T P90 | R mean | R median | R P90 | Relaxed success | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前 Fixed Explicit | 1.7047 | 1.4366 | 3.8495 | 23.84 | 13.42 | 58.35 | 5.6% | 66.7% |
| 最强单一固定候选 | 1.5950 | 1.3001 | 3.4682 | 12.93 | 11.28 | 25.41 | 1.1% | 65.6% |
| Joint Oracle Selection | 1.3891 | 1.2570 | 2.9465 | 16.63 | 10.47 | 34.23 | 5.6% | 65.0% |
| Rotation Oracle Selection | 1.5883 | 1.3738 | 3.4258 | 10.61 | 8.50 | 23.64 | 5.6% | 66.1% |
| Boundary Oracle SE(3) | 0.0149 | 0.0050 | 0.0371 | 0.25 | 0.08 | 0.55 | 100% | 0% |

Joint Oracle 使用：

```text
平均平移误差（米） + 平均旋转误差（弧度）
```

相对当前 Fixed Explicit：

```text
平均平移改善：0.3156 m
平均旋转改善：7.22 度
P90 平移改善：0.9031 m
P90 旋转改善：24.12 度
99.4% case 的最优候选不是当前固定候选
```

相对更严格的“最强单一固定候选”：

```text
平均平移改善：0.2060 m
joint cost 改善：0.1415
71.1% case 会选择其他候选
但平均旋转反而增加 3.70 度
```

因此候选互补性成立，但增益没有最初相对弱 baseline 时看起来那么大。

## 5. 分数据源结果

| 数据源 | 当前 Fixed | Joint Oracle | 主要现象 |
|---|---|---|---|
| AvatarReX | 1.259 m / 6.8 度 | 1.187 m / 7.5 度 | 平移小幅改善，旋转没有改善 |
| MVHuman100 | 3.364 m / 42.5 度 | 2.530 m / 33.2 度 | 候选选择有较大收益，但所有候选仍很差 |
| MVHuman200 | 1.717 m / 44.5 度 | 1.390 m / 20.5 度 | 旋转候选互补性最明显 |
| THuman | 0.482 m / 6.7 度 | 0.450 m / 6.3 度 | 原方法已经相对稳定，Oracle 增益较小 |

MVHuman 的 Boundary Oracle 仍只有约 `0.026-0.030 m / 0.39-0.61 度`，所以 MVHuman 的大误差不是 GT convention 或 transform 方向错误，而是当前人体/pointmap 候选没有估计到正确 shot transform。

## 6. 候选胜出分布

Joint Oracle 最常选择：

```text
human_last_full_no_refine                 52
human_mean_full_no_refine                 36
human_last_pointmap_wait_b2_standard      27
human_last_pointmap_last_standard         22
human_mean_pointmap_history_strict        13
human_mean_pointmap_wait_b2_standard      11
human_mean_pointmap_history_loose         11
```

没有一个候选在所有场景中占绝对优势，单帧人体、历史人体和等待 B2 的 pointmap 候选确实互补。

## 7. 是否训练 Selector

当前结论是：**暂时不训练 Candidate Selector，先改进候选生成。**

理由：

1. Oracle Selection 明显优于原先的固定 Explicit，说明选择问题真实存在；
2. 但与最强单一固定候选相比，Oracle 上界只获得中等幅度 joint cost 改善；
3. relaxed success rate 没有提高，catastrophic failure 只从 65.6% 降到 65.0%；
4. Joint Oracle 仍有 `1.389 m / 16.63 度`，而 Boundary Oracle 接近零，候选集合与真实解之间仍有巨大差距；
5. 现在训练 Selector，即使完美拟合 Oracle，也无法解决大多数失败样本。

下一步应优先补充能产生真正新解的候选，而不是继续堆相似的 ICP 阈值：

```text
人体朝向的前后歧义处理
人体 root 深度与尺度可靠性诊断
地面/重力/相机高度约束
背景平面退化检测与非平面区域候选
多帧人体运动连续性候选
scene 与 human 分开估计后的一致性候选
```

当 Oracle Candidate Selection 能明显提高成功率并显著降低 catastrophic failure 后，再训练 Geometry-only 和 Token + Geometry Selector。

## 8. 实现与输出

独立实验脚本：

```text
scripts/v10_oracle_candidate_selection_probe.py
```

结果目录：

```text
output/v10_candidate_selection/oracle_gt_4source/
  oracle_candidate_selection_metrics.json
  oracle_candidate_selection_metrics.md
  oracle_candidate_selection_cases.csv
  selected_records.jsonl
  cases/*/case_metrics.json
  cases/*/human3r_local_reset/
```

当前缓存占用约 2.4 GB，可直接复用进行后续候选扩展，不需要重新运行 Human3R。
