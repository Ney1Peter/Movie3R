# V10 Boundary Gauge Partial-Oracle 边界坐标分解实验

日期：2026-07-17

## 1. 实验目的

前一轮已经证明：cut 后必须 reset Human3R；给定正确 Boundary SE(3) 后，后续 shot 几乎可以完美恢复；但现有 12 个显式候选即使做 Oracle Selection，灾难性失败率仍约 65%。本实验进一步将 Boundary SE(3) 拆成旋转、平移、重力、人体 root 和人体 torso heading，判断下一步真正应该开发哪个模块。

实验不训练网络、不运行 Selector、不重新运行 Human3R，直接复用上一轮 180 个 AABB case 的 local-reset 输出与候选缓存。

## 2. 数据与约束

使用完全相同的 180 个 case：AvatarReX 48 个、THuman 48 个、MVHuman100 48 个、MVHuman200 36 个。

所有方法都使用 GT cut_idx，cut 后使用 fresh recurrent state，每个 shot 只确定一个固定 SE(3)，不做 BA 或全局轨迹优化。GT 仅用于 Partial-Oracle 诊断。

数据没有统一保存 ground-normal GT，因此 `GT Gravity` 使用 GT SMPL-X torso-up 作为重力方向代理。AvatarReX 和 THuman 的 body25 world 字段为零填充，本实验统一从其 SMPL-X GT 参数生成 camera-frame joints，再通过各自 GT camera 变到 world；MVHuman 也走同一评测坐标流程。

## 3. 对比方法

实验比较：

```text
Current Best Explicit
Current Candidate Oracle
Candidate Rotation Oracle + 固定旋转重求平移
Candidate Translation Oracle
Factorized Candidate Oracle
GT Rotation + Predicted Translation
Predicted Rotation + GT Translation
GT Gravity Proxy
GT Human Root
GT Human Torso Heading
GT Human + GT Gravity
Full Boundary Oracle
```

固定旋转重求平移时，使用 Human3R 预测 pelvis 初始化，再执行固定旋转的 translation-only pointmap refinement。Factorized Candidate Oracle 会对现有 12 个候选的每个旋转分别重新求平移，再使用 GT 选择最终误差最低的组合，不直接拼接两个候选的 R 和 t。

## 4. 总体结果

| 方法 | T mean | T P90 | R mean | R P90 | Relaxed success | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| Current Best Explicit | 1.5950 | 3.4682 | 12.93 | 25.41 | 1.1% | 65.6% |
| Current Candidate Oracle | 1.3891 | 2.9465 | 16.63 | 34.23 | 5.6% | 65.0% |
| Candidate Rotation Oracle + Resolved T | 1.5521 | 3.3944 | 10.61 | 23.64 | 6.1% | 65.0% |
| Candidate Translation Oracle | 1.1297 | 2.0378 | 45.45 | 152.90 | 6.1% | 65.0% |
| Factorized Candidate Oracle | 1.4083 | 3.1759 | 14.93 | 33.10 | 6.1% | 64.4% |
| GT Rotation + Predicted Translation | 1.4964 | 3.2725 | 0.25 | 0.55 | 11.7% | 66.1% |
| Predicted Rotation + GT Translation | 0.0150 | 0.0371 | 12.93 | 25.41 | 44.4% | 3.9% |
| GT Gravity Proxy | 1.5250 | 3.2818 | 10.33 | 24.73 | 6.1% | 65.0% |
| GT Human Root | 1.1027 | 2.0964 | 12.93 | 25.41 | 1.1% | 41.1% |
| GT Human Torso Heading | 1.5437 | 3.3385 | 5.96 | 9.94 | 3.3% | 66.7% |
| GT Human + GT Gravity | 0.9476 | 1.9766 | 0.25 | 0.54 | 19.4% | 37.8% |
| Full Boundary Oracle | 0.0149 | 0.0371 | 0.25 | 0.55 | 100% | 0% |

## 5. 旋转还是平移

将旋转直接替换为 GT 后，平移只从 `1.5950 m` 降到 `1.4964 m`，改善 `6.2%`。因此当前大平移误差不是由错误旋转间接造成的。即使旋转完全正确，现有 Human3R pelvis 初始化和 pointmap translation solver 仍无法找到正确世界平移。

反过来，使用 GT translation、保留当前预测旋转时，camera translation 已达到 `0.0150 m`，说明主要缺口确实在 translation。

Current transform 的平均绝对平移分量误差为：

```text
x：0.665 m
y：0.348 m
z：1.266 m
```

GT Human Root 将 z 降到 `0.622 m`，但 x 仍为 `0.653 m`。GT Human + Gravity 后仍有 `x=0.616 m、y=0.294 m、z=0.465 m`。所以缺失的不只是单一深度，而是完整的场景级世界平移。

## 6. 重力与人体朝向

GT Gravity Proxy 只把旋转从 `12.93 deg` 降到 `10.33 deg`，仅解释约 `20.6%` 的 GT Rotation 收益，因此 roll/pitch 或地面倾斜不是当前主要旋转瓶颈。

GT Human Torso Heading 将旋转降到 `5.96 deg`，明显优于 Gravity。这说明当前旋转更缺少人体或场景 heading，而不是单纯缺少地面法向。Torso heading 可以作为旋转模块的重要辅助，但它几乎不能解决平移，translation 仍为 `1.544 m`。

## 7. GT Human Root 的关键反例

GT Human Root 把预测 pelvis 精确放到 GT pelvis：

```text
aligned predicted pelvis -> GT pelvis error：约 0
camera translation error：1.103 m
```

GT Human + Gravity 同时把人体位置和旋转都做到 oracle：

```text
pelvis-to-GT：约 0
rotation：0.25 deg
camera translation：0.948 m
```

Full Boundary Oracle 则相反：

```text
camera translation：0.015 m
predicted pelvis -> GT pelvis：0.949 m
```

这说明 Human3R reset 后预测的人体局部位置与正确 camera/scene gauge 之间本身存在约 `0.95 m` 的跨视角不一致。把人体强行放到 GT 世界位置，能够让人体贴对，却会把相机和场景世界坐标带错；使用正确 camera gauge 后，Human3R 的局部人体预测仍可能偏离 GT。

因此人体只能作为软约束或 motion prior，不能作为跨 shot 绝对平移的唯一硬锚点。

## 8. 是否应该拆分 R/T

Factorized Candidate Oracle 相对 Current Candidate Oracle 的 joint cost 只改善约 `0.6%`，灾难性失败率只从 `65.0%` 变为 `64.4%`。当前没有证据支持仅靠“先从候选选旋转，再用现有 pointmap 重求平移”就能解决问题。

R/T 分解在结构上仍合理，但必须先有更可靠的 translation source；否则只改变求解顺序没有意义。Candidate Translation Oracle 能达到 `1.130 m`，但旋转高达 `45.45 deg`，说明候选中偶尔存在较好的平移数值，却与错误旋转和错误物理解绑定。

## 9. 分数据源结果

| 数据源 | Current | GT Human + Gravity | Full Oracle |
|---|---|---|---|
| AvatarReX | 1.187 m / 8.35 deg | 0.684 m / 0.04 deg | 0.004 m / 0.04 deg |
| MVHuman100 | 3.076 m / 17.01 deg | 1.914 m / 0.39 deg | 0.026 m / 0.39 deg |
| MVHuman200 | 1.540 m / 18.19 deg | 0.999 m / 0.61 deg | 0.030 m / 0.61 deg |
| THuman | 0.563 m / 9.50 deg | 0.206 m / 0.04 deg | 0.003 m / 0.04 deg |

MVHuman 的差距最大，说明人体局部深度、点云尺度或跨视角局部 gauge 的数据域差异尤其严重。THuman 相对容易，但即使 GT Human + Gravity 仍不能完全达到 Boundary Oracle。

## 10. 最终结论

下一步优先级为：

```text
1. 场景重定位 / 世界坐标记忆
2. Human Motion / Root 作为平移软约束
3. Human Torso Heading 作为旋转约束
4. Gravity / Ground 作为辅助约束
```

真正缺失的是一个严格流式的 scene relocalization 或 compact world-coordinate memory：利用 cut 前 scene/state 摘要和 cut 后 fresh scene geometry，提供 shot-level translation/gauge；人体 motion/root 负责辅助和消除动态歧义，torso heading 负责旋转方向，gravity 负责稳定倾斜。

在候选上界提高前，不应继续盲目增加相似显式候选，也不应立刻训练 Selector。

## 11. 实现与输出

实验脚本：

```text
scripts/v10_boundary_gauge_partial_oracle_probe.py
```

结果目录：

```text
output/v10_candidate_selection/boundary_gauge_partial_oracle/
  boundary_gauge_partial_oracle_metrics.json
  boundary_gauge_partial_oracle_metrics.md
  boundary_gauge_partial_oracle_summary.csv
  case_progress.jsonl
  gt_cache/*.npz
```

GT cache 和结果总占用约 9.4 MB，Human3R 大缓存继续复用前一轮输出。
