# V14.2 Canonical Human Memory for Alignment and Continuity

## Executive Conclusion

V14.2 得到的是题目中的 **情况 B**：V14.1 canonical human memory 能稳定改善
人体连续性，但不能改善 V18 Boundary alignment。因此它应保留为 continuity
module，不应宣称为 camera-cut alignment module，也不应把历史 shape 强制作为 V18
的 metric calibration reference。

最关键的因果结论是：历史 Human3R 人体很稳定，但稳定在错误的绝对尺度上。历史
memory 能减少 frame-to-frame 的 beta/身体大小噪声，却无法恢复 MVHuman 缺失的
`0.5417/0.65` world-scale factor。GT body-scale scalar 能显著降低 projected depth，
GT beta proportions 单独不能，说明瓶颈是绝对尺度值而不是 shape 抖动。

最终保留：

```text
cut-time scene/camera hard reset
+ V16 torso-motion rotation, global 20 degree bound
+ existing explicit Boundary candidate
+ isolated canonical human memory for continuity only
+ Align-Then-Commit for world quantities
```

停止：

```text
canonical Human3R shape/scale -> V18 Boundary translation
```

除非后续先由 DA3 或其他 source-independent metric cue 校准绝对 body scale，否则
不再把长期 Human3R shape memory 与 Boundary translation 绑定。

## Protocol

Single-cut 主实验使用 V18 已缓存的 180 个真实 cross-camera cuts：

| Source | Cases |
|---|---:|
| AvatarReX | 48 |
| THuman | 48 |
| MVHuman100 | 48 |
| MVHuman200 | 36 |

所有 alignment 版本固定：

- frozen Human3R；
- V16 torso-motion rotation 和统一 `20 deg` residual bound；
- cut 后当前 predicted pose；
- frozen detector 当前 2D torso joints；
- 当前 crop 后 intrinsics；
- last pre-cut projected world root motion model；
- 一个统一、固定的 shot-level Boundary SE(3)。

唯一变量是投影求深度时使用的 SMPL-X beta 和身体物理尺度。历史 local pose 只进入
continuity probe，从未进入 camera-depth solve。Camera GT、GT shape 和 GT scale
只用于评测或明确标注的 oracle。

Human3R 没有独立的 predicted world-scale head。本实验将 body-scale scalar 明确定义
为 8 组固定 pelvis/hip/shoulder/leg 线段的平均米制长度，从而把：

- beta：身体比例；
- scalar：整个人体物理大小；

分开消融。这个 scalar 是从 Human3R SMPL-X geometry 推导的，并不是一条新的外部
尺度观测。

入口：

```bash
PYTHONPATH=src:. .venv/bin/python \
  versions/v12/experiments/v14_2_canonical_human_memory_probe.py \
  --device cuda:0

PYTHONPATH=src:. .venv/bin/python \
  versions/v12/experiments/v14_2_multicut_memory_replay.py \
  --device cuda:0
```

机器可读结果：

```text
output/v14_2_canonical_human_memory/single_cut/
output/v14_2_canonical_human_memory/multicut_replay/
```

## Stage 0: Pre/Post Scale Consistency

| Body convention | T mean | T P90 | View error | Post root-depth error |
|---|---:|---:|---:|---:|
| Current independent pre/post | 0.872 | 2.079 | 0.464 | 0.437 |
| Only post uses canonical | 0.871 | 2.120 | 0.470 | 0.436 |
| Canonical alpha=0.25 | 0.871 | 2.088 | 0.464 | 0.435 |
| Same canonical pre/post | 0.874 | 2.123 | 0.471 | 0.436 |
| Correct pre / wrong-video post | 0.886 | 2.143 | 0.481 | 0.454 |
| Same wrong-video body pre/post | 0.904 | 2.176 | 0.493 | 0.454 |
| GT beta + GT physical scale | 0.472 | 0.977 | 0.191 | 0.083 |

前后使用错误的不同身体会退化，说明投影尺度一致性确实重要；但前后统一使用正确
historical canonical body 并未优于 current independent body。也就是说，一致性是
必要的几何卫生条件，却不是当前缺失绝对尺度的来源。

所有 shape/reference 版本的 rotation 最大差异为 `0`，确认 memory 没有意外改变
V16 rotation。

## Single-Cut Alignment

| Candidate | T mean | Median | P90 | P95 | View | T-cat | Harmful vs current |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715 | 1.422 | 3.718 | 4.123 | 0.988 | 65.6% | 87.8% |
| V18 current post shape | **0.872** | 0.582 | 2.079 | 2.329 | 0.464 | 34.4% | 0.0% |
| V14.1 canonical alpha=0.25 | 0.871 | 0.580 | 2.088 | 2.328 | 0.464 | 34.4% | 1.7% |
| Canonical beta only | 0.873 | 0.580 | 2.087 | 2.337 | 0.464 | 35.0% | 0.0% |
| Canonical scale only | 0.872 | 0.544 | 2.110 | 2.327 | 0.469 | 33.3% | 12.8% |
| Canonical beta + scale | 0.874 | 0.541 | 2.123 | 2.327 | 0.471 | 33.3% | 15.0% |
| Last pre-cut reference | 0.870 | 0.527 | 2.089 | 2.286 | 0.469 | 33.9% | 13.3% |
| Historical median | 0.873 | 0.544 | 2.083 | 2.328 | 0.471 | 33.9% | 15.6% |
| Best quality reference | 0.875 | 0.545 | 2.084 | 2.342 | 0.472 | 33.9% | 17.2% |
| Top-3 consensus | 0.873 | 0.547 | 2.083 | 2.329 | 0.471 | 33.9% | 16.7% |
| Oracle best historical | 0.853 | 0.522 | 2.059 | 2.278 | 0.452 | 33.3% | 8.9% |
| Wrong-video memory | 0.904 | 0.599 | 2.176 | 2.432 | 0.493 | 33.3% | 36.7% |
| GT scale only | **0.462** | 0.321 | 0.981 | 1.160 | **0.183** | **9.4%** | 12.8% |
| GT beta + scale | 0.472 | 0.348 | **0.977** | **1.151** | 0.191 | **9.4%** | 18.9% |

Canonical alpha=0.25 相对 current 的 paired mean 仅改善 `0.0006 m`，Wilcoxon
`p=0.777`，没有统计或工程意义。完整 canonical beta+scale 反而退化 `0.0022 m`。

Oracle best historical 在知道 GT post depth 后才能选择，mean 仅改善 `0.019 m`。
这说明历史中存在很小的 reference variation，但上限远小于剩余 `0.87 m`，不足以
支持继续开发 fixed selector。

## Beta vs Absolute Scale

| Body information | Root depth | Boundary T |
|---|---:|---:|
| Current predicted body | 0.437 | 0.872 |
| Canonical beta only | 0.438 | 0.873 |
| Canonical scale only | 0.434 | 0.872 |
| Canonical beta + scale | 0.436 | 0.874 |
| GT beta only | 0.456 | 0.895 |
| GT physical scale only | **0.075** | **0.462** |
| GT beta + scale | 0.083 | 0.472 |

GT beta 单独无效，GT scalar 单独恢复绝大部分 depth。这直接回答了 V14.2 最重要
的问题：有用的不是更稳定的 beta proportions，而是历史 Human3R memory 中根本
不存在的绝对 world-scale scalar。

五帧历史本身已经非常稳定：predicted body-scale std mean 为 `0.00171`，shape
std mean 为 `0.0524`。稳定并不等于正确，尤其是：

```text
MVHuman100 target world scale = 0.5417
MVHuman200 target world scale = 0.6500
Human3R historical canonical body remains near scale 1
```

因此 median、EMA、Top-3 都只能对错误尺度做更稳定的平均，不能产生缺失的 metric
observation。

## Cross-Source Generalization

Canonical beta+scale 相对 current shape 的 mean 变化：

| Source | Current | Canonical | Delta | P90 delta |
|---|---:|---:|---:|---:|
| AvatarReX | 0.212 | 0.219 | +0.007 | -0.000 |
| THuman | 0.341 | 0.350 | +0.009 | +0.030 |
| MVHuman100 | 1.812 | 1.806 | -0.006 | +0.014 |
| MVHuman200 | 1.207 | 1.204 | -0.003 | +0.002 |

它只在两个 MVHuman source 上带来毫米级 mean 改善，同时破坏 AvatarReX/THuman，
四个 source 的 P90 也没有一致方向，不满足“三源同向且不破坏 THuman”的标准。

Oracle best historical 四源 mean 都略降，但改善仅 `0.003-0.042 m`，且真实系统不
知道该选哪帧。固定 quality、reprojection、shape stability、pixel size、view
similarity 和 Top-3 规则全部未复现 oracle 收益。

正面、背面和侧面组都没有一致收益；180/180 样本的 6 个 torso joints 均达到当前
阈值，因此本协议不能正式评价严重 torso truncation。

## Camera Metric Is Not Final Geometry

| Candidate | Camera T | Final raw-Human3R root residual |
|---|---:|---:|
| Fixed Explicit | 1.715 | **0.287** |
| V18 current body | 0.872 | 0.918 |
| Canonical beta + scale | 0.874 | 0.928 |
| GT physical scale only | **0.462** | 1.465 |
| Boundary Oracle | 0.000 | 1.499 |

GT scale 和 Boundary Oracle 能把 camera pose 拉向 GT，但 post-cut raw Human3R
local SMPL-X geometry 本身仍处在错误 depth/gauge。结果是相机指标越接近 GT，显示
的人体反而可能越不重合。这复现了 V18 三维审计，说明 V14.2 不能作为最终部署
Boundary candidate。

Random beta 的 camera mean 偶然达到 `0.850 m`，但 final root residual 恶化到
`0.959 m`；wrong-video memory 在部分 identity replay 中也会偶然更好。这是错误
尺度相互抵消，不是正确 human memory 对 alignment 的因果作用。

## Continuity Contribution

相同 180 cuts，仅对最终人体输出做保守 continuity memory：

| Method | Shape jump | Scale jump | Local-pose residual | GT beta error | GT scale error |
|---|---:|---:|---:|---:|---:|
| Hard Reset | 0.718 | 0.00751 | 5.37 deg | 1.634 | 0.05738 |
| Shape + scale memory | **0.558** | **0.00577** | 5.37 deg | **1.599** | **0.05674** |
| Shape + scale + local pose | **0.558** | **0.00577** | **4.58 deg** | **1.599** | **0.05674** |

Shape jump 下降 `22.3%`，scale jump 下降 `23.2%`，local-pose residual 下降
`14.8%`。四个 source 的 shape jump、scale jump 和 local-pose residual 全部同向
改善；总体 GT beta/scale accuracy 也没有因平滑而退化。

四象限实验完全解耦两个作用：

| Quadrant | Camera T | Shape jump | Pose residual | Final root residual |
|---|---:|---:|---:|---:|
| No memory | 0.872 | 0.718 | 5.37 deg | 0.918 |
| Continuity only | 0.872 | **0.558** | **4.58 deg** | 0.918 |
| Alignment only | 0.874 | 0.718 | 5.37 deg | 0.928 |
| Alignment + continuity | 0.874 | **0.558** | **4.58 deg** | 0.928 |

这证明 continuity 收益不依赖 Boundary 改善，alignment memory 也没有贡献。

## Multi-Cut and Commit

V14.1 真正 recurrent 8-cut rollout 的保守 `alpha=0.25` 结果：

| Metric | Hard Reset | Selective Align |
|---|---:|---:|
| 8-cut shape jump | 0.592 | **0.435** |
| 8-cut shape drift | 0.582 | **0.484** |
| 8-cut scale jump | 0.00553 | **0.00409** |
| Memory world-root error | n/a | 0.931 |

Immediate Commit 的 8-cut memory root error 为 `1.135 m`，Align-Then-Commit 为
`0.931 m`。Single-cut V14.1 也为 `0.997 -> 0.747 m`。World root 和 global
orientation 依赖 Boundary gauge，因此 Align-Then-Commit 仍是长期 world memory
的必要协议。

额外对 13 个重复 identity 进行了 1/2/4/8-cut causal reference replay。它使用
独立 V18 cuts，不能作为可拼接的全局 camera trajectory；只能比较每个 boundary
和 error sum。8-cut 结果：

| Strategy | T mean | Error sum | Endpoint shape drift | Root residual |
|---|---:|---:|---:|---:|
| No memory | 1.065 | 8.522 | 0.803 | 1.057 |
| V14.1 alpha=0.25 | 1.067 | 8.533 | **0.713** | 1.052 |
| Running median | 1.084 | 8.669 | 0.670 | 1.033 |
| First quality reference freeze | 1.095 | 8.759 | **0.604** | 1.029 |
| Top-3 consensus | 1.094 | 8.750 | 0.679 | 1.024 |
| Wrong-video memory | 1.037 | 8.298 | 0.794 | 1.093 |

长期 memory 策略继续降低 shape drift，但没有改善 alignment。Wrong-video 的
camera metric 偶然更低而 root residual 更高，再次否定 alignment 因果性。

V14.1 no-cut 实际运行检查：

```text
camera max difference   = 0
pointmap max difference = 0
SMPL-X shape difference = 0
```

V14.2 alignment probe 在 frozen Human3R 输出之后运行，不读取或写入 recurrent
scene/camera state，因此不会改变无 cut 路径。

## Final Answers

1. **Canonical shape/scale 能否改善 post-cut human depth？** 只能将 mean depth
   `0.4373 -> 0.4350 m`，约 `2 mm`，没有实质价值。
2. **能否改善 Boundary translation？** 不能。保守 alpha 变化低于 `1 mm`，完整
   canonical 略微退化，跨 source/P90 不一致。
3. **真正有用的是 beta 还是 scale？** 对 alignment 真正有用的是正确绝对
   body-scale scalar；beta 单独无效。当前 memory 只有稳定但不准确的内部尺度。
4. **Pre/post 同一 canonical body 是否更稳定？** 它避免明显 mismatch 伤害，但
   不优于各自 current body，不能恢复缺失的 world scale。
5. **Best historical 是否优于 last/median？** 否。所有固定选择器无效；GT oracle
   只给出约 `1.9 cm` 的小上限。
6. **Memory 对 alignment 是否有因果性？** 没有。Correct memory 未稳定优于
   current/wrong/random controls，错误 memory 还会偶然改善 camera-only metric。
7. **Memory 对 continuity 是否多 cut 更明显？** 是。单 cut 和 8-cut 均降低
   shape/scale drift，四源同向。
8. **Alignment 和 continuity 是否使用相同内容/强度？** 不应。当前 alignment
   不使用 canonical memory；continuity 使用 shape/scale `alpha=0.25` 和 local pose
   `alpha=0.15`。
9. **Memory 应改写人体还是只作 calibration reference？** 当前应保守稳定最终
   人体输出，不应用于 camera calibration。若未来有外部 metric scalar，可再把
   canonical proportions 作为内部投影模板。
10. **Align-Then-Commit 是否必要？** 是。世界位置和朝向必须先进入统一 old-world
    gauge，再写长期 memory。
11. **最终定义？** V14.2 是 **Shot-aware Human Continuity Memory**，不是 unified
    alignment-and-continuity module。

## Limitations

- 当前协议 `max_humans=1`，不能正式报告多人 Re-ID、IDF1 或 wrong-person association。
- 180 cuts 的 torso visibility 全部通过阈值，不能评价严重遮挡/截断。
- V18 cache 只有第一张 post-cut 帧；真实 recurrent offset 1/2/4/8 和 no-cut 结论
  来自 V14.1 rollout，V14.2 repeated-identity replay 不是可拼接全局轨迹。
- Camera GT 只用于评测；GT shape/scale 只用于 oracle，不能进入部署版本。
