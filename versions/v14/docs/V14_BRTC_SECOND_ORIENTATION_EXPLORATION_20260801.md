# V14 BRTC 第二 orientation 候选探索（2026-08-01）

## 结论

本轮没有找到一个能够严格替代或叠加在现有 `individual Kabsch` 上的第二 orientation
方案，最终结论是 **NO-GO**。当前应继续只保留已冻结的 person-local TORSO4 Kabsch：

```text
frozen B0 camera
-> frozen BRTC translation/root
-> current per-person TORSO4 bounded Kabsch
-> post-shot causal propagation
```

没有冻结新模型，也没有打开 `three offset1`、`dance`、`box` 或 EgoHumans 做调参。
原因不是候选完全没有平均收益，而是收益无法仅用 runtime 可观测量在每个 timestamp 上
稳定识别；继续看 aggregate 会掩盖局部回退。

## 1. 不可改变的比较基线

本轮不是相对 BRTC v1 证明“有改善”，而是必须严格超过已经 qualified 的 individual
Kabsch：

```text
joint ids       = (1, 2, 16, 17)
max angle       = 25 deg
rotation fraction = 0.5
observable improvement gate = 0
rotation pivot  = native Human3R root
```

每个新候选必须保持：

- frozen B0 camera exact；
- frozen BRTC native root exact；
- BRTC rejected/unmatched exact；
- pair-root distance/vector exact；
- 不读图像、GT、identity、future post frame 或新预训练模型；
- 相对 current Kabsch 的 raw joint/vertex、pelvis-centered joint/vertex 和 person tail
  harm 都不差。

## 2. 扩大 orientation 对应点：TORSO8

### 2.1 方法

把 current Kabsch 的四点对应：

```text
TORSO4 = (1, 2, 16, 17)
```

扩大为躯干链上的八点：

```text
TORSO8 = (1, 2, 3, 6, 9, 12, 16, 17)
```

输入仍是同一个 matched person 的 last-pre/current-post predicted joints。两组点分别减去
各自 native root，Kabsch 求 `post -> pre` 的 SO(3)，再使用 current policy 做半步、最大
25 度的有界旋转。输出是绕已经 BRTC-corrected native root 旋转后的 joints/vertices；root
和 camera 不变。

### 2.2 结果

在 `three offset0` 的 108 个 BRTC-accepted person 上，TORSO8 相对 current TORSO4 的
aggregate raw joint/vertex 改善约 `-0.051/-0.031 mm`，但 timestamp 不稳定：

| Timestamp | dJoint mm | dVertex mm |
|---:|---:|---:|
| 500 | +0.274 | +0.297 |
| 700 | -0.302 | +0.461 |
| 900 | -0.399 | -0.629 |
| 1000 | -0.029 | -0.018 |
| 1100 | -0.056 | +0.056 |
| 1300 | -0.033 | -0.349 |
| 1500 | -0.123 | +0.334 |

所以“直接把 TORSO4 换成 TORSO8”是 NO-GO。

## 3. 不训练的 observable residual selector

### 3.1 方法

同时计算 TORSO4/TORSO8 rotation，然后只用 22 个 predicted body joints 的 pre/post
连续性 residual 选择较小者。这个规则完全可部署，不读 GT。

### 3.2 结果

它选择 TORSO8 `62/108` 次，aggregate raw joint/vertex 相对 current Kabsch 改善
`-0.055/-0.070 mm`，但：

- timestamp 500 回退 `+0.281/+0.327 mm`；
- timestamp 700 的 vertex 回退 `+0.166 mm`。

说明“让 predicted skeleton 自己更连续”并不等价于“更接近 GT body”。该确定性 selector
也是 NO-GO。

## 4. mapped-pelvis rotation pivot

### 4.1 动机和方法

EgoHumans 严格门中的 fixed-world root 不是 runtime 保存的 native Human3R root，而是先把
SMPL-X vertices 映射到 SMPL，再用 neutral-SMPL joint regressor 得到的 pelvis。current
Kabsch 绕 native root 旋转，所以 native root bit-exact 时，mapped pelvis 仍可能移动。这正是
Ego 上唯一的 strict-zero 回退 `+0.034 mm` 的来源。

测试过的 split-pivot 变体保持 native root 字段不变，并让 mesh 绕 predicted mapped pelvis
旋转，希望 mapped-pelvis proxy 不因 orientation 更新而移动。这个 pivot 完全来自预测 mesh，
不是 GT。

### 4.2 结果

在 `three offset0` 上，joint aggregate 与 current Kabsch 相同，但 vertex aggregate回退
约 `+0.048 mm`，且多数 timestamp 不是稳定改善。它只能针对评测 proxy 保持一个点，代价
是使整张 mesh 的固定世界坐标误差变差，因此判定 NO-GO，不能为修正 Ego 的 `0.034 mm`
而使用。

## 5. 四候选 oracle 只说明存在上限，不是 runtime 方法

诊断候选为：

```text
b  = TORSO4 + native-root pivot（current baseline）
t  = TORSO8 + native-root pivot
bp = TORSO4 + split pelvis pivot
tp = TORSO8 + split pelvis pivot
```

若用 GT `joint+vertex` 对每个人事后选最优项，108 个 accepted person 的选择计数为：

```text
t=35, tp=29, bp=23, b=21
```

其中 `87/108` person 能相对 `b` 改善，oracle 平均潜力约 `0.839 mm/person`。这只证明
候选集合里存在互补性；GT oracle 不可部署，也没有被用作 runtime gate。

## 6. Observable grouped-CV selector

### 6.1 输入、模型与输出

最终测试了一个保守的浅层回归树。输入全部是 cut 时已经可见的预测量：

- TORSO4/TORSO8/all-22 joints 在两种 rotation 前后的 residual 和相对 residual；
- 两个 applied angle，以及两个 rotation 的 SO(3) geodesic difference；
- predicted joint0 与 native root 的 pre/post offset；
- stable-bone pre/post log-ratio median/MAD；
- frozen BRTC action、ray gap、MAD、ray sine；
- frozen B0 pre/post camera 的相对 rotation/translation。

训练 target 仅在 development 构造：TORSO8 相对 TORSO4 的 `joint delta + vertex delta`。
runtime 输出不是新几何，而是二选一 decision：

```text
低置信或 OOD -> exact current TORSO4 Kabsch
预测收益超过阈值 -> TORSO8 Kabsch
```

模型只扫描 `max_depth={2,3,4}`、`min_samples_leaf={3,5,8}` 和预定的 0--4 mm
conservative threshold；OOD 使用训练折的 diagonal `max |z| <= 5`。没有把 timestamp、case、
camera id、person identity 或 GT 放入 feature。

### 6.2 严格协议

开发数据只使用 `three offset0`，按 timestamp 做七折 leave-one-group-out：

```text
500 / 700 / 900 / 1000 / 1100 / 1300 / 1500
```

每个留出折必须同时满足：

- raw joint、raw vertex 不差；
- pelvis-centered joint、pelvis-centered vertex 不差；
- selected person 的所有四个指标 `>1 cm`、`>5 cm` harm count 均为 0；
- 至少一次非 baseline action；
- camera/root/rejected/pair-root invariants exact。

### 6.3 最终结果：0 个 eligible config

共评估 369 个小网格组合，严格 eligible 数为 `0`。最接近的配置是：

```text
max_depth=3
min_samples_leaf=8
predicted_gain_threshold=1.8 mm
OOD max_abs_z=5
```

它在 LOGO-CV 中选择 TORSO8 `9/108` 次，aggregate 四项全改善且无 >1 cm harm：

| Metric | Aggregate delta vs current Kabsch (mm) |
|---|---:|
| raw joint | -0.016748 |
| raw vertex | -0.043760 |
| pelvis-centered joint | -0.029885 |
| pelvis-centered vertex | -0.038154 |

500/700 均因保守回退而 exact current Kabsch；但 timestamp 1000 的唯一 selected person
导致：

| Timestamp 1000 | Delta (mm) |
|---|---:|
| raw joint | -0.029053 |
| raw vertex | -0.148032 |
| pelvis-centered joint | -0.026945 |
| pelvis-centered vertex | **+0.052441** |

因此它仍违反“每个 timestamp、每个 body metric 都不差”的门。不能因为 aggregate 改善而
放行。

## 7. 最终决策

本轮状态：`NO_GO_SECOND_ORIENTATION_SELECTOR`。

- 不生成 frozen selector policy；
- 不运行 offset1/dance/box/Ego held-out；
- 不修改现有 individual Kabsch runtime；
- mapped-pelvis pivot 同样归档为 NO-GO；
- current TORSO4 individual Kabsch 仍是唯一 qualified orientation 模块。

结果文件：

- `output/v14/fine_alignment_research/brtc_person_orientation_selector/GROUPED_CV_DEV.json`
- `output/v14/fine_alignment_research/brtc_person_orientation_selector/GROUPED_CV_DEV.md`
- `versions/v14/probe_brtc_person_orientation_observable_selector.py`
- `tests/test_v14_brtc_person_orientation_observable_selector.py`

这次失败给出的有效结论是：TORSO4/TORSO8/split-pivot 的优劣确实因 person 而异，但当前
仅依赖单个 cut 的 predicted geometry residual，无法在严格跨 timestamp 条件下可靠识别。
下一条主线不应继续扩大同类关节点集合或扫 selector 网格。
