# Frozen BRTC shared-group alpha selector: held-out validation

Frozen policy SHA256: `45fc7c9de7da0f5cfd8fda0937c4b468d3ff2e5bfad31a2c70f9864f5ad8da98`.

## 结论

该分支应归档为 **NO-GO**，不能替换 BRTC v1，也不能作为当前精对齐主线。

它确实证明了“只缩小 shared group translation”有时可以降低人体的绝对
root/joint/vertex 误差：严格分组 CV、three offset1 和 EgoHumans 的人体绝对误差都
有改善。但是这个收益不具有跨数据共性：offset1 的 pair distance 变差，dance 的
两项 pair 都变差，box 的 root/joint/vertex/pair-vector 变差；EgoHumans 上 W、WA、
pair 两项和 world-root Accel 也一起变差。因此它只是一个保守的误差校准器，不是
同时解决绝对位置、多人布局和时间一致性的精对齐方法。

## 方法与严格协议

输入仍然只有冻结 B0 后、BRTC v1 已使用的当前边界信息：前一 shot 最后一帧与当前
shot 第一帧的相机、匿名匹配人体、core-joint ray triangulation 证据，以及 BRTC v1
产生的 shared group shift、individual shift 和 observable layout lambda。没有读图像、
GT、未来帧、身份名或新预训练模型。

选择器从这些量提取 25 个 aggregate observable feature，用一个冻结的三分类线性模型
选择 `alpha∈{0.8,0.9,1.0}`。最终第 i 个人的位移为：

```text
final_i = alpha * group + lambda * (individual_i - group)
```

只允许缩放 shared group 项；individual residual 完全不变。特征超出开发分布、分类器
置信度不足、策略格式错误，或分类器选择 1.0 时，输出都是 exact BRTC v1。相机、未
匹配人体和 BRTC 拒绝的人体始终不变。

开发只读取 `three offset0` 的 41 cuts / 122 people。7 个 timestamp 逐组留一交叉验证，
同 timestamp 的不同 camera pair 不会跨 train/validation。扫描 96 组
`C × class_weight × confidence × OOD margin`，冻结前的入选条件要求：root、joint、
vertex、pair distance、pair vector、harm >1cm 和 harm >5cm 全部不差于 v1，并且至少
有一次非 1.0 动作。

oracle 标签分布为 `0.8/0.9/1.0 = 18/6/17`；24/41 个 case 的 oracle 可以改善 root，
oracle case-mean 收益为 10.485 mm。96 个候选中 7 个通过开发安全门。最终冻结：

```text
C=1.0, class_weight=balanced, confidence=0.6, OOD margin=0
SHA256=45fc7c9de7da0f5cfd8fda0937c4b468d3ff2e5bfad31a2c70f9864f5ad8da98
```

## 冻结前 grouped out-of-fold 结果

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm | Actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 0.225088 | 0.270404 | 0.248604 | 0.102222 | 0.260536 | 15.6% | 6.6% | 0 |
| selector | 0.222445 | 0.268510 | 0.246200 | 0.102178 | 0.259766 | 15.6% | 6.6% | 6/41 |

OOF 中 35/41 次 exact-v1 fallback：20 次 OOD，10 次低置信，5 次模型选择 1.0。
该结果满足冻结条件，但它只决定是否有资格进入盲测，不代表最终晋级。

## three_offset1

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm | Actions | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 0.231437 | 0.274493 | 0.252451 | 0.098351 | 0.258779 | 16.8% | 7.2% | 0 | 0 |
| selector | 0.227227 | 0.271856 | 0.249529 | 0.099248 | 0.258001 | 16.0% | 5.6% | 10 | 32 |

- α counts: `{'0.8': 5, '0.9': 5, '1.0': 32}`; fallback: `{'classifier_selected_exact_v1': 9, 'feature_out_of_development_support': 12, 'low_classifier_confidence': 11}`.
- Post-shot root Accel: `unavailable`.

## dance

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm | Actions | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 0.125131 | 0.177804 | 0.152914 | 0.044141 | 0.078318 | 14.8% | 3.3% | 0 | 0 |
| selector | 0.124816 | 0.177688 | 0.152680 | 0.044794 | 0.078656 | 14.8% | 3.3% | 1 | 60 |

- α counts: `{'0.8': 1, '0.9': 0, '1.0': 60}`; fallback: `{'classifier_selected_exact_v1': 30, 'feature_out_of_development_support': 23, 'low_classifier_confidence': 7}`.
- Post-shot root Accel: `26.579933379249432`.

## box

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm | Actions | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 0.372345 | 0.421610 | 0.434528 | 0.063069 | 0.427334 | 11.5% | 6.4% | 0 | 0 |
| selector | 0.372655 | 0.422069 | 0.434942 | 0.062939 | 0.427634 | 11.5% | 6.4% | 7 | 71 |

- α counts: `{'0.8': 7, '0.9': 0, '1.0': 71}`; fallback: `{'classifier_selected_exact_v1': 18, 'feature_out_of_development_support': 47, 'low_classifier_confidence': 6}`.
- Post-shot root Accel: `53.234274574536435`.

## EgoHumans 001_legoassemble (same-forward CPU cache)

| Method | W | WA | Root | Joint | Vertex | Pair distance | Pair vector | Root Accel | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b0_brtc_lc | 314.059 | 202.461 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 23.8% |
| b0_brtc_group_alpha_selector | 315.942 | 203.019 | 378.471 | 382.755 | 383.173 | 177.381 | 335.286 | 116.631 | 23.8% |

- α counts: `{'0.8': 2, '1.0': 4}`; fallback: `{'low_classifier_confidence': 1, 'feature_out_of_development_support': 3}`.

- Held-out winner: **False**.
- Decision: **NO_GO_ARCHIVE**.
- No held-out result was used to alter the frozen policy.

## 如何解释这些结果

- `three offset1` 说明模型确实学到一小部分可迁移的“BRTC group 过冲”模式，root 从
  231.437 降到 227.227 mm，且 >5cm harm 从 7.2% 降到 5.6%；但 pair distance 增加
  0.896 mm，因此不是全指标 winner。
- dance 只有 1/61 个 case 动作，绝对人体误差略降，但 pair distance/vector 分别增加
  0.653/0.338 mm。严格 OOD 与置信门保护了大多数 case，却也说明 offset0 的可观察
  分布不能覆盖该场景。
- box 有 7/78 个 case 动作，root 增加 0.311 mm，joint/vertex 也变差。这是明确的跨
  场景失败，而不是覆盖率太低造成的“没有效果”。
- EgoHumans 的两次 0.8 动作使 root/joint/vertex 分别改善 2.183/1.974/2.065 mm，
  但 W/WA 增加 1.883/0.558 mm，pair distance/vector 增加 0.356/1.416 mm，world-root
  Accel 增加 0.617 mm/frame²。它把单帧绝对人体位置拉近了，却破坏了轨迹与多人相对
  布局，这正是不能只优化 root mean 的证据。
- dance/box 表中的 post-shot Accel 与 v1 完全相同，因为部署时同一边界 translation
  会常量传播到后续帧，三帧内部二阶差分会抵消。它不覆盖切换瞬间；包含切换的
  EgoHumans full-stream Accel 才暴露出 116.014→116.631 的退化。

## 实现与完整性检查

- runtime：`versions/v14/b0_person_triangulation_group_alpha_selector.py`
- train/freeze/heldout probe：`versions/v14/probe_brtc_group_alpha_selector.py`
- 开发输出：`output/v14/fine_alignment_research/brtc_group_alpha_selector/DEV_CV.json`
- 冻结策略：`output/v14/fine_alignment_research/brtc_group_alpha_selector/FROZEN_POLICY_BEFORE_HELDOUT.json`
- 盲测输出：`output/v14/fine_alignment_research/brtc_group_alpha_selector/HELDOUT_RESULTS.json`

冻结策略在盲测前后 SHA256 完全相同。EgoHumans camera max change 为 0，unmatched max
change 为 0；runtime 公式误差为 `8.33e-17`，individual residual reconstruction 误差
与 rejected-person change 都为 0。四个单元测试覆盖 alpha=1 exact v1、alpha=0.8 只
改 group、OOD exact v1 和低置信 exact v1，全部通过。

后续不应围绕这个 selector 继续调阈值。更值得投入的是直接建立人体朝向/刚体结构的
显式对应（例如当前并行的 Kabsch 候选），因为本实验已经显示 scalar group damping
无法统一兼顾绝对人体位置、多人布局与跨 shot 轨迹。
