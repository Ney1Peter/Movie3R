# V14.5 Final Geometry, Leakage, and Streaming Audit

> 最终几何正确性、信息泄漏与真实流式审计
>
> 审计日期：2026-07-22
>
> 冻结起点 commit：`3822715d8f3d2fbcd9e0867cdf787bb99f05abf4`
>
> 审计代码 commit：`a14be4b`
>
> 审计原则：不改算法、不调阈值，只检查已有结论是否真实、可复现、无泄漏且满足流式约束。

## 1. 最终结论

### 1.1 一句话结论

V11.4 的单 Boundary camera/human 改善能够在原始 RGB 重跑和全新 capture-disjoint holdout 上复现，但它存在显著 scene trade-off、absolute human-root 排名受到 anchor 定义影响，而且真实 8-cut recurrent rollout 的累计误差仍然很大，因此**不能按 V14.5 的严格标准冻结为完整的 camera-human-scene 长期流式主方法**。

### 1.2 建议冻结的准确定位

可以保留：

```text
pre-decode hard reset
+ Fixed Explicit
+ V16 torso-motion rotation (20 deg bound)
+ frozen Conditional VGGT rotation tail rescue
+ V11.4 uniform shot similarity
+ one fixed shot-level Boundary
```

但论文中应定位为：

> **单 Boundary / 短时域、camera-human 优先的流式 re-anchoring 方法。**

当前不能声称：

- camera、human、scene 三者都同时改善；
- absolute human-root 优势完全来自 V11.4 scale；
- 已经证明 8-cut 长期 world mapping 稳定；
- 完整 raw-RGB candidate graph 已经通过所有 GT/metadata 扰动的动态 taint test；
- 达到实时视频帧率。

V14.3/V14.4 适合保留为必要性分析：

- coupled root 能保证 camera-human equation closure；
- projection consistency 不等价于完整 3D consistency；
- naive sequential composition 会重复修正；
- human-projection cue 不能统一当前 human/scene gauge。

V14.2 continuity 仍只应作为 alignment 后的可选轻量稳定器，不能参与 scale、rotation 或 Boundary 求解。

## 2. 审计判定总表

| 审计项 | 判定 | 核心证据 |
|---|---:|---|
| 原始 RGB 重跑 | 部分通过 | 保持原 180-case manifest 顺序时，8 个代表样本的 Human3R、Fixed transform 和指标差异全为 0；单独抽出 8-case 重跑会触发 manifest-order RNG 差异 |
| GT/source/metadata 泄漏 | 部分通过 | 180 个缓存 candidate 的 GT camera、GT human、source、camera ID、路径和标签扰动均不改变 candidate signature；尚未对所有扰动做完整 raw-RGB 动态重跑 |
| 合成 Sim(3) | 通过 | scene/root/body/camera/projection 均恢复到 `1e-14` 至 `1e-16` 量级 |
| 独立 evaluator | 通过 | 180 cuts 上 camera/root/joints/vertices/scene 最大差异均小于 `1e-5 m` |
| Common Anchor | 部分通过 | V11.4 camera 优势保留；旧 absolute root 排名在 common anchor 后消失 |
| Gauge 与单位 | 通过 | 完整解释 Fixed `1.715 m -> 0.712 m`；GT scale 只进入 evaluation conversion |
| Scale Pareto | 部分通过 | V11.4 scale 更接近 camera-optimal，而非 human/scene-optimal；scene 最优仅 14.6% 接近 predicted scale |
| Scene 保持 | 失败 | 147-case 上 V11.4 scene `0.483 -> 0.532 m`，81.6% 样本退化，`p=1.36e-11` |
| Conditional VGGT 泛化 | 通过但有尾部风险 | 419 个 post-freeze cases 上 rotation `14.91 -> 13.16 deg`；60-case holdout 上 `17.62 -> 14.08 deg`，但仍有 1 个大于 5 deg 的退化 |
| 真实 recurrent 8-cut | 失败 | V11.4 camera drift `0.946 m`、rotation drift `59.03 deg`，未满足长期稳定标准 |
| Untouched holdout | camera/human 通过，scene 失败 | V11.4 camera `0.663 -> 0.450 m`，human root `0.234 -> 0.195 m`；scene `0.475 -> 0.546 m` |
| No-cut exact no-op | 通过，沿用冻结 V14.4 检查 | camera、pointmap、SMPL-X max diff 均为 0 |
| 速度与确定性 | 确定性通过，实时性不足 | normal path `3.57 FPS`；额外 cut cue 平均 `2.49 s`；三次重复 scale/Boundary/trigger 完全一致 |

V14.5 原定冻结条件要求全部同时成立。Scene、8-cut 和无条件 root 因果解释没有通过，因此总判定为：

> **不按“完整最终主线”无条件冻结。保留为有明确收益和明确边界的 effect-first 方法。**

## 3. 冻结配置

### 3.1 模型与权重

| 模型 | 何时使用 | 作用 | SHA256 |
|---|---|---|---|
| Human3R | 每个正常帧；cut 后第一帧解码前 reset | shot-local camera、pointmap、SMPL-X 重建 | `1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377` |
| Keypoint R-CNN | 仅 Boundary scale/depth cue | 提供人体 2D keypoints，不回归 SE(3) | `fc266e953d2b302cdcbb9ae66f71f6b0d4649928bf02dc573961e361e4918926` |
| DA3Metric-Large | 仅 cut 时，对 pre/post Boundary 图像运行 | 提供人体/背景 metric-depth 和 shot scale cue | `bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776` |
| VGGT-1B | 仅 Conditional VGGT trigger 命中的 rotation tail | 困难大旋转 rescue，不估计最终 scale | `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0` |

Human3R checkpoint 大小为 4,670,554,642 bytes，DA3 为 1,336,734,448 bytes，VGGT 为 5,026,874,952 bytes。

### 3.2 主要冻结参数

| 参数 | 冻结值 |
|---|---:|
| V16 rotation bound | `20 deg` |
| V11.4 scale clip | `[0.35, 3.0]` |
| scene confidence threshold | `1.5` |
| scene mask dilation | `11` |
| scene sampled points | `1200` |
| camera success | translation `<0.5 m` 且 rotation `<20 deg` |
| camera catastrophic | translation `>2.0 m` 或 rotation `>45 deg` |
| harmful translation | 增加 `>0.05 m` |
| continuity shape/scale alpha | `0.25` |
| continuity local-pose alpha | `0.15` |
| random seed | `20260721` |

Conditional VGGT 的 large residual、spread、texture、consensus 和 cap 阈值全部从冻结实现读取，审计期间未调整。

### 3.3 数据规模

| 审计 | 数据量 |
|---|---:|
| 统一旧测试协议 | 180 real cross-camera cuts |
| common scene-valid subset | 147 cuts |
| Conditional VGGT 全历史诊断 | 1079 cuts |
| Conditional VGGT post-freeze subset | 419 cuts |
| raw-RGB/cache integrity | 8 个代表 cuts |
| scale sensitivity | 48 cuts，每个 source 12 个 |
| true recurrent rollout | 4 个真实 A/B/A/B 序列，每源 1 个，最多 8 cuts |
| untouched capture-disjoint holdout | 60 cuts |
| runtime | 8 cases x 3 repeats |

Untouched holdout 包含 AvatarReX 16、THuman 16、MVHuman100 16、MVHuman200 12。它与历史 829 个 capture 的 case overlap 和 capture overlap 都为 0，冻结 selection SHA256 为：

```text
c4b45bf0a8e1323b1153dc2ac79447724d1857eb82871a85ebf92f55127fc68f
```

## 4. 原始 RGB 与缓存完整性

### 4.1 保持原 manifest 上下文时的结果

在 GPU 上从原始 RGB 重跑完整原始顺序的 180-case V10/Human3R 路径，再抽取事先选择的 8 个代表样本与旧 cache 比较：

| 比较项 | 最大差异 |
|---|---:|
| Human3R 数值数组 | `0` |
| Fixed transform | `0` |
| Fixed translation metric | `0 m` |
| Fixed rotation metric | `0 deg` |

这说明：

- 在相同 commit、权重、输入顺序和 seed 下，旧 Human3R cache 可从 raw RGB 精确重现；
- 没有发现旧 checkpoint、旧 crop/intrinsics 或过期 local reset cache 混入这 8 个样本；
- 完整上下文重跑与旧缓存完全一致，而不只是“误差很小”。

### 4.2 发现的 manifest-order RNG 风险

若只把同样 8 个样本抽出，组成新的 8-case manifest 独立运行：

| 比较项 | 最大差异 |
|---|---:|
| Human3R 数值数组 | `0` |
| Fixed transform | `0.755731` |
| Fixed translation metric | `0.301407 m` |
| Fixed rotation metric | `25.307866 deg` |

Human3R 本身仍完全一致，差异发生在 Fixed candidate 的随机 point sampling。旧 V10 使用了类似：

```text
candidate_seed = seed + case_index * 1000
```

因此同一个 case 在 manifest 中的 index 改变时，candidate 的采样 seed 也改变。

这不是 GT leakage，也不是 stale cache，但它是一个真实的 determinism/cache-key 风险：

- 当前结果对“原 manifest 顺序 + seed”可复现；
- 当前 candidate 不对 case identity 保持 order invariance；
- 部署或重组数据列表时，不能仅凭 case 内容假设输出完全相同；
- 若后续修复，应使用由稳定 case ID/hash 派生的 seed，并从 raw RGB 重跑全部结果。V14.5 按冻结原则没有修改这一点。

### 4.3 本项审计边界

旧 180-case 上做了 raw Human3R + Fixed 的精确 cache 对比。全新 60-case holdout 则从 raw RGB 重新生成了 V10、V15、V16、V18 stream、2D keypoints、DA3 scale 和最终 V14.4 evaluator，60/60 无 inference failure。

但是，没有对旧 180-case 的全部 DA3/VGGT/最终 Boundary cache 做逐数组 raw-RGB 一一重放比较。因此“完整主结果可从 raw RGB 完整复现”的严格回答是：

> Human3R/Fixed 精确通过，fresh holdout 的完整候选链通过；旧 180-case 的每一个后续 cue 尚未全部做 cache-by-cache raw 重放，故判为部分通过。

## 5. GT、Source 与 Metadata 泄漏

### 5.1 已完成的扰动

在全部 180 个 serialized candidates 上分别执行：

- GT camera 随机打乱；
- GT camera 替换为单位矩阵；
- GT human 随机打乱；
- source ID 随机修改；
- camera ID / camera-pair ID 随机修改；
- 文件路径和 sequence 名重命名；
- 删除 evaluation labels，只保留 cut trigger。

所有扰动下 candidate signature 均保持不变，Conditional VGGT branch 180/180 可重现。

### 5.2 运行时依赖表

| 变量 | 来源 | 推理可用 | 含 GT | 影响 candidate | 仅评测 |
|---|---|---:|---:|---:|---:|
| cut trigger | 实验中 GT cut index；部署时 automatic detector | 是 | 只含触发时刻 | 是 | 否 |
| Fixed Explicit | 历史/当前 Human3R human 与 background pointmap | 是 | 否 | 是 | 否 |
| V16 torso rotation | 历史/当前 predicted SMPL-X torso | 是 | 否 | 是 | 否 |
| Conditional VGGT trigger | torso/VGGT residual、方向、spread、RGB texture | 是 | 否 | 是 | 否 |
| V11.4 scale | frozen DA3/root 与 background metric calibration | 是 | 否 | 是 | 否 |
| GT camera/human/scene | dataset annotation | 否 | 是 | 否 | 是 |
| source/camera/path ID | loader 和 cache lookup | 是 | 否 | 不进入几何公式 | 否 |

### 5.3 准确结论

没有发现 GT camera、GT human、source ID、camera pair 或文件名直接进入 deployable candidate 数值。

但当前泄漏扰动主要在 serialized/cached candidate 层执行，而不是对每一种 perturbation 都从 raw RGB 完整重跑所有模型。因此最严谨的表述是：

> **未发现泄漏证据；缓存候选层泄漏测试通过。尚不能把它扩大为完整 raw-RGB 动态 taint proof。**

另外，holdout 的 angle bucket 和 capture metadata 只用于冻结测试集抽样，不进入 candidate generation。

## 6. 独立几何实现与合成单元测试

### 6.1 合成 Sim(3)

独立构造 camera trajectory、scene points、root/body points，并施加已知 `s=1.37`、`R`、`t`：

| 检查 | 误差 |
|---|---:|
| scale recovery | `0` |
| scene recovery max | `1.78e-15 m` |
| root equation | `0 m` |
| body transform | `0 m` |
| projection invariance | `5.68e-14 px` |
| camera-origin scaling | `2.62e-16 m` |
| c2w/w2c round trip | `4.44e-16` |

这证明独立实现中的：

- c2w / w2c convention；
- camera origin scaling；
- root-centered body scaling；
- scene/root/body 共用 Sim(3)；
- uniform scale 下的 2D projection invariance；

都符合理论。

### 6.2 180-cut 双 evaluator

独立 evaluator 不调用主方法 transform helper。对 V11.4、Unified Human、Unified DA3 的全部 180 cuts 重新计算：

| 指标 | 主/独立 evaluator 最大绝对差异 | 容差 |
|---|---:|---:|
| camera translation | `5.70e-7 m` | `1e-5 m` |
| camera rotation | `0.001207 deg` | `0.002 deg` |
| human root | `7.69e-7 m` | `1e-5 m` |
| joints | `6.55e-7 m` | `1e-5 m` |
| vertices | `6.82e-7 m` | `1e-5 m` |
| reprojection | `3.62e-6 px` | `1e-4 px` |
| scene | `8.33e-8 m` | `1e-5 m` |

rotation 的容差略大，是因为 identity 附近 `trace/arccos` 对 float roundoff 较敏感；矩阵空间和所有 metric 误差仍远低于 `1e-5`。

### 6.3 No-cut 与统一几何卫生

冻结 V14.4 已报告：

```text
camera-human closure max       3.21e-7 m
projection invariance max      9.35e-6 px
camera no-cut max diff          0
pointmap no-cut max diff        0
SMPL-X no-cut max diff          0
shared scale all cases          true
no extra contact patch          true
root calibration count          1
```

V14.5 没有修改这些算法代码。No-cut 结论来自冻结版本已有 exact check，而不是在 V14.5 中重新设计一条路径。

## 7. Common Anchor 审计

### 7.1 原协议结果

| post-cut root cue | camera mean | absolute root mean |
|---|---:|---:|
| V11.4 raw root | `0.4027 m` | `0.1625 m` |
| Human projection root | `0.6741 m` | `0.3639 m` |
| DA3 root | `0.4345 m` | `0.1842 m` |

### 7.2 强制 common raw anchor 后

所有方法使用完全相同的 pre-cut anchor、last-root motion、rotation 和有效人体：

| post-cut root cue | camera mean | absolute root mean |
|---|---:|---:|
| raw root | `0.4027 m` | `0.1625 m` |
| Human projection root | `0.5422 m` | `0.1625 m` |
| DA3 root | `0.4394 m` | `0.1625 m` |

### 7.3 解释

Coupled placement 使用：

```text
t = a_pre - R * r_post
r_world_final = R * r_post + t = a_pre
```

因此只要 final root 和 camera translation 使用同一个 `r_post`，最终 world root 就被代数上锁定到所选 `a_pre`。不同 root cue 会改变 camera translation，但不会改变 final root 对该 anchor 的 closure。

所以：

- V11.4 的 camera 优势在 common anchor 后仍然存在；
- 旧报告中 V11.4 对 absolute human root 的排名，不能独立归因于 scale/root cue；
- absolute root 主要评价了 pre-cut anchor 和 motion model；
- V11.4 对 joints/vertices 仍可能通过统一 body scale 产生作用，但 root 本身不能作为独立 scale 因果证据。

这是本轮最重要的 confound 发现之一。

## 8. Gauge 与单位审计

### 8.1 为什么 Fixed 从 1.715 m 变成 0.712 m

| 协议 | Fixed translation mean | 含义 |
|---|---:|---|
| 旧协议 | `1.7151 m` | raw Human3R first-frame gauge，没有 deployable metric pre-shot scale |
| 统一协议 | `0.7118 m` | 先用 deployable pre-shot scale 缩放 Human3R camera translation 和 geometry，再评估同一类 Fixed transform |

这不是单纯的 SE(3) 坐标改名。统一协议在 candidate 进入 world 前改变了预测的物理单位，所以误差数值可以明显改变。

### 8.2 分数据源尺度

| Source | common pre-scale mean | dataset GT world scale | dataset camera baseline mean |
|---|---:|---:|---:|
| AvatarReX | `0.6927` | `1.0` | `2.987 m` |
| MVHuman100 | `0.3820` | `0.5417` | `1.904 m` |
| MVHuman200 | `0.5957` | `0.6500` | `2.095 m` |
| THuman | `1.0279` | `1.0` | `5.162 m` |

`dataset GT world scale` 是 GT dataset gauge 到 evaluator gauge 的转换量，只用于把 annotation 放入评测坐标系。审计确认它不进入 candidate generation。

### 8.3 坐标流程

```text
raw RGB
  -> frozen Human3R shot-local c2w / pointmap / SMPL-X
  -> camera cut: pre-decode hard reset
  -> deployable DA3/V11.4 pre/post scale
       * scale camera translation
       * scale pointmap
       * scale human root
       * scale root-centered joints/vertices/body offsets
  -> Fixed + V16 + optional Conditional VGGT: one R
  -> predicted pre-cut human anchor + predicted post-cut root: one t
  -> one fixed shot Boundary: camera / scene / complete SMPL-X

GT dataset c2w/human/scene
  -> evaluation-only alignment into frozen pre-shot gauge
  -> metrics only
```

结论：坐标、单位和 transform convention 本身通过；旧 Fixed 数字变化来自合法但不同的 metric prediction gauge，不应把两个旧数字直接横向比较，也不能把变化说成 Fixed 算法本身的新收益。

## 9. Scale Pareto 与敏感性

在每源 12 个、总计 48 个冻结样本上，围绕 predicted scale 做固定 grid 扫描。

### 9.1 predicted scale 接近各指标 grid optimum 的比例

| 指标 | 比例 |
|---|---:|
| camera | `54.17%` |
| human joints | `22.92%` |
| scene | `14.58%` |

### 9.2 predicted scale 上下 5% 的局部误差范围

| 指标 | mean range | P90 range |
|---|---:|---:|
| camera | `0.0930 m` | `0.1569 m` |
| joints | `0.0178 m` | `0.0352 m` |
| scene | `0.0334 m` | `0.0643 m` |

### 9.3 解释

- V11.4 predicted scale 处于较合理的 camera Pareto 区域；
- 它不是普遍的 human-optimal scale，更不是 scene-optimal scale；
- camera 对小幅 scale 变化最敏感；
- AvatarReX 的 joint-optimal multiplier 常落在 grid 上界 `1.25`，而 scene-optimal 常偏向更小 scale；
- MVHuman 的 camera、human 和 scene optimum 也并不重合。

因此 V11.4 的正确描述是：

> 当前 scale cue 是 camera-oriented shared scale，并非已经证明的 camera-human-scene 共同 metric optimum。

## 10. Scene 指标与 Trade-off

### 10.1 180-cut common 147-case subset

| 方法 | symmetric pointmap | background-only bidirectional |
|---|---:|---:|
| Fixed | `0.4829 m` | `0.5032 m` |
| V11.1 | `0.5223 m` | `0.5410 m` |
| V11.4 | `0.5323 m` | `0.5545 m` |
| Unified DA3 | `0.5565 m` | `0.5776 m` |
| Boundary Oracle | `0.5578 m` | `0.5788 m` |

V11.4 相比 Fixed：

| 统计量 | symmetric | background-only |
|---|---:|---:|
| mean degradation | `+0.0493 m` | `+0.0513 m` |
| improved rate | `18.37%` | `17.69%` |
| harmed rate | `81.63%` | `82.31%` |
| harmful `>0.05 m` | `51.02%` | `51.70%` |
| Wilcoxon p | `1.36e-11` | `4.31e-12` |

四个 source 都是同方向退化：

| Source | Fixed symmetric | V11.4 symmetric |
|---|---:|---:|
| AvatarReX | `0.5570 m` | `0.6114 m` |
| MVHuman100 | `0.1380 m` | `0.1843 m` |
| MVHuman200 | `0.1943 m` | `0.2506 m` |
| THuman | `0.7390 m` | `0.7800 m` |

### 10.2 如何理解 Boundary Oracle 也更差

Boundary Oracle 在这两个 scene 指标上也不优于 Fixed。这表明问题不只是 V11.4 transform 实现错误，还包含：

- post-cut Human3R local pointmap 与 GT camera gauge 并不完全相容；
- Fixed 的错误 camera transform 有时会偶然让两份 pointmap 更靠近；
- pointmap discontinuity 不是 camera correctness 的替代指标；
- 当前 local scene geometry 可能含 view-dependent 或 non-rigid depth error。

但是，两种独立 scene metric 都给出显著同方向结果，不能因此忽略 trade-off。论文必须准确写成：

> V11.4 显著改善 camera/human，但在当前 scene consistency 指标下存在约 5 cm 的统计显著退化。

## 11. Conditional VGGT 独立验证

### 11.1 Post-freeze 419 cases

| 方法 | mean | median | P90 | P95 |
|---|---:|---:|---:|---:|
| No VGGT | `14.91 deg` | `8.42 deg` | `35.59 deg` | `45.77 deg` |
| Always VGGT | `34.40 deg` | `12.34 deg` | `117.51 deg` | `158.34 deg` |
| Frozen Conditional | `13.16 deg` | `7.75 deg` | `32.58 deg` | `40.25 deg` |
| Best-of-two Oracle | `8.75 deg` | `5.07 deg` | `20.27 deg` | `28.27 deg` |

其他统计：

- trigger rate：`12.89%`；
- triggered improved：`90.74%`；
- triggered harmed：`9.26%`；
- improvement `>5 deg`：46 cases；
- harm `>5 deg`：4 cases。

AvatarReX 和 THuman 的 post-freeze trigger rate 为 0；收益主要来自 MVHuman100/200 的困难大旋转 tail。

### 11.2 Untouched 60-case holdout

| 统计 | 结果 |
|---|---:|
| trigger | `8/60 = 13.33%` |
| no-VGGT rotation mean | `17.62 deg` |
| Conditional rotation mean | `14.08 deg` |
| P90 | `47.91 -> 35.21 deg` |
| P95 | `62.57 -> 41.69 deg` |
| improvement `>5 deg` | 7 |
| harm `>5 deg` | 1 |
| paired Wilcoxon p | `0.0687` |

小 holdout 上均值和 tail 同方向改善，但 8 个 trigger 的统计功效不足，整体 paired p 没有低于 0.05。结合 419 个 post-freeze cases，Conditional VGGT 可以保留为 rotation-tail rescue；不能使用 Always VGGT。

Trigger 的冻结重算不依赖 source label 或 GT camera。仍需在论文中报告少量 harmful tail，而不是宣称 selector 完全可靠。

## 12. 真实连续 Multi-Cut Rollout

### 12.1 协议

- 每源选择一个真实序列，共 4 个；
- 构造 18-frame `A -> B -> A -> B` 交替流；
- 每个 shot 2 帧，最多 8 个真实 cuts；
- Human3R 真正逐帧 recurrent 运行；
- 每次 cut 都在第一帧 decode 前实际 reset；
- 前一次 predicted world 成为下一次 anchor；
- 每次 cut 不回到 GT gauge；
- scale 和 Boundary 每 shot 只计算一次；
- GT 只在 candidate 全部生成后评测。

这不是 single-cut error transform 的离线相乘。

### 12.2 V11.4 随 cut 数增加

| cuts | camera drift | rotation drift | root drift | joint drift | scene discontinuity |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.229 m` | `7.81 deg` | `0.093 m` | `0.150 m` | `0.259 m` |
| 2 | `0.326 m` | `23.97 deg` | `0.094 m` | `0.208 m` | `0.280 m` |
| 4 | `0.698 m` | `37.99 deg` | `0.134 m` | `0.278 m` | `0.271 m` |
| 8 | `0.946 m` | `59.03 deg` | `0.193 m` | `0.277 m` | `0.308 m` |

### 12.3 8-cut 方法比较

| 方法 | camera | rotation | root | joints | scene |
|---|---:|---:|---:|---:|---:|
| Hard Reset + Fixed | `0.994 m` | `41.87 deg` | `0.240 m` | `0.349 m` | `0.221 m` |
| V11.1 | `0.964 m` | `59.03 deg` | `0.180 m` | `0.267 m` | `0.287 m` |
| V11.4 | `0.946 m` | `59.03 deg` | `0.193 m` | `0.277 m` | `0.308 m` |
| Unified DA3 | `0.953 m` | `59.03 deg` | `0.198 m` | `0.279 m` | `0.337 m` |

V11.4 的 camera translation 最好，但优势只有约 1.8 cm 相对 V11.1、4.8 cm 相对 Fixed。与此同时：

- rotation 明显累积到 `59 deg`，比 Fixed 还差；
- root/joints 不优于 V11.1；
- scene discontinuity 高于 Fixed；
- 4 个序列样本量很小，只足以暴露失败，不能给出稳定泛化估计。

所以真实 8-cut 标准失败。当前方法只能称为 single-boundary 或 short-horizon re-anchoring，不能称为长期稳定 multi-shot world mapping。

本轮没有把 optional V14.2 continuity 接入这 4 个 recurrent alignment rollout。由于 continuity 不允许修改 R/s/t，它不会修复这里的 camera rotation drift；但其 shape/pose 稳定效果仍应单独保留，不能用本结果替代 continuity 评测。

## 13. 全新 Untouched Holdout

### 13.1 总体结果

表中 camera 为 mean / median / P90 / P95：

| 方法 | camera translation | rotation mean / P90 | root | joints | vertices | scene | catastrophic | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed | `0.663/0.541/1.318/1.625` | `23.05/61.60` | `0.234` | `0.291` | `0.285` | `0.475` | `15.0%` | `41.7%` |
| V11.1 | `0.502/0.413/0.939/1.099` | `14.08/35.21` | `0.195` | `0.245` | `0.239` | `0.536` | `5.0%` | `56.7%` |
| V11.4 + Conditional VGGT | `0.450/0.322/0.822/1.008` | `14.08/35.21` | `0.195` | `0.241` | `0.236` | `0.546` | `5.0%` | `66.7%` |
| Unified DA3 | `0.487/0.360/0.934/1.095` | `14.08/35.21` | `0.206` | `0.264` | `0.260` | `0.564` | `5.0%` | `60.0%` |

Scene 使用所有方法共同有效的 46/60 子集。

### 13.2 V11.4 相比 Fixed 的 paired 结果

| 指标 | Fixed -> V11.4 | 改善样本 | harmful `>0.05 m` | Wilcoxon p |
|---|---:|---:|---:|---:|
| camera | `0.663 -> 0.450 m` | `78.3%` | `16.7%` | `3.78e-6` |
| root | `0.234 -> 0.195 m` | `70.0%` | `20.0%` | `0.00675` |
| joints | `0.291 -> 0.241 m` | `66.7%` | `13.3%` | `0.00136` |
| vertices | `0.285 -> 0.236 m` | `66.7%` | `13.3%` | `0.00199` |
| scene | `0.475 -> 0.546 m` | `19.6%` | `52.2%` | `3.86e-6` |

这清楚复现了 180-case 的主模式：camera/human 改善，scene 显著退化。

### 13.3 分 source

| Source | Fixed -> V11.4 camera | Fixed -> V11.4 root | Fixed -> V11.4 scene | 结论 |
|---|---:|---:|---:|---|
| AvatarReX | `0.250 -> 0.220` | `0.131 -> 0.137` | `0.577 -> 0.632` | camera 小幅改善，root/scene 略退化 |
| MVHuman100 | `0.827 -> 0.496` | `0.233 -> 0.226` | `0.067 -> 0.220`，仅 4 scene-valid | camera 明显改善，scene 风险大 |
| MVHuman200 | `1.135 -> 0.876` | `0.427 -> 0.356` | `0.087 -> 0.217`，10 valid | camera/human 改善但仍是最差 source |
| THuman | `0.559 -> 0.315` | `0.195 -> 0.103` | `0.718 -> 0.746` | camera/human 强改善，scene 轻微退化 |

MVHuman200 的 V11.4 camera mean 仍为 `0.876 m`、rotation mean 为 `32.62 deg`、success 只有 `16.7%`。因此 holdout 没有出现某个 source 相比 Fixed 的 camera 灾难性系统退化，但 worst-source 的绝对精度仍不足。

### 13.4 Root 结果的正确归因

Holdout 上 V11.1 和 V11.4 的 root mean 都是 `0.195479 m`，几乎逐样本一致。V11.4 相比 V11.1 的额外收益主要是：

- camera `0.502 -> 0.450 m`；
- joints `0.245 -> 0.241 m`；
- vertices `0.239 -> 0.236 m`。

因此不能把 Fixed 到 V11.4 的全部 root gain 归因于 uniform scale。大部分 root gain 来自共同的 V16/anchor/coupled formulation；V11.4 scale 的独立贡献主要体现在 camera 和完整 body 尺寸，而不是 final root anchor。

## 14. 实际速度、显存与确定性

### 14.1 测试协议

- GPU：NVIDIA L20；
- Human3R、Keypoint R-CNN、DA3Metric-Large、VGGT-1B 同时常驻；
- 8 cases，每个重复 3 次；
- 不计 GT evaluator、CPU scene metric 和可视化；
- cut cue latency 包含 keypoint、两张图的 DA3、可选 VGGT 和几何后处理；
- cut cue latency 不包含该帧本身的 Human3R base forward。

### 14.2 结果

| 项目 | mean | median | P90 | P95 |
|---|---:|---:|---:|---:|
| cut cue，总体 | `2.488 s` | `1.743 s` | `6.185 s` | `7.037 s` |
| triggered VGGT cut cue | `3.615 s` | `2.319 s` | `6.878 s` | `7.171 s` |
| untriggered cut cue | `0.609 s` | `0.407 s` | `1.084 s` | `1.106 s` |
| normal Human3R | `3.570 FPS` | `3.571 FPS` | `3.579 FPS` | `3.580 FPS` |

运行时小样本故意覆盖困难 branch，trigger rate 为 `62.5%`，明显高于 untouched holdout 的 `13.3%`，因此总体 `2.488 s` 不是自然数据 trigger 频率下的无偏均值。

若所有步骤串行，加入一个 normal Human3R frame 的约 `0.280 s`：

```text
untriggered cut end-to-end estimate ~= 0.889 s
triggered cut end-to-end estimate   ~= 3.895 s
sample-weighted mean estimate       ~= 2.768 s
```

若平均每 `N` 帧一个 cut，基于本测试均值的摊销 frame time 可写为：

```text
T_amortized ~= 0.280 + 2.488 / N seconds
```

例如 `N=30` 时约 `2.76 FPS`，`N=100` 时约 `3.28 FPS`。这仍是 causal streaming，但不是实时 25/30 FPS。

### 14.3 显存

| 状态 | 显存 |
|---|---:|
| 四模型常驻 allocated | `10.46 GiB` |
| 四模型常驻 reserved | `10.57 GiB` |
| peak allocated | `12.21 GiB` |
| peak reserved | `12.94 GiB` |

在 L20 46 GB 上显存充足。

### 14.4 确定性

三次重复：

```text
scale max diff     = 0
Boundary max diff  = 0
trigger change     = false
branch change      = false
```

固定输入顺序时部署 candidate 完全确定。该结论与第 4.2 节的 manifest-order 风险不矛盾：重复同一 manifest 是确定的，改变 manifest index 会改变旧 V10 的随机采样 seed。

## 15. 审计标准逐项判断

1. **从 raw RGB 与 cache 一致：部分满足。** Human3R/Fixed 在原 manifest 上精确一致；完整旧 180 后处理 cache 未全部逐项 raw 重放，并发现 manifest-order RNG 风险。
2. **GT/source/metadata 不改变 deployable 输出：在缓存候选层满足。** 尚缺所有扰动的全 raw-RGB 动态 taint run。
3. **独立 evaluator 一致：满足。** 所有 metric 在冻结容差内。
4. **合成 Sim(3) 数值恢复：满足。** 最大几何误差约 `1.8e-15 m`。
5. **Common anchor 后 V11.4 优势仍在：camera 满足，absolute root 不满足。** Root 排名存在 anchor confound。
6. **Gauge 完整解释且无 GT scale 推理：满足。** `1.715 -> 0.712 m` 的原因已明确。
7. **Predicted scale 位于合理 Pareto：camera 满足，joint/scene 不充分。** 当前 cue 明显 camera-oriented。
8. **Scene trade-off 准确量化：满足审计要求，但方法本身存在显著退化。**
9. **Conditional VGGT held-out 泛化：基本满足。** 419 post-freeze 和 60 untouched 均同方向改善 tail，但存在少量 harmful trigger。
10. **真实 8-cut drift 低于 Fixed/V11.1：不满足。** 只有 camera translation 略好，rotation/root/joints/scene 不同时更好。
11. **Untouched holdout 稳定改善且无 source/capture 灾难：部分满足。** camera/human 复现，scene 退化；MVHuman200 绝对误差仍高。
12. **No-cut exact no-op：满足冻结 V14.4 检查。**
13. **严格流式速度与显存：显存满足，因果满足，实时帧率不满足。**

## 16. 最终必须回答的 10 个问题

### 1. 当前主结果是否可从原始 RGB 完整复现？

**部分可以。** 保持原 manifest 顺序时，8 个代表样本的 raw-RGB Human3R、Fixed transform 和指标精确一致；全新 60-case holdout 也完成了 fresh end-to-end 推理。但旧 180-case 的全部 DA3/VGGT/final Boundary cache 尚未逐项 raw 重放，而且旧 V10 candidate 对 manifest index 敏感。

### 2. 是否存在任何 GT、source 或 metadata 泄漏？

**未发现泄漏证据。** 180 candidates 的 GT/source/camera/path 扰动均不改变 candidate，GT scale 只进入 evaluator。严格限定：这是缓存候选层的充分测试，不是所有 raw-RGB 扰动的完整动态 taint proof。

### 3. 坐标、单位和 gauge 是否完全正确？

**几何实现和 evaluator convention 正确。** 合成 Sim(3)、独立 evaluator、closure、projection 和 c2w/w2c 全部通过。旧 Fixed `1.715 m` 与统一协议 `0.712 m` 的差异来自 deployable pre-shot metric scale 改变了预测物理 gauge，不是 GT scale 泄漏，也不是简单坐标重命名。

### 4. V11.4 的优势是否独立于 pre-cut anchor 定义？

**Camera 优势基本独立，absolute human-root 优势不独立。** Common anchor 后 V11.4 camera 仍为 `0.403 m`，优于 human projection 和 DA3 root cue；但三个 coupled 方法的 final root 都变成同一个 `0.163 m`，证明旧 root 排名受 anchor confound。

### 5. Scene 是否保持，还是存在显著 trade-off？

**存在显著 trade-off。** 180-set 的 symmetric scene 从 `0.483` 退化到 `0.532 m`，81.6% 样本变差；untouched holdout 从 `0.475` 退化到 `0.546 m`。两套 scene metric 和四个 source 都是同方向。

### 6. Conditional VGGT 是否能在真正 held-out 数据上泛化？

**能改善 rotation tail，但不是无风险。** 419 post-freeze cases 上 mean/P90/P95 都改善；60 untouched cases 上 mean `17.62 -> 14.08 deg`、P90 `47.91 -> 35.21 deg`，7 个大幅改善、1 个大幅退化。它适合作为 conditional tail rescue，不适合 always-on。

### 7. 一个真实连续的 8-cut rollout 是否仍然稳定？

**不稳定。** V11.4 最终 camera drift `0.946 m`、rotation drift `59.03 deg`。它仅在 camera translation 上略优于 Fixed/V11.1，不能证明长期 world gauge 稳定。

### 8. 新的 untouched holdout 是否复现 180-cut 的结论？

**复现了主要模式。** V11.4 camera `0.663 -> 0.450 m`，root/joints/vertices 也下降，camera paired `p=3.78e-6`；同时 scene `0.475 -> 0.546 m` 且显著退化。也就是说，它复现的是 camera/human gain + scene trade-off，而不是三者同时提升。

### 9. 当前实际推理速度是否满足严格流式要求？

**满足 causal/online 定义和显存约束，不满足实时视频帧率。** Normal Human3R 为 `3.57 FPS`；额外 cut cue untriggered 约 `0.61 s`、triggered 约 `3.61 s`。模型同时常驻 peak reserved `12.94 GiB`，三次重复完全确定。

### 10. V11.4 是否可以正式冻结为最终论文主线？

**不能按原定义无条件冻结为完整 camera-human-scene、长期 multi-shot 主线。** 可以冻结为效果明确的 single-boundary / short-horizon camera-human-priority 方法，并在论文中显式报告 scene trade-off、anchor confound、manifest-order RNG 风险和 8-cut 失败。若论文主张必须是完整三者统一和长期流式稳定，则当前版本未达到冻结标准。

## 17. 建议的论文表述

推荐：

> We apply a pre-decode reset at camera cuts and estimate one frozen shot-level similarity transform. A bounded torso-motion rotation with conditional VGGT tail rescue and a uniform shot scale substantially improve camera and human placement on both the development protocol and a capture-disjoint holdout. The method is causal and deterministic for a fixed stream order. However, the current Human3R scene reconstruction is not fully explained by the same scalar gauge: pointmap continuity degrades by about 5 cm on average, and recurrent 8-cut experiments still accumulate substantial rotation drift. We therefore position the method as short-horizon camera-human re-anchoring rather than complete long-term camera-human-scene mapping.

不推荐：

> Our unified method simultaneously improves camera, human, and scene and remains stable over arbitrarily many camera cuts.

## 18. 结果与代码位置

主要结果：

```text
output/v14_5_final_audit/offline/v14_5_final_geometry_leakage_audit.json
output/v14_5_final_audit/raw_rgb_full_context/raw_cache_integrity_full_context.json
output/v14_5_final_audit/raw_rgb_rerun/raw_cache_integrity.json
output/v14_5_final_audit/true_recurrent_multicut/v14_5_true_recurrent_multicut.json
output/v14_5_final_audit/runtime/v14_5_runtime_determinism_audit.json
output/v14_5_final_audit/untouched_holdout/records/freeze_summary.json
output/v14_5_final_audit/untouched_holdout/evaluation/v14_5_untouched_holdout.json
```

审计脚本：

```text
scripts/v14_5_final_geometry_leakage_audit.py
scripts/v14_5_raw_cache_integrity.py
scripts/v14_5_true_recurrent_multicut_audit.py
scripts/v14_5_freeze_capture_holdout.py
scripts/v14_5_cache_2d_keypoints.py
scripts/v14_5_untouched_holdout.py
scripts/v14_5_runtime_determinism_audit.py
scripts/v14_5_run_frozen_script.py
```

所有 GPU 审计命令都使用：

```bash
TMPDIR=output/v14_5_final_audit/tmp
```

以避开根文件系统空间不足；模型推理阶段均在 CUDA 上执行。
