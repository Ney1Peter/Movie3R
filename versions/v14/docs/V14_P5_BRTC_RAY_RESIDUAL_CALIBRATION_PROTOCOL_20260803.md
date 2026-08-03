# V14 P5：BRTC 后的可训练射线残差校准

日期：2026-08-03  
状态：**开发协议；尚无结果。**

## 假设

BRTC-LC 的 ray triangulation 是一个显式、可解释的 root-depth proposal，但它的 residual
幅度可能仍受 joint ray gap、parallax 和跨 joint disagreement 的系统性影响。P5 不训练另一个
全局 Boundary，也不复制 V9 shadow human；它只训练一个低容量、可审计的标量校准器：

```text
frozen B0 + frozen BRTC output
  + seven BRTC runtime evidence statistics
  -> bounded residual delta along the raw post root ray
  -> translate that person's root / joints / vertices only
```

相机、scene、B0、pose、shape、orientation、recurrent state 和 association policy 均冻结。
未匹配或 BRTC-rejected row 必须 exact B0/BRTC fallback。这个模型不使用 P2 token；P2 已经
证明 token 不能承担可靠 WHO cue，不能悄悄作为 residual feature。

## 固定数据、特征和标签隔离

P5 development 重用 P1/P2 同 checkpoint schema cache 的 36 个 `three` events。runtime
首先从 P1 读取 geometry match、frozen BRTC output 和下列**固定顺序** feature：

```text
raw_m,
valid_count,
median_gap_m,
max_gap_m,
median_sine,
min_sine,
mad_m
```

每个 BRTC accepted row 的 runtime ray 是 raw B0 post root 到 frozen B0 post camera center 的
unit direction。只有 runtime candidate/feature 都完成后，evaluator 才从 P1 target、P2 的
identity label 取得训练 label：

```text
y = dot(GT_root_world - BRTC_root_world, raw_post_root_ray)
```

GT identity 只用来在训练/评估时标记几何 association 是否正确，绝不输入模型或 runtime gate。
部署时模型会被应用到原本 BRTC 所接受的 anonymous row；因此最终报告必须同时给出正确匹配
stratum 和全部 geometry-match stratum，不能只看 GT-clean subset。

## 固定学习和开发评估

采用唯一预注册模型：`StandardScaler + Ridge(alpha=1.0)`。无 grid、无 MLP、无特征筛选；
输出固定 clip 到 `[-0.30, +0.30] m`。在六个时间戳 group `{500,700,900,1100,1300,1500}`
做 leave-one-timestamp-out：每一 fold 仅用其他五个 timestamp 的 evaluator labels 训练，
再对留出 timestamp 的 runtime rows 产生动作。所有 leave-out action 合并后才评价。

这只是一条 capture-internal development evidence。若通过，才把所有 `three` correct-match
训练样本拟合为一个 JSON-serializable frozen model，并在新 cache 上做 confirmation；本轮 CV
不允许被写成 final generalization。

## Go / No-Go

仅当以下条件同时成立，才生成 frozen model 进入 confirmation：

```text
correct, BRTC-accepted training/evaluation rows >= 60
LOTO correct-stratum root/joint/vertex mean each improve by >= 5 mm vs BRTC
correct-stratum per-metric harm >5 cm <= 10%
all geometry-pair root mean improves by >= 5 mm; root harm >5 cm <= 10%
camera bit-exact; runtime features/actions precede evaluator access
```

否则记录 `NO_GO_BRTC_RAY_RESIDUAL_CALIBRATION`，不扩展 feature、alpha、cap 或 network
capacity。即使通过，也只说明一个 **BRTC residual calibration** 在 `three` development 有
学习信号；在 isolated multi-human confirmation 前不得作为最终方法或 ICLR claim。

## 已完成开发结果（2026-08-03）

开发判定：

```text
GO_TO_FROZEN_CONFIRMATION
```

这不是最终 promotion。P1/P2 的 36-event cache 中，`92` 个 correct geometry match 且
BRTC-accepted rows 被分成六个完整 timestamp group；每个预测都由其余五组的 Ridge 训练，
再作用于留出 timestamp。因此以下数字不是同一 row 训练再测试的拟合误差。

| Stratum / method | N | Root (m) | Joint (m) | Vertex (m) |
|---|---:|---:|---:|---:|
| all geometry pairs: frozen BRTC | 96 | .1724 | .1983 | .1788 |
| all geometry pairs: P5 LOTO Ridge | 96 | **.1574** | **.1864** | **.1671** |
| correct+BRTC accepted: frozen BRTC | 92 | .1638 | .1906 | .1703 |
| correct+BRTC accepted: P5 LOTO Ridge | 92 | **.1482** | **.1782** | **.1581** |

正确关联 stratum 相对 BRTC 的 root/joint/vertex gains 分别为 `15.61/12.44/12.25 mm`；
全部 geometry rows 仍有 `14.96/11.92/11.74 mm` gain。相对 BRTC 的 `>5 cm` harm 分别为
root `4.35%`、joint `4.35%`、vertex `5.43%`（all rows 为 `4.17/4.17/5.21%`），均低于
预注册 `10%` 上限；所有九项 development gate 通过，B0 camera 保持 bit-exact。

冻结候选已保存为：

```text
output/v14/fine_alignment_research/p5_brtc_ray_residual_calibration/
  P5_FROZEN_MODEL_BEFORE_CONFIRM.json
```

该 JSON 包含唯一的 seven feature order、standardization mean/scale、Ridge coefficient、
intercept、`alpha=1.0` 和 `±0.30 m` cap。下一步只能以这一精确模型，在一个与 `three`
P5 label 完全分离的 multi-human cache 上运行；不得据此修改 feature、alpha、cap 或重训
范围。尤其不应把 timestamp-CV 误称为跨 capture 泛化或 Multi-THuMBS 对榜成绩。

## 冻结 confirmation（执行前固定）

confirmation manifest 固定为已有 MultiHuman `dance` / `box` protocol 中的 `offset=0`
first-post rows：`14 + 15 = 29` boundaries。它们历史上已被其他 V14 diagnostics 打开，
故是 **frozen replay confirmation** 而非 pristine test；但从不进入 P5 feature、alpha、cap
或 Ridge fitting。所有 events 都用与 P5 development 相同的 P0 checkpoint 重新 forward，
不复用旧 checkpoint 的 BRTC cache。

模型只能读取已冻结的 `P5_FROZEN_MODEL_BEFORE_CONFIRM.json`，每个 BRTC accepted
anonymous geometry match 都直接执行该 JSON 的 linear inference；不得利用 confirmation GT
identity 筛掉 action。GT identity 仅在输出后给 correct-match 分层。

冻结 confirmation promotion 条件：

```text
correct BRTC-accepted rows >= 24
correct-match root/joint/vertex mean each improve >= 5 mm vs BRTC
all geometry-pair root mean improves >= 5 mm
all geometry-pair root harm >5 cm <= 10%
camera bit-exact; unmatched/rejected exact fallback
```

任何条件失败则 `NO_GO_BRTC_RAY_RESIDUAL_CALIBRATION_CONFIRMATION`；不得在 dance/box
上重训或改变 JSON。通过也只晋级为 current-P0 / MultiHuman 的 qualified person-root
calibration candidate，仍需独立 real-capture confirmation、long-stream and official-style
evaluation 才能成为 ICLR 主线。

## 已完成 frozen replay confirmation（2026-08-03）

确认判定：

```text
QUALIFIED_P5_BRTC_RAY_RESIDUAL_CALIBRATION
```

冻结 JSON 在 confirmation 开始前已存在，且此阶段只做其线性推理。以同一个 P0 checkpoint
重新运行 `dance` / `box` 的 29 个 `offset=0` first-post boundaries，得到 58 个有 evaluator
target 的 geometry rows；其中 `57` 个是 evaluator-only correct association 且 BRTC accepted，
另一个 BRTC rejected row 保持 exact fallback。运行时没有读取这些 identity labels，也没有
按它们筛掉任何 P5 action。

| Stratum / method | N | Root (m) | Joint (m) | Vertex (m) |
|---|---:|---:|---:|---:|
| all geometry pairs: frozen BRTC | 58 | .2692 | .3019 | .2916 |
| all geometry pairs: frozen P5 | 58 | **.2382** | **.2683** | **.2622** |
| correct+BRTC accepted: frozen BRTC | 57 | .2674 | .3011 | .2903 |
| correct+BRTC accepted: frozen P5 | 57 | **.2358** | **.2668** | **.2604** |

正确关联 stratum 的 root/joint/vertex gain 为 `31.61/34.22/29.85 mm`；保留所有 geometry
rows 后仍为 `31.07/33.63/29.34 mm`。相对 BRTC 的 `>5 cm` harm 为：all root/joint/vertex
`1.72/0.00/1.72%`，correct stratum `1.75/0.00/1.75%`，均远小于 `10%` 的冻结上限。所有
confirmation gate 均通过，29 个 event 的 B0 camera SHA256 均复核，camera max change=`0.0`，
没有 future post frame、外部预训练模型或 shadow state commit。

产物：

```text
config/manifests/v14_p5_brtc_ray_residual_confirm_20260803.json
output/v14/fine_alignment_research/p5_brtc_ray_residual_confirmation_cache/
output/v14/fine_alignment_research/p5_brtc_ray_residual_confirmation/
  P5_CONFIRMATION_REPORT.json
```

所以当前可部署的人体 root 精对齐链可以明确写为：

```text
frozen B0 camera
 -> geometry association
 -> frozen BRTC-LC (explicit two-view ray proposal + layout consensus)
 -> frozen 7-feature P5 Ridge residual along raw post root ray
 -> rigid translation of that person's root/joints/vertices
 -> exact fallback for unmatched or BRTC-rejected person
```

这是一条有价值、可训练但非常小的 **calibration**，不是把人重新端到端回归，也不修改
B0。其证据目前覆盖 `three` timestamp-CV 加上已读 dance/box frozen replay；它尚未证明
EgoHumans/general real capture 泛化、automatic WHO 完全解决、orientation/articulation、长
stream temporal stability 或 Multi-THuMBS official comparison。因此可以晋级为当前主线的
root residual module，但不能提前写成 ICLR 完整系统胜利。
