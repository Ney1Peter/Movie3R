# V14 BRTC acceleration-gated FAGD 实验记录

> 日期：2026-08-01。目标是在不使用未来帧或额外模型的前提下，保留 FAGD-0.9 的空间收益，
> 同时避免 EgoHumans world-root Accel 退化。

## 1. 想法

在 shot boundary 到达时，在线系统已拥有同一人物的前两个 pre-shot root。对 frozen BRTC 和
FAGD 两个候选，计算纯预测的边界二阶差分：

```text
score = mean_i ||root_post_i - 2 * root_pre_i + root_preprev_i||
```

只在 FAGD score 不大于 BRTC score 时使用 `alpha=0.9` 的 group-only damping；否则 exact
BRTC v1。输入是已到达的两个 pre root 与当前 post root，不读 GT 或 future post。

## 2. MultiHuman 空间结果

Acceleration gate 在 `three offset0` 上从 30 个可用 FAGD boundary 中保留 18 个：

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|
| v1 | .225088 | .270404 | .248604 | .102222 | .260536 | 6.6% |
| ungated FAGD | .220839 | .267848 | .245232 | .102222 | .260536 | 6.6% |
| acceleration-gated | **.220714** | **.267569** | .245594 | .102222 | .260536 | 6.6% |

在未参与设计的 same-visibility 集上，空间收益也稳定：

| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm | Applied |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| offset1 | v1 | .231437 | .274493 | .252451 | .098351 | .258779 | 7.2% | - |
| offset1 | gated | **.227150** | **.271848** | **.249567** | .098351 | .258779 | 7.2% | 18/42 |
| dance | v1 | .125131 | .177804 | .152914 | .044141 | .078318 | 3.3% | - |
| dance | gated | **.114954** | **.169830** | **.143664** | .044141 | .078318 | 3.3% | 41/61 |
| box | v1 | .372345 | .421610 | .434528 | .063069 | .427334 | 6.4% | - |
| box | gated | **.352668** | **.402909** | **.414406** | .063069 | .427334 | **5.8%** | 56/78 |

因为每个 boundary 最终仍只改变公共 group translation，pair layout 与 v1 bit-exact。

## 3. EgoHumans 终验：安全但没有动作

在三条同-forward EgoHumans chain 上，strict full-one-to-one/all-accepted 与历史均可用的
boundary 只有一个。该边界的 observable score 为：

```text
BRTC = 0.20503 m
FAGD = 0.21057 m
```

因此 gate 在 `0/6` boundary 应用 FAGD，整条 candidate 与 v1 bit-exact：

| Method | W | WA | Root | Joint | Vertex | Pair distance | Pair vector | Root Accel | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 314.059 | 202.461 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 23.8% |
| gated | 314.059 | 202.461 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 23.8% |

它满足 non-regression，但没有产生任何提升，所以不能计作“优于当前版本”的方法：

```text
NO_GO_ACCELERATION_GATED_STRICT_FAGD
```

## 4. 对称 alpha grid 失败

还测试了完全可观察的 `alpha∈{0.8,0.9,1.0,1.1,1.2}`，选择 predicted acceleration score
最低者。它在 development 已失败：root `.225088→.225134`、joint
`.270404→.270472`、vertex `.248604→.249596`、harm `6.6%→9.0%`。这说明 Human3R 的预测
二阶差分不能直接当作 GT acceleration 的替代目标。

## 5. 结论与产物

前两帧预测 root 的 acceleration consistency 可以保守筛掉 FAGD，但目前过于保守；放宽阈值
会重新引入未经确认的 Accel 风险。它适合作为安全 veto，不是新主线。后续时序改进应训练
显式 calibration 或使用可靠 persistent identity/velocity state，而不是继续手调 score margin。

```text
versions/v14/probe_brtc_acceleration_gated_fagd.py
versions/v14/eval_brtc_acceleration_gated_fagd_egohumans.py
output/v14/fine_alignment_research/brtc_acceleration_gated_fagd/
```
