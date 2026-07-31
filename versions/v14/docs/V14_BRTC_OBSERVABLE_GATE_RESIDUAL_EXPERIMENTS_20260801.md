# V14 BRTC 可观测 gate / residual 独立实验

> 日期：2026-08-01
>
> 约束：不引入预训练模型；不使用 GPU；严格 last-pre / first-post 在线；相机保持
> bit-exact frozen B0；GT 仅在 candidate 输出以后用于评价；不修改 completeness runtime、
> V9 或 frozen BRTC-LC v1。

> **后续终验说明：**本文保留 FAGD 的 development/same-visibility 空间规律，但第 7 节的
> 初步推荐已被 `V14_BRTC_STRICT_DEPLOYABLE_FAGD_20260801.md` 和
> `V14_BRTC_VARIABLE_VISIBILITY_RESULTS_20260801.md` supersede。线性 completeness、soft
> completeness 与 strict FAGD 均未通过完整部署门槛；frozen BRTC-LC v1 仍是默认 runtime。

## 1. 结论

本轮测试了两个只依赖 BRTC 当前可观测量的候选。

第一个候选根据 raw action 大小连续缩放 action，在 `three offset0` development 上表现很好，
但冻结后在 box 明显失败。因此结论为：

```text
NO_GO_OBSERVABLE_ACTION_MAGNITUDE_SHRINKAGE
```

第二个候选不再缩放完整 person action，而是利用 frozen BRTC 已有的分解：

```text
final_shift_i = group_shift + lambda * individual_residual_i
```

只有当当前 boundary 的所有 matched people 都通过 frozen ray gate 时，才执行：

```text
final_shift_i = 0.9 * group_shift + lambda * individual_residual_i
```

否则保持 frozen BRTC v1 原样。该方法称为：

```text
Full-Accept Group-Only Damping，简称 FAGD-0.9
```

它在 offset0 development 冻结后，在 offset1、dance、box 上同时改善 root、joint、vertex，
pair distance/vector 与 v1 bit-exact，>5 cm harm 不增加。在 EgoHumans 同-forward CPU cache
上，W、WA、root、joint、vertex 和两项 layout 也都改善，但 world-root Accel 从
`116.014` 变为 `117.973 mm/frame²`，退化 `1.959 mm/frame²`。

因此最终判断是：

```text
GO_FULL_ACCEPT_GROUP_ONLY_DAMPING_AS_SPATIAL_CANDIDATE
NO_GO_AS_STANDALONE_SPATIOTEMPORAL_FINAL_METHOD
```

它可以作为 completeness-weighted BRTC 之外的第二条明确空间候选，但不能替代后续因果
temporal stabilizer。

## 2. 数据与冻结协议

### 2.1 Development

仅使用：

```text
three offset0
41 cuts / 122 people
```

开发阶段可以读取 GT 评价指标，但 candidate 只读取 BRTC 已有的：

- frozen raw ray action；
- frozen accepted/rejected gate；
- accepted individual shift；
- 多人 group median shift；
- frozen observable layout 选择的 residual lambda；
- matched/accepted count。

### 2.2 冻结后的验证

策略文件在运行下面三组验证之前写入：

```text
three offset1：42 cuts / 125 people
dance：61 cuts / 122 people
box：78 cuts / 156 people
EgoHumans：3 条 15-frame chain / 6 cuts
```

FAGD 冻结策略：

```json
{
  "alpha": 0.9,
  "gate": "accepted_count == matched_count > 0",
  "application": "scale group median only; keep individual residual exact"
}
```

策略 SHA256：

```text
619789e07790054f18cdcddae1a1385c9b1cc66f55b654b464d5f52712a11453
```

## 3. 失败候选：Observable Action-Magnitude Shrinkage

### 3.1 想法

开发数据中 `0.05--0.15 m` 的小 action 符号可靠性明显低于大 action，因此尝试：

```text
confidence = min(1, abs(raw_action) / full_trust_action)
scaled_action = max_scale * confidence * raw_action
```

只在 offset0 扫描 `max_scale` 与 `full_trust_action`。要求 root、joint、vertex、pair distance、
pair vector、coverage、>1 cm harm、>5 cm harm 全部不差于固定 `0.8` strong baseline。

冻结结果：

```text
max_scale = 0.95
full_trust_action = 0.20 m
policy SHA256 = 71f02b0ebbf97d1a283fe955505e296005279e499448becb080f86cef86c2b50
```

### 3.2 Development 结果

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | .225088 | .270404 | .248604 | .102222 | .260536 | 15.6% | 6.6% |
| fixed 0.8 | .223813 | .270686 | .247476 | .103756 | .259358 | 13.1% | 5.7% |
| magnitude shrinkage | **.221037** | **.267818** | **.244868** | **.102350** | **.257719** | **11.5%** | **4.1%** |

Development 上它严格通过所有安全门。

### 3.3 冻结验证结果

| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---|---:|---:|---:|---:|---:|---:|
| offset1 | BRTC v1 | .231437 | .274493 | .252451 | .098351 | .258779 | 7.2% |
| offset1 | magnitude | **.228136** | **.271904** | **.249006** | .099852 | .258405 | **3.2%** |
| dance | BRTC v1 | .125131 | .177804 | .152914 | **.044141** | **.078318** | 3.3% |
| dance | magnitude | **.116156** | **.169875** | **.144708** | .047616 | .081204 | **0.0%** |
| box | BRTC v1 | .372345 | .421610 | .434528 | **.063069** | **.427334** | 6.4% |
| box | fixed 0.8 | **.341341** | **.392024** | **.401604** | **.058194** | **.421938** | 6.4% |
| box | magnitude | .362866 | .412748 | .424862 | .064499 | .428005 | **5.1%** |

失败原因很清楚：action magnitude 只能识别弱小 action 的低信噪比；box 的主要问题是大量
大 action 整体过激。`|raw action| >= 0.2 m` 后该策略几乎固定执行 `0.95×`，因此在 box
退回接近 v1，明显不如固定 `0.8`，两项 layout 甚至略差于 v1。

这个失败也说明，继续用 gap/MAD/parallax 去重新估计相近的 ray center 不是最关键的；更
重要的是区分共享 group translation 与真正需要保留的 individual residual。

## 4. 成功空间候选：Full-Accept Group-Only Damping

### 4.1 原理

Frozen BRTC 先得到每个人的 individual ray shift，再取多人 group median，并用 pre/post
observable layout 选择 residual lambda：

```text
individual_i = group + residual_i
final_i = group + lambda * residual_i
```

过去固定 `0.8` 是把 individual proposal 整体缩放后重新做 group/layout consensus。这会
同时改变 group 与 individual residual，因此可能损害多人 pair layout。

FAGD 只缩 group：

```text
if all matched ray gates pass:
    final_i = alpha * group + lambda * residual_i
else:
    final_i = frozen BRTC final_i
```

当所有 matched people 都接受时，`alpha * group` 对所有人完全相同。对任意两人：

```text
(root_i + alpha*group) - (root_j + alpha*group)
= root_i - root_j
```

所以 pair distance 与 pair vector 数学上保持不变；只允许共同人体层深度/位置变得更保守。
这正对应当前实验观察：BRTC 的多人相对 residual 有价值，但共享 action 稍微过激。

### 4.2 Development 扫描

只在 offset0 扫描：

```text
alpha = 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
```

要求五项误差、coverage 与 harm 全部不差于 BRTC v1，再最小化 root/joint/vertex。最终冻结
`alpha=0.9`。

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | .225088 | .270404 | .248604 | .102222 | .260536 | 15.6% | 6.6% |
| fixed 0.8 | .223813 | .270686 | .247476 | .103756 | .259358 | 13.1% | 5.7% |
| FAGD-0.9 | **.220839** | **.267848** | **.245232** | .102222 | .260536 | 15.6% | 6.6% |

相对 v1，root/joint/vertex 分别改善 `4.25/2.56/3.37 mm`，pair 两项 bit-exact，harm 不变。

## 5. offset1、dance、box 冻结验证

### 5.1 Three offset1

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|
| BRTC v1 | .231437 | .274493 | .252451 | .098351 | .258779 | 7.2% |
| fixed 0.8 | .229095 | .274085 | .250446 | .101313 | .258331 | 5.6% |
| FAGD-0.9 | **.226928** | **.271807** | **.248941** | .098351 | .258779 | 7.2% |

相对 v1，root/joint/vertex 改善 `4.51/2.69/3.51 mm`；pair bit-exact；harm 不增加。

### 5.2 Dance

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|
| BRTC v1 | .125131 | .177804 | .152914 | .044141 | .078318 | 3.3% |
| fixed 0.8 | .122179 | .171557 | .145379 | .046840 | .081514 | **0.0%** |
| FAGD-0.9 | **.113956** | **.168753** | **.142597** | **.044141** | **.078318** | 1.6% |

相对 v1，root/joint/vertex 改善 `11.18/9.05/10.32 mm`；pair bit-exact；harm 从 `3.3%`
降至 `1.6%`。

### 5.3 Box

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|
| BRTC v1 | .372345 | .421610 | .434528 | .063069 | .427334 | 6.4% |
| fixed 0.8 | **.341341** | **.392024** | **.401604** | **.058194** | **.421938** | 6.4% |
| FAGD-0.9 | .352626 | .402936 | .414447 | .063069 | .427334 | **5.8%** |

FAGD 没有超过 box 上的固定 `0.8`，但相对 v1 仍稳定改善 root/joint/vertex
`19.72/18.67/20.08 mm`，pair bit-exact，harm 下降 `0.6` 个百分点。

因此它提供了与 fixed 0.8 不同的明确 trade-off：

- fixed 0.8 在 box 绝对空间误差更低，但会改变两项 pair layout；
- FAGD 保证完整集合的 v1 layout 不变，同时显著降低人体绝对空间误差。

## 6. EgoHumans 同-forward CPU cache

这仍是自建的三条 15-frame `001_legoassemble` chain，不是 Multi-THuMBS 官方 split。没有
重跑 Human3R，只重放现有 22 MB current-checkpoint CPU geometry cache。

| Method | W | WA | Root | Joint | Vertex | Pair distance | Pair vector | Root Accel | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 350.614 | 235.207 | 420.163 | 416.226 | 414.913 | 188.485 | 388.351 | 160.517 | - |
| BRTC v1 | 314.059 | 202.461 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | **116.014** | 23.8% |
| FAGD-0.9 | **313.777** | **202.407** | **378.887** | **382.957** | **383.288** | **175.814** | **333.575** | 117.973 | 23.8% |

相对 v1：

- W 改善 `0.282 mm`；
- WA 改善 `0.054 mm`；
- root/joint/vertex 改善 `1.77/1.77/1.95 mm`；
- pair distance/vector 改善 `1.21/0.30 mm`；
- harm 不增加；
- camera max change 为 `0`；
- world-root Accel 退化 `1.959 mm/frame²`。

Ego 的 pair 不再严格相等，是因为 correction 在第一 cut commit 后会成为第二 cut 的 pre
state，且存在 incomplete/rejected boundary；最终仍是净改善。

## 7. 当时的候选排序（已被后续终验 supersede）

本阶段结束时曾区分三种作用不同的策略：

1. **Completeness-weighted BRTC**：EgoHumans 上 W/WA/root/Accel/harm 改善，但后续独立
   variable-visibility 证明线性阻尼损失 root/joint/vertex，因此未晋级。
2. **FAGD-0.9**：处理完整接受集合中的共享 group action 略过激；保证当前 boundary 的
   individual residual 和完整集合 layout 不变，是第二空间候选。
3. **固定 0.8**：在 box 的绝对空间误差最好，但会改变 layout，应保留为 strong baseline，
   不作为唯一默认动作。

后续确实实现并验证了更严格 wrapper：

```text
complete one-to-one + all accepted -> FAGD group-only 0.9
otherwise                          -> exact frozen BRTC v1
```

终验发现人数变化可 exact-v1，但等人数 identity replacement 仍误触发，且 Ego world-root
Accel 退化，所以最终为 `NO-GO_STRICT_FAGD_DEPLOYABLE`。

## 8. 可复现产物

```text
versions/v14/probe_brtc_observable_action_shrinkage.py
versions/v14/probe_brtc_full_accept_group_damping.py
versions/v14/eval_brtc_group_damping_egohumans.py

output/v14/fine_alignment_research/brtc_observable_action_shrinkage/
output/v14/fine_alignment_research/brtc_full_accept_group_damping/
```

命令：

```bash
.venv/bin/python versions/v14/probe_brtc_observable_action_shrinkage.py --phase dev
.venv/bin/python versions/v14/probe_brtc_observable_action_shrinkage.py --phase freeze
.venv/bin/python versions/v14/probe_brtc_observable_action_shrinkage.py --phase validate

.venv/bin/python versions/v14/probe_brtc_full_accept_group_damping.py --phase dev
.venv/bin/python versions/v14/probe_brtc_full_accept_group_damping.py --phase freeze
.venv/bin/python versions/v14/probe_brtc_full_accept_group_damping.py --phase validate

.venv/bin/python versions/v14/eval_brtc_group_damping_egohumans.py
```

所有新产物均位于 `/data/wangzheng/iJCV-CODE/Movie3R` 下，没有向系统根目录写文件。
