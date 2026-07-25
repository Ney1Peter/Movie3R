# V13 Phase 5：Causal Shot-Persistent Human Identity State

## 1. 结论摘要

本阶段完成了固定大小的跨 shot identity state、因果式 margin rollout、Top-6 identity
hypothesis 生成，以及冻结 Phase 2 Boundary 的 WHO-WHERE hypothesis probe。

最终结论分为两部分：

1. **Shot-persistent identity state 有效，Stage 0/1 通过。**
2. **当前手工 WHO-WHERE joint scorer 无效，不进入 commit，也不进入 frozen evaluation。**

在 MultiHuman `three` 的 15 条真实 multi-cut development streams、90 个 cuts 上：

```text
stateless unfiltered IDF1          = 0.8202
running-mean state candidate IDF1  = 0.9255
gain                               = +0.1053

causal zero-wrong accepted         = 133 / 255 matches
causal multi activation coverage   = 50.0%
wrong accepted                     = 0
Top-6 GT assignment recall         = 92.22%
reverse/random order agreement     = 100%
```

但是联合评分没有超过 identity-only margin：

```text
zero-wrong multi coverage:

identity-only margin = 51.11%
best joint score     = 35.56%
geometry-only        =  5.56%
```

因此当前不能宣称：

```text
Causal Joint WHO-WHERE Hypothesis Search 已成立
```

冻结决定仍是：

```text
GT-ID Uniform Multi-Human Consensus：保留为有效 geometry Oracle
Shot-persistent identity state：保留为有效 identity research component
Automatic joint multi-human commit：关闭
部署默认：Single-Human Movie3R
```

---

## 2. 实现边界

### 2.1 WHO state

新增：

```text
versions/v13/shot_persistent_identity.py
```

每个 external track 使用固定大小状态：

```text
track ID
last normalized feature
running mean
Welford running variance
five-observation medoid buffer
observation count
valid appearance count
quality sum
active/inactive
last seen
TTL
```

state 只保存人物 feature 和生命周期，不保存或预测 SE(3)。支持：

- read-only query；
- explicit commit；
- fixed TTL；
- fixed maximum track count；
- partial one-to-one assignments；
- unmatched/dustbin；
- `K <= 6` 的 bounded hypothesis enumeration。

### 2.2 WHERE geometry

新增的 geometry glue：

```text
versions/v13/causal_who_where.py
```

它不改变 Phase 2 求解器，只完成：

```text
identity hypothesis
-> anonymous geometry slots
-> frozen human_candidates()
-> Fixed Explicit
-> V16 torso residual, 20 degree bound
-> mean_raw_t Uniform Consensus
-> ONE shared Boundary
```

同一个 Boundary 同时应用于 camera pose、point cloud 和所有人体 root/joints/vertices。

为支持 inactive identity，另维护每个 track 最近 5 条**已经对齐到当前 world gauge**的人体几何。
这仍是固定大小 state。未对齐的 post-cut geometry 不会写入历史。

---

## 3. 数据和流协议

Development 只使用 MultiHuman Real-World-Capture `three`。

使用 5 个 start times：

```text
500, 700, 900, 1100, 1300
```

每个 start 构造三条 7-shot stream：

```text
cycle:  0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0
wide:   0 -> 3 -> 1 -> 4 -> 2 -> 5 -> 0
return: 0 -> 3 -> 0 -> 3 -> 0 -> 3 -> 0
```

总计：

```text
15 streams
90 cuts
70 unique shot caches
5 frames per shot
```

每个 shot 都使用原版冻结 Human3R `src/human3r_896L.pth` 独立推理：

```text
2048 x 2048 full frame
-> resize to 512 x 512
-> fresh scene/camera recurrent state
```

没有 full-frame crop。DINOv2 crop 来自 Human3R predicted bbox。GT identity、GT camera 和 GT
body 只在 candidate 生成后进入 evaluator。

---

## 4. 因果 commit 修正

初版 smoke 存在一个会高估 state 的协议错误：

```text
unfiltered Top-1 commit
-> 完整 stream 结束
-> post-hoc 选择 zero-wrong margin
```

这个结果不是可部署的，因为 risk gate 没有真正控制长期 state 更新。

正式实现改为：

```text
在 development 上生成候选 margin
-> 每个 margin 从 stream 起点完整 causal replay
-> cut 当时按固定 margin 决定 commit/dustbin
-> rejected detection 建立新 tentative track
-> 不更新旧 identity prototype
-> 按 zero wrong first, coverage second 选择工作点
```

冻结 margin：

```text
0.042287553364318435
```

该修正非常重要。单条 `cycle_f500` smoke 在错误 post-hoc 协议下会得到“继续”，严格 replay
后 Top-6 recall 只有 `83.33%`；完整 15-stream development 最终才稳定达到 `92.22%`。

---

## 5. Stage 0：Persistent state 结果

### 5.1 Stateful vs stateless

| 方法 | Candidate IDF1 | False positive | Top-6 recall |
|---|---:|---:|---:|
| Phase 4 stateless unfiltered | 0.8202 | - | - |
| State running mean | **0.9255** | 19 | 1.0000 before causal gate |
| State mean / variance normalized | 0.9051 | 26 | 0.9667 |
| State medoid | 0.8275 | 44 | 1.0000 |
| State last | 0.7960 | 54 | 0.9444 |

running mean 明显优于每个 cut 临时五帧 prototype。last 和 medoid 不稳定，说明单个 view
observation 很容易保留 shot-specific appearance。

### 5.2 Per-track variance

本阶段测试了：

```text
distance / track historical dispersion
dispersion floor = 0.03 / 0.05 / 0.08 / 0.12
```

四个 floor 的结果相同，IDF1 均为 `0.9051`，低于普通 running mean 的 `0.9255`。

结论：当前 DINOv2 + beta + pose feature 的 historical variance 不是可靠 uncertainty；人物自身
变化大不等于本次匹配更可信。不能把 per-track normalized distance 冻结为默认方法。

### 5.3 Causal safe operating point

正式 causal replay：

```text
accepted precision              = 100%
wrong accepted                  = 0
accepted recall                 = 52.16%
IDF1                            = 0.6856
multi activation coverage       = 50.0%
cut-level all accepted correct  = 100%
Top-6 GT assignment recall      = 92.22%
```

Phase 4 precision gate 在相同 90 cuts 上 multi coverage 为 `14.44%`。Persistent state 将安全
多人覆盖提升了 `+35.56` 个百分点。

### 5.4 Controls

| State control | IDF1 | False positive | Multi coverage | Top-6 recall |
|---|---:|---:|---:|---:|
| correct running state | 0.6856 causal | 0 | 0.5000 | 0.9222 |
| wrong-person state | 0.0617 | 73 | 0.3111 | 0.3778 |
| shuffled state | 0.0980 | 47 | 0.2333 | 0.3333 |
| zero state | 0.0000 | 0 | 0.0000 | 0.7778 |

zero-state Top-6 recall 较高来自小人数排列空间和 tie enumeration，不能解释为 identity 信息；它的
Top-1 IDF1 和可用 coverage 均为 0。correct state 明显优于所有 controls。

### 5.5 Detection-order audit

测试：

```text
每个 cut reverse post order
每个 cut random post order
整条 stream 从 bootstrap 起 reverse pre/post order
整条 stream 从 bootstrap 起 random pre/post order
```

所有 90 cuts 的语义身份映射 agreement 均为 `1.0000`。内部 track ID 数值允许变化，最终
GT-label-to-detection 语义结果不变。

---

## 6. Stage 1：Top-K 结果

固定：

```text
K = 6
one-to-one partial assignment
explicit unmatched penalty
dustbin enabled
```

正式 causal state trajectory 中：

```text
Top-6 GT assignment recall = 92.22%
```

达到预注册的约 `90%` 门槛，因此允许进入 Stage 2 probe。

但仍有 `7/90` cuts 的正确 assignment 不在 Top-6。任何 scorer 都不可能修复这些 cuts，候选
representation/state 仍是明确上限。

---

## 7. Stage 2/3：WHO-WHERE hypothesis probe

### 7.1 Probe 边界

每个 Top-6 hypothesis 都执行一次冻结 Boundary：

```text
matched track geometry history
-> per-human Fixed + V16 candidate
-> equal SO(3) mean
-> raw translation arithmetic mean
-> one B_k
```

joint scorer 没有更新 persistent state。本 probe 先回答“scorer 有没有资格 commit”，避免错误
scorer 污染后续状态。

### 7.2 Geometry score

无训练 score 使用：

```text
rotation candidate dispersion
translation candidate dispersion
leave-one-human-out residual
independent background pointmap residual
missing-candidate penalty
```

正确和错误 hypothesis 的中位数确实有总体差异：

| 信号 | Correct median | Wrong median |
|---|---:|---:|
| rotation dispersion | 27.39 deg | 101.05 deg |
| translation dispersion | 1.084 m | 3.244 m |
| leave-one-out score | 5.742 | 18.448 |
| pointmap residual | 1.515 m | 1.917 m |
| combined geometry score | 16.164 | 40.591 |

但分布尾部明显重叠。它可以改善平均排序，却不能安全判断最危险的少量 swap。

### 7.3 Ungated ranking

| Method | All matches correct | IDF1 | Wrong accepted | Camera composite | Catastrophic |
|---|---:|---:|---:|---:|---:|
| Identity Top-1 | 83.3% | 0.929 | 18 | 1.598 | 26.7% |
| Geometry only | **87.8%** | **0.941** | 15 | 1.759 | 26.7% |
| Best tested joint | 86.7% | 0.933 | 17 | 1.855 | 28.9% |
| Oracle correct in Top-6 | 92.2% | 0.965 | 9 | 1.580 | 26.7% |

`Oracle correct in Top-6` 的 9 个 wrong 来自正确 assignment 根本不在 Top-6 的 7 cuts，及
相应人数差异，不代表 Oracle 主动选择错误。

camera 结果同时受先前 identity-free fallback world drift 影响，只用于比较同一 trajectory，
不能替代 Phase 2 独立 cut benchmark。即便如此，joint scorer 也没有改善 camera。

### 7.4 Best-vs-second risk coverage

在 `three` 上分别选择 zero-wrong margin：

| Method | Zero-wrong multi coverage | Accepted matches |
|---|---:|---:|
| Identity-only | **51.11%** | 136 |
| Joint weight 0.25 | 31.11% | 81 |
| Joint weight 0.5 | 17.78% | 48 |
| Joint weight 1 | 23.33% | 63 |
| Joint weight 2 | 33.33% | 90 |
| Joint weight 4 | **35.56%** | 96 |
| Geometry-only | 5.56% | 15 |

联合 geometry margin 比 identity-only 更不校准。它提高部分平均 assignment，但正确困难样本
和错误高置信样本仍然混在一起。按照 precision-first 原则，不能用更高平均 accuracy 换取错误
identity 进入 shared Boundary。

---

## 8. 停止决定

Phase 5 的 Stage 0/1 通过，但 Stage 3 joint scoring 未通过：

```text
joint zero-wrong coverage < identity-only zero-wrong coverage
joint camera composite > identity-only camera composite
wrong accepted tail 未消除
```

因此停止：

- 不让 joint scorer 参与 state commit；
- 不实现多轮 hypothesis search；
- 不调 geometry 权重；
- 不修改 Uniform Multi-Human Consensus；
- 不在 `dance`、`box`、EgoHumans 上重新选 threshold；
- 不声称 automatic WHO-WHERE 已保留 70% GT-ID 收益。

由于 development scorer 已经不优于 identity-first，按预注册停止条件没有运行 frozen
`dance/box/EgoHumans` joint endpoint。运行这些数据再修改 scorer 会违反 frozen evaluation。

---

## 9. 对预注册问题的回答

1. **Persistent state 是否优于 stateless？**
   是。Candidate IDF1 提升 `+0.1053`，zero-wrong multi coverage 从 `14.44%` 提到 `50%`。

2. **Mean/variance 是否解决固定阈值问题？**
   否。Track-normalized variants 低于普通 running mean。

3. **正确 assignment 是否通常在小 Top-K 中？**
   是，Top-6 recall 为 `92.22%`，但仍有 7 cuts 不可恢复。

4. **Joint WHO-WHERE 是否优于 identity-first？**
   否。Zero-wrong coverage `35.56% < 51.11%`。

5. **Leave-one-human-out 是否有帮助？**
   有平均可分性，但错误尾部重叠，不能安全 commit。

6. **Best-vs-second margin 能否控制风险？**
   Identity margin 可以；当前 joint margin 更差。

7. **保留多少 GT-ID 收益？**
   Joint route 未通过身份安全 gate，因此不报告部署 retention，避免在选择性容易样本上夸大。

8. **进入、离开、漏检、重现是否已验证？**
   数据结构支持 inactive/TTL/tentative track，但 joint route 在 development 已停止，未进行
   EgoHumans `3 -> 1 -> 3` frozen endpoint，不能宣称通过。

9. **是否仍 causal/fixed-budget？**
   已实现部分满足：state 固定大小、Top-K `<=6`、每 hypothesis 一次 Boundary、无 future
   frame、无 BA、无历史修改。失败原因是评分可靠性，不是计算预算。

10. **是否值得作为最终完整贡献？**
    当前不值得把 joint search 作为最终方法。可以保留 shot-persistent state 和 Top-K Oracle
    分析，但部署主线仍应是 Single-Human Movie3R。

---

## 10. 复现

Stage 0/1：

```bash
PYTHONPATH=src:. .venv/bin/python \
  versions/v13/experiments/phase5_causal_identity_state.py \
  --sequence three --role development --mode all --device cuda:4
```

只使用已有 cache 重做分析：

```bash
PYTHONPATH=src:. .venv/bin/python \
  versions/v13/experiments/phase5_causal_identity_state.py \
  --sequence three --role development --mode analyze
```

Stage 2/3 probe：

```bash
PYTHONPATH=src:. .venv/bin/python \
  versions/v13/experiments/phase5_joint_hypothesis_probe.py \
  --sequence three
```

结果：

```text
output/v13/phase5_identity/three/v13_phase5_stage0_state.json
output/v13/phase5_identity/three/v13_phase5_stage0_state.md
output/v13/phase5_joint/three/v13_phase5_joint_probe.json
output/v13/phase5_joint/three/v13_phase5_joint_probe.md
```
