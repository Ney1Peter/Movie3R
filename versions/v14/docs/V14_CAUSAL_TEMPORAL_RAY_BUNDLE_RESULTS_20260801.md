# V14 严格因果 Temporal Ray Bundle 实验报告

> 日期：2026-08-01。候选只读取已到达帧，不使用未来 post frame、不引入额外模型，
> camera 始终为 bit-exact frozen B0。本实验只复用已有 cache，没有重跑 Human3R。

## 1. 问题与结论

BRTC-LC 已经证明，冻结 B0 后用 last-pre / first-post 的人体关节射线做逐人深度修正，能够
显著改善跨 shot 的人体 root、joint、vertex 和多人 layout。不过，如果每个 post frame 都
独立求一次修正，射线噪声会使修正量随时间抖动，可能恶化 Multi-THuMBS 关心的 Accel。

本实验验证了一个可复现规律：

```text
多个已见 pre frame 的 ray proposal 做 robust 聚合
+ 对按时间到达的 post correction 做 causal EMA
= 保留 BRTC-LC 大部分空间收益，同时降低 correction jitter 和 Accel
```

这个规律在开发集和冻结后的确认集都成立，但当前版本还不能单独作为最终主线：box 上它
虽然比逐帧 BRTC-LC 更平滑，却仍没有在 joint/vertex Accel 上超过完全不做人层修正的 B0。
因此它应作为 BRTC-LC 后面的**因果稳定器**，而不是替代更准确的单帧显式深度证据。

## 2. 严格因果输入、模块和输出

### 2.1 输入

对每个 shot boundary，缓存包含：

- 已经到达的 5 个 pre frame；
- post 当前时刻 `k`，本实验严格按 `k=0 -> 1 -> 2` 到达；
- 每帧 Human3R 人体 root、127 joints、10475 vertices；
- 冻结 B0 对当前 post frame 的 shared-world Boundary；
- B0 root+torso+joints Hungarian 自动身份关联；
- BRTC-LC 已冻结的 ray-gap、parallax、MAD 和 action-cap gate。

候选预测阶段不读取 GT。GT camera、root、joint、vertex 和 identity correctness 只在预测完成
后用于 evaluator。

### 2.2 处理流程

```text
seen pre frames + current post frame
        |
        v
frozen B0：把当前 post camera/humans 放进 pre shared gauge
        |       camera 从这里开始永久冻结
        v
automatic WHO association
        |
        v
每个人、每个已选 pre frame：
  torso-5 joint rays 与当前 post joint rays 两两求最近点
  -> 得到当前人体沿 post pelvis ray 的 depth proposal
        |
        v
ray-gap / parallax / MAD / cap gate
        |
        v
最近 3 个 pre frame 的有效 proposal 做 Huber 聚合
        |
        v
多人 group shift + 可观察 pre-layout 选择 individual residual strength
        |
        v
严格因果 EMA：shift_k = 0.75 proposal_k + 0.25 shift_{k-1}
        |
        v
把同一个 rigid translation 加到当前人的 root/joints/vertices
camera 不变；不回写历史输出；状态递推到 k+1
```

输出是当前 post frame 中每个人修正后的 root、joints、vertices，以及用于下一时刻的一个
person-local correction state。候选不改变 pose、shape、camera 或 scene pointmap。

### 2.3 严格在线约束

- `k=0` 只看全部 pre history 与 post `k=0`；
- `k=1` 只增加 post `k=1`，不能读取 `k=2`；
- `k=2` 才能读取 `k=2`；
- EMA 每到一帧立即 commit，不做双向 smoothing；
- camera candidate max absolute change 必须为 0；
- GT 不参与 proposal、gate、policy selection 或 identity matching。

## 3. 开发、冻结与确认协议

### 3.1 数据可用性

`three offset0/offset1` 的已保存正式报告主要是单 boundary frame，不足以构成一条连续的
三帧轨迹。底层 `dance/box` cache 则保留了同一 boundary 的 `k=0,1,2,4,8`，其中
`k=0,1,2` 是真实连续 frame triple，每个时刻都是只用当前 post frame 的独立 causal reset。

本实验使用：

- development：`dance`，12 streams、36 frames、72 person-frames、24 identity triples；
- confirmation：`box`，15 streams、45 frames、90 person-frames、30 identity triples。

两者自动关联准确率均为 100%。但 dance/box 历史上已经被 V14 其他 post-hoc 分析看过，
所以这里是“对本候选程序严格先冻结、再确认”，不能包装成从未暴露的论文级 holdout。

### 3.2 开发扫描

只在 dance 扫描 108 个确定性策略组合：

- pre history：1 / 3 / 5；
- pre robust aggregation：median / Huber；
- post filter：causal median / causal EMA；
- post state length：1 / 2 / 3；
- EMA alpha：0.50 / 0.75；
- 当前证据失败时：回退零修正 / carry 历史状态。

预先固定的 selection rule 是：相对逐帧 BRTC-LC，root/joint/vertex 均最多退化 3%，layout
最多退化 5%，coverage 最多下降 5 个百分点，`>5 cm` harm 最多增加 2 个百分点；满足后按
joint Accel、root、layout 依次最小化。

选出的策略为：

```json
{
  "pre_history": 3,
  "pre_aggregate": "huber",
  "post_filter": "ema",
  "post_history": 1,
  "ema_alpha": 0.75,
  "carry_on_reject": false
}
```

冻结文件在 box 确认前先落盘：

```text
DEV_SCAN.json                         2026-08-01 03:03:08 +0800
FROZEN_POLICY_BEFORE_CONFIRM.json     2026-08-01 03:05:02 +0800
CONFIRM_RESULTS.json                  2026-08-01 03:05:12 +0800
```

确认结果生成后没有回头改策略。

## 4. 指标定义

空间指标都在同一个固定 pre-shot world gauge 中计算：

- root：pelvis/root 的未对齐世界坐标误差；
- joint：127 joints 的未对齐世界坐标平均误差；
- vertex：10475 SMPL-X vertices 的未对齐世界坐标平均误差；
- layout distance/vector：同帧人物两两 root 距离/向量误差；
- coverage：当前方法实际应用或持有有效 correction 的 person-frame 比例；
- harm：相对 B0 root 误差增加超过 5 cm 的比例。

Accel 只在真实连续 `k=0,1,2` 且同一 identity 三帧均存在时计算：

```text
delta2 X_1 = X_2 - 2 X_1 + X_0
Accel error = mean_j ||delta2 X_pred,j - delta2 X_GT,j||
```

同时报告 root、joint、vertex Accel，以及 correction shift 自身的二阶差分。单位为
`mm/frame^2`。这不是 Multi-THuMBS 已确认的官方口径；论文主文没有公开 Accel 的坐标系、
fps、单位或 missing-frame aggregation。

## 5. Development 结果：dance

| Method | Root | Joint | Vertex | Layout vector | Root Accel | Joint Accel | Shift jitter | Coverage | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 382.1 | 387.9 | 380.1 | 101.2 | 26.58 | 66.43 | - | - | - |
| BRTC-LC | 121.6 | 173.7 | 148.7 | 75.7 | 21.31 | 63.98 | 27.72 | 100% | 2.78% |
| Causal bundle | **120.7** | **172.9** | **147.9** | **75.4** | **19.73** | **62.20** | **17.21** | **100%** | **2.78%** |

距离单位为 mm；Accel/jitter 为 mm/frame²。相对逐帧 BRTC-LC，冻结候选在 development：

- root 改善 0.76%；
- joint 改善 0.42%；
- vertex 改善 0.54%；
- joint Accel 改善 2.79%；
- root Accel 改善 7.41%；
- correction jitter 改善 37.93%；
- coverage/harm 不变。

## 6. 冻结确认结果：box

| Method | Root | Joint | Vertex | Layout distance | Layout vector | Root Accel | Joint Accel | Vertex Accel | Shift jitter | Coverage | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 528.2 | 554.3 | 553.8 | 59.2 | 432.9 | 53.23 | **81.37** | **62.90** | - | - | - |
| BRTC-LC | **337.8** | **381.3** | **388.3** | 55.8 | 390.2 | 54.41 | 86.99 | 66.05 | 58.55 | 100% | **6.67%** |
| Causal bundle | 340.1 | 382.6 | 390.5 | **54.9** | **387.9** | **51.24** | 84.11 | 63.86 | **47.07** | 100% | 7.78% |

相对逐帧 BRTC-LC，冻结候选：

- root 退化 0.68%，joint 退化 0.34%，vertex 退化 0.56%；
- layout distance 改善 1.71%，layout vector 改善 0.59%；
- root Accel 改善 5.83%，joint Accel 改善 3.32%，vertex Accel 改善 3.31%；
- correction jitter 改善 19.60%；
- coverage 保持 100%，harm 增加 1.11 个百分点；
- camera max change 为 `0.0`。

它满足开发前冻结的所有空间/coverage/harm 容忍条件，并在未调参 box 上复现了 Accel 与
jitter 改善，说明规律是真的，不是 dance 单序列偶然。

## 7. 失败点与核心判断

box 上逐帧 BRTC-LC 相对 B0：

```text
root:        528.2 -> 337.8 mm    大幅改善
joint:       554.3 -> 381.3 mm    大幅改善
vertex:      553.8 -> 388.3 mm    大幅改善
joint Accel:  81.37 -> 86.99      反而恶化 6.91%
```

causal bundle 把 joint Accel 救回到 `84.11`，但仍比 B0 的 `81.37` 差 3.37%。vertex Accel
也有同样现象。因此当前明确答案是：

1. BRTC-LC 的逐帧空间 correction 有高价值；
2. correction 随时间抖动是明确问题；
3. robust pre-ray bundle + causal EMA 能稳定缓解，却不能完全消除；
4. 只靠平滑 correction 无法同时拿到最优 position 和 Accel；更准确、低方差的单帧显式
   person-depth observation 才是下一步主线；
5. BRTC-LC 最合理的部署方式仍是 boundary 时估一次可靠 person-local offset，然后在同一
   shot 内持有该 offset，而不是每帧独立重估。需要新的连续长 post cache 验证这种
   boundary-commit 策略。

## 8. 下一步最小实验

### 8.1 Boundary commit，而非逐帧重估

在 `k=0` 用 ray bundle 得到一次 offset，后续同 shot 先保持不变；只有在连续多个当前帧
proposal 形成高置信共识时才小步更新。这会让 correction 的二阶差分接近 0，同时保留 root
收益。当前只有三帧 sparse reset cache，可做 proof，但正式结论需要 15--30 帧连续 post。

### 8.2 引入显式 person-local depth，而非更强平滑

优先加入不会改 camera 的证据：person mask 内高置信 scene depth、mesh-to-pointmap depth、
silhouette/2D reprojection 和 persistent body-size。目标是先降低 proposal 方差，再由因果
state 做保守稳定，而不是用强 EMA 掩盖错误 observation。

### 8.3 Multi-THuMBS 对榜要求

正式对榜需要在 EgoHumans 的同一连续轨迹上保存：每帧 B0 camera、自动 identity、B0 与
BRTC/candidate 的 24-joint/6890-vertex 轨迹、GT visibility 和真实 frame index。作者尚未
公开 supplementary 评测协议，因此当前 Accel 只能作为内部明确公式，不得直接和论文
`27.3` 宣称胜负。

## 9. 可复现产物

```text
versions/v14/probe_causal_temporal_ray_bundle.py
output/v14/fine_alignment_research/causal_temporal_ray_bundle/DEV_SCAN.json
output/v14/fine_alignment_research/causal_temporal_ray_bundle/FROZEN_POLICY_BEFORE_CONFIRM.json
output/v14/fine_alignment_research/causal_temporal_ray_bundle/CONFIRM_RESULTS.json
output/v14/fine_alignment_research/causal_temporal_ray_bundle/RESULTS.md
```

复现命令：

```bash
.venv/bin/python versions/v14/probe_causal_temporal_ray_bundle.py --phase dev
.venv/bin/python versions/v14/probe_causal_temporal_ray_bundle.py --phase freeze
.venv/bin/python versions/v14/probe_causal_temporal_ray_bundle.py --phase confirm
```
