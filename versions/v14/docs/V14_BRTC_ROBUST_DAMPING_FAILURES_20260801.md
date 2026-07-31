# V14 BRTC robust estimator 与固定阻尼实验记录

> 日期：2026-08-01。本文记录两个没有晋级的严格在线候选，防止后续重复试验。
> 两者都冻结 B0 camera，只改 matched post person 的刚性平移；不引入额外预训练模型，
> 不使用未来帧，GT 只用于预测后的评测。

## 1. 结论

固定 action damping 与 Huber-IRLS 都没有稳定支配冻结的 BRTC-LC v1：

- 固定 `scale=0.8` 在 confirmation 上改善 root、vertex、pair-vector 和伤害率，但
  pair-distance 从 `0.09835 m` 退到 `0.10131 m`，joint 只改善 `0.00041 m`；
- 按预先写定的“所有五项主指标、coverage、harm 都不差于 v1”规则，开发集最后仍选出
  `scale=1.0`，即没有产生新方法；
- Huber-IRLS 在 `three offset1` 的 root/joint/vertex 有小幅收益，但 pair-distance 从
  `0.09835 m` 退到 `0.10076 m`，coverage 从 `88.0%` 降到 `86.4%`；
- Huber-IRLS 在 dance+box 上同样恶化 pair-distance 和 pair-vector，未超过独立提供的
  fixed-0.8 强比较器。

因此两条路线均标记为 `NO-GO`。fixed-0.8 可作为 harm-oriented ablation，不能替代主线。

## 2. 固定 action damping

### 2.1 方法

对 BRTC-LC gate 已接受的标量 ray action 乘同一个常数：

```text
raw action -> frozen gate/cap -> scale * action
           -> group median + layout-consensus residual
           -> post root/joints/vertices rigid shift
```

扫描 `0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00, 1.10`。
selection 只看 `three offset0`；要求 root、joint、vertex、pair-distance、pair-vector、harm 和
coverage 全部不差于 v1，再选择 root 最低者。冻结后才读取 `three offset1` 和 dance+box。

### 2.2 结果

| Split | Scale | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| three offset0 | 1.0 | 0.2251 | 0.2704 | 0.2486 | 0.1022 | 0.2605 | 6.6% |
| three offset1 | 1.0 | 0.2314 | 0.2745 | 0.2525 | 0.0984 | 0.2588 | 7.2% |
| three offset1 | 0.8 | 0.2291 | 0.2741 | 0.2504 | 0.1013 | 0.2583 | 5.6% |
| dance+box | 1.0 | 0.2639 | 0.3146 | 0.3109 | 0.0548 | 0.2742 | 5.0% |
| dance+box | 0.8 | 0.2452 | 0.2953 | 0.2892 | 0.0532 | 0.2725 | 3.6% |

`scale=0.8` 的跨数据行为说明 BRTC 的 action 有时偏大，但一个全局常数不能识别何时应该
缩小。下一步必须使用因果可观察的可靠性或集合完整度来决定缩放，而不是继续扫固定常数。

## 3. Huber-IRLS 多射线 robust estimator

### 3.1 方法

旧版先取 torso-5 个 joint ray proposal 的 median。候选改为对 proposal residual 做
Huber-IRLS，并扫描 robust cutoff、最小 inlier 数等策略；政策只在 development 上选择并在
confirmation 前写入冻结文件。输入、camera freeze、no-op fallback 与 BRTC-LC 一致。

### 3.2 结果

| Split | Method | Coverage | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| offset1 | BRTC-LC | 88.0% | 0.23144 | 0.27449 | 0.25245 | 0.09835 | 0.25878 | 7.2% |
| offset1 | Huber-IRLS | 86.4% | 0.22891 | 0.27347 | 0.25107 | 0.10076 | 0.25861 | 7.2% |
| dance+box | BRTC-LC | 98.9% | 0.26386 | 0.31462 | 0.31094 | 0.05476 | 0.27417 | 5.0% |
| dance+box | Huber-IRLS | 98.9% | 0.25943 | 0.30996 | 0.30628 | 0.05958 | 0.27901 | 3.2% |

Huber-IRLS 能减少部分单人深度 outlier，但会改变多人之间的相对 correction；这解释了
单人空间误差略好、两种 layout 指标却不稳。当前没有证据支持替换 v1 的 median estimator。

## 4. 可复现文件

```text
versions/v14/b0_person_triangulation_damped.py
versions/v14/probe_brtc_action_damping.py
versions/v14/tests/test_b0_person_triangulation_damped.py
versions/v14/b0_brtc_huber_irls.py
versions/v14/probe_b0_brtc_huber_irls.py
output/v14/fine_alignment_research/brtc_action_damping/
output/v14/fine_alignment_research/b0_brtc_huber_irls/
```

运行：

```bash
.venv/bin/python versions/v14/probe_brtc_action_damping.py
.venv/bin/python versions/v14/probe_b0_brtc_huber_irls.py --phase dev
.venv/bin/python versions/v14/probe_b0_brtc_huber_irls.py --phase freeze
.venv/bin/python versions/v14/probe_b0_brtc_huber_irls.py --phase confirm
pytest -q versions/v14/tests/test_b0_person_triangulation_damped.py
```
