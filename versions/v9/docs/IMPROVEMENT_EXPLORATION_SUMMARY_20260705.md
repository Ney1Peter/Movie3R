# V9 改进探索总结

日期：2026-07-05

这份文档用于简单记录最近围绕 V9 baseline 做过的模型和 loss 改进探索。重点不是完整复现实验，而是说明：我们为什么改、改了什么、效果如何、下一步建议是什么。

## 0. 指标口径

本文的指标优先使用已经做过的较大训练结果；如果某个消融只做过小规模 probe，就使用小 probe 指标。不同规模的数据分布和样本量不同，所以不要把“大训练”和“小 probe”的绝对 loss 直接横向比较，主要看同一张表内的相对变化。

主要指标含义：

- `AABB avg loss`：跨视角 / 跨镜头跳变序列上的平均 correction loss，越低越好。
- `AAAA avg loss`：连续同视角稳定序列上的平均 loss，越低说明越不容易误修。
- `AABB pose loss`：AABB 上 camera pose correction 的 loss，越低越好。
- `AABB human trans loss`：AABB 上人体 translation correction 的 loss，越低越好。
- `Cam Trans / Cam Rot`：旧 benchmark 中的 camera 平移 / 旋转误差，越低越好。

## 1. 当前 Baseline

当前比较稳定的 baseline 是 V9 的标准 human-pose correction 版本：

- correct token 由三类信息构成：semantic、alignment、momentum。
- 三类 correct token 一起进入 decoder。
- decoder 输出后，对 correct tokens 做 mean pooling。
- pooled correct feature 分别给 pose correction head 和 human latent correction head 使用。
- pose 分支修正 camera pose latent。
- human 分支修正 human latent / SMPL translation 相关信息。
- 使用 pose head LoRA 和 human head LoRA。
- 训练监督包含 camera pose correction 和 human translation correction。

这个版本的特点是比较稳。它不一定在每一个 AABB case 上纠正最强，但 AAAA 连续序列不容易被过度纠正，因此整体作为 baseline 是合理的。

已有指标：

| Variant | 规模 | Epoch | AABB avg ↓ | AAAA avg ↓ | AABB pose ↓ | AABB human ↓ | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline all-mean | 4source 大训练，running latest | 143 | 1.3738 | 0.4377 | 1.3532 | 5.42e-04 | 正式 baseline 大训练还在跑，使用 2026-07-06 最新日志 |
| baseline all-mean | 4source 小 probe | 63 | 0.0813 | 0.0154 | 0.0451 | 2.74e-04 | 用于和 token / body-part 小消融同口径比较 |

结论：baseline 目前是最稳的默认版本。

## 2. Correct Token 构造消融

我们尝试过去掉或改动 correct token 里的不同信息来源，例如：

- 去掉 semantic token。
- 去掉 alignment token。
- 去掉 momentum token。
- 把多个 token 提前压成 single token。
- 使用 mean pooling、concat MLP、learnable pooling 等不同汇聚方式。

### 2.1 三个 token 是否都需要

4source 小 probe 指标：

| Variant | 规模 | Epoch | AABB avg ↓ | AAAA avg ↓ | AABB pose ↓ | AABB human ↓ | 观察 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline all-mean | 4source 小 probe | 63 | 0.0813 | 0.0154 | 0.0451 | 2.74e-04 | 当前小 probe baseline，稳定性最好 |
| no semantic | 4source 小 probe | 60 | 0.1071 | 0.0343 | 0.0185 | 0.0013 | pose 变低，但整体和 human 明显变差 |
| no alignment | 4source 小 probe | 60 | 0.1021 | 0.0385 | 0.0185 | 0.0011 | 去掉对齐信息后 AAAA 误修更明显 |
| no momentum | 4source 小 probe | 60 | 0.0924 | 0.0305 | 0.0120 | 7.57e-04 | AABB pose 低，但整体稳定性不如 baseline |

早期 token 模块消融旧 benchmark 指标：

| Variant | Cam Trans ↓ | Cam Rot ↓ | 观察 |
|---|---:|---:|---|
| all_mean | 0.1543 m | 3.86 deg | 完整 baseline |
| no semantic | 0.5447 m | 7.51 deg | 明显变差 |
| no alignment | 0.1215 m | 2.64 deg | 单例旋转好，但不能证明全局最好 |
| no momentum | 0.0968 m | 4.45 deg | 单例平移好，但旋转差 |
| single_token | 0.1758 m | 5.16 deg | 信息压缩太早 |
| learned_pooling | 0.4015 m | 6.08 deg | 小数据下不稳定 |

### 2.2 pooling / contact-style 融合方式

较大 4source 训练中，目前只对 `pose concat-MLP + human mean` 做了正式对照，训练仍在进行中：

| Variant | 规模 | Epoch | AABB avg ↓ | AAAA avg ↓ | AABB pose ↓ | AABB human ↓ | 观察 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline all-mean | 4source 大训练，running latest | 143 | 1.3738 | 0.4377 | 1.3532 | 5.42e-04 | 稳定下降中 |
| pose concat-MLP + human mean | 4source 大训练，running latest | 132 | 1.2377 | 0.6100 | 1.2204 | 5.72e-04 | AABB 略低，但 AAAA 更差，当前还不能说明优于 baseline |

早期 `benchmark_mixed_small18` pooling 指标：

| Variant | AABB cam 降低量 ↑ | AABB gain ↑ | AABB human 降低量 ↑ | AAAA gate ↓ | Loss ↓ |
|---|---:|---:|---:|---:|---:|
| global_weighted | 0.177 m | 0.177 m | 0.151 m | 0.147 | 1.412 |
| all_concat / contact-style | 0.213 m | 0.213 m | 0.181 m | 0.055 | 0.892 |

总体观察：

- 三类 token 都有价值，不建议简单删掉。
- semantic 提供当前帧和历史上下文的语义信息，去掉后模型更难判断当前画面和过去是否属于同一状态。
- alignment 直接描述当前 pose / human 与 memory 之间的差异，对判断错位很重要。
- momentum 提供上一帧纠正状态，让模型不要每一帧都从零开始判断。
- single token 或 learned pooling 在小数据上没有稳定超过 baseline，容易因为数据太少学出不稳定权重。

结论：当前三 token 结构是合理的，不需要马上大改。

## 3. Human Anchor / Human Alignment 方向

因为我们的任务里人体很重要，所以尝试过让 correct token 更偏向人体信息，例如：

- human-only token。
- human + semantic。
- semantic + alignment + human。
- human alignment token。
- 单序列过拟合测试。

4source 小 probe 指标：

| Variant | 规模 | Epoch | AABB avg ↓ | AAAA avg ↓ | AABB pose ↓ | AABB human ↓ | 观察 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline all-mean | 4source 小 probe | 63 | 0.0813 | 0.0154 | 0.0451 | 2.74e-04 | 最稳 |
| human-only | 4source 小 probe | 60 | 0.1113 | 0.0405 | 0.0194 | 0.0018 | 只看人体不稳定，容易误修 |
| human + semantic | 4source 小 probe | 60 | 0.1026 | 0.0307 | 0.0180 | 7.00e-04 | 比 human-only 好，但仍不如 baseline |
| semantic + alignment + human | 4source 小 probe | 60 | 0.1067 | 0.0251 | 0.0142 | 7.71e-04 | 主观看起来不错，但客观整体没有超过 baseline |
| human alignment | 4source 小 probe | 60 | 0.1001 | 0.0329 | 0.0137 | 9.33e-04 | AABB pose 好，但 human / AAAA 变差 |
| human pose alignment | 4source 小 probe | 60 | 0.1090 | 0.0284 | 0.0165 | 0.0010 | 仍没有稳定超过 baseline |

单序列过拟合指标：

| Variant | 规模 | Epoch | Avg loss ↓ | Pose loss ↓ | Human trans ↓ | Drift loss ↓ | 观察 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline all-mean | single-seq overfit | 120 | 0.0529 | 0.0058 | 2.83e-06 | 0.0657 | 能拟合，但不是最低 |
| pose concat-MLP + human mean | single-seq overfit | 120 | 0.0246 | 0.0049 | 2.82e-06 | 0.0330 | 单序列拟合更强 |
| human-only | single-seq overfit | 120 | 0.0447 | 0.0061 | 1.87e-06 | 0.2020 | 能拟合，但 drift/gate 不稳 |
| semantic + alignment + human | single-seq overfit | 120 | 0.0553 | 0.0066 | 2.61e-06 | 0.1465 | 单序列也没有明显优势 |
| pelvis / hip / feet anchor | single-seq overfit | 120 | 0.0364 | 0.0049 | 2.04e-06 | 0.0296 | 单序列上有效，但多 source 不稳定 |
| human ref pairwise | single-seq overfit | 120 | 0.0267 | 0.0049 | 2.25e-06 | 0.1076 | 局部拟合好，但泛化风险待验证 |

观察：

- 在单个训练序列上，这些方法都能拟合，说明模型确实可以利用人体信息做纠正。
- 但在小规模多 source probe 上，这些方法没有稳定超过 baseline。
- pure human 容易把“人体位置一致”当成唯一目标，如果人真的在运动，可能会错误地把人拉回去。

结论：人体是有效锚点，但不能只靠人体。更合理的是让人体 cue 作为辅助信号，并且由 gate 控制强弱。

## 4. Body-Part Cue 方向

后来进一步尝试显式引入人体部位 cue，主要关注 pelvis、hip、feet 等更稳定的人体部位。

做过的版本包括：

- `body_part_shared_w2`：body-part token 参与整体 correction。
- `body_part_human_w2`：body-part token 主要影响 human branch，body-part residual 权重为 2.0。
- `body_part_human_w0.5`：同上，但把 body-part residual 权重降到 0.5。
- `body_part_aux_only`：body-part token 进入 decoder，但 pose/human pooling 忽略它，只让 body-part residual head 使用它。

小 probe 结果：

| Variant | 规模 | Epoch | AABB avg ↓ | AAAA avg ↓ | AABB pose ↓ | AABB human ↓ | 观察 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | 4source 小 probe | 63 | 0.0813 | 0.0154 | 0.0451 | 2.74e-04 | 最稳 |
| body_part_shared_w2 | 4source 小 probe | 60 | 0.1010 | 0.0325 | 0.0138 | 6.59e-04 | pose 变好，但整体变差 |
| body_part_human_w2 | 4source 小 probe | 60 | 0.0898 | 0.0350 | 0.0119 | 3.84e-04 | body-part 里较好，但 AAAA 伤得明显 |
| body_part_human_w0.5 | 4source 小 probe | 60 | 0.0928 | 0.0353 | 0.0148 | 2.71e-04 | 降权后没有解决 AAAA 问题 |
| body_part_aux_only | 4source 小 probe | 60 | 0.1002 | 0.0311 | 0.0125 | 6.83e-04 | 辅助-only 也没有超过 baseline |
| pelvis / hip / feet anchor | 4source 小 probe | 60 | 0.1014 | 0.0423 | 0.0125 | 6.74e-04 | 单序列指标好，多 source 上过度纠正更明显 |

MVHuman 单序列 body-part 过拟合指标：

| Variant | 规模 | Epoch | Avg loss ↓ | Pose loss ↓ | Human trans ↓ | Drift loss ↓ | 观察 |
|---|---|---:|---:|---:|---:|---:|---|
| body_part_human_only | single-seq overfit | 30 | 0.0830 | 0.0066 | 1.77e-06 | 0.6469 | 训练不够充分，drift/gate 明显不稳 |
| body_part_residual | single-seq overfit | 120 | 0.0280 | 0.0050 | 2.29e-06 | 0.0271 | 单序列可以拟合得很好 |

关键现象：

- body-part cue 对 AABB 的 pose correction 很有效，AABB pose loss 明显下降。
- 但 AAAA 连续序列 loss 明显变差，说明模型在不该修的时候也在修。
- 降低 body-part loss 权重或让 body-part 只做 auxiliary，并没有根本解决过度纠正问题。

结论：body-part cue 是有用的，但需要更强的“什么时候使用”的控制。

## 5. Loss / Gate 方向

目前最大问题不是模型没有纠正能力，而是纠正力度和触发条件不够稳。

已有指标侧面说明了这个问题：

| 实验 | 规模 | 指标现象 | 说明 |
|---|---|---|---|
| baseline 小 probe | 4source 小 probe | AABB `0.0813`，AAAA `0.0154` | 整体最稳，连续序列不容易被误修 |
| body_part_human_w2 | 4source 小 probe | AABB pose `0.0119`，但 AAAA `0.0350` | 修正能力增强，但不该修时也会修 |
| body_part_aux_only | 4source 小 probe | AABB pose `0.0125`，AAAA `0.0311` | 即使 body-part 只辅助，也仍会带来过度纠正 |
| all_concat / contact-style | 旧 small18 benchmark | AAAA gate `0.055`，loss `0.892` | 早期指标显示 gate 更克制，但大训练仍需确认 |
| pose concat-MLP + human mean | 4source 大训练，running latest | AABB `1.2377`，AAAA `0.6100` | 大训练当前 AABB 略好，但 AAAA 更差 |

已有观察：

- AABB 上需要模型大胆修。
- AAAA 上模型应该尽量少修。
- body-part cue 会增强修正能力，但也会增加过度纠正风险。
- shared gate 对 pose / human / body-part 的控制还不够细。

因此，后续如果继续改，重点应该是 gate 和 no-correction 约束，而不是继续盲目增加更多 token。

## 6. 当前建议

短期建议：

1. 继续把 baseline 作为默认稳定版本。
2. 保留 `body_part_human_w2` 作为一个有潜力的分支。
3. 不建议直接用原始 `body_part_human_w2` 开大规模训练，因为它虽然 AABB pose loss 好，但 AAAA 明显变差。
4. 下一步更值得做的是 `body_part_human_w2 + body-part-specific gate`。

具体改进方向：

- 给 body-part cue 单独加 gate。
- 对 AAAA 加强 no-correction / small-residual 约束。
- 让 body-part loss 根据 raw drift 大小动态加权：偏移大时加强，偏移小时减弱。
- 避免所有帧都强行使用人体部位 cue。

大规模训练建议：

- 如果现在要训大规模，优先训 baseline 或 `w2 + gate`，不要直接训原始 `w2`。
- 原始 `w2` 可以作为对照组，但不适合作为最终主线。
- 更合理的节奏是先做 50-60h 中等规模验证，看泛化和主观效果，再决定是否扩大。

## 7. 一句话总结

这段探索证明了：人体信息，尤其是 body-part cue，确实能增强跨镜头纠正能力；但当前版本的问题是容易过度纠正连续序列。所以下一步重点不是继续增加人体信息，而是让模型学会“什么时候该相信人体 cue，什么时候应该保持不动”。
