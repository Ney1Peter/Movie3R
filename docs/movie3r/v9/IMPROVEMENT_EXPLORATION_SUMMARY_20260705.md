# V9 改进探索总结

日期：2026-07-05

这份文档用于简单记录最近围绕 V9 baseline 做过的模型和 loss 改进探索。重点不是完整复现实验，而是说明：我们为什么改、改了什么、效果如何、下一步建议是什么。

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

小 probe 最终结果大致为：

| Variant | AABB avg loss ↓ | AAAA avg loss ↓ | AABB pose loss ↓ | AABB human trans ↓ |
|---|---:|---:|---:|---:|
| baseline | 0.0813 | 0.0154 | 0.0451 | 0.000274 |

结论：baseline 目前是最稳的默认版本。

## 2. Correct Token 构造消融

我们尝试过去掉或改动 correct token 里的不同信息来源，例如：

- 去掉 semantic token。
- 去掉 alignment token。
- 去掉 momentum token。
- 把多个 token 提前压成 single token。
- 使用 mean pooling、concat MLP、learnable pooling 等不同汇聚方式。

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

| Variant | AABB avg ↓ | AAAA avg ↓ | AABB pose ↓ | AABB human ↓ | 观察 |
|---|---:|---:|---:|---:|---|
| baseline | 0.0813 | 0.0154 | 0.0451 | 0.000274 | 最稳 |
| body_part_shared_w2 | 0.1010 | 0.0325 | 0.0138 | 0.000659 | pose 变好，但整体变差 |
| body_part_human_w2 | 0.0898 | 0.0350 | 0.0119 | 0.000384 | body-part 里较好，但 AAAA 伤得明显 |
| body_part_human_w0.5 | 0.0928 | 0.0353 | 0.0148 | 0.000271 | 降权后没有解决 AAAA 问题 |
| body_part_aux_only | 0.1002 | 0.0311 | 0.0125 | 0.000683 | 辅助-only 也没有超过 baseline |

关键现象：

- body-part cue 对 AABB 的 pose correction 很有效，AABB pose loss 明显下降。
- 但 AAAA 连续序列 loss 明显变差，说明模型在不该修的时候也在修。
- 降低 body-part loss 权重或让 body-part 只做 auxiliary，并没有根本解决过度纠正问题。

结论：body-part cue 是有用的，但需要更强的“什么时候使用”的控制。

## 5. Loss / Gate 方向

目前最大问题不是模型没有纠正能力，而是纠正力度和触发条件不够稳。

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
