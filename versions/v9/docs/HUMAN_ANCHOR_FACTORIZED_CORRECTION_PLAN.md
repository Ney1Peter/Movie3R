# Human-Anchor Factorized Correction Plan

Date: 2026-07-06

## 背景

当前 v9 baseline 已经证明：在 Human3R 基础上增加 correction branch，可以通过前馈方式同时纠正 camera pose 和 human pose。这个方向是有效的。

但最近一系列 correct token 消融也暴露出一个问题：继续只改变 semantic / alignment / momentum / human token 的组合，收益有限。原因是这些 token 大多仍然来自 Human3R 已有特征，模型需要自己从混合信息里判断：

- 当前偏移应该归因于 camera 还是 human；
- 什么时候应该信人体锚点；
- 什么时候人体在运动，不能强行拉回；
- 什么时候应该保留历史 state；
- 什么时候发生镜头跳变，需要重新对齐。

因此，下一步不应该只继续堆 token，而应该把问题改成更明确的流式因子化纠正。

## 目标

在保持以下约束的前提下提升效果：

1. 单目输入。
2. 前馈推理。
3. 流式处理。
4. 不依赖全局 BA 或离线优化。
5. 输出 corrected camera pose 和 corrected human pose。

方法上的核心故事可以表述为：

> 用人体作为可学习的局部锚点状态，在一次前馈流程中先估计更可靠的人体锚点，再以这个人体锚点为条件修正相机位姿，从而缓解 camera translation 和 human translation 的耦合问题。

## 核心想法

当前 baseline 里，human correction head 和 pose correction head 基本并行地从 correct token 里取信息：

```text
Human3R tokens
  -> correct tokens
  -> human correction head
  -> pose correction head
```

这个设计容易让模型把 camera 和 human 的位移耦合在一起。

下一步希望改成模型内部的顺序推理：

```text
Human3R raw tokens / raw outputs
  -> human-anchor observation
  -> human correction
  -> corrected human anchor
  -> camera correction conditioned on corrected human anchor
  -> corrected camera pose + corrected human pose
```

注意：这不是推理时运行两个独立模型，也不是必须做两次完整 Human3R forward。它可以是在一个模型 forward 里完成两个有顺序的 correction step。

## 模型设计

### 1. Human-Anchor Observation

从 Human3R 中提取与人体稳定性相关的信息，构成 `human_anchor_observation`：

- current human latent；
- previous / memory human latent；
- human latent difference；
- pose token；
- image token；
- recurrent state；
- 可选的 pelvis / torso / hip / feet 等 body-part cue。

这个 observation 不是最终纠正结果，只是告诉模型：当前人体在哪里，和历史人体是否一致，当前人体是否适合作为锚点。

### 2. Human-Anchor State

增加一个轻量的 streaming state：

```text
S_human,t
```

它记录局部时间窗口内的人体锚点状态，例如：

- 稳定的人体 root / pelvis 位置；
- torso 朝向；
- human latent template；
- 当前人体锚点可信度；
- 当前是否可能是镜头跳变；
- 当前是否可能是人体真实运动。

这个 state 应该是流式更新的，只依赖过去和当前，不看未来。

### 3. 第一层 Correction：修 Human

第一层 correction 先输出 human residual：

```text
delta_human
```

作用是把 raw human latent / human translation 修成更稳定的 corrected human anchor。

这里的重点不是提升 SMPL 重建质量，而是修正人体在世界坐标中的位置关系，使其更适合作为后续 camera correction 的锚点。

### 4. 第二层 Correction：基于 Corrected Human 修 Camera

pose correction head 不再只看原始 correct token，而是额外读取 corrected human anchor：

```text
pose correction input = pose/correct token + corrected human anchor + state
```

这样 camera correction 不是盲目修相机，而是基于“人体已经被放到更合理的位置”之后，再判断 camera pose 应该如何修。

这可以减少 camera translation 和 human translation 互相抢监督的问题。

## 训练策略

可以先从轻量分阶段训练开始，不需要马上做真正 two-pass 推理。

### Stage 1: Human Anchor Warmup

目标：先让 human correction 学会把人体放对。

设置：

- human correction loss 权重大；
- camera correction loss 可以关闭或降低；
- pose head / pose correction branch 可以冻结或弱训练；
- 使用相同训练数据。

### Stage 2: Camera Conditioned on Human

目标：让 camera correction 学会基于 corrected human anchor 修相机。

设置：

- 使用 Stage 1 训练出的 human correction；
- pose correction head 读取 corrected human anchor；
- 可选：对 corrected human anchor 使用 stop-gradient，避免 camera loss 把 human branch 拉乱；
- camera pose loss 权重恢复正常。

### Stage 3: Joint Fine-tune

目标：让 human correction 和 camera correction 最后适配到一起。

设置：

- 小学习率；
- human loss + camera loss 联合训练；
- 保留 gate / confidence 监督；
- 观察是否过拟合或 human 被 camera loss 拉偏。

## 推理形式

最终推理仍然应该是前馈流式：

```text
input frames + previous state
  -> Human3R backbone
  -> human-anchor state update
  -> human correction
  -> camera correction conditioned on corrected human
  -> output corrected human + corrected camera + next state
```

不使用全局 BA，不需要把整段视频全部放进优化器，也不需要未来帧。

## 可能的实验顺序

等当前 `human_anchor_single` 和 `human_anchor_multi` 两个小实验结束后，可以按下面顺序做：

1. 保持 baseline token 结构，只改训练策略为 Stage 1 / Stage 2 / Stage 3。
2. 增加 corrected human anchor 到 pose correction head 输入。
3. 测试 Stage 2 中 corrected human anchor 是否需要 stop-gradient。
4. 再增加 human-anchor state。
5. 最后考虑是否做 teacher two-pass，再蒸馏回 single-pass。

## 评估重点

除了总 loss，还要重点观察：

- camera translation error；
- camera rotation error；
- human translation error；
- corrected human 和 GT human 的相对高度/深度；
- aaaa 下是否过度纠正；
- aabb / abab / abba 下是否能处理跳变；
- 12 帧或更长序列下 state 是否漂移；
- 对未见数据的泛化是否比 baseline 更稳定。

## 当前结论

这个方向的价值不在于“又加一个 token”，而在于把任务拆成更清楚的因子化流程：

1. 先建立人体锚点。
2. 再基于人体锚点修相机。
3. 用流式 state 维护局部一致性。

如果实验成功，它比单纯 correct token 消融更容易形成论文里的方法贡献：feed-forward streaming human-anchor correction without global BA.
