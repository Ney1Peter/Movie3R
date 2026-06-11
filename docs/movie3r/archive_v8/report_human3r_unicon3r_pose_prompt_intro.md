# Human3R / UniCon3R / Pose Correction 汇报简版

## 1. Human3R 的模型流程

可以把 Human3R 的在线推理流程简化成下面三步：

```text
输入视频帧
  -> Encoder / tokenizer
  -> 提取 image token、camera/pose token、human token

image token + camera/pose token + human token + recurrent state token
  -> Decoder 做 attention / cross attention
  -> 当前帧信息和历史 state 融合

Decoder 输出 refined tokens
  -> image token / pose token / human token
  -> 各自进入 head
      - DPT / pointmap head: 输出 depth、pointmap、confidence、scene geometry
      - pose head: 输出 camera pose
      - human head: 输出 SMPL / SMPL-X 参数、人体 mesh / joints / mask
```

更通俗地说：

```text
Human3R 每来一帧，
先把图像变成 token，
再让当前 token 和历史 state 交互，
最后用不同 head 输出场景、相机位姿和人体。
```

当前问题是：当输入视频中发生镜头切换时，虽然场景和时间仍然连续，但 Human3R 可能把切换后的相机 pose 预测到错误位置，导致前后帧的人和场景在同一个 world 里对不上。

## 2. UniCon3R 在 Human3R 基础上的改法

Human3R 原始流程可以抽象为：

```text
video frame
  -> encoder
  -> image / camera / human tokens
  + recurrent state
  -> decoder
  -> image / camera / human / pose tokens
  -> heads
  -> scene + camera pose + human reconstruction
```

UniCon3R 的核心改动是：在这个流程中额外加入 contact prompt / contact token。这个 token 会和 image / camera / human tokens 一起进入 decoder，在 decoder 里面通过 attention 和 human token 交互。decoder 输出 refined contact token 后，再用它预测 contact 和 human latent residual，反过来修正 human reconstruction。

```text
video frame
  -> encoder
  -> image / camera / human tokens

human token + local scene token + geometry cue + history memory
  -> [新增] contact token / contact prompt

image / camera / human tokens + [新增] contact token + recurrent state
  -> decoder
  -> refined image / camera / human / pose tokens + [新增] refined contact token

refined human token + HMR prior
  -> refined human latent

[新增] refined contact token + HMR prior
  -> contact latent

contact latent
  -> contact head
  -> 预测 SMPL mesh 上每个 vertex 的 contact probability

contact latent
  -> residual head
  -> 预测 human latent residual: delta H

refined human latent + delta H
  -> corrected human latent
  -> human head
  -> contact-aware human reconstruction

image / pose tokens
  -> scene / camera heads
  -> scene + camera pose
```

也就是说，UniCon3R 不是完全重做一个模型，而是在 Human3R 这种 foundation backbone 上加一个轻量的 contact-guided prompt / refinement 分支。

这里要避免一个歧义：不是简单把 refined contact token 和 refined human token 拼起来，然后一起过两个 head。更接近论文公式的流程是：decoder 后得到 `H'_t` 和 `C'_t`；`H'_t` concat HMR prior 得到 `H_tilde`，`C'_t` concat HMR prior 得到 `C_tilde`。然后 `C_tilde` 分别进入 contact head 和 residual head；residual head 输出的 `delta H` 加回 `H_tilde`，再交给 human head 输出最终 SMPL / mesh。

更重要的是，contact token 有两个作用，不能只理解成一个 contact 分类输出：

```text
作用 1：decoder 内交互
contact token 和 human token 一起进入 decoder
  -> human token 可以吸收 contact / scene / geometry 信息

作用 2：decoder 后 refinement
refined contact token
  -> contact head: 预测哪里接触
  -> residual head: 预测 delta H
  -> delta H 加回 refined human latent
  -> human head 输出更合理的 SMPL / mesh
```

它的核心思想是：

```text
人体和场景应该有合理接触关系。
如果人漂浮、穿地、脚和地面关系不合理，
就用 contact token 作为额外提示去修正人体重建。
```

## 3. Human3R Decoder 前后可用的信息

设计 pose correction token 时，必须先分清楚哪些信息在当前帧 decoder 前能拿到，哪些信息必须等当前帧 decoder/head 输出后才有。

一个容易混淆的点是：**当前帧 pointmap 不是 encoder 后直接生成的**。encoder 只输出 image tokens；当前帧 pointmap / confidence / camera pose / SMPL 参数都要经过 decoder 和对应 head 后才输出。

但在线模型可以使用上一帧已经输出过的结果，因为这些不破坏 causal / streaming 设置。这里的上一帧 pointmap 是上一帧 decoder tokens 经过 scene / DPT head 后得到的最终显式结果；同理，上一帧 SMPL / joints / body anchors 也是上一帧 human head 输出后可以缓存的显式结果。

| 信息 | 当前帧 decoder 前能不能拿 | 类型 | 是否适合做 pose correction token |
|---|---:|---|---|
| 当前 image tokens | 能 | 隐式 token | 适合，表示当前帧视觉上下文 |
| 当前 human token / smpl query | 能 | 隐式 token | 很重要，表示当前检测到的人 |
| 当前 human detection score / location | 能 | 显式/半显式 | 适合，做人体 anchor 和可靠性判断 |
| 当前 pose token | 能 | 隐式 token | 适合，代表模型当前准备估计 pose 的 query |
| recurrent state token | 能 | 隐式 token | 适合，代表历史 scene / temporal memory |
| pose memory `mem` | 能 | 隐式 token | 适合，代表历史 pose / motion memory |
| 上一帧 raw / corrected pose | 能，需要自己缓存 | 显式 history | 很重要，提供相机运动先验 |
| 上一帧 pointmap / confidence | 能，需要自己缓存 | 上一帧 scene head 输出 | 适合，可构造 local geometry cue |
| 上一帧 SMPL / joints / body anchors | 能，需要自己缓存 | 上一帧 human head 输出 | 很重要，可构造人体运动和 anchor consistency |
| 当前帧 pointmap / confidence | 不能 | 当前帧 scene head 输出 | decoder/head 后才有，不能用于构造当前帧 decoder-in `A_corr_t` |
| 当前帧 raw camera pose | 不能 | 当前帧 pose head 输出 | pose head 后才有，不能用于构造当前帧 decoder-in `A_corr_t` |
| 当前帧 SMPL joints / mesh | 不能 | 当前帧 human head 输出 | human head 后才有，不能用于构造当前帧 decoder-in `A_corr_t` |

所以如果我们要完全模仿 UniCon3R，把 correction prompt 放进当前帧 decoder 前，第一版应该优先使用：

```text
当前 image / human / pose tokens
+ recurrent state / pose memory
+ 上一帧 pointmap / confidence
+ 上一帧 SMPL / joints / human anchors
+ 上一帧 corrected pose 和 motion history
```

因此当前主线不是 decoder 后修正，而是 UniCon-style decoder-in prompt：`A_corr_t` 必须在当前帧 decoder 前构造，并和 image / pose / human / state tokens 一起进入 decoder。

## 4. UniCon3R 的 Contact Token 怎么设计

这里按功能理解 contact token，而不是逐行复现论文实现。

| Contact token 来源 | 通俗解释 | 提供的信息 |
|---|---|---|
| Human token / human prompt | 先知道当前要分析的是哪个人 | 人体整体位置、人体 latent、human query |
| Body / pose feature | 看人体姿态，尤其是脚、腿、躯干等部位 | 哪些身体部位可能接触场景，人体姿态是否支持 contact |
| Local scene token | 读取人体附近或脚附近的场景 token | 人旁边是什么，脚下是不是地面，附近有没有可接触物体 |
| Local geometry cue | 显式看 3D 几何关系 | 脚到地面的距离、是否穿地、局部平面、human-scene 距离 |
| Recurrent state memory | 读取模型保存的历史场景 / 历史人体信息 | 上一段时间里场景和人的状态，帮助当前帧更稳定 |
| Temporal momentum / previous contact | 使用上一帧的 contact 状态 | 接触关系一般不会突然乱跳，提供时序连续性 |
| Contact confidence / gate | 判断当前 contact cue 是否可靠 | 避免在遮挡、跳跃、检测错误时强行修正 |

可以概括成一句话：

```text
UniCon3R 的 contact token =
当前人 + 人体部位 + 局部场景 + 显式几何 + 历史 contact / memory + 可靠性判断
```

然后这个 contact token 不是只拿来输出一个 contact 结果，而是进入 decoder / correction branch，作为中间提示去修正 human reconstruction。

更细一点看，UniCon3R 最终进入 decoder 的 contact token 不是显式 contact 表，而是一个 latent prompt token。论文里的核心形式可以理解为：

```text
C_t = MLP(H_t + U_scene + G_t + M_t)
```

其中哪些是显式信息、哪些是隐式信息，可以这样拆：

| 组成 | 含义 | 显式/隐式 |
|---|---|---|
| `H_t` | human prompt / human token，表示当前人 | 隐式 token |
| `U_scene` | 当前图像 token 和 recurrent state 读出来的人附近场景上下文 | 隐式 token |
| `G_t` | 从上一帧 world pointmap 采样的人附近局部 3D 坐标 | 显式几何，再转成 token |
| `M_t` | temporal momentum，来自上一帧 refined contact token `C'_{t-1}` | 隐式历史 token |
| `gamma` | 当前 scene cue 和 memory scene cue 的融合 gate | 隐式 gate |
| human anchor `u_t` | 当前人的 2D anchor，用来定位采样窗口 | 显式位置，但主要用于采样，不是最终 token 主体 |

这里最典型的显式信息是 `G_t`。它进入 decoder 前会先被 token 化：

```text
上一帧 world pointmap X_{t-1}
+ 当前人 2D anchor u_t
  -> 在人附近取局部窗口
  -> RoIAlign 得到局部 3D 坐标 patch
  -> pooling 得到一个 R3 geometry descriptor
  -> MLP 映射到 decoder token 维度
  -> geometry token G_t
```

所以 decoder 里看到的不是原始 `(x, y, z)` 坐标，而是已经被 MLP 编码后的 geometry token。这个设计说明：显式 cue 和隐式 token 并不冲突，显式几何可以先转成 token，再和 human token、scene token、history token 融合。

## 5. 我的 Pose Correction 如何参考 Contact Token

我的任务不是修 contact，而是修 camera pose drift。

所以可以把 UniCon3R 的逻辑从：

```text
contact cue -> 修正 human reconstruction
```

改成：

```text
human-centric pose cue -> 修正 camera pose
```

对应关系如下：

| UniCon3R contact token | 我的 pose correction token | 为什么可以对应 |
|---|---|---|
| Human token / human prompt | `A_human` / `A_body_part` | 两者都以人为中心。先确定当前人，再围绕人找稳定线索。 |
| Body / pose feature | pelvis、torso、left foot、right foot token | UniCon3R 用人体部位判断 contact；我用人体稳定部位判断 pose 是否漂移。 |
| Contact token | human anchor history / support-foot cue | contact 关注脚和场景是否接触；pose correction 关注人体 anchor 在 world 里是否突然跳走。 |
| Local scene token | near-human / near-foot scene token | UniCon3R 看人附近场景；我也可以看人体附近背景是否能辅助 pose 对齐。 |
| Local geometry cue | human-scene geometry / floor normal / anchor residual | UniCon3R 用几何判断 contact 合不合理；我用几何判断当前 camera pose 下人体和场景是否自洽。 |
| Recurrent state memory | `A_state_memory` | 两者都可以读取模型内部历史 state，但目前需要验证是否稳定有效。 |
| Temporal momentum | `A_history_human` / previous corrected anchors | 这是目前最强线索：同一个人的 pelvis / torso / feet 在连续时间里不应突然跳走。 |
| Contact confidence / gate | `A_gate` / pose correction gate | 判断当前帧是否需要强修正，避免正常帧过度修正。 |

因此，我的第一版 pose correction prompt 可以先设计成：

```text
A_corr_t =
  current body-part prompt tokens
  + previous human-anchor history
  + previous pose / pose-memory cue
  + correction gate
```

其中最核心的是：

```text
pelvis token
torso token
left foot token
right foot token
previous corrected human anchors
previous pose / correction history
```

它进入模型的方式应该模仿 UniCon3R：

```text
image token + pose token + human token + A_corr_t + recurrent state
  -> decoder
  -> refined pose token + refined A_corr_t

refined A_corr_t
  -> residual head
  -> delta pose latent

refined pose token + delta pose latent
  -> pose head
  -> corrected camera pose
```

## 6. 当前已经验证到什么

目前做了一个 online 显式 baseline，用来验证这个思路是否成立。

做法：

```text
frame 0:
  没有历史，不修
  用 raw pose 初始化 world
  保存 pelvis / torso / left foot / right foot 的 world anchors

frame t:
  取当前帧 Human3R 预测的人体四个部位
  和上一帧保存的 corrected anchors 对比
  如果差很小，说明 raw pose 还稳定，不修
  如果差很大，说明 pose 可能漂了，反推 camera pose 把人体拉回去
```

注意：这个 baseline 还不是纯 token head。它用的是 Human3R 输出的 SMPL joints 显式坐标。

但它是 token-aligned 的：

```text
我们已经验证 pelvis / torso / left foot / right foot 在 token heatmap 中能找到，
所以这些显式 anchor 可以看作未来 token-level 模块的可解释替代。
```

结果：

```text
A -> A 正常连续帧:
  anchor residual 小
  gate = 0
  不修

A -> B 镜头切换帧:
  anchor residual 大
  gate = 1
  修正后人体 anchor 明显拉回

B -> B 后续帧:
  如果仍有偏移
  gate 继续触发
  修正后保持人体历史连续
```

这说明：

```text
历史 human motion 不只是能检测 drift，
它确实可以在 online 条件下纠正 pose。
```

## 7. 接下来可以验证什么

下一步不是马上训练大模型，而是逐步把显式 baseline 变成 token-level prompt。

建议验证顺序：

| 阶段 | 验证内容 | 目的 |
|---|---|---|
| 1 | 保持显式四点 anchor，测试更多 AABB case | 看 online human-motion correction 是否稳定泛化 |
| 2 | 用 token heatmap / body-part token 替代显式部位坐标 | 验证 token 层面是否真的能提供同样信息 |
| 3 | 构造 decoder-in `A_corr_t` prompt token | 把 body token、history、pose memory、gate 融合起来 |
| 4 | 把 `A_corr_t` 和 Human3R tokens 一起送入 decoder | 验证 prompt 是否能在 decoder 内和 pose / human / state 交互 |
| 5 | 训练 residual latent head | 用 refined `A_corr_t` 修正 refined pose token，再由 pose head 输出 corrected pose |

第一版最推荐的 token 组合是：

```text
A_corr_t =
  A_body_part
  + A_history_human
  + A_camera_motion
  + A_gate
```

暂时不优先使用：

```text
background feature
floor normal
raw recurrent memory cosine
```

原因是目前实验显示：人体四部位 + 历史 motion 是最强、最直接、最稳定的 cue。
