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

UniCon3R 的核心改动是：在这个流程中额外加入 contact prompt / contact token，并且用它预测一个 correction offset，反过来修正 human latent / human reconstruction。

```text
video frame
  -> encoder
  -> image / camera / human tokens

human token + local scene token + geometry cue + history memory
  -> [新增] contact token / contact prompt

image / camera / human tokens + [新增] contact token + recurrent state
  -> decoder
  -> image / camera / human / pose tokens + [新增] refined contact token

[新增] refined contact token + human token
  -> contact-guided correction head
  -> 预测 human correction offset / residual
  -> 应用回 human latent / human prediction

corrected human token / corrected human latent
  -> human head
  -> contact-aware human reconstruction

image / pose tokens
  -> scene / camera heads
  -> scene + camera pose
```

也就是说，UniCon3R 不是完全重做一个模型，而是在 Human3R 这种 foundation backbone 上加一个轻量的 contact-guided correction 分支。

更重要的是，contact token 不是只作为一个额外监督目标输出出来，而是会参与后续修正：

```text
contact token
  -> 预测 correction residual
  -> residual 应用到 human latent / human prediction
  -> 输出更合理的 human reconstruction
```

它的核心思想是：

```text
人体和场景应该有合理接触关系。
如果人漂浮、穿地、脚和地面关系不合理，
就用 contact token 作为额外提示去修正人体重建。
```

## 3. UniCon3R 的 Contact Token 怎么设计

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

## 4. 我的 Pose Correction 如何参考 Contact Token

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
  current body-part tokens
  + previous human-anchor history
  + raw camera motion cue
  + correction gate
```

其中最核心的是：

```text
pelvis token
torso token
left foot token
right foot token
previous corrected human anchors
current-vs-history residual
```

## 5. 当前已经验证到什么

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

## 6. 接下来可以验证什么

下一步不是马上训练大模型，而是逐步把显式 baseline 变成 token-level prompt。

建议验证顺序：

| 阶段 | 验证内容 | 目的 |
|---|---|---|
| 1 | 保持显式四点 anchor，测试更多 AABB case | 看 online human-motion correction 是否稳定泛化 |
| 2 | 用 token heatmap / body-part token 替代显式部位坐标 | 验证 token 层面是否真的能提供同样信息 |
| 3 | 构造 `A_corr_t` prompt token | 把 body token、history、camera motion、gate 融合起来 |
| 4 | 训练小 MLP / adapter 预测 pose error score | 先判断是否需要修正，不直接输出 pose |
| 5 | 再训练 residual SE(3) correction head | 输出 `delta_xi_t`，修正 `T_raw_t` |

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
