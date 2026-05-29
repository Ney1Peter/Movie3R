# Movie3R V8 Plan: Human-Centric Pose Correction Prompt

## 1. 任务背景

### 1.1 Human3R 的基本流程

当前可以把 Human3R 的在线推理流程简化成三步：

1. 输入视频帧，经过 encoder，从图像中提取 image tokens；模型内部同时维护 camera / pose token、human prompt / human token 等。
2. 这些 token 和上一帧保留下来的 recurrent state token 一起进入 decoder，通过 attention 融合当前帧信息和历史记忆。
3. decoder 输出 refined image token、pose token、human token，然后分别经过不同 head：
   - DPT / pointmap head 输出 depth、pointmap、scene geometry；
   - pose head 输出 camera pose；
   - human head 输出 SMPL / SMPL-X 人体参数。

### 1.2 当前存在的问题

Human3R 在普通连续视频中通常可以保持稳定，但在一种特殊情况中容易出错：

- 场景是同一个；
- 时间是连续的；
- 人和背景仍然属于同一个世界；
- 但输入视频中间发生了镜头切换，比如前半段来自相机 A，后半段来自相机 B。

这时 Human3R 容易在镜头切换后的第一帧产生明显 camera pose / world alignment 偏移。也就是说，模型没有很好理解“这是同一个连续场景，只是换了一个相机视角”。

### 1.3 想要纠正的思路

目标不是重新训练 Human3R / CUT3R 大模型，也不是做完整离线 global alignment，而是在冻结主干的前提下，增加一个轻量 pose correction 模块。

核心想法是：

> 把人体、脚、局部场景、接触关系和历史状态变成 pose correction anchor，用它们判断 raw camera pose 是否漂移，并为后续 residual pose correction 提供信息。

换句话说，V8 希望把人体从“动态干扰物”转化为“结构化的 pose correction 线索”。

## 2. UniCon3R 的做法

### 2.1 UniCon3R 解决的问题

UniCon3R 也是基于 Human3R / CUT3R 的在线 human-scene reconstruction 方法。它发现 Human3R 虽然能同时重建人体和场景，但人体经常会出现：

- 漂浮在地面上；
- 穿进场景；
- 和局部场景接触关系不合理；
- world-frame human motion 不够稳定。

所以 UniCon3R 的目标是用 human-scene contact 来纠正人体重建。

### 2.2 基于 Human3R 的流程改进

Human3R 原本可以简化为：

```text
image token + pose token + human token + state
-> decoder
-> scene / camera / human output
```

UniCon3R 增加了一个 contact prompt / contact token：

```text
image token + pose token + human token + contact token + state
-> decoder
-> scene / camera / human / contact output
```

关键不是“多预测一个 contact”，而是：

```text
contact token
-> decoder 融合
-> refined contact token
-> residual 修正 human latent
```

也就是说，contact 被当成内部纠错信号，而不是单纯的辅助输出。

### 2.3 Contact Token 的构造

UniCon3R 的 contact token 主要来自几类信息：

- 当前 human token：以人为中心作为 query；
- 当前 image / scene tokens：读取当前帧的人体和场景关系；
- recurrent state memory：读取历史场景记忆；
- local geometry：从上一帧 world pointmap 中，在人体附近采样局部 3D 几何；
- temporal momentum：复用上一帧 refined contact token，提供接触历史。

通俗地说，UniCon3R 是在问：

> 当前这个人在哪里？附近场景是什么？上一帧接触状态是什么？局部 3D 几何是否支持这个 contact？

然后它把这些信息融合成 contact prompt，用这个 prompt 反过来修正人体重建。

## 3. Movie3R V8 计划

### 3.1 如何仿照 UniCon3R 做 Pose Correction

V8 不复现 UniCon3R 的 contact branch，而是借鉴它的 prompt-based correction 思路，把 correction 目标从 human reconstruction 换成 camera pose correction。

对应关系是：

```text
UniCon3R:
human token + scene token + contact geometry + contact history
-> contact prompt
-> 修正 human latent

Movie3R V8:
human token + body structure token + local scene token + geometry token + pose/history token
-> pose correction prompt
-> 判断 / 修正 camera pose drift
```

第一阶段先不训练 delta pose head，也不把 token concat 回 Human3R decoder。先做一个 sidecar validation pipeline：

```text
Human3R frozen output
-> 提取候选 pose correction token pool
-> 可视化和验证每个候选 token 是否有意义
-> 分析这些候选 token 是否和 pose drift 相关
```

### 3.2 第一阶段先构造候选 Token Pool

当前不急着固定 `A_corr_t` 的最终组合。第一阶段更重要的是先把可能有用的 human-centric pose correction 线索都提取出来，逐个可视化、统计和验证，最后再决定哪些 token 真正进入最终 prompt。

候选 token pool 可以先包括：

```text
CandidatePool_t = {
  camera_motion_token,
  human_root_token,
  body_orientation_token,
  body_part_token,
  support_contact_token,
  near_human_scene_token,
  near_foot_scene_token,
  human_scene_geometry_token,
  temporal_history_token,
  reliability_token
}
```

每个候选 token 的含义：

- `camera_motion_token`：表示 raw camera pose 的相对运动、旋转变化、平移变化和异常跳变；
- `human_root_token`：表示 pelvis / root 的位置和速度，是人体整体轨迹 anchor；
- `body_orientation_token`：表示 torso、hip、shoulder 构成的人体朝向，用来观察 shot switch 后人体朝向是否异常；
- `body_part_token`：表示 pelvis、torso、hip、shoulder、feet 等结构化人体点；
- `support_contact_token`：表示脚、支撑状态、foot sliding、foot-to-ground relation；它是强 cue，但不是唯一 cue；
- `near_human_scene_token`：表示人体附近的静态场景区域，例如 human mask 外扩背景 ring；
- `near_foot_scene_token`：表示脚附近地面或局部支撑区域；
- `human_scene_geometry_token`：表示人体和周围 pointmap / depth / local plane 的显式 3D 几何关系；
- `temporal_history_token`：表示上一帧 pose、pelvis、body orientation、feet、support state 等历史状态；
- `reliability_token`：表示当前人体、pointmap、局部场景、contact cue 是否可信。

这里和 UniCon3R 的关系不是简单照搬 contact。更准确的对应是：

```text
UniCon3R:
human + local scene + contact geometry + contact history
-> contact prompt

Movie3R V8:
human structure + local scene + human-scene geometry + pose/history
-> pose correction prompt
```

也就是说，V8 的核心不是一定要依赖脚或 contact，而是以人为中心收集能够解释 camera pose drift 的候选证据。脚和 contact 只是其中一类很强但有条件的证据。

候选信号验证有效之后，再组合成最终 prompt：

```text
A_corr_t = SelectAndFuse(
  useful candidate tokens from CandidatePool_t
)
```

后续如果验证有效，再扩展为：

```text
A_corr_t -> pose_error_score
```

再进一步：

```text
A_corr_t -> delta_xi_t + gate_t
T_corr_t = exp(gate_t * delta_xi_t) @ T_raw_t
```

### 3.3 第一版如何验证

第一阶段重点不是直接修正 pose，而是验证候选 token 里的信息是否真的有用。

需要做两类验证。

第一类是可视化验证，检查这些东西是否提取正确：

- pelvis / torso / feet 投影是否落在人体正确位置；
- body orientation 是否能稳定反映人体朝向；
- foot / support contact token 是否真的对应脚部和支撑状态；
- near-human scene region 是否覆盖人体周围静态区域；
- near-foot scene region 是否覆盖脚下地面，而不是人体区域；
- human mask 是否能排除动态人体；
- local pointmap near human / near foot 是否有合理 3D 几何；
- local plane / local geometry 是否稳定。

第二类是数值相关性验证，观察 shot switch 或 pose drift 发生时，下面指标是否明显异常：

- support foot world jump；
- foot sliding residual；
- pelvis / root world acceleration；
- body orientation jump；
- torso / pelvis trajectory jump；
- foot-to-ground distance；
- local plane residual；
- pointmap confidence drop；
- pose jump score；
- near-foot geometry inconsistency。

如果这些指标在镜头切换后第一帧明显升高，说明对应候选 token 里确实包含 pose correction 所需的信息。后续再根据这些验证结果决定最终 `A_corr_t` 的组合。

第一阶段成功标准：

> 不训练任何 correction head，仅通过 Human3R 输出构造候选 pose correction tokens，就能在 drift 帧附近找到可解释、可视化、可量化的异常信号。

这可以证明 V8 的核心假设成立：人体结构、人体周围局部场景、显式 3D 几何和历史状态可以作为 Human3R camera pose drift correction 的结构化 anchor。支撑脚和 contact 是重要候选，但不预先假设它们一定是唯一或最优的选择。

### 3.4 版本切分：V8.1 先跑通，V8.2 再处理人体真实运动

当前 human-only 显式实验已经说明：pelvis、torso、left foot、right foot 这四个 token-aligned body anchors 对镜头切换后的 pose drift 很敏感，并且可以把错误 pose 拉回去。

但是这个实验隐含了一个较强假设：

```text
人短时间内基本停在原地。
```

如果人正在走路、跳舞、蹲下或者快速运动，不能简单把当前人体位置硬对齐到上一帧人体位置。否则真实的人体运动会被误认为 camera drift。

因此 V8 可以分成两个阶段：

| 版本 | 目标 | 核心假设 |
|---|---|---|
| V8.1 | 先验证隐式人体 token 是否能支撑 pose correction 闭环 | 短时间人体位移较小，先用 previous corrected anchors 做历史参照 |
| V8.2 | 加入 motion-aware human history | 人不是静态锚点，而是连续运动锚点 |

V8.1 的重点是先跑通最小闭环：

```text
frozen Human3R
-> dump current human/body-part tokens
-> 构造 token-aligned A_corr_t
-> 小 probe / MLP 预测 pose error 或 delta pose
-> 可视化 corrected pose
```

这一步回答的问题是：

```text
不依赖显式 GT 人体锚点时，
Human3R 内部的人体 token 是否真的包含足够信息来纠正 camera pose drift？
```

V8.2 再把“人体静止锚点”升级为“人体连续运动锚点”：

```text
previous corrected anchors:
  p_{t-1}, p_{t-2}, ...

estimated human velocity:
  v_{t-1} = p_{t-1} - p_{t-2}

predicted current anchor:
  p_pred_t = p_{t-1} + v_{t-1}

motion residual:
  p_raw_t - p_pred_t
```

这样 correction 不再假设人不动，而是假设：

```text
人在相邻短时间内的运动应该连续，不应该在 world frame 里突然瞬移。
```

V8.2 的 pose correction token 可以增加：

```text
A_history_human_motion =
  previous corrected anchors
  + previous velocity
  + previous acceleration
  + previous gate / correction confidence

A_motion_residual =
  current raw anchors
  - predicted current anchors from motion history
```

如果所有稳定人体部位的 residual 方向和大小比较一致，更像 camera pose drift；如果只有单个部位异常，更像人体动作、遮挡或 SMPL 误差，应该降低 correction gate。
