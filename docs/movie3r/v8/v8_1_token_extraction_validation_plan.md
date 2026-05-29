# Movie3R V8.1 Plan: UniCon-Style Pose Prompt Validation

## 1. 目标

V8.1 的目标是跑通一个 UniCon-style 的 pose correction prompt：`A_corr_t` 在当前帧 decoder 前构造，并和 image / pose / human / state tokens 一起进入 decoder。

当前最重要的是验证两件事：

1. `A_corr_t` 的候选组成是否能从当前帧 decoder 前信息和上一帧缓存中构造出来；
2. `A_corr_t` 进入 decoder 后，refined correction token 是否能通过小 residual head 修正 pose latent。

因此，V8.1 要验证的是：像 UniCon3R 的 contact token 一样，把 pose correction prompt 放进 decoder 内部交互，是否能改善 camera pose drift。

## 2. 构造 Pose Correction 候选 Token Pool

当前不提前固定最终的 `A_corr_t`。第一阶段先尽量多提取可能有用的候选信息，然后逐个验证。

候选 token pool 可以包括：

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
  state_memory_token,
  temporal_history_token,
  reliability_token
}
```

各类候选信息的作用：

- `camera_motion_token`：进入 decoder 前只能来自上一帧 raw/corrected pose、pose memory、当前 coarse pose token 等；当前帧 `T_raw_t` 只能作为 decoder 后监督或评估，不作为 `A_corr_t` 输入。
- `human_root_token`：当前帧优先来自 human prompt / human token / learned body-part query；上一帧可以使用 cached pelvis / root 作为历史。
- `body_orientation_token`：来自 torso、hip、shoulder 构成的人体朝向，用来观察人体方向在 world frame 中是否突然跳变。
- `body_part_token`：最终 decoder-in 版本来自 learned body-part query 对当前 image / human tokens 的读取；可视化验证时可以用 SMPL pelvis、torso、feet 投影来检查这些 token 是否落在合理区域。
- `support_contact_token`：来自 foot position、foot velocity、support foot state、foot sliding、foot-to-ground relation。它是重要候选，但不是唯一核心。
- `near_human_scene_token`：来自 human mask 外扩区域或 human bbox 周围的静态背景，重点是以人为中心找局部场景。
- `near_foot_scene_token`：来自脚附近的地面或局部支撑区域，用来观察 foot-scene relation；decoder-in 版本优先使用上一帧缓存和当前图像 token。
- `human_scene_geometry_token`：进入 decoder 前优先来自上一帧 pointmap / depth / confidence，显式计算 human-scene 3D 几何关系，例如 local plane、body-to-scene distance、local residual；当前帧 pointmap 只能作为监督或评估。
- `state_memory_token`：来自 recurrent state 或 decoder 中间 token，用来观察历史场景记忆是否提供有用信息。
- `temporal_history_token`：来自上一帧 raw/corrected pose、pelvis、feet、body orientation、support state、previous correction 等。
- `reliability_token`：来自 pointmap confidence、human confidence、joint visibility、mask quality、near-scene confidence、pose jump score 等，用来判断当前 cue 是否可靠。

这些候选 token 最后不一定都会进入最终 prompt。V8.1 的任务是先把它们 dump 出来，并验证哪些真的 work。

### 2.1 V8.1 当前最小 Pose Correction Token

当前 V8.1 先不使用所有候选信息，而是先测试一个最小可行的 decoder-in prompt：

```text
A_corr_t =
  A_body_part_t
  + A_history_human_t
  + A_camera_motion_t
  + A_reliability_gate_t
```

直观拆解如下：

| 组成 | 具体内容 | 显式/隐式 | 作用 |
|---|---|---|---|
| `A_body_part_t` | pelvis、torso、left foot、right foot 四个 learned body-part query 从当前 image / human tokens 中读取到的部位 token | 隐式 token | 提供当前人体结构锚点 |
| `A_history_human_t` | 上一帧 corrected human anchors / previous body-part state | 显式历史缓存，后续可替换为隐式 memory token | 提供历史参照，判断当前人体是否在 world frame 中突然跳走 |
| `A_camera_motion_t` | 上一帧 raw/corrected pose、pose memory、当前 coarse pose token | 显式历史 + 隐式 pose prior，经过 MLP token 化 | 提供相机运动先验 |
| `A_reliability_gate_t` | human score、body-part token confidence、历史 anchor consistency、previous confidence | 显式/隐式混合，经过 MLP token 化 | 决定当前 correction 是否应该强触发 |

其中真正的当前帧人体语义部分是隐式的：

```text
pelvis token
torso token
left foot token
right foot token
```

但 `A_corr_t` 整体不是全隐式，因为它还会融合上一帧 corrected anchors、上一帧 pose history、gate score 这些显式或半显式信息。它们的用法和 UniCon3R 的 `G_t` 类似：先用数值形式计算，再通过 MLP 映射到 token 维度，最后和人体 token 融合。

V8.1 的核心验证问题是：

```text
在显式 human-only correction 已经成立的前提下，
能否把这些 token / tokenized features 作为 A_corr_t 放进 decoder，
并让 refined A_corr_t 修正 pose latent？
```

## 3. 验证 Token 是否提取正确

高维 token 不能直接可视化，但可以验证它的来源位置、相似性、跨帧一致性和对应的几何含义。

### 3.1 图像位置验证

对 image encoder token / decoder image token：

1. 用 Human3R 输出或数据集 GT 的 SMPL pelvis、torso、left foot、right foot 等 3D joints 投影回 RGB 图像，作为验证标尺；
2. 根据投影位置找到对应 patch token，检查 body-part query 读到的 token 是否和这些区域相近；
3. 在图像上画出采样 patch 区域或 attention / similarity heatmap；
4. 检查这些区域是否真的落在目标身体部位或目标背景区域。

注意：SMPL 投影在这里主要是验证工具，不是最终 decoder-in `A_corr_t` 在当前帧使用的输入。最终当前帧 body-part token 应该由 learned query 从当前 image / human tokens 中读取。

需要重点检查：

- pelvis token 是否落在人体中心附近；
- torso / shoulder / hip token 是否落在对应身体区域；
- left / right foot token 是否落在脚部或脚附近；
- near-human scene token 是否在 human mask 外部；
- near-foot scene token 是否覆盖脚下或脚旁边的局部场景；
- human mask 是否成功排除了动态人体区域。

### 3.2 Similarity Heatmap 验证

对选中的高维 token，例如 foot token、pelvis token、near-scene token，可以和整张图的 image tokens 做 cosine similarity：

```text
selected token
-> cosine similarity with all image tokens
-> reshape to patch grid
-> upsample to image size
-> overlay heatmap on RGB
```

如果提取正确，期望看到：

- foot token 的高相似区域集中在脚、鞋、脚下地面附近；
- pelvis / torso token 的高相似区域集中在人体主体附近；
- near-scene token 的高相似区域集中在人体周围背景，而不是人体内部；
- shot switch 前后，同一类 token 的语义响应仍然合理。

### 3.3 几何和区域验证

对 pointmap / geometry 相关 token，需要检查：

- local pointmap near human / near foot 是否有合理 3D 分布；
- local ground / support plane 是否稳定；
- foot-to-scene distance 是否数值合理；
- body-to-scene residual 是否受 human mask 影响；
- confidence map 是否能过滤低质量 pointmap 区域。

### 3.4 State / Memory Token 验证

state token 不一定有直接图像位置，因此先不要求解释每个 state token 的语义。

可以先做弱验证：

- 用 human / body-part query 和 state tokens 做 similarity，观察 top-k state token 是否跨帧稳定；
- 比较 current-only token 与 current+state token 的 drift proxy 表现；
- 观察 shot switch 附近 state similarity 是否出现异常变化；
- 如果模型中能拿到 attention map，再检查 human / pose token 是否在异常帧读取了不同的 state 区域。

V8.1 中 state token 的目标不是完全解释清楚，而是先判断它是否可能提供有用的历史场景记忆。

## 4. 验证 Token 是否有助于 Pose Correction

提取正确不代表一定能纠正 pose。第二步需要验证候选信号是否和 pose drift 相关。

### 4.1 不训练 Head，先做 Proxy 曲线

先计算每一帧的 proxy 指标：

- raw camera translation jump；
- raw camera rotation jump；
- pelvis / root world jump；
- pelvis / root world acceleration；
- body orientation jump；
- torso / pelvis trajectory jump；
- left / right foot world jump；
- foot sliding residual；
- foot-to-ground distance；
- near-human scene confidence；
- near-foot geometry residual；
- local plane residual；
- pointmap confidence drop；
- state similarity drop；
- reliability / gate prior score。

然后检查这些曲线在 shot switch 后第一帧是否出现明显异常。

如果某个 proxy 在 drift 帧附近稳定升高，说明对应 token 可能包含 pose correction 信息。

### 4.2 有 GT Pose 时的相关性验证

如果有 ground-truth camera pose 或 teacher pose，可以计算：

- camera translation error；
- camera rotation error；
- ATE / RTE；
- corrected / raw pose relative error；
- per-frame pose error curve。

然后计算候选 proxy 与 pose error 的相关性，例如：

```text
foot_sliding_residual vs pose_error
body_orientation_jump vs pose_error
pelvis_world_acceleration vs pose_error
local_geometry_residual vs pose_error
state_similarity_drop vs pose_error
confidence_drop vs pose_error
```

目标不是立刻得到完美预测器，而是找出哪些候选 token 和 pose error 最相关。

### 4.3 没有 GT Pose 时的 Proxy 验证

如果没有 GT pose，可以使用弱监督 proxy：

- shot boundary 后第一帧是否有异常峰值；
- support foot 是否突然在 world frame 大幅滑动；
- pelvis / torso world trajectory 是否突然断裂；
- body orientation 是否突然不合理跳变；
- local scene geometry 是否和上一帧不一致；
- pointmap / confidence 是否在失败帧变差。

这些 proxy 不能完全替代 pose error，但可以帮助判断候选 token 是否值得继续使用。

### 4.4 组合验证

当单个候选 token 有初步信号后，再做简单组合验证：

```text
camera only
camera + human root
camera + body orientation
camera + body parts
camera + support contact
camera + local scene
camera + human-scene geometry
camera + temporal history
camera + state memory
all candidates
```

第一阶段的组合验证也应围绕 decoder-in prompt 展开。可以先让 refined `A_corr_t` 预测一个 pose error / drift gate：

```text
image / pose / human tokens + A_corr_t + state
-> decoder
-> refined A_corr_t
-> small gate head
-> pose_error_score
```

如果 `pose_error_score` 能在 drift 帧附近升高，说明这些 token 进入 decoder 后仍然保留了诊断 pose drift 的能力。

之后再进入真正的 latent pose correction：

```text
image / pose / human tokens + A_corr_t + state
-> decoder
-> refined pose token + refined A_corr_t
-> residual head(refined A_corr_t)
-> delta pose latent + gate

corrected pose token = refined pose token + gate * delta pose latent
corrected pose token -> pose head -> T_corr_t
```

### 4.5 V8.1 最小闭环测试：UniCon-Style Human Pose Prompt

在完成 token 提取和显式 proxy 验证后，V8.1 做一个最小闭环训练。目标不是解决所有人体运动情况，而是验证：

```text
A_corr_t 进入 Human3R decoder 后，
refined A_corr_t 是否能产生有用的 pose latent residual，
并通过 pose head 输出 corrected camera pose。
```

第一版输入尽量保持简单，只使用已经验证过的 token-aligned 人体部位：

```text
decoder 前构造:
A_corr_t =
  pelvis token
  + torso token
  + left foot token
  + right foot token
  + previous corrected human-anchor memory
  + previous pose / pose memory cue
  + reliability gate cue

decoder 内:
image / pose / human tokens + A_corr_t + state
  -> decoder attention

decoder 后:
refined A_corr_t
  -> residual head
  -> delta pose latent

refined pose token + delta pose latent
  -> pose head
  -> corrected pose
```

这里的 previous corrected human-anchor memory 可以先用上一帧显式 corrected anchors 作为历史输入，后续再替换成纯 token memory。关键是：当前帧 correction token 仍然进入 decoder，并在 decoder 内和 pose / human / image / state tokens 交互。

建议按三个级别推进：

| 级别 | 做法 | 目的 |
|---|---|---|
| Level 1 | refined `A_corr_t` 预测 pose error score / drift gate | 验证 correction prompt 进入 decoder 后还能识别错误帧 |
| Level 2 | refined `A_corr_t` 预测 pose latent residual，修正 pose token | 验证 correction prompt 能影响 pose latent |
| Level 3 | corrected pose token 经过 pose head 输出 `T_corr_t` 并用 viewer 可视化 | 验证完整 decoder-in prompt-to-pose 闭环是否成立 |

这一阶段可以先使用一个或少量 AABB case 做 overfit / sanity check：

```text
input:
  current image / pose / human tokens
  A_corr_t
  recurrent state
  previous-frame cached anchors / pose history

output:
  pose_error_score
  delta pose latent / gate
  corrected Human3R viewer result
  raw vs corrected camera trajectory comparison

supervision:
  GT camera pose 或 explicit human-only correction teacher
```

如果这个 decoder-in 闭环在小样本上都无法拟合，说明当前 `A_corr_t` 设计或注入位置不够；如果可以拟合，再扩大到 10 组 AABB 做 train / validation split。

## 5. V8.1 成功标准

V8.1 的成功标准不是最终修好 pose，而是完成以下验证：

1. 能稳定 dump Human3R / CUT3R 的 raw pose、pointmap、confidence、SMPL joints / mesh、mask，以及可访问的 image / human / state tokens。
2. pelvis、torso、feet、near-human scene、near-foot scene 的 token 来源位置能被可视化，并且和预期区域一致。
3. similarity heatmap 能说明高维 token 的响应区域大致合理。
4. local geometry、body orientation、foot/contact、state/history 等 proxy 指标能被逐帧计算。
5. 在已知 drift / shot switch 位置，至少一部分候选 proxy 出现可解释异常。
6. 能通过单 token 和组合 ablation 判断哪些候选信息最有用。
7. 最小 UniCon-style decoder-in prompt 能在小样本 sanity check 中拟合 pose error / pose latent residual，并输出可视化 corrected pose，用来判断隐式人体 token 是否具备 correction 闭环能力。

只有当 V8.1 验证通过后，才进入下一阶段：确定最终 `A_corr_t` 的组成，并扩大训练数据和 ablation。
