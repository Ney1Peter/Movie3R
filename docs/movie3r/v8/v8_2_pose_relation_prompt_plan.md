# Movie3R V8.2 Plan: UniCon-Style Pose Relation Prompt

本文档记录 V8.2 的最新思路。核心变化是：不再把 pose correction prompt 简单理解成几个固定人体部位 token，而是把它设计成一个 **current-history alignment relation token**。

通俗地说，V8.2 要让模型学习：

```text
当前帧的人、背景、相机状态
和上一帧的历史记忆是否还在同一个世界里对齐。

如果不对齐，就通过一个 decoder-in correction token
去修正 pose latent，最后纠正 camera pose。
```

## 1. 为什么需要 V8.2

V8.1 已经验证了几个重要事实：

1. 人体确实是很强的 pose correction anchor。显式使用 pelvis、torso、left foot、right foot 四个点，可以把 A -> B 镜头跳变拉回来。
2. encoder token 层面确实能在 pelvis / torso / feet 附近看到有意义的响应，但 feet 左右区分不一定稳定。
3. 只在 AvatarReX 的一个小分布上训练，容易出现过拟合。换到新视频或新序列时，当前模型不一定泛化。
4. 当前 V8.1 代码里的“四个 body queries”本质上是 learnable queries，并没有被显式监督成 pelvis / torso / left foot / right foot。

所以 V8.2 需要修正设计重点：

```text
V8.1 早期理解：
四个 body-part token -> 对齐人体位置 -> 修正 pose

V8.2 更合理理解：
human-centric current-history relation token
-> 判断当前帧和历史世界是否一致
-> 修正 pose latent
```

也就是说，人体部位仍然很重要，但它们应该作为 pose relation token 的候选 cue / 辅助监督，而不是强行假设每个 learnable query 天然就是某个身体部位。

## 2. 从 UniCon3R 得到的关键启发

UniCon3R 并不是直接找一个“脚 token”然后手工纠正人体。它的做法更像是构造一个 contact relation prompt：

```text
human prompt
+ current scene context
+ recurrent state memory
+ explicit local geometry cue
+ previous contact token
  -> contact token C_t

image / pose / human tokens + C_t + state
  -> decoder
  -> refined contact token C'_t

C'_t
  -> contact head: 显式监督 contact
  -> residual head: 输出 delta H
  -> 修正 human latent
  -> human head 输出更合理的人体
```

这里最重要的不是“contact token 里面某一项单独起作用”，而是：

1. 它专门为 contact 关系拉了一条分支；
2. 这个 token 进入 decoder，和原来的 human / image / pose / state tokens 交互；
3. decoder 后有 contact head 做显式 contact 监督；
4. residual head 用 refined contact token 去修正 human latent。

因此，UniCon3R 的 contact token 可以理解成：

```text
一个专门学习 human-scene contact relation 的 latent prompt。
```

V8.2 要模仿的是这个范式，而不是机械复制 contact 或脚部特征。

## 3. V8.2 要纠正什么

V8.2 纠正的是 Human3R / CUT3R 输出的 camera pose drift，尤其是：

- 镜头切换后第一帧 pose 跳变；
- 后续几帧 pose 不稳定；
- 同一场景、连续时间下，前后帧 world alignment 不一致；
- 人体或场景在同一个世界坐标下出现明显错位。

最终希望得到：

```text
raw pose token
+ pose correction residual
  -> corrected pose token
  -> pose head
  -> corrected camera pose
```

注意：V8.2 仍然是 online / causal 的。当前帧 decoder 前不能使用当前帧 pose head、scene head、human head 的输出。可以使用：

- 当前帧 encoder 后的 image / human / pose tokens；
- 当前 recurrent state / pose memory；
- 上一帧已经输出并缓存的 corrected pose、SMPL / joints、pointmap、confidence；
- 上一帧 refined correction token / delta / gate。

## 4. V8.2 靠什么信息纠正

V8.2 的核心 token 记为：

```text
A_corr_t
```

更准确地说，它是一个 pose relation prompt，而不是单纯的 body-part token。

为了更贴近 UniCon3R，可以把构造 `A_corr_t` 所用的信息概括成三类：

```text
A_corr_t = Fuse(
  Semantic Pose-Scene Context,
  Explicit Metric Geometry / Alignment,
  Temporal Momentum
)
```

这三类信息和 UniCon3R 的 contact token 是一一对应的，只是任务目标从 contact / human refinement 换成了 camera pose correction。

### 4.1 Semantic Pose-Scene Context

UniCon3R 中这一类信息是：

```text
当前帧 image / scene tokens
+ 上一时刻 recurrent scene memory
+ learned gate
  -> semantic scene context
```

它的作用是让 contact token 知道：

```text
当前人周围看起来是什么？
历史里这个场景是什么？
当前视觉证据和历史 memory 哪个更可信？
```

对应到 V8.2，semantic context 不应该只看 scene，还应该以人为中心读当前 pose / human / image tokens：

```text
U_curr_t = CA(Q_corr, current image / human / pose tokens)
U_mem_t  = CA(Q_corr, recurrent state / pose memory)
gamma_t  = sigmoid(MLP(U_curr_t, U_mem_t, reliability cue))

S_sem_t = gamma_t * U_curr_t + (1 - gamma_t) * U_mem_t
```

通俗理解：

```text
当前帧提供即时证据：
  人在哪里，人体朝向如何，脚/躯干/骨盆附近的视觉 token 是什么样。

历史 memory 提供稳定证据：
  前面几帧里这个场景、人和相机关系大概是什么样。

learned gate 决定：
  当前帧更可信，还是历史 memory 更可信。
```

这一步对应 UniCon3R 的 `U_curr / U_mem / gamma`。区别是我们不只为了判断 contact，而是为了判断当前帧和历史世界是否对齐。

### 4.2 Explicit Metric Geometry / Alignment

UniCon3R 中这一类信息是：

```text
上一帧 world pointmap
+ 当前 human anchor 附近 RoIAlign
  -> local 3D geometry descriptor
  -> MLP
  -> geometry token
```

它解决的问题是：语义 token 不一定知道真实 3D 距离。例如“脚看起来在地面附近”不等于“脚真的离地面 0cm”。所以 UniCon3R 显式加入局部 3D 几何，让 contact token 知道人体附近的场景表面在哪里。

对应到 V8.2，这一类可以叫：

```text
G_align_t: explicit metric alignment cue
```

它不是为了强行做后处理 ICP，而是把显式的当前-历史对齐线索转成 token：

```text
上一帧 corrected pose
+ 上一帧 human anchors / SMPL joints
+ 上一帧 pointmap / confidence
+ 当前帧 decoder 前 human / image anchor cue
+ pose memory jump / consistency score
  -> explicit alignment features
  -> MLP
  -> G_align_t
```

可以包含的显式量包括：

| 显式量 | 含义 |
|---|---|
| previous corrected body anchors | 上一帧人体在 corrected world 中的位置 |
| short-term human anchor velocity | 人如果在走动，应该沿着短时速度继续变化，而不是被假设原地不动 |
| previous local pointmap / confidence | 上一帧人体附近的场景几何和可靠性 |
| pose memory jump score | 当前 pose memory / pose token 是否出现异常跳变 |
| human-state consistency | 当前 human token 与历史 memory 是否一致 |
| local scene reliability | 人附近背景、地面、墙面等是否可信 |

这里要特别注意 causal 限制：当前帧 decoder 前不能使用当前帧 pointmap、当前帧 raw pose、当前帧 SMPL joints，因为它们都要等当前帧 decoder/head 之后才有。V8.2 可以使用上一帧已经输出的显式结果，也可以使用当前帧 encoder 后的 token / human anchor cue。

### 4.3 Temporal Momentum

UniCon3R 中这一类信息是：

```text
上一帧 refined contact token C'_{t-1}
  -> 对齐当前人数
  -> temporal momentum M_t
```

它的直觉是：contact 在视频中通常连续，脚着地不会一帧有一帧无，所以 previous contact token 可以稳定当前 contact 预测。

对应到 V8.2，temporal momentum 不能简单写成 camera pose smoothness。原因是我们的 AABB 场景里相机本来就可以从 A 视角切到 B 视角，如果强行让 camera pose 平滑，反而会错。

V8.2 的 temporal momentum 应该是：

```text
previous refined A_corr token
+ previous correction residual / gate
+ previous corrected pose token
+ previous human-anchor motion
  -> pose-relation momentum M_pose_t
```

它稳定的是：

```text
当前帧和历史世界的对齐关系
```

而不是强行让相机运动平滑。

通俗说：

```text
如果上一帧已经判断出 raw pose 有偏，并且修正方向是某种方式，
那么当前帧大概率还需要沿着相似的 relation correction 继续稳定下来。

如果人确实在走动，
momentum 应该参考上一帧人体运动趋势，
而不是把当前人强行拉回上一帧原地。
```

这也是 V8.2 区别于简单 human-only 对齐的关键点：它不是假设人不动，而是利用短时人体运动、历史 correction token 和 state memory，学习“合理的连续世界关系”。

### 4.4 V8.2 的三类 token 对应表

| UniCon3R contact token | 作用 | V8.2 pose relation prompt |
|---|---|---|
| Semantic Scene Context | 当前视觉证据 + 历史 scene memory + gate | 当前 human/image/pose tokens + recurrent state/pose memory + gate |
| Explicit Metric Geometry | 上一帧 pointmap 中人体附近局部 3D 几何 | 上一帧 corrected pose / human anchors / pointmap / confidence 构成的显式对齐 cue |
| Temporal Momentum | 上一帧 refined contact token | 上一帧 refined `A_corr_t`、delta、gate、corrected pose token、人体运动趋势 |

最终可以写成：

```text
A_corr_t = Fuse(S_sem_t, G_align_t, M_pose_t)
```

其中 `gate / reliability / drift_score` 不一定单独作为输入 token，也可以作为 `A_corr_t` decoder 后的输出 head：

```text
refined A_corr_t
  -> drift score / gate
  -> pose latent residual
```

## 5. 人体部位在 V8.2 中怎么用

人体部位仍然非常重要，但 V8.2 不应该直接说：

```text
四个 learnable queries = pelvis / torso / left foot / right foot
```

除非后续给它们加了明确监督。

更合理的用法有三种：

### 5.1 作为候选 cue

保留 pelvis、torso、left foot、right foot 这些部位，因为实验说明它们对 pose correction 有用。

但是在模型里可以把它们叫成：

```text
body anchor queries
```

而不是直接宣称它们已经是准确的 named body-part tokens。

### 5.2 作为辅助监督

如果希望四个 query 真的对应 pelvis / torso / left foot / right foot，需要额外加监督，例如：

```text
body anchor attention
  -> 应该集中到 GT 投影附近
```

或者用 GT-projected body-part tokens 做 teacher：

```text
learned body query 读出来的 token
  -> 接近 GT body-part patch token
```

这样以后才能更有把握地说：

```text
这个 query 对应 pelvis
这个 query 对应 torso
这个 query 对应 left / right foot
```

### 5.3 作为验证工具

即使训练时不直接使用 GT，验证阶段仍然可以用 GT SMPL 投影检查：

- query 是否看向人体；
- pelvis / torso 是否比 feet 更稳定；
- feet 是否容易左右混淆；
- 人体 token 是否比背景 token 更适合 pose correction。

## 6. V8.2 怎么监督

UniCon3R 的 loss 不是只监督最后人体 mesh。它大致可以理解成三类：

1. CUT3R / Human3R 原本继承来的 4D 重建损失，例如 pointmap、camera pose、appearance / temporal consistency；
2. SMPL-X 人体监督，例如 body pose、shape、mesh、joint、reprojection；
3. 新增 contact 分支的专属监督，例如 vertex-level contact 的 Focal BCE，以及 DECO 风格的 part-level contact loss。

这个设计很关键：contact token 之所以能学到 contact relation，不只是因为最后 human mesh 更准了，而是因为它有一个明确的 contact head 和 contact loss。

所以 V8.2 也不能只用：

```text
corrected camera pose vs GT pose
```

否则 `A_corr_t` 很容易退化成普通 pose regression shortcut。更合理的是把 loss 也分成三类。

### 6.1 原任务保持 / 4D 重建损失

这一类对应 UniCon3R 继承的 CUT3R 4D reconstruction loss。

对 V8.2 来说，它的作用是：

```text
新增 A_corr_t 后，不应该破坏原本的 scene / pointmap / appearance / temporal reconstruction。
```

可选监督包括：

| Loss | 含义 | 备注 |
|---|---|---|
| `L_pointmap_keep` | V8.2 输出 pointmap 不要偏离 frozen Human3R baseline 太多 | 没有可靠 metric depth 时优先用 keep loss，不用 DA3 depth |
| `L_conf_keep` | confidence 分布不要异常塌陷 | 防止 correction token 影响 scene head |
| `L_pose_gt` | corrected camera pose 接近 GT pose | 这是主 pose 监督，但不是唯一监督 |
| `L_pose_rel` | 相邻帧相对 pose / AABB 相对 pose 接近 GT | 比单帧绝对 pose 更符合在线连续设置 |

如果第一阶段冻结 encoder / decoder / scene head / human head / pose head，只训练 prompt 和 residual head，这类 loss 仍然有意义：它会约束 `A_corr_t` 进入 decoder 后不要把原有 image / scene / human token 搅坏。

### 6.2 人体 / Anchor 辅助监督

这一类对应 UniCon3R 的 SMPL-X human supervision。区别是我们不是主要修人体，而是利用人体作为 pose relation anchor。

可选监督包括：

| Loss | 含义 | 作用 |
|---|---|---|
| `L_smpl_keep` | V8.2 human output 不要比 frozen Human3R baseline 更差 | 防止 pose correction token 破坏 human branch |
| `L_joint_reproj_keep` | joints / mesh 投影不要明显偏离原预测或 GT mask | 保持人体检测稳定 |
| `L_body_anchor_aux` | body anchor query attention 接近 pelvis / torso / foot GT 投影区域 | 如果要解释四个 body queries，这个必须加 |
| `L_anchor_token_teacher` | learned body query 读出的 token 接近 GT-projected body-part patch token | 让 query 真正学会看对应人体部位 |
| `L_anchor_motion` | corrected world 下的人体 anchor 短时运动合理 | 防止把走动的人强行拉回上一帧原地 |

这里的重点是：如果后续想在论文里说“我们使用 pelvis / torso / feet anchor”，就需要 `L_body_anchor_aux` 或 `L_anchor_token_teacher` 这类监督。否则只能说它们是 learnable body anchor queries。

### 6.3 Pose Relation / Correction 专属监督

这一类是 V8.2 最重要的新 loss，对应 UniCon3R 的 contact loss。

UniCon3R 是：

```text
contact token
  -> contact head
  -> vertex contact / part contact loss
```

V8.2 应该是：

```text
A_corr_t
  -> drift / alignment head
  -> pose-relation loss

A_corr_t
  -> residual head
  -> pose latent correction
```

可选监督包括：

| Loss | 监督对象 | 作用 |
|---|---|---|
| `L_drift_score` | `A_corr_t` 输出 raw pose 是否漂移 / 漂移程度 | 类似 contact head，让 token 显式学习“当前是否错位” |
| `L_gate` | correction gate | 让模型知道不是每一帧都要强修 |
| `L_pose_residual` | corrected pose vs GT pose | 训练 residual branch 真正把 pose 修对 |
| `L_improvement_margin` | corrected pose error 应小于 raw pose error | 避免修正后比原版更差 |
| `L_current_history_alignment` | corrected pose 下当前人 / 局部场景与历史 world memory 更一致 | 直接监督“当前-历史对齐关系” |
| `L_temporal_relation` | corrected pose / correction token 短时连续 | 稳定 B 段后续几帧，不只修 shot 第一帧 |
| `L_residual_small` | residual norm | 防止每帧过度修正 |

`L_drift_score` 的弱标签可以由 GT pose 自动生成：

```text
raw pose error 大 -> drift label 高
raw pose error 小 -> drift label 低
```

`L_current_history_alignment` 可以用人体 anchor 和局部场景 memory 构造：

```text
上一帧 corrected world anchors
+ 上一帧 anchor velocity
+ 当前帧 corrected world anchors
  -> 当前是否符合短时运动趋势
```

注意这不是假设人静止，而是要求：

```text
人可以走动，但短时间内不应该出现由 camera pose drift 导致的异常世界跳变。
```

### 6.4 第一版建议总 loss

第一版不要一下子全开，可以从下面这个组合开始：

```text
L_total =
  L_pose_gt
+ lambda_drift * L_drift_score
+ lambda_gate  * L_gate
+ lambda_margin * L_improvement_margin
+ lambda_res   * L_residual_small
+ lambda_keep  * (L_pointmap_keep + L_smpl_keep)
```

如果要让 body queries 变得可解释，再加：

```text
+ lambda_anchor * (L_body_anchor_aux 或 L_anchor_token_teacher)
```

最重要的设计原则是：

```text
L_pose_gt 负责最终 pose 对不对；
L_drift_score / L_gate 负责 A_corr_t 是否知道哪里错；
L_current_history_alignment 负责 A_corr_t 是否真的学到当前-历史对齐关系；
keep / human losses 负责不破坏原来的 Human3R 输出。
```

## 7. Gate 应该怎么理解

gate 的作用不是单独负责“预测正确 pose”，而是控制修正强度：

```text
gate 小：相信原模型，少修正
gate 大：当前可能有 drift，允许强修正
```

V8.1 中 gate 的问题是：它没有足够明确的监督，所以有时只是跟着 residual head 间接学习，不一定真的学会“什么时候该修”。

V8.2 更合理的做法是把 gate 和 drift score 绑定起来：

```text
A_corr_t
  -> drift_score head
  -> gate
  -> 控制 residual 强度
```

drift score 可以用 GT camera pose 生成弱标签：

```text
raw pose error 大 -> drift label 高
raw pose error 小 -> drift label 低
```

这样 gate 就不是无意义的装饰，而是一个可解释的“是否需要修正”的判断。

## 8. V8.2 模型流程

整体流程建议如下：

```text
当前帧 RGB
  -> Human3R encoder / tokenizers
  -> image token, pose token, human token

当前 token + recurrent state + 上一帧缓存
  -> [新增] build A_corr_t

image token + pose token + human token + A_corr_t + recurrent state
  -> frozen decoder
  -> refined image / pose / human / A_corr tokens

refined A_corr_t
  -> drift score / gate head
  -> pose residual head
  -> delta pose latent

refined pose token + gate * delta pose latent
  -> corrected pose token
  -> frozen pose head
  -> corrected camera pose
```

第一阶段建议冻结：

```text
encoder
decoder
scene head
human head
pose head
```

只训练：

```text
A_corr_t prompt builder
drift score / gate head
pose latent residual head
可选 body anchor auxiliary head
```

后续再做 ablation，比较是否解冻 pose head / human head。

## 9. 与 V8.1 的关键区别

| 问题 | V8.1 早期版本 | V8.2 新版本 |
|---|---|---|
| token 定义 | 容易把 4 个 learnable queries 说成固定人体部位 | 明确叫 pose relation prompt，body anchors 只是其中一类 cue |
| correction 依据 | 更像人体 anchor 对齐 | 学习当前帧和历史世界的一致性关系 |
| gate | 弱监督或自然学习 | 用 drift / alignment score 显式监督 |
| UniCon 对应 | 更像只用了 human cue | 更接近 contact token: relation prompt + explicit head + residual refinement |
| 泛化风险 | 容易 overfit 某个 AvatarReX 分布 | 加入 relation / drift 监督和跨数据划分，减少只记住训练组合 |
| 可解释性 | query 是否真是 pelvis / foot 不够明确 | 可通过 body anchor auxiliary loss 单独验证 |

## 10. 后续实验计划

### 10.1 先做小规模可控实验

数据仍然使用 AvatarReX，但训练 / 测试要跨 `lbn1 / zxc / zzr` 划分，避免只在一个人或一个场景上过拟合。

建议对照：

| 实验 | 目的 |
|---|---|
| Raw Human3R | 原始基线 |
| V8.1 body-query only | 复现当前方案 |
| V8.2 relation prompt + `L_pose` | 看 relation token 是否能修正 pose |
| V8.2 + drift score / gate loss | 看 gate 是否真的学会判断 drift |
| V8.2 + recurrent state memory | 看 state memory 是否提升泛化 |
| V8.2 + body anchor auxiliary loss | 看四个 query 是否能更稳定对应人体部位 |

### 10.2 再做泛化测试

测试集至少包括：

- AvatarReX 同数据集不同序列；
- AvatarReX 不同人物 / 不同相机组合；
- AIST / H36M 这类非 AvatarReX 视频；
- 不同角度：60 / 90 / 120 / 180 度；
- 不同动作：站立、走动、转身、跳舞、蹲下。

重点观察：

- 是否只在训练人物上有效；
- 人走动时是否还能修正，而不是强行把人拉回原地；
- 背景低纹理时是否仍然稳定；
- gate 是否在 pose drift 大的帧更高；
- correction 是否破坏原本已经正确的帧。

## 11. 一句话总结

V8.2 的核心不是“用四个人体点做后处理对齐”，而是：

```text
模仿 UniCon3R 的 contact relation prompt，
构造一个 decoder-in pose relation prompt A_corr_t，
让它学习当前帧与历史世界是否对齐，
再用显式 pose / drift 监督训练 residual branch，
最终修正 Human3R 的 camera pose latent。
```

## 12. 当前最小实现版本

2026-06-02 已经按 V8.2 思路加了第一版训练前置代码。

### 12.1 代码文件

核心新增 / 修改：

```text
src/dust3r/v8_pose_prompt.py
  - V82PoseRelationPrompt
  - V82PoseRelationResidualHead

src/dust3r/model.py
  - 新增 v8_pose_prompt_variant='relation_v8_2'
  - V8.2 复用原 V8 decoder-in 接入位置
  - 额外缓存 previous delta，作为 temporal momentum

src/dust3r/losses.py
  - V82PoseRelationLoss

config/train_v8_2_pose_relation_small.yaml
  - V8.2 小批量训练配置

scripts/v8_2_check_pose_relation_shapes.py
  - V8.2 shape check
```

旧 V8.1 默认不受影响。只有配置中显式写：

```text
v8_pose_prompt_variant='relation_v8_2'
```

才会启用 V8.2 relation prompt。

### 12.2 当前 A_corr_t 的实际组成

第一版只保留 3 个 correction tokens：

```text
A_corr_t = [S_sem_t, G_align_t, M_pose_t]
```

| token | 当前代码里怎么来 | 对应设计 |
|---|---|---|
| `S_sem_t` | 一个 relation query 同时读当前 image / human / pose tokens 和 recurrent state / pose memory，再用 learned gate 融合 current 和 memory | Semantic Pose-Scene Context |
| `G_align_t` | `pose_token`、`previous corrected pose token`、二者 latent difference、memory token 经过 MLP | Explicit Metric Geometry / Alignment 的最小 latent 版本 |
| `M_pose_t` | previous refined `A_corr_t`、previous applied delta、previous gate 经过 MLP | Temporal Momentum |

当前还没有加入 pointmap、floor normal、near-scene、contact 或 named pelvis/foot auxiliary。这样做是为了先验证最小 relation prompt 是否稳定，避免一次塞太多 cue 后难以 ablation。

### 12.3 当前 correction head

decoder 后：

```text
refined A_corr_t
  -> drift_head
  -> drift_logit
  -> gate = sigmoid(drift_logit)

refined A_corr_t
  -> delta_head
  -> delta_pose_latent

corrected_pose_token =
  refined_pose_token + gate * delta_pose_latent

corrected_pose_token
  -> frozen pose head
  -> corrected camera pose
```

也就是说，gate 不再是一个独立乱学的值，而是和 drift score 绑定。

### 12.4 当前 loss

第一版训练配置使用：

```text
V82PoseRelationLoss(
  L_pose_gt,
  L_drift_score / gate,
  L_improvement_margin,
  L_residual_small
)
```

具体含义：

| Loss | 当前用途 |
|---|---|
| `L_pose_gt` | corrected pose 接近 raw calibration GT pose |
| `L_drift_score` | 用 raw pose error 自动生成 drift target，监督 drift logit / gate |
| `L_improvement_margin` | corrected error 不应比 raw error 更差 |
| `L_residual_small` | 防止 residual latent 过大 |

当前第一版暂时没有启用：

```text
L_current_history_alignment
L_temporal_relation
L_body_anchor_aux
L_anchor_token_teacher
pointmap / smpl keep loss
```

这些后续再逐项 ablation。

### 12.5 当前训练数据

配置：

```text
config/train_v8_2_pose_relation_small.yaml
```

使用之前优化过的小批量数据：

```text
manifest root:
  output/v8_1_aabb_manifests/round1_ablation_small

train:
  240 clips

val:
  60 clips

test:
  60 clips
```

数据特点：

```text
groups:
  lbn1 / zxc / zzr 各自均衡

view angle:
  >= 120 deg

frame reuse:
  split 内不重复使用 RGB frame

depth:
  load_da3_depth=False

pose target:
  raw calibration camera pose
```

训练时冻结：

```text
encoder
decoder
scene head
human head
pose head
```

只训练：

```text
V82PoseRelationPrompt
V82PoseRelationResidualHead
```

### 12.6 已完成检查

已完成：

```text
PYTHONPATH=src:. .venv/bin/python scripts/v8_2_check_pose_relation_shapes.py
PYTHONPATH=src:. .venv/bin/python -m py_compile src/dust3r/v8_pose_prompt.py src/dust3r/model.py src/dust3r/losses.py scripts/v8_2_check_pose_relation_shapes.py
PYTHONPATH=src:. .venv/bin/python src/train.py --config-name train_v8_2_pose_relation_small --cfg job
```

结果：

```text
V8.2 shape check passed
Hydra config can expand
V82PoseRelationLoss minimal forward passed
```
