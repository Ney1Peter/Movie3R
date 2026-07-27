# Movie3R-Hybrid V14.0：一次性隐式粗对齐与多人显式精对齐的严格流式系统

> 2026-07-27 状态说明：本文保留 V14 的总体设计空间，其中仍包含早期的四帧
> shadow-rollout 方案。当前已实现的 V14.1 是三帧 event-only capacity probe，权威规范见
> `V14_1_ONE_SHOT_SHADOW_CORRECTION_TRAINING.md`。若两份文档在 V14.1 输入长度、state
> 语义、loss 或实现状态上冲突，以 V14.1 文档为准。state commit 与显式 shot Boundary
> propagation 尚属于 V14.2。

## 0. 你的身份与任务

你现在是一名负责 Movie3R 研究开发的资深三维视觉研究员和代码工程师。

项目仓库：

```text
https://github.com/Ney1Peter/Movie3R
```

Movie3R 基于 Human3R，研究单目、多分镜、存在 camera cut 的流式 camera-human-scene 三维重建。

你的任务是：

1. 完整阅读当前 Movie3R 仓库；
2. 理解 V9、V12、V13 的设计、代码和实验；
3. 在不破坏现有三个正式版本的前提下，设计并实现新的正式版本：

```text
Movie3R-Hybrid V14.0
```

V14 的核心思想是：

```text
V9-style one-shot latent coarse re-anchoring
+
V13-style identity-conditioned multi-human explicit residual refinement
+
V12-style hard reset、fixed shot Boundary 和安全单人 fallback
```

更准确地说：

```text
camera cut
→ 在 reset 前保存上一 shot 的 raw recurrent state 只读快照
→ 对第一张 post-cut frame 共享执行一次 image / human encoding
→ raw main rollout：hard reset 后执行原版 Human3R，并提交唯一主 state
→ V9 shadow rollout：从 pre-cut raw state 出发，执行完整 correct-token decoder refine
→ shadow rollout 的 corrected state、pose memory 和 tracking 全部丢弃
→ 从 raw/corrected camera 输出中提取 shot-to-shot 显式相对变换 ΔB0
→ 与上一 shot 的固定 world Boundary 组合得到当前粗 Boundary B0
→ B0 只负责将新镜头拉回大致正确的方向和世界区域
→ 在 B0 粗对齐后的空间中进行跨镜人物身份匹配
→ 使用可靠人物身份进行 V13-style residual Boundary refinement
→ 得到一个最终 shared Boundary B*
→ 整个新 shot 固定使用 B*
→ 后续帧不再做 latent correction，也不重新估计 Boundary
```

V14 必须是严格因果、严格流式、固定内存、无未来帧、无全局 BA、无离线回看优化的系统。

这里的双 rollout 不等于双长期 state。系统在 cut 结束后始终只保留：

```text
一个 raw Human3R shot-local recurrent state
+
一个固定大小的显式 shot Boundary state
+
一个固定大小的 external identity/world memory
```

`PreCutContext` 和 shadow rollout state 只在当前 cut 的固定预算计算中存在，完成
`B0` 后立即释放，不进入下一帧。

---

# 1. 首先必须理解的研究背景

## 1.1 Human3R 的正常流式结构

Human3R 可以抽象为：

```text
当前 RGB I_t
+
上一帧 recurrent state S_{t-1}
+
上一帧 pose memory M_{t-1}
        ↓
image encoder / human encoder
        ↓
recurrent encoder-decoder
        ↓
pose token z_t
image tokens F_t
human tokens H_t
new recurrent state S_t
        ↓
camera head → camera pose
scene head  → pointmap/depth/confidence
human head  → SMPL-X/root/joints/vertices
```

形式上：

\[
S_t = F(S_{t-1}, I_t)
\]

\[
Y_t = H(S_t)
\]

其中输出 \(Y_t\) 包括：

- camera pose；
- scene pointmap；
- SMPL-X；
- human root；
- joints；
- vertices。

在普通连续镜头内，Human3R 的流式重建通常是稳定的。camera cut 的困难并不是新镜头内部无法重建，而是 cut 发生的一瞬间同时产生了两个问题。

### 问题一：State contamination

如果第一张 post-cut frame 继续读取旧镜头的 recurrent state：

```text
old shot state
→ first post-cut frame
→ contaminated first write
→ 后续整个新 shot 的 recurrent trajectory 被带偏
```

### 问题二：World-gauge discontinuity

即使在 cut 处 hard reset，新镜头也只会产生一个干净但独立的 shot-local reconstruction：

```text
old shot world gauge
≠
new shot local gauge
```

两者之间可能存在：

- rotation；
- translation；
- metric scale；
- camera orientation；
- human placement；

等差异。

因此必须明确区分：

```text
State transition
```

和：

```text
Gauge transition
```

它们不是同一个问题。

---

# 2. V9：学习式隐式 token correction

## 2.1 V9 的基本设计

V9 在 Human3R decoder 内增加 relation prompt，并在 latent 层分别纠正：

- pose token；
- human token。

当前 V9 relation prompt 主要由以下信息组成：

```text
当前 image tokens
当前 pose token
当前 human tokens
旧 recurrent state
旧 pose memory
上一帧 correction prompt
上一帧 latent residual
上一帧 gate
```

relation prompt 包含：

```text
semantic token
alignment token
momentum token
```

随后：

```text
relation prompt
→ decoder interaction
→ pose latent residual Δz_t
→ human latent residual ΔH_t
→ learned correction gate g_t
```

最终：

\[
\hat z_t = z_t + g_t \Delta z_t
\]

\[
\hat H_t = H_t + g_t \Delta H_t
\]

再交给 Human3R 原有 pose/human heads：

```text
corrected pose token → camera head → corrected camera
corrected human token → human head → corrected SMPL-X
```

V9 还给 pose head 和 human head增加了 LoRA，使原有 head 能够更好地解释 corrected latent。

---

## 2.2 V9 的优点

V9 已经证明：

1. Human3R latent 中存在一定的 camera-human correction 信息；
2. pose token 和 human token 可以通过小型 residual branch 被纠正；
3. AABB/AAAA 联合训练可以让 learned gate 区分跳变和正常连续帧；
4. learned correction 可以改善一部分 camera translation 和 rotation；
5. camera correction 不能只修 pose，human placement 也必须同步考虑；
6. latent branch 可以较小侵入地接入 Human3R，不需要重新训练整个 backbone。

因此 V9 不是没有价值。

它最重要的意义是证明：

```text
旧状态、当前图像、pose token 和 human token 之间
存在可以预测跨镜粗关系的隐式信号。
```

---

## 2.3 V9 的根本不足

### 2.3.1 原有 token 中的信息量有限

V9 使用的并不是专门为跨镜世界对齐学习的 representation，而是重新组合 Human3R 已有的：

- image token；
- camera/pose token；
- human token；
- recurrent state；
- pose memory。

这些 token 首先服务于单镜头重建，而不是精确的跨镜 SE(3) 恢复。

因此 V9 的能力上限有限：

```text
它可能识别出“方向明显不对”
但很难恢复精确的 rotation 和 translation。
```

V9 更适合：

```text
粗对齐
```

而不是：

```text
最终精确 Boundary。
```

---

### 2.3.2 V9 当前是逐帧 correction

原 V9 每一帧都会：

```text
重新构造 relation prompt
→ 重新预测 gate
→ 重新预测 latent residual
→ 决定当前帧是否修正
```

但 V12 的实验已经表明：

```text
camera cut 的核心错误集中在跳变边界和第一次错误 state write。
```

新镜头一旦获得：

- 干净的 fresh state；
- 一个正确或近似正确的 shot-to-world Boundary；

后续 shot-local reconstruction 通常可以自然延续。

因此不需要在 post-cut shot 的每一帧持续修正。

逐帧 correction 反而会产生：

- 每帧 residual 不一致；
- camera 抖动；
- human placement 抖动；
- shot 内 world gauge 缓慢变化；
- correction history 误差积累；
- 正常帧被过度修正；
- corrected token 污染 pose memory。

---

### 2.3.3 相同 latent residual 不等价于相同 SE(3)

假设第一张 post-cut frame 得到：

\[
\Delta z_0
\]

直接把它复用到后续帧：

\[
\hat z_t = z_t + \Delta z_0
\]

不代表每一帧都发生了相同的显式刚体变换。

因为 pose head 是非线性函数：

\[
C_t = h_{\text{pose}}(z_t)
\]

一般不存在：

\[
h_{\text{pose}}(z_t + \Delta z_0)
=
B h_{\text{pose}}(z_t)
\]

对不同帧都成立。

一阶近似下：

\[
h_{\text{pose}}(z_t+\Delta z_0)-h_{\text{pose}}(z_t)
\approx
J_h(z_t)\Delta z_0
\]

不同帧的 Jacobian \(J_h(z_t)\) 不同，因此同一个 latent residual 会被解释成不同的显式 rotation/translation。

human latent residual 问题更严重，因为 human token 同时编码：

- body pose；
- root translation；
- global orientation；
- shape；
- expression；
- 当前视角；
- 人体上下文。

所以第一帧的 human residual 不能直接复制给后续人物 token。

---

### 2.3.4 V9 没有 one shared Boundary 保证

V9 分别修改：

- camera latent；
- human latent；
- pose head；
- human head。

它没有数学上保证：

```text
camera correction
=
scene correction
=
human correction
=
同一个 SE(3)
```

scene head 也没有独立的 latent correction。

因此可能出现：

```text
camera 看起来被拉回来了
human 也发生了一定变化
但 camera、scene、human 不属于同一个严格 world transform
```

---

### 2.3.5 当前代码会把 corrected pose token 写入 pose memory

当前 `src/dust3r/model.py` 中，pose latent residual 被加到 `pose_token_for_head` 后，修正后的 pose token 会被用于更新 pose retriever memory。

这意味着逐帧 correction 不只是改变当前输出，还会影响后续帧的 pose memory。

V14 必须改变这一点：

```text
raw pose token
→ 更新主 pose memory

corrected pose token
→ 只进入 proposal head
→ 不写入主 pose memory
```

---

### 2.3.6 原 V9 训练分布偏窄

正式 V9 主要训练于：

```text
4-frame AABB / AAAA
max_humans = 1
```

因此：

- 长 shot；
- 多人；
- 多 cut；
- variable person count；
- first-frame-only correction；

都不完全属于原训练分布。

V14 不能简单加载旧 V9 checkpoint 然后假设它自然变成 one-shot multi-human coarse aligner。

旧 V9 权重可以用于初始化和诊断，但需要按 first-post-cut-only 协议重新训练完整的
V9 shadow correction branch。不是把 V9 替换成一个弱 event-level MLP。

---

# 3. V12：单人显式 shot-level 对齐

## 3.1 V12 的基本结构

V12 的正式结构是：

```text
streaming RGB + cut trigger
→ cut 前保存允许跨镜保留的人体历史
→ first post-cut frame 解码前 hard reset
→ fresh Human3R shot-local reconstruction
→ Fixed Explicit coarse anchor
→ V16 torso-motion rotation
→ V11.4 shared shot scale
→ explicit translation solve
→ one fixed shot-level Boundary
→ camera + scene + complete SMPL-X 共享同一 Boundary
→ Align-Then-Commit
```

Boundary 只在 cut 处估计一次，并在整个新 shot 内复用。

---

## 3.2 V12 的优点

### 3.2.1 正确处理了 state lifecycle

V12 明确证明：

```text
reset 必须发生在 first post-cut frame 进入 decoder 之前。
```

如果先让第一张 post-cut frame 读取 old state，再在输出层修正 camera，无法消除 state contamination。

---

### 3.2.2 正确使用 shot-level fixed Boundary

V12 不逐帧重新估计 transform，而是：

```text
cut 时估计一次 B
→ 整个新 shot 固定应用 B
```

这符合 camera cut 的实际结构：

```text
一个 shot 内部属于同一个 local gauge。
```

---

### 3.2.3 单人 torso motion 是有效的 rotation cue

V16 利用 cut 前人体 torso angular motion，预测 first post-cut 时刻的人体方向，再对 coarse rotation 做 bounded refinement。

实验表明它可以明显降低 rotation ambiguity。

---

### 3.2.4 一个 shared Boundary 保持内部几何一致性

V12 强制：

- camera；
- pointmap；
- human root；
- global orientation；
- root-centered body offsets；
- joints；
- vertices；

使用同一个 Boundary。

这避免了 camera、human 和 scene 各自使用不同 correction 的结构性错误。

---

## 3.3 V12 的不足

### 3.3.1 依赖单人 anchor

单人 torso/root 可能受到：

- 当前姿态；
- 遮挡；
- 检测误差；
- root depth；
- 视角；
- torso heading ambiguity；

影响。

单个人的几何噪声会直接决定整个 shot 的 Boundary。

---

### 3.3.2 它仍然是显式后处理式对齐

V12 完全依赖显式输出：

- camera；
- root；
- torso；
- pointmap。

它没有利用 Human3R latent 中可能存在的跨镜关系先验。

当 camera cut 的视角跨度很大时，显式 Fixed initialization 可能落入错误的 rotation branch。

---

### 3.3.3 scene consistency 存在 trade-off

V12 的 human-dominant translation closure 能改善 camera-human placement，但不保证 scene 同时改善。

V12 当前不能宣称完整的：

```text
camera-human-scene closure。
```

---

### 3.3.4 多 cut 后会累计漂移

每个新 shot 都基于上一个 predicted world 继续构造 Boundary。

因此：

```text
B_1 → B_2 → B_3 → ...
```

误差会逐次累积。

V12 适合：

```text
short horizon
sparse cuts
```

而不是无限长度全局 mapping。

---

### 3.3.5 当前 runtime 代码还不是理想的 frame-by-frame API

当前 V12 runner 会先对完整输入序列运行 Human3R，再基于输出构造各个 Boundary。

虽然算法设计是因果的，但新的 V14 不能继续依赖这种完整序列式 runner。

V14 必须提供真正的：

```text
pipeline.step(frame, intrinsics, cut_event)
```

逐帧接口。

---

# 4. V13：多人 shared-Boundary 精对齐

## 4.1 V13 的基本结构

V13 在正确身份条件下，为每个 matched human 构造一个 Boundary candidate：

```text
human i
→ root-motion target
→ torso-motion target
→ Fixed + V16
→ R_i
→ t_i
```

然后融合：

```text
R = SO(3) mean(R_1, R_2, ..., R_N)
t = arithmetic mean(t_1, t_2, ..., t_N)
B = [R, t]
```

这里必须严格保留 V13 Phase 2 已验证的 `mean_raw_t`：每个人先在自己的
rotation candidate 下得到 raw translation candidate，再对这些 `t_i` 做算术平均。
不得默认改成在 shared rotation 下重新求 translation。已有结果为：

```text
mean_raw_t naive multi mean        composite = 0.657
shared-R recomputed translation    composite = 0.710
```

后者已被证明更差，只能作为消融，不能写成冻结的 V13 默认方法。

所有：

- camera；
- pointmap；
- 所有人体；

共同使用同一个 Boundary。

核心原则是：

```text
Identity answers WHO.
Geometry answers WHERE.
All humans share ONE Boundary.
```

---

## 4.2 V13 的优点

### 4.2.1 多人能够降低单人 rotation ambiguity

多个正确匹配的人提供独立 torso/root observation。

实验已证明：

```text
1 人 → 2 人 → 3 人
```

随着有效人数增加，camera rotation 和 composite error 呈改善趋势。

多人收益主要来自：

```text
rotation redundancy。
```

---

### 4.2.2 WHO 和 WHERE 被明确分离

V13 的 identity 模块只负责：

```text
谁和谁对应。
```

它不能直接输出：

- SE(3)；
- scale；
- Boundary；
- fusion weight。

最终几何仍由显式 solver 决定。

这是正确的职责划分。

---

### 4.2.3 已经具备 causal identity memory 基础

当前 V13 已有：

- external track IDs；
- TTL；
- fixed-size prototype history；
- tentative match；
- commit；
- Match-Then-Align；
- Align-Then-Commit；
- shot-persistent identity state。

这些结构可以作为 V14 的身份基础。

---

## 4.3 V13 的不足

### 4.3.1 自动跨镜身份仍不可靠

V13 当前可靠的多人几何依赖 GT-ID。

自动身份使用过：

- Human3R refined human token；
- CUT3R head token；
- Multi-HMR token；
- beta；
- local pose；
- DINO appearance；
- causal persistent identity state。

但目前仍没有形成稳定的默认 automatic multi-human system。

---

### 4.3.2 错误身份会产生灾难性 Boundary

如果把人物 A 错误对应到人物 B：

```text
错误 root target
+
错误 torso target
→ 错误 R_i/t_i
→ 错误 candidate 进入 shared consensus
→ camera、scene、所有人一起被变换到错误世界
```

错误身份的风险是高度非对称的：

```text
拒绝一个正确匹配
→ 只是失去多人收益

接受一个错误匹配
→ 可能污染整个新 shot
```

因此必须优先控制：

```text
wrong accept
```

而不是只追求平均 IDF1。

---

### 4.3.3 最新 identity state 已提高 coverage，但 automatic commit 仍未成立

V13 Phase 5 已证明 shot-persistent running-mean identity state 比 stateless prototype
更有效。在 MultiHuman `three` development streams 上：

```text
stateless IDF1                  = 0.8202
stateful running-mean IDF1      = 0.9255
zero-wrong multi coverage       = 50.0%
Top-6 GT assignment recall      = 92.22%
detection-order agreement       = 100%
```

但当前手工 WHO-WHERE joint scorer 仍未超过 identity-only margin：

```text
identity-only zero-wrong coverage = 51.11%
best joint score coverage         = 35.56%
```

因此 V14 的 identity baseline 必须从 Phase 5 stateful identity 开始，而不是退回
Phase 4 的低覆盖 stateless gate。真正需要解决的是：

```text
在低 wrong-accept 风险下获得非零且有用的多人覆盖
+
提供可靠的单人和零匹配 fallback。
```

---

### 4.3.4 现有 joint WHO-WHERE scorer 尚未验证成功

当前 V13 的 causal persistent identity state 是积极结果，但现有手工 joint identity-geometry scorer 没有超过 identity-only scorer。

因此 V14 不应假设：

```text
加入几何 cost 一定改善身份。
```

V14 必须先验证：

```text
V9 粗对齐后的 geometry
是否真的提高 correct-ID ranking 和 margin。
```

粗几何应该首先作为：

- soft compatibility；
- rejection cue；
- top-K reranking cue；

而不是直接覆盖 identity-only evidence。

---

# 5. 为什么必须提出 V14

## 5.1 前因后果

目前三条路线分别解决了问题的一部分。

### V9 证明

```text
latent 中存在跨镜粗 correction signal。
```

但 V9：

- 信息有限；
- 逐帧 correction；
- 没有 fixed Boundary；
- 没有严格 shared geometry；
- 不够精确。

### V12 证明

```text
camera cut 应该只在边界处理一次。
```

而且：

```text
hard reset
+
fixed shot Boundary
```

是正确的 state/gauge 管理方式。

但 V12：

- 单人；
- 显式 initialization 可能落入错误 branch；
- 没有利用 latent 中的粗跨镜关系。

### V13 证明

```text
正确身份下，多人可以提高 Boundary 精度。
```

但 V13：

- 自动身份困难；
- wide-view cut 后人物 ID 容易混淆；
- wrong ID 会造成灾难性 shared-Boundary failure。

---

## 5.2 V14 的核心研究假设

V14 的核心假设是：

> 不要求 latent branch 精确恢复最终世界坐标，只要求它将新镜头从一个完全错误的 gauge 拉回到大致正确的方向和位置区域。粗对齐之后，人物 root、torso 和多人布局进入可比较范围，跨镜身份更容易判断。获得可靠身份后，再使用多人的显式几何约束求精确 residual Boundary。

可以概括为：

```text
粗 WHERE
→ WHO
→ 精 WHERE
```

对应分工：

```text
V9-style latent branch：
解决大范围 rotation/gauge jump，提供粗初始化。

V13-style multi-human solver：
利用可靠身份和多人运动约束，求小范围 residual refinement。

V12-style protocol：
负责 hard reset、fixed Boundary、single fallback 和 Align-Then-Commit。
```

---

## 5.3 V14 不是什么

V14 不是：

```text
旧 V9 每帧 correction
+
旧 V13 后处理
```

也不是：

```text
把第一帧 latent residual 原样复制给整个 shot。
```

更不是：

```text
每一帧重新估计一个 SE(3)。
```

V14 正确的定义是：

```text
第一帧 latent residual
→ 只生成一次粗显式 Boundary B0
→ 将 B0 传播给整个 shot
→ cut 时再估计一次 residual ΔB
→ 最终整个 shot 固定使用 B* = ΔB B0
```

---

# 6. V14 的总体架构

正式名称建议为：

```text
Movie3R-Hybrid V14.0
```

副标题建议为：

```text
One-Shot Latent Coarse-to-Fine Re-anchoring for Streaming Multi-Shot Human Reconstruction
```

注意：仓库中已经存在大量 `v14_*` 历史实验编号，它们属于 V12 的历史组件和审计，不是新的正式 V14 release。

因此必须：

```text
新建 versions/v14/
```

并在文档中明确：

```text
versions/v12/experiments/v14_* = legacy experiment IDs
versions/v14/                  = formal Movie3R-Hybrid V14.0
```

禁止重命名、删除或覆盖旧实验文件。

---

# 7. V14 的状态划分

V14 必须将状态分成四类，但只有第一类是 Human3R 的长期递归 state。V14 不维护
两条跨帧运行的 corrected/raw recurrent trajectories。

## 7.1 Human3R shot-local recurrent state

```text
LocalRecurrentState
├── recurrent_state
├── recurrent_state_pos
├── pose_memory
├── pose_memory_pos
├── raw last pose token
├── raw last human tokens / IDs
└── other original Human3R within-shot tracking state
```

职责：

```text
负责当前 shot 内部的局部重建。
```

camera cut 时：

```text
先生成只读快照，再必须 hard reset。
```

cut 后第一帧的 raw rollout产生的新 state是唯一允许提交的 Human3R state。后续帧
只沿这条 raw trajectory 运行。

---

## 7.2 Pre-cut read-only context

```text
PreCutContext
├── full old raw recurrent state snapshot
├── full old raw pose-memory snapshot
├── old state / memory positions
├── last raw pose token
├── last raw human tokens and validity mask
├── optional fixed-size raw token history
└── required original Human3R decoder metadata
```

职责：

```text
为 one-shot V9 shadow rollout提供上一 shot 的完整 decoder context。
```

camera cut 时：

```text
snapshot
detach
read-only
```

它不能作为 post-cut raw recurrent decoder 的主状态，也不能在 shadow rollout 后被
写回。它只在一个 cut 内存活，得到 `B0` 后立即释放。

默认 shadow branch 直接读取完整 pre-cut raw state，而不是只读取其 pooling summary。
这是为了保留 V9 中 state、pose memory、image/human token 和 correct token 的完整
decoder attention 约束。只使用 summary/MLP 的弱 proposal作为消融，不是正式默认。

`G_prev` 和 external identity/world memory不进入 V9 shadow decoder；它们分别由
ShotBoundaryState和 IdentityWorldMemory持有，只在显式 Boundary composition、WHO和
Align-Then-Commit阶段使用。

---

## 7.3 Shot Boundary state

```text
ShotBoundaryState
├── shot_id
├── previous shot-to-world boundary G_prev
├── relative coarse bridge ΔB0
├── coarse_boundary B0
├── residual_boundary ΔB
├── final_boundary B*
├── boundary_confidence
├── rotation_uncertainty
├── translation_uncertainty
└── optional shot scale
```

职责：

```text
把当前 shot-local reconstruction 映射到统一 streaming world。
```

一个 shot 内：

```text
B* 必须固定。
```

若 shadow corrected camera 位于上一 shot 的 raw local gauge，且 fresh raw camera 位于
当前 shot 的 raw local gauge，则：

\[
\Delta B_0
=
C_0^{\text{shadow, prev-local}}
\left(C_0^{\text{raw, curr-local}}\right)^{-1}
\]

\[
B_0^{\text{world}}
=
G_{\text{prev}}
\Delta B_0
\]

因此 raw recurrent state 可以始终保持 shot-local，无需向 latent state 注入 world
transform。多 cut 只组合固定大小的显式矩阵，不需要保留历史 state 图。

处理下一次 cut 时：

```text
G_prev = 当前 shot 已冻结的 final Boundary B*
```

不保存之前所有 `ΔB0` 列表，也不重新优化旧 Boundary。

---

## 7.4 External identity/human world memory

```text
IdentityWorldMemory
├── external track IDs
├── identity prototypes
├── world-space root history
├── world-space torso history
├── appearance history
├── beta/local-pose history
├── last_seen
├── TTL
└── active/inactive status
```

职责：

- WHO；
- 跨 shot 人体物理运动；
- V12/V13 几何 anchor；
- 人物进入、离开和重新出现。

更新时必须遵循：

```text
Match-Then-Align
Align-Then-Commit
```

未经最终 Boundary 对齐的 local human 不能写入 world memory。

---

# 8. V14 的逐帧输入输出协议

必须实现真正的逐帧 API，概念接口为：

```python
world_output, streaming_state = pipeline.step(
    rgb=current_rgb,
    intrinsics=current_K,
    cut_event=cut_event,
    state=streaming_state,
)
```

每次调用只能读取：

- 当前帧；
- 已经到达的过去帧状态；
- 固定大小历史。

禁止读取：

- 下一帧；
- 当前 shot 的未来帧；
- 完整视频；
- GT camera；
- GT identity；
- GT SMPL-X；
- GT depth。

初期允许使用 GT cut index，但它只能作为：

```text
cut trigger
```

不能提供任何几何值。

---

# 9. No-cut 普通帧路径

当：

```text
cut_event = false
```

必须执行：

```text
当前 RGB
→ 原始 Human3R shot-local recurrent inference
→ 得到 raw local camera/scene/humans
→ 使用当前 shot 固定 Boundary B*
→ 转成 world output
→ 更新 aligned identity/world human memory
```

普通帧必须满足：

1. 不调用 V14 latent coarse branch；
2. 不重新估计 Boundary；
3. 不运行 V13 residual solver；
4. 不运行额外 DA3/VGGT；
5. 不改变 Human3R 原始 shot-local recurrent trajectory；
6. 只增加一次固定刚体变换和有限 identity memory 更新。

正常帧额外开销应接近：

```text
shared transform + bounded identity update
```

而不是第二次 Human3R forward。

---

# 10. Cut 帧完整路径

当 frame \(t\) 是第一张 post-cut frame 时，执行以下严格顺序。

---

## Module 0：Cut Event Router

### 输入

```text
cut_event = true
当前 frame t
旧 LocalRecurrentState
旧 ShotBoundaryState
旧 IdentityWorldMemory
```

### 操作

1. 在任何 reset 之前复制完整旧 raw local state，构造 `PreCutContext`；
2. 保存最近固定数量的 world human/root/torso history；
3. 将所有保存内容 detach；
4. 保存上一 shot 固定 Boundary `G_prev`；
5. 为 raw main branch 清空旧 Human3R local recurrent state和 pose memory；
6. 保留外部 world/identity memory；
7. 为新 shot 创建新的 `shot_id`。

### 输出

```text
PreCutContext
Fresh LocalRecurrentState
Old IdentityWorldMemory
G_prev
```

### 强制要求

```text
reset 必须发生在 first post-cut frame 进入 recurrent decoder 前。
```

---

## Module 1：Fresh Shot-Local Human3R Reconstruction

### 输入

```text
当前 post-cut RGB I_t
当前 intrinsics K_t
fresh local state
```

### 操作

先对当前 RGB 执行一次共享 image / Multi-HMR encoding。然后只对 raw main branch执行
一次 fresh Human3R decoder rollout，V9 correct token和 V9 LoRA 全部关闭，得到：

```text
raw image tokens F_0
raw pose token z_0
raw human tokens H_0
raw new recurrent state S_0^L
raw new pose memory M_0^L
raw local camera C_0^L
raw local pointmap X_0^L
raw local SMPL-X humans
```

共享 encoder 的输出允许 shadow branch复用，但 raw decoder输出和 raw state不得读取
任何 corrected token或 corrected LoRA结果。

### 输出

```text
FreshLocalOutput
FreshTokens
FreshLocalStateAfterWrite
```

### 状态写入规则

主 Human3R state 只能写入：

```text
raw new recurrent state
raw pose token / raw pose memory
raw local tracking information
```

禁止将 V14 corrected pose/human token 写入主 local state。

---

## Module 2：One-Shot Latent Coarse Proposal

这是完整保留 V9 refine能力的 one-shot shadow branch，不是一个
`old/new feature -> MLP -> B0` 的弱临时 head。

它只在：

```text
first post-cut frame
```

执行一次。

### 输入

```text
完整、只读的 PreCutContext
共享的当前 image tokens F_0
共享的当前 Multi-HMR / human prompt tokens H_0
上一 shot raw pose token和 raw human token context
当前 human validity mask
```

### 不允许输入

```text
未来 post-cut frames
GT identity
GT camera
GT SMPL-X
完整 post-cut shot
```

### Shadow decoder 设计

默认首先保持 V9 的显式 correct-token结构：

```text
current image / pose / human tokens
+ pre-cut raw recurrent state
+ pre-cut raw pose memory
        ↓
V9 relation correction prompt builder
        ↓
A_corr,0 = semantic / alignment / event token(s)
        ↓
[pose ; A_corr,0 ; image ; human]
        ↓
完整 Human3R recurrent decoder attention
        ↓
refined corrected pose token + corrected human token
        ↓
V9 pose correction head + pose LoRA
V9 human correction head + human LoRA
```

shadow decoder从 `PreCutContext` 中的完整上一 shot raw state开始，而不是从 fresh raw
state summary开始。这样旧 shot 的约束直接参与全部 decoder layers，避免把跨 shot
关系压缩成单个弱向量。

V9 原有逐帧 momentum history、correction cache和 gate propagation不作为 V14 默认状态。
第一版在 cut event 已知时只执行一次 correction；`previous correction / delta / gate`
默认置空。是否保留一个只读的 last-raw-pose/human context由 V14.1 消融决定。

### 输出

```text
shadow corrected camera proposal Ĉ_0^(prev-local)
shadow corrected human proposal Ĥ_0^(prev-local/camera)
refined correct token
pose latent residual Δz_0（diagnostic）
per-human latent residual ΔH_0（diagnostic）
rotation confidence / uncertainty
translation confidence / uncertainty
human consistency confidence
```

不再输出“当前帧修不修”的逐帧 gate。

已知 cut 发生时，需要预测的是：

```text
这次 coarse proposal 有多可靠
```

而不是：

```text
是否发生 correction。
```

### Shadow transaction 与状态提交

shadow branch可以在内部生成：

```text
shadow new_state_feat
shadow new pose memory
shadow tracking token / ID
shadow correction history
```

但这些值只能用于当前 cut 的 diagnostic，不能提交。完成 corrected camera/human head
输出后，整个 shadow transaction被丢弃。

注意：

```text
shadow new_state_feat 不写入主 recurrent state
shadow corrected pose token 不写入主 pose memory
shadow corrected human token不写入主 tracking或 identity memory
shadow smpl_id不覆盖 raw / external identity
shadow correction history不跨帧保存
```

它们只用于：

- 估计粗 Boundary；
- Boundary consistency；
- confidence；
- identity matching workspace；
- diagnostics。

最终公开输出不直接使用 corrected proposal。

cut 处理结束后只保留 raw rollout产生的 `FreshLocalStateAfterWrite`。因此 V14 虽然在
cut 帧临时执行两个 decoder rollout，但整个系统仍只有一条长期递归 trajectory。

---

## Module 3：显式粗 Boundary 提取

### 输入

```text
raw fresh camera C_0^L
V9 shadow corrected camera proposal Ĉ_0^(prev-local)
previous fixed shot-to-world Boundary G_prev
raw human proposal
corrected human proposal
uncertainty
```

假设 camera pose 解码后采用 camera-to-world 约定。首先求当前 shot raw local gauge
到上一 shot raw local gauge 的相对 bridge：

\[
\Delta B_0
=
\hat C_0^{\text{prev-local}}
\left(C_0^{\text{curr-local}}\right)^{-1}
\]

再组合到统一 streaming world：

\[
B_0^{\text{world}}
=
G_{\text{prev}}
\Delta B_0
\]

第一 shot 取 `G_prev = I`。具体实现必须先把 pose encoding解码成统一的 4x4 c2w，
再做群运算，并用单元测试确认左乘方向。禁止直接对 pose encoding向量做减法。

### V14 Core 默认

第一阶段建议固定：

```text
B0 ∈ SE(3)
scale = 1
```

原因：

1. V9 latent 不一定可靠恢复 metric scale；
2. V13 当前多人几何本身固定 `s=1`；
3. 先验证 rotation/translation coarse-to-fine 主假设；
4. 避免 DA3/Keypoint 复杂度干扰核心结论。

V12 的 shared scale 可以作为后续独立扩展：

```text
V14-Core: SE(3)
V14-Scale: optional Sim(3)
```

不能一开始就把 scale、DA3、Keypoint、VGGT 全部混入主系统。

### Camera 是 B0 的主要定义来源

粗 Boundary 首先由 camera proposal定义。

human corrected proposal不应独立产生另一套最终 Boundary。

human branch用于：

1. 检查 camera-defined \(B_0\) 是否也能解释 human root；
2. 检查 global orientation；
3. 计算 coarse proposal 置信度；
4. 提供 identity geometry；
5. 约束 local body pose 不被错误修改。

例如：

\[
E_{\text{root-consistency}}
=
\left\|
\hat r_j^{\text{prev-local}}
-
\Delta B_0 r_j^{\text{curr-local}}
\right\|
\]

\[
E_{\text{orientation-consistency}}
=
d_{\mathrm{SO(3)}}(
\hat R_j^{\text{prev-local}},
R(\Delta B_0) R_j^{\text{curr-local}}
)
\]

Human3R 的 `smpl_transl` 起始于 camera frame。上述 root/orientation consistency
必须先分别使用 raw c2w和 shadow c2w转到各自 branch 的 shot-local frame。禁止直接
对两个 branch 的 camera-frame `smpl_transl` 做差并将其称为 world translation。

V14.1 默认从 camera pair唯一提取 `ΔB0`，human head用于 shared-rigidity loss和
confidence。将 camera/human candidates再做解析式 SE(3) projection可以作为后续消融，
但不得输出独立的 camera/human transform。

如果 camera 与 human proposal 不一致，应降低 B0 confidence，而不是让 camera 和 human分别使用不同 transform。

### 输出

```text
relative bridge ΔB0
world coarse Boundary B0 = G_prev ΔB0
coarse rotation uncertainty
coarse translation uncertainty
camera-human consistency
coarse validity flag
```

---

## Module 4：Coarse-Aligned Identity Workspace

### 输入

```text
raw post-cut local humans
B0
IdentityWorldMemory
pre-cut predicted root/torso histories
appearance/beta/local-pose features
```

### 操作

只在临时匹配空间中应用 B0：

\[
\tilde r_j^W = R_0 r_j^L + t_0
\]

\[
\tilde T_j^W = R_0 T_j^L
\]

注意：

```text
此时不能提交 identity memory。
```

### Identity evidence

身份匹配应区分两类信息。

#### Gauge-invariant identity evidence

```text
appearance
beta
root-centered local pose
persistent identity prototype
native human token
```

#### B0-aligned geometry evidence

```text
coarse world root
coarse torso heading
predicted root motion
predicted torso motion
pairwise human layout
person ordering/layout consistency
```

可定义：

\[
\begin{aligned}
D_{ij} = {}&
\lambda_{\text{app}}D_{\text{app}}(i,j)
+
\lambda_\beta D_\beta(i,j)
+
\lambda_{\text{pose}}D_{\text{pose}}(i,j)\\
&+
\lambda_{\text{root}}
\rho\left(
\frac{
\|\hat a_i^W-\tilde r_j^W\|
}{
\sigma_r
}
\right)\\
&+
\lambda_{\text{torso}}
d_{\mathrm{SO(3)}}(
T_{i,\mathrm{target}}^W,
\tilde T_j^W
)
\end{aligned}
\]

其中：

- \(\hat a_i^W\)：旧 world 中人物 \(i\) 对当前时刻的预测 root；
- \(T_{i,\mathrm{target}}^W\)：预测 torso orientation；
- \(\rho\)：Huber 或 truncated robust loss。

### 重要约束

当前 V13 已经表明简单 joint WHO-WHERE scorer 不一定优于 identity-only。

因此必须同时评测：

```text
identity-only
identity + B0 root
identity + B0 torso
identity + B0 root + torso
identity + layout
```

geometry 必须先作为：

- soft cue；
- compatibility gate；
- rejection cue；
- top-K reranker；

不能在没有实验依据的情况下压过 identity evidence。

### Top-K 身份假设

对于 2–3 人场景，应保留：

```text
top-K assignments
+
unmatched/dustbin hypotheses
```

不要在 coarse stage 只保留一次 Hungarian 结果。

三人的排列数量很小，可以显式枚举或使用 K-best assignment。

### 输出

```text
identity hypotheses
identity-only score
geometry compatibility score
accepted/rejected mask
identity margin
unmatched/new-person mask
```

仍然不提交 identity state。

---

## Module 5：V13-Style Residual Fine Alignment

V13 不再从 identity transform 开始估计完整 Boundary。

它应在 B0 基础上只估计：

\[
\Delta B \approx I
\]

最终：

\[
B^*
=
\Delta B B_0
\]

### 输入

```text
B0
可靠 identity hypothesis
pre-cut world root histories
pre-cut torso histories
post-cut raw local roots/torsos
B0-aligned roots/torsos
```

### 每个人的 residual candidate

对 matched human \(i\)：

```text
predicted world root a_i
predicted torso target T_i
B0-aligned post root
B0-aligned post torso
        ↓
bounded residual rotation ΔR_i
residual translation evidence
```

得到：

```text
(ΔR_i, Δt_i)
```

### Rotation fusion

对于两个及以上可靠身份：

\[
\Delta R
=
\operatorname{SO3Mean}
(
\Delta R_1,
\Delta R_2,
...,
\Delta R_N
)
\]

### Translation 求解

严格复用 V13 Phase 2 冻结的 `mean_raw_t`。每个人先在自己的 residual rotation
candidate 下得到 raw residual translation：

\[
\Delta t_i
=
a_i^W - \Delta R_i \left(B_0 r_{\pi(i)}^L\right)
\]

然后：

\[
\Delta t
=
\operatorname{Mean}_i(\Delta t_i)
\]

最终：

\[
\Delta B=[\Delta R,\Delta t],
\qquad
B^* = \Delta B B_0
\]

禁止默认在 shared `ΔR` 下重新计算所有 translation candidate。该
`shared-R recomputed translation` 在 V13 Phase 2 中 composite 为 `0.710`，明显差于
冻结 `mean_raw_t` 的 `0.657`。它只能保留为实验消融。

### Residual prior

因为 B0 已经提供粗初始化，fine solver 应偏向小 residual：

\[
E_{\text{prior}}
=
\|\log(\Delta B)\|_{\Sigma_0^{-1}}^2
\]

较大的 \(\Delta B\) 可以作为 coarse proposal失效或 identity冲突的风险信号，但 V13
已经证明 geometry residual不能可靠判断冲突双方谁的身份正确。因此以下信号只能做
保守的 hypothesis-level uncertainty，不得重新设计 hard reject或 per-human fusion weight：

- residual magnitude；
- candidate dispersion；
- torso residual；
- layout residual；
- identity margin；

进行拒绝。

### 重要要求

fine alignment：

```text
只在 first post-cut frame 估计一次。
```

禁止：

```text
每个后续帧重新估计 ΔB。
```

---

## Module 6：安全路由与 fallback

根据可靠 matched identity 数量执行：

```text
N >= 2
→ B0 + V13 multi-human residual refinement

N == 1
→ B0 + V12 single-human residual refinement

N == 0
→ B0 only
  或经过验证的 scene-only safe refinement
```

禁止默认：

```text
N == 0
→ identity-free human Fixed
```

因为弱 identity-free human anchor 可能在 wide-view cut 上产生灾难性结果。

### B0 也不可靠时

若：

```text
B0 confidence 低
+
无可靠人物
+
无可靠 scene cue
```

应执行安全策略：

```text
保留 fresh shot-local gauge
创建 disconnected local submap
world_valid = false
输出低置信 / unanchored 标记
不伪造高置信 world alignment
```

或者使用明确验证过的 conservative fallback。

不能为了输出一个 world pose 而强行接受不可靠 Boundary。

`B0` 无效时不得把 shot-local root、torso或 camera当成 world值写入 external memory。
只有未来某个因果时刻获得可靠 bridge时，才允许从该时刻开始建立新的 world-valid
segment；已经输出的历史仍不得回改。

---

## Module 7：One Shared Final Boundary

最终得到：

```text
B* = ΔB B0
```

它必须共同作用于：

- camera；
- scene pointmap；
- depth-derived points；
- 所有 human roots；
- global orientation；
- joints；
- vertices；
- body offsets。

对于 rigid SE(3)：

\[
C_t^W = B^* C_t^L
\]

\[
X_t^W = R^* X_t^L + t^*
\]

\[
r_{t,j}^W = R^* r_{t,j}^L + t^*
\]

\[
R_{t,j}^W = R^* R_{t,j}^L
\]

具体实现时必须根据 Human3R 的：

- camera-to-world；
- camera-frame root；
- local/global orientation；
- pointmap coordinate convention；

集中实现坐标变换，禁止在不同模块中复制不同版本的矩阵逻辑。

---

## Module 8：Align-Then-Commit

只有完成最终 Boundary 后，才能更新：

```text
world human root history
world torso history
identity appearance prototype
identity local pose prototype
active/inactive identity state
shot Boundary state
```

正确顺序：

```text
raw local prediction
→ temporary coarse alignment
→ identity hypothesis
→ final fine alignment
→ final world output
→ verify
→ commit identity/world memory
```

错误顺序：

```text
raw local prediction
→ 先写 identity memory
→ 再进行 alignment
```

未经 alignment 的 local root 绝不能污染长期 world memory。

---

# 11. 后续 shot 内帧如何处理

对同一个新 shot 的后续帧 \(t>t_0\)：

```text
RGB I_t
→ raw fresh Human3R local recurrent inference
→ 使用固定 B*
→ world output
→ aligned memory commit
```

严禁：

```text
再次运行 latent correction
再次预测 gate
再次估计 B0
再次运行 V13 Boundary solver
每帧产生新的 SE(3)
```

传播给整段的是：

```text
显式 B*
```

而不是：

```text
第一帧 latent residual Δz_0 / ΔH_0。
```

这是 V14 最核心的设计原则。

---

# 12. V14 完整数据流图

```text
                             frame t
                               |
                         external cut?
                               |
              +----------------+----------------+
              |                                 |
            no cut                            camera cut
              |                                 |
      raw Human3R local path        snapshot full pre-cut raw state
              |                    preserve G_prev + world/ID memory
              |                                 |
              |                   encode first post-cut RGB once
              |                                 |
              |              +------------------+------------------+
              |              |                                     |
              |       raw main rollout                     V9 shadow rollout
              |       hard reset                           read S_prev directly
              |       correction/LoRA off                  correct token + full decoder
              |       raw state commit                     corrected pose/human heads
              |              |                                     |
              |              |                         discard all shadow state
              |              |                                     |
              |              +------ ΔB0 = Ccorr inv(Craw) --------+
              |                                 |
              |                         B0 = G_prev ΔB0
              |                                 |
              |                   B0-aligned identity workspace
              |                                 |
              |          persistent identity + appearance + beta + pose
              |                    + coarse root/torso/layout
              |                                 |
              |                      top-K identity hypotheses
              |                                 |
              |             +-------------------+-------------------+
              |             |                   |                   |
              |          N >= 2               N == 1              N == 0
              |       V13 multi residual    V12 single residual    B0 only
              |             |                   |                   |
              |             +-------------------+-------------------+
              |                                 |
              |                          final B* = ΔB B0
              |                                 |
              +---------------- fixed shared Boundary -------------+
                                                |
                  camera + pointmap + complete SMPL-X in one world
                                                |
                                      Align-Then-Commit
```

---

# 13. V14 的训练设计

## 13.1 训练目标

V14.1 不再训练逐帧 correction controller，也不先训练一个弱的直接 SE(3) MLP。
它训练的是完整的 first-post-cut V9 shadow correction：

```text
pre-cut raw state
+ first post-cut image/human tokens
+ explicit correct token
→ full decoder refine
→ corrected pose/human heads
→ explicit relative bridge ΔB0
```

目标是：

```text
只读取已经到达的 pre-cut history和第一张 post-cut frame
→ 隐式纠正第一帧
→ 提取一个能解释整个新 shot 的显式粗 Boundary。
```

---

## 13.2 两类 target 构造必须分开

### A. 同 camera / 同 shot pseudo-reset

对没有真实 view jump 的连续视频，可以使用不断流 frozen Human3R teacher：

```text
I_0, I_1, ..., I_{t-1}, I_t, I_{t+1}, ...
```

student 在 `t` 前执行 pseudo-reset，训练恢复 teacher gauge：

```text
pre-cut frames
→ snapshot full raw PreCutContext
→ raw branch hard reset
→ V9 shadow branch从 PreCutContext执行一次 correction
```

此设置只验证 state-reset consistency和短时 propagation。因为 teacher没有跨真实 wide-view
cut，它不能单独证明 cross-camera re-anchoring。

### B. 真实 cross-camera cut

对 wide-view camera cut，禁止把“不 reset、直接跨 cut运行的 Human3R”当 teacher。该
teacher本身会受到旧 state contamination，不能提供可靠 world target。

具有 calibrated GT camera的数据先把 dataset GT gauge归一化到 pre-shot predicted gauge。
设最后一个 pre-cut camera为：

```text
C_pre_pred：Human3R 在 previous raw local gauge中的 c2w
C_pre_gt：dataset world中的 GT c2w
C_post_gt：第一张 post-cut frame的 GT c2w
C_post_raw：fresh raw branch在 current raw local gauge中的 c2w
```

则 dataset world到 previous predicted local gauge的变换为：

\[
A_{\text{gt}\rightarrow\text{prev-local}}
=
C_{\text{pre-pred}}
\left(C_{\text{pre-gt}}\right)^{-1}
\]

post camera target为：

\[
C_{\text{post-target}}^{\text{prev-local}}
=
A_{\text{gt}\rightarrow\text{prev-local}}
C_{\text{post-gt}}
\]

relative Boundary target为：

\[
\Delta B_{\text{target}}
=
C_{\text{post-target}}^{\text{prev-local}}
\left(C_{\text{post-raw}}^{\text{curr-local}}\right)^{-1}
\]

GT camera只用于训练 target和 evaluation，不进入 shadow branch输入。

两种训练设置必须单独报告，不能将 same-camera pseudo-reset结果包装成 wide-view cut能力。

训练 shadow branch只允许读取：

```text
detached pre-cut raw context
+
当前 first post-cut frame。
```

---

## 13.3 可以使用未来帧作为训练 loss，但不能作为模型输入

训练时可以取得：

```text
I_t, I_{t+1}, ..., I_{t+K}
```

用来检查 first-frame 预测的 \(B_0\) 是否能解释后续 shot。

但是：

```text
I_{t+1...t+K}
```

只能进入 loss，不能进入 V14 adapter。

也就是说：

```text
future supervision is allowed
future conditioning is forbidden
```

这仍然符合因果推理。

---

## 13.4 多人训练输入

V14 必须支持 variable human count。

human-set 编码必须：

- permutation-invariant 或 permutation-equivariant；
- 使用 validity mask；
- 不能依赖 detection index 的固定顺序；
- 不能假设始终只有一个人；
- 不能要求 cut 前后先有正确 ID 才能输出 B0。

粗 latent branch解决的是：

```text
identity-free coarse WHERE。
```

因此它可以使用：

- pooled human-set context；
- aggregate torso/root distribution；
- current human count；
- image/scene token；

但不能依赖 GT cross-shot identity。

raw branch与 shadow branch处理的是同一张 post-cut图像，并共享 detection / human prompt
顺序，所以 branch间的同 index human consistency不需要跨 shot identity。GT identity仍然
不得用于产生 `ΔB0`。

---

## 13.5 参数冻结策略

V14.1 第一版建议从 V9 pose+human checkpoint初始化 shadow adapter，但保持 raw main
branch严格等价于原版 Human3R：

```text
冻结 Human3R image encoder
冻结 recurrent encoder/decoder
冻结 scene head
冻结 base pose head
冻结 base human head
raw branch关闭所有 V9 correct token与 LoRA
shadow branch启用并训练 V9 correct-token builder
shadow branch训练 pose latent residual / pose LoRA
shadow branch训练 human latent residual / human LoRA
训练 uncertainty head
```

可比较：

```text
A. frozen V9 one-shot，完全不训练
B. 从 V9 初始化，只训练 correct-token和 residual modules
C. 从 V9 初始化，同时训练 shadow-only pose/human LoRA
```

LoRA必须只在 shadow branch生效。若 LoRA 同时作用于 raw main branch，就无法满足 no-cut
parity，也无法把 branch差异解释为单独的 `ΔB0`。

训练过程中 `PreCutContext`、raw outputs和 raw state全部 detach；梯度不能通过 raw
Human3R trajectory回传。shadow产生的 state不进入下一帧。

---

# 14. V14 的损失函数

## 14.1 Boundary rotation loss

\[
\mathcal L_R
=
d_{\mathrm{SO(3)}}(
R(\Delta B_0),
R(\Delta B_{\text{target}})
)
\]

---

## 14.2 Boundary translation loss

\[
\mathcal L_t
=
\rho(
t(\Delta B_0)-t(\Delta B_{\text{target}})
)
\]

建议使用 Huber/robust loss，避免极端样本支配训练。

---

## 14.3 First-frame camera loss

\[
\mathcal L_{\text{camera-first}}
=
d(
\hat C_0^{\text{prev-local}},
C_{0,\text{target}}^{\text{prev-local}}
)
\]

---

## 14.4 Shared camera-human Boundary consistency

要求 corrected human proposal 能由 camera-defined B0 解释：

\[
\mathcal L_{\text{shared-root}}
=
\sum_j
\left\|
\hat r_j^{\text{prev-local}}
-
\Delta B_0 r_j^{\text{curr-local}}
\right\|
\]

\[
\mathcal L_{\text{shared-orient}}
=
\sum_j
d_{\mathrm{SO(3)}}(
\hat R_j^{\text{prev-local}},
R(\Delta B_0)R_j^{\text{curr-local}}
)
\]

所有 root和 torso必须先从各自 branch 的 camera frame转换到对应 shot-local frame。
该 loss不能直接作用于 camera-frame `smpl_transl`。

---

## 14.5 Segment-level propagation loss

把第一帧预测的 B0 应用于后续 raw local reconstruction：

\[
\mathcal L_{\text{segment}}
=
\sum_{\tau=0}^{K}
w_\tau
d(
\Delta B_0 C_{t+\tau}^{\text{curr-local}},
C_{t+\tau,\text{target}}^{\text{prev-local}}
)
\]

human 只使用具有明确坐标语义的 root、torso、joints或 vertices，不使用符号 `H`
指代 latent human token：

\[
\mathcal L_{\text{segment-human}}
=
\sum_{\tau=0}^{K}
w_\tau
d(
\Delta B_0 r_{t+\tau,j}^{\text{curr-local}},
r_{t+\tau,j,\text{target}}^{\text{prev-local}}
)
\]

该 loss 是 V14 的关键。

它确保：

```text
第一帧 shadow correction产生的不是只对第一帧有用的偶然修正，
而是可以作为整个 shot 固定 Boundary 的粗解。
```

---

## 14.6 Local body preservation loss

V14 human latent correction不应随意改变：

- beta；
- body pose；
- expression；
- root-centered body geometry。

可以使用：

\[
\mathcal L_{\text{local-preserve}}
=
\lambda_\beta
\|\hat\beta-\beta^L\|
+
\lambda_\theta
\|\hat\theta_{\text{body}}-\theta_{\text{body}}^L\|
\]

以及 root-centered joints：

\[
\left\|
(\hat J-\hat r)
-
R(\Delta B_0)(J^L-r^L)
\right\|
\]

若 shadow human head产生无法被同一个 `ΔB0` 解释的独立 root translation或 articulation
变化，应由 shared/local-preserve loss惩罚，而不是把这些独立变化写入最终人体结果。

---

## 14.7 Latent residual regularization

\[
\mathcal L_{\text{latent}}
=
\|\Delta z_0\|^2
+
\lambda_H\|\Delta H_0\|^2
\]

避免 latent branch 重新生成完整 reconstruction。

---

## 14.8 Uncertainty calibration

不再训练逐帧 gate，而是训练：

```text
rotation uncertainty
translation uncertainty
human consistency uncertainty
```

可以使用 heteroscedastic regression 或 calibration loss，使 confidence 与实际 Boundary error 对应。

---

## 14.9 No-cut / false-trigger robustness

可以加入 AAAA 或 false-trigger 样本：

```text
没有真实 gauge jump
但错误触发了 V14 event branch
```

此时 target：

\[
B_{\text{target}}\approx I
\]

用于防止未来自动 cut detector 的 false positive 造成大幅 correction。

但这不是恢复逐帧 gate，而是：

```text
event-level no-op robustness。
```

---

# 15. 推荐的训练阶段

## Phase A：Frozen V9 one-shot diagnostic

暂时不训练新模型。

使用当前 V9 checkpoint：

1. 只在第一张 post-cut frame启用 correction；
2. 保存 raw camera 和 corrected camera；
3. 构造 \(B_0\)；
4. 禁止 corrected token写回 state；
5. 将 B0 应用于后续 shot；
6. 检查 coarse rotation、translation 和 identity rank。

目的：

```text
验证旧 V9 是否已经具备可利用的粗 Boundary 信号。
```

---

## Phase B / V14.1：训练 one-shot V9 shadow correction

保留 V9 的 correct-token、完整 decoder refine和两个 corrected heads，只改变 correction
的生命周期：

```text
pre-cut raw state + first post-cut tokens
→ V9 correct token
→ full shadow decoder
→ corrected pose/human heads
→ ΔB0
→ discard shadow state
```

只研究：

- camera Boundary；
- human consistency；
- segment propagation。

暂时使用 GT-ID 进行后续 V13 refinement，隔离 coarse module 的作用。

---

## Phase C：验证 coarse alignment 是否帮助 identity

固定 V14 coarse head，比较：

```text
identity-only
identity + B0 root
identity + B0 torso
identity + B0 layout
identity + all coarse geometry
```

重点报告：

- correct-ID Recall@1；
- Recall@K；
- assignment margin；
- accepted precision；
- accepted coverage；
- wrong-accept rate；
- multi activation coverage。

只有 coarse geometry 确实改善身份分离后，才进入完整 automatic V14。

---

## Phase D：完整 coarse-to-fine V14

```text
one-shot B0
→ automatic identity
→ V13 multi residual / V12 single residual
→ final fixed B*
```

---

## Phase E：可选 scale 和 scene extension

只有 V14 Core 的 SE(3) 路线通过后，才单独评估：

- V12 shared scale；
- anthropometric scale；
- contact scale；
- scene-based scale；
- scene residual verification。

禁止在初期把所有组件同时加入，导致无法解释收益来源。

---

# 16. 当前代码需要重点处理的问题

## 16.1 不要继续扩大 `src/dust3r/model.py`

当前 `model.py` 已经包含大量历史 V2–V9 分支。

V14 应采用最小侵入式 hook：

```text
导出 / 导入完整 raw recurrent state snapshot
共享当前帧 encoder / human prompt结果
允许 branch-local correction与 LoRA开关
返回 raw与 shadow camera/human outputs
显式控制 state / memory / tracking commit
```

新逻辑尽量放在独立模块中。

建议新增：

```text
src/dust3r/v14_shadow_rollout.py
src/dust3r/v14_outputs.py
src/dust3r/v14_token_hooks.py
```

---

## 16.2 必须拆开 raw main rollout 与 corrected shadow rollout

当前代码中 correct token在 head 前已经进入 decoder attention，因此同时改变
`new_state_feat`；corrected pose token随后又进入 pose memory。只拆开 head token或最后
一次 state assignment不够。

V14 必须明确：

```text
raw rollout：correction/LoRA off，允许 commit
shadow rollout：correction/LoRA on，禁止任何 commit
```

以及：

```text
raw state / raw pose memory / raw tracking：长期保留
shadow state / shadow pose memory / shadow tracking：cut 内临时，随后释放
```

两个 rollout共享当前帧 encoder和 human detection结果，但从不同 recurrent state开始：

```text
raw branch    <- hard-reset fresh state
shadow branch <- detached full pre-cut raw state
```

主状态只能接收 raw rollout结果。

---

## 16.3 禁止使用旧 latent cache 作为正式传播方案

当前代码存在：

```text
v9_oracle_correction_cache
```

它会缓存 pose/human latent residual并在后续帧复用。

V14 可以保留它作为消融：

```text
cached latent residual baseline
```

但正式 V14 禁止依赖它。

正式传播对象必须是：

```text
B0 / B*
```

---

## 16.4 V9 需要独立 V14 runner

当前 `versions/v9/run.py` 只是通用推理脚本 wrapper。

V14 必须实现独立的逐帧入口：

```text
versions/v14/run.py
versions/v14/streaming_pipeline.py
```

不能继续依赖一次处理完整序列的 generic runner。

---

## 16.5 不要从 V12 archive/experiment 脚本直接拼 runtime

当前 V12 runner依赖多个：

```text
archive/
experiments/
外部 DA3 path
```

V14 应将真正需要的稳定功能抽取为正式模块，而不是从历史实验脚本 import。

建议：

```text
versions/v14/boundary/
├── shared_transform.py
├── single_residual.py
├── multi_residual.py
├── motion_prediction.py
└── boundary_quality.py
```

---

## 16.6 不要把 `gt_id_consensus.py` 直接作为 runtime 模块

V13 的 `gt_id_consensus.py` 是大型 benchmark/evaluator 脚本。

需要从中提取：

- robust velocity；
- torso angular prediction；
- per-human residual candidate；
- SO(3) fusion；
- `mean_raw_t` translation fusion；

到可复用模块。

GT-ID、数据集读取和 evaluator 不得进入 V14 推理依赖。

---

## 16.7 扩展现有 causal identity memory

可以复用：

```text
versions/v13/identity_bridge.py
versions/v13/shot_persistent_identity.py
```

但需要新增独立的：

```text
CoarseGeometryContext
```

例如：

```text
CoarseGeometryContext
├── B0
├── sigma_R
├── sigma_t
├── predicted_world_roots
├── predicted_world_torsos
├── aligned_post_roots
├── aligned_post_torsos
└── pairwise_layout
```

identity memory 的 tentative/commit 语义必须保留。

---

# 17. 推荐的新目录结构

```text
versions/v14/
├── README.md
├── LATEST_MODEL.md
├── manifest.json
├── run.py
├── train.py
├── viewer.py
├── streaming_pipeline.py
├── config/
│   ├── train_v14_event_coarse.yaml
│   ├── eval_v14_core.yaml
│   └── eval_v14_identity.yaml
├── model/
│   ├── shadow_rollout.py
│   ├── v9_one_shot_correction.py
│   ├── boundary_extractor.py
│   └── uncertainty.py
├── state/
│   ├── streaming_state.py
│   ├── pre_cut_context.py
│   └── identity_world_memory.py
├── identity/
│   ├── geometry_context.py
│   ├── causal_matcher.py
│   └── hypothesis.py
├── geometry/
│   ├── transform_conventions.py
│   ├── apply_shared_boundary.py
│   ├── motion_prediction.py
│   ├── single_residual.py
│   ├── multi_residual.py
│   └── fallback_router.py
├── experiments/
│   ├── one_shot_v9_diagnostic.py
│   ├── latent_cache_vs_explicit_boundary.py
│   ├── coarse_identity_rank_probe.py
│   ├── gt_id_residual_probe.py
│   └── full_v14_benchmark.py
├── docs/
│   ├── V14_METHOD.md
│   ├── V14_STATE_AND_GAUGE.md
│   ├── V14_STREAMING_PROTOCOL.md
│   ├── V14_TRAINING.md
│   └── V14_EXPERIMENTS.md
└── tests/
    ├── test_streaming_state.py
    ├── test_cut_reset_order.py
    ├── test_proposal_no_state_write.py
    ├── test_boundary_composition.py
    ├── test_shared_transform.py
    ├── test_identity_commit_order.py
    └── test_no_future_leakage.py
```

可以根据仓库现有风格调整，但职责必须保持分离。

---

# 18. 严格流式和在线约束

V14 必须满足以下全部要求。

## 18.1 因果性

处理 frame \(t\) 时，只能读取：

```text
frames <= t
```

禁止读取：

```text
frames > t
```

---

## 18.2 First-frame-only correction

latent coarse branch只在：

```text
first post-cut frame
```

运行一次。

---

## 18.3 Fixed Boundary

每个 shot 只产生一个：

```text
B*
```

在下一个 camera cut 到来前保持不变。

---

## 18.4 有限内存

所有 history 都必须有固定长度：

```text
K-frame root/torso history
prototype window
TTL
fixed recurrent state
```

内存不能随视频长度无限增长。

`PreCutContext` 是上一 shot raw state 的临时只读快照，大小与一个 Human3R state固定相同；
shadow rollout结束后释放。它不是第二条长期 trajectory。

---

## 18.5 固定预算

cut 时允许额外计算，但必须是固定预算：

- 一次共享 image / human encoding；
- 一次 fresh raw decoder rollout；
- 一次 full V9 shadow decoder rollout；
- 有限数量 identity hypotheses；
- 有限人数 residual solve；
- 无迭代到收敛的全局优化。

---

## 18.6 不回改历史输出

一旦 frame \(t\) 输出，不允许在未来重新修改该 frame 的结果。

因此：

- 无 smoothing over future；
- 无 backward pass over sequence；
- 无离线 BA；
- 无 future identity correction；
- 无 retrospective trajectory rewrite。

---

## 18.7 无独立 camera/human/scene transform

最终只能有一个：

```text
B*
```

禁止：

```text
camera 用 B_camera
human 用 B_human
scene 用 B_scene
```

---

## 18.8 无未对齐状态提交

未经 B* 对齐的：

- root；
- torso；
- identity observation；

不得进入 world memory。

---

# 19. 必须实现的单元测试

## 19.1 No-cut parity

在没有 cut 时：

```text
V14 local Human3R output
≈
原版 Human3R output
```

camera、pointmap、SMPL-X 应在数值容差内一致。

---

## 19.2 Pre-decode reset

验证：

```text
first post-cut frame
```

在 recurrent decoder 前已经使用 fresh state。

---

## 19.3 Proposal branch 不写主状态

比较：

```text
启用 proposal branch
关闭 proposal branch
```

主 recurrent state 和 raw pose memory应保持一致。

还必须逐项比较：

```text
new_state_feat
pose memory
last_smpl_tk / last_smpl_id
raw pose/human token history
external identity memory commit count
```

shadow on/off 后，除 proposal diagnostic外都必须一致。

---

## 19.4 Boundary composition

验证：

\[
B^*=\Delta B B_0
\]

的矩阵方向和 c2w convention正确。

---

## 19.5 Shared transform

随机生成 camera、pointmap、root、joints、vertices，验证它们使用同一 B 后保持内部相对几何。

---

## 19.6 Detection permutation

交换 post-cut detection 顺序，identity 和最终 Boundary 不应依赖 detection index。

---

## 19.7 Wrong-ID rejection

构造错误身份 hypothesis，验证：

- residual magnitude；
- torso error；
- layout error；
- identity margin；

可以阻止错误 commit。

---

## 19.8 Zero-match fallback

没有可靠人物时，不得调用错误 human anchor。

---

## 19.9 No-future leakage

随机改变未来帧内容，当前 frame 输出必须完全不变。

---

## 19.10 Multi-cut composition

验证多次 cut 的 Boundary composition 不出现：

- 矩阵顺序反转；
- local/world gauge 混淆；
- 重复应用；
- double correction。

明确验证：

\[
B_{0,s}^{W}
=
B_{s-1}^{W}\Delta B_{0,s}^{s-1\leftarrow s}
\]

而不是把 `ΔB0` 直接误当成 world Boundary。

---

# 20. 必须进行的实验

## 20.1 Latent propagation 基础消融

比较：

| 方法 | first-frame latent | 后续 latent | 显式 Boundary |
|---|---:|---:|---:|
| Raw reset | 否 | 否 | 否 |
| One-shot latent only | 是 | 否 | 否 |
| Cached pose residual | 是 | 复用 | 否 |
| Cached pose+human residual | 是 | 复用 | 否 |
| V14 B0 propagation | 是 | 否 | 固定 B0 |
| V14 full | 是 | 否 | B0 + ΔB |

对 cached latent residual，计算每帧隐式诱导的：

\[
B_t^{\text{latent}}
=
\hat C_t(C_t^L)^{-1}
\]

报告：

\[
d(B_t^{\text{latent}}, B_0)
\]

用于证明：

```text
相同 latent residual 不对应固定 shot-level SE(3)。
```

---

## 20.2 Coarse Boundary 能力

比较：

```text
fresh reset
old V9 one-shot
new V14 event head
Fixed Explicit
GT coarse Boundary
```

报告：

- rotation mean/P90/P95；
- translation mean/P90/P95；
- coarse root distance；
- torso angle；
- camera-human consistency；
- catastrophic coarse failure；
- uncertainty calibration。

---

## 20.3 Identity rank

比较：

```text
identity-only
+ B0 root
+ B0 torso
+ B0 layout
+ B0 root/torso/layout
```

报告：

- Recall@1；
- Recall@2；
- Recall@K；
- assignment margin；
- accepted precision；
- accepted coverage；
- wrong accept；
- multi coverage；
- GT assignment 是否进入 top-K。

---

## 20.4 GT-ID fine alignment

固定正确身份，比较：

```text
V13 from-scratch Boundary
B0 only
B0 + V12 single residual
B0 + V13 multi residual
GT B0 + V13 residual
Fixed B0 + V13 residual
```

用于区分：

```text
B0 是否改善 WHERE optimization
```

和：

```text
B0 是否只帮助 WHO。
```

---

## 20.5 完整 automatic V14

比较：

```text
Human3R continue
Human3R reset
旧 V9 per-frame
V12 single
V13 current automatic identity
V14 B0 only
V14 + identity-only
V14 + coarse geometry identity
V14 + single/multi residual routing
V14 GT-ID upper bound
```

---

## 20.6 流式时间指标

分别报告：

```text
normal-frame latency
cut-frame extra latency
identity latency
residual solve latency
peak GPU memory
persistent CPU/GPU state size
```

不得只报告平均 FPS，而忽略 cut-frame latency。

---

# 21. 统一评价指标

必须至少报告：

## Camera

```text
translation error
rotation error
P90/P95
success rate
```

## Human

```text
root error
global orientation
MPJPE
PVE
torso error
```

## Scene

```text
pointmap/scene discontinuity
background alignment
```

## Identity

```text
IDF1
wrong accepted matches
accepted precision
accepted coverage
multi coverage
ID switches
top-K assignment recall
```

## Boundary

```text
coarse B0 error
residual ΔB magnitude
final B* error
candidate rotation dispersion
candidate translation dispersion
```

## Failure

```text
catastrophic failure rate
fallback usage
B0 invalid rate
single/multi/zero-match比例
```

## Streaming

```text
future leakage
normal-frame parity
memory upper bound
cut-frame compute
multi-cut accumulation
```

---

# 22. V14 的核心消融

必须至少包含：

```text
pose residual only
human residual only
pose + human residual
```

```text
proposal token写入 pose memory
vs
proposal-only，不写 pose memory
```

必须扩展为真正的结构消融：

```text
Inline-write one-shot V9：corrected decoder state和 pose memory提交
vs
Shadow proposal-only V9：完整 decoder refine，但所有 corrected state丢弃
```

只关闭 corrected pose-memory write而仍提交 corrected `new_state_feat` 不算有效消融。

```text
旧逐帧 gate
vs
one-shot event trigger
```

```text
latent cache
vs
explicit fixed B0
```

```text
first-frame loss only
vs
segment-level propagation loss
```

```text
无 shared consistency
vs
camera-human shared consistency
```

```text
B0 rotation-only
vs
B0 full SE(3)
```

```text
camera-pair-derived ΔB0
vs
camera + human analytic rigid projection
```

```text
identity-only
vs
geometry-assisted identity
```

```text
V13 full-from-scratch
vs
V13 residual-on-B0
```

```text
single only
vs
multi only
vs
single/multi safe routing
```

---

# 23. V14 的成功标准

V14 不能只以某一个平均 camera 指标改善作为成功。

至少需要满足：

1. no-cut 路径不破坏原 Human3R；
2. first post-cut frame的 raw main rollout确实在 reset 后解码；
3. latent correction只运行一次；
4. corrected proposal不写入主 local state；
   其中包括 decoder `new_state_feat`、pose memory、tracking和 correction history；
5. B0 对整个 shot有效，而不只对第一帧有效；
6. B0 能降低大范围 rotation/gauge error；
7. B0 能提高正确身份的 rank 或 margin；
8. automatic identity wrong-accept 不增加；
9. 多人可靠时，V13 residual 能兑现多人收益；
10. 单人时能够稳定退化到 V12-style residual；
11. 零匹配时不会进入灾难性的错误 human fallback；
12. camera、scene、所有 humans 使用同一个 fixed Boundary；
13. 后续帧不重新估计 Boundary；
14. 当前帧结果不依赖未来帧；
15. 内存与计算不会随视频长度无限增加；
16. cut-frame latency 有明确审计；
17. 在 held-out capture/camera/subject 上复现；
18. scene trade-off 必须如实报告；
19. 多 cut 漂移必须单独报告；
20. 不得将 GT-ID 结果描述为可部署 automatic system。

---

# 24. V14 最终论文叙事

V14 的方法故事应统一成：

> Camera cut is not a per-frame correction problem. It is a discrete state transition followed by a shot-level gauge transition. We therefore keep one reset shot-local Human3R trajectory, run one full V9-style correction rollout as a non-committing shadow transaction at the first post-cut frame, and project its implicit refinement into an explicit coarse gauge bridge. The coarse alignment facilitates causal identity association, after which a small identity-conditioned multi-human residual is estimated. A single fixed Boundary is applied to the camera, scene and all humans throughout the new shot.

中文概括：

> Camera cut 不是逐帧 refinement 问题，而是一次离散的状态切换和 shot-level 世界坐标切换。V14 只提交一条 hard-reset 后的原版 Human3R shot-local trajectory；同时在第一张 post-cut frame使用 pre-cut raw state执行一次完整但不提交的 V9 shadow correction，并将隐式 refine投影成显式粗 gauge bridge。粗 world geometry用于辅助流式人物身份关联，可靠身份再提供多人显式 residual Boundary。整个新 shot 的 camera、scene 和所有人物始终共享同一个固定 Boundary。

V14 的核心贡献可以组织为：

```text
1. One-shot latent coarse re-anchoring
2. Full V9-style shadow correction without recurrent-state contamination
3. Coarse-gauge-assisted causal identity association
4. Identity-conditioned multi-human residual Boundary refinement
5. Single/multi/zero-match safe routing
6. One fixed shared Boundary under strict streaming constraints
```

---

# 25. 实施顺序

不要一开始直接实现完整系统。

严格按照下面顺序工作。

## Step 1：代码和坐标系审计

先输出：

```text
V14_CODE_AUDIT.md
```

内容包括：

- Human3R recurrent state写入位置；
- pose memory写入位置；
- V9 latent correction位置；
- raw/corrected token流向；
- camera c2w/w2c convention；
- human root coordinate convention；
- V12 Boundary应用方式；
- V13 identity tentative/commit路径；
- 可以复用和必须重构的函数；
- 所有潜在 double correction 风险。

在该审计完成前，不要大规模改代码。

## Step 2：实现 raw main / V9 shadow 双 rollout分流

先保证：

```text
共享一次 encoder
raw branch从 fresh state运行且只提交 raw state
shadow branch从 detached pre-cut state运行完整 V9 refine
V14 corrected proposal不会修改主 state / pose memory / tracking。
```

完成对应单元测试。

## Step 3：实现 frozen V9 one-shot diagnostic

先验证：

```text
旧 V9 是否能产生有用 B0。
```

## Step 4：实现显式 B0 整段传播

比较：

```text
latent cache
vs
fixed explicit Boundary。
```

## Step 5 / V14.1：训练 first-post-cut V9 shadow correction

加入：

- one-shot cut event trigger；
- full correct-token decoder interaction；
- shadow-only pose/human corrected heads和 LoRA；
- explicit `ΔB0 = C_corr inv(C_raw)` loss；
- segment-level loss；
- shared consistency；
- uncertainty。

## Step 6：扩展 causal identity matcher

加入 B0-aligned geometry，但保留 identity-only 对照。

## Step 7：实现 residual V13 solver 和 V12 fallback

最终形成：

```text
multi / single / zero-match
```

安全路由。

## Step 8：完整 streaming audit

验证：

- no future；
- bounded memory；
- fixed Boundary；
- no retroactive update；
- no GT leakage；
- multi-cut；
- runtime。

---

# 26. 最终代码交付要求

完成每个阶段时必须给出：

1. 新增和修改的文件；
2. 每个文件的职责；
3. 关键数据结构；
4. 状态读取和写入路径；
5. coordinate convention；
6. 运行命令；
7. 测试命令；
8. 实验配置；
9. 输出目录；
10. 结果 JSON schema；
11. 当前通过的测试；
12. 尚未解决的问题；
13. 与 V9/V12/V13 的公平对照；
14. 是否使用 GT；
15. 是否存在未来帧泄漏；
16. 是否真正是逐帧 streaming。

不要只给出概念性代码。

不要隐藏失败实验。

不要把 GT-ID、GT camera 或未来帧结果误写成可部署推理结果。

不要修改 V9/V12/V13 已冻结的 checkpoint、manifest、tag 和正式结论。

---

# 27. 最终必须始终遵守的原则

```text
第一，camera cut 时 raw main branch必须先 reset，再 decode；只有不提交的 shadow
branch可以读取 pre-cut raw state。

第二，latent correction 只在第一张 post-cut frame执行一次。

第二点补充，correction必须在完整 shadow decoder中 refine，但 shadow state不得提交。

第三，第一帧 latent residual 不直接复制给后续帧。

第四，传播给整个 shot 的是显式 Boundary，不是 latent residual。

第五，V9-style branch只负责 coarse proposal，不负责最终精确 alignment。

第六，V13-style branch只估计 B0 上的小 residual，而不是从头求大范围 Boundary。

第七，Identity 只回答 WHO，Geometry 只回答 WHERE。

第八，wrong identity 宁可拒绝，也不能进入 shared Boundary。

第九，多身份不可靠时必须退化到可靠单人；无可靠人物时使用 B0 或安全 fallback。

第十，camera、scene、所有 humans 始终共享一个固定 Boundary。

第十一，raw local Human3R state保持 shot-local，world gauge由外部 Boundary管理。

第十二，任何 identity/world memory都必须 Align-Then-Commit。

第十三，不使用未来帧，不进行全局 BA，不回改历史输出。

第十四，所有内存和计算必须是固定预算。

第十五，正常帧不运行额外 latent correction或 Boundary solver。
```

最终 V14 应当形成下面这一条清晰且不可混淆的主线：

```text
Snapshot Pre-Cut Raw State
→ Shared Post-Cut Encoding
→ Hard-Reset Raw Main Rollout + Commit Raw State
→ One-Shot Full V9 Shadow Rollout + Discard Shadow State
→ Relative Bridge ΔB0 = Ccorr inv(Craw)
→ World Coarse Boundary B0 = Gprev ΔB0
→ Coarse-Gauge-Assisted WHO
→ Identity-Conditioned Residual WHERE
→ Final B* = ΔB B0
→ One Fixed Shared Boundary
→ Align-Then-Commit
```
