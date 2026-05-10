# ShotToken V6 Plan: Local Scene Re-Anchor Tokens

## 1. 核心目标

V6 的目标是把现在的 global ShotToken：

```text
一个无空间锚点的全局控制 token
```

改成：

```text
多个有明确位置、有明确匹配关系、有明确监督的 local scene re-anchor tokens
```

原来的 global ShotToken 更像在告诉模型：

```text
镜头跳了，整体相机应该调整。
```

V6 希望改成：

```text
当前帧这些静态背景 patch，应该接回历史帧 / keyframe 里的这些位置。
```

也就是说，ShotToken 不再直接像“全局遥控器”一样控制 camera / world，而是提供局部重定位证据，让 camera / pose token 根据这些证据自己推理对齐关系。

## 2. 为什么要改

之前实验说明，global ShotToken 如果直接进入 decoder full attention，会污染很多分支：

```text
image token
pointmap
background reconstruction
human token
camera pose
recurrent state
```

主要原因是它没有明确空间位置，也没有明确局部对象。它表达的是全局变化，但 decoder 里所有 token 都可能读取它，于是它很容易变成一个不受控的全局控制信号。

Human3R 的 human token 能进入 decoder，是因为它有明确锚点和明确任务：

```text
head patch -> human prompt -> 恢复这个具体的人
```

所以 V6 希望 ShotToken 也变成类似形式：

```text
static background patch / matched patch -> re-anchor prompt -> 帮助 camera/world 对齐
```

## 3. Token 形式

V6 不再使用单个：

```text
q_t = ShotGenerator(F_{t-1}, F_t)
```

而是生成 K 个局部 scene anchor tokens：

```text
R_t = {R_1, R_2, ..., R_K}
```

每个 anchor token 表达：

```text
当前帧某个静态背景 patch
对应历史帧 / keyframe / memory 里的某个静态背景 patch
这个匹配有多可靠
```

可以写成：

```text
R_k = Projection(
    F_cur[u_k],
    F_ref[v_k],
    PE_cur[u_k],
    PE_ref[v_k],
    match_confidence_k,
    optional depth / 3D / visibility info
)
```

这里的 `R_k` 不直接携带完整 SE(3) camera correction。它只提供局部匹配证据，例如：

```text
当前帧这个墙角 patch，对应旧帧里的那个墙角。
当前帧这个门框 patch，对应旧世界里的那个门框。
当前帧这块地面 patch，对应历史帧里的那块地面。
```

camera / pose token 再根据多个 anchor evidence 共同推理相机应该如何重新对齐。

## 4. 职责拆分

V6 不让一个 token 同时负责所有事情，而是拆成三类模块。

```text
1. Global transition gate
2. Local scene anchor tokens
3. Camera / pose token
```

Global transition gate 只负责：

```text
是否发生 shot change
当前是否 low-overlap
是否需要启用 correction
correction gate 应该多大
```

Local scene anchor tokens 负责：

```text
当前 patch 对应哪个历史 patch
这个 patch 是否是静态背景
这个匹配是否可靠
当前帧和参考帧是否有有效 overlap
```

Camera / pose token 负责：

```text
汇总 anchor evidence
推理 refined camera pose
完成 world re-anchor
```

这样可以避免一个 global token 同时承担“判断跳变、表达跳变、修 pose、更新 world、更新 state”等过多职责。

## 5. V6.0 安全版本

V6.0 建议先不要让 anchor tokens 进入主 decoder sequence。

采用更安全的 pose-only adapter：

```text
z_l = z_l + gamma_l * g_t * Adapter(z_l, R_t)
```

其中：

```text
z_l: 第 l 层 decoder 后的 pose token
g_t: global transition gate
R_t: local scene anchor tokens
```

这个版本的约束是：

```text
scene anchor tokens 只影响 pose / camera token
image token 不直接 attend anchor tokens
human token 不直接 attend anchor tokens
pointmap head 不直接吃 anchor tokens
```

这样最接近当前 V5.1 的结构，也最容易验证 anchor evidence 是否真的帮助 camera alignment，同时最大程度保护 pointmap / background / human branch。

## 6. V6.1 冻结背景 Encoder

Human3R 额外引入了冻结的 Multi-HMR encoder 来提取人体相关 token。V6 也可以类似引入一个冻结的背景 / 匹配 encoder，用来提供更可靠的静态场景 anchor。

候选方向包括：

```text
DINOv2 / MAE 类通用视觉特征
SAM / semantic segmentation 类背景区域提取
SuperPoint + LightGlue 类局部匹配
DUSt3R / MASt3R 类几何匹配特征
Depth Anything / monocular depth 辅助过滤动态和低质区域
```

这个 encoder 初期可以冻结，不参与训练，只作为 anchor proposal 和 matching 的辅助来源。

它的作用是：

```text
帮助选择可靠背景 patch
帮助排除人体和动态区域
帮助判断 patch match 是否可信
提供比主模型 image token 更稳定的局部匹配特征
```

这和 Human3R 使用专门人体 encoder 的思路类似：主模型不一定自己承担所有感知任务，可以借助冻结专家模型产生更稳定的 prompt / anchor。

## 7. V6.2 Masked Decoder

如果 V6.0 / V6.1 证明 anchor tokens 有效，再考虑让 anchor tokens 进入 decoder。

但必须加 attention mask：

```text
anchor token <-> pose token: yes
anchor token <-> image token: no
anchor token <-> human token: no
```

否则即使 token 是 local anchor，也仍然可能污染 reconstruction。

V6.2 的目标是同时做到两点：

```text
token 语义更精准：local scene evidence
token 权限更受控：只服务 camera / pose branch
```

这会比 V6.0 工程量更大，因为当前 decoder attention 还不支持 mask，需要改底层 attention 调用链。

## 8. 主要困难

### 8.1 Anchor 选什么

这是 V6 最大难点。

理想 anchor 应该是：

```text
静态背景
纹理足够明显
不在人体 mask 内
不在动态物体区域
不是天空 / 纯色墙 / 低纹理地面
在当前帧和历史帧都有可见对应
match confidence 高
```

错误 anchor 会很危险，因为它会给 camera token 提供错误几何证据。

### 8.2 Shot Change 未必有 Overlap

电影镜头切换时可能出现：

```text
A2 和 B1 几乎没有重叠背景
相机视角完全不同
背景遮挡严重
```

这种情况下不能强行 re-anchor，否则会把 camera pose 拉错。

因此 V6 需要 overlap / validity 判断：

```text
有可靠 overlap -> 启用 anchor correction
没有可靠 overlap -> gate 变小或关闭
```

### 8.3 Mask 不一定可靠

如果人体区域或动态区域没有排干净，anchor 可能选到人身上。

这会造成：

```text
把动态人体当作静态世界点
camera token 根据错误点对齐
pose 更乱
```

所以需要可靠的人体 mask、动态区域过滤，或者 anchor inlier 判断。

### 8.4 匹配监督不直接可用

即使数据集有 GT camera pose，也不一定直接有 patch-level correspondence 标签。

实现时需要考虑：

```text
patch-level GT correspondence 不一定直接可用
depth / pointmap 可能有噪声
不同相机视角下同一背景 patch 不一定容易匹配
low-overlap 时可能根本没有可靠匹配
```

初期可以用 weak supervision 和几何一致性约束辅助训练。

### 8.5 工程成本更高

V6 比 V5.1 大很多，涉及：

```text
anchor proposal
anchor matching
frozen encoder 接入
anchor token projection
pose adapter
可能的 attention mask
新的监督和日志
```

因此建议分阶段做，不要一次性全部加上。

## 9. 推荐路线

建议按下面顺序推进：

```text
V6.0: local anchor tokens + pose-only adapter
V6.1: frozen background / matching encoder 辅助 anchor 选择
V6.2: masked decoder with anchor prompts
```

这样可以逐步回答三个问题：

```text
局部 anchor evidence 是否比 global ShotToken 更有用？
冻结专家 encoder 是否能提高 anchor 可靠性？
anchor tokens 是否可以安全进入 decoder attention？
```

## 10. 最终目标

V6 的最终目标是把 ShotToken 从 global command token：

```text
镜头跳了，整体应该这样变。
```

改成 local scene evidence tokens：

```text
当前帧这些静态背景 patch，对应旧世界里的这些位置。
```

这样做的好处是：

```text
1. token 有明确来源和空间锚点。
2. token 有明确任务和监督。
3. camera correction 来自局部证据，而不是全局命令。
4. image / human / pointmap 更不容易被污染。
5. 方法更容易解释，也更适合作为论文方法。
```
