# ShotToken 近期训练与测试汇报

## 1. 总体结论

总的来说，近期实验说明：ShotToken 这个方向不是完全错的，但早期版本的问题在于它的权限太大、语义太泛，导致它不只是修 camera pose，还会干扰 pointmap、background reconstruction 和 human branch。

目前可以把结论概括成三句话：

```text
V2：ShotToken 作为 global token 直接进 decoder，能力太大，污染严重。
V4：ShotToken 只服务 pose/camera，方向正确，但修正幅度有限。
V5/V6：后续应该进一步把 ShotToken 限制成 pose-only 或 local scene evidence，而不是全局遥控器。
```

## 2. V2 版本

### 2.1 模型设计

V2 的核心想法是：让模型自己通过一个全局 ShotToken 学会镜头跳变后的对齐关系。

具体做法是：

```text
1. 用当前帧和上一帧的 decoder image token 生成一个 q_t。
2. 把 q_t 当作普通 token append 到 decoder token 序列中。
3. decoder 里的 pose token、image token、human token 都可以和 q_t 做 full attention。
4. 额外使用 PoseLoRA / HumanLoRA / WorldLoRA 修正 camera、human、pointmap。
```

这套设计当时的直觉是：镜头跳变会同时影响 camera、world 和人体位置，所以让 ShotToken 进入 decoder，让所有分支一起读取它，理论上可以学到全局对齐。

### 2.2 训练与测试结果

V2 在训练指标上看起来是能学的，比如 shot classification 相关 loss 会下降，`shot_acc` 也会上升。

但实际 demo 和训练集内 A5B5 视频测试都失败了。典型现象是：

```text
1. 关闭 Shot Adaptation 时，背景和人物重建基本正常，只是 B 段相机整体有偏移。
2. 打开 Shot Adaptation 后，背景 / pointmap 明显变差。
3. 相机位置像被压到第一帧附近，camera pose 不稳定。
4. 尺度也会出现异常。
```

我们专门构造了训练集内连续时间相机切换视频：

```text
A 段：/workspace/data/Avatarrex/avatarrex_zzr_output/Training/22010708/rgb/00000300.png - 00000304.png
B 段：/workspace/data/Avatarrex/avatarrex_zzr_output/Training/22010710/rgb/00000305.png - 00000309.png
```

这个视频不是训练集外泛化测试，而是训练分布内的 A5B5 相机切换测试。V2 在这个视频上仍然失败，说明问题主要不在数据分布，而在结构设计。

### 2.3 失败原因

V2 的主要问题是 ShotToken 太像一个“全局遥控器”。

它没有明确空间位置，也没有明确局部对象，却可以被所有 token 读取：

```text
pose token 可以读它；
image token 可以读它；
human token 可以读它；
pointmap 分支间接受它影响；
recurrent state 也可能受到它影响。
```

结果是，模型没有只学到“怎么修 camera pose”，而是把 ShotToken 当作一个全局扰动信号，影响了原本稳定的 image token 表达。

所以 V2 的失败可以总结为：

```text
ShotToken 的语义不是完全错，错在它进入 decoder 后权限太大。
它本来应该服务 shot boundary 和 camera alignment，结果却污染了 reconstruction 和 human branch。
```

## 3. V4 版本

### 3.1 模型设计

V4 的核心思路是：不再让 ShotToken 影响所有 token，只让它服务 camera / pose 分支。

具体做法是：

```text
1. ShotTokenGenerator 仍然生成 q_t。
2. q_t 不再作为普通 decoder token 进入 full attention。
3. decoder 正常输出 pose token z_out。
4. 通过 PoseAlignmentAdapter，让 pose token 和 q_t 做受限 attention。
5. Adapter 只修正最终 camera pose，不直接修改 image token、human token 或 pointmap。
```

通俗来说，V4 不让 ShotToken 去“改画面”，只让它给 camera pose 一个对齐提示。

### 3.2 训练与测试结果

V4 30 epoch 权重目录是：

```text
/workspace/code/Movie3R/experiments/training-4gpu-bz24-30ep-shot-v4-20260509-024512
```

我们用同一个 A5B5 视频，对比了关闭和开启 Shot Adaptation 后的 camera pose，并用 AvatarReX GT camera pose 做数值评估。

评估方式是：

```text
1. 读取 GT camera pose。
2. 读取 demo 输出的 predicted camera pose。
3. 都统一到第 0 帧坐标系：T_rel[i] = inv(T[0]) @ T[i]。
4. 比较每一帧 camera translation error。
```

结果如下：

| 指标 | Shot Off | Shot On | 变化 |
|---|---:|---:|---:|
| A 段平均 translation error | 0.0542 | 0.0555 | 基本不变 |
| B 段平均 translation error | 2.7805 | 2.4469 | 改善约 12% |
| 全段平均 translation error | 1.4174 | 1.2512 | 改善约 12% |
| 跳变边界 4->5 translation error | 2.1122 | 1.8516 | 改善约 12% |

这里的 `translation error` 是三维平移向量误差的 L2 norm，不是单独某一个方向的误差。因为 `T_rel[i] = inv(T[0]) @ T[i]`，所以平移向量都表达在第 0 帧相机坐标系下。

坐标方向按 Human3R / AvatarReX 当前 camera convention 理解为：

```text
x: 第 0 帧相机图像向右
y: 第 0 帧相机图像向下
z: 第 0 帧相机光轴向前
```

V4 per-axis 误差补充如下。表中 `mean |dx|/|dy|/|dz|` 是逐帧绝对误差均值，单位与 camera translation 一致。

| 区间 | 模式 | mean abs dx | mean abs dy | mean abs dz | mean L2 |
|---|---|---:|---:|---:|---:|
| A 段 0-4 | Shot Off | 0.0045 | 0.0013 | 0.0539 | 0.0542 |
| A 段 0-4 | Shot On | 0.0047 | 0.0030 | 0.0549 | 0.0555 |
| B 段 5-9 | Shot Off | 1.9021 | 1.7787 | 0.8844 | 2.7805 |
| B 段 5-9 | Shot On | 1.6530 | 1.5290 | 0.8704 | 2.4469 |
| 全段 0-9 | Shot Off | 0.9533 | 0.8900 | 0.4692 | 1.4174 |
| 全段 0-9 | Shot On | 0.8288 | 0.7660 | 0.4627 | 1.2512 |

跳变边界 `4 -> 5` 的 translation vector 对比如下：

| 项 | x | y | z | L2 |
|---|---:|---:|---:|---:|
| GT jump | 0.2666 | -1.8634 | -0.2356 | - |
| Shot Off jump | -0.7003 | -0.0188 | 0.1266 | - |
| Shot On jump | -0.4604 | -0.2726 | 0.3751 | - |
| Shot Off jump error | 0.9669 | 1.8445 | 0.3621 | 2.1138 |
| Shot On jump error | 0.7269 | 1.5908 | 0.6107 | 1.8526 |

从 per-axis 看，V4 的主要改善来自 `x` 和 `y` 方向，尤其 `y` 方向的 jump 从 `-0.02` 被拉到 `-0.27`，更接近 GT 的 `-1.86`。但 `z` 方向反而变差一些，因此 V4 只是把 camera jump 往正确方向拉了一点，并没有真正学会完整的跨镜头对齐。

也就是说，V4 确实把 B 段 camera pose 往正确方向拉了一点，但改善幅度不大，肉眼上不一定明显。

### 3.3 结果猜想

V4 的方向是正确的，因为它基本避免了 V2 的严重背景污染，同时 camera pose 数值上有一定改善。

但 V4 的能力仍然有限，原因可能有三点：

```text
1. 它是在 decoder 完成之后再做 camera pose correction，介入时间比较晚。
2. q_t 仍然是 global token，信息比较粗，不知道具体哪个局部背景点应该对齐。
3. PoseAlignmentAdapter 的 correction 幅度有限，可能只能做小修，无法处理 A->B 这种较大的相机跳变。
```

从 A5B5 数值看，真正的 GT jump 在 y 方向变化很大：

```text
GT jump y ≈ -1.86
Shot Off jump y ≈ -0.02
Shot On  jump y ≈ -0.27
```

所以 V4 学到了一点“应该往正确方向修”，但还远远没有学到完整的跨镜头 camera alignment。

## 4. V5 版本设计

### 4.1 设计目标

V5 的目标是：保持 V4 的安全性，但让 ShotToken 更早参与 pose token 的形成过程。

V4 是 decoder 结束后再修 camera pose，像一个后处理 correction。V5 希望把 ShotToken 推进到 decoder 内部，但仍然不让它污染 image / human / pointmap。

### 4.2 V5.1 模型设计

V5.1 的做法是 layerwise pose-only attention。

结构可以理解成：

```text
decoder 正常处理 [pose token, image token, human token]
每一层 decoder block 结束后：
    pose token 作为 query
    [pose token, ShotToken] 作为 key/value
    只更新 pose token
```

这样做的好处是：

```text
1. ShotToken 不再是 decoder 后处理，而是更早影响 pose token。
2. image token 和 human token 仍然不能直接读取 ShotToken。
3. pointmap / background reconstruction 理论上不会被 ShotToken 直接污染。
4. camera pose 的修正更像是逐层推理出来的，而不是最后硬加 correction。
```

### 4.3 V5.1 初步结果

V5.1 5 epoch 训练已经完成，但实际 demo 效果不理想。

主观观察是：

```text
背景仍然错；
camera pose 也错；
整体效果比预期差。
```

这说明，虽然 V5.1 在设计上限制了 ShotToken 的访问范围，但只要它逐层改变 pose token，仍然可能通过 pose token 间接影响后续几何输出；同时 global q_t 本身信息仍然太粗，不能提供可靠的局部对齐证据。

### 4.4 V5.2 后备设计

V5.2 的想法是 masked decoder。

也就是：

```text
ShotToken 可以进入 decoder token 序列，
但 attention mask 限制它只能和 pose token 交互，
不能被 image token / human token 直接读取。
```

目标关系是：

```text
ShotToken <-> pose token: yes
ShotToken <-> image token: no
ShotToken <-> human token: no
```

这个方案可以更严格地控制“谁能看谁”，但当前 decoder attention 还不支持 mask，需要改底层 attention 调用链，所以工程量更大。

## 5. V6 版本设计

### 5.1 设计目标

V6 的目标是解决 ShotToken 的语义问题。

V2 到 V5 主要都在处理“ShotToken 能影响谁”的问题，但另一个核心问题是：global ShotToken 本身太泛了。

它现在表达的是：

```text
当前帧和上一帧整体差异很大，可能发生了 shot change。
```

但它没有告诉模型：

```text
当前帧哪个背景点对应旧世界里的哪个背景点。
```

所以 V6 希望把 global ShotToken 改成 local scene re-anchor tokens。

### 5.2 V6 模型设计

V6 不再只用一个全局 q_t，而是生成多个局部 scene anchor tokens：

```text
R_t = {R_1, R_2, ..., R_K}
```

每个 `R_k` 表达的是：

```text
当前帧某个静态背景 patch，
对应历史帧 / keyframe 里的某个静态背景 patch，
这个匹配有多可靠。
```

这更像 Human3R 的 human token：

```text
Human3R: head patch -> human prompt -> 恢复这个具体的人
V6: static background patch -> re-anchor prompt -> 帮助 camera/world 对齐
```

### 5.3 为什么 V6 更合理

V6 的核心变化是：ShotToken 不再直接下命令，而是提供局部证据。

原来的 global ShotToken 像是在说：

```text
镜头跳了，整体相机应该调整。
```

V6 的 local anchor tokens 则是在说：

```text
当前帧这个墙角，对应旧帧里的那个墙角。
当前帧这个门框，对应旧世界里的那个门框。
当前帧这块地面，对应历史帧里的那块地面。
```

这样 camera / pose token 可以根据多个局部证据自己推理 camera correction，而不是依赖一个无位置的全局控制 token。

### 5.4 V6 推荐实现路线

V6 建议分三步做：

```text
V6.0: local anchor tokens + pose-only adapter
V6.1: 引入冻结背景 / 匹配 encoder 辅助 anchor 选择
V6.2: masked decoder with anchor prompts
```

其中 V6.1 可以参考 Human3R 引入冻结 Multi-HMR encoder 的思路。Human3R 用专门的人体 encoder 提取人体相关 prompt，V6 也可以引入冻结的背景 / 匹配 encoder 来选择静态场景 anchor。

候选模块包括：

```text
DINOv2 / MAE：通用视觉特征
SAM：前景 / 背景区域分割
SuperPoint + LightGlue：局部特征匹配
DUSt3R / MASt3R：几何匹配和跨视角对应
Depth Anything：辅助过滤低质量区域和深度异常区域
```

### 5.5 V6 的主要困难

V6 最大难点不是 adapter，而是 anchor 是否可靠。

理想 anchor 应该满足：

```text
是静态背景；
不在人身上；
不是动态物体；
纹理足够明显；
当前帧和历史帧都有可见对应；
match confidence 足够高。
```

如果 anchor 选错，比如选到人体、遮挡区域、低纹理墙面，camera token 可能会被错误证据拉偏。

所以 V6 需要额外考虑：

```text
anchor proposal 怎么做；
匹配置信度怎么估计；
低 overlap shot change 时是否应该关闭 correction；
attention mask 是否必须加；
如何保证 pointmap / human branch 不再被污染。
```

## 6. 最终建议

当前最清晰的路线是：先承认 V4 是一个方向正确但能力有限的版本，再把后续重点放到 V6。

更具体地说：

```text
1. V2 证明了 global ShotToken full attention 风险很大。
2. V4 证明了 pose-only 方向更安全，并且 camera pose 有小幅改善。
3. V5.1 说明仅仅把 global q_t 提前注入 pose token 仍然不够稳。
4. V6 应该把 ShotToken 从 global command token 改成 local scene evidence tokens。
```

最终目标不是让 ShotToken 直接告诉模型“相机应该怎么变”，而是让它提供更可靠的局部重定位证据：

```text
当前帧这些静态背景 patch，应该接回旧世界里的这些位置。
```

这样更安全、更可解释，也更适合作为后续论文方法。
