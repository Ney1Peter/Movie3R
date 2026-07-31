# Movie3R V14：多人单目 Multi-Shot 视频处理流程通俗说明

日期：2026-07-29

用途：用尽量通俗的语言解释，一段多人、单目、包含多个 camera cuts 的视频进入
Movie3R V14 后，会依次经过哪些模块。文末同时解释主要专业名词。

可以把最终 Movie3R V14 理解为：

> Human3R 负责重建每个镜头内部，V14 负责在 camera cut 时把不同镜头重新接回同一个
> 世界。

---

## 1. 逐帧读取和预处理

视频按照时间顺序逐帧输入，不会一次读取完整视频，也不读取未来帧。

每张 RGB 图像先做 Human3R 标准预处理，例如：

- 调整图像尺寸；
- 中心裁剪；
- 像素归一化。

这里的“单目”指系统每个时刻只接收一张普通 RGB 图像，不需要同步多相机输入。

---

## 2. Encoder 提取图像和人物特征

图像进入 Human3R encoder。

Encoder 不直接输出三维结果，而是把图像转换成一组高维特征，也就是 tokens：

- image tokens：场景、纹理、物体和空间结构信息；
- pose token：用于表示当前相机状态；
- human tokens：每个检测人物对应的人体特征；
- scene features：用于预测 pointmap 和深度。

Token 可以理解为模型内部使用的“压缩信息单元”。

Human3R 同时检测画面里有几个人，并为每个人构造 human token。部署时使用的是模型
预测的人体位置，不使用 GT bbox 或 GT identity。

---

## 3. 判断当前帧是否发生 Camera Cut

系统需要判断当前帧是否从一个镜头突然切到了另一个镜头。

例如：

```text
上一帧：正面相机
当前帧：突然切到背面相机
```

最终系统会使用一个轻量 cut detector，根据相邻 RGB 或冻结的图像特征判断是否发生
镜头切换。

需要明确：**自动 detector 目前还没有正式实现**。现有实验是直接告诉模型 cut 在哪一帧
发生。

这个判断必须在 recurrent decoder 写入当前帧状态之前完成。

---

## 4. 如果没有 Camera Cut

如果当前帧属于同一个连续镜头，就正常运行原版 Human3R。

Decoder 接收：

- 当前 image tokens；
- 当前 pose token；
- 当前 human tokens；
- 上一帧 recurrent state。

Recurrent state 可以理解为模型对当前镜头历史的内部记忆，包括此前看到的场景、相机和
人体信息。

这些 tokens 在 decoder 中通过 attention 相互交换信息，然后由不同 head 输出：

- Camera Head：相机位置和旋转；
- Scene Head：三维 pointmap；
- Human Head：SMPL-X 人体、root、joints、vertices；
- Native Tracker：同一个镜头内的人物编号。

最后，再应用当前镜头已经确定的固定 Boundary，把局部结果转换到跨镜头的统一世界
坐标。

正常帧不会运行额外的 shadow correction，也不会重新计算 Boundary。

---

## 5. 如果检测到 Camera Cut

Camera cut 发生时，V14 不会让新图像直接继续使用旧 recurrent state。

因为旧 state 记录的是上一个镜头，直接继续使用会污染新镜头。

系统会暂时保存上一镜头的状态，但只允许读取，不能修改。然后针对第一张 post-cut
图像运行两条不同分支：

1. Raw-reset branch；
2. Shadow correction branch。

这两条分支处理的是同一张 post-cut 图像，但职责不同。

---

## 6. Raw-Reset Branch：获得干净的新镜头重建

Raw branch 会先执行 Hard Reset。

Hard Reset 的意思是：

> 清空上一镜头的 Human3R recurrent state，然后把当前 post-cut 图像当作新镜头第一帧
> 重新解码。

它输出：

- 新镜头局部坐标下的 camera；
- pointmap；
- 多个 SMPL-X 人体；
- 人体 root、torso、joints；
- 一条干净的新 recurrent state。

这条 raw-reset state 是后续真正保留和继续使用的状态。

但它存在一个问题：虽然重建是干净的，却位于一个新的局部坐标系里，和上一镜头的
世界坐标没有连接。

---

## 7. Shadow Branch：利用旧信息猜测粗对齐

Shadow branch 会读取上一镜头的状态，并处理第一张 post-cut 图像。

它会显式构造 V9 风格的 correct tokens：

- Semantic Token：当前图像、人物和旧状态在语义上是否一致；
- Alignment Token：当前相机 latent 相对旧状态发生了什么变化；
- Momentum Token：参考此前 correction 的变化趋势。

这些 correct tokens 和 image、pose、human tokens 一起进入完整 decoder，通过 attention
进行联合 refine。

Decoder 后有两个 correction head：

- Camera correction head：输出 pose-token latent residual；
- Human correction head：输出 human-token latent residual。

它们不是直接预测最终 4x4 SE(3)，而是先纠正模型内部 latent，再由原来的
camera/human head 输出显式相机和人体结果。

这一分支得到一个“参考旧镜头后纠正过的 post-cut camera”。

最关键的是：

> Shadow branch 的输出可以读取，但 shadow recurrent state 永远不会提交给后续帧。

它只是一次临时计算，完成后立即丢弃。

---

## 8. 从 Raw 和 Shadow 结果提取粗 Boundary `B0`

现在同一张 post-cut 图像有两个 camera：

- `C_raw`：Hard Reset 后的新镜头局部 camera；
- `C_shadow`：利用旧镜头状态纠正后的 camera。

于是可以计算：

```text
B0 = C_shadow @ inverse(C_raw)
```

`B0` 是一个显式 SE(3) 变换，包括：

- 三维旋转；
- 三维平移。

它表示：

> 应该怎样移动 raw-reset 的新镜头坐标系，才能大致接回上一镜头的世界坐标。

这里的 gauge 就是“整个重建使用的世界坐标系”。Hard Reset 会产生新的 gauge，`B0`
负责连接新旧 gauge。

---

## 9. 使用 `B0` 后再判断人物身份

Post-cut 检测到的人一开始都是匿名的。

例如 cut 前有：

```text
人物 A
人物 B
人物 C
```

cut 后模型只知道：

```text
detection 0
detection 1
detection 2
```

直接比较 cut 前后人体位置通常会失败，因为两个镜头根本不在同一个坐标系中。

因此先使用 `B0` 把 post-cut 人体映射到旧世界，再比较：

- root 位置；
- torso 朝向；
- 可选的人体局部结构。

然后使用 Hungarian assignment 找到一对一对应关系。

Hungarian 是一种求最小总代价匹配的标准算法。例如它会判断：

```text
人物 A -> detection 2
人物 B -> detection 0
人物 C -> detection 1
```

这就是 `B0-before-WHO`：

> 先粗略解决“在哪里”，再判断“谁是谁”。

当前实现已经有 `B0 + root/torso + Hungarian`，但暂时只完整验证了 cut 前后可见人物
集合相同的情况。人物进入、离开和 dustbin 尚未完整接入。

---

## 10. 每个人独立产生几何修正候选

如果人物身份匹配可靠，每个人都可以根据 cut 前历史和 post-cut 重建产生一个 Boundary
candidate：

```text
B_i = [R_i, t_i]
```

主要使用：

- cut 前 root motion；
- cut 后 root；
- torso 朝向变化；
- Fixed Explicit 初始化；
- pointmap 局部 refinement；
- V16 torso residual；
- 统一 20 degree rotation bound。

每个人只提供一个几何候选，不拥有独立世界坐标，也不会单独移动。

---

## 11. 在 `B0` 附近做多人精修

当前旧实现会直接用多人候选重新计算完整 Boundary，但实验发现这可能破坏已经较准的
`B0` translation。

下一步计划改成：

> 不重新估计完整 SE(3)，只允许多人在 `B0` 附近提供一个小 residual。

需要比较：

- `B0` only；
- `B0 + rotation residual`；
- `B0 + translation residual`；
- `B0 + bounded full residual`。

如果多人精修不能在保持 camera 安全的同时改善 human/layout，就不使用精修，直接把
`B0` 作为最终 Boundary。

这一步属于下一阶段计划，当前 operational runner 仍使用旧的完整多人 Boundary 覆盖
`B0`。

---

## 12. 根据可靠人物数量选择结果

最终采用安全 fallback：

- 至少 2 个可靠身份：尝试多人 Boundary refinement；
- 只有 1 个可靠身份：使用单人几何修正或保守 `B0`；
- 没有可靠身份：直接使用 `B0`。

不会为了启用多人模式而强制匹配人物。

这套完整人数变化 fallback 尚未全部接入当前 runtime。

---

## 13. 产生 ONE Shared Boundary

最终只保留一个 Boundary：

```text
B_final
```

它统一应用于：

- camera；
- pointmap；
- 所有人体 root；
- 所有人体 joints；
- 所有人体 vertices。

不能出现：

```text
camera 使用一个变换
人物 A 使用另一个变换
人物 B 再使用第三个变换
```

所有对象必须共享同一个世界坐标变化，这就是 ONE Shared Boundary。

---

## 14. 提交状态

Boundary 和身份验证完成后，系统执行：

```text
Match -> Align -> Verify -> Commit
```

最终只提交：

- raw-reset Human3R recurrent state；
- 当前 shot 的固定 `B_final`；
- 已验证的人物身份信息。

Shadow state、shadow tracking 和临时 latent 全部删除。

完整 external identity commit/lifecycle 目前仍属于计划模块；已经实现并验证的核心语义是
raw-reset state 保留、shadow state 丢弃。

---

## 15. 处理新镜头的后续帧

后续 post-cut 帧不再运行 shadow correction，也不重新估计 Boundary。

它们只执行：

```text
普通 Human3R streaming
-> 得到 shot-local camera/scene/humans
-> 应用同一个固定 B_final
-> 输出 persistent-world 结果
```

所以额外计算主要只发生在 cut 第一帧。

---

## 16. 再次发生 Camera Cut

如果之后从镜头 B 切到镜头 C，重复同样过程。

新的相对 Boundary 会和已有 world Boundary 组合：

```text
B_world_C = B_world_B @ DeltaB_BC
```

这样理论上可以处理：

```text
A -> B
A -> B -> C
A -> B -> A
```

这部分真实 multi-cut runtime 是接下来需要实现和验证的重点。

---

## 17. 当前已经完成到哪里

当前已经实现或通过 probe 跑通：

```text
Human3R encoder/decoder 和正常流式推理
外部给定 cut index
first-post-cut V14 shadow correction
fresh Human3R Hard Reset raw branch
B0 = C_shadow @ inverse(C_raw)
B0-assisted root+torso Hungarian
Fixed Explicit + V16 per-human candidates
旧版 uniform multi-human consensus
ONE fixed shared Boundary segment propagation
demo.py 格式的长段可视化
```

当前仍未完成：

```text
automatic cut detector
正式 broad-trained V14.1 checkpoint
variable visibility/dustbin lifecycle
B0-centered bounded residual
真实 A->B->C multi-cut runtime
完整 no-cut parity 和 runtime benchmark
```

---

## 18. 专业名词解释

### Encoder

把 RGB 图像转换成模型内部 tokens 的网络。它主要负责提取信息，还没有输出最终三维
世界。

### Decoder

让图像、相机、人体和历史 tokens 通过 attention 交互，并产生用于各个输出 head 的
refined tokens。

### Recurrent State

Human3R 在连续视频中保存的内部历史记忆。它让模型知道此前看过什么，但 camera cut
后也可能把旧镜头信息错误带入新镜头。

### Hard Reset

在 camera cut 的第一张新图 decode 前，清空旧镜头 recurrent state。

### Shadow Branch / Shadow Transaction

临时读取旧状态，对第一张 post-cut 图像执行一次纠正，但绝不提交它产生的 state。

### Token

模型内部表示图像、相机、人体或纠正信息的高维向量。

### Attention

让不同 tokens 相互读取信息的机制。例如 pose token 可以读取人物和场景 token，判断
当前相机是否合理。

### Latent Residual

对模型内部特征的修正，不是直接手工修改最终三维坐标。

### Head

把 refined token 解码成具体输出的小型网络。例如 Camera Head 输出相机，Human Head
输出 SMPL-X。

### SE(3)

三维刚体变换，由 rotation 和 translation 组成，不包含人体独立形变。

### SO(3) Mean

对多个三维旋转进行几何平均。普通数字平均不能直接用于 rotation matrix。

### Gauge

整段三维重建采用的世界坐标系。两个 hard-reset shots 即使各自重建正确，也可能处于
两个不同 gauge。

### Boundary

将新 shot 的局部坐标转换到 persistent world 的显式 SE(3)。

### `B0`

由 shadow camera 与 raw-reset camera 差值产生的第一版 coarse Boundary。

### WHO

人物身份问题，即 cut 前后“谁是谁”。

### WHERE

几何位置问题，即 camera、scene 和人物在统一世界中“在哪里”。

### Hungarian Assignment

在一组人物和一组 detections 之间求总代价最小的一对一匹配算法。

### Dustbin

允许一个人物或 detection 暂时“不匹配任何人”的出口，用于处理人物进入、离开、漏检
和低置信度。

### Pointmap

图像中每个像素对应的预测三维点。

### SMPL-X

参数化三维人体模型，能够表示 body pose、shape、手、脸和人体网格。

### Root / Torso

Root 是人体整体位置锚点；torso orientation 是人体躯干朝向，用于短时几何和动作兼容性。

### Causal / Streaming

只读取当前和过去帧，不读取未来，不回头修改已经输出的历史。

### Commit

把当前计算结果正式写入后续会继续使用的长期 state 或 memory。Shadow 结果只读，不
commit；raw-reset state 才 commit。

---

## 19. 最简总结

整个方法可以概括为：

> 正常镜头交给 Human3R；发生 cut 时，先 reset 获得干净的新状态，再用一次不提交的
> shadow correction 估计新旧世界之间的 `B0`，在 `B0` 对齐后判断人物身份，必要时做
> 保守多人精修，最后让 camera、scene 和所有人体共享同一个固定 Boundary。
