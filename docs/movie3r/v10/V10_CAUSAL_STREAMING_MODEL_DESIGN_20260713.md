# V10 因果流式模型设计

日期：2026-07-13

## 1. 核心想法

V10 不应该是一个“每一帧都强行修 Human3R 输出”的 correction 分支。最终模型应该是一个严格流式的系统：

> detector 先控制 Human3R 的 local state；Human3R 负责每个 shot 内的局部重建；V10 维护一个跨 shot 的 global state，把 camera、human 和 point cloud 接到同一个全局坐标系里。

这里最重要的约束是严格流式：

- 模型只能使用历史帧和当前帧；
- 不能先跑完整个 segment，再回头修改前面的帧；
- detector 必须在 Human3R state 更新之前运行；
- Human3R head 输出当前帧之后，后处理或对齐只能影响当前帧和未来帧的 global 输出，不能回头改过去。

所以最终设计不是：

```text
Human3R 输出 -> SMPL 对齐 -> 结束
```

而应该是：

```text
detector -> state 控制 -> Human3R 局部重建
         -> 粗对齐
         -> 细对齐
         -> 因果 motion / global-state integrator
         -> global camera + human + point cloud
```

## 2. 完整前向流程

### Step 1. 输入当前帧

视频是一帧一帧输入的。在第 `t` 帧时，模型只能看到：

- 当前 RGB 帧；
- 历史帧缓存下来的特征或 state；
- 历史已经输出的 global camera / human / point cloud；
- 历史 detector 判断结果。

模型不能看到未来帧。

### Step 2. 前端 shot detector

detector 是整个流程的第一步。

它可以使用轻量的因果特征：

- RGB 图像差异；
- bbox 变化；
- 2D 关键点变化；
- 图像匹配特征；
- 可选的历史缓存图像特征。

它输出一个二分类或 soft 判断：

```text
继续当前 shot
或者
开启新的 shot
```

这个判断必须发生在当前帧进入 Human3R encoder / decoder 之前。原因是 Human3R 的 recurrent state 需要提前知道：这一帧是应该沿用上一帧 state，还是应该 reset / fork 一个新的 local state。

detector 不一定是主要创新点。第一版可以是简单规则，也可以是 image-only classifier。

### Step 3. Human3R 之前的 state 控制

detector 的判断会变成一个 state-control 信号。

如果没有 shot boundary：

```text
Human3R local state: continue
global state: 正常更新
```

如果检测到 shot boundary：

```text
Human3R local state: reset 或 fork
global state: 保留历史记忆
```

这表示新的 shot 会用一个干净的 Human3R local state 开始，但 global state 不会被丢掉。global state 是长期的世界坐标参考，用来把新 shot 的 local reconstruction 接回历史世界。

### Step 4. Human3R 局部重建

当前帧进入原版 Human3R encoder-decoder。

Human3R 输出的是当前 local coordinate system 下的局部重建：

- local camera pose；
- local SMPLX / human mesh / joints；
- local point cloud 或 depth；
- 可选的 pose token / human token / state token summary。

对于同一个连续 shot 内的帧，Human3R 的 local 输出通常已经比较稳定。V10 应该尽量不要破坏这些帧。

对于新 shot，Human3R 的输出在本 shot 内可能是合理的，但它可能和上一个 shot 落在不同的坐标系里。

### Step 5. 粗对齐：segment-to-global alignment

Human3R heads 输出当前帧的 local camera / human / point cloud 之后，V10 先计算一个粗对齐变换 `T_geo`。

`T_geo` 的作用是把当前 local shot 对齐到历史 global state。

可以使用的 cue 包括：

- pelvis / root 位置；
- 左右 hip 方向；
- torso / spine 方向；
- head 方向；
- feet anchor；
- floor normal 或 upright cue；
- 可选的稀疏 scene anchor，如果它足够可靠。

这一步和 HumanMM 的 OAM 有相似思想：用显式几何去解决一个强约束的跨 shot 对齐问题。

但区别是，V10 不只是对齐人体朝向。V10 要估计一个 segment-to-global transform，并且把它同时作用到：

- camera pose；
- human / SMPLX；
- point cloud / scene points。

`T_geo` 是一个强几何粗 proposal，可以不训练。

### Step 6. 细对齐

粗对齐之后仍然可能有误差：

- yaw 对了，但 roll / pitch 有倾斜；
- 高度有偏差；
- 某个 anchor 不可靠；
- 人体看起来对齐了，但相机轨迹不自然；
- floor / point cloud 和全局不一致。

因此需要一个 fine alignment module 来细修 `T_geo`。

输入可以包括：

- `T_geo`；
- 当前 local Human3R 输出；
- 当前 human anchors；
- 当前 floor / normal cue，如果可用；
- pose token / human token / state token summary；
- global state 预测的当前 anchor 和 camera trend；
- detector confidence。

输出包括：

- anchor reliability 或 anchor weights；
- alignment confidence；
- 围绕 `T_geo` 的小 residual transform；
- global state 的 update gate。

fine module 不应该直接预测一个自由的完整 SE(3)。之前实验已经说明，直接回归完整 SE(3) 容易过拟合，也不稳定。学习部分应该是保守的：

```text
T_final = small_residual * T_geo
```

这样学习模块有贡献，但输出范围是受约束的。

### Step 7. 因果 motion / global-state integrator

这是最终模型里最重要的模块。

固定缓存一个 segment transform 只是 probe 阶段的简化版本。它能把一个 shot 接到另一个 shot 上，但对于长序列、运动人物、多次 shot transition 来说不够。

最终模型需要一个因果的 motion / global-state integrator。

它的作用是随时间维护和更新 global state：

- global human root / pelvis / torso / feet 的历史；
- 人体速度和朝向变化趋势；
- global camera trajectory trend；
- 当前 segment-to-global transform；
- anchor 和 transform 的 confidence；
- 可选的 scene / floor / point-cloud anchors。

在对齐新 shot 之前，这个模块要先预测：

```text
如果没有发生镜头切换，当前帧的人和相机应该在全局坐标里的哪里
```

对于静止人物：

```text
预测 anchor 基本不变
```

对于运动人物：

```text
预测 anchor 会沿着最近的 root velocity / motion trend 往前走
```

这样可以避免一个很粗暴的规则：永远把当前人贴回上一帧的人。对于运动场景，这种规则会把真实运动抹掉。

integrator 必须是因果的。它不能看完整个序列之后再做全局优化，而是每一帧往前更新一次 state。

### Step 8. 最终输出

每一帧，V10 最终输出：

- global camera pose；
- global SMPLX / human mesh / joints；
- global point cloud / scene points；
- detector decision；
- segment-to-global transform；
- alignment confidence；
- anchor weights；
- 用于调试的 global state summary。

## 3. 和 HumanMM 的对应关系

HumanMM 的流程可以概括为：

```text
多分镜 RGB 视频
  -> bbox / 2D keypoint / image features
  -> shot detection
  -> 每个 shot 单独用 Masked LEAP-VO 估计相机
  -> 每个 shot 单独用 GVHMR 初始化人体
  -> 用 OAM 做跨 shot 人体整体朝向对齐
  -> 用 ms-HMR 修复跨 shot 身体姿态
  -> 预测 root velocity 和 foot contact
  -> trajectory refinement
  -> 连续世界坐标下的人体动作 + 相机轨迹
```

V10 可以这样对应：

| HumanMM 模块 | V10 对应模块 |
|---|---|
| bbox / 2D keypoint / image feature extraction | 因果 detector 的前端特征 |
| shot detector | Human3R state 更新之前的 V10 shot detector |
| 每个 shot 单独估计相机 | 原版 Human3R 的 local camera reconstruction |
| 每个 shot 单独初始化人体 | 原版 Human3R 的 local SMPLX / human output |
| OAM geometry alignment | V10 的粗对齐 `T_geo`，做 segment-to-global alignment |
| ms-HMR pose recovery | 不直接照搬；V10 用 anchor reliability 和 fine alignment 替代 |
| root velocity / foot contact prediction | V10 的因果 motion-state prediction |
| trajectory refiner | V10 的因果 global-state integrator |
| 最终人体动作 + 相机 | V10 的 global camera + human + point cloud |

关键区别是：

HumanMM 主要恢复全局人体运动。V10 要对齐的是 Human3R 的完整 local reconstruction，包括 camera、human 和 point cloud。

HumanMM 可以做 whole-sequence refinement。V10 必须保持 streaming 和 causal。

## 4. 建议的训练拆分

完整系统不应该作为一个大模型端到端一起训练，而应该分模块训练。

### Stage A. Detector

目标：

```text
判断当前帧是不是新 shot 的开始
```

输入：

- RGB features；
- bbox；
- 2D keypoints；
- image matching features。

loss：

- boundary 二分类 loss；
- 特别强调 stable frame 上的低 false positive rate。

这一阶段不需要 SMPLX GT，也不需要 camera GT。

### Stage B. 粗几何 baseline

目标：

```text
先衡量显式 T_geo 不训练时能做到什么程度
```

第一版使用 oracle boundary。

评估：

- local reset only；
- fixed human-anchor `T_geo`；
- floor / upright assisted `T_geo`；
- 可选 scene-assisted `T_geo`。

这是强 baseline。后续学习模块至少不能比它更差。

### Stage C. Fine Alignment Module

目标：

```text
在不替代 T_geo 的前提下改善 T_geo
```

训练内容：

- anchor weights；
- confidence；
- small residual transform；
- state update gate。

loss：

- camera rotation / translation loss；
- human anchor loss；
- body frame loss；
- vertical / floor consistency loss；
- proposal improvement loss；
- residual prior loss。

关键规则：

learned residual 应该很小。如果 residual 变成了主变换，说明模型很可能在过拟合。

### Stage D. 因果 Motion / Global-State Integrator

目标：

```text
在不看未来帧的情况下维持全局时序连贯
```

训练它从历史 state 预测当前 global human / camera 状态：

- root velocity；
- body orientation trend；
- contact / foot stability；
- camera motion trend；
- segment transform smoothness。

loss：

- state prediction loss；
- root velocity loss；
- 可选 foot contact loss；
- camera trend loss；
- transform smoothness loss；
- 连续帧 no-op loss；
- 最终 global camera / human alignment loss。

这一阶段最接近 HumanMM 的 trajectory predictor 和 trajectory refiner，但必须改成因果流式版本。

## 5. 训练数据构造

真实 multi-shot GT 很少，不能只依赖它。

一个更实际的训练方式是：

1. 取普通连续视频。
2. 用 frozen original Human3R 跑出连续结果。
3. 把这条连续 Human3R 输出当作 pseudo global trajectory。
4. 人为把视频切成多个 segment。
5. 对后面的 segment 加随机 SE(3) 扰动，模拟 shot gauge change。
6. 训练 V10 只能使用历史和当前信息，把扰动后的 segment 接回原来的连续 pseudo trajectory。

这样就可以训练 global-state integrator，而不需要大量真实的 multi-shot camera + SMPLX GT。

对于 AIST / H36M / MVHuman / AvatarReX 这类有 GT 的数据，能用 GT camera / human 的地方就用于更严格的监督和评估。

## 6. 为什么这是有意义的贡献

这个设计不是简单的 SMPL 后处理对齐。

真正的贡献是：

1. 把 shot-discontinuous Human3R reconstruction 定义成一个因果 segment gauge alignment 问题。
2. detector 放在 Human3R state 更新之前，让 state control 真正流式。
3. 区分 Human3R 的 local state 和跨 shot 的 global state。
4. 用显式几何做强粗对齐，用学习模块做 reliability、fine correction 和 state update。
5. 加入因果 motion / global-state integrator，让运动人物不会被简单贴回上一帧。
6. 同时对齐 camera、human 和 point cloud，而不是只修人体运动。

## 7. PPT 简短版本

Human3R 擅长重建连续 shot 内的 camera、human 和 point cloud，但它不知道如何把两个不连续 shot 接到同一个世界坐标系里。V10 在 Human3R 上增加一个 streaming global state。detector 先判断当前帧应该继续还是 reset local state。Human3R 输出 local camera、human、point cloud 后，V10 先做显式几何粗对齐，再用一个因果的 learned state module 做细修和时序连贯维护。最终输出是全局一致的 camera-human-scene reconstruction，同时整个过程保持前向、流式、不回头。
