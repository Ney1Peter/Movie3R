# V9 Method Overview

这份文档给我们自己看，目标是通俗但尽量完整地解释：Human3R 是什么，UniCon3R 怎么改，Movie3R V9 现在怎么设计，以及 loss 在监督什么。

## 1. Human3R 在做什么

Human3R 可以理解成一个前馈式、流式的人体 + 场景联合重建模型。它每来一帧，就同时估计：

- 场景：pointmap / depth / confidence。
- 相机：当前帧相对世界的 camera pose。
- 人体：SMPL / SMPL-X 参数、人体 mesh、joints、mask。
- 状态：用于下一帧的 recurrent state / memory。

简化流程是：

```text
video frame
  -> image encoder / tokenizer
  -> image tokens

video frame
  -> DINOv2/ViT-based Multi-HMR human encoder
  -> human detection / human tokens

image tokens + pose token + human tokens + recurrent state
  -> decoder
  -> refined image / pose / human tokens

refined image tokens
  -> scene / pointmap head

refined pose token
  -> pose head
  -> camera pose

refined human tokens
  -> human head
  -> SMPL / mesh / joints
```

这里的 Multi-HMR 不是一个简单名字标签，而是 Human3R 里专门做人检测和人体 token 的分支。代码里它使用 DINOv2 ViT backbone 提取人体相关视觉特征，再结合 human detection / tokenization / transformer 模块生成 human tokens。因此可以简短说成：

```text
Human3R 使用一个 DINOv2/ViT-based、经过人体任务训练的 Multi-HMR human encoder。
```

Human3R 的优点是前馈、流式、人体和场景一起输出。问题是遇到镜头跳变、跨视角拼接或弱纹理场景时，它可能把后几帧放到错误的 world gauge 里，导致相机、人和场景相对前几帧错位。

## 2. UniCon3R 的启发

UniCon3R 不是完全重写 Human3R，而是在 Human3R backbone 上加了一个 contact-guided prompt / refinement branch。

它的直觉是：

```text
如果人体和场景重建正确，脚、地面、身体和附近物体之间应该有合理接触关系。
```

所以它增加 contact token：

```text
human token + scene context + local geometry + history contact memory
  -> contact token

image / pose / human tokens + contact token + recurrent state
  -> decoder
  -> refined contact token

refined contact token
  -> contact head
  -> contact probability / contact loss

refined contact token
  -> residual head
  -> human latent residual
  -> corrected human latent
  -> human head
```

UniCon3R 最重要的启发不是“必须显式找脚”，而是这个范式：

1. 为一个关系问题单独构造 latent prompt token。
2. 让这个 token 进入 decoder，和原 image / pose / human tokens 通过 attention 交互。
3. decoder 后为这个 token 加专门 head 和专门 loss。
4. 用 refined token 输出 residual，修正原 backbone 的 latent，而不是完全重做模型。

UniCon3R 的 contact token 可以概括为三类信息：

| 信息 | 通俗解释 | 对应作用 |
|---|---|---|
| Semantic Scene Context | 当前帧视觉 token + 历史 scene memory | 知道人附近是什么，场景上下文是否支持 contact |
| Explicit Metric Geometry | 从上一帧 world pointmap 的人体附近采样局部 3D 几何 | 知道人附近表面在哪里，例如脚是否接近地面 |
| Temporal Momentum | 上一帧 refined contact token / contact state | contact 通常连续，不应该一帧有一帧无 |

## 3. Movie3R V9 的核心想法

我们的问题不是 contact，而是 camera pose 和 human 在 world 里的错位。

所以 V9 把 UniCon3R 的思路改成：

```text
contact relation prompt -> 修 human

变成：

pose/human relation correction prompt -> 同时修 camera pose 和 human latent
```

当前新增 token 叫 `A_corr_t`。它不是四个固定人体部位 token，也不是显式匹配脚/骨盆/躯干。它是一个 relation prompt，让 attention 自己从当前 image / pose / human token 和历史 memory 中学习“当前帧和历史 world 是否对齐”。

当前实现的 `A_corr_t` 有 3 个 relation tokens：

| token | 来自哪里 | 想表达什么 |
|---|---|---|
| semantic token | 当前 image / pose / human tokens 和 state / pose memory 软融合 | 当前视觉和历史世界是否一致 |
| alignment token | 当前 pose token、上一帧 pose token、pose latent 差分 | 当前相机运动是否像正常运动 |
| momentum token | 上一帧 refined corr token、上一帧 delta、上一帧 gate | 上一帧修正状态对当前帧的时间先验 |

简化流程：

```text
current image tokens
+ current pose token
+ current human tokens
+ recurrent state memory
+ pose memory
+ previous corr token / delta / gate
  -> A_corr_t

image / pose / human tokens + A_corr_t + recurrent state
  -> Human3R decoder
  -> refined image / pose / human tokens + refined A_corr_t
```

## 4. Camera 怎么修

Camera correction 是 latent residual，不是直接输出一个显式 4x4 `delta T`。

```text
refined A_corr_t
  -> pose residual head
  -> delta pose token + gate

raw refined pose token + gate * delta pose token
  -> corrected pose token
  -> original Human3R pose head
  -> corrected camera pose
```

这样做的好处是：仍然使用 Human3R 原来的 pose head，让新增分支只学习“如何把 token 推到更合理的位置”，而不是另起炉灶预测相机矩阵。

## 5. Human 怎么修

早期我们只修 camera pose，发现相机对了以后，人体仍可能飞起来或在深度上错位。这说明 Human3R 的人体输出还有自己的 camera-space / latent 平移误差。

显式 `smpl_transl` correction 已经证明问题可解：

```text
refined A_corr_t -> delta smpl_transl -> corrected SMPL translation
```

但这更像诊断分支，不够 UniCon-style。V9 当前更合理的做法是修 human latent：

```text
refined A_corr_t + refined human token + corrected pose token
  -> human latent residual head
  -> delta human token + shared gate

raw decoder human token + gate * delta human token
  -> corrected human token
  -> original Human3R human head
  -> corrected SMPL
```

也就是说，V9 不直接手写 `smpl_transl`，而是让 human head 自己解释被修正后的 human token。这更接近 UniCon3R 的 contact token residual 逻辑。

## 6. Gate 是什么

不是所有帧都应该修。正常连续的 AAAA clip 或原版 Human3R 已经很准的样本，correction 应该很小。

因此 V9 有一个 correction gate：

```text
gate 低：当前帧基本不修，保持 Human3R 原输出
gate 高：当前帧可能 drift，需要更大 residual
```

当前 gate 来自 pose residual head 的 drift logit，并被 human latent correction 分支共享。它不是人工写死“后两帧修、前两帧不修”，而是通过 raw pose error 构造 drift target 来训练。

后续可以继续比较：

- 四帧都允许修。
- 前两帧加 no-op loss，鼓励稳定帧少修。
- AAAA 正常连续样本增加比例，防止模型过度修正。

## 7. Loss 设计

当前主要使用 `V82PoseRelationLoss`，虽然名字还沿用 V8.2，但已经包含 V9 需要的 camera / gate / human 监督。

核心 loss 可以这样理解：

| loss / metric | 监督什么 | 为什么需要 |
|---|---|---|
| pose loss | corrected camera pose 接近 GT pose | 直接保证相机位姿变对 |
| drift / gate loss | gate 预测 raw pose 是否 drift | 让模型知道什么时候该修 |
| improvement margin | corrected 要比 raw 更好 | 防止模型只学到一个很小或无意义的修正 |
| pose residual small | delta pose token 不要无界变大 | 避免用过大 latent residual 硬凑 |
| human trans loss | corrected SMPL translation 接近 GT | 让人体位置真正回到正确 world / camera gauge |
| human delta small | human latent delta 不要过大 | 保持 residual correction 的“小修正”性质 |
| no-op loss | 稳定帧 delta 尽量小 | 防止 AAAA 或原版已经准的帧被强行修改 |
| LoRA norm loss | 如果启用 LoRA，约束 LoRA 变化幅度 | 防止微调 head 破坏原版能力 |

训练时 GT 可以参与 loss；推理时不能参与。推理时模型只能拿到当前输入、Human3R internal tokens、历史 state 和上一帧 correction memory。

## 8. 当前验证结果

V8.9 / V9 起点的 AvatarReX 单 clip overfit 已经验证成功。

设置：

```text
dataset: AvatarReX lbn1 AABB
seqA: lbn1/22070935
seqB: lbn1/22053926
start_frame: 1671
view_angle_deg: 143.418318
resize_mode: resize_only_16
base weights: original Human3R checkpoint
```

单 clip 训练后：

```text
raw human trans err       0.708 m
corrected human trans err 0.0037 m
raw camera trans err      0.592 m
corrected camera trans err 0.016 m
raw rot err               24.04 deg
corrected rot err          0.119 deg
gate mean                  0.339
```

这个结果说明：

1. `A_corr_t` 能提供有效 correction signal。
2. pose latent residual 能修 camera。
3. human latent residual 能修人体平移错位。
4. 坐标系修正后，AvatarReX 上的“人飞起来”主要是监督坐标系问题，不是模型能力完全不行。

## 9. 后续实验方向

当前合理顺序是：

1. 复现单 clip，从原版 Human3R 权重开始。
2. 扩到 5 clips，检查是否仍能 overfit。
3. 扩到 AvatarReX + THUman mixed train/test，显式保留未见 test。
4. 加 AAAA 正常连续样本，训练 gate 不要乱修。
5. 做 LoRA ablation：只训 correction branch、加 pose head LoRA、加 human head LoRA、pose+human LoRA。
6. 指标对齐 Human3R 论文里的 MPJPE / PA-MPJPE，同时保留我们自己的 camera trans/rot 和 human trans error。

### 9.1 备选方向：human-aware mask / static scene attention

Trophies 提供了一个值得后续借鉴的点：在估计静态场景和 camera pose 时，动态人体区域可能会污染时序几何理解。尤其是人物快速移动、人物占画面较大、背景纹理较弱时，模型可能把真实人体运动误判成相机或场景漂移，然后把人错误地拉回历史位置。

当前不要直接把人从输入图像中抹掉。原因是 V9 仍然依赖 Human3R 的统一前馈结构，human head 需要完整看到人体来输出 SMPL / SMPL-X；如果把人体区域变黑或变灰，可能会伤害人体重建和 human latent correction。

更合理的后续方案是保留原图，但利用 person mask 做 patch-level 的 human-aware attention：

```text
原始 RGB image tokens
  -> human branch / human head: 正常看人体区域
  -> scene / camera / memory branch: 对人体 patch 降权
  -> A_corr,t: 同时接收 human context 和 background/static context
```

实现上可以先做轻量实验：

1. dataloader 将已有 `mask/{frame}.png` 转成 patch-level human mask。
2. 在 `V82PoseRelationPrompt` 里额外构造 `background_context`，只聚合非人体 patch 或对人体 patch 降权。
3. camera pose residual head 更偏向使用 background/static context。
4. human latent correction head 仍然使用 human token，不屏蔽人体。
5. 加入快速移动 AAAA/AABB 样本，让 gate 学会区分“背景稳定但人真实移动”和“镜头跳变导致人/场景错位”。

这个方向暂时只作为备选，不进入当前 120h 训练。只有当后续发现 V9 在快速运动人物上持续过度拉回、或者 camera/scene 明显被人体动态干扰时，再考虑加入。
