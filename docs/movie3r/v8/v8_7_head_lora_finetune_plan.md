# V8.7 Pose / Human Head LoRA Finetune Plan

## 背景

当前 V8.6 四帧全修版本已经验证：

```text
pose correction token 进入 decoder
  -> residual head 修 camera pose
  -> human translation correction head 修 SMPL transl
```

这个方向是有效的，尤其在 5 个 THuman 大角度 clip overfit 上，camera 和 human translation 都能被明显修正。

但它还有一个问题：

```text
目前主要是新拉一条 correction 分支在做修正，
原本 Human3R 的 pose head / human head 基本没有适配这个新分支。
```

也就是说，新增 token 已经提供了关系信息，但原始 head 不一定知道怎么最好地使用这些信息。UniCon3R 的 contact 分支也不是只训练一个孤立 head，而是让 contact-aware 信息参与人体输出。因此下一步要测试：

```text
在保持 Human3R 主体能力不被破坏的前提下，
是否可以用 LoRA 轻量微调 pose head 和 human head，
让原 head 更好地适配 correction token。
```

## 当前 Delta 和 LoRA 的区别

当前 V8.6 的 correction delta 可以理解成一种广义 adapter，但不是严格 LoRA。

### 当前做法

```text
correct token 进入 decoder
  -> refined correction token
  -> residual head 预测 delta
  -> delta 加到 pose token 或 SMPL transl 上
```

它的特点是：

```text
冻结原模型大部分参数，
额外训练一个分支，
由这个分支直接预测修正量。
```

所以它是：

```text
residual branch / adapter-like correction
```

但不是严格的 LoRA。

### 严格 LoRA

LoRA 是在原 head 的线性层旁边加一个低秩旁路：

```text
原始层:
  y = W x

LoRA 层:
  y = W x + scale * B(A(x))
```

其中：

```text
W 冻结
A / B 是新增的小矩阵
只训练 A / B
```

所以 LoRA 不是直接输出一个最终 delta，而是让原 head 的内部计算产生一个很小的可训练偏移。

## LoRA 的好处

| 好处 | 通俗解释 | 对当前任务的意义 |
|---|---|---|
| 不破坏原模型 | 原 head 主权重冻结 | Human3R 原本能做对的样本不容易被训坏 |
| 参数少 | 只训练低秩 A/B 小矩阵 | 比全量解冻 pose/human head 更稳，也更省显存 |
| 适合小数据 | 可训练参数少，过拟合风险相对低 | 当前 correction 数据还不够大时更合适 |
| 能适配新 token | 让 head 内部学会使用 correction 后的 feature | 比单独分支更像模型内部协同 |
| 易做对照 | 可以单独开 pose LoRA / human LoRA / both | 能分析到底是哪个 head 需要适配 |
| 初始化安全 | LoRA B 可以初始化为 0 | 初始输出和原 Human3R 完全一致，训练更稳 |

一句话总结：

```text
LoRA 是一种不大改原模型、只给原 head 加小旁路的轻量微调方法。
它比全量解冻更安全，也比完全孤立的 correction head 更像模型内部适配。
```

## V8.7 总体结构

V8.7 不替换 V8.6，而是在当前最好的四帧全修版本上加 head LoRA。

```text
video frames
  -> encoder
  -> image / camera / human / pose tokens

image / camera / human / pose tokens
  + pose-human correction token
  + recurrent state
  -> decoder
  -> refined image / camera / human / pose tokens
  -> refined correction token

refined correction token
  -> pose residual head
  -> delta pose token
  -> corrected pose token
  -> pose head + [新增] pose-head LoRA
  -> corrected camera pose

refined correction token
  -> human translation correction head
  -> delta SMPL transl

refined human token / corrected human feature
  -> human head + [新增] human-head LoRA
  -> SMPL params
  -> apply delta SMPL transl
  -> corrected SMPL
```

核心思想：

```text
correction token 仍然是主信息来源。
LoRA 不是替代 correction token，而是让原 pose/human head 更会使用 correction token 带来的变化。
```

## 四组对照实验

先固定同一批 5 个大角度 overfit clips，使用当前 V8.6 full-long 设置，做四组最小对照。

| 组别 | 训练内容 | 目的 |
|---|---|---|
| A. baseline | 当前 V8.6 full-long | 看没有 LoRA 时的基础表现 |
| B. pose LoRA | V8.6 + pose head LoRA | 看 pose head 轻量适配是否让 camera 更准 |
| C. human LoRA | V8.6 + human head LoRA | 看 human head 轻量适配是否让 SMPL 更准 |
| D. pose + human LoRA | V8.6 + pose head LoRA + human head LoRA | 看两个 head 一起适配是否有互补效果 |

训练时四组都保持：

```text
encoder 冻结
decoder 冻结
scene / DPT head 冻结
原 pose head 主权重冻结
原 human head 主权重冻结
只训练 correction branch + 对应 LoRA 参数
```

注意：

```text
不要再使用 v8_pose_prompt_pose_head 这种全量解冻 pose head 的方式。
那不是 LoRA，会破坏原 head 能力，之前 pose-head 全量训练已经证明风险较大。
```

## 代码改动计划

### 1. 新增 LoRA Linear Wrapper

新增一个通用 `LoRALinear`：

```text
base linear:
  frozen W, frozen bias

lora branch:
  down: in_dim -> rank
  up: rank -> out_dim

forward:
  base(x) + alpha / rank * up(down(x))
```

初始化策略：

```text
down 正常小随机初始化
up 初始化为 0
```

这样训练刚开始时：

```text
LoRA 输出为 0
整体模型输出等于原 Human3R
```

这对稳定性很重要。

建议文件：

```text
src/dust3r/lora.py
```

或者如果只服务 V8：

```text
src/dust3r/v8_head_lora.py
```

### 2. 增加 Head LoRA 注入函数

需要写一个递归函数，只替换指定 head 内的 `nn.Linear`。

候选函数：

```text
inject_lora_to_linear_modules(module, rank, alpha, target_name_filter)
```

要求：

```text
只替换 pose_head / human_head 内部层
不碰 encoder / decoder / DPT scene head
不碰整个 downstream_head
```

原因是：

```text
DPT / scene / backbone 一旦被 LoRA 改动，会让变量太多，
很难判断到底是谁导致效果变化。
```

### 3. 模型配置开关

在模型初始化配置中新增：

```text
v8_pose_head_lora: false
v8_human_head_lora: false
v8_head_lora_rank: 8
v8_head_lora_alpha: 8
v8_head_lora_dropout: 0.0
```

可选再细分：

```text
v8_pose_head_lora_rank
v8_human_head_lora_rank
```

第一版建议先统一 rank，减少变量。

### 4. 新增 Freeze Mode

新增一个 freeze mode，例如：

```text
freeze: v8_pose_prompt_head_lora
```

这个模式下：

```text
冻结所有原始 Human3R 权重
开启 v8_pose_prompt
开启 v8_pose_residual_head
开启 v8_human_trans_corr_head
如果配置打开 pose LoRA，则训练 pose LoRA 参数
如果配置打开 human LoRA，则训练 human LoRA 参数
```

训练参数应该只包括：

```text
v8_pose_prompt.*
v8_pose_residual_head.*
v8_human_trans_corr_head.*
*.lora_down.*
*.lora_up.*
```

必须打印并保存 trainable parameter list，确认没有误把原 head 全量打开。

### 5. 保存和加载

checkpoint 中需要保存：

```text
correction branch 参数
pose head LoRA 参数
human head LoRA 参数
```

加载时要注意：

```text
必须先根据 config 注入 LoRA 模块，
再 load checkpoint。
```

否则 checkpoint 里的 LoRA key 会找不到。

### 6. Sanity Check

训练前必须做三个检查：

| 检查 | 标准 |
---|---|
| LoRA 初始化等价 | 开 LoRA 但不训练时，输出应接近原 V8.6 |
| trainable params | 只有 correction branch 和 LoRA 参数可训练 |
| viewer 坐标 | raw Human3R / GT / corrected 三者坐标系正确 |

尤其是 viewer：

```text
raw Human3R 必须来自正确的 demo / saved Human3R payload，
不能再用错误 gauge 的 raw pose dump。
```

## 训练计划

### Stage 0. 单 clip 过拟合 sanity

目的：

```text
确认 LoRA 实现没有破坏当前 full-long 能力。
```

数据：

```text
使用之前 V8.6 成功的大角度 THuman clip
或者同一个 5-clip overfit set 中最明显的一组
```

训练：

```text
epochs: 50-100
batch size: 1
learning rate: 低于 correction branch 或相同先试
rank: 8
alpha: 8
```

成功标准：

```text
至少不差于当前 full-long。
如果加入 human LoRA 后 SMPL world error 下降，说明 human head 适配有价值。
```

### Stage 1. 五 clip 对照

目的：

```text
在当前最熟悉的 5 个 clips 上做 A/B/C/D 对照。
```

四组：

```text
A: V8.6 full-long baseline
B: V8.6 + pose LoRA
C: V8.6 + human LoRA
D: V8.6 + pose LoRA + human LoRA
```

每组训练：

```text
epochs: 100 或 loss 稳定后停止
batch size: 1 或当前稳定设置
optimizer: 只优化 trainable params
```

观察：

```text
camera trans / rot
SMPL head / pelvis / mean joint world error
gate mean
pose residual norm
human transl residual norm
LoRA parameter norm
```

### Stage 2. 小批量混合训练

如果 Stage 1 有提升，再进入小批量训练。

数据：

```text
AvatarReX + THuman
AABB + AAAA
保留显式 test split
```

比例建议：

```text
AABB drift 样本比例提高
AAAA stable 样本保留
```

原因：

```text
如果 drift 样本太少，模型会学成少修或不修。
如果 AAAA 没有，模型可能正常帧也乱修。
```

### Stage 3. 正式 benchmark

在固定 test set 上输出：

```text
AABB large angle
AABB medium angle
AAAA normal continuous
AvatarReX held-out
THuman held-out
```

每组都要比较：

```text
raw Human3R
V8.6 full-long
V8.7 pose LoRA
V8.7 human LoRA
V8.7 pose + human LoRA
```

## Loss 设计

V8.7 不是重写 loss，而是在 V8.6 的基础上继续使用同一套监督。

核心 loss：

| Loss | 作用 |
|---|---|
| `L_camera_pose` | corrected camera pose 接近 GT |
| `L_pose_residual_small` | 相机 residual 不要无意义变大 |
| `L_smpl_head_world` | SMPL 头部 world 坐标接近 GT |
| `L_smpl_pelvis_world` | SMPL 身体中心 world 坐标接近 GT |
| `L_smpl_mean_joint_world` | 全身关节平均位置接近 GT |
| `L_human_trans_residual_small` | human translation residual 不要乱动 |
| `L_gate` | AABB drift 高 gate，AAAA stable 低 gate |
| `L_aaaa_noop` | 正常连续帧尽量保持原输出 |

LoRA 本身可以加一个很轻的正则：

```text
L_lora_norm
```

作用：

```text
防止 LoRA 变成新的全量 head，
保证它只是轻量适配。
```

第一版建议：

```text
先不额外加复杂 LoRA loss，
只监控 LoRA 参数 norm。
如果发现 LoRA 改动过大，再加 L_lora_norm。
```

## 指标和可视化

每个 checkpoint 都至少保存：

```text
camera_trans_error_raw
camera_trans_error_corrected
camera_rot_error_raw
camera_rot_error_corrected

smpl_head_world_error_raw
smpl_head_world_error_corrected
smpl_pelvis_world_error_raw
smpl_pelvis_world_error_corrected
smpl_mean_joint_world_error_raw
smpl_mean_joint_world_error_corrected

gate_mean
pose_delta_norm
human_delta_norm
pose_lora_norm
human_lora_norm
```

viewer 统一规则：

```text
GT camera: red
raw Human3R camera: gray
corrected camera: yellow
pointcloud / SMPL 使用 corrected output
raw Human3R 必须来自正确 demo payload
```

## 预期结果判断

### 如果 pose LoRA 有效

说明：

```text
pose head 原本对 correction 后的 pose token 不够适配。
```

后续可以把 pose LoRA 保留为主线。

### 如果 human LoRA 有效

说明：

```text
human head 原本不太会利用 correction 分支带来的新关系信息。
```

这对当前任务更重要，因为 V8.6 已经暴露出：

```text
camera 对了，不代表人一定对。
```

### 如果 pose + human LoRA 最好

说明：

```text
相机和人体 head 都需要轻量适配，
correction token 应该作为 joint pose-human correction prompt 继续发展。
```

### 如果 LoRA 没提升

说明可能是：

```text
当前瓶颈不在 head 适配，
而在 token 信息不足、loss 设计不足、数据比例不足或坐标监督噪声。
```

这时不应该继续加大 LoRA，而应该回到：

```text
token 构造
AABB drift 数据比例
human / scene relation supervision
```

## 最小执行顺序

建议按这个顺序做：

1. 读取 pose head / human head 结构，确认哪些 `nn.Linear` 可以加 LoRA。
2. 实现 `LoRALinear` 和注入函数。
3. 加配置开关和 `v8_pose_prompt_head_lora` freeze mode。
4. 做 LoRA 初始化等价 sanity check。
5. 用一个 clip 跑通 pose LoRA / human LoRA / both。
6. 用 5 clips 训练 A/B/C/D 四组对照。
7. 用同一批 viewer 和 benchmark 评估。
8. 只把有效的 LoRA 方案扩展到更大数据集。

## 当前结论

V8.7 的目标不是替代 V8.6 的 correction token，而是验证：

```text
在 UniCon-style decoder-in correction token 已经有效的基础上，
原始 pose head / human head 是否也需要轻量适配。
```

如果 LoRA 有效，最终模型结构会比单独 residual 分支更完整：

```text
新增 token 提供 correction relation，
residual head 给出显式修正，
pose/human head LoRA 让原 Human3R 输出头轻量适配这种新信息。
```

## 2026-06-10 代码落地状态

第一版 V8.7 LoRA 能力已经接入，但尚未启动正式训练。

新增代码：

```text
src/dust3r/v8_head_lora.py
  LoRALinear
  inject_lora_to_linear_modules
  mark_lora_trainable
  lora_parameter_l2
```

模型改动：

```text
src/dust3r/model.py
  新增 config:
    v8_pose_head_lora
    v8_human_head_lora
    v8_head_lora_rank
    v8_head_lora_alpha
    v8_head_lora_dropout

  新增 freeze mode:
    v8_pose_prompt_head_lora
```

当前 freeze 规则：

```text
冻结 encoder / decoder / scene head / 原 pose head / 原 human head
训练 v8_pose_prompt
训练 v8_pose_residual_head
训练 v8_human_trans_corr_head
训练 LoRA 的 lora_down / lora_up
```

不会训练：

```text
downstream_head.pose_head 原始 weight / bias
downstream_head.deccam/decpose/decshape/decexpression 原始 weight / bias
DPT scene head
MHM-R detector
backbone
decoder
```

loss 改动：

```text
src/dust3r/losses.py
  V82PoseRelationLoss 增加:
    pose_lora_norm_weight
    human_lora_norm_weight

  默认权重为 0，先只记录:
    v82_pose_head_lora_l2
    v82_human_head_lora_l2
```

五 clip 对照配置：

```text
config/train_v8_7_head_lora_thuman5_pose_lora.yaml
config/train_v8_7_head_lora_thuman5_human_lora.yaml
config/train_v8_7_head_lora_thuman5_pose_human_lora.yaml
```

三份配置都从当前 V8.6 full-long 权重初始化：

```text
output/v8_6_human_correction_thuman_overfit/
  v8_6_joint_gate_pose_human_thuman5_aabb_overfit_full_long/
    checkpoint-final.pth
```

sanity check 结果：

| 配置 | pose LoRA layers / params | human LoRA layers / params | bad trainable |
|---|---:|---:|---:|
| pose LoRA | 2 / 55,352 | 0 / 0 | 0 |
| human LoRA | 0 / 0 | 8 / 445,096 | 0 |
| pose + human LoRA | 2 / 55,352 | 8 / 445,096 | 0 |

`bad trainable = 0` 表示没有误打开原始 Human3R head、encoder、decoder 或 scene head 参数。
