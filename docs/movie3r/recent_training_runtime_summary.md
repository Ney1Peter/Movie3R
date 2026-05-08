# Movie3R 近期训练小结

这份文档主要解释几个容易误会的问题：

- 现在是不是用 DDP？
- Gradient Accumulation 有没有算对？
- 多卡训练时 batch size 和更新步数怎么算？
- 为什么之前 `batch_size=2` 显存看起来很离谱？
- 上一个版本为什么训练 loss 看起来能降，但推理结果不对？
- 后面为什么要对 ShotToken 做约束？

## 第一部分：训练配置说明

### 1. 当前训练到底用的是什么分布式框架

结论：**当前用的是 Accelerate 管理的 DDP。**

不是单纯手写 DDP，也不是没有 DDP。

启动脚本是：

```bash
python -m torch.distributed.run --nproc_per_node=4 train.py ...
```

这一步会启动 4 个进程，每个进程用一张 GPU。

然后代码里创建了：

```python
accelerator = Accelerator(
    gradient_accumulation_steps=args.accum_iter,
    mixed_precision="bf16",
    kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
)
```

再调用：

```python
optimizer, model, data_loader_train = accelerator.prepare(
    optimizer, model, data_loader_train
)
```

这一步会把模型包装成 DDP，把 optimizer 和 dataloader 也交给 Accelerate 管理。

所以准确说法是：

```text
torchrun 启动多进程
Accelerate 负责封装 DDP、混合精度、梯度累积、梯度同步
底层模型同步仍然是 DDP
```

### 2. 当前 batch size 怎么理解

正式训练脚本：`scripts/cmd_4gpu_train_bz24.sh`。

当前设置是：

```text
GPU 数量 = 4
每张 GPU 的 batch_size = 24
accum_iter = 1
```

注意：代码里的 `batch_size=24` 是 **每张卡的 batch size**，不是全局 batch size。

所以全局一次 forward/backward 看到的样本数是：

```text
global batch = 24 × 4 = 96
```

因为现在 `accum_iter=1`，所以每次处理完这 96 个样本就更新一次参数。

也就是：

```text
每次参数更新使用 96 个样本
```

### 3. Gradient Accumulation 有没有用？有没有算错？

当前正式训练配置是：

```yaml
accum_iter: 1
```

所以现在其实 **没有额外做梯度累积**。

这意味着：

```text
1 个 dataloader iteration
= 4 张 GPU 各处理 24 个样本
= 全局 96 个样本
= backward 一次
= optimizer.step() 更新一次
```

所以当前不存在“累积次数是不是算错”的问题，因为现在累积次数就是 1。

### 4. 如果以后 accum_iter 不等于 1，怎么算？

通用公式是：

```text
有效 batch size = 每卡 batch_size × GPU 数量 × accum_iter
```

比如当前每卡 `24`，4 张 GPU。

如果 `accum_iter=1`：

```text
有效 batch size = 24 × 4 × 1 = 96
```

如果以后改成 `accum_iter=2`：

```text
有效 batch size = 24 × 4 × 2 = 192
```

直观理解：

```text
accum_iter=1：每处理 1 批就更新一次
accum_iter=2：先处理 2 批，把梯度攒起来，再更新一次
accum_iter=4：先处理 4 批，把梯度攒起来，再更新一次
```

### 5. 每个 epoch 有多少步？

当前训练集配置大约是：

```text
800 × 6 = 4800 个样本 / epoch
```

当前全局 batch 是：

```text
24 × 4 = 96
```

所以每个 epoch 大约有：

```text
4800 / 96 = 50 个 iteration
```

因为当前 `accum_iter=1`，每个 iteration 都更新一次参数。

所以：

```text
每个 epoch 大约更新 50 次参数
30 个 epoch 大约更新 1500 次参数
```

如果以后 `accum_iter=2`：

```text
每 2 个 iteration 更新一次
每个 epoch 大约更新 25 次参数
```

### 6. NativeScalerWithGradNormCount 是干什么的

这个名字有点容易误导。现在它不是在做传统 fp16 的手动 loss scaling。

当前训练用的是：

```python
mixed_precision="bf16"
```

bf16 一般不需要 fp16 那种动态 loss scaler。

现在 `NativeScalerWithGradNormCount` 更像是一个小封装，主要做三件事：

```text
1. 调用 accelerator.backward(loss)
2. 做 gradient clipping
3. 调用 optimizer.step()
```

关键点是：optimizer 已经被 Accelerate 包装过了。

所以即使代码里看起来每个 iteration 都写了：

```python
optimizer.step()
optimizer.zero_grad()
```

Accelerate 也会根据 `accum_iter` 判断什么时候真的更新、什么时候只是继续累积。

当前 `accum_iter=1`，所以每个 iteration 都会真的更新。

### 7. 参数到底什么时候更新

当前正式训练可以简单理解成：

```text
每张 GPU 拿 24 个样本
4 张 GPU 一共处理 96 个样本
算 loss
backward
DDP 同步梯度
clip grad
optimizer.step 更新参数
zero_grad 清空梯度
进入下一批
```

也就是说：

```text
当前是每 96 个样本更新一次参数
```

不是每张卡单独乱更新，也不是 4 张卡各自更新出 4 份不同模型。

DDP 会同步梯度，保证 4 张卡更新的是同一个模型。

### 8. 为什么之前 batch_size=2 显存接近 48GB 很奇怪

之前观察到：

| batch size | 修复前显存峰值 | 修复后显存峰值 |
|---:|---:|---:|
| 2 | 约 `45.8 GB` | 约 `8.5 GB` |
| 4 | 约 `48.2 GB` | 约 `11.1 GB` |
| 8 | 约 `53.0 GB` | 约 `16.4 GB` |

`batch_size=2` 就接近 48GB 明显不合理。

后面定位到主要有两个原因。

#### 原因一：cudnn.benchmark=True

`cudnn.benchmark=True` 会让 cuDNN 自动寻找最快的卷积算法。

问题是：最快的算法不一定省显存。

有些算法会在 backward 时申请非常大的临时 workspace。

所以之前的情况是：

```text
模型常驻显存其实不高
但是 backward 过程中 cuDNN 临时申请了很大的 workspace
导致显存峰值突然冲到 45GB 以上
```

修复方式：

```yaml
benchmark: False
```

#### 原因二：pretrained checkpoint 曾经直接加载到 GPU

之前代码相当于：

```python
ckpt = torch.load(args.pretrained, map_location=device)
```

这会让 GPU 上除了模型本身，还额外临时放一份 checkpoint 权重。

修复后改成：

```python
ckpt = torch.load(args.pretrained, map_location="cpu")
model.load_state_dict(...)
del merge_state_dict
torch.cuda.empty_cache()
```

这样 checkpoint 先放 CPU，加载完就释放，不再额外占 GPU 显存。

### 9. 修复后为什么选择每卡 batch_size=24

修复后测试结果更合理：

| 每卡 batch size | 显存峰值 | 平均耗时 |
|---:|---:|---:|
| 8 | 约 `17.6 GB` | `14.35 s/it` |
| 16 | 约 `29.3 GB` | `27.85 s/it` |
| 24 | 约 `40.9 GB` | `38.87 s/it` |
| 32 | 约 `52.6 GB` | `52.89 s/it` |

H800 单卡显存约 80GB。

`batch_size=32` 也能跑，但显存更高，余量更小。

`batch_size=24` 显存约 41GB，吞吐也比较好，所以当前正式训练选择每卡 `24`。

4 卡训练时就是：

```text
每卡 24
全局 96
每 96 个样本更新一次
```

### 10. 训练配置部分的简短结论

可以直接这样解释：

```text
我们现在是 torchrun 启 4 个进程，Accelerate 在内部封装 DDP。
每张卡 batch_size=24，所以全局 batch 是 96。
当前 accum_iter=1，没有额外梯度累积，所以每个 iteration 更新一次参数。
训练集大约 4800 个样本，所以每个 epoch 大约 50 次参数更新，30 个 epoch 大约 1500 次更新。
之前 batch_size=2 显存接近 48GB 是异常峰值，主要是 cudnn.benchmark 导致 backward workspace 过大，以及 checkpoint 曾经加载到 GPU。
修复后 batch_size=2 约 8.5GB，batch_size=24 约 40.9GB，显存表现已经正常。
```

补充一点：如果以后真的把 `accum_iter` 改成大于 1，建议把 gradient clipping 显式放在 `accelerator.sync_gradients=True` 时执行。当前正式训练 `accum_iter=1`，所以不受影响。

## 第二部分：训练结果分析

### 1. 上一个版本的问题是什么

上一个正式版本是 LoRA64 + ShotToken 的版本。

训练日志表面上看是正常的：loss 会下降，训练过程没有明显报错。

但实际 demo / 推理结果不对，主要表现是：

```text
pointmap 尺度不稳定
camera / pose 结果不稳定
SMPL transl 和人体尺度容易崩
连续镜头本来应该接近原 Human3R，但结果被明显扰动
```

所以问题不是“训练跑不起来”，而是：

```text
训练 loss 看起来能优化，但 ShotToken 进入模型后破坏了原本稳定的重建能力
```

### 2. 为什么怀疑是 ShotToken 的问题

我们做了几组消融。

第一组：只跑原始 Human3R。

```text
结果正常
```

说明 base model 本身没有问题。

第二组：加载 LoRA64 checkpoint，但关闭 shot adaptation。

```text
结果基本正常
```

这说明 checkpoint 里的主体权重、LoRA 权重本身不是主要崩坏原因。

第三组：同一个 LoRA64 checkpoint，打开 shot adaptation。

```text
结果明显崩坏
```

这说明只要 ShotToken 路径参与 forward，结果就会变差。

第四组：把 LoRA 的影响尽量关掉，只保留 trained ShotToken 的影响。

```text
结果仍然会崩
```

这一步很关键，因为它把 LoRA 和 ShotToken 分开了。

如果关掉 LoRA 后问题还在，就说明主要问题不在 LoRA，而在 ShotToken 注入 decoder 这条路径。

### 3. 为什么这种消融能定位问题

原因很简单：每次只改变一个开关。

可以理解成下面这个对照表：

| 实验 | LoRA checkpoint | ShotToken | 结果 | 说明 |
|---|---|---|---|---|
| 原始 Human3R | 无 | 关 | 正常 | base model 正常 |
| LoRA64 + shot 关 | 有 | 关 | 正常 | LoRA / checkpoint 不是主因 |
| LoRA64 + shot 开 | 有 | 开 | 崩坏 | ShotToken 路径触发问题 |
| 弱化 LoRA + shot 开 | 基本关掉 | 开 | 仍崩 | 问题更集中在 ShotToken |

所以结论是：

```text
问题不是模型整体训练失败，也不是 DDP / batch size / optimizer 的问题。
主要问题是旧版 ShotToken 进入 decoder 后，对原本稳定的 token 表达产生了过强扰动。
```

### 4. 旧版 ShotToken 为什么容易出问题

旧版思路是：检测到镜头变化后，生成一个 ShotToken，然后把它送进 decoder，让 decoder 感知“这里发生了镜头跳变”。

这个方向本身是合理的，因为 Movie3R 要解决的就是多镜头电影级数据里的 shot change 问题。

但旧版有一个问题：

```text
ShotToken 注入 decoder 的强度没有被很好约束
```

具体来说，旧版 ShotToken 容易出现：

```text
连续帧也产生较强 shot token
shot token 的 norm 偏大
decoder 原本的 image token 被额外 token 干扰
连续镜头下本来应该 no-op，但实际不是 no-op
```

我们前面也量过一个现象：

```text
trained q_t 的 norm 明显大于 decoder image token 的 norm
连续帧和跳变帧的 q_t 强度没有拉开
```

这意味着 ShotToken 不只是“提醒模型这里有 shot change”，而是可能变成了一个很强的全局扰动。

对 decoder 来说，它看到的是一个额外 token。这个 token 如果太强，就可能影响 cross-attention，进而影响 pointmap、camera、SMPL 等所有后续预测。

### 5. 为什么不能直接删掉 ShotToken

不能简单删掉 ShotToken，因为我们的目标确实需要处理镜头切换。

原 Human3R 更适合连续视频或视角变化比较平滑的情况。

Movie3R 的目标是电影级多镜头人体重建，镜头切换时会出现：

```text
视角突然变化
人物尺度突然变化
背景和构图突然变化
时序连续性被打断
```

所以模型需要知道“这里是 shot change”。

真正的问题不是 ShotToken 这个方向错了，而是旧版 ShotToken 没有被限制住。

所以后面的改进方向是：

```text
保留 ShotToken，但让它只在需要的时候起作用；
连续帧时尽量不影响原 Human3R；
跳变帧时才允许它提供额外信息。
```

### 6. 后面怎么改进 ShotToken

后面我们做的是 ShotToken V2。

核心思想是给 ShotToken 加三个约束。

第一个约束：加 shot probability gate。

旧版是直接生成 `q_t` 并注入 decoder。

新版是先预测一个 `shot_prob`：

```text
shot_prob 越接近 0，说明更像连续帧
shot_prob 越接近 1，说明更像镜头跳变
```

然后用它控制 ShotToken 强度：

```text
q_t = shot_scale × shot_prob × normalized(q_raw)
```

这样连续帧的 ShotToken 会被压小。

第二个约束：加 shot classification supervision。

也就是让模型显式学习：

```text
当前 pair 是连续帧，还是 shot change
```

对应 loss 是：

```text
shot_bce
```

这样 `shot_prob` 不是随便学出来的，而是被 shot label 监督。

第三个约束：连续帧 no-op。

连续帧时，我们希望 ShotToken 尽量不要改变原模型输出。

所以训练时会比较：

```text
打开 ShotToken 的输出
关闭 ShotToken 的输出
```

对于连续帧，这两个输出应该尽量接近。

这就是：

```text
shot_noop_loss
```

直观理解是：

```text
如果不是镜头跳变，就不要乱动原模型结果。
```

### 7. V2 具体约束了哪些东西

新版主要加了这些 loss / 指标：

| 名称 | 作用 |
|---|---|
| `shot_bce` | 监督模型判断是不是 shot change |
| `shot_q0_loss` | 连续帧时压低 ShotToken 能量 |
| `shot_noop_loss` | 连续帧时要求 shot-on 和 shot-off 输出接近 |
| `shot_prob_gap` | 观察跳变帧和连续帧的 shot probability 是否拉开 |
| `shot_q_energy_cont` | 观察连续帧 ShotToken 是否足够小 |
| `shot_q_energy_jump` | 观察跳变帧 ShotToken 是否可以更强 |

当前配置里对应权重是：

```yaml
shot_loss_weight: 0.1
shot_q0_loss_weight: 0.1
shot_noop_loss_weight: 1.0
shot_scale_init: 0.05
```

其中 `shot_scale_init=0.05` 的意思是：一开始不要让 ShotToken 太强，先从很小的扰动开始学。

### 8. 这次训练重点看什么

这次训练不能只看总 loss。

因为上一个版本已经说明：

```text
loss 能降，不代表 demo 一定正常
```

这次重点要看：

| 指标 | 希望看到的趋势 |
|---|---|
| `shot_bce` | 下降，说明 shot change 分类在学 |
| `shot_acc` | 上升，说明 shot change 判断更准 |
| `shot_prob_gap` | 变成正数并拉开，说明跳变帧概率高于连续帧 |
| `shot_q_energy_cont` | 保持较小，说明连续帧不乱注入 |
| `shot_q_energy_jump` | 可以高于 continuous，说明跳变帧允许使用 ShotToken |
| `shot_noop_loss` | 下降或保持较小，说明连续帧 shot-on / shot-off 输出接近 |

最后还是要跑 demo / 推理消融。

理想结果是：

```text
连续镜头：接近原 Human3R，不明显扰动
镜头跳变：ShotToken 能提供帮助
pointmap / camera / SMPL transl 不再崩
```

### 9. 训练结果分析部分的简短结论

可以这样总结：

```text
上一个版本训练 loss 能降，但推理结果不对。
通过消融发现，base Human3R 正常，LoRA64 checkpoint 在关闭 shot adaptation 时也正常，只有打开 ShotToken 后结果崩坏。
进一步弱化 LoRA 后问题仍然存在，所以主要问题定位到 ShotToken 进入 decoder 这条路径。
原因是旧版 ShotToken 注入强度缺少约束，连续帧也可能产生强扰动，破坏 decoder 原本稳定的 token 表达。
新版没有删除 ShotToken，而是给它加 gate、scale、shot 分类监督、连续帧 q0 约束和 no-op 约束。
目标是让连续帧时 ShotToken 尽量不影响原模型，只有镜头跳变时才发挥作用。
```

## 第三部分：ShotToken V2 失败现象记录

### 1. 本次额外验证视频

为了排除 `h36.mp4` 是训练集外数据导致异常，我们从 AvatarReX 训练集里构造了一个 10 帧测试视频：

```text
data/avatarrex_train_camjump_contiguous_0300_0309_10f.mp4
```

构造方式是同一训练集时间连续、相机切换：

```text
A 段：Training/22010708/rgb/00000300.png 到 00000304.png
B 段：Training/22010710/rgb/00000305.png 到 00000309.png
```

也就是：

```text
A(t) A(t+1) A(t+2) A(t+3) A(t+4) B(t+5) B(t+6) B(t+7) B(t+8) B(t+9)
```

这个视频的目的不是测试泛化，而是验证：在训练分布内、时间连续但相机突然切换的情况下，Shot Adaptation 是否能稳定工作。

### 2. 推理现象

使用 checkpoint：

```text
experiments/formal_training-4gpu-bz24-shot-v2/checkpoint-best.pth
```

关闭 Shot Adaptation 时：

```text
前 5 帧基本正确。
第 6 帧以后因为相机切换，整体位置出现偏移。
但后 5 帧的重建本身仍然是对的，说明 base reconstruction 没有明显坏掉。
```

启用 Shot Adaptation 时：

```text
只有第 1 帧基本正确。
后续帧位置明显不对。
相机位姿表现得像都被压到第一帧附近。
尺度也不对。
背景 / pointmap 出现错误。
人物本身相对没那么坏，说明 HumanLoRA 当前只修 smpl_transl 的策略没有明显破坏人体形状。
```

### 3. 当前判断

这次训练集内连续时间 A5B5 测试失败，说明 V2 的问题不是简单的 out-of-distribution 泛化问题。

更可能的问题是：

```text
ShotToken V2 的训练 loss 虽然下降，但 Shot Adaptation 路径仍然没有学成可用的相机切换修正。
```

更具体地说，当前 Shot Adaptation 的作用范围太大：

```text
1. q_t 被 append 到 decoder token 序列，会通过 attention 影响 pose token、image tokens、SMPL tokens。
2. image tokens 被影响后，下游 pointmap / background reconstruction 也会被影响。
3. WorldLoRA 又显式修改 pts3d_in_self_view 和 pts3d_in_other_view。
4. 最终效果不只是修相机位姿，而是把几何重建和尺度一起扰动了。
```

所以当前 V2 的失败现象更符合下面这个解释：

```text
ShotToken + WorldLoRA 过度干预了 decoder 几何表达。
模型没有学到“只在 shot change 时修正相机/人体位置”的局部行为。
相反，启用 Shot Adaptation 后破坏了原 Human3R 稳定的重建 token。
```

### 4. 对 LoRA 的进一步判断

当前不能把问题简单归因到所有 LoRA。

HumanLoRA 已经改成只修：

```text
smpl_transl
```

不再修：

```text
smpl_shape
smpl_rotmat
smpl_expression
```

这和观察到的“人物本身没那么坏”基本一致。

更危险的是：

```text
decoder q_t 注入
WorldLoRA 对 pointmap 的显式修改
PoseLoRA residual 幅度没有足够强的输出限制
```

### 5. V3 前的排查方向

下一步不要直接重训 30 epoch，而是先做推理级细粒度消融。

建议添加 demo 开关：

```text
--disable_shot_decoder_token
--disable_world_lora
--disable_human_lora
--disable_pose_lora
```

第一组重点实验：

```text
关闭 decoder q_t 注入。
关闭 WorldLoRA。
只保留 PoseLoRA 和 HumanLoRA。
q_t 不进入 decoder attention，只作为 PoseLoRA / HumanLoRA 的 condition 使用。
```

这组实验要验证：

```text
如果 pointmap / background reconstruction 恢复正常，说明主要问题来自 decoder q_t 和 WorldLoRA。
如果仍然崩，说明 PoseLoRA / HumanLoRA residual 本身也需要限幅或重新设计。
```

进一步建议的 V3 方向：

```text
1. 不再把 q_t append 到 decoder。
2. 关闭 WorldLoRA，先不修改 pointmap。
3. 优先只做 PoseLoRA，必要时保留只修 smpl_transl 的 HumanLoRA。
4. 对 pose translation residual 加 tanh 限幅。
5. 对 rotation residual 改成小角度修正，或用很小的 quaternion delta。
6. 连续帧使用 hard no-op gate，保证 shot-off 和 shot-on 在非跳变帧几乎一致。
```

### 6. 本阶段结论

ShotToken V2 不能作为最终方案。

当前最重要的结论是：

```text
训练 loss 下降和 shot_acc 上升不代表 Shot Adaptation 路径可用。
即使在训练集内构造的连续时间相机切换视频上，V2 仍然会破坏 camera pose、scale 和 background reconstruction。
V3 应该先缩小作用范围，只修相机/人体位置，不再直接改 decoder image tokens 和 pointmap。
```
