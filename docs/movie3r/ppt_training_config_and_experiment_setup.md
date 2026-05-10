# PPT 第一部分：训练配置和实验设置

## 1. 训练配置和实验设置

### 1.1 训练配置

**总述：当前训练不是从零训练整个 Human3R，而是在已有 Human3R/CUT3R 权重上微调新增的 Shot-Aware 模块。**

分开来说，主干 encoder、decoder、人体分支和原始预测头基本冻结，只训练和镜头跳变相关的小模块，这样训练成本更低，也更方便观察 ShotToken 是否真的有效。

**总述：训练框架采用 `torchrun + Accelerate + DDP`，没有使用 FSDP 或 DeepSpeed。**

分开来说，`torchrun` 负责多进程启动，Accelerate 负责封装 DDP、bf16 混合精度和梯度累积；因为目前可训练参数量较小，H800 80GB 显存也足够，所以不需要更复杂的分布式优化框架。

**总述：模型主体仍然是 ARCroco3DStereo，也就是 Human3R/CUT3R 的原始重建框架。**

分开来说，当前配置使用 `DINOv2 ViT-L` backbone，encoder depth 为 24，decoder depth 为 12，普通图像输入尺寸是 512 系列分辨率，Multi-HMR 分支输入尺寸是 896。

**总述：训练模式使用 `freeze='shot_adaptation'`，也就是只训练 Shot-Aware 相关模块。**

分开来说，V4 主要训练 `ShotTokenGenerator` 和 `PoseAlignmentAdapter`；V5.1 主要训练 `ShotTokenGenerator` 和 `LayerwisePoseShotAdapter`，主干重建能力尽量保持不变。

**总述：每个训练 sample 包含 4 个 view。**

分开来说，`num_views=4`，所以 `batch_size=24` 实际表示每张卡一次处理 `24 × 4 = 96` 张图像；如果是 4 卡训练，全局一次更新就是 `96 sample = 384 view`。

补充说明：之前 `batch_size=2` 就占用接近 48GB 显存是异常峰值，不是正常 batch 开销。

- 原因一：`num_views=4` 会让 `batch_size=2` 实际变成 8 张图同时进入模型，所以显存不能按单图理解。
- 原因二：当时 `cudnn.benchmark=True`，cuDNN 在 backward 时可能选择速度快但 workspace 很大的算法，导致显存峰值突然冲高。
- 原因三：当时 pretrained checkpoint 曾经直接加载到 GPU，加载模型时 GPU 上会临时多放一份权重，占用额外显存。
- 修复后：关闭 `benchmark` 并把 checkpoint 先加载到 CPU 后，`batch_size=2` 显存降到约 8.5GB，`batch_size=24` 约 40.9GB，显存表现恢复正常。

**总述：训练数据主要使用 AvatarReX 的 Video 和 AABB 两种采样。**

分开来说，Video 样本用于学习连续视频中的稳定性，AABB 样本用于构造相机切换场景；当前训练集是 3 套 AvatarReX 数据，每套包含 Video 和 AABB，各取 800 个样本，总共 `4800 samples / epoch`。

**总述：验证集和测试集也使用 AvatarReX Video + AABB。**

分开来说，val/test 各是 `600 samples / epoch`，即 3 套数据 × Video/AABB × 每类 100 个样本，用来观察连续视频和相机跳变两类场景的指标变化。

**总述：主 loss 仍然围绕 3D 重建、RGB 和 SMPL 三部分。**

分开来说，基础训练项包括 `Regr3DPoseBatchList`、`RGBLoss` 和 `SMPLLoss`；Shot-Aware 相关项会额外记录或加入 `shot_bce`、`shot_q0_loss`、`shot_noop_loss`、`shot_pointmap_keep_loss`、`shot_pose_residual_loss` 等指标。

**总述：V5.1 在 V4 基础上额外加入 camera jump 相关监督。**

分开来说，V5.1 增加了 `shot_boundary_abs_loss`、`shot_jump_rel_loss` 和 `shot_anchor_loss`，目的是更直接地约束镜头切换边界处的 camera pose。

**总述：优化器和学习率配置保持稳定。**

分开来说，当前使用 `lr=1e-4`、`min_lr=1e-6`、`weight_decay=0.05`、`warmup_epochs=5`，并开启 `gradient_checkpointing=true` 和 `amp=1` 的 bf16 混合精度训练。

**总述：由于容器 `/dev/shm` 只有 64MB，实际训练强制使用 `num_workers=0`。**

分开来说，虽然配置文件里有默认 `num_workers`，但训练脚本会显式传入 `num_workers=0`，避免多进程 DataLoader 出现 bus error 或 shared memory 不足。

### 1.2 实验设置

**总述：V4 正式训练使用 4 卡，每卡 batch size 为 24。**

分开来说，V4 的全局 batch size 是 `4 × 24 = 96`，每个 epoch 有 `4800 / 96 = 50` 个训练 step，30 epoch 总共约 `1500` 次参数更新。

| 项目 | V4 正式训练 |
|---|---:|
| GPU 数量 | 4 |
| 每卡 batch size | 24 |
| 全局 batch size | 96 |
| 每个 sample 的 view 数 | 4 |
| 每次更新实际 view 数 | 384 |
| 训练集大小 | 4800 samples / epoch |
| step 数 | 50 steps / epoch |
| 训练 epoch | 30 |
| 总参数更新次数 | 1500 |
| 显存占用 | 约 26.6GB / 卡 |
| 单 step 时间 | 约 4.3-4.4s |
| eval 频率 | 每 5 epoch 一次 |

**总述：V5.1 调试训练使用单卡，每卡 batch size 也是 24。**

分开来说，单卡时全局 batch size 就是 24，每个 epoch 有 `4800 / 24 = 200` 个训练 step，5 epoch 总共约 `1000` 次参数更新。

| 项目 | V5.1 debug 训练 |
|---|---:|
| GPU 数量 | 1 |
| batch size | 24 |
| 全局 batch size | 24 |
| 每个 sample 的 view 数 | 4 |
| 每次更新实际 view 数 | 96 |
| 训练集大小 | 4800 samples / epoch |
| step 数 | 200 steps / epoch |
| 训练 epoch | 5 |
| 总参数更新次数 | 1000 |
| 显存占用 | 约 44.7GB |
| 单 epoch 时间 | 约 2小时03分钟 |
| 总训练时间 | 11小时37分钟 |
| eval 频率 | 每 1 epoch 一次 |

**总述：短实验一般跑 5-10 epoch，用来判断结构方向是否值得继续训练。**

分开来说，如果是 4 卡 batch 24，5 epoch 是 `250` 次更新，10 epoch 是 `500` 次更新；如果是单卡 batch 24，5 epoch 是 `1000` 次更新，10 epoch 是 `2000` 次更新。

**总述：当前梯度更新没有使用梯度累积。**

分开来说，`accum_iter=1`，所以每个训练 step 都会完成一次 `backward + optimizer step + zero_grad`；V4 30 epoch 对应约 `1500` 次实际参数更新，V5.1 5 epoch 对应约 `1000` 次实际参数更新。

## 2. 后续两种主流改进思路及可能问题

### 2.1 思路一：修改 ShotToken，让它变成 local background anchor

**总述：这条路线的核心是把 ShotToken 从“全局命令”改成“局部场景证据”。**

分开来说，不再让一个 token 负责所有镜头跳变信息，而是从静态背景里提取多个 anchor，例如墙角、门框、地面纹理，让 camera 根据多个局部证据共同推理对齐关系。

- 问题一：最难的是如何稳定找到可信的 static background anchor，因为电影画面里人、遮挡和动态物体很多。
- 问题二：anchor 不能选到人体上，否则模型会把动态人体误当作静态世界点来对齐 camera。
- 问题三：低纹理区域很难作为 anchor，例如白墙、天空、模糊地面通常没有足够可靠的匹配信息。
- 问题四：跨镜头不一定有足够 overlap，如果 A 镜头和 B 镜头几乎没有共同背景，强行 re-anchor 可能会拉错 camera。
- 问题五：逐 patch 提取比逐帧提取更精细，但会带来更多 token、更高计算量和更复杂的筛选逻辑。
- 问题六：anchor 的置信度需要额外估计，否则模型不知道哪些局部匹配可信、哪些匹配应该忽略。
- 问题七：如果像 Human3R 提取人头一样提取背景 anchor，需要一个可靠的“背景关键点/静态区域”检测器，但这个模块目前还没有现成答案。
- 问题八：可以引入冻结的 DINOv2、SAM、LightGlue 或 MASt3R 辅助提取 anchor，但会增加工程复杂度和推理成本。
- 问题九：anchor token 如果仍然能被 image/human token 随意读取，依然可能污染 pointmap 和人体分支。
- 问题十：local anchor 只提供局部匹配证据，不应该直接预测完整 camera correction，否则又会变成另一个全局控制器。

### 2.2 思路二：增加 attention mask，限制 token 之间的交互权限

**总述：这条路线的核心是控制“谁能看谁”，防止 ShotToken 或 anchor token 干扰不该改动的分支。**

分开来说，可以在 decoder 内增加 attention mask，让 anchor token 只和 pose token 交互，不让 image token 和 human token 直接读取它。

```text
anchor token <-> pose token: yes
anchor token <-> image token: no
anchor token <-> human token: no
```

- 问题一：当前 decoder attention 原生不支持 mask，需要修改 `Attention`、`CrossAttention`、`DecoderBlock` 和 `_decoder()` 调用链。
- 问题二：如果只是在 attention 里加 mask、不改变参数形状，理论上不会破坏预训练权重的加载。
- 问题三：虽然预训练权重还能加载，但 attention 路由变了，模型行为仍然可能和原始预训练分布不完全一致。
- 问题四：mask 如果设计太严格，pose token 可能拿不到足够上下文，camera correction 的能力会受限。
- 问题五：mask 如果设计太宽松，anchor token 仍然可能泄露到 image/human 分支，继续造成背景或人体污染。
- 问题六：加入新 token 后，即使 mask 做对了，也要确认原本 image token 之间的 attention 没有被意外改变。
- 问题七：attention mask 会增加代码路径复杂度，需要额外测试 shot on/off、连续帧、跳变帧三类情况。
- 问题八：mask 本身不能提供更好的几何证据，它只能减少污染，不能解决 global token 信息太粗的问题。
- 问题九：如果 anchor token 的内容本身是错的，mask 只能限制错误影响范围，不能保证 pose correction 正确。
- 问题十：masked decoder 工程量比 pose-only adapter 更大，因此更适合作为 V6 后半阶段，而不是第一步就全量改。

### 2.3 两种方法结合

**总述：最理想的长期方案是两种方法一起用：local anchor 解决“token 表达什么”，attention mask 解决“token 能影响谁”。**

分开来说，local anchor 提供更可靠的局部背景匹配证据，attention mask 保证这些证据只服务 camera / pose，不直接污染 image、human 和 pointmap。

- 问题一：两种方法一起做效果可能最好，但工程风险也最大，因为同时改 token 来源和 decoder attention。
- 问题二：建议先做安全版 local anchor + pose-only adapter，验证 anchor evidence 是否有效。
- 问题三：如果 anchor evidence 有效，再加入 attention mask，让 anchor token 更安全地进入 decoder。
- 问题四：最终目标不是让一个 token 直接“命令相机怎么动”，而是让多个局部背景证据帮助 camera 自己推理对齐。
