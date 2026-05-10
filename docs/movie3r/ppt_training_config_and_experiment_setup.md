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
