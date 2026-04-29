# Movie3R 训练代码详解

## 1. 整体架构

训练使用 **Accelerate** 库封装分布式训练细节，配合 **Hydra** 进行配置管理。

```
train.py
├── train()              # 主函数，初始化和训练循环
├── train_one_epoch()    # 单个 epoch 的训练逻辑
├── test_one_epoch()     # 验证/测试逻辑
├── build_dataset()      # 构建 DataLoader
└── run()                # Hydra 入口
```

---

## 2. train() 主函数流程 (L120-426)

### 2.1 Accelerator 初始化

```python
accelerator = Accelerator(
    gradient_accumulation_steps=args.accum_iter,  # 梯度累积
    mixed_precision="bf16",                        # 混合精度
    kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True),
        InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
    ],
)
```

**作用**：
- 自动处理多卡分布式（`torchrun` 启动时）
- 自动管理 bf16 混合精度
- 自动处理梯度累积

### 2.2 数据集构建 (L162-197)

```python
data_loader_train = build_dataset(args.train_dataset, args.batch_size, ...)
data_loader_test = {dataset.split("(")[0]: build_dataset(dataset, ...)}
data_loader_val = {dataset.split("(")[0]: build_dataset(dataset, ...)}  # 可选
```

`build_dataset()` 内部调用 `get_data_loader()`，使用 `accelerator.prepare()` 包装 DataLoader。

### 2.3 模型初始化 (L199-254)

```python
model: PreTrainedModel = eval(args.model)  # e.g., "ARCroco3DStereo"
smpl_model: SMPLModel = SMPLModel(device, model_args=...)

# 加载预训练权重
if args.pretrained and not args.resume:
    ckpt = torch.load(args.pretrained, ...)
    merge_state_dict = strip_module(ckpt["model"])
    model.load_state_dict(merge_state_dict, strict=False)
```

### 2.4 优化器配置 (L256-260)

```python
param_groups = misc.get_parameter_groups(model, args.weight_decay)
optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
loss_scaler = NativeScaler(accelerator=accelerator)  # bf16 loss scaler
```

`misc.get_parameter_groups()` 按照 timm 的策略：对 bias 和 norm 层不施加 weight decay。

### 2.5 Accelerator Prepare (L262-265)

```python
optimizer, model, data_loader_train = accelerator.prepare(
    optimizer, model, data_loader_train
)
```

**关键**：这一步会将 model 包装成 `DistributedDataParallel`（多卡）并移动到对应设备。

### 2.6 训练循环 (L324-420)

```python
for epoch in range(args.start_epoch, args.epochs + 1):

    # 1. 验证 (在训练之前)
    if data_loader_val is not None and epoch % args.eval_freq == 0:
        val_stats = test_one_epoch(...)

    # 2. 测试
    if epoch % args.eval_freq == 0:
        test_stats = test_one_epoch(...)

    # 3. Early Stopping 检查
    if monitor_loss < best_so_far:
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += args.eval_freq
        if epochs_without_improvement >= early_stop_patience:
            break

    # 4. 训练
    train_stats = train_one_epoch(...)
```

**注意**：验证在训练之前执行，好处是可以基于上一个 epoch 的模型状态做 early stopping 判断。

---

## 3. train_one_epoch() 详解 (L463-641)

### 3.1 初始化

```python
model.train(True)
metric_logger = misc.MetricLogger(delimiter="  ")
optimizer.zero_grad()
```

### 3.2 核心训练循环

```python
for data_iter_step, batch in enumerate(data_loader):

    with accelerator.accumulate(model):  # 梯度累积控制
        epoch_f = epoch + data_iter_step / len(data_loader)

        # 学习率调整
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        # 前向传播
        result = loss_of_one_batch(
            batch, model, criterion, accelerator,
            symmetrize_batch=False, use_amp=bool(args.amp),
            smpl_model=smpl_model
        )

        loss, loss_details = result["loss"]

        # 梯度反向传播
        if not result.get("already_backprop", False):
            loss_scaler(
                loss, optimizer,
                parameters=model.parameters(),
                update_grad=True,
                clip_grad=1.0,
            )
            optimizer.zero_grad()

        # 记录指标
        metric_logger.update(loss=loss_value, **loss_details)
```

### 3.3 关键点

#### 3.3.1 `accelerator.accumulate(model)`

控制是否执行梯度更新。当 `data_iter_step % accum_iter != 0` 时，只做前向传播和 `loss.backward()`（梯度累积），不执行 `optimizer.step()`。

#### 3.3.2 `loss_of_one_batch()`

这是核心前向+loss计算函数，位于 `dust3r.inference`：
```python
result = loss_of_one_batch(batch, model, criterion, ...)
# result = {
#     'pred': [...],      # 模型预测
#     'loss': (loss, loss_details),  # 总 loss 和详细 loss
#     ...
# }
```

#### 3.3.3 bf16 Loss Scaling

```python
loss_scaler(loss, optimizer, parameters=..., update_grad=True, clip_grad=1.0)
```

`NativeScaler` 来自 `croco.utils.misc`，专门处理 bf16 训练中的 loss scaling，防止下溢。

#### 3.3.4 梯度裁剪

`clip_grad=1.0` 限制梯度范数不超过 1.0，防止梯度爆炸。

### 3.4 可视化 (L592-627)

当 `(data_iter_step + 1) % accum_iter == 0` 且达到 `print_img_freq` 周期时：
```python
depths_self, gt_depths_self = get_render_results(batch, result["pred"], self_view=True)
depths_cross, gt_depths_cross = get_render_results(batch, result["pred"], self_view=False)
gt_msks, pr_msks, gt_hms, pr_hms, gt_smpls, pr_smpls = get_render_smpl(...)

imgs_stacked_dict = get_vis_imgs_new(loss_details, ...)
log_writer.add_images("train" + "/" + name, imgs_stacked, step)
```

生成训练过程的可视化图像（深度图、SMPL 渲染等）。

---

## 4. test_one_epoch() 详解 (L644-755)

```python
@torch.no_grad()
def test_one_epoch(model, criterion, data_loader, accelerator, device, epoch, ...):

    model.eval()  # 评估模式

    for _, batch in enumerate(data_loader):
        batch = todevice(batch, device)
        result = loss_of_one_batch(batch, model, criterion, accelerator, ...)

        loss_value, loss_details = result["loss"]
        metric_logger.update(loss=float(loss_value), **loss_details)

    # 返回统计结果
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in [("avg", "global_avg"), ("med", "median")]
    }
    return results
```

---

## 5. 分布式训练细节

### 5.1 多卡启动

使用 `torchrun` 启动时：
```bash
torchrun --nproc_per_node=8 train.py epochs=40 batch_size=8
```

Accelerate 自动检测到多个进程，包装 model 为 `DistributedDataParallel`。

### 5.2 关键配置

```python
accelerator = Accelerator(
    gradient_accumulation_steps=args.accum_iter,  # 全局梯度累积
    mixed_precision="bf16",                        # bf16 混合精度
)
accelerator.even_batches = False  # 允许不均匀 batch（最后一个不完整）
```

### 5.3 loss_of_one_batch 中的处理

```python
result = loss_of_one_batch(
    batch, model, criterion, accelerator,
    symmetrize_batch=False,   # 不对称 batch
    use_amp=bool(args.amp),   # 使用 bf16
    smpl_model=smpl_model
)
```

---

## 6. 训练流程图

```
开始
  │
  ▼
Accelerator 初始化 (bf16, gradient_accumulation)
  │
  ▼
构建数据集 (train/val/test DataLoader)
  │
  ▼
创建模型 (ARCroco3DStereo) + criterion (Loss)
  │
  ▼
加载预训练权重 (可选)
  │
  ▼
optimizer = AdamW(param_groups, lr)
  │
  ▼
accelerator.prepare(optimizer, model, data_loader)
  │
  ▼
┌─────────────────────────────────────────┐
│  for epoch in range(args.epochs):       │
│    │                                    │
│    ├─► 验证 (test_one_epoch)            │
│    │                                    │
│    ├─► Early Stopping 检查               │
│    │                                    │
│    ├─► 训练 (train_one_epoch)            │
│    │   │                                │
│    │   ├─► for batch in data_loader:    │
│    │   │   │                            │
│    │   │   ├─► accelerator.accumulate() │
│    │   │   │   │                        │
│    │   │   │   ├─► loss_of_one_batch() │
│    │   │   │   │   │                    │
│    │   │   │   │   └─► 前向 + criterion│
│    │   │   │   │                        │
│    │   │   │   ├─► loss.backward()      │
│    │   │   │   │   (累积梯度)           │
│    │   │   │   │                        │
│    │   │   │   └─► if accum_iter==0:   │
│    │   │   │       optimizer.step()    │
│    │   │   │       loss_scaler.update()│
│    │   │   │       optimizer.zero_grad()│
│    │   │   │                            │
│    │   │   └─► metric_logger.update()  │
│    │   │                                │
│    │   └─► return train_stats           │
│    │                                    │
│    ├─► 保存 checkpoint (last/best/final)│
│    │                                    │
│    └─► Early Stopping 触发? ──► 退出    │
└─────────────────────────────────────────┘
  │
  ▼
保存最终模型 (checkpoint-final.pth)
  │
  ▼
结束
```

---

## 7. 关键文件对应关系

| 功能 | 文件位置 |
|------|----------|
| 模型定义 | `dust3r/model.py` |
| Loss 计算 | `dust3r/inference.py` (loss_of_one_batch) |
| 数据加载 | `dust3r/datasets/` |
| 训练入口 | `train.py` |
| 配置文件 | `config/train.yaml` |

---

## 8. 常见问题

### 8.1 为什么 `accelerator.accumulate(model)` 需要？

控制梯度累积。当 `accum_iter > 1` 时，不是每个 batch 都更新参数，而是累积 `accum_iter` 个 batch 的梯度后再更新。

### 8.2 `NativeScaler` 的作用？

bf16 训练中，loss 可能非常小导致下溢。Loss scaler 会在反向传播前将 loss 放大，前向传播后再还原，保证梯度有效。

### 8.3 `find_unused_parameters=True` 的影响？

某些模块（如 Shot Adaptation）的输出可能不在 loss 中使用。设为 True 允许 DDP 跳过这些参数的同步，减少通信开销。
