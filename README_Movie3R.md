# Human3R + AvatarReX 训练环境配置

## 1. 环境要求

| 组件 | 版本要求 |
|------|---------|
| Python | **3.10**（必须） |
| CUDA | **12.4**（PyTorch 运行时通过 cu124 wheel 提供） |
| 系统 nvcc | **11.8**（仅用于编译 curope，非必须；H800 环境通常已自带） |
| 显卡驱动 | **≥ 525**（支持 CUDA 12.4；H800 服务器通常满足） |
| 显存 | **≥ 46GB**（如 L20 48GB、H800 80GB） |

> **重要说明**：
> - PyTorch 2.4.0 使用 `cu124` wheel，自带 CUDA 12.4 runtime 动态库。训练时 `torch.version.cuda` 返回 `12.4`。
> - 系统 `nvcc --version` 显示的是 NVIDIA toolkit 编译器版本（可能是 11.8），与 PyTorch 运行时的 CUDA 版本是**两个不同概念**。
> - curope（RoPE CUDA 加速）编译需要系统有 nvcc（11.8 或 12.x 均可），其他组件全部由 PyTorch wheel 提供。
> - **只要显卡驱动支持 CUDA 12.4，H800 服务器即可直接使用本项目的 PyTorch wheel，无需额外安装 CUDA toolkit。**

## 2. 使用 uv 创建虚拟环境

```bash
# 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # 或重新打开终端

# 创建 Python 3.10 虚拟环境
uv venv .venv --python 3.10

# 激活环境
source .venv/bin/activate

# 升级 pip
pip install --upgrade pip
```

## 3. 安装 PyTorch（CUDA 12.4）

```bash
# 安装 PyTorch 2.4.0 + CUDA 12.4（推荐）
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# 验证安装
python -c "import torch; print(f'torch {torch.__version__}, CUDA runtime {torch.version.cuda}')"
# 输出应为：torch 2.4.0+cu124, CUDA runtime 12.4

# 注意：nvcc 版本（系统编译器）可能显示 11.8，这是正常的，不影响 PyTorch 运行
nvcc --version 2>/dev/null || echo "nvcc not in PATH（无需在意，不影响训练）"
```

## 4. 安装其他依赖

```bash
# 安装不含 PyTorch 的依赖列表
pip install -r requirements_avatarrex.txt

# 安装 PyTorch 后自动解决大部分依赖
# 如有遗漏，手动安装
```

## 5. 编译 RoPE (curope)

**必须编译**，这是模型中 Positional Embedding 的 CUDA 实现。

```bash
cd src/croco/models/curope

# 编译（会自动检测 CUDA_ARCH，使用系统 nvcc）
python setup.py build

# 编译产物为 .so 文件，检查是否生成：
ls *.so
# 应该看到：curope.cpython-310-x86_64-linux-gnu.so
```

> **编译前提**：系统有 `nvcc`（11.8 或 12.x 均可）。H800 服务器通常自带 nvcc 11.8。
> **如遇编译错误**：确认 `nvcc` 在 PATH 中，执行 `which nvcc` 确认。
> 编译成功后，将 `.so` 文件路径加入 `PYTHONPATH` 或确保从 `src/` 目录启动训练。
> **其他组件**（模型推理、训练）不需要 nvcc，全部由 PyTorch cu124 提供。

## 6. 目录结构

```
Human3R/
├── src/                          # 源代码
│   ├── dust3r/                   # 核心模型
│   │   ├── datasets/             # 数据集（含 AvatarReX）
│   │   └── smpl_model.py         # SMPL/SMPLX 前向
│   ├── croco/                    # 预训练 backbone
│   │   └── models/curope/        # RoPE CUDA 实现
│   └── train.py                  # 训练入口
├── models/                       # SMPLX 模型权重
│   └── smplx/
├── config/
│   └── train.yaml               # 训练配置（全量微调）
├── .venv/                       # Python 虚拟环境
├── requirements_avatarrex.txt    # 依赖版本
└── README_AvatarReX.md           # 本文件
```

## 7. 数据准备

### AvatarReX 数据集

需要两组数据（来自 AvatarReX 官方）：

```
数据根目录/
├── AvatarRex4Human3R/Training/    # zzr 数据
│   └── {seq_id}/
│       ├── rgb/{frame:08d}.png
│       ├── cam/{frame:08d}.npz
│       ├── smpl/{frame:08d}.pkl
│       ├── depth/{frame:08d}.npy
│       └── mask/{frame:08d}.png
└── AvatarRex_lbn1_4Human3R/Training/  # lbn1 数据
    └── {seq_id}/ ...
```

**如需预处理**（原始 AvatarReX → 本格式），运行：
```bash
cd /path/to/AvatarReX/dataset
python /path/to/Human3R/datasets_preprocess/preprocess_avatarrex_fast.py \
    -i ./raw_data \
    -o ./AvatarRex4Human3R

# 生成深度图（需要 Depth-Anything-3）
python /path/to/Human3R/datasets_preprocess/generate_depth_avatarrex.py \
    --root ./AvatarRex4Human3R \
    --da3_root /path/to/Depth-Anything-3
```

## 8. 训练配置

### train.yaml 关键参数

```yaml
# 全量微调（freeze=none）
model: ARCroco3DStereo(ARCroco3DStereoConfig(freeze='none', ...))

# 数据集（AvatarReX Video + AABB）
train_dataset: 2000 @ AvatarReX_Video(zzr)
            + 2000 @ AvatarReX_Video(lbn1)
            + 2000 @ AvatarReX_AABB(zzr)
            + 2000 @ AvatarReX_AABB(lbn1)

# 数据路径（根据实际修改）
# AvatarReX_Video: ROOT="../../../Movie3R-dataset/AvatarRex4Human3R"
# AvatarReX_AABB:   ROOT="../../../Movie3R-dataset/AvatarRex4Human3R"
```

### 数据集采样格式

| 类型 | 4视图内容 |
|------|---------|
| **Video** | 相机A @ t, t+1, t+2, t+3（同一相机连续帧） |
| **AABB** | 相机A @ t, t+1, 相机B @ t+2, t+3（跳视角但时间连续） |

## 9. 启动训练

### 单卡训练

```bash
cd src

# 全量微调（batch_size=1 在 46GB 显卡上刚好够）
python train.py \
    epochs=1 \
    batch_size=1 \
    print_freq=50 \
    eval_freq=0 \
    output_dir=../experiments/avatarrex_zzr_lbn1
```

### 多卡训练（推荐）

```bash
cd src

# 2卡
python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 \
    train.py epochs=1 batch_size=1 print_freq=50 eval_freq=0

# 4卡
python -m torch.distributed.run --nproc_per_node=4 --master_port=29501 \
    train.py epochs=1 batch_size=1 print_freq=50 eval_freq=0
```

> **显存估算**：全量微调 batch_size=1 单卡约 42-43GB。4卡各 batch_size=1 等效 batch_size=4，速度快约 4 倍。

### 使用训练脚本（自动选择空闲GPU）

```bash
# 1卡（自动检测空闲卡）
./train.sh

# 2卡（自动选择2张空闲卡）
./train.sh 2

# 查看GPU状态
./train.sh 0
```

## 10. 已知问题与解决

### SMPL 坐标参考系

AvatarReX 的 `smplx_transl` 存储于 **mocap 世界坐标系**，而非相机坐标系。代码中已做正确变换：
- mocap → 相机坐标：`smpl_cam = R_c2w.T @ (smpl_world - t_c2w)`
- AABB 过滤条件：`camera_z > -0.5m`

### OOM（显存不足）

- 方案1：减小 batch_size（单卡 batch_size=1 是最小）
- 方案2：切换为**头部微调**（只训练 decoder+head，显存约 29GB）
  ```yaml
  model: ARCroco3DStereo(ARCroco3DStereoConfig(freeze='encoder_and_decoder_and_head', ...))
  ```

## 11. 环境快速验证

```bash
cd src

# 验证 Python 和 torch
python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')"

# 验证 curope (RoPE)
python -c "from croco.models.curope import curope2d; print('curope OK')"

# 验证 SMPLX
python -c "import smplx; print('smplx OK')"

# 验证数据集加载
python -c "
import sys; sys.path.insert(0, '.')
from dust3r.datasets.avatarrex import AvatarReX_Video
ds = AvatarReX_Video(ROOT='/path/to/AvatarRex4Human3R', split='Training')
print(f'AvatarReX_Video: {len(ds)} samples OK')
"
```

## 12. 训练输出

```
experiments/avatarrex_zzr_lbn1/
├── checkpoints/      # 模型权重
├── logs/             # TensorBoard 日志
└── code/             # 代码备份（训练时自动保存）
```

## 13. H800 服务器特别说明

H800 80GB 服务器部署要点：

| 项目 | 要求 |
|------|------|
| 驱动 | ≥ 525（支持 CUDA 12.4） |
| Python | 3.10 |
| PyTorch | `torch==2.4.0` + `torchvision==0.19.0` + **cu124** |
| 显存 | 80GB，batch_size=2 轻松跑满 |
| 多卡 | 推荐 4×batch_size=2=8，NVLink 利用率高 |
| Dinov2 backbone | 离线模式：`export TORCH_HOME=$HOME/.cache/torch` |

**完整环境搭建流程（H800）**：

```bash
# 1. 创建环境
uv venv .venv --python 3.10
source .venv/bin/activate

# 2. 安装 PyTorch CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# 3. 安装其他依赖
pip install -r requirements_Movie3R.txt

# 4. 编译 curope（需要系统有 nvcc）
cd src/croco/models/curope && python setup.py build && cd ../../../..

# 5. 启动训练
./train.sh 4 40 2   # 4卡，每卡 batch=2，等效 batch=8
```
