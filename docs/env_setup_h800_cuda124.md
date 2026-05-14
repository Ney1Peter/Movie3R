# Movie3R 环境配置文档

## 环境概览

| 项目 | 版本/信息 |
|------|----------|
| GPU | NVIDIA H800 (80GB) |
| CUDA 驱动 | 560.35.03 |
| 系统 | Debian (bookworm) |
| Python | 3.11 |
| PyTorch | 2.4.0 (CUDA 12.4) |
| CUDA Toolkit | 12.4 |

## 配置步骤

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

### 2. 配置 pip 镜像源（永久）

```bash
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml << 'EOF'
[pip]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
EOF
```

### 3. 创建虚拟环境

```bash
cd /workspace/code/Movie3R
uv venv .venv --python 3.11
source .venv/bin/activate
```

### 4. 安装 PyTorch

```bash
uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
```

### 5. 安装基础依赖

```bash
uv pip install numpy==1.26.4
uv pip install pip
```

### 6. 安装项目依赖

```bash
uv pip install accelerate einops gradio h5py "huggingface-hub[torch]>=0.22" hydra-core lpips OpenEXR==3.4.9 "pyglet<2" pyrender pyvista roma scikit-learn scipy smplx spaces tensorboard tqdm transformers trimesh viser gsplat --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

> 注意：gsplat 从清华镜像安装即可，无需从 GitHub 安装

### 7. 安装 chumpy（GitHub）

```bash
uv pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17 --no-build-isolation
```

### 8. 系统依赖（apt）

```bash
# 配置清华镜像源
cat > /etc/apt/sources.list << 'EOF'
deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free
deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free
deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free
EOF

apt-get update
apt-get install -y g++ gcc python3.11-dev libxml2 libgl1-mesa-glx libglib2.0-0
```

### 9. 安装 CUDA Toolkit（用于编译 RoPE kernel）

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
chmod +x cuda_12.4.0_550.54.14_linux.run
./cuda_12.4.0_550.54.14_linux.run --silent --toolkit --override
```

### 10. 编译 curope（RoPE CUDA kernel）

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
cd /workspace/code/Movie3R/src/croco/models/curope/
python setup.py build_ext --inplace
```

### 11. 设置环境变量（永久）

```bash
# CUDA
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc

# torch lib（运行时库路径，根据实际 venv 位置调整）
echo 'export LD_LIBRARY_PATH=$WORKING_DIR/.venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Dinov2 backbone 缓存（训练需要）
echo 'export TORCH_HOME=$HOME/.cache/torch' >> ~/.bashrc

source ~/.bashrc
```

> 注意：将 `$WORKING_DIR` 替换为实际项目路径，如 `/workspace/code/Movie3R`

### 12. 安装评估工具

```bash
uv pip install evo open3d --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 验证安装

```bash
# 检查 curope
python -c "import sys; sys.path.insert(0, 'src/croco/models/curope'); import curope; print('curope OK')"

# 检查 PyTorch + GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
```

## 下载模型

```bash
# SMPLX
bash scripts/fetch_smplx.sh

# Human3R 权重
huggingface-cli download faneggg/human3r human3r_896L.pth --local-dir ./src
```

## Dinov2 backbone 缓存

训练需要 Dinov2 权重，缓存位置 `$TORCH_HOME`（默认 `~/.cache/torch`）。

如果网络通畅，训练时会自动从 GitHub 下载。如果网络不通，需要从有缓存的服务器拷贝：

```bash
# 拷贝 Dinov2 缓存到目标服务器
scp -r user@源服务器:/root/.cache/torch/hub /root/.cache/torch/
```

验证缓存是否完整：
```bash
find ~/.cache/torch -name "*dinov2*" | head -5
# 应看到 facebookresearch_dinov2_main 目录
```

## 各组件关系说明

| 组件 | 作用 |
|------|------|
| gcc/g++ | 编译 C/C++ CPU 代码 |
| nvcc | NVIDIA CUDA 编译器，编译 GPU 代码 |
| CUDA Toolkit | 包含 nvcc、CUDA 头文件、CUDA 库 |
| PyTorch wheel | 预编译的 Python 库，含 CUDA 运行时库 |
| ninja/cmake | 构建工具，用于编译 curope |

## 镜像源汇总

| 用途 | 镜像 |
|------|------|
| pip 包 | https://pypi.tuna.tsinghua.edu.cn/simple |
| PyTorch | https://download.pytorch.org/whl/cu124 |
| Debian 系统包 | https://mirrors.tuna.tsinghua.edu.cn/debian |
