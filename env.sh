#!/bin/bash
# Human3R 环境激活脚本

# 1. 激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# 2. 配置当前服务器运行环境
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$SCRIPT_DIR/.venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="${TORCH_HOME:-/data/${USER}/.cache/torch}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-$TORCH_HOME/hub}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/${USER}/.cache/torch_extensions}"

# 3. 验证
echo "Human3R 环境已激活"
echo "   Python: $(which python) → $(python --version)"
echo "   PyTorch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null)"
