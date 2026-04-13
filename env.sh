#!/bin/bash
# Human3R 环境激活脚本

# 1. 激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# 2. 验证
echo "Human3R 环境已激活"
echo "   Python: $(which python) → $(python --version)"
echo "   PyTorch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null)"
