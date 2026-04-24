#!/bin/bash
set -e

# ============================================
# Human3R 1GPU 测试脚本
# ============================================

# 环境变量
export TORCH_HOME=/workspace/cache/torch
export TORCH_HUB_USE_HEURISTICS=0
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "环境检查"
echo "=========================================="

# 激活虚拟环境
cd /workspace/code/Movie3R
source .venv/bin/activate

# Python 信息
echo "Python: $(which python)"
python --version
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# 路径检查
echo ""
echo "=========================================="
echo "路径检查"
echo "=========================================="
echo "Data path: /workspace/data"
[ -d /workspace/data ] && echo "  ✓ 存在" || echo "  ✗ 不存在"

echo "Torch cache: $TORCH_HOME"
[ -d "$TORCH_HOME" ] && echo "  ✓ 存在" || echo "  ✗ 不存在"

echo "Project dir: /workspace/code/Movie3R"
[ -d /workspace/code/Movie3R ] && echo "  ✓ 存在" || echo "  ✗ 不存在"

echo "Train code: /workspace/code/Movie3R/src"
[ -d /workspace/code/Movie3R/src ] && echo "  ✓ 存在" || echo "  ✗ 不存在"

# ============================================
# 1GPU 训练测试 (100 steps 快速验证)
# ============================================
echo ""
echo "=========================================="
echo "启动 1GPU 训练测试 (100 steps)"
echo "预计时间: ~7 分钟"
echo "=========================================="

cd /workspace/code/Movie3R/src
export CUDA_VISIBLE_DEVICES=0

mkdir -p ../experiments/avatarrex_zzr_lbn1_test

python train.py \
    epochs=1 \
    batch_size=1 \
    num_workers=0 \
    print_freq=10 \
    eval_freq=0 \
    output_dir=../experiments/avatarrex_zzr_lbn1_test