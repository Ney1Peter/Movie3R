#!/bin/bash
set -e

# ============================================
# Human3R 4GPU 正式训练脚本 (30 epochs)
# ============================================

# 环境变量
export TORCH_HOME=/workspace/cache/torch
export TORCH_HUB_USE_HEURISTICS=0
export PYTHONUNBUFFERED=1

# NCCL 配置
export NCCL_DEBUG=INFO

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

# 创建输出目录
mkdir -p /workspace/code/Movie3R/experiments/avatarrex_zzr_lbn1

# ============================================
# 4GPU 正式训练 (30 epochs)
# ============================================
echo ""
echo "=========================================="
echo "启动 4GPU 正式训练"
echo "=========================================="
echo "Epochs: 30"
echo "Batch/GPU: 2"
echo "等效Batch: 8"
echo "Early Stopping: 10"
echo "预计时间: ~75-110 小时 (3-5天)"
echo "=========================================="

cd /workspace/code/Movie3R/src

python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29501 \
    train.py \
    epochs=30 \
    batch_size=2 \
    num_workers=0 \
    print_freq=50 \
    eval_freq=1 \
    early_stopping_patience=10 \
    print_img_freq=999999 \
    output_dir=../experiments/avatarrex_zzr_lbn1