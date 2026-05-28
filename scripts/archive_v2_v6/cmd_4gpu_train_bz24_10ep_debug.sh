#!/bin/bash
set -euo pipefail

# ============================================
# Movie3R 4GPU 快速测试训练脚本
# 每卡 batch_size=24，全局 batch_size=96
# 默认只跑 10 epochs，用于快速验证结构和 demo 质量
# ============================================

export TORCH_HOME=${TORCH_HOME:-/workspace/cache/torch}
export TORCH_HUB_USE_HEURISTICS=0
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-24}
EPOCHS=${EPOCHS:-10}
EVAL_FREQ=${EVAL_FREQ:-5}
MASTER_PORT=${MASTER_PORT:-29502}
# **========== 原始代码备份：V2 debug run name ==========**
# RUN_NAME=${RUN_NAME:-debug_training-4gpu-bz24-10ep-shot-v2-$(date +%Y%m%d-%H%M%S)}
# **========== 结束 ==========**
# **========== V3 当前代码备份：debug run name ==========**
# RUN_NAME=${RUN_NAME:-debug_training-4gpu-bz24-10ep-shot-v3-$(date +%Y%m%d-%H%M%S)}
# **========== 结束 ==========**
RUN_NAME=${RUN_NAME:-debug_training-4gpu-bz24-10ep-shot-v4-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/code/Movie3R/experiments/${RUN_NAME}}

# 如果调度系统已经设置了 CUDA_VISIBLE_DEVICES，则尊重调度系统设置。
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

echo "=========================================="
echo "Movie3R 4GPU debug training"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPUs:                 ${NUM_GPUS}"
echo "Epochs:               ${EPOCHS}"
echo "Batch/GPU:            ${BATCH_SIZE}"
echo "Global batch:         $((NUM_GPUS * BATCH_SIZE))"
echo "Eval frequency:       ${EVAL_FREQ}"
echo "num_workers:          0"
echo "Output dir:           ${OUTPUT_DIR}"
echo "Master port:          ${MASTER_PORT}"
echo "=========================================="

cd /workspace/code/Movie3R
source .venv/bin/activate

python - <<'PY'
import torch
print(f"Python CUDA available: {torch.cuda.is_available()}")
print(f"Visible GPU count: {torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(idx)
    print(f"GPU {idx}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")
PY

mkdir -p "${OUTPUT_DIR}"

cd /workspace/code/Movie3R/src

python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train.py \
    epochs=${EPOCHS} \
    batch_size=${BATCH_SIZE} \
    num_workers=0 \
    print_freq=50 \
    structured_log_freq=50 \
    eval_freq=${EVAL_FREQ} \
    early_stopping_patience=10 \
    print_img_freq=999999 \
    save_code=false \
    output_dir=${OUTPUT_DIR}
