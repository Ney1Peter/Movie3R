#!/bin/bash
set -euo pipefail

# ============================================
# Movie3R 4GPU 正式训练脚本
# 每卡 batch_size=24，全局 batch_size=96
# ============================================

export TORCH_HOME=${TORCH_HOME:-/workspace/cache/torch}
export TORCH_HUB_USE_HEURISTICS=0
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

NUM_GPUS=4
BATCH_SIZE=24
EPOCHS=${EPOCHS:-30}
MASTER_PORT=${MASTER_PORT:-29501}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/code/Movie3R/experiments/formal_training-4gpu-bz24-shot-v2}

# 如果调度系统已经设置了 CUDA_VISIBLE_DEVICES，则尊重调度系统设置。
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

echo "=========================================="
echo "Movie3R 4GPU training"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPUs:                 ${NUM_GPUS}"
echo "Epochs:               ${EPOCHS}"
echo "Batch/GPU:            ${BATCH_SIZE}"
echo "Global batch:         $((NUM_GPUS * BATCH_SIZE))"
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
    eval_freq=1 \
    early_stopping_patience=10 \
    print_img_freq=999999 \
    save_code=false \
    output_dir=${OUTPUT_DIR}
