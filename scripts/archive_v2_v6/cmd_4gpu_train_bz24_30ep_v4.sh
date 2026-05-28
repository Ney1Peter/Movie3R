#!/bin/bash
set -euo pipefail

# ============================================
# Movie3R V4 4GPU 30 epoch training script
# 每卡 batch_size=24，全局 batch_size=96
# 用于在 10 epoch debug 有正向结果后继续验证 V4 收敛趋势
# ============================================

export TORCH_HOME=${TORCH_HOME:-/workspace/cache/torch}
export TORCH_HUB_USE_HEURISTICS=0
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-24}
EPOCHS=${EPOCHS:-30}
EVAL_FREQ=${EVAL_FREQ:-5}
# 30 epoch 观察完整收敛趋势，默认不让 early stopping 在第 10 epoch 提前截停。
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-999999}
MASTER_PORT=${MASTER_PORT:-29504}
RUN_NAME=${RUN_NAME:-training-4gpu-bz24-30ep-shot-v4-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/code/Movie3R/experiments/${RUN_NAME}}

# 如果调度系统已经设置了 CUDA_VISIBLE_DEVICES，则尊重调度系统设置。
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

echo "=========================================="
echo "Movie3R V4 4GPU 30 epoch training"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPUs:                 ${NUM_GPUS}"
echo "Epochs:               ${EPOCHS}"
echo "Batch/GPU:            ${BATCH_SIZE}"
echo "Global batch:         $((NUM_GPUS * BATCH_SIZE))"
echo "Eval frequency:       ${EVAL_FREQ}"
echo "Early stop patience:  ${EARLY_STOPPING_PATIENCE}"
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
    early_stopping_patience=${EARLY_STOPPING_PATIENCE} \
    print_img_freq=999999 \
    save_code=false \
    output_dir=${OUTPUT_DIR}
