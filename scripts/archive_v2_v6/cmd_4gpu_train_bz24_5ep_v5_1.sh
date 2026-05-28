#!/bin/bash
set -euo pipefail

# ============================================
# Movie3R V5.1 single-GPU 5 epoch debug training script
# batch_size=24，用于单卡快速验证
# 用于快速验证 layerwise pose-shot adapter 与 jump/anchor loss
# ============================================

export TORCH_HOME=${TORCH_HOME:-/workspace/cache/torch}
export TORCH_HUB_USE_HEURISTICS=0
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# **========== 原始脚本备份：4GPU 默认配置 ==========**
# NUM_GPUS=${NUM_GPUS:-4}
# MASTER_PORT=${MASTER_PORT:-29505}
# RUN_NAME=${RUN_NAME:-debug_training-4gpu-bz24-5ep-shot-v5_1-$(date +%Y%m%d-%H%M%S)}
# **========== 结束 ==========**
NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-24}
EPOCHS=${EPOCHS:-5}
EVAL_FREQ=${EVAL_FREQ:-1}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-999999}
MASTER_PORT=${MASTER_PORT:-29515}
RUN_NAME=${RUN_NAME:-debug_training-1gpu-bz24-5ep-shot-v5_1-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/code/Movie3R/experiments/${RUN_NAME}}

# 单卡调试默认使用物理 0 号卡；如果这张卡在跑 V4，请运行前显式指定空闲卡：
# CUDA_VISIBLE_DEVICES=3 ./scripts/cmd_4gpu_train_bz24_5ep_v5_1.sh
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "=========================================="
echo "Movie3R V5.1 single-GPU 5 epoch debug training"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPUs:                 ${NUM_GPUS}"
echo "Epochs:               ${EPOCHS}"
echo "Batch size:           ${BATCH_SIZE}"
echo "Global batch:         ${BATCH_SIZE}"
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
