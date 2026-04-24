#!/bin/bash
# Human3R 多卡训练脚本
# 用法: ./train.sh [num_gpus] [epochs] [batch_size]
# 示例:
#   ./train.sh        # 自动检测空闲卡，默认1卡
#   ./train.sh 1      # 1卡
#   ./train.sh 2      # 2卡
#   ./train.sh 4      # 4卡
#   ./train.sh 0      # 查看GPU状态

set -e

# Torch hub offline mode - required for Dinov2Backbone even with pretrained=False
export TORCH_HOME=$HOME/.cache/torch
export TORCH_HUB_USE_HEURISTICS=0

NUM_GPUS=${1:-auto}
EPOCHS=${2:-1}
BATCH_SIZE=${3:-1}

# 查找空闲GPU（显存使用 < 2GB 的卡视为空闲）
find_free_gpus() {
    local free_gpus=()
    while read -r gpu_idx mem_used; do
        # 去掉 gpu_idx 末尾的逗号（如 "1," -> "1"）
        gpu_idx="${gpu_idx%,}"
        # mem_used 格式如 "1234MiB" 或 "0MiB"
        mem_num=$(echo "$mem_used" | grep -oP '\d+')
        if [ "$mem_num" -lt 2000 ]; then
            free_gpus+=("$gpu_idx")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null)
    echo "${free_gpus[@]}"
}

show_gpu_status() {
    echo "=========================================="
    echo "GPU 状态"
    echo "=========================================="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | \
    while IFS=, read -r idx name mem_used mem_total util; do
        mem_num=$(echo "$mem_used" | grep -oP '\d+' | head -1)
        if [ "$mem_num" -lt 2000 ]; then
            status="空闲"
        else
            status="占用中"
        fi
        printf "  GPU %s: %s (%s used / %s) [%s]\n" \
            "$(echo $idx | xargs)" \
            "$(echo $name | xargs)" \
            "$(echo $mem_used | xargs)" \
            "$(echo $mem_total | xargs)" \
            "$status"
    done
    echo "=========================================="
}

# 查看GPU状态
if [ "$NUM_GPUS" = "0" ]; then
    show_gpu_status
    exit 0
fi

# 自动检测
if [ "$NUM_GPUS" = "auto" ]; then
    FREE_GPUS=($(find_free_gpus))
    if [ ${#FREE_GPUS[@]} -eq 0 ]; then
        echo "错误: 没有检测到空闲GPU（< 2GB 使用中）"
        show_gpu_status
        exit 1
    fi
    NUM_GPUS=${#FREE_GPUS[@]}
    echo "自动检测到 ${NUM_GPUS} 张空闲GPU: ${FREE_GPUS[*]}"
fi

AVAILABLE_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "错误: 请求 ${NUM_GPUS} 卡但只有 ${AVAILABLE_GPUS} 张"
    exit 1
fi

echo "=========================================="
echo "Human3R 训练"
echo "=========================================="
echo "GPU数量:     ${NUM_GPUS}"
echo "Epochs:     ${EPOCHS}"
echo "Batch/卡:    ${BATCH_SIZE}"
echo "等效batch:   $((${BATCH_SIZE} * ${NUM_GPUS}))"
echo "=========================================="

# 查找并选择真正空闲的GPU
ALL_FREE_GPUS=($(find_free_gpus))
if [ ${#ALL_FREE_GPUS[@]} -lt "$NUM_GPUS" ]; then
    echo "错误: 请求 ${NUM_GPUS} 卡但只有 ${#ALL_FREE_GPUS[@]} 张空闲: ${ALL_FREE_GPUS[*]}"
    show_gpu_status
    exit 1
fi
# 取前 NUM_GPUS 个空闲卡
SELECTED_GPUS=$(IFS=','; echo "${ALL_FREE_GPUS[*]:0:$NUM_GPUS}")
echo "使用GPU:     ${SELECTED_GPUS}"
echo ""

cd "$(dirname "$0")/src"
source ../.venv/bin/activate

if [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES="$SELECTED_GPUS"
    echo "启动单卡训练..."
    python train.py \
        epochs=${EPOCHS} \
        batch_size=${BATCH_SIZE} \
        num_workers=0 \
        print_freq=50 \
        eval_freq=0 \
        output_dir=../experiments/avatarrex_zzr_lbn1
else
    # 多卡：torchrun 自己分配 GPU，不设置 CUDA_VISIBLE_DEVICES
    # 注意：多卡时也需要 num_workers=0 避免 /dev/shm 不足
    echo "启动多卡训练..."
    python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=29501 \
        train.py \
        epochs=${EPOCHS} \
        batch_size=${BATCH_SIZE} \
        num_workers=0 \
        print_freq=50 \
        eval_freq=0 \
        output_dir=../experiments/avatarrex_zzr_lbn1
fi
