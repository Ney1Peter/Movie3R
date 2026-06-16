#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/wangzheng/iJCV-CODE/Movie3R"
MPL_DIR="${ROOT}/output/tmp/mpl"

POSE_SESSION="v9_120h_pose_lora_gpu6"
POSE_HUMAN_SESSION="v9_120h_pose_human_lora_gpu7"

POSE_CMD="cd ${ROOT} && MPLCONFIGDIR=${MPL_DIR} CUDA_VISIBLE_DEVICES=6 .venv/bin/python src/train.py --config-name train_v9_120h_avatarrex_thuman_pose_lora_bs16"
POSE_HUMAN_CMD="cd ${ROOT} && MPLCONFIGDIR=${MPL_DIR} CUDA_VISIBLE_DEVICES=7 .venv/bin/python src/train.py --config-name train_v9_120h_avatarrex_thuman_pose_human_lora_bs16"

mkdir -p "${MPL_DIR}" "${ROOT}/output/v9_120h_mixed"

if [[ "${1:-}" != "--start" ]]; then
  cat <<EOF
Prepared V9 120h dual LoRA training commands.

Pose LoRA on GPU6:
  tmux new-session -d -s ${POSE_SESSION} "${POSE_CMD}"

Pose + human LoRA on GPU7:
  tmux new-session -d -s ${POSE_HUMAN_SESSION} "${POSE_HUMAN_CMD}"

Run this script with --start only when GPU6/GPU7 are ready:
  bash scripts/training/run_v9_120h_dual_lora.sh --start
EOF
  exit 0
fi

if tmux has-session -t "${POSE_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${POSE_SESSION}" >&2
  exit 1
fi

if tmux has-session -t "${POSE_HUMAN_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${POSE_HUMAN_SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${POSE_SESSION}" "${POSE_CMD}"
tmux new-session -d -s "${POSE_HUMAN_SESSION}" "${POSE_HUMAN_CMD}"

cat <<EOF
Started:
  ${POSE_SESSION}
  ${POSE_HUMAN_SESSION}

Monitor:
  tmux capture-pane -pt ${POSE_SESSION} -S -80
  tmux capture-pane -pt ${POSE_HUMAN_SESSION} -S -80
  nvidia-smi
EOF
