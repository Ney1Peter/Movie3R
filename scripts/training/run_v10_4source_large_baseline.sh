#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/wangzheng/iJCV-CODE/Movie3R"
MPL_DIR="${ROOT}/output/tmp/mpl"
CONFIG="train_v10_4source_angle60_large_pose_concat_human_mean_lora_bs10"
GPU="${GPU:-7}"
SESSION="v10_4source_large_baseline_gpu${GPU}"

CMD="cd ${ROOT} && MPLCONFIGDIR=${MPL_DIR} CUDA_VISIBLE_DEVICES=${GPU} .venv/bin/python src/train.py --config-name ${CONFIG}"

mkdir -p "${MPL_DIR}" "${ROOT}/output/v10_4source_large_baseline"

if [[ "${1:-}" != "--start" ]]; then
  cat <<EOF
Prepared V10 four-source large baseline training.

Config:
  ${CONFIG}

GPU:
  ${GPU}

Command:
  tmux new-session -d -s ${SESSION} "${CMD}"

Start only when the target GPU is ready:
  GPU=${GPU} bash scripts/training/run_v10_4source_large_baseline.sh --start
EOF
  exit 0
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" "${CMD}"

cat <<EOF
Started:
  ${SESSION}

Monitor:
  tmux capture-pane -pt ${SESSION} -S -120
  nvidia-smi
EOF
