#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/wangzheng/iJCV-CODE/Movie3R"
MPL_DIR="${ROOT}/output/tmp/mpl"

EXP_A_SESSION="v9_60h_h3_imp075_gpu5"
EXP_B_SESSION="v9_60h_h3_hcam_ref_gpu6"
EXP_C_SESSION="v9_60h_h3_hcam_pair_gpu7"

EXP_A_CMD="cd ${ROOT} && MPLCONFIGDIR=${MPL_DIR} CUDA_VISIBLE_DEVICES=5 .venv/bin/python src/train.py --config-name train_v9_60h_h3_imp075_pose_human_lora_bs10"
EXP_B_CMD="cd ${ROOT} && MPLCONFIGDIR=${MPL_DIR} CUDA_VISIBLE_DEVICES=6 .venv/bin/python src/train.py --config-name train_v9_60h_h3_hcam_ref_pose_human_lora_bs10"
EXP_C_CMD="cd ${ROOT} && MPLCONFIGDIR=${MPL_DIR} CUDA_VISIBLE_DEVICES=7 .venv/bin/python src/train.py --config-name train_v9_60h_h3_hcam_ref_pairwise_pose_human_lora_bs10"

mkdir -p "${MPL_DIR}" "${ROOT}/output/v9_60h_loss_followup"

if [[ "${1:-}" != "--start" ]]; then
  cat <<EOF
Prepared V9 60h loss follow-up commands.

A. H3 improvement 0.075 on GPU5:
  tmux new-session -d -s ${EXP_A_SESSION} "${EXP_A_CMD}"

B. H3 + human-camera reference loss on GPU6:
  tmux new-session -d -s ${EXP_B_SESSION} "${EXP_B_CMD}"

C. H3 + human-camera reference + pairwise motion loss on GPU7:
  tmux new-session -d -s ${EXP_C_SESSION} "${EXP_C_CMD}"

Start all three only when GPU5/GPU6/GPU7 are ready:
  bash scripts/training/run_v9_60h_loss_followup.sh --start
EOF
  exit 0
fi

for session in "${EXP_A_SESSION}" "${EXP_B_SESSION}" "${EXP_C_SESSION}"; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 1
  fi
done

tmux new-session -d -s "${EXP_A_SESSION}" "${EXP_A_CMD}"
tmux new-session -d -s "${EXP_B_SESSION}" "${EXP_B_CMD}"
tmux new-session -d -s "${EXP_C_SESSION}" "${EXP_C_CMD}"

cat <<EOF
Started:
  ${EXP_A_SESSION}
  ${EXP_B_SESSION}
  ${EXP_C_SESSION}

Monitor:
  tmux capture-pane -pt ${EXP_A_SESSION} -S -80
  tmux capture-pane -pt ${EXP_B_SESSION} -S -80
  tmux capture-pane -pt ${EXP_C_SESSION} -S -80
  nvidia-smi
EOF
