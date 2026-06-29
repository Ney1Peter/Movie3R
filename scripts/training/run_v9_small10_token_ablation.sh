#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/wangzheng/iJCV-CODE/Movie3R"
MPL_DIR="${ROOT}/output/tmp/mpl"

declare -A CONFIGS=(
  [v9_small10_ablate_all_mean]="train_v9_small10_ablate_all_mean"
  [v9_small10_ablate_single_token]="train_v9_small10_ablate_single_token"
  [v9_small10_ablate_no_semantic]="train_v9_small10_ablate_no_semantic"
  [v9_small10_ablate_no_alignment]="train_v9_small10_ablate_no_alignment"
  [v9_small10_ablate_no_momentum]="train_v9_small10_ablate_no_momentum"
  [v9_small10_ablate_learned_pooling]="train_v9_small10_ablate_learned_pooling"
)

declare -A GPUS=(
  [v9_small10_ablate_all_mean]=1
  [v9_small10_ablate_single_token]=1
  [v9_small10_ablate_no_semantic]=2
  [v9_small10_ablate_no_alignment]=2
  [v9_small10_ablate_no_momentum]=3
  [v9_small10_ablate_learned_pooling]=3
)

ORDER=(
  v9_small10_ablate_all_mean
  v9_small10_ablate_single_token
  v9_small10_ablate_no_semantic
  v9_small10_ablate_no_alignment
  v9_small10_ablate_no_momentum
  v9_small10_ablate_learned_pooling
)

mkdir -p "${MPL_DIR}" "${ROOT}/output/v9_small10_token_ablation"

print_commands() {
  cat <<EOF
Prepared V9 small10 correct-token ablation runs.

Each run uses the same fixed 7 AvatarReX + 3 THuman clips, original Human3R
initialization, pose+human LoRA, and 150 epochs / 1050 optimizer steps.

EOF
  for name in "${ORDER[@]}"; do
    local gpu="${GPUS[$name]}"
    local cfg="${CONFIGS[$name]}"
    cat <<EOF
${name} on GPU${gpu}:
  tmux new-session -d -s ${name} "cd ${ROOT} && MPLCONFIGDIR=${MPL_DIR} CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=src:. .venv/bin/python src/train.py --config-name ${cfg}"

EOF
  done
  cat <<EOF
Start all:
  bash scripts/training/run_v9_small10_token_ablation.sh --start

Monitor:
  tmux capture-pane -pt v9_small10_ablate_all_mean -S -80
  tmux capture-pane -pt v9_small10_ablate_single_token -S -80
  tmux capture-pane -pt v9_small10_ablate_no_semantic -S -80
  tmux capture-pane -pt v9_small10_ablate_no_alignment -S -80
  tmux capture-pane -pt v9_small10_ablate_no_momentum -S -80
  tmux capture-pane -pt v9_small10_ablate_learned_pooling -S -80
  nvidia-smi
EOF
}

if [[ "${1:-}" != "--start" ]]; then
  print_commands
  exit 0
fi

for name in "${ORDER[@]}"; do
  if tmux has-session -t "${name}" 2>/dev/null; then
    echo "tmux session already exists: ${name}" >&2
    exit 1
  fi
done

for name in "${ORDER[@]}"; do
  gpu="${GPUS[$name]}"
  cfg="${CONFIGS[$name]}"
  cmd="cd ${ROOT} && MPLCONFIGDIR=${MPL_DIR} CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=src:. .venv/bin/python src/train.py --config-name ${cfg}"
  tmux new-session -d -s "${name}" "${cmd}"
done

cat <<EOF
Started V9 small10 token ablations:
EOF
for name in "${ORDER[@]}"; do
  echo "  ${name} on GPU${GPUS[$name]}"
done
