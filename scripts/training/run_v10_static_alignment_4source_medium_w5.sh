#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/wangzheng/iJCV-CODE/Movie3R"
MPL_DIR="${ROOT}/output/tmp/mpl"
GPU="${GPU:-6}"
SAMPLES_PER_SOURCE="${SAMPLES_PER_SOURCE:-200}"
STEPS="${STEPS:-2500}"
SESSION="v10_static_align_w5_medium_s${SAMPLES_PER_SOURCE}_gpu${GPU}"
OUT_DIR="${OUT_DIR:-${ROOT}/output/v10_static_alignment_probe/medium_4source_angle60_w5_s${SAMPLES_PER_SOURCE}}"
MANIFEST_MAP="${ROOT}/config/manifests/v10_static_alignment_4source_large_angle60/manifest_map.json"

CMD="cd ${ROOT} && MPLCONFIGDIR=${MPL_DIR} CUDA_VISIBLE_DEVICES=${GPU} .venv/bin/python scripts/v10_static_alignment_4source_probe.py --output_dir ${OUT_DIR} --manifest_map ${MANIFEST_MAP} --samples_per_source ${SAMPLES_PER_SOURCE} --steps ${STEPS} --body_frame_weight 1.0 --body_vector_weight 1.0 --body_anchor_weight 5.0 --body_vertical_weight 5.0 --log_every 100 --overwrite"

mkdir -p "${MPL_DIR}" "${ROOT}/output/v10_static_alignment_probe"

if [[ "${1:-}" != "--start" ]]; then
  cat <<EOF
Prepared V10 static streaming-alignment medium W5 probe.

GPU:
  ${GPU}

Samples:
  ${SAMPLES_PER_SOURCE} per source, 4 sources total.

Steps:
  ${STEPS}

Output:
  ${OUT_DIR}

Command:
  tmux new-session -d -s ${SESSION} "${CMD}"

Start:
  GPU=${GPU} SAMPLES_PER_SOURCE=${SAMPLES_PER_SOURCE} STEPS=${STEPS} bash scripts/training/run_v10_static_alignment_4source_medium_w5.sh --start
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

Output:
  ${OUT_DIR}

Monitor:
  tmux capture-pane -pt ${SESSION} -S -120
  nvidia-smi
EOF
