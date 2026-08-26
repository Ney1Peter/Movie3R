#!/usr/bin/env bash
# Publication-only queue for the frozen Harmony4D multi-cut protocol.
# By default, a task starts only after an L20 reports both 0 MiB allocated and
# 0% GPU utilization.  A caller may explicitly permit safe shared use through
# MAX_EXISTING_GPU_MEMORY_MB and MAX_GPU_UTILIZATION_PCT.  Per-GPU flock files
# prevent this queue from putting two of its own workers on the same card.  It
# never terminates or otherwise interferes with another user's process.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 TASK [TASK ...]" >&2
  exit 2
fi

repo=/data/wangzheng/iJCV-CODE
runner="$repo/Movie3R/publication/bridge3r_iclr2027/run_harmony4d_multicut.py"
evaluator="$repo/Movie3R/publication/bridge3r_iclr2027/evaluate_harmony4d_multicut.py"
multi_manifest="$repo/Movie3R/publication/bridge3r_iclr2027/multicut/manifests/harmony4d_multicut_v1.jsonl"
nocut_manifest="$repo/Movie3R/publication/bridge3r_iclr2027/multicut/manifests/harmony4d_nocut_v1.jsonl"
runs="$repo/Movie3R/publication/bridge3r_iclr2027/multicut/runs"
locks="$runs/.gpu_locks"
max_existing_memory_mb="${MAX_EXISTING_GPU_MEMORY_MB:-0}"
max_gpu_utilization_pct="${MAX_GPU_UTILIZATION_PCT:-0}"
mkdir -p "$locks"

acquire_idle_gpu() {
  local gpu memory utilization lock
  selected_gpu=""
  while true; do
    while IFS=',' read -r gpu memory utilization; do
      gpu="${gpu//[[:space:]]/}"
      memory="${memory//[[:space:]]/}"
      utilization="${utilization//[[:space:]]/}"
      (( memory <= max_existing_memory_mb && utilization <= max_gpu_utilization_pct )) || continue
      lock="$locks/gpu_${gpu}.lock"
      exec 9>"$lock"
      flock -n 9 || { exec 9>&-; continue; }
      # Recheck while holding this queue's lock to avoid stale sampling.
      IFS=',' read -r memory utilization < <(
        nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
          --format=csv,noheader,nounits | head -n 1
      )
      memory="${memory//[[:space:]]/}"
      utilization="${utilization//[[:space:]]/}"
      if (( memory <= max_existing_memory_mb && utilization <= max_gpu_utilization_pct )); then
        selected_gpu="$gpu"
        return 0
      fi
      flock -u 9
      exec 9>&-
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    printf '%s waiting for a GPU within the shared-use limits (%s MiB, %s%%)\n' \
      "$(date -Is)" "$max_existing_memory_mb" "$max_gpu_utilization_pct" >&2
    sleep 60
  done
}

run_task() {
  local task="$1" line mode stage output manifest gpu
  case "$task" in
    multi3) line=3; mode=multi; stage=train_14_mma3; output=case_03.npz; manifest="$multi_manifest" ;;
    multi4) line=4; mode=multi; stage=train_15_mma4; output=case_04.npz; manifest="$multi_manifest" ;;
    nocut1) line=1; mode=nocut; stage=train_10_karate2; output=nocut_01.npz; manifest="$nocut_manifest" ;;
    nocut2) line=2; mode=nocut; stage=train_11_karate3; output=nocut_02.npz; manifest="$nocut_manifest" ;;
    nocut3) line=3; mode=nocut; stage=train_14_mma3; output=nocut_03.npz; manifest="$nocut_manifest" ;;
    nocut4) line=4; mode=nocut; stage=train_15_mma4; output=nocut_04.npz; manifest="$nocut_manifest" ;;
    *) echo "unknown task: $task" >&2; return 2 ;;
  esac
  if [[ -f "$runs/$output.runtime.json" ]]; then
    echo "$(date -Is) $task already has a runtime ledger; skipping"
    return 0
  fi
  # Do not use command substitution here: the file descriptor holding the
  # per-GPU flock must remain open until this task exits.
  acquire_idle_gpu
  gpu="$selected_gpu"
  echo "$(date -Is) starting $task on cuda:$gpu"
  local extra=()
  [[ "$mode" == nocut ]] && extra+=(--no-cut)
  "$repo/Movie3R/.venv/bin/python" "$runner" \
    --manifest "$manifest" --line "$line" \
    --extracted-root "$repo/data/Bridge3R_multicut_harmony4d/staging/$stage" \
    --output "$runs/$output" --device "cuda:$gpu" "${extra[@]}"
  if [[ "$mode" == multi ]]; then
    "$repo/Movie3R/.venv/bin/python" "$evaluator" \
      --cache "$runs/$output" --runtime-report "$runs/${output%.npz}.runtime.json" \
      --extracted-root "$repo/data/Bridge3R_multicut_harmony4d/staging/$stage" \
      --output "$runs/${output%.npz}.evaluation.json"
  fi
  echo "$(date -Is) completed $task"
  flock -u 9
  exec 9>&-
}

for task in "$@"; do
  run_task "$task"
done
