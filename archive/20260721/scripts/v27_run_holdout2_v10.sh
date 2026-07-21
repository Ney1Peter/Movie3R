#!/usr/bin/env bash
set -uo pipefail

repo_root="/data/wangzheng/iJCV-CODE/Movie3R"
holdout_root="$repo_root/output/v27_consensus_holdout2"
manifest="$holdout_root/records/manifest_map.json"
output_root="$holdout_root/v10"
log_root="$holdout_root/logs"
mkdir -p "$output_root" "$log_root"

sources=(avatarrex thuman mvhuman100 mvhuman200)
devices=(cuda:1 cuda:2 cuda:4 cuda:5)
pids=()
for index in "${!sources[@]}"; do
    source="${sources[$index]}"
    "$repo_root/.venv/bin/python" \
        "$repo_root/scripts/v10_oracle_candidate_selection_probe.py" \
        --manifest_map "$manifest" \
        --sources "$source" \
        --samples_per_bucket 8 \
        --seed 20260723 \
        --device "${devices[$index]}" \
        --output_dir "$output_root/$source" \
        --overwrite \
        >"$log_root/v10_${source}.log" 2>&1 &
    pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'v10 %s complete\n' "${sources[$index]}"
    else
        printf 'v10 %s failed\n' "${sources[$index]}" >&2
        status=1
    fi
done
exit "$status"
