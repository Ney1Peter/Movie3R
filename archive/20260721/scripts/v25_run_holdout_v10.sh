#!/usr/bin/env bash
set -uo pipefail

repo_root="/data/wangzheng/iJCV-CODE/Movie3R"
manifest="$repo_root/output/v25_holdout_rotation_validation/records/manifest_map.json"
output_root="$repo_root/output/v25_holdout_rotation_validation/v10"
log_root="$repo_root/output/v25_holdout_rotation_validation/logs"
mkdir -p "$output_root" "$log_root"

sources=(avatarrex thuman mvhuman100 mvhuman200)
devices=(cuda:1 cuda:2 cuda:4 cuda:5)
pids=()

for index in "${!sources[@]}"; do
    source_name="${sources[$index]}"
    device="${devices[$index]}"
    "$repo_root/.venv/bin/python" \
        "$repo_root/scripts/v10_oracle_candidate_selection_probe.py" \
        --manifest_map "$manifest" \
        --sources "$source_name" \
        --samples_per_bucket 8 \
        --seed 20260722 \
        --device "$device" \
        --output_dir "$output_root/$source_name" \
        --overwrite \
        >"$log_root/v10_${source_name}.log" 2>&1 &
    pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'v10 %s complete\n' "${sources[$index]}"
    else
        printf 'v10 %s failed; inspect %s\n' \
            "${sources[$index]}" "$log_root/v10_${sources[$index]}.log" >&2
        status=1
    fi
done
exit "$status"
