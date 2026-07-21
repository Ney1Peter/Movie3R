#!/usr/bin/env bash
set -uo pipefail

repo_root="/data/wangzheng/iJCV-CODE/Movie3R"
holdout_root="$repo_root/output/v25_holdout_rotation_validation"
records="$holdout_root/records/holdout_records.jsonl"
candidate_root="$holdout_root/v10_merged/cases"
output_root="$holdout_root/v15"
empty_v14="$holdout_root/empty_v14"
log_root="$holdout_root/logs"
mkdir -p "$output_root" "$empty_v14" "$log_root"

devices=(cuda:1 cuda:2 cuda:4 cuda:5)
pids=()
for shard in 0 1 2 3; do
    "$repo_root/.venv/bin/python" \
        "$repo_root/scripts/v15_wide_baseline_boundary_bridge_candidates.py" \
        --records "$records" \
        --candidate_root "$candidate_root" \
        --v14_candidate_dir "$empty_v14" \
        --output_dir "$output_root" \
        --device "${devices[$shard]}" \
        --num_shards 4 \
        --shard_index "$shard" \
        --overwrite \
        >"$log_root/v15_shard_${shard}.log" 2>&1 &
    pids+=("$!")
done

status=0
for shard in 0 1 2 3; do
    if wait "${pids[$shard]}"; then
        printf 'v15 shard %s complete\n' "$shard"
    else
        printf 'v15 shard %s failed; inspect %s\n' \
            "$shard" "$log_root/v15_shard_${shard}.log" >&2
        status=1
    fi
done
exit "$status"
