#!/usr/bin/env bash
set -uo pipefail

repo_root="/data/wangzheng/iJCV-CODE/Movie3R"
holdout_root="$repo_root/output/v27_consensus_holdout2"
records="$holdout_root/records/holdout_records.jsonl"
candidate_root="$holdout_root/v10_merged/cases"
output_root="$holdout_root/v15"
log_root="$holdout_root/logs"
mkdir -p "$output_root" "$holdout_root/empty_v14" "$log_root"

devices=(cuda:1 cuda:2 cuda:4 cuda:5)
pids=()
for shard in 0 1 2 3; do
    "$repo_root/.venv/bin/python" \
        "$repo_root/scripts/v15_wide_baseline_boundary_bridge_candidates.py" \
        --records "$records" \
        --candidate_root "$candidate_root" \
        --v14_candidate_dir "$holdout_root/empty_v14" \
        --output_dir "$output_root" \
        --device "${devices[$shard]}" \
        --enable_da3_correspondence \
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
        printf 'v15 shard %s failed\n' "$shard" >&2
        status=1
    fi
done
exit "$status"
