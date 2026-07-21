#!/usr/bin/env bash
set -uo pipefail

repo_root="/data/wangzheng/iJCV-CODE/Movie3R"
holdout_root="$repo_root/output/v29_rotation_rule_holdout3"
records="$holdout_root/records/holdout_records.jsonl"
log_root="$holdout_root/logs"
devices=(cuda:1 cuda:2 cuda:4 cuda:5)
mkdir -p "$log_root" "$holdout_root/empty_v14"

pids=()
for shard in 0 1 2 3; do
    "$repo_root/.venv/bin/python" \
        "$repo_root/scripts/v15_wide_baseline_boundary_bridge_candidates.py" \
        --records "$records" \
        --candidate_root "$holdout_root/v10_merged/cases" \
        --v14_candidate_dir "$holdout_root/empty_v14" \
        --output_dir "$holdout_root/v15" \
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
if [[ "$status" -ne 0 ]]; then
    exit "$status"
fi

pids=()
for shard in 0 1 2 3; do
    "$repo_root/.venv/bin/python" \
        "$repo_root/scripts/v16_human_torso_candidates.py" \
        --records "$records" \
        --candidate_root "$holdout_root/v10_merged/cases" \
        --v15_candidate_dir "$holdout_root/v15" \
        --output_dir "$holdout_root/v16" \
        --device "${devices[$shard]}" \
        --num_shards 4 \
        --shard_index "$shard" \
        --overwrite \
        >"$log_root/v16_shard_${shard}.log" 2>&1 &
    pids+=("$!")
done
status=0
for shard in 0 1 2 3; do
    if wait "${pids[$shard]}"; then
        printf 'v16 shard %s complete\n' "$shard"
    else
        printf 'v16 shard %s failed\n' "$shard" >&2
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    exit "$status"
fi

"$repo_root/.venv/bin/python" \
    "$repo_root/scripts/v25_holdout_rotation_validation.py" \
    --v15_dir "$holdout_root/v15" \
    --v16_dir "$holdout_root/v16" \
    --output_dir "$holdout_root/evaluation"

"$repo_root/.venv/bin/python" \
    "$repo_root/scripts/v27_holdout_failure_candidate_audit.py" \
    --records "$records" \
    --v15_dir "$holdout_root/v15" \
    --evaluation_report "$holdout_root/evaluation/v25_holdout_rotation_validation.json" \
    --output_dir "$holdout_root/failure_candidate_audit"
