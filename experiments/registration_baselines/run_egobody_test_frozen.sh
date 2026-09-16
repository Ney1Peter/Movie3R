#!/usr/bin/env bash
# Continue the frozen EgoBody Test protocol after the detached R0 supervisor.
# The ordering below is deliberate: no evaluator manifest or GT asset is
# opened until all R0--R3 predictions and transforms have been sealed.

set -euo pipefail

workspace=/data/wangzheng/iJCV-CODE
cd "$workspace"

python_bin=Movie3R/.venv/bin/python
output_root=Movie3R/output/shot3r_registration_baselines_v1_20260915
runtime_manifest="$output_root/work/frozen_manifests/egobody/egobody_cs150_test.runtime.jsonl"
evaluator_manifest="$output_root/work/frozen_manifests/egobody/egobody_cs150_test.evaluator.jsonl"
config="$output_root/config/frozen_egobody.json"
r0_root="$output_root/predictions"
gt_root="$output_root/work/gt_cache/egobody/test"
seal="$output_root/provenance/egobody_test_prediction_seal.json"
r0_supervisor_pid="${1:?R0 supervisor PID is required}"

export PYTHONPATH="$output_root/work/vendor:Movie3R"
export OMP_NUM_THREADS=4
export MALLOC_ARENA_MAX=2

while supervisor_state=$(ps -o stat= -p "$r0_supervisor_pid" 2>/dev/null) \
  && [[ "${supervisor_state:0:1}" != "Z" ]]; do
  count=$(find "$output_root/predictions/egobody/test" -maxdepth 1 -name '*.r0_scene.npz' 2>/dev/null | wc -l)
  printf 'waiting for sealed-input R0 caches: %s/129\n' "$count"
  sleep 60
done

r0_count=$(find "$output_root/predictions/egobody/test" -maxdepth 1 -name '*.r0_scene.npz' 2>/dev/null | wc -l)
if [[ "$r0_count" -ne 129 ]]; then
  printf 'R0 inference stopped with %s/129 caches; refusing to open Test evaluator inputs\n' "$r0_count" >&2
  exit 1
fi

fit_pids=()
for shard in 0 1 2 3 4 5; do
  "$python_bin" Movie3R/experiments/registration_baselines/fit.py \
    --runtime-manifest "$runtime_manifest" \
    --r0-root "$r0_root" \
    --output-root "$output_root" \
    --dataset egobody \
    --split test \
    --config "$config" \
    --manifest-sha256 8a5861bd3e4ee55dd1639c86526d21c96a73bb44fe07ff9848ef7b6b7645b02b \
    --shard-index "$shard" \
    --num-shards 6 \
    >"$output_root/logs/fit_egobody_test_worker${shard}.log" 2>&1 &
  fit_pids+=("$!")
done

fit_status=0
for pid in "${fit_pids[@]}"; do
  wait "$pid" || fit_status=1
done
if [[ "$fit_status" -ne 0 ]]; then
  printf 'At least one frozen Test registration shard failed; refusing to seal\n' >&2
  exit 1
fi

prediction_count=$(find "$output_root/predictions/egobody/test" -maxdepth 1 -name '*.npz' ! -name '*.r0_scene.npz' | wc -l)
if [[ "$prediction_count" -ne 129 ]]; then
  printf 'Registration produced %s/129 prediction files; refusing to seal\n' "$prediction_count" >&2
  exit 1
fi

"$python_bin" Movie3R/experiments/registration_baselines/seal_predictions.py \
  --runtime-manifest "$runtime_manifest" \
  --output-root "$output_root" \
  --config "$config" \
  --dataset egobody \
  --expected-cases 129 \
  >"$output_root/logs/seal_egobody_test.log" 2>&1

if [[ ! -f "$seal" ]]; then
  printf 'Prediction seal was not created; refusing to open Test GT\n' >&2
  exit 1
fi

# This is the first stage that opens the evaluator-only manifest and official
# Test body/calibration archives.
mkdir -p "$gt_root"
CUDA_VISIBLE_DEVICES=5 "$python_bin" Movie3R/versions/v20/egobody/prepare_gt.py \
  --manifest "$evaluator_manifest" \
  --outer-root data/OnlineHMR_work_v1/work_egobody/outer \
  --model-root Movie3R/src/models \
  --output-root "$gt_root" \
  --device cuda:0 \
  --batch-size 32 \
  --fail-fast \
  >"$output_root/logs/gt_egobody_test.log" 2>&1

gt_count=$(find "$gt_root" -maxdepth 1 -name '*.gt.npz' | wc -l)
if [[ "$gt_count" -ne 129 ]]; then
  printf 'GT generation produced %s/129 caches\n' "$gt_count" >&2
  exit 1
fi

eval_pids=()
for shard in 0 1 2 3 4 5; do
  "$python_bin" Movie3R/experiments/registration_baselines/evaluate.py \
    --runtime-manifest "$runtime_manifest" \
    --evaluator-manifest "$evaluator_manifest" \
    --prediction-root "$output_root/predictions" \
    --gt-root "$output_root/work/gt_cache" \
    --output-root "$output_root" \
    --dataset egobody \
    --split test \
    --shard-index "$shard" \
    --num-shards 6 \
    --prediction-seal "$seal" \
    >"$output_root/logs/evaluate_egobody_test_worker${shard}.log" 2>&1 &
  eval_pids+=("$!")
done

eval_status=0
for pid in "${eval_pids[@]}"; do
  wait "$pid" || eval_status=1
done
if [[ "$eval_status" -ne 0 ]]; then
  printf 'At least one frozen Test evaluator shard failed\n' >&2
  exit 1
fi

evaluation_count=$(find "$output_root/metrics/evaluations/egobody/test" -maxdepth 1 -name '*.evaluation.json' | wc -l)
if [[ "$evaluation_count" -ne 129 ]]; then
  printf 'Evaluator produced %s/129 reports\n' "$evaluation_count" >&2
  exit 1
fi

printf 'EgoBody frozen Test complete: R0=129, predictions=129, GT=129, evaluations=129\n'
