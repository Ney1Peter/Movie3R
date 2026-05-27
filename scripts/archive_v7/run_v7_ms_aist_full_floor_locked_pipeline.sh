#!/usr/bin/env bash
set -euo pipefail

MOVIE3R_ROOT="/data/wangzheng/iJCV-CODE/Movie3R"
DATASET_ROOT="/data/wangzheng/iJCV-CODE/Movie3R-dataset"
DATA_ROOT="/data/wangzheng/iJCV-CODE/data"

REFINED_ROOT="${DATA_ROOT}/data-V7-shot-change-clips-refined/ms-aist"
SHOT_CLIP_ROOT="${DATA_ROOT}/data-V7-shot-change-clips/ms-aist"
STAGE_A_ROOT="${DATA_ROOT}/data-V7-stage-a/ms-aist/full_floor_locked_human"
FULL_INPUT_MANIFEST="${STAGE_A_ROOT}/all_refined_accepted_manifest.json"

REFINE_PY="${DATASET_ROOT}/.venv_data/bin/python"
MOVIE3R_PY="${MOVIE3R_ROOT}/.venv/bin/python"

mkdir -p "${STAGE_A_ROOT}/logs"

if [[ ! -f "${REFINED_ROOT}/shot3_30f_manifest.json" ]]; then
  "${REFINE_PY}" "${DATASET_ROOT}/scripts/archive_v7/refine_shot2_boundary_clips.py" \
    --clip_root "${SHOT_CLIP_ROOT}/videos/shot3" \
    --source_manifest "${SHOT_CLIP_ROOT}/manifest.json" \
    --output_root "${REFINED_ROOT}" \
    --split_name shot3_30f
fi

if [[ ! -f "${REFINED_ROOT}/shot4_30f_manifest.json" ]]; then
  "${REFINE_PY}" "${DATASET_ROOT}/scripts/archive_v7/refine_shot2_boundary_clips.py" \
    --clip_root "${SHOT_CLIP_ROOT}/videos/shot4" \
    --source_manifest "${SHOT_CLIP_ROOT}/manifest.json" \
    --output_root "${REFINED_ROOT}" \
    --split_name shot4_30f
fi

cd "${MOVIE3R_ROOT}"
PYTHONPATH=src:scripts:scripts/archive_v7:. "${MOVIE3R_PY}" scripts/archive_v7/build_v7_manifest_from_refined_clips.py \
  --refined_manifest "${REFINED_ROOT}/shot2_30f_manifest.json" \
  --refined_manifest "${REFINED_ROOT}/shot3_30f_manifest.json" \
  --refined_manifest "${REFINED_ROOT}/shot4_30f_manifest.json" \
  --output_manifest "${FULL_INPUT_MANIFEST}" \
  --require_existing_videos

PYTHONPATH=src:scripts:scripts/archive_v7:. "${MOVIE3R_PY}" scripts/archive_v7/build_v7_floor_locked_human_stage_a.py \
  --input_manifest "${FULL_INPUT_MANIFEST}" \
  --output_root "${STAGE_A_ROOT}" \
  --model_path src/human3r_896L.pth \
  --target_count 3 \
  --device cuda \
  --align_device cuda \
  --token_device cuda
