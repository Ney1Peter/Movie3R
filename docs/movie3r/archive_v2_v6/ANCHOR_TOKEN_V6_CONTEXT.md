# Movie3R AnchorToken V6 Context Handoff

Last updated: 2026-05-14

This file is the compact context entry point for a new chat/window. Read this first before touching code or outputs.

## Current Goal

We are designing and validating Movie3R/Human3R ShotToken V6 for multi-shot cinematic human reconstruction. The current direction is to replace unsafe global shot tokens with local scene anchors.

The concrete idea is:

- Use XFeat semi-dense matching to propose cross-view/cross-shot correspondences.
- Use RICH official static scene mesh + camera calibration as a teacher/validator to keep only true static-scene anchor matches.
- Map those verified 2D anchors into Human3R encoder patch tokens.
- Show that Human3R encoder tokens still preserve correspondence signal at those anchor locations.
- Later use cached AnchorTokens to guide pose/camera adaptation without modifying the encoder or appending anchors to the full decoder token sequence.

## User Preferences And Constraints

- The user wants visual, presentation-ready evidence at every step.
- The current report should be clean and minimal, not cluttered overlays.
- The user prefers independent diagnosis/visualization scripts first, not invasive model-path changes.
- Do not modify the encoder for the first implementation.
- Do not insert anchor tokens into the complete decoder token sequence for the first implementation.
- Training should use offline anchor caches/manifests, not online XFeat+mesh generation.
- Inference cannot use RICH mesh, so inference anchors must come from online XFeat semi-dense matching plus lightweight geometry/confidence filtering.
- AABB format is fixed as `[A@t, A@t+1, B@t+2, B@t+3]`.
- The key shot-boundary pair is `A@t+1 -> B@t+2`.
- Project code convention from `CLAUDE.md`: when modifying code, preserve old code in comment blocks instead of deleting it unless the user explicitly asks otherwise.

## Important Roots

- Repo: `/workspace/code/Movie3R`
- Python env: `/workspace/code/Movie3R/.venv/bin/python`
- Human3R checkpoint: `/workspace/code/Movie3R/src/human3r_896L.pth`
- RICH processed root: `/workspace/data/RICH/RICH_4Human3R/Training`
- RICH raw/calibration root: `/workspace/data/RICH`
- RICH static mesh: `/workspace/data/RICH/scan_calibration/BBQ/scan_camcoord.ply`
- RICH camera XMLs: `/workspace/data/RICH/scan_calibration/BBQ/calibration/*.xml`
- XFeat repo/scripts: `/workspace/code/accelerated_features`
- Main report root: `/workspace/code/Movie3R/output/anchor_token_report_v1`

## Current Report State

The Step1 report folder has been intentionally cleaned. It now contains only three sample folders, and each sample folder contains exactly five presentation images.

Step1 root:

`/workspace/code/Movie3R/output/anchor_token_report_v1/01_aabb_step1`

Current sample folders:

- `/workspace/code/Movie3R/output/anchor_token_report_v1/01_aabb_step1/BBQ_001_guitar_cam01_cam03_f00000005`
- `/workspace/code/Movie3R/output/anchor_token_report_v1/01_aabb_step1/BBQ_001_guitar_cam06_cam07_f00000244`
- `/workspace/code/Movie3R/output/anchor_token_report_v1/01_aabb_step1/BBQ_001_juggle_cam02_cam01_f00000197`

Each sample folder currently contains only:

- `00_semidense_xfeat_mesh_inliers.jpg`
- `01_ref_human3r_crop.jpg`
- `02_cur_human3r_crop.jpg`
- `03_human3r_patch_anchor_correspondences.jpg`
- `04_similarity_map_anchor_00.jpg`

Removed from these folders by user request:

- `aabb_comparison.jpg`
- `summary.json`
- `pair_*` folders
- old `clean_step1_*` outputs
- extra `04_similarity_map_anchor_01/02/03.jpg` files

## What The Five Step1 Images Mean

- `00_semidense_xfeat_mesh_inliers.jpg`: raw/semi-dense XFeat matches filtered by RICH static mesh geometry, showing true static-scene inlier correspondences.
- `01_ref_human3r_crop.jpg`: Human3R preprocessed reference crop with patch grid and anchor patch locations.
- `02_cur_human3r_crop.jpg`: Human3R preprocessed current crop with patch grid and anchor patch locations.
- `03_human3r_patch_anchor_correspondences.jpg`: mesh-verified anchors mapped to Human3R patch-token pairs.
- `04_similarity_map_anchor_00.jpg`: for one selected reference anchor patch, current-view encoder-token cosine similarity heatmap; magenta is true mesh anchor, small numbers are top token-similarity candidates.

Heatmap color convention:

- warm colors usually mean higher cosine similarity.
- blue/purple usually mean lower cosine similarity.
- the heatmap alone does not prove uniqueness; use rank, cosine, top-k markers, and mesh-verified true correspondence together.

## Latest Step1 Sample Facts

These are the important metrics from the latest clean generation. The current cleaned folders no longer include `summary.json`, so keep these numbers here.

| Sample | Boundary Pair | Mesh Inliers | Unique Anchor Patch Pairs | Ref Grid | Cur Grid | Notes |
|---|---|---:|---:|---|---|---|
| `BBQ_001_guitar_cam01_cam03_f00000005` | cam01 frame 6 -> cam03 frame 7 | 9 | 7 | 23 x 32 | 21 x 32 | weak/low-overlap example |
| `BBQ_001_guitar_cam06_cam07_f00000244` | cam06 frame 245 -> cam07 frame 246 | 77 | 41 | 23 x 32 | 23 x 32 | strong primary presentation example |
| `BBQ_001_juggle_cam02_cam01_f00000197` | cam02 frame 198 -> cam01 frame 199 | 490 | 179 | 23 x 32 | 23 x 32 | strong high-anchor example |

Previously observed encoder-token correspondence quality:

- `guitar cam06->cam07 f244`: true-match rank median around 4, true cosine clearly above random negatives.
- `juggle cam02->cam01 f197`: true-match rank median around 3, true cosine clearly above random negatives.
- `guitar cam01->cam03 f5`: weak sample, only 7 patch anchors, true-match rank median around 38, still above random in many cases.

## Current Clean Step1 Script

Script:

`/workspace/code/Movie3R/scripts/redraw_rich_step1_clean.py`

What it does:

- Loads the AABB boundary pair `A@t+1 -> B@t+2`.
- Runs XFeat `match_xfeat_star` semi-dense matching.
- Validates matches against RICH mesh visibility/reprojection.
- Loads the original Human3R checkpoint.
- Encodes Human3R crops.
- Maps mesh inliers into patch-token coordinates.
- Draws the five clean report images.

Important detail:

- Default `--num_similarity_examples` is `4`.
- To keep exactly five output images per sample, pass `--num_similarity_examples 1`.
- The script creates/overwrites individual files but does not clean old files. If exact five-file output is required, clean the target directory first.

Regeneration commands used most recently:

```bash
PYTHONPATH="src:." .venv/bin/python scripts/redraw_rich_step1_clean.py --source_sequence BBQ_001_guitar --cam_a 1 --cam_b 3 --start_frame 5 --num_similarity_examples 1 --out_dir "/workspace/code/Movie3R/output/anchor_token_report_v1/01_aabb_step1/BBQ_001_guitar_cam01_cam03_f00000005"
```

```bash
PYTHONPATH="src:." .venv/bin/python scripts/redraw_rich_step1_clean.py --source_sequence BBQ_001_guitar --cam_a 6 --cam_b 7 --start_frame 244 --num_similarity_examples 1 --out_dir "/workspace/code/Movie3R/output/anchor_token_report_v1/01_aabb_step1/BBQ_001_guitar_cam06_cam07_f00000244"
```

```bash
PYTHONPATH="src:." .venv/bin/python scripts/redraw_rich_step1_clean.py --source_sequence BBQ_001_juggle --cam_a 2 --cam_b 1 --start_frame 197 --num_similarity_examples 1 --out_dir "/workspace/code/Movie3R/output/anchor_token_report_v1/01_aabb_step1/BBQ_001_juggle_cam02_cam01_f00000197"
```

## Other Report Sections

Main report root:

`/workspace/code/Movie3R/output/anchor_token_report_v1`

Other useful report folders:

- `02_correction_proxy`: affine/translation/no-correction proxy and evidence visualizations.
- `03_anchor_token_prototype`: AnchorToken residual prototype visualizations.
- `04_specificity_controls`: negative controls and specificity checks.
- `05_topk_quality_gate`: top-K and quality-gate validation.
- `README.md`: report summary written earlier.

Presentation-friendly clean overlays already generated:

- `/workspace/code/Movie3R/output/anchor_token_report_v1/02_correction_proxy/BBQ_001_guitar_cam06_cam07_f00000244/correction_prediction_overlay_clean.jpg`
- `/workspace/code/Movie3R/output/anchor_token_report_v1/03_anchor_token_prototype/BBQ_001_guitar_cam06_cam07_f00000244/anchor_token_lookup_overlay_clean.jpg`

## Main Evidence So Far

Step1 conclusion:

- External XFeat+mesh anchors can be mapped into Human3R patch-token indices.
- Human3R encoder tokens at true mesh-verified anchor pairs have stronger similarity than random/shuffled negatives.
- This supports using external local anchors as model-internal token evidence.

Correction proxy conclusion:

- A simple `mean(delta_uv)` translation is not robust.
- Coarse affine re-anchoring is much more stable than no correction or naive translation.
- This motivates global affine evidence plus local AnchorToken residuals.

Representative correction proxy results:

| Sample | No Correction | Translation | Affine |
|---|---:|---:|---:|
| `guitar cam06->cam07 f244` | 3.16 | 4.32 | 1.04 |
| `juggle cam02->cam01 f197` | 3.16 | 1.04 | 0.76 |
| `guitar cam01->cam03 f5` | 10.03 | 2.47 | 0.48 |

AnchorToken prototype conclusion:

- Best first design is `global affine coarse re-anchor + local AnchorToken residual`.
- Correct tokens should outperform affine-only and degrade under shuffled/wrong-boundary controls.

Representative leave-one-out prototype results:

| Sample | Same Position | Affine | Token Soft | Token Affine Residual |
|---|---:|---:|---:|---:|
| `guitar cam06->cam07` | 3.16 | 1.15 | 1.41 | 0.82 |
| `juggle cam02->cam01` | 3.16 | 0.82 | 1.58 | 0.66 |
| `guitar cam01->cam03` | 10.03 | 1.05 | 1.46 | 1.14 |

Specificity/negative-control conclusion:

- Correct token usually beats shuffled and wrong-boundary controls.
- Spatial-only can be competitive in very easy/near-contiguous samples, so negative controls are important.

Top-K/quality-gate conclusion:

- We do not need all anchors.
- `8-16` high-quality, spatially diverse AnchorTokens are usually enough.
- Suggested fallback: `<8` unique anchor patch pairs should fall back or have very low weight.

## Offline Anchor Caches Already Generated

Small guitar cache:

`/workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_v1`

- `20/20` cached.
- `frame_stride=30`.
- `top_k_tokens=16`.
- size about `145K`.

High-overlap guitar cache:

`/workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1`

- `185/185` cached.
- `0` skipped.
- `frame_stride=10`.
- `camera_pairs=6-7,5-6,4-5,3-4,1-2`.
- `top_k_tokens=16`.
- mean `quality_gate=0.7926676730851869`.
- mean `unique_anchor_patch_pairs=120.48648648648648`.
- size about `923K`.

## Key Scripts

Step1 and anchor validation:

- `/workspace/code/Movie3R/scripts/verify_rich_anchor_encoder_similarity.py`
- `/workspace/code/Movie3R/scripts/verify_rich_aabb_anchor_step1.py`
- `/workspace/code/Movie3R/scripts/redraw_rich_step1_clean.py`

Correction/evidence/prototype:

- `/workspace/code/Movie3R/scripts/analyze_rich_aabb_anchor_correction.py`
- `/workspace/code/Movie3R/scripts/build_rich_anchor_evidence.py`
- `/workspace/code/Movie3R/scripts/prototype_rich_anchor_tokens.py`
- `/workspace/code/Movie3R/scripts/validate_anchor_token_specificity.py`
- `/workspace/code/Movie3R/scripts/validate_rich_anchor_token_selection.py`

Cache generation:

- `/workspace/code/Movie3R/scripts/batch_generate_rich_guitar_anchor_cache.py`

Clean overlay generation:

- `/workspace/code/Movie3R/scripts/make_anchor_report_clean_overlays.py`

XFeat/RICH geometry helpers in the accelerated_features repo:

- `/workspace/code/accelerated_features/scripts/test_rich_aabb_xfeat_mesh_geometry.py`
- `/workspace/code/accelerated_features/scripts/compute_rich_camera_overlap.py`
- `/workspace/code/accelerated_features/scripts/visualize_rich_mesh_projection.py`
- `/workspace/code/accelerated_features/scripts/visualize_rich_mesh_correspondences.py`

Human3R/Movie3R model files:

- `/workspace/code/Movie3R/src/dust3r/model_human3r.py`: original Human3R model loader used for visualization.
- `/workspace/code/Movie3R/src/dust3r/model.py`: Movie3R modified model path; do not use for original Human3R evidence visualization unless intentionally testing Movie3R.
- `/workspace/code/Movie3R/src/dust3r/shot_adaptation.py`: Shot-aware adaptation module.

Docs/work logs:

- `/workspace/code/Movie3R/docs/movie3r/shot_token_v6_plan.md`
- `/workspace/code/Movie3R/tasklist/work_log.md`
- `/workspace/code/Movie3R/tasklist/TODO.md`: current concise TODO list.
- `/workspace/code/Movie3R/tasklist/archive/work_compact.md`: archived compact historical summary.
- `/workspace/code/Movie3R/tasklist/archive/TODO_legacy_20260514.md`: archived old long TODO.
- `/workspace/code/Movie3R/docs/env_setup_h800_cuda124.md`: H800 / CUDA 12.4 environment setup notes.
- `/workspace/code/Movie3R/output/anchor_token_report_v1/README.md`

## Important Technical Notes

- RICH XML `CameraMatrix` is treated as world-to-camera for projecting scan/world coordinates into camera views.
- RICH mesh correspondence logic uses the same static mesh vertex visible in both cameras, z-buffer visibility filtering, and optional human masks to remove dynamic humans.
- Camera intrinsics/extrinsics alone only give epipolar lines, not unique point-to-point correspondence; the mesh is what gives true static 2D-2D correspondences.
- RICH generated depth is not used as cross-camera metric GT here; static mesh + calibration is the trusted geometry teacher.
- Homography RANSAC is not the final gate for cross-camera non-planar scenes; mesh validation is the main gate.
- `match_xfeat_star()` gives semi-dense matches and does not use the same `min_cossim` logic as sparse `match_xfeat()`.
- Human3R image preprocessing resizes long side to `512`, keeps aspect ratio, center-crops, aligns crop size to multiples of `16`, then uses patch size `16`.
- Patch grids can differ across images because of aspect ratio/crop alignment. Example weak sample has ref `23 x 32`, current `21 x 32`.

## Current Status

Completed:

- Built RICH mesh-verified anchor pipeline.
- Verified anchors map into Human3R encoder patch tokens.
- Built and cleaned Step1 presentation outputs for three samples.
- Built correction proxy and evidence vector analysis.
- Built AnchorToken residual prototype and negative controls.
- Built top-K/quality-gate validation.
- Generated guitar offline anchor caches.
- Committed clean Step1 report outputs, clean overlay scripts, migration notes, and tasklist/archive cleanup.

Not completed:

- No model-path integration yet for cached AnchorTokens.
- No dataset/loader integration yet for anchor cache files.
- No training run using AnchorTokens yet.
- No inference-time anchor generator yet without RICH mesh.
- Latest local tasklist cleanup commit is `f89a708`; push status may still need checking before deleting the old server copy.

## Recommended Next Steps

For report polishing:

- Use `BBQ_001_guitar_cam06_cam07_f00000244` as the primary strong Step1 example.
- Use `BBQ_001_juggle_cam02_cam01_f00000197` as the high-anchor strong example.
- Use `BBQ_001_guitar_cam01_cam03_f00000005` as the weak/low-overlap fallback example.
- If the user asks for a slide deck or final report, reference the five Step1 images plus the two clean overlay images from `02_correction_proxy` and `03_anchor_token_prototype`.

For implementation:

- Add dataset/loader logic to read offline `.npz` anchor cache files.
- Read `ref_patch_idx`, `cur_patch_idx`, `affine_forward`, `affine_inverse`, and `quality_gate` from cache.
- Implement minimal cached AnchorToken adapter that affects only pose/camera path.
- Keep encoder unchanged.
- Keep full decoder token sequence unchanged for the first implementation.
- Use quality gates: strong enable at `unique_anchor_patch_pairs >= 16`, weak/lower weight for `8-15`, fallback below `8`.

For inference research:

- Replace mesh teacher with online XFeat semi-dense matches.
- Add lightweight geometric/confidence filtering.
- Select top-K spatially diverse anchors.
- Fall back when confidence or anchor count is low.

## If You Need To Continue Immediately

Best first read after this file:

`/workspace/code/Movie3R/output/anchor_token_report_v1/README.md`

Best first image to inspect:

`/workspace/code/Movie3R/output/anchor_token_report_v1/01_aabb_step1/BBQ_001_guitar_cam06_cam07_f00000244/04_similarity_map_anchor_00.jpg`

Best first script to modify for current visualization requests:

`/workspace/code/Movie3R/scripts/redraw_rich_step1_clean.py`

Best first code path for future model integration:

`/workspace/code/Movie3R/src/dust3r/shot_adaptation.py`
