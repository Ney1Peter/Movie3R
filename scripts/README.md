# Scripts

This directory keeps only current generic entry points and small utilities.

## Current Utilities

| Script | Purpose |
|---|---|
| `run_human3r_save_output.py` | Run original Human3R inference and save camera/color/conf/depth/SMPL outputs. |
| `view_human3r_saved_output.py` | View saved Human3R outputs. |
| `detect_video_shot_changes.py` | Generic video shot-change detection helper. |
| `export_avatarrex_aabb_video.py` | Export an AvatarReX/RICH AABB sample as a quick mp4. |
| `fetch_bedlam.sh`, `fetch_model.sh`, `fetch_smplx.sh` | Download/setup helpers. |

## Historical Archives

| Directory | Contents |
|---|---|
| `archive_v2_v6/` | ShotToken, AnchorToken, RICH anchor, V4-V6 training command, and related diagnostic scripts. |
| `archive_v7/` | Offline floor/human/scene correction, post-shot teacher, pseudo-label, implicit-token student, and related diagnostic scripts. |

New V8 work should not extend archived scripts by default.
