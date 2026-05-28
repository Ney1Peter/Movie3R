# V7 Script Archive

This directory contains archived V7 research scripts for MS-AIST/H36M pseudo-label generation, token dumping, implicit adapter training, prediction, long-run monitoring, and offline floor/human/scene correction diagnostics.

V7 is no longer the active direction. These scripts are kept for reproducibility and review, but new work should not extend them by default.

Important notes:

- Some archived scripts still contain historical paths under `/data/wangzheng/iJCV-CODE/data/data-V7-*`.
- Most V7 helper scripts now live in this directory. Some historical paths may still appear in comments or old outputs; if a historical run must be reproduced, prefer the archived path under `scripts/archive_v7/`.
- Core source hooks such as `src/dust3r/v7_pose_adapter.py` remain in place for import/checkpoint compatibility and are not treated as the active V8 direction.
