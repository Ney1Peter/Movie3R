# V7 Script Archive

This directory contains archived V7 research scripts for MS-AIST/H36M pseudo-label generation, token dumping, implicit adapter training, prediction, and long-run monitoring.

V7 is no longer the active direction. These scripts are kept for reproducibility and review, but new work should not extend them by default.

Important notes:

- Some archived scripts still contain historical paths under `/data/wangzheng/iJCV-CODE/data/data-V7-*`.
- Some scripts call other V7 scripts using their original `scripts/` paths. If a historical run must be reproduced, use the commit history or adjust those paths explicitly.
- Core source hooks such as `src/dust3r/v7_pose_adapter.py` remain in place for import/checkpoint compatibility and are not treated as the active V8 direction.
