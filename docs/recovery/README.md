# Shot3R disaster-recovery entry

Last verified: 2026-09-16 (Asia/Shanghai).

This directory is the authoritative starting point when the original server is
unavailable. The Git repository backs up source code, frozen configurations,
small model components, and numerical evidence. Raw datasets, full prediction
caches, licensed body models, virtual environments, and multi-gigabyte model
checkpoints are intentionally external.

The paper workspace is not mirrored here by design. Its current local copy is
documented only in `LOCAL_ASSETS.md`.

## Recovery order

1. Clone `https://github.com/Ney1Peter/Movie3R.git` and check out the recovery
   tag `shot3r-recovery-20260916` (or a later commit that contains it).
2. Recreate the environment using `ENVIRONMENT.md`.
3. Download the weights listed in `WEIGHTS.md`, place them at the recorded
   relative paths, and run `python scripts/verify_shot3r_recovery.py`.
4. Obtain only the datasets needed for the intended experiment and rebuild
   their manifests as described in `DATASETS_AND_PROTOCOLS.md`.
5. Use `EXPERIMENTS.md` to locate the frozen protocol, entry point, and
   numerical source of truth.
6. Apply local baseline patches from `baseline_patches/` only when reproducing
   an external method; see `BASELINES.md`.

## What is protected by GitHub

- the Shot3R/Human3R-derived implementation and frozen configuration;
- the streaming boundary transaction, association, and evaluation code;
- the traditional-registration control implementation;
- the audited 25 KB causal transition detector;
- final aggregate and per-case metrics used for the paper;
- training curves/logs and runtime measurements;
- exact external-baseline commits and local patch series;
- checksums and path contracts for all external weights.

## What must be restored separately

- final Shot3R checkpoint (4.93 GB file; SHA-256 locked in `WEIGHTS.md`);
- Human3R and optional V9 initializer checkpoints;
- SMPL/SMPL-X licensed assets;
- EgoBody, EgoHumans, Harmony4D, AIST++, MVHuman, and training data;
- full per-frame prediction caches and qualitative rendering workspaces;
- the paper/Overleaf workspace and editable PPT files.

## Integrity rule

Never identify a weight only by filename. Verify its byte size and SHA-256.
The file called `checkpoint-final.pth` exists in several runs and represents
different models.

