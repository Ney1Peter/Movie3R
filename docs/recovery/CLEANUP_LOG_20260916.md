# Workspace cleanup log — 2026-09-16

This cleanup was performed only after the Shot3R recovery snapshot had passed
its integrity checks and the GitHub `master` reference was verified. Raw
datasets, model checkpoints, formal prediction outputs, current paper assets,
and runnable baseline environments were deliberately retained.

## Recovery gate

- GitHub repository: `https://github.com/Ney1Peter/Movie3R.git`
- Recovery snapshot commit: `6d0ab0e69ced39e73fd7a3b407682c4628a0ee75`
- Recovery tag: `shot3r-recovery-20260916`
- Lightweight manifest: 242/242 SHA-256 entries passed.
- Test suite: 105/105 tests passed with `PYTHONPATH=src:.`.
- The temporary repository deploy key used to reach GitHub through the SSH
  443 gateway was deleted immediately after the push; the API reported zero
  matching keys afterward.

## Permanently removed

- `output/shot3r_registration_baselines_v1_20260915/work/`
  (38,807,526,307 bytes): evaluator staging duplicated by the sealed outputs.
- `output/Shot3R_Traditional_Registration_Baselines_v1_20260915_FULL.zip`
  (39,446,385,630 bytes): duplicate archive of the retained result tree.
- runtime benchmark `runs/`, contended/failed archives, and copied
  dependencies (1,167,714,513 bytes): regenerable from the tracked protocol.
- eight ignored multi-cut `.npz` replay caches (826,420,029 bytes).
- three superseded teaser-generation workspaces and the licensed SMPL-X ZIP
  from the current Git tree. The teaser payloads were removed locally; the
  SMPL-X ZIP remains local but is no longer tracked.
- paper snapshots v055–v066 and v068–v071, old v063 release files, and older
  draft copies. Local paper retention is now v067 plus v072 only.
- obsolete qualitative handoff/cache folders, old figure prompts, temporary
  image previews, installer archives, and Python/pytest caches.

The measured removal was approximately 82 GB. Files deleted from previously
tracked Git paths remain recoverable from repository history; untracked
staging, caches, and duplicate archives are not recoverable except by rerun.

## Retained locally

- all three raw benchmark archives under `/data/wangzheng/iJCV-CODE/data/`;
- 37 GB of formal traditional-registration predictions;
- all other formal Shot3R outputs (the complete `output/` remains about 228 GB);
- final Shot3R, V9 initializer, and original Human3R checkpoints listed in
  `WEIGHTS.md`;
- external baseline repositories, official weights, and runnable virtual
  environments;
- paper versions v067 and v072 (paper files are not mirrored to GitHub);
- five current editable PPT sources listed in `LOCAL_ASSETS.md`;
- the active EgoHumans qualitative viewer and its catalog.

## Outstanding external action

Upload the large checkpoints to Hugging Face, then add their stable download
URLs to `WEIGHTS.md`. Until that upload is complete, the checksums in the
weight manifest are the authoritative artifact identifiers.
