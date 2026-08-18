# Movie3R-v15 × Harmony4D

This directory implements the frozen `Movie3R-Harmony4D-CrossShot-v1`
protocol.  It does not change v15 checkpoints, thresholds, or geometry.  GT
is read only by the dataset-index and evaluator processes; the GPU runtime is
RGB-only and records `gt_in_runtime=false`.

Large data are staged only below
`/data/wangzheng/iJCV-CODE/data/Harmony4D_work`.  The outer archive is never
deleted.  Compact outputs live below `output/v15_harmony4d`.

The public Multi-THuMBS Harmony4D values are literature references because
its exact sequence/camera/cut/evaluator manifest is not public.  Direct
Movie3R ablations use the fully frozen protocol in this directory.

Main entry points:

- `stage_archive.py`: resumable nested-ZIP SHA/CRC verification, coordinate
  audit, deterministic capture selection, and test-manifest freeze;
- `run_harmony_batch.py`: resumable CUDA inference plus GT-isolated evaluator;
- `reevaluate_harmony.py`: deterministic evaluator-only refresh over immutable
  prediction caches;
- `aggregate_harmony.py`: clip/sequence/micro aggregation, bootstrap CI,
  paired tests, CSV, and LaTeX output;
- `audit_test_results.py`: frozen-manifest, runtime-contract, cache-hash, and
  explicit evaluator-unavailable audit;
- `build_paper_artifacts.py`: stratified CSVs, LaTeX tables, runtime summaries,
  paper-result JSON, and PDF/PNG figures;
- `export_harmony_qualitative.py`: restores dense demo backgrounds from frozen
  checkpoints while keeping formal-test camera/human geometry immutable;
- `probe_temporal_identity.py`: train/dev-only identity-strategy audit.  Its
  selected single-boundary permutation is frozen in M13--M16; test data never
  enter strategy selection.

Long jobs must run in the `movie3r_h4d_iclr` tmux session.  `TMPDIR` must stay
under `/data/wangzheng/iJCV-CODE/data/Harmony4D_work/tmp`; the outer
`Harmony4D.zip` is never deleted.
