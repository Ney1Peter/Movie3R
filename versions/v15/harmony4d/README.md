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
