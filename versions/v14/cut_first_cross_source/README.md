# V14 Cut-First Cross-Source Correction

## Question

Does the existing V14 event-only shadow correction generalize when it is
trained with cross-camera supervision from AvatarReX, THuman, MVHuman100, and
MVHuman200 instead of one AvatarReX event?

This directory is isolated from the V9 decoder probes and all existing V14
weights.  It does not introduce a new architecture.

## Frozen Contract

```text
old pre-cut state (read-only)
-> correct only the first post-cut camera and human outputs
-> compute B0 from corrected camera vs independent raw-reset camera
-> discard shadow outputs and shadow state
-> apply one B0 to the complete raw-reset post-cut segment
```

The model retains the complete V9 semantic/alignment/momentum correct-token
path, decoder refinement, camera/human residual heads, event-only head LoRA,
and the geometry-preservation losses from
`train_v14_1_cut_event_single_v9_event_only_geometry.yaml`.

Only the labelled first post-cut frame receives correction and loss.  The
human correction is auxiliary supervision for the shadow decoder; shadow
humans are never committed at inference.

## Data Stages

`build_manifests.py` converts existing AABB records to:

```text
frames:      [t-1, t, t]
sequences:   [camera A, camera A, camera B]
shot_labels: [0, 0, 1]
```

It excludes every unordered camera pair in the frozen ten-event suite.  The
generated scales are nested:

- `train10`: 3/2/3/2 examples by source;
- `train24ps`: 24 examples per source;
- `train96ps`: 96 examples per source, gated by the 24/source result.

## Promotion Gates

1. Ten-cut training must finish without routing errors or NaNs and improve
   held-out first-post-cut camera metrics over the one-Avatar checkpoint.
2. The 24/source model must improve mean and P90 composite and must not add a
   catastrophic failure on the frozen ten-event and 180-cut diagnostics.
3. The 96/source run starts only if gate 2 passes.
4. No-event/reset-only outputs must remain numerically equal to the raw
   Human3R path, and later post-cut frames must use only raw-reset state plus
   the cached shared `B0`.

Failure at a gate is a result: do not compensate by changing the decoder,
fusion, loss weights, or evaluation set in the same experiment.

## Commands

```bash
.venv/bin/python versions/v14/cut_first_cross_source/build_manifests.py

cd src
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=..:. ../.venv/bin/python train.py \
  --config-name train_v14_1_cut_first_cross_source_10

cd ..
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:. .venv/bin/python \
  versions/v14/cut_first_cross_source/evaluate_cut_events.py \
  --model-path <checkpoint> --device cuda:0 --output-dir <eval-output>

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:. .venv/bin/python \
  versions/v14/cut_first_cross_source/evaluate_four_source_b0.py \
  --model-path <checkpoint> --device cuda:0 --output-dir <eval-output>

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:. .venv/bin/python \
  versions/v14/cut_first_cross_source/audit_reset_only_parity.py \
  --event-model <checkpoint> --device cuda:0 --output-dir <audit-output>
```

All outputs go under `output/v14_cut_first_cross_source/`.  Existing V9/V14
checkpoints are read-only initialization or evaluation baselines.
