# V9 Multi-Layer Information Probe

This directory contains an isolated feasibility experiment for the V9 correction
architecture. It does not modify Human3R, V9 checkpoints, existing configs, or
the recurrent model implementation.

The experiment asks whether V9 loses useful camera-cut evidence because it uses
only final encoder features and strongly pooled correction tokens.

## Sources

- CUT3R encoder blocks: quarter, half, three-quarter, final
- DINOv2 blocks: quarter, half, three-quarter, final
- Human3R decoder blocks: quarter, half, three-quarter, final

All backbone parameters remain frozen. Forward hooks collect selected outputs
and immediately convert each pre/post pair into a fixed-size relation descriptor.
Raw feature maps are not persisted.

## Protocol

1. Single-case overfit on `v14_1_single_lbn1_1192`.
2. Ten-cut capacity test using the frozen V14.1 four-source manifests.
3. Held-out test: fit on separate records sampled from the existing V9
   four-source manifests, excluding the ten evaluation cuts.
4. Compare direct absolute and raw-Human3R-residual heads across individual
   layers, multi-layer groups, and cross-backbone combinations.
5. Select ridge regularization by camera-pair-grouped training-only CV, then
   report repeated data-scale curves and paired bootstrap uncertainty.
6. Evaluate the frozen formal V9 checkpoint on the identical ten cuts.

The direct head is a diagnostic. It predicts a single relative camera transform;
it never predicts independent camera and human transforms.

## Reproduction

Extract the scaled frozen cache on one physically free GPU:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src:. .venv/bin/python \
  versions/v9/multilayer_information_probe/run_probe.py \
  --device cuda:0 \
  --train-per-source 96 \
  --output-dir output/v9_multilayer_information_probe/scale96 \
  --extract-only \
  --overwrite-cache
```

Run the robust CPU analysis:

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=src:. .venv/bin/python \
  versions/v9/multilayer_information_probe/robust_analysis.py
```

Evaluate formal V9 on the same frozen cuts:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src:. .venv/bin/python \
  versions/v9/multilayer_information_probe/evaluate_v9_checkpoint.py \
  --device cuda:0
```

## Outputs

Default outputs are written under:

```text
output/v9_multilayer_information_probe/
```

The cache stores compact descriptors, target relative camera transforms, and raw
Human3R camera predictions. It does not contain model weights or full feature
maps.

The final interpretation is in `FINAL_REPORT.md`. A multi-layer evidence-token
variant is intentionally not trained unless multi-layer frozen features clearly
beat the best single layer on held-out residual correction.
