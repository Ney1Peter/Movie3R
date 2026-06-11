# Movie3R V9

V9 starts from the verified V8.9 implicit human-pose correction branch.

Current base commit:

```text
a79eb18 feat: add v8.9 implicit human-pose correction
```

Current preserved checkpoint:

```text
output/v9_saved_weights/v9_implicit_avatarrex_single_checkpoint-best.pth
output/v9_saved_weights/v9_implicit_avatarrex_single_checkpoint-final.pth
```

Starting point:

```text
A_corr_t enters the decoder as a UniCon-style streaming relation token.
The refined correction token predicts both pose-token and human-token residuals.
Camera correction is applied before the original pose head.
Human correction is applied before the original Human3R human head.
GT camera and SMPL are only used for loss, metrics, and visualization overlays.
```

The detailed V9 design will be filled in after the next design pass.

