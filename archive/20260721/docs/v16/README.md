# V16 Explicit-First Human-Aware Rotation Residual

V16 tests a fixed cascade at camera cuts:

```text
Explicit coarse SE(3)
-> bounded torso/motion rotation residual
-> fixed-rotation scene translation re-solving
-> one shot-level SE(3)
```

The geometry component passes. On 180 real cuts, predicted one-frame torso motion reduces Fixed Explicit rotation from `24.20` to `16.04` degrees and R-P90 from `62.30` to `39.33` degrees. It improves all four sources and remains effective after V15 coarse pose.

The learned token component does not pass. Token-only residual regression degrades rotation to `41.18` degrees, while token confidence or gate models do not add stable held-out-source gains over explicit torso geometry. The retained route is therefore geometry-only:

```text
Explicit coarse
-> torso motion heading residual
-> scene translation re-solving
```

Main report:

- [V16 Explicit-First Human-Aware Rotation Residual Probe](V16_EXPLICIT_FIRST_HUMAN_AWARE_ROTATION_RESIDUAL_20260719.md)

Code:

```text
scripts/v16_rotation_residual_partial_oracle.py
scripts/v16_human_torso_candidates.py
scripts/v16_human_torso_eval.py
scripts/v16_loso_human_token_probe.py
```
