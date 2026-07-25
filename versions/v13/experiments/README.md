# V13 Experiments

- `fusion_optimization.py`: Phase 2 fusion-only analysis on the frozen strict
  GT-ID candidates from Phase 1.
- `phase3_cross_shot_identity.py`: Phase 3 external identity-bank feature
  extraction, cross-shot WHO probe and automatic-ID Uniform Consensus.
- `phase3_egohumans_identity.py`: apply the identity rule frozen on `three` to
  the recurrent EgoHumans multi-cut stream without retuning it.
- `phase4_precision_identity.py`: extract predicted-bbox DINOv2 appearance,
  scan the `three` precision gate, freeze it, and evaluate MultiHuman
  `dance/box` with the unchanged Uniform Multi-Human Boundary.
- `phase4_egohumans_identity.py`: reconstruct deployable predicted SMPL-X
  bboxes from compact Human3R outputs and audit the frozen Phase-4 rule on
  three recurrent EgoHumans streams, including `3 -> 1 -> 3`.
- `phase5_causal_identity_state.py`: run true multi-cut streams with a fixed-size
  external identity state, causal margin replay, Top-6 assignment generation,
  and whole-stream detection-order audits.
- `phase5_joint_hypothesis_probe.py`: give every Top-6 identity hypothesis one
  frozen Phase-2 Boundary solve and audit identity/geometry/joint risk coverage.

These experiments keep Human3R, hard reset, Fixed Explicit, V16 and `s=1`
fixed. They do not enable DA3, VGGT or V11.4 scale. Phase 3 changes only the
GT-ID association layer; tokens never predict SE(3) or fusion weights.
Phase 4 changes only WHO and keeps the same geometry contract. Its result is a
negative deployment decision: zero wrong accepted is achieved only at
insufficient coverage, so automatic multi-human alignment remains disabled.
Phase 5 validates persistent identity state on development data, but its tested
joint geometry scores reduce zero-wrong coverage relative to identity-only
margin. The joint scorer is probe-only and never updates persistent state.
