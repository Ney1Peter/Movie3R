# V13 Experiments

- `fusion_optimization.py`: Phase 2 fusion-only analysis on the frozen strict
  GT-ID candidates from Phase 1.
- `phase3_cross_shot_identity.py`: Phase 3 external identity-bank feature
  extraction, cross-shot WHO probe and automatic-ID Uniform Consensus.
- `phase3_egohumans_identity.py`: apply the identity rule frozen on `three` to
  the recurrent EgoHumans multi-cut stream without retuning it.

These experiments keep Human3R, hard reset, Fixed Explicit, V16 and `s=1`
fixed. They do not enable DA3, VGGT or V11.4 scale. Phase 3 changes only the
GT-ID association layer; tokens never predict SE(3) or fusion weights.
