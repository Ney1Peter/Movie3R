# BRIDGE3R method--code map

This internal map binds the v030 paper description to the retained final
implementation. It is not part of the anonymous Overleaf package.

| Paper operation | Canonical implementation | Verified scope |
|---|---|---|
| Event-conditioned correction tokens | `Movie3R/src/dust3r/v8_pose_prompt.py`, `V82PoseRelationPrompt` | Semantic, alignment, and temporal-continuity tokens are constructed from current features and causal memory. |
| Decoder insertion and camera residual | `Movie3R/src/dust3r/model.py`, `_route_v9_corr_tokens_pre_decoder` and recurrent forward sites | Tokens are inserted after the native pose token; a learned residual changes the decoded read-only camera. |
| Auxiliary human residual and head LoRA | `Movie3R/src/dust3r/model.py`; final training configuration | Enabled in training/inference, but decoded read-only people are discarded and do not enter reported packed outputs. |
| Causal event proposal and two boundary evaluations | `Movie3R/versions/v15/harmony4d/run_harmony_case.py`, `run_transaction` | A proposal invokes a read-only continuation and a clean reset. Only the latter state is propagated. |
| Coarse gauge | `run_transaction` camera composition | `T0 = C_shadow @ inv(C_reset)`; no separately regressed transform is claimed. |
| Prediction-only association | `run_harmony_case.py`, `anonymous_match`; `versions/v14/probe_b0_identity_matching.py` | Hungarian assignment uses predicted pelvis, torso, and root-centred joint descriptors. |
| Persistent anonymous IDs | `run_harmony_case.py`, `boundary_permutation_post`; `versions/v16/harmony4d/causal_stabilization.py` | IDs are transported at the first post-cut frame; no GT identity is read at runtime. |
| Shared camera--human translation | `versions/v16/harmony4d/causal_stabilization.py`, `coupled_boundary_register` | `0.5 * coordinate-wise median` of predicted pelvis residuals is applied to cameras, joints, and vertices. |
| Fixed method contract | `Movie3R/publication/bridge3r_iclr2027/PAPER_METHOD_LOCK.json` | Camera coefficient 1, translation coefficient 0.5, no reliability gate, no root filter. |
| Causal composition tests | `Movie3R/publication/bridge3r_iclr2027/runtime_contract.py` | Prefix immutability and increasing-order multi-cut composition are checked. |

## Explicit exclusions

- Historical BRTC/C1, reliability gates, root filters, ray refinement, and
  dataset-specific coefficient selection are not the reported method.
- The read-only state, people, and point map are discarded after forming the
  camera gauge.
- Dense point maps do not receive the final shared translation; the paper
  therefore makes no dense-scene accuracy claim.
- Same-checkpoint inference masks are sensitivity tests, not independently
  trained token ablations.

