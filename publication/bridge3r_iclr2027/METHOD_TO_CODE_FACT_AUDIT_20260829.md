# BRIDGE3R method-to-code fact audit

This internal audit is not part of the anonymous paper package. Its purpose is
to prevent manuscript formulas from describing a historical branch rather than
the formal BRIDGE3R route.

| Paper component | Canonical implementation | Verified behavior |
|---|---|---|
| Event-conditioned learned coarse branch | `src/dust3r/v8_pose_prompt.py`, `V82PoseRelationPrompt` | Builds semantic, pose-alignment, and temporal-continuity tokens from current tokens and causal memory. |
| Decoder insertion and pose residual | `src/dust3r/model.py`, `_route_v9_corr_tokens_pre_decoder` and the recurrent-forward sites | Inserts correction tokens immediately after the native pose token, refines them in decoder attention, and adds the gated residual to the decoded pose token. |
| Optional human-latent residual and head LoRA | `src/dust3r/model.py`; inference switches in `versions/v15/harmony4d/run_harmony_case.py` | These are enabled in the reported full learned forward. Formal token/head ablations are fixed-checkpoint inference masks, not separately retrained architectures. |
| Causal detector and shadow/clean first post-cut evaluations | `versions/v15/harmony4d/run_harmony_case.py`, `run_transaction` | A detector proposal triggers a read-only continuation/shadow evaluation and a clean-reset post-cut evaluation. The shadow-to-clean camera relation forms the B0 coarse gauge. |
| Prediction-only boundary association | `versions/v15/harmony4d/run_harmony_case.py`, `anonymous_match`; `versions/v14/probe_b0_identity_matching.py`, `identity_cost_components` | Hungarian assignment at the boundary uses normalized predicted root distance, torso orientation, and root-centred joint descriptors. |
| Persistent-ID transfer | `versions/v15/harmony4d/run_harmony_case.py`, `boundary_permutation_post`; `versions/v16/harmony4d/causal_stabilization.py`, `boundary_permutation_ids` | Matched IDs transfer only at the first post-cut frame; later frames retain native post-cut slots and unseen slots start new tracks. |
| Shared camera--human translation | `versions/v16/harmony4d/causal_stabilization.py`, `coupled_boundary_register` | With the formal candidate, the translation is 0.5 times the median predicted pelvis offset over boundary matches and is left-applied to camera-to-world poses, joints, and vertices. |
| Formal fixed configuration | `versions/v19/egohumans/frozen_final_candidate.json`; `publication/bridge3r_iclr2027/PAPER_METHOD_LOCK.json` | `camera_alpha=1`, translation boundary correction, blend 0.5, no velocity target, root filter, or reliability gate. |
| Multi-cut composition and no-cut invariant | `publication/bridge3r_iclr2027/runtime_contract.py`, `apply_locked_transaction` and `apply_locked_multicut` | Applies boundary operations in increasing order and checks that each later operation leaves its preceding prefix unchanged. |

## Audit boundaries

- `versions/v19/egohumans/causal_identity.py` is a retained historical
  framewise re-tracker. The final formal candidate leaves its `identity` field
  null and does not invoke it; it must not be cited as the formal identity
  mechanism.
- The standard-array publication core consumes clean-reset/B0 arrays and
  prediction-only pairs supplied by dataset adapters. It does not itself run
  the RGB reconstructor or solve the boundary assignment.
- The bridge is evaluated as a Camera--Human operation. Point-map outputs of
  the base reconstructor are not an input to its final shared-translation
  estimate and are not assigned a dense-scene metric claim.
