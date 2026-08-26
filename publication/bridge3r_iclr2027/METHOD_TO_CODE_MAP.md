# Bridge3R publication method-to-code map

`PAPER_METHOD_LOCK.json` is the scientific contract.  This map records the
current implementation pieces and their publication role; it intentionally
does not pretend that a historical dataset runner is already a single,
backbone-independent release entry point.

| Locked component | Current code/evidence | Publication role | Status |
|---|---|---|---|
| Human3R-derived recurrent backbone | `src/` and the hash in `PAPER_METHOD_LOCK.json` | scene-aware camera/person estimator | frozen checkpoint contract |
| clean reset and post-shot ownership | `versions/v20/egobody/deployment_runtime.py` | EgoBody formal runtime branch | implemented; historical adapter replay requires restored RGB/cache |
| read-only B0 shadow gauge | `versions/v20/egobody/deployment_runtime.py`, cached `m3_b0_only` outputs | coarse pre-to-post gauge proposal | implemented in retained protocol paths |
| permutation-aware association | `versions/v20/egobody/deployment_runtime.py`, `versions/v19/egohumans/causal_identity.py` | persistent anonymous identity across the boundary | implemented in dataset runners and accepted by the standard-array API |
| fixed shared translation | `versions/v20/egobody/deployment_runtime.py`, `versions/v19/egohumans/joint_correction.py`, `versions/v17/harmony4d/unified_half_translation_audit_candidate.json` | final correction: `camera_alpha=1`, translation, blend `0.5` | locked; Harmony result is cache materialization |
| no gate / no root filter | EgoBody deployment validation, EgoHumans final candidate, Harmony unified audit candidate | excludes historical adaptive policy from final method | locked |
| single publication transaction core | `publication/bridge3r_iclr2027/runtime_contract.py` and `bridge3r.py` | evaluator-free standard-array entry point, lock validation, no-cut path, and multi-cut composition | implemented and contract-tested; EgoHumans adapter equivalence is numerically verified on 116 retained Test caches |
| causal cut proposal | EgoBody formal runtime detector ledger and `publication/bridge3r_iclr2027/evidence/egohumans_detector_equivalence.json` | cut trigger before the transaction | detector-driven equivalence is verified on all 116 frozen EgoHumans Test runtime reports |
| archive staging | `versions/v15/harmony4d/stage_archive.py` | reads Harmony4D ZIP archives with optional top-level folder | patched and smoke-tested |
| Harmony4D external publication artifact | `../../external_baselines/bridge3r_eval/build_harmony4d_final_artifacts.py` | validates availability and creates five-method supplementary tables | implemented and unit-tested |

## Required integration work

`bridge3r.py` is the single formal entry for the locked post-cut transaction;
it accepts only standard clean-reset/B0 arrays and emits a uniform runtime
ledger. Dataset-specific code still stages RGB, invokes the Human3R-derived
backbone, and supplies the causal cut/prediction-only association. The
EgoHumans historical adapter is numerically identical to this entry on all
116 retained Test caches (exact finite values and the same NaN padding mask;
see `evidence/egohumans_publication_entry_equivalence.json`). The original
EgoBody RGB/cache and Harmony4D staging cache were intentionally removed for
disk management, so those two adapters must be replayed only after the source
data or caches are restored; this limitation is recorded rather than hidden.
Historical BRTC, C1, adaptive-joint, gate, and root-filter branches remain
available only as ablations or archived analysis.
