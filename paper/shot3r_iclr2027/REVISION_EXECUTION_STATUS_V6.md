# BRIDGE3R revision execution status

This checklist translates the author-approved decisions in
`REVISION_DECISIONS_V6.md` into executable work.  **Implemented** means the
versioned manuscript or runtime source has been changed and verified;
**Deferred** means the decision intentionally requires new data, a new run, or
author-provided figures. **Partially implemented** means its paper-facing
structure is complete but an explicitly named audit still requires data or a
new computation. It does not mean the issue was dropped.

| ID | Decision | Execution state | Evidence / next action |
|---|---|---|---|
| D01 | Fixed publication route | Implemented | Scope is fixed; no new method or three-dataset rerun. |
| D02 | Harmony4D presentation | Implemented | v016 reconciles the paper with retained provenance: the final unified configuration is fixed across datasets, but historical Harmony4D test captures informed an earlier parent diagnosis. The 88-case table is therefore described as a controlled fixed-configuration regression, not an untouched held-out selection result; it retains the mixed W/WA/ATE/IDF1 outcome. |
| D03 | Blend sweep | Implemented | v017 completes the pre-registered train-only Harmony4D sensitivity study. The pre-inference lock (SHA-256 `26688b70aa3d132d3e21994849135458e79c1c237cd18c2fdeb47b4f7a5dd749`) permits only `train/02_grappling`, `train/07_ballroom`, and `train/12_mma`, and forbids test archives. All 12 fixed cases completed base inference; 9 are evaluator-complete for every parent/$\lambda\in\{0.25,0.50,0.75,1.00\}$ row and 3 Ballroom cases are uniformly evaluator-unavailable before candidate-specific scoring. The summary (SHA-256 `e90d0ea94c29bebd86f069113a02fb1d548cb210cfb404144abba1938028d879`) reports all planned mean and median metrics descriptively. The publication $\lambda=0.5$ and all primary tables remain unchanged. |
| D04 | EgoHumans shared denominator | Implemented | v007 main table uses the 90-case internal subset; the 116-case public-method ledger remains supplementary. |
| D05 | EgoBody executable references | Implemented | v007 retains a separate availability-aware TRACE/PromptHMR-SPEC table. |
| D06 | Camera--Human scope | Implemented | v007 limits claims to camera--human consistency and does not claim dense-scene gains. |
| D07 | Harmony4D mixed result | Implemented | v007 main table uses (N=88), reports the adverse W value, and retains WA/ATE/IDF1 gains. |
| D08 | Standardized runtime | Deferred | v007 has a TODO-EXPERIMENT slot; it makes no standardized runtime claim. |
| D09 | Direct association study | Partially implemented | v008 writes the formal Harmony4D audit into the supplement. The frozen 88-case runtime manifest (`evidence/harmony4d_boundary_association/final_runtime_manifest_v1.jsonl`, SHA-256 `a4de6a52da5e019d06e6910f2b1229d1130b61e64a4422223b713f110a16e021`) verifies every final-audit boundary pair against its frozen RGB report. The completed audit (`formal_v1/final_v1.json`, SHA-256 `6bb7a6c2af2c85758f0666d68e5a1ae1135d0885eeb90e7150317c7db7cd9e75`) covers all 88 cases: 142/147 evaluator-valid endpoint pairs are correct (pair-micro 96.60%; case-macro 94.05%, bootstrap 95% CI [88.10%, 98.81%]); 84/88 cases have at least one evaluable pair; continuation coverage is 142/176 (80.68%); and runtime abstention is 0/256. The evaluator-only all-frame oracle upper bound is 69.41% IDF1 versus 63.61% in the final audit. The evidence remains explicitly Harmony4D-only until the same protocol is run on restored EgoBody/EgoHumans. |
| D10 | Detector scope | Implemented | v007 defines it only as a causal RGB event proposal for hard viewpoint cuts. |
| D11 | Paired EgoBody statistics | Implemented | v013 adds a retrospective, frozen-Test recording-paired analysis over all 43 retained EgoBody recordings. `evidence/egobody_paired_statistics/formal_test_v1.json` is bound to the retained `recording_metrics.csv` by SHA-256 and reports paired bootstrap intervals, median gain, win/tie/loss, and fixed-seed two-sided sign-flip tests for the five v011 main-table quantities. It changes neither the frozen candidate nor the Test protocol. |
| D12 | Multi-cut evidence | Implemented | v007 retains the completed four-capture auxiliary control only in the supplement. A replay subset containing exactly the four frozen captures is retained at `data/Bridge3R_harmony4d_retention/Harmony4D_multicut_replay_v1.tar.gz` (SHA-256 `a2ea8faa9e810673b8c63030e7b8a828e9d2a7182c45825f52831195f9948682`); archive listing validation passed. |
| D13 | Backbone scope | Implemented | v007 limits all claims to a HUMAN3R-derived recurrent reconstructor. |
| D14 | Natural edited video | Deferred | v007 lists a scoped no-GT continuity/failure study as TODO-EXPERIMENT. |
| D15 | Qualitative visualisation | Deferred | v007 has explicit author-facing TODO-FIGURE specifications; real assets remain required. |
| D16 | Title | Implemented | v007 uses the confirmed BRIDGE3R same-scene title. |
| D17 | Abstract evidence hierarchy | Implemented | v007 abstract uses strict internal evidence and omits external-baseline headline numbers. |
| D18 | Unavailable-work comparison | Implemented | v007 uses a supplementary HumanMM/Multi-THuMBS protocol table, not direct numerical ranking. |
| D19 | Main-paper evidence budget | Implemented | v007 keeps concise main tables and moves ledgers, multi-cut, and deferred work to the supplement. |
| D20 | Academic terminology | Implemented | v007 removes paper-facing transaction/locked/adapter and obsolete-branch terminology. |
| D21 | Supplementary method detail | Implemented | v007 specifies detector, correction tokens, association, coordinate update, fallback, and repeated cuts. |
| D22 | Replay package | Deferred | v007 records restoration of data and per-case predictions as a prerequisite. Harmony4D now has a verified four-capture multi-cut replay subset, but the three-dataset anonymous replay package remains pending restored EgoBody/EgoHumans inputs and per-case artifacts. |
| D23 | Camera decomposition | Deferred | v007 lists the required Sim(3)/SE(3) study as TODO-EXPERIMENT. |
| D24 | AI use statement | Implemented | v007 contains the approved factual scope of language-model assistance. |
| D25 | Explicit TODO policy | Implemented | v007 visibly renders TODO-EXPERIMENT and TODO-FIGURE placeholders. |
| D26 | Anonymous release | Deferred | Build only after final paper, data replay, and TODO clearance. |
| D27 | Harmony4D denominator | Implemented | v007 main table uses common (N=88); denominator detail is supplementary. |
| D28 | Oracle controls | Implemented | v007 confines oracle-boundary controls to the component table. |
| D29 | Causal state/evidence separation | Implemented | v007 introduction, method, and conclusion make it the core contribution. |
| D30 | Learned correction-token coarse gauge | Partially implemented | v007 method/provenance bind it to V14.1 P0 and causal routing; three-dataset result-binding audit remains pending. |
| D31 | Full two-stage ablation | Deferred | v007 adds the exact TODO-EXPERIMENT chain; execute after replay data are restored. |
| D32 | Training/data audit | Partially implemented | v014 completes the checkpoint-bound parameter audit: the saved V14.1 P0 checkpoint has 34,415,333 trainable parameters out of 1,232,459,373 (2.79%) under its frozen configuration; `evidence/checkpoint_parameter_audit/v14_1_p0_e6.json` binds this result to SHA-256 `de2430ed...282265`. A complete event-/clip-level training/benchmark-overlap audit still requires the retained source manifests and archive identities. |
| D33 | Detector-driven EgoHumans runner | Implemented | Runtime now consumes GRU first-positive and fails closed on cache mismatch; 116/116 equivalence audit passes. |

## Additional retained evidence completed after the decision ledger

| ID | Study | Execution state | Evidence / scope |
|---|---|---|---|
| E01 | EgoHumans capture-paired statistics | Implemented | v015 adds a retrospective strict-versus-Bridge3R paired analysis on the frozen common 90-case Test, clustered over 27 captures. `evidence/egohumans_paired_statistics/formal_test_v1.json` is bound to the Test summary and all 29 candidate reports. It supports W, ATE-Sim3, and IDF1 improvements; WA and coverage remain descriptive, not significance claims. |

## Update convention

On completing an item, replace **In progress** with **Implemented** and add
the manuscript version plus validation evidence in the final column.  Deferred
items remain explicit until their required experiment or author asset exists.
