# Bridge3R ICLR 2027: Final Paper Content Plan (v1)

## 0. Scope and status

This is a planning document only.  It is grounded in the sealed `v019_20260827_aist_multicut_formal` evidence and does **not** alter any result, denominator, or claimed comparison.

The target is one anonymous ICLR submission PDF: at most **nine pages before references** for the initial submission, followed by references and an unrestricted supplementary section in the *same* PDF.  Figure and table numbering must remain continuous across the main paper and supplement.

The paper's claim must remain narrow:

> Bridge3R causally bridges a same-scene physical-camera viewpoint cut for recurrent multi-person reconstruction by transferring a limited camera--human gauge while giving all future recurrent-state ownership to a clean post-cut branch.

It is not a dense scene-reconstruction paper, an arbitrary video-editing system, or a latency-equivalent replacement for offline full-video methods.

## 1. Evidence inventory and publication role

| Evidence | What is fixed and supported | Publication role | Claim boundary |
|---|---|---|---|
| EgoBody-CS150 | 129 fixed cases / 43 recording macro; Bridge3R improves W, WA, ATE-Sim3 and IDF1 over strict Human3R at equal coverage.  Executable TRACE and PromptHMR records exist with their own availability. | **Main result** and core ablation dataset. | External geometric means have method-specific accepted-match availability; they must not be called a paired leaderboard. |
| EgoHumans-CS100 | Common internal subset of 90 cases from 27 captures; Bridge3R improves W, WA point estimate, ATE-Sim3 and IDF1 over strict Human3R, with slightly lower coverage. | **Main transfer result**; complete public-method table in supplement. | Capture-cluster evidence supports W, ATE-Sim3 and IDF1, not a resolved WA/coverage significance claim. |
| Harmony4D-CS150 | Common 88-case internal subset; Bridge3R improves WA, ATE-Sim3 and IDF1, but has worse W and slightly lower coverage. | **Main robustness result** and component/audit source. | The adverse W result must remain visible.  This is a fixed-configuration regression evaluation, not a new test-time selection result. |
| EgoBody viewpoint strata | Pre-specified small/medium/extreme pairs; the gain persists in all retained strata. | One sentence in main; full table in supplement. | Do not treat strata as independent new datasets. |
| EgoBody public implementations | TRACE and PromptHMR-SPEC receive the same RGB clips.  Their coverage and accepted-match support are far lower than Bridge3R's, and their native camera interfaces differ. | Compact availability-aware public-reference block in main; complete ledger in supplement. | Never bold conditional geometric cells as a fully paired cross-method SOTA ranking. |
| AIST++ CS150 | Frozen 100-source official `pose_test` manifest.  Bridge3R improves causal strict Human3R in PA-MPJPE, seam orientation and relative camera rotation, but is worse in Anchor-MPJPE and seam root.  Offline PromptHMR is stronger on local/anchor geometry but worse in relative camera rotation. | **Supplementary single-person validation.** Mention only its scope in main. | Causal and offline rows are not latency-equivalent; do not claim overall AIST superiority. |
| AIST++ MC150-3 / MC150-4 | Two separate sealed 100-source repeated-cut internal studies.  They measure event composition and expose retained trade-offs. | Supplementary repeated-event/component evidence. | Do not pool with CS150 or compare to offline PromptHMR. |
| Harmony4D direct association + four-capture multi-cut control | Evaluator-only association audit and a small, pre-registered, real-image repeated-cut control with matched no-cut negatives. | Brief main-text mechanism support; full tables in supplement. | Not a universal association or arbitrary-edit benchmark. |
| AvatarReX `lbn1`, THuman02 | Both have excellent calibration/SMPL structure and broad angle coverage, but the fixed V14.1 training manifest contains `lbn1` and `thuman02`. | **Exclude from formal paper tables.** | They cannot be described as unseen test data or used to support generalization. |

## 2. Recommended main-paper architecture

The main paper should be a coherent method paper, rather than a catalogue of every available table.  The following layout is designed to fit the nine-page initial-submission limit once real figures replace placeholders.

| Main-paper component | Target space | Required content |
|---|---:|---|
| Title, abstract | 0.35 page | State the same-scene viewpoint-cut setting, causal mechanism, and the three multi-person datasets.  Avoid an unqualified “outperforms all methods” claim. |
| 1. Introduction + Fig. 1 teaser | 1.10 pages | Explain the continue-versus-reset failure mode, define the narrow task, show a real fixed multi-person cut, and give three precise contributions. |
| 2. Related work | 0.55 page | Organize by recurrent 3D/4D reconstruction, human--camera reconstruction, and temporal/cross-view association.  Distinguish HumanMM/Multi-THuMBS by task framing without inventing numerical comparisons. |
| 3. Method + Fig. 2 | 2.25 pages | Problem formulation; read-only learned coarse gauge; clean-state ownership; prediction-only association; shared translation; causal invariants and training loss.  This is the technical centre of the paper. |
| 4. Protocol | 0.55 page | Frozen RGB-only manifests, evaluator isolation, aggregation units, W/WA/ATE/IDF1/coverage interpretation, and causal-versus-offline distinction. |
| 5. Main multi-person results + Table 1 | 1.25 pages | Three multi-person datasets, strict Human3R control, Bridge3R, and a compact clearly-labelled EgoBody public-reference block.  Keep the Harmony4D W trade-off in the table. |
| 6. Mechanism evidence + Table 2 | 0.90 page | EgoBody chain: strict continuation, clean reset, learned coarse gauge, association, shared translation, final causal route.  Explain that oracle-boundary rows are counterfactual controls. |
| 7. Boundary/repeated-cut evidence + Fig. 3 | 0.95 page | One concise Harmony4D association paragraph, one concise repeated-cut/no-cut paragraph, and two or three real qualitative cases. |
| Conclusion and reproducibility/ethics pointers | 0.25 page | Restate scope and direct readers to the supplement/release package. |

The page allocation is a target rather than an excuse to shrink core method detail.  After actual figures are inserted, the compiled PDF is the authority: material must move to the supplement until the main text ends on page 9 before references.

### 2.1 Table 1: one honest primary result table

Replace the current internally-only presentation with one compact table titled approximately **“Multi-person same-scene viewpoint cuts and executable references.”**

1. Retain the paired strict-Human3R and Bridge3R rows for EgoBody, EgoHumans, and Harmony4D on their stated common subsets.
2. Retain only `W`, `WA`, `ATE-Sim3`, `IDF1`, and `Coverage` in the main table.
3. Add a small indented EgoBody-only block for TRACE and PromptHMR-SPEC, including accepted-match support `N` and `N/A` where a native camera trajectory is unavailable.
4. Bold only comparisons that are valid under the same evaluator and same denominator.  The external rows are an availability-aware executable reference, not a single aggregate leaderboard.
5. Preserve the Harmony4D adverse W and coverage values.  The paper gains credibility by making the mechanism-specific trade-off explicit.

This is preferable to silently hiding public baselines in the supplement, while still avoiding invalid cross-interface ranking.

### 2.2 Table 2: mechanism ablation, not another baseline table

Keep the EgoBody 43-recording macro and the following causal decomposition:

1. strict streaming continuation;
2. clean reset with oracle boundary;
3. learned coarse gauge with oracle boundary;
4. prediction-only association under the causal route;
5. Bridge3R with shared translation under the causal route.

The caption must label the oracle-boundary rows as non-deployable counterfactual controls.  After the missing token-removal ablation is completed, add the no-token shadow row here only if it uses the same cases, checkpoint, evaluator, and detector policy.  Do not combine it with external baselines.

### 2.3 Required figures

All final figures must be based on fixed real cases and real outputs; generated illustrations may assist only with a clean method schematic, not with qualitative evidence.

| Figure | Main-paper purpose | Required content |
|---|---|---|
| Fig. 1 — teaser | Establish the problem before the method. | One fixed EgoBody/EgoHumans/Harmony4D multi-person case: pre-cut RGB, first post-cut RGB, strict continuation versus Bridge3R camera/human overlay, consistent person colours/IDs, and a concise gauge-drift callout. |
| Fig. 2 — method | Make the causal state transaction visually unambiguous. | Detector proposal; read-only historical/shadow branch with correction tokens; clean-reset future branch; prediction-only association; shared translation; “history is emitted and immutable / future state is clean.”  Clearly distinguish runtime inputs from evaluator-only GT. |
| Fig. 3 — qualitative evidence | Make the numerical claim interpretable. | Two or three pre-registered fixed cases, preferably covering a large-angle cut and a crowded case.  Show temporal progression around the cut, cameras, meshes/skeletons, and persistent IDs for strict versus Bridge3R.  Include no cherry-picked claim; list case IDs in the caption or supplement. |

## 3. Supplementary-material architecture

The supplement should be a complete audit trail, not a duplicate main paper.  Reorganize the current sections into the following reader order.

1. **A. Complete protocol and evaluator isolation**
   - Dataset-specific manifest construction, camera-pair strata, RGB-only runtime contract, GT-only evaluator contract, metric definitions, topology mapping, and aggregation.
   - Explain the difference between coverage, accepted-match availability, and conditional geometry.

2. **B. Implementation, training provenance, and external adapters**
   - Frozen checkpoint/configuration, trainable parameter scope, causal detector, adapter versions, hardware, runtime/memory measurement, and executable-baseline eligibility rules.
   - Add the completed training/benchmark-overlap audit here.  No unresolved “audit pending” language may remain in the final PDF.

3. **C. Complete multi-person results**
   - EgoBody: recording macro, uncertainty/paired analysis, local/body and camera results, boundary/temporal values, angle strata, detector accounting, full public-method ledger.
   - EgoHumans: complete external table, camera-angle/action strata, capture-cluster paired analysis.
   - Harmony4D: complete common-set table, train-only blend sensitivity, association audit, multi-cut/no-cut control.

4. **D. Single-person and repeated-event evidence**
   - AIST++ CS150 table with causal and offline blocks visibly separated.
   - MC150-3 and MC150-4 tables, their immutable manifests, and the composition-only interpretation.
   - Document the GVHMR availability pilot and why it was not allowed into the global AIST table.

5. **E. Full component and sensitivity studies**
   - Completed correction-token/no-token, camera-only/human-only, association, and shared-translation ablations.
   - Retain the Harmony4D train-only blend grid, clearly marked as descriptive and non-reselecting.

6. **F. Additional qualitative results and failure taxonomy**
   - More fixed examples, no-cut negatives, cases where global W does not improve, and a short scoped failure taxonomy.

7. **G. Scope, related-protocol context, and release checklist**
   - HumanMM/Multi-THuMBS comparison context without fabricated direct numbers, limitation statement, licenses, release contents, reproducibility checklist, ethics statement, and AI-use disclosure in the form required by the official template.

The final supplement must not contain red `TODO-*` labels, “work in progress,” “pending,” or a table of experiments that the authors did not complete.  Uncompleted material should be omitted and any necessary scope limitation stated plainly.

## 4. Submission-critical work, in priority order

### P0 — lock the evidence and create a new working revision

- Preserve `v019` unchanged as a sealed evidence source.
- Create the next manuscript version only after this plan is accepted.
- Record source hashes for every table/figure, the selected checkpoint, immutable manifests, evaluator versions, and public baseline adapters.
- Do not add AvatarReX or THuman02 formal results under the current checkpoint.

### P1 — mandatory scientific closure before submission

1. **Complete the correction-token ablation.**  The learned coarse correction-token pathway is a central contribution, so the final paper cannot retain a `TODO-EXPERIMENT` in its place.  Re-upload the minimal replay-ready EgoBody evidence needed to run the same-manifest chain; freeze the row list before reading results.
2. **Complete the training/benchmark overlap audit.**  Establish event/clip-level intersections between the V14.1 fine-tuning inputs and EgoBody, EgoHumans, Harmony4D, and AIST.  If any overlap exists, revise dataset labels and claims rather than concealing it.
3. **Produce real qualitative figures.**  Select cases by a declared rule (for example, one fixed large-angle case plus one crowded case), not after looking for the most favourable rendering.  Preserve case IDs and source outputs.
4. **Measure runtime and peak memory.**  Use one fixed GPU, fixed resolution, fixed clip length, warm-up policy, and report causal Bridge3R versus strict Human3R.  Do not compare causal per-frame latency directly with offline whole-video PromptHMR latency.
5. **Remove all author-facing placeholders.**  No TODO text, local absolute paths, version-history prose, author identity, or unverified statement may enter the submission PDF.

### P2 — strongly recommended strengthening

1. Replay the direct boundary-association audit on EgoBody and EgoHumans after inputs are restored.  Then report a cross-dataset statement only if every dataset uses the identical evaluator-only protocol.
2. Verify camera error decomposition (translation, rotation, and scale) against the common evaluator; retain adverse values where present.
3. Curate a small licensed natural edited-video qualitative set only if its evaluation claim can stay strictly no-GT and scoped.  It is not required for the central benchmark claim.
4. Add release scripts that regenerate every published table from frozen ledgers, together with one anonymous installation/inference smoke test.

### P3 — manuscript rewrite and compression

1. Rewrite the abstract, introduction, results paragraphs, and conclusion around the narrow supported claim in Section 0.
2. Promote the compact availability-aware public-reference block to Table 1; move all wide ledgers and every per-stratum table to the supplement.
3. Keep detailed AIST CS150 and MC150 evidence in the supplement; a main-paper sentence may direct readers to it but must not claim an unqualified single-person win.
4. Merge repeated descriptions of manifest freezing, evaluator isolation, and metric semantics so each appears once in the main text and once in detailed supplementary form.
5. Replace provisional prose such as “retained” or “working paper” with final academic wording while preserving the factual caveats.

### P4 — ICLR packaging and verification gate

- Build from `main.tex` into one PDF only; do not submit separate main/supplement PDFs.
- Confirm that page 9 contains the end of main text and that references start afterwards.
- Run LaTeX/BibTeX until references and cross-references stabilize; require zero fatal errors, no undefined citations/references, no overfull critical tables, and legible figures at 100% zoom.
- Inspect the anonymous metadata, ICLR style version, page size, PDF fonts, hyperlinks, captions, continuous numbering, and all supplementary references.
- Search the complete source/PDF for `TODO`, `pending`, `placeholder`, `work in progress`, local paths, and author-identifying text.
- Export a clean Overleaf ZIP containing only sources, bibliography, style files, final figure assets, and documented build instructions; keep raw data, checkpoints, logs, and evidence ledgers outside the archive but record stable release locations.

## 5. Explicit exclusions and non-negotiable reporting rules

- Do not fabricate numerical rows for HumanMM or Multi-THuMBS; their code/data are unavailable for a shared execution contract.
- Do not merge causal streaming and offline full-video methods into a single latency-equivalent ranking.
- Do not hide the Harmony4D W/coverage trade-off, AIST anchor/seam-root trade-off, baseline failures, or evaluator-unavailable cases.
- Do not select a different model, camera blend, test subset, or qualitative example based on final-test performance without documenting the selection rule and revising the protocol status.
- Do not claim AvatarReX `lbn1` or THuman02 as unseen data under the fixed V14.1 checkpoint.

## 6. Completion definition

The paper is ready for final ICLR packaging only when all P1 items are complete, all main and supplementary tables/figures trace to frozen evidence, the page/anonymous-template checks pass, and the final PDF contains neither TODOs nor deferred-study placeholders.
