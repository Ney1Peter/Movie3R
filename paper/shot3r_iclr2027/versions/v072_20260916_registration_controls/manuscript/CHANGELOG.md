# Manuscript changelog

## v072 typography and supplementary-table normalization (2026-09-18)

- Changed the `Shot3R` and `Human3R` macros from forced sans serif to the
  surrounding Times-compatible Roman face; bold table entries now use the
  actual Nimbus Roman Bold font.
- Added explicit regular, bold, italic, and bold-italic mappings for local
  XeTeX validation so that missing font-family inference cannot silently
  replace bold Roman text with regular text. The official pdfLaTeX route
  continues to use the ICLR template's `times` package.
- Standardized every rendered supplementary-table body at 9 pt and retained
  8 pt only for table notes.
- Removed whole-table scaling from the rendered supplement. Reflowed the
  wide scope table and split the OnlineHMR and paired-registration summaries
  into stacked panels so that all values remain legible at the common size.
- Rebuilt the English and Chinese PDFs. The English build has no overfull
  boxes, undefined citations/references, Type 3 fonts, or unembedded fonts.

## v072 traditional geometric-registration controls

- Added complete-test-set geometric controls built from identical Human3R per-shot-reset predictions: pelvis translation, prediction-only robust human-joint SE(3), and offline full-shot scene FPFH/RANSAC+ICP.
- Added a compact main-paper table reporting W, camera ATE, IDF1, and EgoHumans fitting success; retained the qualified claim that Shot3R has the best aggregate joint result without claiming a statistically significant universal W advantage over human-joint SE(3).
- Added a dedicated supplementary section with complete W/WA/ATE/IDF1/Coverage, 100,000-draw clustered bootstrap intervals, fixed viewpoint strata, fitting failures, person-count analysis, boundary diagnostics, and fitting-only runtime.
- Added viewpoint and failure visualizations and classic RANSAC, Kabsch, FPFH, and ICP citations.
- Bolded only the first occurrence of Shot3R in each Abstract and synchronized the supplementary cover with the final title.
- Synchronized the English and Chinese main-paper versions and verified 40-page English and 14-page Chinese builds with no undefined citations/references, fatal errors, or overfull boxes.

## v069 front-matter and Method narrative revision

- Selected the provisional title `Shot3R: Streaming Multi-Person 4D Reconstruction across Video Shots` from the recorded candidate list.
- Retained the confirmed v068 Abstract, Introduction, and Related Work narrative without changing experimental evidence.
- Added a compact unnumbered Method Overview around the conflicting roles of recurrent history at shot boundaries.
- Reorganized Method into recurrent reconstruction at shot boundaries, history-conditioned shot alignment, and cross-shot spatial and identity consistency.
- Integrated the former standalone alignment--state-propagation subsection into the recurrent formulation and consistency mechanism to remove repetition.
- Made the temporary lifetime of the history-conditioned path and the persistent lifetime of the reinitialized reconstruction state explicit in text and equations.
- Clarified that the shared transform is inferred causally at the boundary rather than fitted post hoc between complete reconstructed shots.
- Corrected and disambiguated the camera-to-world transform convention, accumulated transforms, association normalization, root/pelvis roles, and induced world-space scene mapping.
- Synchronized the English text, Chinese source comments, Chinese reading version, Figure 2 caption, and subsection terminology.
- Changed no model checkpoint, experimental result, experiment section, conclusion, or supplementary evidence.
- Verified 36-page English and 14-page Chinese builds with no undefined citations/references, fatal errors, or overfull boxes.

## v068 Abstract and Introduction narrative revision

- Reframed the Abstract around the conflicting roles of recurrent history at shot boundaries rather than a list of modules.
- Retained only the dataset-balanced W-MPJPE, mean IDF1, and EgoBody extreme-viewpoint headline numbers in the Abstract.
- Reorganized the Introduction into four substantive paragraphs and merged the former one-sentence multi-shot definition and generic result summary into their relevant paragraphs.
- Defined multi-shot input against synchronized multi-view observations and defined the common world frame as the reconstruction frame initialized by the first shot.
- Recast the central insight as reading history for alignment without recurrently propagating it into the new shot.
- Organized contributions as problem characterization, method, and empirical evidence.
- Renamed the first Related Work theme to world-frame human--scene reconstruction and aligned its scope with the human--camera--scene setting introduced earlier.
- Rephrased the stateful-online distinction so pre-transition history remains valuable as an alignment reference but is not propagated as new-shot recurrent state.
- Retained all Related Work citations and synchronized the English text, Chinese source comments, and Chinese reading version.
- Synchronized English source comments and the Chinese reading version, and standardized the Shot3R/Human3R macros to sans-serif styling.
- Changed no method, checkpoint, experimental result, figure, table, or supplementary evidence.

## v066 unified result audit and opening-page teaser

- Applied all twelve confirmed decisions in the v065 result audit without changing the method, checkpoint, or any retained numerical result.
- Removed MVHuman stress, AIST++ single-person, custom EgoHumans boundary-association, and same-checkpoint fine-grained masking results from the rendered submission while preserving their original records outside the anonymous package.
- Retained the three multi-person protocols, complete Harmony4D trade-offs, large-viewpoint evidence, state-propagation control, shared-translation sensitivity, local body diagnostics, and limited three-shot control with appropriately scoped claims.
- Added a per-metric Shot3R--JOSH common-support comparison on EgoBody, EgoHumans, and Harmony4D.
- Replaced the main runtime headline with standardized reconstruction throughput: 3.106 FPS for Shot3R versus 3.224 FPS for Human3R; retained separate seven-method full-system timing in the supplement.
- Moved Figure 1 to a non-floating block immediately below the author information and above the Abstract; scaled it so the complete English Abstract remains on page 1.
- Expanded the supplementary contents with linked Harmony4D entries for shared translation, association, and the three-shot control.
- Synchronized the English and Chinese reading versions and verified a 34-page anonymous English build with nine scientific pages and 65 visible references.

## v065 reference coverage expansion

- Expanded the visible bibliography from 41 to 66 cited works after auditing the reviewer communities around human mesh recovery, world-coordinate motion, online reconstruction, multi-shot video, human--scene reconstruction, and person association.
- Added direct multi-shot predecessors on TV-series tracking, cross-shot human recovery, and actor--environment reconstruction.
- Added representative HMR, online geometry, feed-forward reconstruction, 3D-aware tracking, and re-identification work without introducing a fourth Related Work theme.
- Rewrote and synchronized the English and Chinese Related Work while keeping the scientific paper at nine pages.
- Changed no method, experiment, retained numerical result, figure, or table.
- Verified a 37-page anonymous build: scientific paper pages 1--9, statements page 10, references from page 11, and supplementary material from page 16.

## v064 review consolidation and supplementary navigation

- Reframed the central claim around the conflicting effects of recurrent history: it supports inter-shot alignment, direct carry-over can cause within-shot drift, and reinitialization alone loses useful cross-shot context.
- Reorganized the state evidence as reset-only failure, same-checkpoint carry-over drift, the aligned state-propagation control, and the complete Shot3R result.
- Distinguished camera-local scene geometry from its transformed world-space realization in the method, equations, supplement, and captions.
- Added complete detector input, architecture, training, seed, threshold, trigger policy, and EgoBody/EgoHumans/Harmony4D first-trigger statistics.
- Simplified the main comparison to temporal category, transition handling, accuracy, identity, Coverage, and evaluable support; moved preprocessing and output scope to the supplement.
- Enlarged four key qualitative comparisons in the main paper and retained all seven methods in a dedicated supplementary subsection.
- Moved training convergence to the supplement and placed the supplied editable shot-alignment module beside its method explanation.
- Reformatted the supplement with a linked three-part contents page, a compact reading guide, and immediate continuation into Part I, following the supplied reference layout.
- Updated verified publication metadata while retaining Multi-THuMBS as arXiv and the Yang et al. IJCV article number.
- Verified a 35-page anonymous build: scientific paper pages 1--9, statements page 10, references from page 11, and supplementary contents on page 14.

## v063 method narrative and evidence plots

- Reordered Method to follow the inference pipeline: formulation, learned history-conditioned shot alignment, separation from recurrent state propagation, and camera--human consistency.
- Moved the alignment-module training objective, trainable-parameter scope, and convergence curve into Section 3.2.
- Reduced Abstract/Introduction result duplication while retaining representative quantitative evidence in the Abstract.
- Expanded the Conclusion around the dual role of history and the same-checkpoint state-isolation evidence.
- Added absolute viewpoint-response curves with capture-bootstrap intervals, post-transition camera-drift curves, and a seven-method full-system runtime plot.
- Moved the complete runtime and boundary-analysis tables to the supplementary implementation section.
- Kept the scientific paper at nine pages and retained the existing limitation scope without claiming validation on real film footage.
- Synchronized the Chinese reading edition and verified both compilation targets.

## v062 supplied teaser and pipeline replacement

- Replaced Figure 1 with the supplied `Shot3R_teaser_cropped_enlarged_editable_v9.pptx` and regenerated vector PDF/SVG exports.
- Replaced Figure 2 with the supplied `Shot3R_pipeline_minimal_corrected_20260909.pptx`; the layout is retained while `Camera latent residual` and `Geometry-based person association` match the method description.
- Cleared creator and last-editor metadata from the retained editable PPTX copies.
- Updated only hidden Chinese in-figure label comments and figure provenance; captions, scientific prose, methods, results, tables, and references are unchanged.
- Added a separately compiled Chinese reading edition that shares the updated figures and quantitative content without changing the anonymous English submission source.
- Verified that the complete English manuscript remains 33 pages and that the scientific paper, including the complete Conclusion, remains nine pages.

## v061 teaser subtitle removal

- Removed the small `Common world frame` subtitle from the upper-left teaser label as requested.
- Retained `Streaming 4D reconstruction` and `Multi-shot video stream` as the two concise input/output labels.
- Regenerated the editable PPTX-derived vector PDF and SVG.
- Changed no caption, scientific prose, method, result, table, citation, or other figure from v060.

## v060 smooth-timeline teaser replacement

- Replaced Figure 1 with the supplied `Shot3R_teaser_smooth_timeline_editable_v7.pptx` design.
- Changed `Unified 4D reconstruction` to `Streaming 4D reconstruction` so the teaser names the online/streaming output rather than suggesting a generic unified architecture.
- Changed `One world frame` to the paper-standard `Common world frame`.
- Changed `Streaming video` to `Multi-shot video stream` so the input modality is explicit without repeating the output label.
- Cleared editable-document author metadata and retained the updated editable PPTX together with vector PDF and SVG exports.
- Updated the Chinese source comment for all changed in-figure labels; the Figure 1 caption remains scientifically accurate and unchanged.
- Changed no scientific prose, method, experiment, numerical value, table, citation, or other figure from v059.
- Verified that the 33-page build still has nine scientific pages, with the complete Conclusion on page 9 and no undefined references/citations or overfull boxes.

## v059 nine-page main-paper and citation revision

- Reduced the scientific paper from the temporary v058 overflow to exactly nine pages without changing the ICLR template, font size, margins, or using negative spacing.
- Rewrote and compressed the Introduction around one state conflict: history is required for inter-shot alignment, but the complete old-shot recurrent state should not govern reconstruction in the new shot.
- Reorganized Related Work into three compact themes and added directly relevant recent work on world-coordinate human recovery and streaming reconstruction.
- Consolidated Method into four subsections while retaining the dual-path formulation, typed alignment tokens, camera--human transform, association rule, training objective, and trainable-parameter scope.
- Consolidated Experiments around setup, main reconstruction results, and analysis/ablations; retained all core tables and figures required to support the claims.
- Moved the complete seven-method full-system timing table into the main paper and kept its protocol distinct from controlled reconstruction-only timing.
- Added compact main-paper accounts of typed-token sensitivity and repeated-transition controls, including reported exceptions rather than claiming uniform dominance.
- Standardized `alignment--state-propagation` terminology in the controlled experiment and removed the potentially ambiguous `future-state` label.
- Added or activated citations for GLAMR, PACE, DUSt3R, Long3R, Point3R, ST4R, TROPHIES, AIST++, SMPL-X, AdamW, and related directly used methods.
- Kept paired Chinese source comments for rewritten English passages and captions.
- Verified a 33-page anonymous US-Letter build: scientific paper pages 1--9, statements page 10, references from page 11, and appendix contents on page 14.
- Verified no fatal errors, undefined references/citations, or overfull boxes; only three non-fatal underfull warnings remain.

## v058 pipeline wording replacement

- Replaced Figure 2 with the wording-corrected editable pipeline supplied on 2026-09-08.
- Regenerated the vector PDF and SVG from the supplied PowerPoint source.
- Changed no LaTeX text, caption, equation, table, citation, method, or experiment result.
- Verified a 35-page anonymous US-Letter build and a clean extraction build of the Overleaf archive.

## v057 P1/P2 completion

- Added Yang et al. 2026 as complementary joint multi-view work and clarified that Shot3R receives temporally ordered monocular frames rather than synchronized views.
- Tightened the contribution hierarchy around temporary alignment history and the independently propagated new-shot state, retaining association and the shared transform as supporting mechanisms.
- Added the 129/129-case EgoBody alignment--future-state control. Continued state improves W/WA without explicit association, whereas reinitialized state improves ATE; the paper reports this mixed outcome rather than claiming uniform gains from reset.
- Added capture-clustered viewpoint/camera-gain statistics on 90 EgoHumans cases from 27 captures with 50,000 bootstrap draws.
- Added the three-setting association-cue ablation on all 88 Harmony4D cases.
- Added a fixed-300-frame transition-count control with 0, 1, 3, and 5 transitions, while avoiding linear-scaling and bounded-memory claims.
- Updated the main paper, supplement, tables, reproducibility statement, and build documentation without changing prior results or checkpoints.
- Verified a 35-page anonymous US-Letter build with no undefined references, citations, or overfull boxes.

## v055 early qualitative placement

- Moved the full-width seven-method qualitative comparison to the top of the
  Experiments opening page, before Section 4.1.
- Integrated its short interpretation into the Experiments overview so that
  the visual evidence precedes protocol and metric details.
- Changed no figure panel, verified viewpoint angle, result, or method claim.

## v054 qualitative comparison

- Added the supplied three-row, seven-method qualitative comparison as a
  full-width main-paper figure after the quantitative reconstruction results.
- Added a concise qualitative-results paragraph and a self-contained caption
  that preserve method-specific output scope.
- Filled the three per-case viewpoint changes with the verified top-to-bottom
  values of 134.36, 149.85, and 176.75 degrees.
- Retained both the editable PPTX source and its vector PDF export.

## v053 recurrent-state isolation

- Added a same-checkpoint state-isolation control on the fixed EgoHumans
  formal90 protocol: 90 two-shot cases grouped into 27 capture macros.
- Compared the original Human3R checkpoint with state continued across the
  transition against the same checkpoint reinitialized on identical post-cut
  RGB frames; no Shot3R module is enabled in either control.
- Measured camera motion relative to the first post-cut pose, which cancels
  any fixed inter-shot world transform and isolates drift within the new shot.
- Added the primary 16-offset result, capture-level bootstrap intervals,
  paired Wilcoxon tests, and 10/32/49-offset sensitivity analysis to the
  EgoHumans supplement.
- Revised the Introduction, Method, main ablation discussion, Conclusion, and
  reproducibility statement so the loss of world-frame continuity under reset
  is no longer conflated with recurrent-state carry-over.
- Changed no method, checkpoint, training procedure, prior experiment value,
  or external-baseline result.

## v052 seven-method full-system runtime

- Cleared author and last-editor metadata from both editable PowerPoint figure
  sources before assembling the anonymous Overleaf package; figure content is
  unchanged.
- Replaced all six H4D FPS placeholders in the main comparison with audited
  micro-average throughput on five fixed 150-frame streams.
- Added the measured 0.894 FPS and carefully scoped 2.7--6.0x throughput result
  to the metric-bearing abstract.
- Clarified that multi-stage releases sum all required prediction-stage times
  and that the retained PromptHMR/TRACE records conservatively include their
  lightweight hardlink staging; no runtime value is changed.
- Added TRAM to the efficiency study and a seven-method supplementary table
  reporting mean $\pm$ population-standard-deviation seconds, FPS, temporal
  access, and camera/scene output scope.
- Defined full-system time as fresh process launch through native prediction
  save, including model loading and method-required prediction stages while
  excluding rendering and evaluation.
- Reported that Shot3R is 4.2--6.0$\times$ faster than PromptHMR, OnlineHMR,
  TRAM, and JOSH, while explicitly retaining the higher throughput of Human3R
  and TRACE and contextualizing TRACE's output scope and coverage.
- Kept the previous controlled 100-frame reconstruction benchmark as a
  separate component-level analysis and did not mix its FPS with full-system
  throughput.
- Changed no method, checkpoint, accuracy protocol, or existing result.

## v051 comprehensive main comparison

- Reorganized the primary comparison as one row per method with Offline,
  Semi-online, and Online groups.
- Reported PromptHMR, JOSH, OnlineHMR, TRACE, Human3R, and Shot3R together on
  EgoBody and EgoHumans, with method properties and five reconstruction
  measures per dataset.
- Preserved every method-specific evaluable count and stated explicitly that
  conditional geometry means with different support do not form one ranking.
- Added an FPS placeholder for a future standardized end-to-end profile rather
  than mixing it with the existing reconstruction-only timing.
- Added the complete six-method Harmony4D comparison to the supplement,
  including the metrics on which Shot3R is not best.
- Consolidated the former viewpoint summary table and angle-stratified plot
  into one full-width signed-gain Figure 3, preserving all reported values.
- Reorganized the mechanism table around boundary cue, transition operation,
  and explicitly unit-labelled EgoBody metrics; removed the ambiguous Online
  column from controlled configurations.
- Standardized the association/detection/cost table, including detector
  denominators and the scope of the reconstruction-only L20 timing.
- Changed no method, checkpoint, protocol, or existing result value.

## v050 terminology and notation audit (continuation)

- Standardized reader-facing scope terms: `multi-shot` for the input/task,
  `inter-shot` for camera geometry and spatial alignment, `cross-shot` for
  person association and trajectory continuity, `shot transition` for the
  event, and `shot boundary` for its frame location.
- Unified the learned representations as semantic, camera-alignment, and
  temporal-context `typed alignment tokens`; removed competing boundary-token,
  native-token, and correction-token wording from the compiled manuscript.
- Standardized person matching as geometry-based cross-shot person association
  and the output operation as a shared camera--human transform.
- Replaced ambiguous or implementation-log wording in the main paper and
  supplement, including `same-checkpoint`, `causal streaming`, `seam`, and
  relative-camera table labels, without changing any result.
- Disambiguated the pooled memory summary from pelvis locations by renaming it
  from $r_{b-1}$ to $m_{b-1}$, and wrote the shared translation as a homogeneous
  transform so that the coordinate composition is dimensionally explicit.
- Updated the editable Figure 2 deck and PDF, replacing `Future frames` with
  `Subsequent frames` to avoid suggesting look-ahead access.
- Verified the complete 33-page PDF with no undefined references, citations,
  or overfull boxes. The current draft allows the conclusion to extend onto
  main-paper page 10; final page reduction is deferred.

## v050 alignment-token sensitivity (continuation)

- Reframed the three internal tokens as typed alignment tokens with
  consistent semantic, camera-alignment, and temporal-context terminology.
- Added an inference-time masking analysis using the same jointly trained
  checkpoint on all ten
  held-out extreme-viewpoint MVHuman sequences, using annotated transition
  timing to isolate the alignment tokens from detector behaviour.
- Reported every retained human and camera metric while limiting the prose
  conclusion to the supported finding that the typed tokens primarily
  stabilize inter-shot relations.
- Kept the incomplete early V9 probes outside the submitted evidence and made
  no change to training, checkpoints, primary results, or existing baselines.
- The complete manuscript ends on page 32 after adding the supplementary table.

## v050 supplementary-material audit (continuation)

- Clarified that the two annotated-boundary rows are controlled diagnostics,
  not deployable settings, and removed the stale JOSH3R row from the related-
  method scope table because JOSH is already evaluated in the main table.
- Made the EgoHumans association table self-contained by naming evaluable
  clips and evaluator-excluded pair slots explicitly.
- Corrected dataset-specific camera metrics: EgoBody uses ATE-Sim3 and
  EgoHumans uses ATE-SE3; clarified the mixed-dataset viewpoint table and the
  Harmony4D boundary-translation column.
- Replaced Figure 2 with the terminology-checked editable pipeline deck,
  exported its vector PDF/SVG, and marked the calibration-assisted output
  panel as a schematic world-frame illustration.
- Preserved all methods, checkpoints, protocols, numerical values, and the
  9-page main-paper / 30-page complete-manuscript layout.

## v044_20260907_alignment_terminology

- Unified the temporary route as the history-conditioned alignment path and
  its learnable component as the shot-alignment module.
- Reserved alignment-token terminology for the module's three internal token
  representations and removed competing correction-token and relation-module
  aliases from reader-facing prose.
- Updated the editable method schematic and its PDF export to use the same
  direct state-role description.
- Preserved all v043 methods, formula content, experiments, and table values.

## v043_20260907_contribution_reframing

- Reorganized the contribution list into the state conflict, the Shot3R
  realization, and the corresponding reconstruction/runtime evidence.
- Replaced the broad ``general principle'' and ``information lifetimes''
  wording with the concrete distinction between temporary alignment history
  and the recurrent state propagated through the new shot.
- Preserved all v042 methods, formulas, experimental results, tables, and
  figures.

## v042_20260907_introduction_and_runtime_audit

- Reframed the Introduction around streaming multi-person reconstruction from
  sequential monocular shots and the discontinuities created by shot changes.
- Positioned existing multi-shot processing against the desired streaming
  setting before introducing the state conflict at a shot boundary.
- Clarified the central observation: historical state may be read temporarily
  for alignment but should not control recurrent reconstruction in the new
  shot.
- Preserved all v041 methods, experimental results, tables, and figures.

## v041_20260906_josh_three_dataset_integration

- Added the completed JOSH offline full-sequence reference on all 129
  EgoBody, 90 EgoHumans, and 88 Harmony4D inputs without changing any prior
  method or result.
- Expanded the main comparison table with the audited EgoBody and Harmony4D
  JOSH rows and preserved each metric's valid support.
- Rewrote only the JOSH-related comparison text to distinguish reconstruction
  accuracy, identity, Coverage, temporal access, and native failures.
- Added supplementary execution/support accounting for all 307 streams and
  retained both Shot3R- and JOSH-favouring outcomes.
- Added versioned machine-readable JOSH summaries and independent audits for
  the two newly completed datasets.
- Compactly rephrased the unchanged conclusion claim so that the complete
  conclusion and limitation remain on main-paper page 9.

## v039_20260906_onlinehmr_extension_integration

- Integrated the verified 354-stream OnlineHMR supplementary campaign while
  preserving all existing results and artifacts.
- Added the strongest repeated-transition AIST++ paired result to the main
  paper, together with the complementary local and boundary outcome.
- Added OnlineHMR to the AIST++ single/repeated-cut, MVHuman, and Harmony4D
  three-shot supplementary tables.
- Added AIST++ paired intervals and MVHuman viewpoint-stratified paired
  results, with complete failure and support accounting.
- Preserved OnlineHMR-favouring metrics, MVHuman early triggers, and the small
  Harmony4D multi-cut denominator in the rendered interpretation.

## v038_20260906_baseline_scope_and_josh_audit

- Added JOSH and CVPR 2022 Multishot to the method positioning.
- Replaced the broad "code unavailable" explanation for HumanMM and
  Multi-THuMBS with a precise account of missing executable inference,
  pretrained models, and complete same-protocol settings.
- Added a supplementary capability and executability table for related
  methods absent from the same-input numerical tables.
- Recorded the JOSH Gate A/B result: all available dependencies and models
  were audited, but the official demo stopped at gated SAM3 access before
  producing predictions; no development or test sample was consumed.
- Preserved all v037 experimental artifacts, numerical tables, figures, and
  the nine-page main-paper layout.

## v037_20260906_supplement_revision

- Reorganized the supplement into five coherent sections and ordered the
  additional experiments by research question.
- Reduced the rendered table set from 36 to 24 without changing retained
  measurements.
- Removed repeated, all-constant, and one-row tables; combined closely related
  results into compact presentations.
- Simplified captions and removed internal audit/manifest terminology from
  rendered supplementary prose.
- Added colored method-property and component indicators to main Tables 1 and
  3, following Human3R's compact table style.
- Added a dedicated appendix table of contents with automatically resolved
  page references for Sections A--E and D.1--D.7.
- Kept all figure content unchanged.

## v036_20260906_final_training_curve

- Established the final model's 72-epoch AvatarReX/THuman training
  configuration as the sole training account.
- Restored the exact relation-correction objective and its 19.62M trainable
  parameters.
- Added a log-recovered training/validation curve and machine-readable source
  values to the supplement.
- Preserved the metric-bearing abstract and all existing result-table values.

## v035_20260905_metric_abstract_and_training_details

- Selected the concrete-metric abstract without inserting unmeasured
  end-to-end runtime placeholders.
- Replaced the grouped Method loss with the exact implemented event objective,
  weights, and term descriptions.
- Added training-source counts, update count, resolution, optimizer schedule,
  precision, and checkpoint policy to Experimental Setup.
- Preserved all retained results and the nine-page main-paper layout.

## v033_20260905_shot3r_bilingual_first_draft

- Adopted the Shot3R title, method name, and alignment--state decoupling
  narrative throughout the compiled manuscript.
- Rewrote Sections 1--5 and added paired Chinese source comments.
- Consolidated Related Work and Experiments, integrated OnlineHMR, and removed
  the standalone Discussion.
- Reorganized the method into Overview, Stateful Online Human Reconstruction,
  Decoupling Alignment from State Propagation, Camera--Human Consistency
  across Shots, and Learning Alignment at Shot Transitions.
- Rebuilt the three main figures and four main tables while retaining source
  images and numerical evidence.
- Fixed main-table horizontal overflow and kept the scientific paper within
  nine pages.

## v031_20260902_read_reset_register_and_baseline_plan

- Adopted the final Read--Reset--Register title and method vocabulary.
- Rewrote the abstract without per-dataset metric enumeration.
- Recast the contributions as four concise claims and added OnlineHMR, TRAM,
  and JOSH to Related Work.
- Preserved every v030 experimental value, denominator, and limitation.

## v030_20260901_fact_audit_and_evidence_revision

- Reframed the method as a causal same-scene camera--human boundary operation
  with an explicit information set and propagated-state invariant.
- Added the temporal-token equation, complete coarse-gauge composition,
  transformation scope, and prediction-only association cost.
- Added data-generated viewpoint-interaction and detector-generalization
  artifacts and integrated them into the main paper and supplement.
- Rebuilt the mechanism table so oracle timing is diagnostic and only the
  causal row is deployable.
- Added dual association denominators and zero active runtime abstention.
- Retained Harmony4D W degradation, AIST++ anchored-root degradation,
  repeated-cut camera drift, and MVHuman early detector triggers.
- Added final recoverable training settings and preserved unknown
  hardware/duration/multi-seed details as explicit limitations.
- Included AI Use, Ethics, and Reproducibility statements before References.
- Verified nine scientific pages and one continuous 28-page combined PDF.

## v029_20260830_mvhuman_audit_and_submission_package

- Added the held-out MVHuman stress audit, portable source package, and real
  extreme-view qualitative comparison.
- Preserved public-method support limitations and complete negative results.
