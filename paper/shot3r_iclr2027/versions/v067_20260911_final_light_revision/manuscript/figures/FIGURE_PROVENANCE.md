# Figure provenance

All figures are anonymous and contain no experiment-version identifiers.

| File | Source | Data driven? | Scope |
|---|---|---:|---|
| teaser.pdf / .svg | `Shot3R_teaser_article_figure1_editable.pptx`, copied from the supplied cropped/enlarged v9 deck. The RGB panels are source frames from a temporally concatenated EgoHumans single-view-shot sequence; the central point-cloud/person/camera composition is explicitly schematic and retains the source deck's illustrative assets. | Partly | Figure 1 overview of three successive monocular shots, two cuts, streaming reconstruction, and persistent identity colours. The central composition is not a raw prediction or quantitative result. |
| method.pdf / .svg | `Shot3R_pipeline_Harmony4D_article_figure2_editable.pptx`, copied from the supplied minimally corrected 2026-09-09 deck. The layout is unchanged from the preceding pipeline; `Camera latent residual` and `Geometry-based person association` are the two terminology corrections. | No | Editable method schematic following the temporary history-conditioned alignment, reinitialized recurrence, geometry-based association, shared camera--human transform, and streaming-output protocol. The output panel is schematic and makes no quantitative claim. |
| egohumans_angle_strata.pdf / .svg | `tools/plot_egohumans_angle_strata.py` applied to the case-level formal-90 result file, with capture-cluster bootstrap. | Yes | W-MPJPE and IDF1 as a function of camera-pair angle. It contains no RGB, ground truth rendering, evaluator annotation, or synthesized image. |
| Shot3R_qualitative_comparison.pdf / editable source .pptx | Supplied editable three-row comparison deck. All RGB and reconstruction images are preserved; the verified viewpoint changes are inserted from top to bottom as 134.36, 149.85, and 176.75 degrees. | Yes | Main-paper qualitative comparison across six public references and Shot3R. Each panel retains the output scope and rendering supplied for that method; no absent camera or scene output is imputed. |
| qualitative_real_large_view.pdf / .svg | Real EgoHumans RGB and method-native outputs selected by the recorded metric rule; the SVG supplies editable layout and callouts. | Yes | Archived qualitative comparison retained for audit; it is not the current Figure 1 asset. No observation or reconstruction panel is synthesized. |

The angle-stratified plot is a pre-specified aggregate analysis; its
machine-readable means and intervals are stored next to the manuscript table
artifact. Its publication PDF is exported from the retained SVG with CairoSVG
so that the submission contains no Type-3 glyphs; this export changes no plot
data or geometry. The qualitative teaser is explicitly metric-selected rather than
representative, and its separate provenance record gives the selection rule,
panel hashes, and method-interface boundaries.

The editable Figure 2 source is retained next to the manuscript figures. The
visible output labels use the paper terminology ``World-frame output'' and
``Schematic illustration''; the original calibrated v3 deck remains preserved
in the publication asset directory.
