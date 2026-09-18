# Shot3R ICLR 2027 manuscript — v066

This revision consolidates all twelve decisions in the v065 result audit without changing the method, checkpoint, or any retained measurement. It keeps 65 visible references spanning human mesh recovery, world-coordinate motion, streaming geometry, multi-shot video, joint human--scene reconstruction, and cross-shot identity association.

The complete scientific paper remains nine pages. Figure 1 appears immediately below the author block and above the Abstract, following the Human3R-style opening layout; the Abstract remains complete on page 1. The conclusion ends on scientific-paper page 9; statements, references, and supplementary material follow without modifying the ICLR style or using negative spacing. The supplement opens with a three-part contents page, linked page numbers, and a compact reading guide.

`main.tex` is the only compilation root. It produces one anonymous US-Letter PDF containing the scientific paper, required statements, references, and supplementary material.

The manuscript presents streaming multi-person 4D reconstruction from temporally successive monocular shots. Its central design uses preceding-shot history only through a temporary alignment path, while an independently reinitialized state is propagated through the new shot. Geometry-based cross-shot person association and a shared camera--human transform support identity and world-frame consistency.

The retained evidence includes:

- six-method accuracy comparisons and seven-method full-system timing;
- a same-weight recurrent-state isolation study on 90 EgoHumans cases from 27 captures;
- an alignment--state-propagation control on all 129 EgoBody cases;
- capture-clustered viewpoint/camera-gain statistics on EgoHumans;
- an association-cue ablation on all 88 Harmony4D cases;
- a controlled 300-frame runtime/VRAM study with 0, 1, 3, and 5 transitions;
- large-viewpoint, three-shot, detector, and qualitative analyses.

The rendered paper omits the MVHuman weak-texture stress test, AIST++ single-person experiments, the custom EgoHumans boundary-association diagnostic, and same-checkpoint fine-grained masking tables. These original results remain outside the anonymous submission package. Harmony4D trade-offs, Coverage, common-support counts, and local body diagnostics remain visible.

The main qualitative figure enlarges OnlineHMR, JOSH, Human3R, and Shot3R; the complete seven-method comparison remains in the supplement. The main accuracy table retains every completed method on EgoBody and EgoHumans, while a supplementary scope table records preprocessing, camera, scene, and transition-handling properties.

Related Work remains organized around world-coordinate human reconstruction, stateful online reconstruction, and multi-shot human reconstruction. Direct multi-shot predecessors and representative HMR, online geometry, joint reconstruction, and identity-association work are placed within those three themes rather than added as a generic literature survey.

Reader-facing terminology is standardized as `multi-shot` for the input/task, `inter-shot` for camera geometry and spatial alignment, `cross-shot` for person association and trajectory continuity, `shot transition` for the event, and `shot boundary` for its frame location. The learned representations are semantic, camera-alignment, and temporal-context typed alignment tokens.

Rewritten English passages and captions are preceded by Chinese `% 中文：...` comments. These comments remain visible in Overleaf source but are omitted from the PDF. Editable PPTX/SVG files and private machine-readable audits remain in the working version and are excluded from the anonymous Overleaf archive.

See `BUILD.md` for compilation and validation.
