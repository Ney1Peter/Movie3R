# Provenance: real large-view qualitative figure

## Scientific status

This figure is a **metric-selected qualitative example**, not a random or
representative sample. It uses only RGB images and geometry renderings already
present in the frozen two-dataset demonstration export. No image, person,
camera, scene, or missing geometry was generated, inpainted, redrawn, or copied
between methods. The SVG adds only typography, borders, a cut arrow, layout,
and metric-consistent callouts describing identity continuity, world-scale
drift, global placement, and camera--human gauge coherence.

- Dataset: EgoHumans
- Case ID: `ego_test_legoassemble_003_legoassemble_extreme_cam03_cam04_b00301`
- Camera-rotation span: 176.74916252624004 degrees (displayed as 176.7 degrees)
- People: 3
- Pre-cut panel: export-local frame 4, original clip frame 49
- Post-cut and all method panels: export-local frame 5, original clip frame 50
- Cut: between original clip frames 49 and 50
- Figure metrics: case-level values from the frozen export manifest, not
  per-frame measurements of frame 50

## Case-selection rule

The rule was applied to the cases available in the frozen large-view demo
export:

1. require a camera-rotation span of at least 150 degrees;
2. require valid same-case, same-frame outputs for Strict Human3R, Bridge3R,
   official TRACE, and PromptHMR with official SPEC;
3. require Bridge3R to improve over Strict Human3R in W-MPJPE, WA-MPJPE, and
   IDF1;
4. prefer a case where Bridge3R is best on all three metrics among those four
   methods; and
5. use the camera-rotation span only as a final tie-break.

The EgoHumans case meets these conditions. This selection was made using the
reported case-level metrics and must not be described as random sampling or as
evidence that the chosen case is representative of the complete test set.

## Method boundaries

- Strict Human3R and Bridge3R have a scene channel in their respective demo
  payloads. Each scene channel comes from that method's own frozen-checkpoint
  replay; neither borrows geometry from the other method.
- TRACE and PromptHMR are mesh-only in this export. They do not provide dense
  scene point clouds, and none was added to their panels.
- TRACE's exported `cameras_c2w` is a diagnostic body-root proxy, not an
  official physical camera trajectory. The figure therefore makes no claim
  about an official TRACE camera reconstruction.
- The Strict Human3R and Bridge3R human/camera geometry used by the export comes
  from the immutable formal test cache. Ground truth was not used to construct
  the visualized predictions (`gt_used: false`).

## Source assets and transformations

Paths below are repository-relative. PNG crops were produced with ImageMagick
as exact integer crops and losslessly re-encoded. The source panels themselves
received no color, geometric, or generative editing.

| Figure panel | Original source | SHA-256 of original | Panel transformation | Packaged panel SHA-256 |
|---|---|---|---|---|
| (a) pre-cut RGB | `Movie3R/output/bridge3r_two_dataset_demo_v2/egohumans/_shared/color/000004.png` | `42376ff28775eeac6d17beab39bfffa3518f1be45f2878fa108d0fd58be095cc` | byte-for-byte copy; 640x450 | `42376ff28775eeac6d17beab39bfffa3518f1be45f2878fa108d0fd58be095cc` |
| (b) post-cut RGB | `Movie3R/output/bridge3r_two_dataset_demo_v2/egohumans/_shared/color/000005.png` | `b1b412fd00eab404fa577eb9c8b4911893a536da084208a9c5b4a55b4db96d75` | byte-for-byte copy; 640x450 | `b1b412fd00eab404fa577eb9c8b4911893a536da084208a9c5b4a55b4db96d75` |
| (c) Strict Human3R | `Movie3R/output/bridge3r_two_dataset_demo_v2/egohumans/strict/frames/000005.png` | `a27b6078e02053105fe613937cf5284cda7c610bbb6013fd6e80f05ecea59aa2` | crop `(x=640,y=85,w=640,h=437)` from 1280x522 | `0f00b209628ba76effe65f2e88a97298529bf51e74e6179ed10b4d3490d45ca6` |
| (d) TRACE | `Movie3R/output/bridge3r_two_dataset_demo_v2/egohumans/trace/frames/000005.png` | `cecf5fc5e7df83483dbaca4dfd0948222246739f3b08e1f53e486bdd61300042` | crop `(x=640,y=85,w=640,h=437)` from 1280x522 | `32658465f9bfc53371ba7a0a6fbe27b36ad491d2c0dc84f4e4c85e2feda87935` |
| (e) PromptHMR (SPEC) | `Movie3R/output/bridge3r_two_dataset_demo_v2/egohumans/prompthmr_spec/frames/000005.png` | `a1a0324a41436d9609b13241c5480291a25806174d22e2f137049bf9ddc5c115` | crop `(x=640,y=85,w=640,h=437)` from 1280x522 | `263d86e31dcb1428112d58376c1540c768ea20374b77270de28624d5f023125e` |
| (f) Bridge3R | `Movie3R/output/bridge3r_two_dataset_demo_v2/egohumans/bridge3r/frames/000005.png` | `c9a7f30921813ec3f8702db85708d3e8fdb5a0641459a38d0774ef3bf2175a50` | crop `(x=640,y=85,w=640,h=437)` from 1280x522 | `ecce8c110588a50ae1588190066064b9be1120675fb4171a772a3f8186958180` |

Packaged RGB images are displayed at 542x381.1 SVG units. Packaged method
renderings are displayed at 276x188.4 SVG units. All use
`preserveAspectRatio="xMidYMid meet"`; display scaling does not alter the
packaged source files.

## Quantitative values printed in the figure

| Method | W-MPJPE (mm) down | WA-MPJPE (mm) down | IDF1 up |
|---|---:|---:|---:|
| Strict Human3R | 393.1 | 208.1 | 0.514 |
| TRACE (official) | 2359.0 | 895.5 | 0.143 |
| PromptHMR (official SPEC) | 1056.1 | 971.8 | 0.422 |
| Bridge3R | **337.0** | **182.1** | **0.951** |

## Controlling records

- `Movie3R/output/bridge3r_two_dataset_demo_v2/manifest.json`
  (`b71c9b0227f5fe0cbfaff9d7996bd5fec620c04a8a36548d11289d65644f14ee`)
- `Movie3R/output/bridge3r_two_dataset_demo_v2/VIEWER_README.md`
  (`538cac7f3aeca1fb539d9a0a54fcb46e0bc9d41977f2555193651b35eb3dd96a`)
- Strict metadata SHA-256:
  `d0f6909452e6143495697f51c0aeccaa6c3afdefd71a1b3c46a7adf0ba568f17`
- Bridge3R metadata SHA-256:
  `878411e6deae96b8fddd311b1ca09f9e1cc83ebc9e207e7cd6c4c3b60c9b7cfb`
- TRACE metadata SHA-256:
  `b2779a26e9e33ad366c97dccc48d59b0a936eaa7275e200fa21fd6a8b2f0c38f`
- PromptHMR SPEC metadata SHA-256:
  `47cb5b01acab713f372e53490a2599dddc9144ebd5ae5a86b02ee02d2d179628`

## Figure artifacts

- Editable SVG SHA-256:
  `3cdf80389f8168eb6c3772f5ccdcd0c765a4919639746be60b0bb87dacc0d3c2`
- PDF SHA-256:
  `8ba5164aebb6a9c9daea178f6de4a71f13fcd876cf312f09db96e45fcb310456`
- PDF was exported from the SVG with CairoSVG 2.9.0 while allowing its local,
  repository-relative PNG references. The raster panels are embedded in the
  PDF; labels and layout remain vector content.
