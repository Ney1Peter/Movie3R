# Movie3R Latest Frozen Model

> Freeze date: 2026-07-23
>
> Method version: `V14.7 Shot-Aware Uniform Similarity Re-anchoring`
>
> Release tag: `movie3r-single-v14.7`
>
> Scope: short-shot, sparse-camera-cut streaming re-anchoring

This file is the canonical entry point for the current default single-human
method. It is not the catalog for every retained version. The trained V9 model,
this single-human release and the V20 multi-human research release are indexed
separately in `versions/README.md`.

## Current Default

Formal release identity: **Movie3R-Single V14.7**.

```text
streaming RGB + intrinsics + cut trigger
-> frozen Human3R
-> pre-decode hard reset at a camera cut
-> Fixed Explicit coarse anchor
-> V16 bounded torso-motion rotation, 20 deg
-> V11.4 fused DA3/Keypoint shared shot scale
-> one explicit translation solve
-> one fixed shot-level Boundary for the new short shot
-> Align-Then-Commit
```

The Boundary is applied consistently to camera translation, pointmap, SMPL-X
root, root-centered body offsets, joints and vertices. It is estimated once at
the cut and then reused; it is not re-estimated per frame.

## Default-Off Modules

- Conditional VGGT is off. It is an optional rotation-tail rescue enabled only
  with `--enable_vggt` in the recurrent audit/runtime path.
- V14.2 continuity memory is off. It can be applied after alignment for
  shape/scale/local-pose smoothing, but it does not improve Boundary accuracy.
- V14.3 coupled root and V14.4 Unified Human/DA3 are diagnostic branches, not
  the retained default.

## Version Identity

The current standalone method is **V14.7 Shot-Aware Uniform Similarity
Re-anchoring**. Its scale block comes from V11.4, its rotation correction comes
from V16, and V14.6 is the component-necessity audit that froze this selection.
V14.7 introduces a single unambiguous method identity; it does not change the
frozen V14.6 numbers.

Legacy names:

| Legacy name | Current meaning |
|---|---|
| V47 | V11.1 Conditional Wide comparison, not default |
| V46 | V11.2 Contact-Preserving diagnostic |
| V53 | V11.4 Uniform Similarity, retained scale block |

## Main Evidence

All numbers below use the same 180-cut evaluator with VGGT disabled.

| Method | Camera translation | Rotation | Human root | Joints | Scene |
|---|---:|---:|---:|---:|---:|
| Fixed Explicit | 0.712 m | 24.20 deg | 0.234 m | 0.290 m | 0.483 m |
| Fixed + V16, raw scale | 0.518 m | 16.04 deg | 0.163 m | 0.223 m | 0.526 m |
| Current V11.4 | **0.463 m** | **16.04 deg** | **0.163 m** | 0.225 m | 0.536 m |

Paired V16-to-V11.4 camera improvement is significant (`p=0.00107`). The
scene metric worsens from `0.526 m` to `0.536 m` (`p=0.038`). DA3-only and
Keypoint-only scale variants are not independently significant; DA3 and
Keypoint R-CNN are retained as internal cues of the fused V11.4 scale rule.

On the capture-disjoint 60-cut holdout, the no-VGGT default improves camera
translation from `0.663 m` to `0.508 m`, while scene changes from `0.475 m` to
`0.547 m`.

## Valid Scope

This release is a short-horizon camera-human re-anchoring method. It is most
appropriate for one or two sparse cuts followed by a short shot. It is not an
unlimited-horizon mapping system: the true recurrent 8-cut audit reaches
`0.946 m` camera drift and `59.03 deg` rotation drift.

The method prioritizes camera-human placement. It does not establish complete
camera-human-scene closure, and its scene trade-off must be reported.

Current experiments use GT cut indices only as trigger signals. Automatic cut
detection is not yet a validated part of this frozen release.

## Canonical Files

| Purpose | Path |
|---|---|
| All formal releases | `versions/README.md` |
| Single release manifest | `versions/v14.7-single/manifest.json` |
| Full architecture and all ablations | `docs/movie3r/CURRENT_MODEL_FULL_ARCHITECTURE_AND_ABLATION.md` |
| Standalone current method | `docs/movie3r/V14_7_SHOT_AWARE_UNIFORM_SIMILARITY_REANCHORING.md` |
| Short-shot freeze record | `docs/movie3r/V11_4_SHORT_SHOT_METHOD_FREEZE.md` |
| Component audit | `docs/movie3r/V14_6_ALIGNMENT_COMPONENT_NECESSITY_AUDIT.md` |
| Final leakage/streaming audit | `docs/movie3r/V14_5_FINAL_GEOMETRY_STREAMING_AUDIT.md` |
| Unified evaluator and scale ablations | `scripts/v14_4_unified_similarity_reanchoring_probe.py` |
| True recurrent 1/2/4/8-cut audit | `scripts/v14_5_true_recurrent_multicut_audit.py` |
| Recurrent 3D viewer | `scripts/v14_5_multicut_interactive_viewer.py` |
| Arbitrary image multi-cut demo | `scripts/v14_7_custom_multicut_demo.py` |
| Arbitrary image multi-cut viewer | `scripts/v14_7_custom_multicut_viewer.py` |

The viewer currently reads a historical Conditional-VGGT recurrent cache and
labels it as such. It must not be presented as the new no-VGGT default result.

## Reproduction

Component audit:

```bash
TMPDIR=output/v14_5_final_audit/tmp \
.venv/bin/python scripts/v14_4_unified_similarity_reanchoring_probe.py \
  --device cuda:6 \
  --output_dir output/v14_6_alignment_component_necessity/full180_no_vggt
```

True recurrent audit, VGGT off by default:

```bash
TMPDIR=output/v14_5_final_audit/tmp \
.venv/bin/python scripts/v14_5_true_recurrent_multicut_audit.py \
  --device cuda:5
```

Use `--enable_vggt` only for an explicit optional-tail experiment.

## Detailed Reading Order

1. `LATEST_MODEL.md`
2. `docs/movie3r/V14_7_SHOT_AWARE_UNIFORM_SIMILARITY_REANCHORING.md`
3. `docs/movie3r/CURRENT_MODEL_FULL_ARCHITECTURE_AND_ABLATION.md`
4. `docs/movie3r/V14_6_ALIGNMENT_COMPONENT_NECESSITY_AUDIT.md`
5. `docs/movie3r/V14_5_FINAL_GEOMETRY_STREAMING_AUDIT.md`
6. Historical V14.3/V14.4 reports only when studying failed alternatives
