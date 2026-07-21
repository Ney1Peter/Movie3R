# V11 Retained Geometry Integrity Audit

## Problem

Legacy metric scaling improved boundary camera metrics by independently changing camera/root
and scene scales. SMPL-X body dimensions were not transformed by the same
similarity, so feet could penetrate the reconstructed floor even when cameras
looked aligned.

V11.2 tested an explicit root correction that restores the foot/ground contact
proxy and then re-solves boundary translation. This fixes the 3D contact proxy,
but the required root displacement breaks the original Human3R image-space
human reconstruction.

## 180-Cut Results

| Method | Camera T mean/P95 | Rotation mean/P95 | Foot/ground distortion | Human reprojection shift | Rigid local geometry |
|---|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715/4.123 m | 24.20/73.61 deg | 0 | 0 px | yes |
| Legacy independent metric scaling | 0.434/1.040 m | 12.09/37.75 deg | 0.515 m | 29.9 px | no |
| V11.2 Contact-Preserving | 0.465/1.003 m | 12.09/37.75 deg | 0 | 112.1 px | no |
| V11.1 Conditional Wide Rotation | 1.568/3.798 m | 12.09/37.75 deg | 0 | 0 px | yes |

V11.2 requires an average `0.515 m` camera-frame SMPL-X root correction. On the
selected MVHuman viewer examples, this can exceed `0.8 m`, producing hundreds
of pixels of reprojection displacement. It is therefore a diagnostic and not a
valid final bridge.

## Selection

The selected integrity-preserving candidate is:

```text
hard reset Human3R
-> keep the original Human3R shot gauge
-> conditional wide-baseline/torso rotation
-> explicit raw-gauge human-root translation solve
-> one rigid shot-level SE(3) for camera, pointmap, and SMPL-X
```

This candidate gives up the large camera-translation gain from independent scaling, but
it improves both translation and rotation over Fixed Explicit while preserving
the original Human3R body scale, image reprojection, and human/background
contact relation exactly.

## Decision Rule

Future candidates must be rejected if they improve camera metrics by changing
shot scale without a coherent transform of camera, pointmap, and the full
SMPL-X body. Evaluation must jointly include camera error, scene continuity,
human reprojection, body-scale continuity, and foot/ground contact.
