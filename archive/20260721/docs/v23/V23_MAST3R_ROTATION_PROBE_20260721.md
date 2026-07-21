# V23 MASt3R Rotation Probe

## Goal

Test whether the locally available frozen MASt3R metric checkpoint can repair the remaining MVHuman rotation tail after V22, without learning a selector or changing metric translation.

## Candidates

1. Background-only reciprocal MASt3R descriptor matches -> Essential Matrix -> relative rotation.
2. MASt3R new-view pointmap in the old camera frame -> dense 3D-2D PnP -> relative rotation.
3. A fixed physical consensus check requiring:
   - DA3 background/root scale ratio below `0.8`;
   - MASt3R metric baseline / V22 baseline in `[0.7, 1.3]`;
   - Essential and PnP rotations within `20 deg`;
   - minimum Essential and PnP inlier ratios.

## Result

On the 18 cuts selected only by the deployable `background/root < 0.8` instability cue:

| Method | Rotation mean | Median | P90 | P95 |
|---|---:|---:|---:|---:|
| Fixed Explicit | 43.51 | 42.75 | 66.15 | 73.30 |
| V22 | 23.71 | 17.71 | 44.66 | 60.44 |
| MASt3R Essential | 102.49 | 104.61 | 158.94 | 178.52 |
| MASt3R PnP | 96.91 | 100.14 | 158.25 | 165.67 |

The physical consensus accepted only 2 of 18 cases. One improved strongly (`38.3 deg -> 2.2 deg`), while one changed from `3.7 deg -> 5.8 deg`. Across all 180 cuts this would reduce mean rotation by only about `0.19 deg`; P95 and catastrophic rate remain unchanged.

## Decision

Stop the MASt3R branch. Its occasional large gain is real, but too sparse to justify another large cut-time model and consensus path. It is not included in V22.

The remaining MVHuman rotation tail still requires a more reliable explicit wide-baseline rotation cue. Raw frozen MASt3R pair geometry is not that cue in the current data.
