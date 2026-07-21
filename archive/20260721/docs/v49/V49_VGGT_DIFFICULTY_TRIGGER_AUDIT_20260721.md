# V49 VGGT Difficulty Trigger Audit

## Runtime Rule

The V32 trigger does not read source ID or GT view angle. It uses only:

- the remaining torso correction after Fixed Explicit;
- the size of the VGGT correction relative to torso;
- torso/VGGT direction agreement or conflict;
- VGGT forward/reverse rotation spread;
- image texture.

## 1079-Cut Audit

| Source | Count | Triggered | Rate | Mean GT view angle | Base R | Final R |
|---|---:|---:|---:|---:|---:|---:|
| AvatarReX | 288 | 0 | 0.0% | 120.9 deg | 5.77 | 5.77 |
| THuman | 288 | 0 | 0.0% | 120.1 deg | 4.83 | 4.83 |
| MVHuman100 | 287 | 74 | 25.8% | 118.0 deg | 24.59 | 20.88 |
| MVHuman200 | 216 | 84 | 38.9% | 103.8 deg | 28.02 | 20.86 |

AvatarReX and THuman contain the same `60-180 deg` view-angle buckets, but
Fixed plus torso remains around `5 deg` on all four buckets. Their large camera
span is therefore not difficult for the current explicit/human bridge.

Across all sources, VGGT trigger rate decreases rather than increases with GT
view angle:

| View-angle bucket | Trigger rate |
|---|---:|
| 60-90 deg | 20.8% |
| 90-120 deg | 18.8% |
| 120-150 deg | 12.5% |
| 150-180 deg | 3.7% |

The trigger follows Fixed error much more directly:

| Fixed rotation error | Trigger rate |
|---|---:|
| <10 deg | 0.2% |
| 10-30 deg | 6.8% |
| 30-60 deg | 35.5% |
| >=60 deg | 64.4% |

On 158 triggered cuts, base rotation improves from `34.90` to `18.38 deg`.
`130` improve by more than `5 deg` and `10` are harmed by more than `5 deg`.

## Conclusion

The rule is source-blind but its activation is empirically source-correlated:
only MVHuman currently produces the diagnostic pattern that requires VGGT.
Difficulty should be described as residual explicit/human ambiguity, not raw
camera span. The current rule should not be replaced by an angle threshold or
an MVHuman-specific switch.
