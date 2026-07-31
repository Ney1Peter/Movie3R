# V14 Human3R-Internal Root-Depth Feasibility

Date: 2026-07-30

## Question

Can Movie3R reduce post-cut human root-depth error without Depth Anything,
another pretrained model, future frames, or changing the learned coarse
Boundary `B0`?

The tested correction is deliberately camera-local:

```text
Human3R raw SMPL-X
+ Human3R pointmap / mask / pelvis detection
-> optional per-human local translation correction
-> unchanged learned B0
```

It does not recompute camera translation, alter the pointmap, or introduce a
per-person Boundary.

## Protocol

### MultiHuman B0 evaluation

The evaluation contains 180 cuts and 400 accepted GT-associated humans:

| Sequence | Role | Cuts | Humans |
|---|---|---:|---:|
| `three` | development | 41 | 122 |
| `dance` | frozen | 61 | 122 |
| `box` | frozen | 78 | 156 |

GT identity assigns anonymous detections for evaluation only. GT camera and
SMPL-X are used only for metrics. Candidate generation reads predicted outputs.

For every cut, the learned `B0` matrix is loaded unchanged from the existing
B0 identity-matching report. The fresh raw camera exactly matches the existing
Phase 2 cache: maximum translation and rotation difference are both zero.

### Near/far stress evaluation

The miner scanned 2,800 held-out AABB records and ranked the pre/post pair by
GT camera-person distance ratio and mask occupancy ratio:

| Source | Records | Distance ratio median/P95/max | Occupancy ratio median/P95/max |
|---|---:|---:|---:|
| AvatarReX | 600 | 1.226 / 1.720 / 2.071 | 1.168 / 1.609 / 2.460 |
| THuman | 600 | 1.103 / 1.400 / 2.134 | 1.237 / 1.865 / 3.164 |
| MVHuman100 | 800 | 1.083 / 1.289 / 1.560 | 1.240 / 1.830 / 2.352 |
| MVHuman200 | 800 | 1.085 / 1.383 / 3.043 | 1.197 / 1.724 / 3.184 |

The top eight unique stress pairs per source were evaluated with a fresh,
single post-cut Human3R frame and pseudo intrinsics. GT selected stress cases
and measured error but did not generate a correction.

H36M is no longer present in the current data directory. AIST remains under
`Training/asit` but has no compatible cross-camera held-out manifest, so neither
is reported as evaluated.

## Candidates

1. `raw`: unmodified Human3R local root.
2. `pointmap_z`: sample mask-restricted pointmap depth around the internal
   pelvis detection at three radii; preserve the predicted body-surface offset.
3. `mask_translation`: hold SMPL-X pose/shape fixed and optimize only `(x,y,z)`
   against the internal semantic mask extent and pelvis pixel.
4. `candidate_mean`: average valid pointmap and mask proposals.
5. `conservative_gate`: accept pointmap only when three radii agree within
   5 mm, IQR is at most 0.10 m, and the shift is at most 0.20 m and 8% of depth.
6. `persistent_mask_ratio_z`: use the pre-cut Human3R depth as a causal scale
   reference and adjust post depth by the pre/post internal-mask height ratio.
7. `oracle_candidate`: choose the best available candidate after reading GT;
   diagnostic only.
8. `oracle_gt_local_root`: set camera-local root to GT while leaving `B0`
   unchanged; diagnostic only.

The internal mask is a semantic union of all people, not an instance mask.
MultiHuman mask regions are split by predicted pelvis seeds. The one-person
stress evaluation removes this multi-person splitting issue.

## MultiHuman Results

Root values are mean errors in meters. P95 is world-root P95 after unchanged
`B0`. Harm is the fraction whose world-root error increases by more than 5 cm.

| Sequence | Method | World root | Local root | Local depth | Joint | Vertex | P95 | Harm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| three | raw | 0.315 | 0.323 | 0.316 | 0.340 | 0.322 | 0.613 | 0.0% |
| three | pointmap | 0.278 | 0.240 | 0.227 | 0.303 | 0.284 | 0.533 | 17.2% |
| three | frozen gate | 0.306 | 0.312 | 0.304 | 0.331 | 0.312 | 0.613 | 0.0% |
| dance | raw | 0.383 | 0.376 | 0.368 | 0.388 | 0.379 | 0.604 | 0.0% |
| dance | pointmap | 0.338 | 0.313 | 0.299 | 0.350 | 0.337 | 0.621 | 4.9% |
| dance | frozen gate | 0.380 | 0.367 | 0.358 | 0.387 | 0.377 | 0.616 | 2.5% |
| box | raw | 0.556 | 0.460 | 0.397 | 0.587 | 0.592 | 2.183 | 0.0% |
| box | pointmap | 0.525 | 0.411 | 0.343 | 0.559 | 0.564 | 2.296 | 17.9% |
| box | frozen gate | 0.545 | 0.445 | 0.380 | 0.577 | 0.582 | 2.183 | 2.6% |

The mask translation is not viable. Its mean world-root error is `1.013 m` on
`three`, `1.169 m` on `dance`, and `0.817 m` on `box`.

### Gate risk-coverage

The 5 mm multiscale threshold was selected on `three`: 26.2% person coverage,
9.8 mm aggregate gain, and no greater-than-5-cm harm. Without changing any
threshold, frozen evaluation produced:

| Sequence | Person coverage | Mean gain | Greater-than-5-cm harm |
|---|---:|---:|---:|
| three | 26.2% | 9.8 mm | 0.0% |
| dance | 36.9% | 2.4 mm | 2.5% |
| box | 28.2% | 10.7 mm | 2.6% |

A stricter 2 mm threshold removes greater-than-5-cm harm across these three
sequences but gives only about 11% coverage and 0.4-3.8 mm mean gain. On the
four-source stress set it slightly worsens the mean. This is not a useful
deployable operating point.

## Frozen B0 Limit

Even perfect camera-local root depth does not make the world root perfect when
`B0` camera translation remains wrong:

| Sequence | B0 camera translation | Raw world root | GT-local-root world error |
|---|---:|---:|---:|
| three | 0.256 | 0.315 | 0.259 |
| dance | 0.295 | 0.383 | 0.310 |
| box | 0.274 | 0.556 | 0.306 |

This separates two errors:

```text
camera-local human depth error
+ B0 camera translation error
-> final world-root error
```

They sometimes cancel. Therefore, reducing local depth can even increase final
world-root error while producing a more physically correct camera-human layout.
The human root must not be used to overwrite the already stronger B0 camera.

## Near/Far Stress Results

The 32 strongest selected cases produce the following combined local-root
results:

| Method | Mean | Median | P90 | P95 | Improved | Greater-than-5-cm harm |
|---|---:|---:|---:|---:|---:|---:|
| raw | 1.113 | 0.885 | 2.050 | 2.524 | 0.0% | 0.0% |
| pointmap | 1.127 | 0.883 | 2.195 | 2.380 | 40.6% | 34.4% |
| persistent mask ratio | 1.162 | 1.064 | 2.181 | 2.809 | 53.1% | 40.6% |
| mask translation | 1.257 | 1.098 | 2.452 | 2.624 | 31.2% | 62.5% |
| candidate mean | 1.150 | 0.882 | 2.400 | 2.431 | 43.8% | 46.9% |
| frozen gate | 1.114 | 0.885 | 2.050 | 2.524 | 9.4% | 6.2% |
| oracle candidate | 0.925 | 0.807 | 1.889 | 2.000 | 81.2% | 0.0% |

The source-level behavior is also inconsistent. Pointmap worsens AvatarReX
from `0.651` to `0.686 m` and THuman from `0.188` to `0.249 m`; it is almost
neutral on MVHuman100 (`2.174` to `2.171 m`) and improves MVHuman200 from
`1.438` to `1.404 m`.

The persistent mask ratio is not a safe latent scale state. It marginally
improves THuman (`0.188` to `0.184 m`) but worsens AvatarReX, MVHuman100, and
MVHuman200, with a 40.6% aggregate greater-than-5-cm harm rate.

## Interpretation

1. Human3R pointmap contains useful local depth information on ordinary
   MultiHuman cuts, but it shares much of the same monocular gauge error as the
   human head. In severe cases the human root can be wrong by 1-2 m while the
   pointmap proposes only a few centimeters.
2. Pointmap local variance and multiscale agreement measure spatial smoothness,
   not metric correctness. A confidently wrong shared depth gauge passes them.
3. The semantic mask cannot provide metric depth without reliable focal length.
   Union-mask assignment, pose-dependent silhouette size, truncation, and pseudo
   intrinsics create unstable translation proposals. Failure remains in the
   one-person stress set, so multi-person mask splitting is not the sole cause.
4. A pre/post apparent-size ratio cancels body-size error but not focal-length
   changes or pose-dependent projected height. It is insufficient as a
   persistent scale state.
5. The earlier shared-shot-scale conclusion remains unchanged: body and
   multi-human layout scale are mostly stable; the unresolved quantity is
   camera-local root depth plus B0 camera translation, not one global scale.

## Decision

Do not add pointmap, mask, or persistent apparent-size root correction to the
main V14 path. None provides both nontrivial coverage and a safe frozen tail.

The safest current route is:

```text
fresh Human3R local human geometry
-> unchanged learned B0
-> B0-guided identity
-> no deployable per-human root-depth override
```

If root depth must be improved without an external pretrained depth model, the
next technically defensible experiment is a small Human3R-internal focal/root
calibration head trained with existing multi-camera metric GT. It must predict
only camera-local human depth, leave `B0` camera unchanged, and be evaluated
capture-disjoint. A deterministic post-process using the currently available
pointmap/mask signals is not supported by these results.

## Reproducibility

```text
versions/v14/probe_v14_internal_root_depth.py
versions/v14/mine_v14_root_depth_stress.py
versions/v14/probe_v14_root_depth_stress.py

output/v14/internal_root_depth/
output/v14/root_depth_stress_mining/
output/v14/root_depth_stress_probe/
```
