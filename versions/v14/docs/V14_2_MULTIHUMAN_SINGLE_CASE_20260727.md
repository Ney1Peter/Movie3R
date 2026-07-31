# V14.2 Multi-Human Single-Case Probe

## Protocol

The first V14 multi-human segment test uses the previously audited strict-GT-ID
MultiHuman case:

```text
sequence: three
pre:  camera 0, frames 897-900
post: camera 3, frames 900-905
cut:  synchronized frame 900
humans: person0, person1, person2
```

All ten pre/post frames use full `2048x2048` images resized to `512x512`
without person crops. Human3R detects all three people in every frame. GT
identity is used only to associate detections for this controlled geometry
probe.

The causal execution is:

```text
pre-cut history + first post-cut frame -> V14 shadow B0
fresh-reset post-cut stream           -> committed local reconstruction
each GT-ID human                       -> B0 + V16/V12 candidate (R_i, t_i)
three candidates                       -> equal SO(3) mean + mean raw t_i
one fixed shared Boundary              -> all post cameras, pointmaps and humans
```

The shadow state is discarded and no future post-cut frame contributes to the
Boundary.

## Results

| Method | Camera T (m) | Camera R (deg) | Composite | Human root (m) | Cut root jump (m) | Cloud NN median (m) |
|---|---:|---:|---:|---:|---:|---:|
| Raw reset | 1.579 | 54.26 | 2.665 | 0.625 | 0.649 | 0.419 |
| V14 B0 | **0.178** | 4.14 | 0.261 | 0.537 | 0.544 | 1.620 |
| B0 + mean V16 rotation | **0.178** | 3.38 | **0.246** | **0.521** | 0.424 | 1.404 |
| Highest-quality single candidate | 0.635 | 6.86 | 0.772 | 0.599 | 0.145 | 1.261 |
| GT-ID three-human uniform consensus | 0.387 | 3.38 | 0.455 | 0.580 | **0.144** | **1.082** |
| GT-camera-only Boundary | 0.000 | 0.05 | 0.001 | 0.353 | 0.396 | 1.462 |
| GT rotation + multi-anchor translation | 0.371 | 0.05 | 0.372 | 0.540 | 0.150 | 1.113 |

The candidate dispersion is substantial:

```text
mean pairwise rotation dispersion:    11.21 deg
mean pairwise translation dispersion: 0.48 m
```

`person2` contributes a `-13.43 deg` torso residual and has much lower observed
quality than the other two people. The frozen uniform rule still keeps it, as
required by the V13 geometry protocol.

## Interpretation

This case establishes three limited facts:

1. The V14.1 checkpoint and Human3R multi-human head run correctly on a full
   three-person frame and retain all three people through a wide camera cut.
2. Learned V14 B0 already gives the best camera translation. Mean V16 rotation
   improves rotation slightly, but replacing B0 translation with human-root
   candidates worsens the GT camera metric.
3. Three-human consensus substantially improves predicted cross-cut human and
   pointcloud continuity. It does not improve all GT camera/human metrics on
   this case.

Therefore V14 B0 and V13 multi-human translation currently optimize different
objectives. Multi-human root continuity should remain an explicit diagnostic or
guarded refinement until more cases determine whether it improves visible
alignment without systematically moving the camera away from GT.

## Artifacts

Runner:

```text
versions/v14/run_v14_2_multihuman_sequence.py
```

Report and viewer payloads:

```text
/dev/shm/movie3r_v14_2/multihuman_three_t0900_c0_c3
```
