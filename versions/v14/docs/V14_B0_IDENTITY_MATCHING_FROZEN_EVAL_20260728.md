# V14 Learned B0 Identity Matching: Dance/Box Frozen Evaluation

## Protocol

The matching protocol and V14.1 checkpoint were frozen from the MultiHuman
`three` development experiment. No threshold, feature weight, model parameter,
or normalization rule was changed after reading `dance` or `box` results.

```text
sequences:       MultiHuman dance and box
timestamps:      6 per sequence
camera pairs:    0->1, 0->3, 1->4
offsets:         k = 0, 1, 2, 4, 8
candidate cuts:  90 per sequence
pre-cut history: 4 causal frames
post input:      first post-cut frame at t+k
```

Only cuts with the same two GT identities detected on both sides are included
in the controlled matching statistic:

```text
dance: 61/90 eligible, 29 variable-visibility cuts excluded
box:   78/90 eligible, 12 variable-visibility cuts excluded
```

This isolates the effect of learned B0 from a still-unfrozen dustbin policy.
GT identity is used for benchmark selection and scoring only. The B0 model,
cost matrix, and Hungarian assignment do not read GT identity or GT camera.

## Main Results

The primary diagnostic uses the unchanged `root + torso` matcher.

| Sequence | Eligible cuts | Direct accuracy/all-correct | B0 accuracy/all-correct | Fixed cuts | Regressed cuts |
|---|---:|---:|---:|---:|---:|
| dance | 61 | 65.6% | **100.0%** | 21 | 0 |
| box | 78 | 65.4% | **98.7%** | 26 | 0 |

Because these are two-person cuts, assignment accuracy and cut-level
all-correct rate are numerically identical: an incorrect permutation swaps
both people.

### Dance by temporal offset

| Offset | Eligible | Direct all-correct | B0 all-correct | B0 T median/P90 | B0 R median/P90 |
|---|---:|---:|---:|---:|---:|
| k=0 | 13 | 61.5% | **100.0%** | 0.261 / 0.500 m | 3.66 / 4.13 deg |
| k=1 | 12 | 66.7% | **100.0%** | 0.246 / 0.481 m | 3.65 / 4.03 deg |
| k=2 | 12 | 66.7% | **100.0%** | 0.261 / 0.486 m | 3.62 / 3.88 deg |
| k=4 | 12 | 66.7% | **100.0%** | 0.255 / 0.492 m | 3.56 / 4.14 deg |
| k=8 | 12 | 66.7% | **100.0%** | 0.273 / 0.510 m | 3.62 / 4.42 deg |

### Box by temporal offset

| Offset | Eligible | Direct all-correct | B0 all-correct | B0 T median/P90 | B0 R median/P90 |
|---|---:|---:|---:|---:|---:|
| k=0 | 16 | 68.8% | **100.0%** | 0.177 / 0.487 m | 3.75 / 5.47 deg |
| k=1 | 15 | 66.7% | **100.0%** | 0.203 / 0.467 m | 3.82 / 5.31 deg |
| k=2 | 15 | 66.7% | **100.0%** | 0.191 / 0.468 m | 3.79 / 5.39 deg |
| k=4 | 15 | 66.7% | **100.0%** | 0.190 / 0.446 m | 3.75 / 5.60 deg |
| k=8 | 17 | 58.8% | **94.1%** | 0.189 / 0.449 m | 3.79 / 5.57 deg |

For the `>=120 deg` view-span group, both sequences have:

```text
direct all-correct: 0.0%
B0 all-correct:     100.0%
```

The best-vs-second assignment margin also increases after B0 for every offset.
For example, the dance `k=8` median margin rises from `2.929` to `4.818`, and
the box `k=8` median rises from `3.332` to `5.632`.

## The One Remaining Failure

```text
case:       box_t0630_c0_c3_k8
view span:  54.2 deg
direct:     0/2
learned B0: 0/2
GT camera:  0/2
B0 error:   0.158 m / 3.44 deg
```

The learned B0 is accurate in this case. Even exact GT-camera gauge alignment
leaves the predicted roots, torso directions, and root-centered joints closer
to the opposite identity after eight frames. The two-person permutation is
therefore a geometry/identity ambiguity caused by motion or local human
reconstruction, not a coarse camera-alignment failure.

This failure gives the correct module boundary:

```text
learned B0 removes the shared shot gauge jump
appearance/shape answers WHO when geometry changes or crosses
geometry then verifies layout and estimates the shared Boundary
```

## Conclusion

The learned B0-before-ID ordering generalizes from the `three` development
sequence to both frozen two-person sequences and remains effective through
`k=8`. It should be retained as the first stage of cross-shot identity
association.

B0 is not a complete Re-ID method. The next controlled experiment should add
the already available frozen appearance and beta cues after B0, target the
single `box k=8` failure, and then evaluate the 41 excluded variable-visibility
cuts with a frozen dustbin policy.

## Artifacts

```text
output/v13/dance_phase2/v13_gtid_offsets_0_1_2_4_8.json
output/v13/box_phase3/v13_gtid_offsets_0_1_2_4_8.json
output/v14/b0_identity_matching_extended/dance/v14_b0_identity_matching.json
output/v14/b0_identity_matching_extended/box/v14_b0_identity_matching.json
versions/v14/probe_b0_identity_matching.py
```
