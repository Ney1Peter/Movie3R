# V14 Learned B0 Before Multi-Human Identity Matching

## Question

This probe tests whether the learned V9/V14 coarse Boundary should run before
cross-shot multi-human identity matching:

```text
direct: pre/post Human3R geometry -> identity matching
B0:     learned coarse shared Boundary -> transform post geometry -> same matcher
```

The two paths use identical anonymous Human3R root, torso and root-centered
joint costs, identical per-case normalization, and identical Hungarian
assignment. The only changed variable is the shared post-cut transform.

GT identity is used only to select the controlled evaluation subset and score
the resulting assignment. It is not an input to the cost matrix, learned B0,
or Hungarian solver.

## Protocol

Dataset and cuts:

```text
MultiHuman Real-World-Capture / three
synchronized camera cuts (k=0)
63 candidate cuts
41 controlled cuts with the same >=2 detected identities on both sides
22 entry/exit or detector-set-change cuts excluded
```

The controlled subset isolates coordinate alignment from dustbin design. The
excluded cuts must be evaluated separately after a dustbin policy is frozen.

Learned Boundary:

```text
four causal pre-cut frames + first post-cut frame -> shadow V14 inference
first post-cut frame with fresh Human3R state    -> raw inference
B0 = C_shadow_post @ inverse(C_raw_post)
```

No future post-cut frame is read and the shadow state is discarded. The fresh
post-cut Human3R output is in exactly the same gauge as the Phase-2 cache:

```text
raw current vs Phase-2 cache: 0.0 m / 0.0 deg
```

The checkpoint was trained for 80 epochs on one AvatarReX `lbn1_1192` event,
not on MultiHuman `three`. It starts from the V9 mixed checkpoint, so this is a
cross-dataset feasibility result for the V14 correction, not a claim of fully
held-out V9 pretraining provenance.

## Results

### All 41 controlled cuts

| Matcher | Direct accuracy | B0 accuracy | Direct all-correct | B0 all-correct |
|---|---:|---:|---:|---:|
| root | 50.8% | **100.0%** | 41.5% | **100.0%** |
| torso | 41.8% | **100.0%** | 41.5% | **100.0%** |
| root + torso | 46.7% | **100.0%** | 46.3% | **100.0%** |
| root + torso + joints | 44.3% | **100.0%** | 43.9% | **100.0%** |

For `root + torso`, learned B0 corrected 22 cut-level assignment failures and
introduced zero regressions. Its median best-vs-second assignment margin rose
from `0.973` to `3.184`.

### View-span ablation with root + torso

| Camera span | Cases | Direct accuracy | B0 accuracy | Direct all-correct | B0 all-correct |
|---|---:|---:|---:|---:|---:|
| `<60 deg` | 17 | 100.0% | 100.0% | 100.0% | 100.0% |
| `60-120 deg` | 5 | 35.7% | **100.0%** | 40.0% | **100.0%** |
| `>=120 deg` | 19 | 1.8% | **100.0%** | 0.0% | **100.0%** |

The learned B0 camera error over all 41 cuts is:

```text
translation: mean 0.256 m, median 0.210 m, P90 0.443 m
rotation:    mean 4.135 deg, median 4.249 deg, P90 5.919 deg
```

In the `>=120 deg` group it remains accurate enough for identity geometry:

```text
translation median: 0.340 m
rotation median:    4.923 deg
```

Camera-pose continuity without learned correction behaves like direct matching:

```text
root + torso accuracy:        46.7%
cut-level all-correct rate:   46.3%
```

Therefore the gain is caused by the learned coarse Boundary, not by merely
combining the pre/post raw camera predictions.

One representative `173.9 deg` three-person cut shows the mechanism directly:

```text
case: three_t1100_c1_c2_k0

direct:
    correct identities:       0/3
    GT permutation rank:      6/6
    best-vs-second margin:    0.290
    error type:               three-person cyclic swap

learned B0:
    B0 camera error:          0.175 m / 5.42 deg
    correct identities:       3/3
    GT permutation rank:      1/6
    best-vs-second margin:    2.569
```

The coarse Boundary therefore does not need final-alignment accuracy to help
identity. It only needs to remove enough of the shared view jump for the group
layout and torso directions to become discriminative.

## Conclusion

The proposed ordering is strongly supported on the synchronized controlled
probe:

```text
learned coarse B0
-> put all post-cut humans into the approximate pre-cut gauge
-> match identities using shared geometry and identity cues
-> run frozen multi-human Boundary refinement
```

The reason is direct: before B0, root positions and torso orientations from the
two shots describe different coordinate systems. At a roughly 180-degree cut,
the wrong permutation can look geometrically cheaper. B0 removes most of the
shared camera rotation and translation first, after which the correct group
layout becomes separable.

B0 does not solve identity by itself. It still needs appearance/shape cues for
similar people and temporal offsets, and it does not yet handle entry, exit, or
missed detections. The next validation must freeze this ordering and test:

```text
k = 1, 2, 4, 8
MultiHuman dance and box
variable visibility with dustbin
B0 + appearance/shape/pose matching
```

## Artifacts

```text
versions/v14/probe_b0_identity_matching.py
tests/test_v14_b0_identity_matching.py
output/v14/b0_identity_matching/v14_b0_identity_matching.json
output/v14/b0_identity_matching/v14_b0_identity_matching.md
```
