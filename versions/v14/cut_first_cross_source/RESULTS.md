# V14 Cut-First Cross-Source Correction Results

## Scope

This isolated experiment tests one question:

> Can the existing V9/V14 correction path generalize better when it receives
> cross-camera, cross-source supervision and is triggered only on the first
> frame after a camera cut?

No architecture, loss weight, frozen evaluation record, Human3R checkpoint,
V9 checkpoint, or existing V14 checkpoint was modified. All trained models
start independently from:

```text
checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth
```

The runtime contract is:

```text
pre-cut state (read-only)
-> shadow correction on first post-cut frame only
-> B0 = C_shadow @ inverse(C_raw_reset)
-> discard shadow human/scene/state
-> apply the camera-derived B0 to the raw-reset post-cut segment
```

The human correction head remains an auxiliary training signal for decoder
refinement. Shadow human outputs are evaluated diagnostically but are never
committed at runtime.

## Data Protocol

Each event contains three views:

```text
frames:      [t-1, t, t]
sequences:   [camera A, camera A, camera B]
shot_labels: [0, 0, 1]
```

Training uses four sources:

- AvatarReX
- THuman
- MVHuman100
- MVHuman200

The staged scales are:

| Stage | Events by source | Total events | Epochs |
|---|---:|---:|---:|
| cross10 | 3 / 2 / 3 / 2 | 10 | 40 |
| cross24 | 24 each | 96 | 12 |
| cross96 | 96 each | 384 | 6 |

`cross24` and `cross96` exclude every unordered camera pair present in both
the frozen ten-event suite and the frozen 180-event suite. Pair overlap is
zero for every source. `cross10` has zero overlap with frozen10 and one
AvatarReX camera-pair overlap with frozen180, but no exact event overlap.
Therefore frozen10 is the strict result for cross10; its frozen180 result is
auxiliary only.

No frozen record was used for training, early stopping, checkpoint selection,
or hyperparameter tuning. Every reported model is the final checkpoint from a
predeclared schedule.

## Training Behavior

| Stage | Final loss | AvatarReX | THuman | MVHuman100 | MVHuman200 |
|---|---:|---:|---:|---:|---:|
| cross10 | 0.0467 | 0.0593 | 0.0535 | 0.0430 | 0.0263 |
| cross24 | 0.2045 | 0.1037 | 0.1564 | 0.3866 | 0.1711 |
| cross96 | 0.9020 | 1.4232 | 0.3452 | 1.1987 | 0.6408 |

All runs completed without NaNs, OOMs, early stopping, or source-routing
errors. Larger stages are deliberately harder: each event is seen fewer times
and the source/camera diversity is much higher.

Training-set capacity evaluation:

| Stage | N | B0 composite | P90 | Catastrophic |
|---|---:|---:|---:|---:|
| cross10 | 10 | 0.1028 | 0.2018 | 0 / 10 |
| cross24 | 96 | 0.2989 | 0.4881 | 1 / 96 |
| cross96 | 384 | 0.6122 | 1.0406 | 29 / 384 |

This proves that the event-only correction path has substantial fitting
capacity. It also shows that cross96 is not saturated on its training set, so
its remaining error cannot be attributed only to held-out distribution shift.

## Frozen Ten-Event Evaluation

All rows below use the deployable `b0_runtime` path, not committed shadow
human outputs.

| Model | Camera T (m) | Camera R (deg) | Composite | P90 | P95 | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| Raw hard reset | 3.1015 | 133.947 | 5.7805 | 7.2919 | 7.7626 | 10 / 10 |
| Old one-Avatar checkpoint | 1.1980 | 64.863 | 2.4953 | 5.3101 | 5.4701 | 6 / 10 |
| cross10 | 1.1985 | 45.240 | 2.1033 | 3.0773 | 3.7982 | 8 / 10 |
| cross24 | 0.9714 | 51.670 | 2.0048 | 4.4742 | 4.6174 | 5 / 10 |
| cross96 | 1.0861 | 42.427 | **1.9346** | **3.8858** | 5.0526 | **4 / 10** |

The ten-event pilot established capacity but increased catastrophic failures.
Adding source/camera coverage recovered safety: cross24 and cross96 improve
mean, tail, and catastrophic count over the old checkpoint.

## Frozen 180-Event Evaluation

| Model | Camera T (m) | Camera R (deg) | Composite | P90 | P95 | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| Raw hard reset | 3.1006 | 120.838 | 5.5173 | 8.5929 | 9.2995 | 180 / 180 |
| Old one-Avatar checkpoint | 1.0984 | 59.455 | 2.2875 | 4.9074 | 5.6680 | 107 / 180 |
| cross10 (auxiliary) | 1.3093 | 59.280 | 2.4949 | 4.9273 | 5.7202 | 126 / 180 |
| cross24 | 1.0260 | 46.707 | 1.9602 | 4.2133 | 5.2791 | 97 / 180 |
| cross96 | **0.9073** | **41.302** | **1.7333** | **3.9670** | **4.7186** | **86 / 180** |

Relative to the old one-Avatar checkpoint, cross96 reduces:

- translation error by 17.4%;
- rotation error by 30.5%;
- composite error by 24.2%;
- P90 composite by 19.2%;
- catastrophic count from 107 to 86.

The scale trend is consistent from cross24 to cross96, so the improvement is
not explained by memorizing the ten-event suite.

### Cross96 by Source

| Source | N | Camera T (m) | Camera R (deg) | Composite | P90 | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| AvatarReX | 48 | 1.0462 | 35.481 | 1.7558 | 4.0350 | 19 / 48 |
| THuman | 48 | 0.2935 | 1.985 | 0.3332 | 0.4626 | 2 / 48 |
| MVHuman100 | 48 | 1.0751 | 63.249 | 2.3401 | 4.1397 | 36 / 48 |
| MVHuman200 | 36 | 1.3169 | 72.222 | 2.7613 | 5.0122 | 29 / 36 |

The method is strong on THuman and substantially better on AvatarReX, but
wide-view MVHuman camera pairs still dominate the failure tail. More training
data helps on average but does not make the implicit correction intrinsically
safe for arbitrary camera pairs.

## Streaming and State Purity

Reset-only parity was audited after every stage against
`src/human3r_896L.pth`. Camera, pointmap, confidence, and SMPL fields have:

```text
max_abs = 0.0
mean_abs = 0.0
all_shapes_match = true
```

Therefore event-only correction does not alter ordinary no-cut frames. The
post-cut segment can continue with clean Human3R hard-reset state plus one
cached B0; shadow state is never committed.

The B0 camera implementation also matches the shadow camera numerically:
maximum 4x4 matrix disagreement is below `2.4e-7` in all capacity audits.

## Conclusions

### What succeeded

1. The V9 correct-token/decoder/two-head path can be trained as a cut-first,
   non-committing shadow transaction without changing normal-frame outputs.
2. Cross-camera and cross-source supervision produces real held-out gains.
3. Increasing training coverage from 24 to 96 events per source improves
   frozen10 and frozen180 mean, tail, and catastrophic count.
4. Correcting only the first post-cut frame is sufficient to estimate a B0
   that can be applied to the independent raw-reset segment; every later frame
   does not need recurrent correction.

### What did not succeed

1. The route is not safe enough to become the final deployable boundary by
   itself: cross96 still has 86 catastrophic cases out of 180.
2. MVHuman wide-view pairs remain highly unstable, especially in rotation.
3. Larger supervision reduces but does not eliminate the wrong-gauge tail.
4. Shadow human reconstruction improves diagnostically, but the deployable
   camera-derived B0 can still increase raw-reset human-head error because it
   preserves raw human/scene geometry by design.

### Promotion Decision

Retain this route as the learned **coarse B0 proposal** and as evidence for the
causal state/gauge decomposition. Do not promote it as a standalone final
alignment method and do not commit its shadow human, scene, or recurrent state.

The next main-method stage should use the learned B0 only to move the raw-reset
shot into the correct basin, then apply a conservative explicit residual or
fallback that is bounded around B0. Further unstructured increases in
correct-token complexity are not justified by these results.

## Artifacts

Checkpoints and logs:

```text
output/v14_cut_first_cross_source/v14_cut_first_cross_source_10_e40/
output/v14_cut_first_cross_source/v14_cut_first_cross_source_24ps_e12/
output/v14_cut_first_cross_source/v14_cut_first_cross_source_96ps_e6/
```

Frozen and capacity reports:

```text
output/v14_cut_first_cross_source/eval_cross10_frozen10/
output/v14_cut_first_cross_source/eval_cross10_180/
output/v14_cut_first_cross_source/eval_cross24_frozen10/
output/v14_cut_first_cross_source/eval_cross24_180/
output/v14_cut_first_cross_source/eval_cross96_frozen10/
output/v14_cut_first_cross_source/eval_cross96_180/
output/v14_cut_first_cross_source/eval_cross10_train10_capacity/
output/v14_cut_first_cross_source/eval_cross24_train24_capacity/
output/v14_cut_first_cross_source/eval_cross96_train96_capacity/
```

Parity reports:

```text
output/v14_cut_first_cross_source/reset_only_parity_cross10/
output/v14_cut_first_cross_source/reset_only_parity_cross24/
output/v14_cut_first_cross_source/reset_only_parity_cross96/
```
