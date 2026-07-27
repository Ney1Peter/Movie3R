# V14.1 Single-Event Contract Diagnosis and Corrected Upper Bound

Date: 2026-07-27

## 1. Status

The first V14.1 one-event and 10-event reports are invalid as event-only results.
They were produced with a training/demo contract mismatch. They must not be used
as evidence for V14.1 capacity or generalization.

The corrected status is:

```text
single-event routing and preprocessing fixed
single-event upper bound reproduced in the streaming demo path
10-event result withdrawn and not rerun yet
large-scale training blocked on visual inspection
```

## 2. Sample and Intended Contract

Sample:

```text
lbn1/22053926 frame 1191
lbn1/22053926 frame 1192
lbn1/22010716 frame 1192
view angle = 132.853 degrees
shot_labels = [0, 0, 1]
```

The intended V14.1 contract is:

```text
views 0 and 1:
    ordinary context
    no correction token
    no event-only head LoRA
    no direct supervision

view 2:
    explicit cut event
    insert correction token(s)
    enable correction heads
    apply event supervision
```

## 3. Root Causes

### 3.1 `shot_label` was dropped before model forward

`_make_v8_image_only_model_batch()` correctly removed GT camera, SMPL and depth
fields, but it also removed the deployable `shot_label` input. The training model
therefore did not receive `[0, 0, 1]`.

Observed consequence:

```text
training:
    correction was active on all three frames

demo.py:
    correction was active only on frame 2
```

The loss still supervised only frame 2, so a low training validation loss did not
prove that the event-only inference graph worked. This was the primary bug.

The fix preserves `shot_label` while continuing to strip every GT-only field.
A regression test verifies both requirements.

### 3.2 Training and demo used different RGB preprocessing

The old training config used `resize_only_16`; `demo.py` uses Human3R long-edge
resize followed by center crop. Both happened to produce `512x368` tensors, but
the actual normalized RGB tensors differed substantially:

```text
mean absolute difference approximately 0.048
maximum absolute difference approximately 1.45-1.69
```

The dataset now uses `resize_mode='human3r_demo'`. The resulting training RGB
tensors are exactly equal to the demo input tensors.

### 3.3 Patch embedding class differed across train and checkpoint reload

`load_model()` replaces `ManyAR_PatchEmbed` with `PatchEmbedDust3R` for video
inference. The first corrected run still trained with the former and reloaded with
the latter. The numerical effect was smaller than the two bugs above, but an exact
upper-bound experiment cannot keep this discrepancy.

V14.1 configs now train with `PatchEmbedDust3R`, matching checkpoint reload and
`demo.py`.

### 3.4 The simplified V14.1 architecture is not V9-equivalent

The simplified branch changes several V9 choices at once:

| Component | V9-parity | Simplified V14.1 |
|---|---|---|
| Correct tokens | semantic + alignment + momentum | semantic + alignment |
| Reliability feature | enabled | disabled |
| Learned correction gates | enabled | forced on at event |
| Human latent gate | enabled | forced on at event |
| Head LoRA on context | enabled | disabled |
| Event supervision | event only | event only |

This is an architecture ablation, not merely a different event schedule. These
changes must be removed one at a time after the V9-parity reference is frozen.

## 4. Forward-Path Audit

For both corrected checkpoints:

```text
view 0: v9_pre_decoder_append = 0
view 1: v9_pre_decoder_append = 0
view 2: v9_pre_decoder_append = 1
```

Training-style `model.forward()` and demo-style
`forward_recurrent_lighter()` agree to numerical precision:

| Output | Maximum mean absolute difference |
|---|---:|
| camera pose | about 1.9e-4 |
| SMPL translation | about 4.0e-5 |
| cross-view pointmap | about 3.5e-4 |

The training and streaming implementations are therefore not the remaining
explanation for a large visual difference.

## 5. Corrected Single-Event Results

The table below uses the reloaded checkpoint and the same streaming input/routing
as `demo.py`, not an in-memory training-only forward.

| Method | Camera translation | Camera rotation | Human translation |
|---|---:|---:|---:|
| Formal V9 checkpoint | 0.246 m | 7.13 deg | 0.0102 m |
| V14.1 simplified, corrected contract | 0.082 m | 0.13 deg | 0.0126 m |
| V14.1 V9-parity, corrected contract | 0.048 m | 0.25 deg | 0.0050 m |

Interpretation:

1. The corrected event-only branch can overfit the hard 132.9-degree cut.
2. V14.1 is no longer numerically worse than V9 on this sample.
3. The V9-parity branch gives the best camera translation and human translation.
4. The simplified branch also fits, but its translation remains weaker; the
   simplifications cannot yet be accepted as free changes.
5. Pointmap/scene geometry has no GT depth supervision in this pilot, so these
   camera/human metrics do not replace visual inspection.

## 6. Withdrawn Results

The former claims below are withdrawn:

```text
old one-event result:
    0.0891 m / 0.0969 deg / 0.0086 m

old 10-event weighted result:
    0.1025 m / 0.1659 deg / 0.0066 m
```

Both runs used the dropped-`shot_label` training path. They also used
`resize_only_16`, while visualization used the Human3R demo preprocessing. The old
checkpoints and metrics are debugging artifacts only.

## 7. Current Visualization

All viewers use the same three images and `cut_indices=2`:

| Port | Model |
|---:|---|
| 8091 | Original Human3R |
| 8092 | Formal V9 |
| 8093 | Corrected simplified V14.1 upper bound |
| 8094 | Corrected V9-parity V14.1 upper bound |

The two corrected pilot checkpoints are currently stored under `/dev/shm`:

```text
/dev/shm/movie3r_v14_1/v14_1_cut_event_single_simplified_exact_runtime/checkpoint-best.pth
/dev/shm/movie3r_v14_1/v14_1_cut_event_single_v9_parity_exact_runtime/checkpoint-best.pth
```

They are volatile diagnostic artifacts, not release checkpoints.

## 8. Decision and Next Step

The single-event training graph is now valid. Training must remain at one sample
until the 8093/8094 inspection confirms that camera, pointmap and SMPL-X are
coherent in 3D.

If V9-parity is visually coherent but the simplified branch is not, use V9-parity
as the reference and ablate in this order:

```text
1. remove momentum only
2. disable reliability only
3. disable learned pose/human gates only
4. make pose/human head LoRA event-only only
```

Only one change is allowed per run. The 10-event pilot should be rerun only after
one architecture passes both the numerical and subjective single-event checks.
