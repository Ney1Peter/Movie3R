# V9 State Update Ablation Plan

## Motivation

The current V9 model already has a useful output-side correction path:

- semantic / alignment / momentum correct tokens enter the decoder
- pose correct head predicts a pose-token residual and gate
- human latent correct head predicts a human-token residual
- pose / human head LoRA turns the corrected tokens into camera and human outputs

After many token and loss ablations, the correction branch is useful, but it does
not fully solve AABB storyboard misalignment. A likely reason is that the error is
partly upstream: CUT3R / Human3R keeps a recurrent state. When frame 3 comes from
a new shot or a new view, the model still writes it into the same state as frames
1-2. Once the state is contaminated, the output heads can only compensate after
the fact.

So the next direction is to control how the recurrent state is written, while
keeping the existing output correction branch.

## What TTT3R Shows

TTT3R already changes the state update rule by making state writes more
conservative according to cross-attention confidence. In our short AABB test it
slightly improved the result, but did not change the main failure mode.

This suggests two points:

- state update matters
- confidence-only state gating is not specific enough for storyboard alignment

Our task is not just long-sequence stability. The immediate target is short
AABB alignment: frames 3-4 should be interpreted in the same world as frames 1-2.

## Experiment 1: Freeze After A

For a 4-frame AABB input:

- frame 0: A, update state
- frame 1: A, update state
- frame 2: B, forward normally, but do not write state
- frame 3: B, forward normally, but do not write state

This keeps the recurrent state as a reference built from the A segment. B frames
can still read the reference state, run the decoder, and use the V9 correction
heads, but they cannot overwrite the state.

This is an inference-only ablation. It does not require retraining because it only
changes the update mask:

```text
state = new_state * update_mask + old_state * (1 - update_mask)
```

With `freeze_state_after=2`, `update_mask` becomes 0 for frames with original
index >= 2.

The same update mask also freezes:

- global recurrent state
- pose memory
- V8 correct-token history

This is intentional for the first probe. It tests whether preserving the A-side
reference helps B-side alignment.

## Component Ablation

The first freeze-after-A probe freezes three update paths together. To identify
where the short AABB drift mainly comes from, split it into three independent
inference-only switches:

```text
--freeze_state_feat_after 2
```

Freeze only the global recurrent `state_feat`. Frames 3-4 still update pose
memory and V9 correction history, but they cannot rewrite the global scene/state
tokens.

```text
--freeze_pose_memory_after 2
```

Freeze only the pose retriever memory `mem`. Frames 3-4 still update the global
state and V9 history, but the pose-token memory used for later camera prediction
stays anchored to frames 1-2.

```text
--freeze_v9_history_after 2
```

Freeze only the V9 correction history. Frames 3-4 still update the original
Human3R/CUT3R state and pose memory, but the V9 momentum/history tokens stop
absorbing B-side corrections.

```text
--freeze_state_after 2
```

Freeze all three paths together. This is the original oracle freeze-after-A
setting.

These component ablations do not retrain the model. They only change which
internal memory is written after the A segment.

## Expected Outcomes

If freeze-after-A improves AABB alignment, then the B-side state write is hurting
short-sequence alignment. That supports a stronger state-control design.

If it does not improve alignment, then merely preventing B from writing state is
not enough. The model may need a second local B-state or a learned state-write
policy.

## Follow-Up Designs

### Reference State + Local State

Keep two states:

- reference state: stable A-side world anchor
- local state: current shot / current view consistency

B frames read both states. The model can preserve A as the alignment anchor while
still modeling B-frame local consistency.

### Learned State Write Gate

Predict a per-frame or per-token write gate from:

- current image tokens
- previous state
- pose token
- human tokens
- V9 correct tokens / gate

Then update:

```text
state = gate * new_state + (1 - gate) * old_state
```

This removes the hand-coded assumption that frames 2-3 are always the jump. It is
closer to the final streaming use case.

### Shot-Aware / Human-Aware State Update

Use the human as a stability cue when the task assumes a static person:

- strong human consistency -> preserve reference state
- inconsistent human/camera relation -> reduce write strength
- high current-frame reliability -> allow controlled state update

This is more targeted than TTT3R because the gate is designed for our storyboard
alignment problem, not generic long-sequence stability.
