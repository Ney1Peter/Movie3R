# Decoder / Correct-Token Experiment Plan

## Scope

The experiment changes neither V14 geometry nor Human3R reconstruction. It asks
only how V9 should represent and read correction evidence for the first frame
after a known cut.

## Testable hypotheses

### H1: final mean pooling removes role information

Semantic, alignment, and momentum tokens answer different questions. Averaging
them assumes they occupy one interchangeable feature space. Compare each token,
all pairwise combinations, mean, and concatenation at the same decoder depth.

### H2: correction evidence peaks before the final decoder block

Middle decoder layers may contain useful geometric discrepancy before the final
layer specializes for Human3R's original heads. Compare identical token
readouts at blocks 2, 5, 8, and 11.

### H3: a correction token should complement, not replace, the native pose token

Compare correction-only readout with correction plus pose, image-mean, and human
tokens at the same depth. This diagnoses whether the correction prompt lacks
raw pose/context or whether its own construction is weak.

## Candidate order

1. Formal V9 final correction-token mean.
2. Final semantic/alignment/momentum token separately.
3. Final role-preserving concatenation.
4. L5 and L8 versions of the same readouts.
5. L5/L8 correction plus native pose token.
6. Multi-depth fusion only if a single depth is insufficient.

No early DINO/CUT3R feature is reintroduced. No token predicts a human-specific
world transform.

## Stages and stop rules

### Stage 0: frozen information probe

- single-cut and ten-cut capacity controls;
- 24 training cuts per source, camera-pair-disjoint ten-cut evaluation;
- repeat with 96 cuts per source only for the top candidates;
- camera translation, rotation, composite, P90, and per-source results.

Promotion requires a candidate to beat final-mean residual readout by at least
10% on held-out composite and not depend on one source only.

### Stage 1: one-sequence trainable head

Train only a small bounded pose-latent residual head. Human3R, V9 prompt,
decoder, pose head, human head, and all LoRA weights remain frozen.

Promotion requires clear optimization below the frozen formal-V9 result on the
single cut, with finite residual norm and deterministic repeated inference.

### Stage 2: frozen ten-cut test

Train on separate four-source cuts and evaluate the unchanged ten cuts.

Promotion requires lower mean or materially lower P90 than formal V9, with no
new catastrophic cut and no degradation hidden by averaging sources.

### Stage 3: scale-up

Increase to 96 cuts per source, select regularization on training-only
camera-pair grouped CV, and freeze the resulting rule.

If no candidate survives a stage, stop. The valid conclusion is then that the
formal V9 prompt/head is not improved by a small decoder-depth change under the
available data.
