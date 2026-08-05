# V9 Decoder / Correct-Token Probe

This is an isolated continuation of the V9 multi-layer information audit. It
does not modify `src/`, formal V9/V14 configs, or any checkpoint.

The probe freezes the formal V9 checkpoint and records compact token vectors at
the first post-cut frame:

- pre-decoder semantic, alignment, and momentum correction tokens;
- refined correction tokens after decoder blocks 2, 5, 8, and 11;
- pose token, mean image token, and mean human token at the same depths.

Each vector is converted into a causal pre/post relation descriptor. No patch
feature map is saved. A residual probe predicts one shared camera correction
relative to V9's raw pose-head input; it never predicts separate human and
camera transforms.

The staged stop rule is:

1. frozen token/depth ranking;
2. one-sequence trainable mid-depth head only if a candidate beats final-mean;
3. frozen ten-cut test;
4. four-source scale-up only if the ten-cut result remains positive.

Artifacts are written below `output/v9_decoder_correct_token_probe/`.

The completed experiment found that the final decoder pose token is more useful
than the dedicated correction-token mean. A 175k role-preserving pose-relation
head improved frozen mean and P90, but introduced one new catastrophic
MVHuman200 error. It is therefore not promoted into V9/V14.

See `FINAL_REPORT.md` for the complete protocol, metrics, safety audits, and
final decision.
