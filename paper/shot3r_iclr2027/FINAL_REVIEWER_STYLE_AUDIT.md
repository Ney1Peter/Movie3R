# Final Reviewer-Style Audit of v030

## What is now convincing

- The paper asks a narrow, identifiable question: how to preserve a
  camera--human world gauge across same-scene viewpoint cuts without
  propagating view-specific recurrent state.
- The method description matches the retained implementation and clearly
  separates read-only boundary evidence from clean future recurrence.
- The primary evidence is a strict same-backbone, same-RGB, same-evaluator
  comparison on three multi-person protocols.
- The wide-view motivation is tested by a cluster-aware interaction rather
  than inferred from selected qualitative examples.
- Coverage, association denominators, oracle inputs, incompatible public
  interfaces, and negative results are visible.

## What a critical reviewer may still ask

- Whether independent training ablations would justify each correction-token
  design choice.
- Whether the method transfers to another recurrent backbone.
- Whether a detector calibrated on broader appearance domains avoids the
  MVHuman early-trigger failure.
- Whether gains persist under a larger natural repeated-cut protocol and
  temporally asynchronous actors.
- Whether end-to-end runtime, including event detection, is practical.

## Final assessment

The v030 paper is internally consistent and submission-ready as a scoped
Human3R-derived camera--human bridging study. It should not be marketed as a
general video editor, dense scene reconstructor, all-metric winner, or
backbone-independent solution. The remaining questions are legitimate future
experiments, not gaps that can be repaired by stronger prose.
