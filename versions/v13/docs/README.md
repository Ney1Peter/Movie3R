# V13 Documentation

Movie3R-Multi V13.0 is the GT-ID multi-human shared-Boundary research release.
Read `V20_PHASE1_GT_ID_MULTIHUMAN_CONSENSUS_V2.md` for the corrected current
result. The report without `_V2` is retained only to document the earlier
identity-association error.

The V20 filenames are legacy experiment identifiers. The formal release name
is V13, and it is not yet a deployable cross-shot Re-ID system.

`V13_PHASE2_MULTIHUMAN_FUSION_OPTIMIZATION.md` records the fusion-only analysis
and the `dance` cross-sequence pilot. Its current decision is to retain naive
mean rather than any tested soft uncertainty rule.

`V13_PHASE3_CROSS_SHOT_IDENTITY_BRIDGE.md` records the completed automatic WHO
audit. The GT-ID geometry milestone remains valid, but the native token/local
pose bridge fails the gain-retention and catastrophic-swap gates, so automatic
multi-human alignment remains disabled by default.

`V13_PHASE4_PRECISION_FIRST_IDENTITY.md` records the frozen DINOv2 appearance
and precision-first gate study. It reaches zero wrong accepted matches on the
MultiHuman sequences, but only at very low multi-human coverage; EgoHumans
coverage collapses to zero and the identity-free fallback is catastrophic.
Phase 4A therefore fails the deployable gate and Phase 4B adapter training is
not started.

`V13_PHASE5_CAUSAL_IDENTITY_STATE.md` records the causal multi-cut state study.
Running-mean persistent state passes Stage 0/1 and raises safe identity
coverage, but bounded joint WHO-WHERE scoring fails to improve the identity-only
zero-wrong operating point. Automatic multi-human commit therefore remains off.
