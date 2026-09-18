# Remaining Risks after v030

## Scientific risks

1. **No independent token retraining.** The available masks reuse one
   checkpoint and cannot establish that every correction token is necessary.
2. **Single reconstruction backbone.** Evidence supports a Human3R-derived
   system, not backbone-agnostic generality.
3. **Detector domain shift.** On MVHuman, 40/50 first triggers fire early;
   future work needs development-only calibration or retraining, not test-set
   threshold tuning.
4. **Repeated-cut scale.** The retained real-image control is small, and its
   four-stream camera trajectory worsens.
5. **Shared-translation limits.** Harmony4D W-MPJPE worsens and asynchronous
   motion/offset sensitivity is not isolated.
6. **External comparison support.** TRACE and PromptHMR have limited finite
   geometric support; HumanMM and Multi-THuMBS lack a compatible executable
   common evaluator.
7. **Runtime boundary.** The reported measurement covers one clip and excludes
   the detector, preprocessing, and checkpoint initialization.
8. **Training metadata.** Exact hardware/duration and multi-seed variance of
   the final training run could not be recovered reliably and are not guessed.

## Submission risks to recheck later

- Re-run the official ICLR style checker when the final 2027 submission bundle
  is published.
- Replace the anonymous author block only after review.
- Revalidate data/checkpoint redistribution terms before any public artifact
  release.
