# V10 Static Alignment Probe Results

Date: 2026-07-08

## Goal

This probe tests whether explicit body orientation cues help the V10 learnable
streaming alignment route.

The current V10 assumption in this test is:

- Oracle boundary is known.
- Original Human3R reconstructs each local segment.
- A small learnable alignment MLP predicts one global transform for the new
  segment.
- The human is near-static, so post-boundary frames should align back to the
  historical global gauge.

## Data

Small 4-source AABB probe:

- AvatarReX: 2 clips
- THuman: 2 clips
- MVHuman100: 2 clips
- MVHuman200: 2 clips

Total: 8 AABB clips. Each run trains for 2500 steps.

## Variants

| Variant | Body-frame input | Body-frame/vector loss | Meaning |
| --- | --- | --- | --- |
| `ablation_base` | No | No | Only the original joint/camera/prior alignment losses. |
| `ablation_body_loss` | No | Yes | The model is explicitly supervised to align body orientation and body vectors. |
| `ablation_body_input` | Yes | No | The model receives body-frame features, but is not directly supervised on them. |
| `ablation_body_input_loss` | Yes | Yes | Body-frame features and explicit body supervision are both enabled. |

## Overall Metrics

Lower is better for all aligned metrics.

| Variant | Cam Rot ↓ | Cam Trans ↓ | Human ↓ | Amean-B0 ↓ | Amean-B1 ↓ | Body Frame ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw local reset | 118.13 deg | 3.769 m | 0.323 m | 0.296 m | 0.303 m | 114.05 deg |
| `ablation_base` | 4.16 deg | 0.810 m | 0.191 m | 0.160 m | 0.183 m | 10.74 deg |
| `ablation_body_loss` | 4.92 deg | 0.819 m | 0.180 m | 0.151 m | 0.174 m | 3.82 deg |
| `ablation_body_input` | 4.31 deg | 0.828 m | 0.179 m | 0.149 m | 0.169 m | 10.85 deg |
| `ablation_body_input_loss` | 4.92 deg | 0.816 m | 0.180 m | 0.145 m | 0.171 m | 3.92 deg |

## Observations

1. The base learnable alignment already solves most of the segment-to-global
   rotation error: camera rotation drops from about 118 deg to about 4 deg.

2. Explicit body-frame/body-vector loss is useful. It reduces body-frame
   orientation error from about 10.74 deg to about 3.82-3.92 deg, and also
   slightly improves human alignment.

3. Body-frame input alone is not enough. It improves human position metrics a
   little, but body-frame orientation remains around 10.85 deg. This means that
   simply giving the model body-frame information does not force it to use that
   information correctly.

4. Adding both body-frame input and body-frame loss is not clearly better than
   adding body loss only. The current evidence suggests the explicit supervision
   is the important part.

5. MVHuman100 remains the weakest source. Its human metric can get slightly
   worse after alignment, so the next version should inspect MVHuman100 scale,
   joint conventions, and body-frame construction before trusting it as a final
   training signal.

## Recommendation

For the next V10 prototype, keep the learnable streaming alignment module and
add explicit body-frame/body-vector supervision. Do not rely on body-frame input
alone as the main change.

The most conservative next variant is:

- Keep the current alignment MLP feature.
- Add body-frame and body-vector loss.
- Keep body-frame input optional or disabled until larger data proves it helps.

Before scaling to a larger train set, run one held-out check on non-training
AABB clips and then repeat on AIST/H36M 4-frame examples.
