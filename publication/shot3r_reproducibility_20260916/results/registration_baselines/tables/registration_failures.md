# Registration failure analysis

Algorithmic failure is decided from prediction-only fitting diagnostics and triggers the mandatory exact-R0 fallback. Catastrophic evaluation failure is a post-seal diagnostic only: boundary translation error greater than 1 m or boundary rotation error greater than 30 degrees. It never selects a Test transform.

| Dataset | Method | Status counts | Algorithmic failure | Catastrophic evaluation | Fit time mean / median / p90 (s) |
|---|---|---|---:|---:|---:|
| egobody | + pelvis translation | ok: 129 | 0/129 (0.0%) | 112/129 valid (86.8%; unavailable=0) | 0.002 / 0.002 / 0.003 |
| egobody | + human-joint SE(3) | ok: 129 | 0/129 (0.0%) | 30/129 valid (23.3%; unavailable=0) | 0.078 / 0.076 / 0.087 |
| egobody | + scene registration + ICP | ok: 129 | 0/129 (0.0%) | 57/129 valid (44.2%; unavailable=0) | 0.524 / 0.407 / 0.955 |
| egohumans | + pelvis translation | ok: 90 | 0/90 (0.0%) | 90/90 valid (100.0%; unavailable=0) | 0.007 / 0.007 / 0.012 |
| egohumans | + human-joint SE(3) | ok: 74, ransac_failed: 16 | 16/90 (17.8%) | 74/90 valid (82.2%; unavailable=0) | 0.119 / 0.114 / 0.149 |
| egohumans | + scene registration + ICP | icp_failed: 36, ok: 37, ransac_failed: 17 | 53/90 (58.9%) | 88/90 valid (97.8%; unavailable=0) | 39.132 / 35.422 / 76.523 |

## Prediction-only scene-registration diagnostics

| Dataset | Pre-scene points (median / p90) | Post-scene points (median / p90) | Correspondences (median / p90) | Inlier ratio (median / p90) | Residual (median / p90, m) | Global fitness (median / p90) |
|---|---:|---:|---:|---:|---:|---:|
| egobody | 2936 / 4708 (n=129) | 2950 / 5609 (n=129) | 1875 / 3043 (n=129) | 0.624 / 0.825 (n=129) | 0.143 / 0.159 (n=129) | 0.691 / 0.889 (n=129) |
| egohumans | 37632 / 51114 (n=90) | 29787 / 51635 (n=90) | 1572 / 3104 (n=90) | 0.055 / 0.198 (n=90) | 0.056 / 0.058 (n=90) | 0.213 / 0.340 (n=37) |

## Algorithmic failure by pre-defined viewpoint stratum

| Dataset | Stratum | Cases | Pelvis translation | Human-joint SE(3) | Scene + ICP |
|---|---|---:|---:|---:|---:|
| egobody | all | 129 | 0.0% | 0.0% | 0.0% |
| egobody | small | 43 | 0.0% | 0.0% | 0.0% |
| egobody | medium | 43 | 0.0% | 0.0% | 0.0% |
| egobody | extreme | 43 | 0.0% | 0.0% | 0.0% |
| egohumans | all | 90 | 0.0% | 17.8% | 58.9% |
| egohumans | small | 24 | 0.0% | 16.7% | 75.0% |
| egohumans | medium | 22 | 0.0% | 13.6% | 59.1% |
| egohumans | large | 22 | 0.0% | 22.7% | 40.9% |
| egohumans | extreme | 22 | 0.0% | 18.2% | 59.1% |
| egohumans | ge150deg | 22 | 0.0% | 18.2% | 59.1% |

## Algorithmic failure by person count

| Dataset | People | Cases | Pelvis translation | Human-joint SE(3) | Scene + ICP |
|---|---:|---:|---:|---:|---:|
| egobody | 2 | 129 | 0.0% | 0.0% | 0.0% |
| egohumans | 2 | 14 | 0.0% | 0.0% | 100.0% |
| egohumans | 3 | 32 | 0.0% | 3.1% | 25.0% |
| egohumans | 4 | 44 | 0.0% | 34.1% | 70.5% |
