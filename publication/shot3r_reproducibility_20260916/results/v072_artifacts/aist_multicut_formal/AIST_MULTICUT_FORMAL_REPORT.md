# AIST++ multi-cut formal result

Both tables use their own frozen 100-source official pose_test RGB manifests. All rows are causal internal routes; the study is an event-scaling and component analysis, not a latency-equivalent ranking against the offline PromptHMR CS150 row.

## MC150-3

| Method | PA-MPJPE | Anchor-MPJPE | Mean seam-root | Mean seam-orient. | Post-cut camera rot. | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 58.4 | 574.1 | 805.7 | 58.4 | 61.7 | 100.0% |
| Clean reset | 58.3 | 576.3 | 801.8 | 88.8 | 94.8 | 100.0% |
| Coarse alignment only | 58.3 | 1325.4 | 1967.4 | 53.8 | 56.5 | 100.0% |
| Coarse alignment + identity | 58.3 | 1325.4 | 1967.4 | 53.8 | 56.5 | 100.0% |
| Bridge3R (fixed, causal) | 58.3 | 751.0 | 983.3 | 53.8 | 56.5 | 100.0% |

## MC150-4

| Method | PA-MPJPE | Anchor-MPJPE | Mean seam-root | Mean seam-orient. | Post-cut camera rot. | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 59.0 | 687.1 | 819.9 | 58.7 | 57.3 | 100.0% |
| Clean reset | 58.8 | 623.7 | 746.4 | 89.0 | 99.2 | 100.0% |
| Coarse alignment only | 58.8 | 1482.4 | 2031.8 | 57.5 | 58.7 | 100.0% |
| Coarse alignment + identity | 58.8 | 1482.4 | 2031.8 | 57.5 | 58.7 | 100.0% |
| Bridge3R (fixed, causal) | 58.8 | 819.4 | 1015.6 | 57.5 | 58.7 | 100.0% |
