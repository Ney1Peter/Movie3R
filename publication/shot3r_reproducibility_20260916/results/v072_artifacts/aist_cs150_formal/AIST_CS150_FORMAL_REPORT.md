# AIST++ CS150 formal single-person result

All rows use the same frozen 100-source official `pose_test` RGB manifest and a 150-frame timeline. Bridge3R and Strict Human3R are causal streaming routes; PromptHMR is its unchanged official offline full-video pipeline. The rows are therefore not presented as a single latency-equivalent ranking.

| Execution | Method | PA-MPJPE | Anchor-MPJPE | Seam-root | Seam-orient. | Rel. camera rot. | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| causal | Strict Human3R | 57.7 | 492.9 | 776.0 | 59.6 | 59.1 | 100.0% |
| causal | Bridge3R (fixed, causal) | 57.6 | 556.3 | 821.5 | 50.1 | 49.5 | 100.0% |
| offline | PromptHMR (official, offline) | 51.7 | 462.6 | 55.4 | 43.8 | 94.0 | 100.0% |

## Provenance

- Internal aggregate: `/data/wangzheng/iJCV-CODE/Movie3R/experiments/bridge3r_singleperson_aist_v1/test/internal_cs150/reevaluated_v2_label_finite/aggregate.json`
- PromptHMR sealed ledger: `/data/wangzheng/iJCV-CODE/Movie3R/experiments/bridge3r_singleperson_aist_v1/test/prompthmr_official_v1_final_ledger_cuda_isolated_v3/aggregate.json`
- Every row has a 100-case macro denominator and 100% completion. The approximately 99.993% internal frame coverage is preserved before one-decimal display rounding.
- GVHMR is intentionally absent: its pre-registered 12-case availability audit found one-sided raw tracker support on 2/12 hard-cut pilot cases, so it did not enter the global Test table.
