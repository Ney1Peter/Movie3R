# EgoBody-CS150 final result inventory

All rows below use the frozen 43-recording/129-case, 150-frame EgoBody Test.
The primary aggregate averages small/medium/extreme cases within each
recording, then gives each recording equal weight.  The external rows use the
same RGB inputs, topology, matcher, evaluator, and aggregation.

| Method | W-MPJPE (mm) | WA-MPJPE (mm) | MPJPE (mm) | MPVPE (mm) | ATE-Sim3 (m) | IDF1 | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 321.1 | 233.4 | -- | -- | 0.124 | 0.870 | -- |
| Bridge3R runtime parent | 349.2 | 265.2 | -- | -- | 0.019 | 0.983 | -- |
| Bridge3R causal frozen | **258.8** | **201.7** | **70.94** | **79.00** | **0.016** | **0.985** | **98.059%** |
| TRACE† | 576.38 | 276.40 | 264.81 | 327.75 | N/A | 3.627% | 3.501% |
| PromptHMR (official SPEC) | 741.36 | 336.11 | 118.64 | 135.91 | 0.302 | 19.495% | 20.238% |
| PromptHMR (no-SPEC)† | 700.71 | 335.38 | 103.81 | 119.78 | 0.308 | 20.503% | 21.199% |

The first three rows are the audited Bridge3R/Human3R operating points; the
last three are executable external baselines.  TRACE and PromptHMR† have
conditional human metrics and limited availability; official SPEC has W on
34/43 recordings and WA/local on 36/43.  HumanMM and Multi-THuMBS remain
different-protocol or unavailable-code literature context and are not inserted
into this same-protocol table.

The complete official SPEC aggregate, confidence intervals, accounting, and
artifact hashes are in
[`PROMPTHMR_SPEC_EGOBODY_TEST.md`](../../../../external_baselines/bridge3r_eval/PROMPTHMR_SPEC_EGOBODY_TEST.md).
