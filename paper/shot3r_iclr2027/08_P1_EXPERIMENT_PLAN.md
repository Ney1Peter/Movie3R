# P1 enhancement plan after v030

| Priority | Study | Required protocol | Success and failure interpretation | Destination |
|---:|---|---|---|---|
| 1 | Independent token retraining | Native shadow, principal token groups, full, oracle; same development/test split and at least three seeds | Determines component attribution; a null result requires simplifying the token claim | Main + Supplement |
| 2 | Detector-inclusive runtime | 10--20 fixed clips; preprocessing, detector, reconstruction, boundary work, total latency | Supports only measured throughput; no real-time claim without full pipeline | Supplement, main one-line summary |
| 3 | Motion/offset analysis | Offsets 0/1/2/4/8 and motion strata fixed before execution | Tests whether human motion contaminates the shared translation | Supplement; main if interaction is clear |
| 4 | Larger repeated-cut study | 20--50 captures, multiple cut counts and loops | Separates causal composability from long-horizon accuracy | Supplement + drift curve |
| 5 | EgoBody direct association | Same denominator definitions as EgoHumans/Harmony4D | Completes the three-dataset association audit | Supplement |
| 6 | Natural edited-video set | Licensed, pre-specified qualitative manifest; no 3D accuracy claims | Supports qualitative transfer and detector taxonomy only | Supplement/video |
| 7 | Second recurrent backbone | Fixed compatible causal interface | Tests generality; otherwise keep Human3R-derived scope | Main if successful |
| 8 | Blind perceptual study | Method labels hidden, randomized order, agreement and bootstrap CI | Supports perceived continuity only | Supplement |

