# H4D-CS150 runtime results

Five 150-frame sequences (750 frames total); wall-clock process latency on NVIDIA L20.

| Method | Access | s / 150 frames (mean ± std) | FPS ↑ |
|---|---|---:|---:|
| Human3R | online | 120.4 ± 3.2 | 1.246 |
| Shot3R | online | 167.8 ± 4.0 | 0.894 |
| PromptHMR (SPEC) | offline | 708.7 ± 209.0 | 0.212 |
| OnlineHMR | semi-online | 1007.8 ± 124.7 | 0.149 |
| TRACE | online | 45.0 ± 21.5 | 3.334 |
| TRAM | offline | 455.5 ± 60.6 | 0.329 |
| JOSH | offline | 949.6 ± 286.0 | 0.158 |
