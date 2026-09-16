# Traditional registration baselines

| Method | Access | EgoBody W | EgoBody WA | EgoBody ATE-Sim3 | Reg. success | EgoHumans W | EgoHumans WA | EgoHumans ATE-SE3 | Reg. success |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Human3R per-shot reset | per-shot online | 752.6 (n=129) | 589.4 (n=129) | 1.141 (n=129) | N/A | 1248.1 (n=90) | 491.3 (n=90) | 4.411 (n=90) | N/A |
| + pelvis translation | boundary-time | 500.2 (n=129) | 431.8 (n=129) | 0.068 (n=129) | 100.0% | 1242.1 (n=90) | 472.0 (n=90) | 3.326 (n=90) | 100.0% |
| + human-joint SE(3) | boundary-time | 261.5 (n=129) | 215.0 (n=129) | 0.019 (n=129) | 100.0% | 897.0 (n=90) | 339.9 (n=90) | 1.731 (n=90) | 82.2% |
| + scene registration + ICP | offline post-hoc | 733.2 (n=129) | 496.9 (n=129) | 0.020 (n=129) | 100.0% | 1372.8 (n=90) | 489.3 (n=90) | 3.847 (n=90) | 41.1% |
| Shot3R | streaming | 258.8 (n=129) | 201.7 (n=129) | 0.016 (n=129) | N/A | 866.8 (n=90) | 336.9 (n=90) | 1.423 (n=90) | N/A |

EgoBody uses recording-macro aggregation. EgoHumans retains the frozen paper headline case macro; capture-macro values and paired capture bootstrap are included in `aggregate.json`. Traditional methods receive annotated cut boundaries; Shot3R remains streaming with its online detector.
