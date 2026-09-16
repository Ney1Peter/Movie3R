# Viewpoint-stratified traditional registration results

Each cell is the frozen publication aggregation within the pre-defined stratum. Failure denotes prediction-only fitting failure followed by the mandatory R0 fallback.

| Dataset | Stratum | Cases | Method | W (mm) | WA (mm) | ATE (m) | Failure |
|---|---|---:|---|---:|---:|---:|---:|
| egobody | all | 129 | Human3R per-shot reset | 752.6 (n=129) | 589.4 (n=129) | 1.141 (n=129) | N/A |
| egobody | all | 129 | + pelvis translation | 500.2 (n=129) | 431.8 (n=129) | 0.068 (n=129) | 0.0% |
| egobody | all | 129 | + human-joint SE(3) | 261.5 (n=129) | 215.0 (n=129) | 0.019 (n=129) | 0.0% |
| egobody | all | 129 | + scene registration + ICP | 733.2 (n=129) | 496.9 (n=129) | 0.020 (n=129) | 0.0% |
| egobody | all | 129 | Shot3R | 258.8 (n=129) | 201.7 (n=129) | 0.016 (n=129) | N/A |
| egobody | small | 43 | Human3R per-shot reset | 546.4 (n=43) | 423.2 (n=43) | 0.606 (n=43) | N/A |
| egobody | small | 43 | + pelvis translation | 265.3 (n=43) | 216.3 (n=43) | 0.047 (n=43) | 0.0% |
| egobody | small | 43 | + human-joint SE(3) | 194.1 (n=43) | 148.9 (n=43) | 0.018 (n=43) | 0.0% |
| egobody | small | 43 | + scene registration + ICP | 413.9 (n=43) | 287.7 (n=43) | 0.017 (n=43) | 0.0% |
| egobody | small | 43 | Shot3R | 201.4 (n=43) | 160.6 (n=43) | 0.017 (n=43) | N/A |
| egobody | medium | 43 | Human3R per-shot reset | 697.5 (n=43) | 557.8 (n=43) | 1.061 (n=43) | N/A |
| egobody | medium | 43 | + pelvis translation | 517.9 (n=43) | 443.5 (n=43) | 0.073 (n=43) | 0.0% |
| egobody | medium | 43 | + human-joint SE(3) | 249.1 (n=43) | 203.6 (n=43) | 0.020 (n=43) | 0.0% |
| egobody | medium | 43 | + scene registration + ICP | 650.1 (n=43) | 451.9 (n=43) | 0.019 (n=43) | 0.0% |
| egobody | medium | 43 | Shot3R | 220.8 (n=43) | 171.2 (n=43) | 0.017 (n=43) | N/A |
| egobody | extreme | 43 | Human3R per-shot reset | 1013.9 (n=43) | 787.2 (n=43) | 1.756 (n=43) | N/A |
| egobody | extreme | 43 | + pelvis translation | 717.3 (n=43) | 635.5 (n=43) | 0.086 (n=43) | 0.0% |
| egobody | extreme | 43 | + human-joint SE(3) | 341.2 (n=43) | 292.6 (n=43) | 0.020 (n=43) | 0.0% |
| egobody | extreme | 43 | + scene registration + ICP | 1135.7 (n=43) | 751.1 (n=43) | 0.023 (n=43) | 0.0% |
| egobody | extreme | 43 | Shot3R | 354.3 (n=43) | 273.3 (n=43) | 0.016 (n=43) | N/A |
| egohumans | all | 90 | Human3R per-shot reset | 1248.1 (n=90) | 491.3 (n=90) | 4.411 (n=90) | N/A |
| egohumans | all | 90 | + pelvis translation | 1242.1 (n=90) | 472.0 (n=90) | 3.326 (n=90) | 0.0% |
| egohumans | all | 90 | + human-joint SE(3) | 897.0 (n=90) | 339.9 (n=90) | 1.731 (n=90) | 17.8% |
| egohumans | all | 90 | + scene registration + ICP | 1372.8 (n=90) | 489.3 (n=90) | 3.847 (n=90) | 58.9% |
| egohumans | all | 90 | Shot3R | 866.8 (n=90) | 336.9 (n=90) | 1.423 (n=90) | N/A |
| egohumans | small | 24 | Human3R per-shot reset | 1093.6 (n=24) | 499.3 (n=24) | 2.113 (n=24) | N/A |
| egohumans | small | 24 | + pelvis translation | 1158.9 (n=24) | 442.2 (n=24) | 1.355 (n=24) | 0.0% |
| egohumans | small | 24 | + human-joint SE(3) | 872.3 (n=24) | 351.0 (n=24) | 0.870 (n=24) | 16.7% |
| egohumans | small | 24 | + scene registration + ICP | 1136.2 (n=24) | 497.9 (n=24) | 1.953 (n=24) | 75.0% |
| egohumans | small | 24 | Shot3R | 850.9 (n=24) | 357.1 (n=24) | 0.610 (n=24) | N/A |
| egohumans | medium | 22 | Human3R per-shot reset | 1271.4 (n=22) | 513.1 (n=22) | 4.559 (n=22) | N/A |
| egohumans | medium | 22 | + pelvis translation | 1262.3 (n=22) | 505.9 (n=22) | 3.724 (n=22) | 0.0% |
| egohumans | medium | 22 | + human-joint SE(3) | 986.1 (n=22) | 345.9 (n=22) | 1.584 (n=22) | 13.6% |
| egohumans | medium | 22 | + scene registration + ICP | 1381.3 (n=22) | 518.3 (n=22) | 3.964 (n=22) | 59.1% |
| egohumans | medium | 22 | Shot3R | 801.2 (n=22) | 341.5 (n=22) | 1.301 (n=22) | N/A |
| egohumans | large | 22 | Human3R per-shot reset | 1303.0 (n=22) | 449.5 (n=22) | 5.031 (n=22) | N/A |
| egohumans | large | 22 | + pelvis translation | 1242.1 (n=22) | 455.6 (n=22) | 3.693 (n=22) | 0.0% |
| egohumans | large | 22 | + human-joint SE(3) | 928.2 (n=22) | 346.7 (n=22) | 1.685 (n=22) | 22.7% |
| egohumans | large | 22 | + scene registration + ICP | 1660.0 (n=22) | 456.8 (n=22) | 4.193 (n=22) | 40.9% |
| egohumans | large | 22 | Shot3R | 930.7 (n=22) | 331.1 (n=22) | 1.529 (n=22) | N/A |
| egohumans | extreme | 22 | Human3R per-shot reset | 1338.5 (n=22) | 502.6 (n=22) | 6.153 (n=22) | N/A |
| egohumans | extreme | 22 | + pelvis translation | 1312.6 (n=22) | 486.9 (n=22) | 4.712 (n=22) | 0.0% |
| egohumans | extreme | 22 | + human-joint SE(3) | 803.5 (n=22) | 314.9 (n=22) | 2.864 (n=22) | 18.2% |
| egohumans | extreme | 22 | + scene registration + ICP | 1335.2 (n=22) | 483.6 (n=22) | 5.451 (n=22) | 59.1% |
| egohumans | extreme | 22 | Shot3R | 886.1 (n=22) | 315.9 (n=22) | 2.326 (n=22) | N/A |
| egohumans | ge150deg | 22 | Human3R per-shot reset | 1338.5 (n=22) | 502.6 (n=22) | 6.153 (n=22) | N/A |
| egohumans | ge150deg | 22 | + pelvis translation | 1312.6 (n=22) | 486.9 (n=22) | 4.712 (n=22) | 0.0% |
| egohumans | ge150deg | 22 | + human-joint SE(3) | 803.5 (n=22) | 314.9 (n=22) | 2.864 (n=22) | 18.2% |
| egohumans | ge150deg | 22 | + scene registration + ICP | 1335.2 (n=22) | 483.6 (n=22) | 5.451 (n=22) | 59.1% |
| egohumans | ge150deg | 22 | Shot3R | 886.1 (n=22) | 315.9 (n=22) | 2.326 (n=22) | N/A |
