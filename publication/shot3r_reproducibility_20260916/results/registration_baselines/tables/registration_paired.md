# Paired differences with 95% confidence intervals

Positive `vs R0` means the registration method has lower error than R0. Positive `vs Shot3R` means Shot3R has lower error than the registration method. Resampling unit is recording for EgoBody and capture for EgoHumans.

| Dataset | Method | Delta W vs R0 | Delta ATE vs R0 | Delta W vs Shot3R | Delta ATE vs Shot3R | Bootstrap unit |
|---|---|---:|---:|---:|---:|---|
| egobody | + pelvis translation | 252.4 [202.9, 304.3] (n=129, u=43) | 1.072 [0.946, 1.202] (n=129, u=43) | 241.3 [184.8, 298.0] (n=129, u=43) | 0.052 [0.038, 0.067] (n=129, u=43) | recording |
| egobody | + human-joint SE(3) | 491.1 [431.9, 549.9] (n=129, u=43) | 1.122 [0.992, 1.254] (n=129, u=43) | 2.6 [-40.4, 53.8] (n=129, u=43) | 0.003 [0.000, 0.005] (n=129, u=43) | recording |
| egobody | + scene registration + ICP | 19.4 [-95.5, 124.8] (n=129, u=43) | 1.121 [0.993, 1.252] (n=129, u=43) | 474.4 [373.4, 585.7] (n=129, u=43) | 0.003 [0.001, 0.006] (n=129, u=43) | recording |
| egohumans | + pelvis translation | 14.0 [-48.8, 82.2] (n=90, u=27) | 1.141 [0.901, 1.419] (n=90, u=27) | 320.3 [191.1, 452.7] (n=90, u=27) | 1.989 [1.552, 2.471] (n=90, u=27) | capture |
| egohumans | + human-joint SE(3) | 309.5 [174.1, 442.4] (n=90, u=27) | 2.806 [2.002, 3.696] (n=90, u=27) | 24.9 [-102.0, 134.0] (n=90, u=27) | 0.324 [-0.197, 0.846] (n=90, u=27) | capture |
| egohumans | + scene registration + ICP | -107.0 [-207.0, -24.3] (n=90, u=27) | 0.491 [0.272, 0.734] (n=90, u=27) | 441.4 [270.9, 620.5] (n=90, u=27) | 2.638 [1.986, 3.347] (n=90, u=27) | capture |
