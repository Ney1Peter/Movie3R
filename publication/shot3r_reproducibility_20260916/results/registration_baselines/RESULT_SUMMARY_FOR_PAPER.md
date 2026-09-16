# Shot3R traditional-registration experiment: paper summary

## Frozen protocol at a glance

R0 reconstructs both shots independently with the public Human3R checkpoint. R1 estimates one shared pelvis translation, R2 one prediction-only robust human-joint SE(3), and R3 an offline FPFH/RANSAC + point-to-plane ICP transform from both complete shot point clouds. Every failed fit is an exact R0 fallback and remains in the fixed denominator. Traditional methods receive annotated boundaries; Shot3R uses its streaming boundary detector.

| Dataset | Cases / units | Method | W (mm) | WA (mm) | ATE (m) | IDF1 | Coverage | Registration success |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| egobody | 129 / 43 recordings | Human3R per-shot reset | 752.6 | 589.4 | 1.141 | 0.834 | 0.981 | N/A |
| egobody | 129 / 43 recordings | pelvis translation | 500.2 | 431.8 | 0.068 | 0.896 | 0.981 | 100.0% |
| egobody | 129 / 43 recordings | human-joint robust SE(3) | 261.5 | 215.0 | 0.019 | 0.896 | 0.981 | 100.0% |
| egobody | 129 / 43 recordings | scene registration + ICP | 733.2 | 496.9 | 0.020 | 0.834 | 0.981 | 100.0% |
| egobody | 129 / 43 recordings | Shot3R | 258.8 | 201.7 | 0.016 | 0.985 | 0.981 | N/A |
| egohumans | 90 / 27 captures | Human3R per-shot reset | 1248.1 | 491.3 | 4.411 | 0.455 | 0.616 | N/A |
| egohumans | 90 / 27 captures | pelvis translation | 1242.1 | 472.0 | 3.326 | 0.506 | 0.616 | 100.0% |
| egohumans | 90 / 27 captures | human-joint robust SE(3) | 897.0 | 339.9 | 1.731 | 0.510 | 0.616 | 82.2% |
| egohumans | 90 / 27 captures | scene registration + ICP | 1372.8 | 489.3 | 3.847 | 0.455 | 0.616 | 41.1% |
| egohumans | 90 / 27 captures | Shot3R | 866.8 | 336.9 | 1.423 | 0.571 | 0.616 | N/A |

## Answers to the ten protocol questions

1. **Frozen Development settings (egobody).** R1 `r1_pair020`, R2 `r2_p020_i020_q040`, and R3 `r3_v018_loose` were selected once by the declared complete-Development aggregate objective. The settings are dataset-level, passed the one-shot Holdout gate, and were never selected case by case.
1. **Frozen Development settings (egohumans).** R1 `r1_pair035`, R2 `r2_p020_i020_q040`, and R3 `r3_v008_strict` were selected once by the declared complete-Development aggregate objective. The settings are dataset-level, passed the one-shot Holdout gate, and were never selected case by case.

**Aggregation note.** EgoBody headline values are recording-macro over 43 recordings. To reproduce the frozen paper table exactly, EgoHumans headline values retain its established case-macro convention; capture-macro values are stored alongside every metric in `aggregate.json`, and all EgoHumans paired intervals resample the 27 captures. Thus no confidence interval treats the 90 cases as independent.

2. **R0 reproduction.** EgoBody reproduces W/WA/ATE-Sim3 = 752.6/589.4 mm/1.141 m; the preregistered references are 752.6/589.4 mm/1.141 m. EgoHumans reproduces W/WA = 1248.1/491.3 mm versus 1248.1/491.3 mm. All numerical gates pass within the frozen tolerance.

3. **What pelvis translation fixes.** On EgoBody, R1 reduces W from 752.6 to 500.2 mm and ATE from 1.141 to 0.068 m, but boundary rotation error stays 55.7 degrees. On EgoHumans, W changes only from 1248.1 to 1242.1 mm, and its paired W gain is not significant: 14.0 (95% CI -48.8 to 82.2; n=90, units=27). Translation cannot resolve viewpoint rotation or an incorrect identity match.

4. **Does robust human SE(3) improve on translation?** Yes in the fixed-denominator aggregate. Relative to R1, R2 changes W/WA/ATE from 500.2/431.8 mm/0.068 m to 261.5/215.0 mm/0.019 m on EgoBody and reduces boundary rotation from 55.7 to 17.9 degrees. On EgoHumans it changes W/WA/ATE from 1242.1/472.0 mm/3.326 m to 897.0/339.9 mm/1.731 m and reduces boundary rotation from 104.5 to 48.8 degrees. Its W gain over R0 is 491.1 (95% CI 431.9 to 549.9; n=129, units=43) and 309.5 (95% CI 174.1 to 442.4; n=90, units=27), respectively. The limitation is robustness: prediction-only RANSAC fails on 16/90 EgoHumans cases, which remain exact R0 fallbacks, and R2 retains lower IDF1 than Shot3R on both datasets.

5. **How scene registration fails.** R3 reports 129/129 algorithmic successes on EgoBody but W remains 733.2 mm even though ATE falls to 0.020 m; a numerically accepted camera alignment therefore does not imply correct human placement. On EgoHumans it succeeds on only 37/90 cases (41.1%), with 17 global-RANSAC failures and 36 ICP-gate failures; W is 1372.8 mm, worse than R0. Its prediction-only scene correspondence inlier ratio has median 0.055, consistent with weak overlap, and the fixed extreme/at-least-150-degree stratum has 59.1% algorithmic failure.

6. **Do traditional methods improve R0 significantly?** R2 does on W for both datasets because its 95% CIs exclude zero: 491.1 (95% CI 431.9 to 549.9; n=129, units=43) and 309.5 (95% CI 174.1 to 442.4; n=90, units=27). R1 is significant on EgoBody but not EgoHumans W. R3 has no significant EgoBody W gain and significantly degrades EgoHumans W (see `registration_paired.md`).

7. **Shot3R versus the strongest traditional method.** R2 is the lowest-W traditional method on both datasets. Shot3R versus R2 is 258.8 vs 261.5 mm W, 201.7 vs 215.0 mm WA, and 0.016 vs 0.019 m ATE on EgoBody. The R2-minus-Shot3R paired differences are W 2.6 (95% CI -40.4 to 53.8; n=129, units=43), WA 13.3 (95% CI -30.4 to 67.3; n=129, units=43), and ATE 0.003 (95% CI 0.000 to 0.005; n=129, units=43). On EgoHumans the corresponding headline values are W 866.8 vs 897.0 mm, WA 336.9 vs 339.9 mm, and ATE 1.423 vs 1.731 m. The paired differences are W 24.9 (95% CI -102.0 to 134.0; n=90, units=27), WA 2.1 (95% CI -23.2 to 30.1; n=90, units=27), and ATE 0.324 (95% CI -0.197 to 0.846; n=90, units=27).

8. **Viewpoint dependence.** The advantage is not monotonic in W. R2 is lower than Shot3R on EgoBody small W (194.1 vs 201.4 mm) and extreme W (341.2 vs 354.3 mm), while Shot3R is lower on medium W and on extreme WA/ATE. On EgoHumans, R2 is lower on large W (928.2 vs 930.7 mm) and extreme W (803.5 vs 886.1 mm), whereas Shot3R is lower on medium W and on ATE in every listed stratum. These are the pre-defined strata; no result-dependent angle subset was introduced.

9. **Does this prove Shot3R is not simple post-hoc registration?** It supports a qualified, not absolute, statement. Pelvis translation and scene ICP are not interchangeable substitutes: they leave large human or rotation errors, and scene ICP is offline and unstable on EgoHumans. Robust human SE(3) is much stronger and sometimes matches or exceeds Shot3R in stratum-level W, but it receives the annotated boundary, fails on some crowded cases, does not recover Shot3R's IDF1, and uses a separate post-hoc geometric fit. The appropriate claim is that Shot3R jointly provides streaming cross-shot state handling and association rather than merely applying scene ICP or a shared translation to independent reconstructions.

10. **Results that do not support the strongest initial expectation.** Shot3R's overall W advantage over R2 is small and statistically inconclusive on both datasets: 2.6 (95% CI -40.4 to 53.8; n=129, units=43) and 24.9 (95% CI -102.0 to 134.0; n=90, units=27). R2 is better in several fixed W strata, and on EgoBody R3 attains ATE 0.020 m, close to Shot3R's 0.016 m. These outcomes must remain visible; the experiment falls under protocol Decision Rule B, not Rule A.

## Recommended paper placement

Use the compact main table or one short paragraph in the main paper, emphasizing access and joint W/WA/ATE/IDF1 rather than claiming universal W dominance. Put the complete pre-defined angle strata, common-support 100,000-sample bootstrap, fitting diagnostics, valid support, failure taxonomy, and locked qualitative cases in the supplement. State explicitly that R3 is offline post-hoc and that every traditional baseline receives annotated cut boundaries, which favors the baselines.
