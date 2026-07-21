# Movie3R V12

V12 tests whether the V11 Gauge-Neutral First-Write Oracle can be distilled into a deployable one-forward adapter with a learned safety gate.

## Result

The first learned version does not pass the continuation criteria on the completely unseen MVHuman200 split:

- it retains only about 5.5% of the V11 Oracle camera gain;
- strict relative-camera success remains unchanged;
- the gate does not generalize and activates on almost every test case;
- the no-old-state adapter is slightly better than the intended state-query adapter;
- correct old state is not consistently better than zero, shuffled, or wrong state.

The recurrent-state modification route should stop in its current form. The next priority is explicit world relocalization and a redesigned geometry-dominant reliability/fallback predictor.

## Documents

- [V12 Learned Gated Gauge-Neutral First-Write Prompt](V12_LEARNED_GATED_GAUGE_NEUTRAL_FIRST_WRITE_PROMPT_20260719.md)

## Results

```text
output/v12_gated_first_write/teacher_cache_loso_mvhuman200/
output/v12_gated_first_write/training_loso_mvhuman200/
output/v12_gated_first_write/eval_loso_mvhuman200/merged/
```
