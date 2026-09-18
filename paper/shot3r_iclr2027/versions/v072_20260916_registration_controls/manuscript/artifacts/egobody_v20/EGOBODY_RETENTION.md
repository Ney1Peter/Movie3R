# EgoBody v20 retention and cleanup

The EgoBody v20 Development, Holdout, Test, TRACE, and PromptHMR runs are
complete.  The final ICLR-facing evidence is retained in this directory:

- generated recording-macro, local, boundary, angle, detector, safety, and
  runtime tables;
- `recording_macro_main.json/.csv` and the generated TeX tables;
- the frozen protocol and candidate ledgers under
  `data/EgoBody_work_v20/frozen/`;
- the runtime/evaluator manifests and provenance metadata under
  `data/EgoBody_work_v20/manifests/` and `metadata/`;
- the compact aggregate summaries and case/recording CSV files under
  `data/EgoBody_work_v20/external_predictions/*/aggregate/`.

After the final results were frozen, the raw EgoBody archive, staged RGB
frames, GT/prediction caches, raw external-baseline outputs, and intermediate
run predictions were removed to release storage.  Re-running EgoBody from
scratch therefore requires uploading the source dataset again; the retained
tables and hashes are sufficient for the current paper and audit record.
