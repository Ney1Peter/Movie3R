# Bridge3R ICLR 2027 publication lock

This directory is the source of truth for the final paper method.  Read
`PAPER_METHOD_LOCK.json` before changing the runtime, experiments, tables, or
manuscript.  The lock preserves the distinction between historical frozen
artifacts and the post-hoc unified publication contract.

- `configs/egobody_reconstructed_publication_config.json` is a transparent
  reconstruction of a deleted historical candidate configuration; its original
  SHA is retained in the file and in the method lock.
- `RESULT_SOURCE_OF_TRUTH.md` defines exactly which frozen results may appear
  in the manuscript.
- `METHOD_TO_CODE_MAP.md` records current implementation coverage and the
  integration work still required for a release entry point.
- `CLAIM_EVIDENCE_LEDGER.csv` prevents unsupported claims from entering the
  paper while verification work is incomplete.
