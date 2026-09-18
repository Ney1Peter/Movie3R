# Version-control scope

This repository preserves the Shot3R ICLR 2027 paper sources, versioned
evidence tables and manifests, figures, release PDFs, and importable Overleaf
archives. The active numbered manuscript as of 2026-09-18 is
`versions/v072_20260916_registration_controls`.

Long-term paper history and editable presentation assets are centralized in
`paper_archive/`. Under the current retention rule, that archive contains the
complete v067 snapshot, all recoverable v070-related records, four deduplicated
PPTX sources, and the ICLR 2027 template audit. A complete v070 PDF/source/ZIP
snapshot has not been found, so v071/v072 material must not be relabeled as
v070. All archived files are covered by `paper_archive/manifests/SHA256SUMS`.

The v072 directory remains the active working copy. It is not duplicated as a
historical snapshot in `paper_archive/`. Its previous README was actually the
v070 release note; the original text is preserved at
`paper_archive/versions/v070_related_materials/V070_RELEASE_NOTES.md`, and the
v072 README now describes the correct version.

At the user's request, working copies of paper versions v001--v029 and their
old release archives were removed during the 2026-09-06/07 disk cleanup.
In the subsequent 2026-09-07 cleanup, the user raised the retention threshold
to v045. Working version directories v030--v044 and obsolete pre-v045 release
files were removed; v045 onward and Git history are retained. Tracked v030--v041
sources remain recoverable from commit `2b02c23c1d15c1498697393690b231bf19a2e94f`.
Uncommitted v042--v044 were first saved as
`../../cleanup_records/20260907_v045/uncommitted_v042_v044_recovery.tar.gz`.
Previously tracked v001--v029
sources remain recoverable from the pre-cleanup commit `1a221bc`; ignored
build products and untracked discarded copies are not guaranteed recoverable.
See `../../WORKSPACE_CLEANUP_20260906.md` for the deletion and retention log.
See `../../WORKSPACE_CLEANUP_20260907_V045.md` for the newer cleanup and recovery
record. No Git history was rewritten and the new cleanup was not auto-committed.
This paper-version rule does not apply to model or experiment versions in
`Movie3R/versions`, `Movie3R/output`, or checkpoint directories.

Raw datasets, licensed model weights, caches, and large per-frame prediction
tensors remain outside this paper repository.  Their publication-facing
statistics are retained in the versioned `evidence`, `artifacts`, and table
directories, while the executable source of truth lives in
`Movie3R/publication/bridge3r_iclr2027`.

LaTeX auxiliary files are intentionally ignored because they can be rebuilt
from the committed manuscript.  Release PDFs and Overleaf ZIP archives are
tracked intentionally so that every handed-off paper version remains directly
recoverable.

This is currently a local Git repository.  A separate remote or storage backup
is still required to protect against loss of the workspace disk itself.
