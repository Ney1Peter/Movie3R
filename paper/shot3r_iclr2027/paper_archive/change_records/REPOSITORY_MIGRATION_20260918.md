# Repository migration record

On 2026-09-18, the Shot3R ICLR 2027 paper workspace was moved into the Movie3R
main repository.

- Canonical paper root: `Movie3R/paper/shot3r_iclr2027/`
- Active manuscript: `versions/v072_20260916_registration_controls/manuscript/`
- Central archive: `paper_archive/`
- Previous path: `ICLR-paper/bridge3r_iclr2027`
- Compatibility: the previous path is now a relative symlink to the canonical
  Movie3R location.

The nested standalone `.git` directory was removed from the imported working
tree so that the paper is governed by the Movie3R repository rather than by an
embedded repository. The preceding standalone Git history was preserved in
both forms below, outside the Movie3R working tree:

- `ICLR-paper/bridge3r_iclr2027_history_20260918.bundle`
- `ICLR-paper/bridge3r_iclr2027_legacy_git_20260918/`

The Git bundle was verified as a complete history and contains branch `main`,
tag `bridge3r-iclr2027-v031`, and the former HEAD commit
`2b02c23c1d15c1498697393690b231bf19a2e94f`.
