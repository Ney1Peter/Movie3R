#!/usr/bin/env bash
# Create a minimal anonymous Overleaf import ZIP from the v003 manuscript.
#
# The ZIP deliberately contains only files required to compile main.tex.
# Large local evidence ledgers, raw outputs, cached predictions, PDFs from
# previous builds, and absolute local paths remain outside the release.

set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manuscript="${package_root}/versions/v003_20260825_camera_human_rewrite/manuscript"
release_dir="${package_root}/releases"
release_zip="${release_dir}/bridge3r_iclr2027_v003_20260825_overleaf.zip"
staging="$(mktemp -d "${TMPDIR:-/tmp}/bridge3r_overleaf_v003.XXXXXX")"

cleanup() {
  rm -rf "${staging}"
}
trap cleanup EXIT

mkdir -p "${release_dir}"

copy_file() {
  local relative="$1"
  local source="${manuscript}/${relative}"
  if [[ ! -f "${source}" ]]; then
    echo "Missing required release file: ${relative}" >&2
    exit 1
  fi
  mkdir -p "${staging}/$(dirname "${relative}")"
  cp -- "${source}" "${staging}/${relative}"
}

for relative in \
  main.tex math_commands.tex references.bib \
  iclr2027_conference.sty iclr2027_conference.bst \
  natbib.sty fancyhdr.sty README.md BUILD.md CHANGELOG.md \
  artifacts/harmony4d_unified_table.tex \
  artifacts/egobody_v20/recording_macro_primary.tex \
  artifacts/egobody_v20/recording_macro_local.tex \
  artifacts/egobody_v20/recording_macro_boundary.tex \
  artifacts/egobody_v20/angle_strata.tex \
  artifacts/egobody_v20/detector_table.tex \
  artifacts/egohuman_external_table.tex \
  artifacts/egohuman_external_angle_table.tex \
  artifacts/egohuman_external_action_table.tex \
  figures/method.pdf figures/qualitative.pdf figures/teaser.pdf \
  figures/FIGURE_PROVENANCE.md figures/QUALITATIVE_PROVENANCE.json; do
  copy_file "${relative}"
done

while IFS= read -r -d '' source; do
  relative="${source#${manuscript}/}"
  copy_file "${relative}"
done < <(find "${manuscript}/sections" "${manuscript}/tables" -type f -name '*.tex' -print0 | sort -z)

rm -f "${release_zip}"
(
  cd "${staging}"
  zip -q -r "${release_zip}" .
)

if unzip -Z1 "${release_zip}" | rg -q '(^|/)(main\.pdf|.*\.(aux|bbl|blg|log|out|synctex\.gz))$'; then
  echo "Release unexpectedly contains build products." >&2
  exit 1
fi
if unzip -Z1 "${release_zip}" | rg -q '(^|/)artifacts/(harmony4d_final|egobody_v20/.*\.(json|csv|md))'; then
  echo "Release unexpectedly contains non-compilation evidence artifacts." >&2
  exit 1
fi

echo "${release_zip}"
