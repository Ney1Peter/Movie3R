#!/usr/bin/env bash
set -euo pipefail

archive_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${archive_dir}/../.." && pwd)"
manuscript_dir="${1:-${repo_dir}/versions/v072_20260916_registration_controls/manuscript}"
pdf_path="${2:-${manuscript_dir}/main.pdf}"
official_dir="${archive_dir}/official_iclr2027"

for name in iclr2027_conference.sty iclr2027_conference.bst fancyhdr.sty natbib.sty math_commands.tex; do
  cmp "${official_dir}/${name}" "${manuscript_dir}/${name}"
done

pdfinfo "${pdf_path}" | awk -F: '/^(Pages|Page size|Creator|Producer)/ {print}'

if pdffonts "${pdf_path}" | grep -q 'Type 3'; then
  echo "FAIL: Type 3 font detected" >&2
  exit 1
fi

if pdffonts "${pdf_path}" | awk 'NR > 2 && $(NF-4) == "no" {found=1} END {exit !found}'; then
  echo "FAIL: unembedded font detected" >&2
  exit 1
fi

echo "PASS: official style files match; no Type 3 or unembedded fonts detected."
echo "MANUAL CHECK REQUIRED: complete Conclusion must end on or before page 9."
