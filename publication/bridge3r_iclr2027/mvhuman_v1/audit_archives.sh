#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="${BRIDGE3R_WORKSPACE_ROOT:-$(cd "${script_dir}/../../../.." && pwd)}"
audit_root="${workspace_root}/data/MVHuman_test12_audit_20260830"
output_root="${workspace_root}/Movie3R/output/bridge3r_mvhuman_v1/intake"
mkdir -p "${output_root}/per_archive"

audit_one() {
  local archive="$1"
  local name
  name="$(basename "${archive}" .tar.gz)"
  local report="${output_root}/per_archive/${name}.txt"
  local temporary="${report}.partial"
  {
    printf 'archive=%s\n' "${archive}"
    printf 'size_bytes=%s\n' "$(stat -c '%s' "${archive}")"
    printf 'mtime=%s\n' "$(stat -c '%y' "${archive}")"
    printf 'sha256=%s\n' "$(sha256sum "${archive}" | awk '{print $1}')"
    gzip -t "${archive}"
    printf 'gzip_test=PASS\n'
    tar -tzf "${archive}" | awk -F/ '
      BEGIN {members=0; unsafe=0; symlinks=0}
      {
        members += 1
        if ($0 ~ /^\// || $0 ~ /(^|\/)\.\.($|\/)/) unsafe += 1
        if (NF >= 2) top[$2] += 1
      }
      END {
        printf "members=%d\nunsafe_paths=%d\n", members, unsafe
        for (key in top) printf "top_%s=%d\n", key, top[key]
      }
    ' | sort
  } > "${temporary}"
  mv "${temporary}" "${report}"
}

export -f audit_one
export output_root

find "${audit_root}/test1_archives" "${audit_root}/test2_archives" \
  -maxdepth 1 -type f -name '*.tar.gz' -print0 \
  | sort -z \
  | xargs -0 -n1 -P2 bash -c 'audit_one "$1"' _

find "${output_root}/per_archive" -maxdepth 1 -type f -name '*.txt' -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${output_root}/report_checksums.sha256.partial"
mv "${output_root}/report_checksums.sha256.partial" "${output_root}/report_checksums.sha256"
date --iso-8601=seconds > "${output_root}/ARCHIVE_AUDIT_COMPLETE"
