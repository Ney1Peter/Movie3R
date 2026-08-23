#!/usr/bin/env python3
"""Extract only frozen EgoBody JPEG members from a materialized inner ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

from stage_images import image_members_for_row, sha256_file


SCHEMA = "Movie3R-v20-EgoBody-image-stage-v1"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-zip", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--staged-manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = args.manifest.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    prepared, requested = [], set()
    for index, original in enumerate(rows):
        row = dict(original)
        row["image_members"] = image_members_for_row(row, expected_images=150)
        row["image_paths"] = list(row["image_members"])
        row["image_paths_relative_to_stage_root"] = list(row["image_members"])
        prepared.append(row)
        requested.update(row["image_members"])
    requested = sorted(requested)
    root = args.output_root.resolve()
    started = time.time()
    images = []
    with zipfile.ZipFile(args.inner_zip.resolve(), "r") as archive:
        infos = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        for index, member in enumerate(requested, start=1):
            info = infos.get(member)
            if info is None:
                raise FileNotFoundError(f"missing inner ZIP member: {member}")
            target = root / member
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and not target.is_symlink():
                digest = sha256_file(target)
                if target.stat().st_size != info.file_size or digest != hashlib.sha256(archive.read(info)).hexdigest():
                    raise ValueError(f"existing staged member differs: {target}")
                source = "reused_after_inner_zip_verification"
            else:
                data = archive.read(info)
                partial = target.with_suffix(target.suffix + ".partial")
                partial.write_bytes(data)
                os.replace(partial, target)
                source = "random_extracted_from_materialized_inner_zip"
                digest = hashlib.sha256(data).hexdigest()
            images.append({"member": member, "path": str(target.resolve()), "source": source, "size_bytes": int(info.file_size), "sha256": digest, "crc32": f"{info.CRC:08x}"})
            if index % 25 == 0:
                print(f"extracted/verified {index}/{len(requested)} unique images", flush=True)
    staged_manifest = args.staged_manifest_output.resolve()
    staged_manifest.parent.mkdir(parents=True, exist_ok=True)
    partial_manifest = staged_manifest.with_suffix(staged_manifest.suffix + ".partial")
    partial_manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in prepared), encoding="utf-8")
    os.replace(partial_manifest, staged_manifest)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    # The materialized inner ZIP is 329 GB.  Computing a second full-file hash
    # would reread the entire archive after every requested member was already
    # verified by ZIP size/CRC and SHA256.  Keep the archive size and the
    # per-member digests as the integrity evidence for this staging path.
    inner_sha = None
    payload = {
        "schema_version": SCHEMA,
        "status": "complete",
        "updated_unix_seconds": time.time(),
        "manifests": [str(args.manifest.resolve())],
        "manifest_sha256": manifest_sha,
        "output_root": str(root),
        "provenance_output": str(args.provenance_output.resolve()),
        "staged_manifest_output": str(staged_manifest),
        "archive_fingerprint": {"outer_path": str(args.inner_zip.resolve()), "nested_entry": "materialized_kinect_color.zip"},
        "inner_zip_sha256": inner_sha,
        "inner_zip_size_bytes": int(args.inner_zip.stat().st_size),
        "inner_zip_hash_policy": "not computed; every requested member verified against central-directory size/CRC and SHA256",
        "requested_unique_images": len(requested),
        "recorded_images": len(images),
        "limits": {"expected_images_per_case": 150},
        "method": "materialized outer nested ZIP with random central-directory extraction",
        "elapsed_seconds": time.time() - started,
        "images": images,
    }
    atomic_json(args.provenance_output.resolve(), payload)
    print(json.dumps({"status": "complete", "requested_unique_images": len(requested), "elapsed_seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
