#!/usr/bin/env python3
"""Stream selected EgoBody Kinect JPEGs out of a nested ZIP archive.

The EgoBody release stores ``kinect_color.zip`` inside a much larger outer ZIP.
This tool never materializes the inner ZIP and never extracts the complete image
tree.  It scans both ZIP streams once, writes only manifest-requested JPEGs, and
records enough provenance to safely resume or reuse a completed staging run.

Run this script with ``Movie3R/.venv/bin/python``; that environment provides
``stream-unzip`` and Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import time
import uuid
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover - produces a clearer CLI error
    raise SystemExit(
        "Pillow is required. Run with Movie3R/.venv/bin/python."
    ) from exc

try:
    from stream_unzip import stream_unzip
except ImportError as exc:  # pragma: no cover - produces a clearer CLI error
    raise SystemExit(
        "stream-unzip is required. Run with Movie3R/.venv/bin/python."
    ) from exc


SCHEMA = "Movie3R-v20-EgoBody-image-stage-v1"
DEFAULT_INNER_ZIP = "kinect_color.zip"
DEFAULT_EXPECTED_IMAGES = 150
DEFAULT_IO_CHUNK_MIB = 8
DEFAULT_STREAM_CHUNK_MIB = 1
DEFAULT_MAX_IMAGE_MIB = 32.0
DEFAULT_ESTIMATED_IMAGE_MIB = 2.0
DEFAULT_RESERVE_GIB = 20.0
WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=Path, required=True, help="Outer EgoBody.zip")
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more runtime JSONL/JSON manifests. All rows are merged and "
            "all unique frames are extracted in one archive scan."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Staging root; inner relative paths are preserved below this directory",
    )
    parser.add_argument(
        "--inner-zip-entry",
        default=DEFAULT_INNER_ZIP,
        help=f"Nested image ZIP member in the outer archive (default: {DEFAULT_INNER_ZIP})",
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help="Atomic JSON ledger (default: OUTPUT_ROOT/stage_images.provenance.json)",
    )
    parser.add_argument(
        "--staged-manifest-output",
        type=Path,
        help="Manifest with image_members/image_paths (default: OUTPUT_ROOT/runtime_manifest.staged.jsonl)",
    )
    parser.add_argument(
        "--expected-images-per-case",
        type=int,
        default=DEFAULT_EXPECTED_IMAGES,
        help="Require this many images per row; use 0 to accept variable-size test manifests",
    )
    parser.add_argument("--io-chunk-mib", type=int, default=DEFAULT_IO_CHUNK_MIB)
    parser.add_argument(
        "--stream-chunk-mib", type=int, default=DEFAULT_STREAM_CHUNK_MIB
    )
    parser.add_argument("--max-image-mib", type=float, default=DEFAULT_MAX_IMAGE_MIB)
    parser.add_argument(
        "--estimated-image-mib",
        type=float,
        default=DEFAULT_ESTIMATED_IMAGE_MIB,
        help="Per-missing-file estimate used only for the preflight disk budget",
    )
    parser.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--max-pixels", type=int, default=100_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and disk budget without scanning archives or writing files",
    )
    args = parser.parse_args(argv)
    if args.expected_images_per_case < 0:
        parser.error("--expected-images-per-case must be non-negative")
    if args.io_chunk_mib <= 0 or args.stream_chunk_mib <= 0:
        parser.error("chunk sizes must be positive")
    if args.max_image_mib <= 0 or args.estimated_image_mib <= 0:
        parser.error("image sizes must be positive")
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be non-negative")
    if args.max_pixels <= 0:
        parser.error("--max-pixels must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    return args


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def combined_manifest_sha256(inputs: Sequence[Mapping[str, str]]) -> str:
    """Hash an ordered, unambiguous list of manifest content digests."""
    digest = hashlib.sha256()
    for item in inputs:
        content_digest = str(item["sha256"])
        digest.update(len(content_digest).to_bytes(8, "big"))
        digest.update(content_digest.encode("ascii"))
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_path(path.parent, path.parent.resolve())
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        fsync_directory(path.parent)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    data = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    atomic_write_bytes(path, data)


def fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def safe_archive_member(raw: str, *, what: str) -> str:
    """Return one canonical POSIX member name or reject traversal/ambiguity."""
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{what} must be a non-empty string: {raw!r}")
    if "\x00" in raw or "\\" in raw:
        raise ValueError(f"Unsafe {what} (NUL or backslash): {raw!r}")
    if raw.startswith("/") or raw.startswith("//") or WINDOWS_DRIVE.match(raw):
        raise ValueError(f"Unsafe absolute {what}: {raw!r}")
    value = raw[:-1] if raw.endswith("/") else raw
    parts = PurePosixPath(value).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe traversal or ambiguous {what}: {raw!r}")
    canonical = "/".join(parts)
    if canonical != value:
        raise ValueError(f"Non-canonical {what}: {raw!r} -> {canonical!r}")
    return canonical


def decode_member(raw: bytes, *, what: str) -> str:
    try:
        value = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Non-UTF-8 {what}: {raw[:100]!r}") from exc
    return safe_archive_member(value, what=what)


def safe_output_path(output_root: Path, member: str) -> Path:
    root = output_root.resolve()
    candidate = output_root.joinpath(*PurePosixPath(member).parts)
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Output escapes staging root: {member!r} -> {resolved}")
    return candidate


def ensure_no_symlink_path(path: Path, root: Path) -> None:
    """Reject any existing symlink from root through path."""
    root = Path(os.path.abspath(root))
    lexical = Path(os.path.abspath(path))
    if lexical != root and root not in lexical.parents:
        raise ValueError(f"Path escapes root: {path} -> {lexical}")
    current = root
    relative = lexical.relative_to(root)
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink is not allowed below output root: {current}")
    resolved = lexical.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Resolved path escapes root: {path} -> {resolved}")


def file_chunks(path: Path, chunk_size: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return
            yield chunk


class HashingChunks(Iterator[bytes]):
    """One-pass iterator that hashes/counts bytes consumed by a nested parser."""

    def __init__(self, source: Iterable[bytes]) -> None:
        self._source = iter(source)
        self._hash = hashlib.sha256()
        self.size = 0

    def __iter__(self) -> "HashingChunks":
        return self

    def __next__(self) -> bytes:
        chunk = next(self._source)
        self._hash.update(chunk)
        self.size += len(chunk)
        return chunk

    @property
    def hexdigest(self) -> str:
        return self._hash.hexdigest()

    def drain(self) -> None:
        for _ in self:
            pass


def read_manifest(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    if not raw.strip():
        raise ValueError(f"Empty manifest: {path}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {line_number} is not an object")
            rows.append(row)
    else:
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = None
            for key in ("rows", "cases", "entries"):
                if key in parsed:
                    candidates = parsed[key]
                    break
            if candidates is None:
                candidates = [parsed]
        else:
            raise ValueError("JSON manifest must be an object or list")
        if not isinstance(candidates, list) or not all(
            isinstance(row, dict) for row in candidates
        ):
            raise ValueError("Manifest rows/cases/entries must be a list of objects")
        rows = [dict(row) for row in candidates]
    if not rows:
        raise ValueError(f"Manifest has no rows: {path}")
    return rows, raw


def normalized_camera(value: Any, *, field: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{field} cannot be boolean")
    text = str(value).strip()
    if text == "master":
        return text
    if re.fullmatch(r"sub_[0-9]+", text):
        return text
    if re.fullmatch(r"sub[0-9]+", text):
        return "sub_" + text[3:]
    if re.fullmatch(r"[0-9]+", text):
        return "sub_" + text
    raise ValueError(
        f"{field} must be 'master', 'sub_N', or an integer camera index; got {value!r}"
    )


def normalized_component(value: Any, *, field: str) -> str:
    text = str(value).strip()
    canonical = safe_archive_member(text, what=field)
    if "/" in canonical:
        raise ValueError(f"{field} must be one path component: {text!r}")
    return canonical


def frame_numbers(row: Mapping[str, Any], primary: str, alternative: str) -> list[int]:
    values = row.get(primary, row.get(alternative))
    if not isinstance(values, list) or not values:
        raise ValueError(f"Manifest row requires non-empty {primary}")
    output: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Boolean frame number in {primary}")
        number = int(value)
        if number < 0 or str(number) != str(value).strip():
            # Permit JSON integer 1 and string "1", but reject floats and signs/whitespace aliases.
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid frame number in {primary}: {value!r}")
        output.append(number)
    return output


def image_members_for_row(
    row: Mapping[str, Any], *, expected_images: int
) -> list[str]:
    explicit = row.get("image_members")
    if explicit is not None:
        if not isinstance(explicit, list) or not all(
            isinstance(value, str) for value in explicit
        ):
            raise ValueError("image_members must be a list of strings")
        members = [
            safe_archive_member(value, what="image_members entry") for value in explicit
        ]
    else:
        recording_value = row.get("recording", row.get("capture"))
        if recording_value is None:
            raise ValueError(
                "Manifest row needs image_members or recording/capture for path derivation"
            )
        recording = normalized_component(recording_value, field="recording/capture")
        pre_camera = normalized_camera(row.get("pre_camera"), field="pre_camera")
        post_camera = normalized_camera(row.get("post_camera"), field="post_camera")
        pre = frame_numbers(row, "pre_frame_numbers", "pre_frames")
        post = frame_numbers(row, "post_frame_numbers", "post_frames")
        members = [
            f"kinect_color/{recording}/{pre_camera}/frame_{number:05d}.jpg"
            for number in pre
        ] + [
            f"kinect_color/{recording}/{post_camera}/frame_{number:05d}.jpg"
            for number in post
        ]
    if expected_images and len(members) != expected_images:
        case_id = row.get("case_id", "<unknown>")
        raise ValueError(
            f"Case {case_id!r} has {len(members)} images, expected {expected_images}"
        )
    if len(set(members)) != len(members):
        raise ValueError(
            f"Case {row.get('case_id', '<unknown>')!r} contains duplicate image members"
        )
    for member in members:
        if PurePosixPath(member).suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError(f"Selected member is not a JPEG: {member}")
    return members


def prepare_manifest(
    rows: Sequence[Mapping[str, Any]], output_root: Path, expected_images: int
) -> tuple[list[dict[str, Any]], list[str]]:
    staged: list[dict[str, Any]] = []
    requested: set[str] = set()
    case_ids: set[str] = set()
    for index, original in enumerate(rows):
        row = dict(original)
        case_id = str(row.get("case_id", f"row_{index:06d}"))
        if case_id in case_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        case_ids.add(case_id)
        members = image_members_for_row(row, expected_images=expected_images)
        # The inference runner resolves these relative paths under --staged-root.
        # Absolute paths would make the manifest host-specific and fail its safety check.
        paths = list(members)
        row["image_members"] = members
        row["image_paths"] = paths
        row["image_paths_relative_to_stage_root"] = members
        staged.append(row)
        requested.update(members)
    return staged, sorted(requested)


def validate_jpeg(path: Path, *, max_bytes: int, max_pixels: int) -> dict[str, Any]:
    stat = path.stat()
    if stat.st_size <= 4:
        raise ValueError(f"JPEG is too small: {path} ({stat.st_size} bytes)")
    if stat.st_size > max_bytes:
        raise ValueError(f"JPEG exceeds size cap: {path} ({stat.st_size} > {max_bytes})")
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError(f"Missing JPEG SOI marker: {path}")
        handle.seek(-2, os.SEEK_END)
        if handle.read(2) != b"\xff\xd9":
            raise ValueError(f"Missing JPEG EOI marker: {path}")
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise ValueError(f"Expected JPEG, got {image.format!r}: {path}")
            width, height = image.size
            mode = image.mode
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ValueError(f"Unsafe JPEG dimensions {width}x{height}: {path}")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid/truncated JPEG: {path}: {exc}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
    return {
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(path),
        "width": int(width),
        "height": int(height),
        "mode": mode,
    }


def nearest_existing(path: Path) -> Path:
    current = path.absolute()
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(f"No existing ancestor for {path}")
        current = parent
    return current


def outer_member_info(outer: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(outer) as archive:
        matches = [info for info in archive.infolist() if info.filename == member]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one outer member {member!r}, found {len(matches)}"
        )
    info = matches[0]
    if info.flag_bits & 0x1:
        raise ValueError(f"Encrypted outer member is not supported: {member}")
    return {
        "entry": member,
        "uncompressed_size_bytes": int(info.file_size),
        "compressed_size_bytes": int(info.compress_size),
        "crc32": f"{info.CRC:08x}",
        "compression_method": int(info.compress_type),
    }


def archive_fingerprint(
    outer: Path, inner_info: Mapping[str, Any]
) -> dict[str, Any]:
    stat = outer.stat()
    return {
        "outer_path": str(outer.resolve()),
        "outer_size_bytes": int(stat.st_size),
        "outer_mtime_ns": int(stat.st_mtime_ns),
        "nested_entry": dict(inner_info),
    }


def load_provenance(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read provenance ledger {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError(f"Unexpected provenance schema in {path}")
    return value


def record_for_path(
    member: str, path: Path, *, max_bytes: int, max_pixels: int, source: str
) -> dict[str, Any]:
    details = validate_jpeg(path, max_bytes=max_bytes, max_pixels=max_pixels)
    return {
        "member": member,
        "path": str(path.resolve()),
        "source": source,
        **details,
    }


def records_from_previous(
    previous: Mapping[str, Any] | None,
    *,
    requested: Sequence[str],
    output_root: Path,
    manifest_sha256: str,
    fingerprint: Mapping[str, Any],
    max_bytes: int,
    max_pixels: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], bool]:
    """Return trusted previous records, unproven existing candidates, complete flag."""
    trusted: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    compatible = bool(
        previous
        and previous.get("manifest_sha256") == manifest_sha256
        and previous.get("archive_fingerprint") == fingerprint
    )
    previous_by_member = {
        str(record.get("member")): record
        for record in (previous or {}).get("images", [])
        if isinstance(record, dict) and record.get("member")
    }
    for member in requested:
        path = safe_output_path(output_root, member)
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Existing staging target is not a regular file: {path}")
        observed = record_for_path(
            member,
            path,
            max_bytes=max_bytes,
            max_pixels=max_pixels,
            source="existing_unproven",
        )
        old = previous_by_member.get(member)
        if compatible and old is not None:
            if (
                old.get("size_bytes") != observed["size_bytes"]
                or old.get("sha256") != observed["sha256"]
            ):
                raise ValueError(f"Existing staged file changed since provenance: {path}")
            observed["source"] = "reused_from_provenance"
            trusted[member] = observed
        else:
            candidates[member] = observed
    complete = bool(
        compatible
        and previous
        and previous.get("status") == "complete"
        and isinstance(previous.get("inner_zip_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", previous["inner_zip_sha256"])
        and isinstance(previous.get("inner_zip_size_bytes"), int)
        and int(previous["inner_zip_size_bytes"]) > 0
        and set(trusted) == set(requested)
    )
    return trusted, candidates, complete


def base_ledger(
    *,
    status: str,
    args: argparse.Namespace,
    manifest_sha256: str,
    fingerprint: Mapping[str, Any],
    requested_count: int,
    records: Mapping[str, Mapping[str, Any]],
    disk: Mapping[str, Any],
    inner_zip_sha256: str | None = None,
    inner_zip_size_bytes: int | None = None,
) -> dict[str, Any]:
    try:
        stream_unzip_version = importlib.metadata.version("stream-unzip")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        stream_unzip_version = "unknown"
    return {
        "schema_version": SCHEMA,
        "status": status,
        "updated_unix_seconds": time.time(),
        "manifests": list(args.manifest_inputs),
        "manifest_sha256": manifest_sha256,
        "output_root": str(args.output_root.resolve()),
        "provenance_output": str(args.provenance_output.resolve()),
        "staged_manifest_output": str(args.staged_manifest_output.resolve()),
        "archive_fingerprint": dict(fingerprint),
        "inner_zip_sha256": inner_zip_sha256,
        "inner_zip_size_bytes": inner_zip_size_bytes,
        "requested_unique_images": requested_count,
        "recorded_images": len(records),
        "limits": {
            "max_image_bytes": int(args.max_image_mib * 1024**2),
            "max_pixels": int(args.max_pixels),
            "expected_images_per_case": int(args.expected_images_per_case),
        },
        "disk_preflight": dict(disk),
        "software": {
            "python": sys.version.split()[0],
            "stream_unzip": stream_unzip_version,
            "pillow": importlib.metadata.version("Pillow"),
        },
        "images": [dict(records[key]) for key in sorted(records)],
    }


def write_selected_member(
    *,
    member: str,
    declared_size: int,
    chunks: Iterable[bytes],
    target: Path,
    output_root: Path,
    max_bytes: int,
    max_pixels: int,
) -> dict[str, Any]:
    if declared_size <= 4 or declared_size > max_bytes:
        # Drain before raising so stream-unzip can still perform integrity checks.
        for _ in chunks:
            pass
        raise ValueError(
            f"Archive JPEG size is unsafe for {member}: {declared_size} bytes"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_path(target.parent, output_root.resolve())
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging target: {target}")
    partial = target.with_name(
        f".{target.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
    )
    count = 0
    digest = hashlib.sha256()
    try:
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as handle:
            for chunk in chunks:
                count += len(chunk)
                if count > declared_size or count > max_bytes:
                    raise ValueError(f"JPEG stream exceeds declared/capped size: {member}")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if count != declared_size:
            raise ValueError(
                f"JPEG size mismatch for {member}: streamed={count}, declared={declared_size}"
            )
        details = validate_jpeg(partial, max_bytes=max_bytes, max_pixels=max_pixels)
        if details["sha256"] != digest.hexdigest():
            raise ValueError(f"Internal SHA256 mismatch while staging {member}")
        os.replace(partial, target)
        fsync_directory(target.parent)
        return {
            "member": member,
            "path": str(target.resolve()),
            "source": "streamed_from_archive",
            **details,
        }
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def verify_candidate_against_stream(
    *,
    member: str,
    declared_size: int,
    chunks: Iterable[bytes],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for chunk in chunks:
        count += len(chunk)
        digest.update(chunk)
    if count != declared_size:
        raise ValueError(
            f"Archive size mismatch for existing candidate {member}: {count} != {declared_size}"
        )
    if count != candidate.get("size_bytes") or digest.hexdigest() != candidate.get(
        "sha256"
    ):
        raise ValueError(
            f"Existing file differs from archive member; refusing overwrite: {candidate.get('path')}"
        )
    record = dict(candidate)
    record["source"] = "adopted_after_archive_verification"
    return record


def stream_stage(
    *,
    args: argparse.Namespace,
    requested: Sequence[str],
    records: dict[str, dict[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    manifest_sha256: str,
    fingerprint: Mapping[str, Any],
    disk: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], str, int]:
    desired = set(requested)
    seen: set[str] = set()
    inner_zip_name = safe_archive_member(
        args.inner_zip_entry, what="outer nested ZIP member"
    )
    max_bytes = int(args.max_image_mib * 1024**2)
    file_chunk_size = int(args.io_chunk_mib * 1024**2)
    stream_chunk_size = int(args.stream_chunk_mib * 1024**2)
    found_inner = False
    checkpoint_count = 0

    outer_files = stream_unzip(
        file_chunks(args.outer, file_chunk_size), chunk_size=stream_chunk_size
    )
    for raw_outer_name, outer_size, outer_chunks in outer_files:
        outer_name = decode_member(raw_outer_name, what="outer ZIP member")
        if outer_name != inner_zip_name:
            for _ in outer_chunks:
                pass
            continue
        if found_inner:
            raise ValueError(f"Duplicate outer nested ZIP member: {inner_zip_name}")
        found_inner = True
        hashing_inner = HashingChunks(outer_chunks)
        inner_files = stream_unzip(hashing_inner, chunk_size=stream_chunk_size)
        for raw_member, declared_size, member_chunks in inner_files:
            member = decode_member(raw_member, what="inner ZIP member")
            if member in seen:
                for _ in member_chunks:
                    pass
                if member in desired:
                    raise ValueError(f"Duplicate requested member in inner ZIP: {member}")
                continue
            seen.add(member)
            if member not in desired:
                for _ in member_chunks:
                    pass
                continue
            target = safe_output_path(args.output_root, member)
            if member in records:
                # Same archive fingerprint and a rehashed local file make this safe to reuse.
                for _ in member_chunks:
                    pass
            elif member in candidates:
                records[member] = verify_candidate_against_stream(
                    member=member,
                    declared_size=int(declared_size),
                    chunks=member_chunks,
                    candidate=candidates[member],
                )
            else:
                records[member] = write_selected_member(
                    member=member,
                    declared_size=int(declared_size),
                    chunks=member_chunks,
                    target=target,
                    output_root=args.output_root,
                    max_bytes=max_bytes,
                    max_pixels=int(args.max_pixels),
                )
            checkpoint_count += 1
            if checkpoint_count % args.checkpoint_every == 0:
                atomic_json(
                    args.provenance_output,
                    base_ledger(
                        status="in_progress",
                        args=args,
                        manifest_sha256=manifest_sha256,
                        fingerprint=fingerprint,
                        requested_count=len(requested),
                        records=records,
                        disk=disk,
                    ),
                )
                print(
                    f"staged/verified {len(records)}/{len(requested)} unique images",
                    file=sys.stderr,
                    flush=True,
                )
        # stream-unzip consumes through the central directory, but explicitly drain
        # to hash every byte of the nested ZIP and finish the outer member CRC check.
        hashing_inner.drain()
        if hashing_inner.size != int(outer_size):
            raise ValueError(
                f"Nested ZIP size mismatch: streamed={hashing_inner.size}, declared={outer_size}"
            )
        inner_sha256 = hashing_inner.hexdigest
        inner_size = hashing_inner.size
        break
    if not found_inner:
        raise FileNotFoundError(f"Outer ZIP does not contain {inner_zip_name!r}")
    missing = sorted(desired - seen)
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Inner ZIP misses {len(missing)} requested images; first entries:\n{preview}"
        )
    if set(records) != desired:
        unresolved = sorted(desired - set(records))
        raise RuntimeError(f"Internal staging error; unresolved members: {unresolved[:20]}")
    return records, inner_sha256, inner_size


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.outer = args.outer.resolve()
    args.manifest = [path.resolve() for path in args.manifest]
    args.output_root = args.output_root.resolve()
    args.provenance_output = (
        args.provenance_output.resolve()
        if args.provenance_output
        else args.output_root / "stage_images.provenance.json"
    )
    args.staged_manifest_output = (
        args.staged_manifest_output.resolve()
        if args.staged_manifest_output
        else args.output_root / "runtime_manifest.staged.jsonl"
    )
    if not args.outer.is_file() or any(not path.is_file() for path in args.manifest):
        raise FileNotFoundError("--outer and every --manifest must be existing regular files")
    if len(set(args.manifest)) != len(args.manifest):
        raise ValueError("The same manifest path was supplied more than once")
    if args.outer in args.manifest:
        raise ValueError("Outer archive and manifest cannot be the same file")
    if args.provenance_output == args.staged_manifest_output:
        raise ValueError(
            "--provenance-output and --staged-manifest-output must be different files"
        )
    protected_inputs = {args.outer, *args.manifest}
    output_collisions = protected_inputs.intersection(
        {args.provenance_output, args.staged_manifest_output}
    )
    if output_collisions:
        raise ValueError(
            "Staging metadata outputs may not overwrite archive/manifest inputs: "
            + ", ".join(str(path) for path in sorted(output_collisions))
        )
    safe_archive_member(args.inner_zip_entry, what="outer nested ZIP member")

    rows: list[dict[str, Any]] = []
    args.manifest_inputs = []
    for manifest_path in args.manifest:
        manifest_rows, manifest_raw = read_manifest(manifest_path)
        rows.extend(manifest_rows)
        args.manifest_inputs.append(
            {
                "path": str(manifest_path),
                "sha256": sha256_bytes(manifest_raw),
                "row_count": len(manifest_rows),
            }
        )
    staged_rows, requested = prepare_manifest(
        rows, args.output_root, args.expected_images_per_case
    )
    manifest_sha256 = combined_manifest_sha256(args.manifest_inputs)
    inner_info = outer_member_info(args.outer, args.inner_zip_entry)
    fingerprint = archive_fingerprint(args.outer, inner_info)
    previous = load_provenance(args.provenance_output)
    max_bytes = int(args.max_image_mib * 1024**2)
    trusted, candidates, complete = records_from_previous(
        previous,
        requested=requested,
        output_root=args.output_root,
        manifest_sha256=manifest_sha256,
        fingerprint=fingerprint,
        max_bytes=max_bytes,
        max_pixels=int(args.max_pixels),
    )
    missing_or_unproven = len(requested) - len(trusted)
    estimated_required = int(
        missing_or_unproven * args.estimated_image_mib * 1024**2
        + args.max_image_mib * 1024**2
        + max(16 * 1024**2, len(requested) * 1024)
    )
    usage = shutil.disk_usage(nearest_existing(args.output_root))
    reserve = int(args.reserve_gib * 1024**3)
    disk = {
        "free_bytes": int(usage.free),
        "reserve_bytes": reserve,
        "estimated_required_bytes": estimated_required,
        "estimated_image_bytes": int(args.estimated_image_mib * 1024**2),
        "missing_or_unproven_images": missing_or_unproven,
        "passes": bool(usage.free - reserve >= estimated_required),
        "note": "Budget uses --estimated-image-mib; every streamed JPEG is separately capped by --max-image-mib.",
    }
    summary = {
        "status": "dry_run" if args.dry_run else ("reusable" if complete else "ready"),
        "outer": str(args.outer),
        "nested_entry": inner_info,
        "manifests": list(args.manifest_inputs),
        "manifest_sha256": manifest_sha256,
        "case_count": len(staged_rows),
        "requested_image_references": sum(
            len(row["image_members"]) for row in staged_rows
        ),
        "requested_unique_images": len(requested),
        "trusted_existing_images": len(trusted),
        "unproven_existing_images": len(candidates),
        "disk_preflight": disk,
        "output_root": str(args.output_root),
        "provenance_output": str(args.provenance_output),
        "staged_manifest_output": str(args.staged_manifest_output),
    }
    if not disk["passes"]:
        raise OSError(
            "Insufficient staging budget: "
            f"free={usage.free}, reserve={reserve}, estimated_required={estimated_required}. "
            "Adjust --estimated-image-mib/--reserve-gib only after checking the target filesystem."
        )
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_path(args.output_root, args.output_root.resolve())
    if complete:
        atomic_jsonl(args.staged_manifest_output, staged_rows)
        final_ledger = base_ledger(
            status="complete",
            args=args,
            manifest_sha256=manifest_sha256,
            fingerprint=fingerprint,
            requested_count=len(requested),
            records=trusted,
            disk=disk,
            inner_zip_sha256=str(previous["inner_zip_sha256"]),
            inner_zip_size_bytes=int(previous["inner_zip_size_bytes"]),
        )
        final_ledger["staged_manifest_sha256"] = sha256_file(
            args.staged_manifest_output
        )
        atomic_json(args.provenance_output, final_ledger)
        summary.update(
            {
                "status": "reused_complete",
                "recorded_images": len(trusted),
                "inner_zip_size_bytes": int(previous["inner_zip_size_bytes"]),
                "inner_zip_sha256": str(previous["inner_zip_sha256"]),
                "staged_manifest_sha256": final_ledger[
                    "staged_manifest_sha256"
                ],
            }
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    atomic_json(
        args.provenance_output,
        base_ledger(
            status="in_progress",
            args=args,
            manifest_sha256=manifest_sha256,
            fingerprint=fingerprint,
            requested_count=len(requested),
            records=trusted,
            disk=disk,
        ),
    )
    records, inner_sha256, inner_size = stream_stage(
        args=args,
        requested=requested,
        records=trusted,
        candidates=candidates,
        manifest_sha256=manifest_sha256,
        fingerprint=fingerprint,
        disk=disk,
    )
    final_ledger = base_ledger(
        status="complete",
        args=args,
        manifest_sha256=manifest_sha256,
        fingerprint=fingerprint,
        requested_count=len(requested),
        records=records,
        disk=disk,
        inner_zip_sha256=inner_sha256,
        inner_zip_size_bytes=inner_size,
    )
    atomic_jsonl(args.staged_manifest_output, staged_rows)
    final_ledger["staged_manifest_sha256"] = sha256_file(args.staged_manifest_output)
    atomic_json(args.provenance_output, final_ledger)
    summary.update(
        {
            "status": "complete",
            "recorded_images": len(records),
            "inner_zip_size_bytes": inner_size,
            "inner_zip_sha256": inner_sha256,
            "staged_manifest_sha256": final_ledger["staged_manifest_sha256"],
        }
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
