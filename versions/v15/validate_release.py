#!/usr/bin/env python3
"""Validate that Movie3R-v15 frozen assets and contracts are reproducible."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPEC = HERE / "FINAL_RUNTIME_SPEC.json"
TEMPLATE = HERE / "BATCH_MANIFEST_TEMPLATE.jsonl"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_repo(relative: str) -> Path:
    value = Path(relative)
    path = value.resolve() if value.is_absolute() else (ROOT / value).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Asset escapes repository: {relative} -> {path}")
    return path


def validate(check_hash: bool = True) -> dict[str, Any]:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    errors: list[str] = []
    checked: list[dict[str, Any]] = []

    if spec.get("release") != "Movie3R-v15-final":
        errors.append("unexpected release name")
    if spec.get("output_policy", {}).get("device") != "cpu":
        errors.append("runtime device is not frozen to CPU")
    if spec.get("output_policy", {}).get("gt_in_runtime") is not False:
        errors.append("runtime GT contract is not false")

    def check_asset(label: str, path_value: str, expected: str | None) -> None:
        try:
            path = resolve_repo(path_value)
        except Exception as exc:  # pragma: no cover - defensive reporting
            errors.append(f"{label}: {exc}")
            return
        if not path.is_file():
            errors.append(f"{label}: missing {path}")
            return
        actual = digest(path) if check_hash and expected else None
        if expected and actual is not None and actual != expected:
            errors.append(f"{label}: sha256 mismatch expected={expected} actual={actual}")
        checked.append({"label": label, "path": str(path), "sha256": actual or expected})

    for name, item in spec.get("checkpoints", {}).items():
        check_asset(f"checkpoint:{name}", item["path"], item.get("sha256"))
    detector = spec.get("learned_components", {}).get("shot_detector", {})
    check_asset("detector:artifact", detector["artifact"], detector.get("sha256"))
    for name, item in spec.get("geometry_modules", {}).items():
        implementation = item.get("implementation")
        expected = item.get("implementation_sha256")
        if implementation and expected:
            check_asset(f"module:{name}", implementation, expected)
    entrypoints = spec.get("runtime_entrypoints", {})
    entrypoint_hashes = entrypoints.get("sha256", {})
    for name, expected in entrypoint_hashes.items():
        implementation = entrypoints.get(name)
        if implementation and expected:
            check_asset(f"entrypoint:{name}", implementation, expected)

    required = {"case_id", "sequence", "frame", "pre_camera", "post_camera"}
    allowed_sequences = set(spec["case_contract"]["sequence_choices"])
    for line_no, text in enumerate(TEMPLATE.read_text(encoding="utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(f"manifest line {line_no}: {exc}")
            continue
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"manifest line {line_no}: missing {missing}")
        if row.get("sequence") not in allowed_sequences:
            errors.append(f"manifest line {line_no}: invalid sequence {row.get('sequence')!r}")
        if int(row.get("pre_frames", 5)) < 1 or int(row.get("post_frames", 25)) < 3:
            errors.append(f"manifest line {line_no}: invalid frame window")

    result = {"release": spec.get("release"), "check_hash": check_hash, "checked": checked, "errors": errors, "valid": not errors}
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-checkpoint-hash", action="store_true", help="Check existence without reading multi-GB checkpoints")
    args = p.parse_args()
    result = validate(check_hash=not args.skip_checkpoint_hash)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
