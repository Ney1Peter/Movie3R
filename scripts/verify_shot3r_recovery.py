#!/usr/bin/env python3
"""Verify the minimal Shot3R recovery snapshot and optional large weights."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "publication" / "shot3r_reproducibility_20260916"

SMALL_ARTIFACTS = {
    SNAPSHOT / "weights" / "SELECTED_MODEL.pt": (
        24_910,
        "cb84b0da620878515e94f08b30d757206b41c4de82e2ff4091fe2a6e519e498f",
    ),
    SNAPSHOT / "results" / "frozen_internal" / "egobody_cs150_summary.json": (
        122_471,
        "91f0830b7859d678ae4d1afd3d21b053fa6db073b60416f5c7fdcdf1eea26a18",
    ),
    SNAPSHOT / "results" / "frozen_internal" / "egohumans_cs100_summary.json": (
        79_290,
        "2515b3d9dc4f77e4358be3c99193a50cdd3d8b1e83909ea3dbfdbd6666e436fc",
    ),
    SNAPSHOT / "results" / "frozen_internal" / "harmony4d_cs150_summary.json": (
        90_335,
        "51ebde2f0ca2d70e54bd3ba948ab4ea47bc2821aabc6dd34b4ec34728520b7da",
    ),
}

LARGE_WEIGHTS = {
    REPO
    / "output/v14_cut_first_cross_source/"
    "v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth": (
        4_930_639_378,
        "de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265",
    ),
    REPO / "checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth": (
        4_831_184_406,
        "3fb2799420f7fd3caa63a47c9cde73090a6f93383520363484eb5158e446fceb",
    ),
    REPO / "src/human3r_896L.pth": (
        4_670_554_642,
        "1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_size: int, expected_hash: str, hash_file: bool) -> bool:
    try:
        label = path.relative_to(REPO)
    except ValueError:
        label = path
    if not path.is_file():
        print(f"MISSING  {label}")
        return False
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        print(f"BAD SIZE {label}: {actual_size} != {expected_size}")
        return False
    if hash_file:
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            print(f"BAD HASH {label}: {actual_hash} != {expected_hash}")
            return False
    print(f"OK       {label}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hash-large-weights",
        action="store_true",
        help="also hash the three 4-5 GB checkpoints (slow and I/O intensive)",
    )
    args = parser.parse_args()

    ok = True
    for path, (size, digest) in SMALL_ARTIFACTS.items():
        ok &= verify(path, size, digest, hash_file=True)
    for path, (size, digest) in LARGE_WEIGHTS.items():
        if path.exists():
            ok &= verify(path, size, digest, hash_file=args.hash_large_weights)
        else:
            print(f"OPTIONAL MISSING {path.relative_to(REPO)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
