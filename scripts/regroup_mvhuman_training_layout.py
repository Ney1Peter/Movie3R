#!/usr/bin/env python3
"""Move converted MVHuman camera folders from flat to grouped layout.

Old:
  data/Training/mvhuman/100001_CC32871A004/{rgb,mask,cam,smpl}

New:
  data/Training/mvhuman/100001/CC32871A004/{rgb,mask,cam,smpl}
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


FLAT_NAME_RE = re.compile(r"^(?P<subject>\d{6})_(?P<camera>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/Training/mvhuman"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_sequence_dir(path: Path) -> bool:
    return all((path / sub).is_dir() for sub in ("rgb", "cam", "smpl"))


def main() -> None:
    args = parse_args()
    root = args.root
    moves: list[tuple[Path, Path]] = []
    skipped = []

    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        match = FLAT_NAME_RE.match(path.name)
        if match is None or not is_sequence_dir(path):
            skipped.append(path.name)
            continue
        subject = match.group("subject")
        camera = match.group("camera")
        dst = root / subject / camera
        moves.append((path, dst))

    print(f"root: {root}")
    print(f"flat sequence dirs: {len(moves)}")
    print(f"skipped dirs: {len(skipped)}")
    for src, dst in moves[:10]:
        print(f"  {src.name} -> {dst.relative_to(root)}")
    if len(moves) > 10:
        print(f"  ... {len(moves) - 10} more")

    if args.dry_run:
        return

    for src, dst in moves:
        if dst.exists():
            raise FileExistsError(f"Target already exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
    print(f"moved: {len(moves)}")


if __name__ == "__main__":
    main()
