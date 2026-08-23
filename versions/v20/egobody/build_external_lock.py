#!/usr/bin/env python3
"""Build a detached, immutable content lock for the EgoBody CS150 protocol.

This is deliberately an *external* lock: it inventories the inputs and source
bytes that define the experiment, but it was not part of
``protocol_state.run_identity``.  The implementation never parses manifests,
ground truth, checkpoints, or result payloads.  It only obtains filesystem
metadata and streams file bytes through SHA-256.

The canonical JSON deliberately has no creation timestamp.  Given identical
files, mtimes, Git state, and environment metadata, rebuilding it yields the
same bytes.  Existing lock artifacts are accepted only when byte-identical;
they are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "Bridge3R-EgoBody-CS150-external-lock-v1"
PROTOCOL = "Bridge3R-EgoBody-CS150-v1"

MANIFEST_FILENAMES = (
    "egobody_cs150_development.runtime.jsonl",
    "egobody_cs150_development.evaluator.jsonl",
    "egobody_cs150_development.spec.json",
    "egobody_cs150_holdout.runtime.jsonl",
    "egobody_cs150_holdout.evaluator.jsonl",
    "egobody_cs150_holdout.spec.json",
    "egobody_cs150_holdout.filtered.runtime.jsonl",
    "egobody_cs150_holdout.filtered.evaluator.jsonl",
    "egobody_cs150_holdout.filtered.exclusions.json",
    "egobody_cs150_test.runtime.jsonl",
    "egobody_cs150_test.evaluator.jsonl",
    "egobody_cs150_test.spec.json",
    "egobody_cs150_manifest_index.json",
)

STAGING_FILENAMES = (
    "stage_images.filtered.provenance.json",
    "runtime_manifest.filtered.staged.jsonl",
)

CHECKPOINT_PATHS = (
    "output/v14_cut_first_cross_source/"
    "v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth",
    "src/human3r_896L.pth",
    "output/v14/detector_learning_audit/SELECTED_MODEL.pt",
)

TOPOLOGY_ASSET_PATHS = (
    "src/models/smplx/smplx2smpl.pkl",
    "src/models/smpl/SMPL_NEUTRAL.pkl",
    "src/models/smplx/SMPLX_NEUTRAL.npz",
    "src/models/smplx/SMPLX_MALE.npz",
    "src/models/smplx/SMPLX_FEMALE.npz",
)

DIRECT_DEPENDENCY_PATHS = (
    "versions/v13/gt_id_consensus.py",
    "versions/v14/causal_image_detector.py",
    "versions/v14/run_v14_2_single_sequence.py",
    "versions/v14/b0_person_triangulation.py",
    "versions/v14/eval_streaming_within_shot_stability.py",
    "versions/v14/probe_p1_foot_scene_observability.py",
    "versions/v15/harmony4d/run_harmony_case.py",
    "versions/v15/harmony4d/evaluate_harmony.py",
    "versions/v15/harmony4d/dataset.py",
    "versions/v15/harmony4d/topology.py",
    "versions/v16/harmony4d/causal_stabilization.py",
    "versions/v16/harmony4d/probe_causal_stabilization.py",
    "versions/v19/egohumans/causal_identity.py",
    "versions/v19/egohumans/joint_correction.py",
    "scripts/v10_detector_feature_probe.py",
    "src/dust3r/adaptive_joint.py",
    "scripts/v10_image_only_detector.py",
    "versions/v15/FINAL_RUNTIME_SPEC.json",
)

ENVIRONMENT_PACKAGES = (
    "numpy",
    "opencv-python",
    "Pillow",
    "scipy",
    "smplx",
    "torch",
    "torchvision",
    "trimesh",
)

V20_SOURCE_SUFFIXES = (
    ".py",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
)
SOURCE_CACHE_DIRECTORIES = frozenset({"__pycache__", ".pytest_cache"})


@dataclass(frozen=True)
class Layout:
    """Physical roots used to construct workspace-relative inventory paths."""

    workspace_root: Path
    project_root: Path
    work_root: Path
    gt_cache_root: Path
    staged_root: Path
    manifest_root: Path
    egobody_source_root: Path
    dust3r_source_root: Path

    @classmethod
    def from_roots(
        cls,
        workspace_root: Path,
        *,
        project_root: Path | None = None,
        work_root: Path | None = None,
        gt_cache_root: Path | None = None,
        staged_root: Path | None = None,
        manifest_root: Path | None = None,
        egobody_source_root: Path | None = None,
        dust3r_source_root: Path | None = None,
    ) -> "Layout":
        workspace = Path(workspace_root)
        project = Path(project_root) if project_root else workspace / "Movie3R"
        work = (
            Path(work_root)
            if work_root
            else workspace / "data" / "EgoBody_work_v20"
        )
        return cls(
            workspace_root=workspace,
            project_root=project,
            work_root=work,
            gt_cache_root=(
                Path(gt_cache_root) if gt_cache_root else work / "gt_cache"
            ),
            staged_root=(
                Path(staged_root) if staged_root else work / "staged_rgb"
            ),
            manifest_root=(
                Path(manifest_root) if manifest_root else work / "manifests"
            ),
            egobody_source_root=(
                Path(egobody_source_root)
                if egobody_source_root
                else project / "versions" / "v20" / "egobody"
            ),
            dust3r_source_root=(
                Path(dust3r_source_root)
                if dust3r_source_root
                else project / "src" / "dust3r"
            ),
        )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical byte representation used by this lock."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Stream a file through SHA-256 without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _workspace_relative(path: Path, workspace_root: Path) -> str:
    """Return a stable POSIX path and reject escapes or symlink components."""

    root = _lexical_absolute(workspace_root)
    target = _lexical_absolute(path)
    if root.is_symlink():
        raise ValueError(f"Workspace root may not be a symlink: {root}")
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace root: {target} (root {root})") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlinks are not allowed in lock inputs: {current}")
    if not relative.parts:
        raise ValueError("The workspace directory itself is not an inventory file")
    return relative.as_posix()


def _stable_file_digest(path: Path) -> tuple[os.stat_result, str]:
    """Hash one regular file and fail if its identity/metadata changes mid-read."""

    target = Path(path)
    before_path = target.lstat()
    if stat.S_ISLNK(before_path.st_mode):
        raise ValueError(f"Symlink inputs are forbidden: {target}")
    if not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"Expected a regular file: {target}")

    digest = hashlib.sha256()
    with target.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    after_path = target.lstat()
    identity = lambda value: (  # noqa: E731 - compact immutable comparison tuple
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before_path) != identity(before) or identity(before) != identity(after):
        raise RuntimeError(f"File changed while it was being inventoried: {target}")
    if identity(after) != identity(after_path):
        raise RuntimeError(f"File was replaced while it was being inventoried: {target}")
    return after, digest.hexdigest()


def file_entry(path: Path, workspace_root: Path) -> dict[str, Any]:
    """Inventory one file as the required path/bytes/mtime/hash tuple."""

    stable_path = _workspace_relative(path, workspace_root)
    metadata, digest = _stable_file_digest(path)
    return {
        "path": stable_path,
        "bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "sha256": digest,
    }


def inventory_group(
    paths: Iterable[Path],
    workspace_root: Path,
    *,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Inventory a non-empty group of unique regular files."""

    materialized = list(paths)
    if not materialized:
        raise FileNotFoundError(f"Inventory selection matched no files: {selection}")
    entries = [file_entry(path, workspace_root) for path in materialized]
    entries.sort(key=lambda entry: entry["path"])
    stable_paths = [str(entry["path"]) for entry in entries]
    if len(stable_paths) != len(set(stable_paths)):
        raise ValueError(f"Duplicate paths in inventory group: {stable_paths}")
    return {
        "selection": dict(selection),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "entries": entries,
    }


def tree_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    """Hash sorted canonical ``(path, bytes, mtime_ns, sha256)`` tuples."""

    tuples: list[list[Any]] = []
    seen: set[str] = set()
    for entry in entries:
        path = str(entry["path"])
        if path in seen:
            raise ValueError(f"Duplicate path in canonical tree: {path}")
        seen.add(path)
        digest = str(entry["sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid SHA-256 for {path}: {digest!r}")
        tuples.append(
            [path, int(entry["bytes"]), int(entry["mtime_ns"]), digest]
        )
    tuples.sort(key=lambda value: value[0])
    return hashlib.sha256(canonical_json_bytes(tuples)).hexdigest()


def _require_file(path: Path) -> Path:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    if target.is_symlink():
        raise ValueError(f"Symlink inputs are forbidden: {target}")
    if not target.is_file():
        raise ValueError(f"Expected a regular file: {target}")
    return target


def _recursive_files(
    root: Path,
    *,
    suffixes: Iterable[str] | None = None,
    exclude: Iterable[Path] = (),
    excluded_directory_names: Iterable[str] = (),
    exclude_temporary: bool = False,
) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    excluded = {_lexical_absolute(path) for path in exclude}
    allowed_suffixes = (
        {suffix.lower() for suffix in suffixes} if suffixes is not None else None
    )
    excluded_directories = set(excluded_directory_names)
    output: list[Path] = []
    for path in sorted(directory.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(directory)
        if any(part in excluded_directories for part in relative.parts[:-1]):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are forbidden in recursive source groups: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Non-regular recursive source entry: {path}")
        if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
            continue
        if exclude_temporary and _is_temporary_source_file(relative):
            continue
        if _lexical_absolute(path) in excluded:
            continue
        output.append(path)
    if not output:
        raise FileNotFoundError(f"Recursive inventory matched no files under {directory}")
    return output


def _is_temporary_source_file(path: Path) -> bool:
    """Identify editor/checkpoint/partial names even when they end in ``.py``."""

    directory_names = [part.lower() for part in path.parts[:-1]]
    if any(
        part in {"tmp", ".tmp", "temp", ".temp", "partial", ".partial"}
        or part.endswith((".tmp", ".temp", ".partial"))
        for part in directory_names
    ):
        return True
    name = path.name.lower()
    return (
        name.endswith("~")
        or name.startswith((".#", "#"))
        or name.endswith((".tmp", ".temp", ".partial", ".swp", ".swo"))
        or any(marker in name for marker in (".tmp.", ".temp.", ".partial."))
    )


def collect_git_metadata(project_root: Path, workspace_root: Path) -> dict[str, Any]:
    """Collect Git identity/status without requiring a clean worktree."""

    project = Path(project_root)

    def git(*arguments: str, binary: bool = False) -> str | bytes:
        completed = subprocess.run(
            ["git", "-C", os.fspath(project), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if binary:
            return completed.stdout
        return completed.stdout.decode("utf-8", errors="replace").strip()

    repository_root = Path(str(git("rev-parse", "--show-toplevel")))
    raw_status = bytes(
        git(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            binary=True,
        )
    )
    status_entries = [
        value.decode("utf-8", errors="replace")
        for value in raw_status.split(b"\0")
        if value
    ]
    return {
        "repository_root": _workspace_relative(repository_root, workspace_root),
        "head_commit": str(git("rev-parse", "HEAD")),
        "head_tree": str(git("rev-parse", "HEAD^{tree}")),
        "status_format": "git-status-porcelain-v1-z decoded as UTF-8 with replacement",
        "status_porcelain": status_entries,
        "worktree_clean": not status_entries,
        "tracked_clean": not any(not row.startswith("?? ") for row in status_entries),
        "untracked_count": sum(row.startswith("?? ") for row in status_entries),
    }


def collect_environment_metadata() -> dict[str, Any]:
    """Collect reproducibility versions without reading environment variables."""

    packages: dict[str, str | None] = {}
    for distribution in ENVIRONMENT_PACKAGES:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None

    torch_runtime: dict[str, Any] = {"importable": False}
    try:
        import torch  # type: ignore

        torch_runtime = {
            "importable": True,
            "version": str(torch.__version__),
            "compiled_cuda": str(torch.version.cuda) if torch.version.cuda else None,
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                and torch.backends.cudnn.version() is not None
                else None
            ),
        }
    except (ImportError, OSError) as exc:
        torch_runtime = {
            "importable": False,
            "error_type": type(exc).__name__,
        }

    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "torch_runtime": torch_runtime,
    }


def _relative_root(path: Path, workspace_root: Path) -> str:
    target = _lexical_absolute(path)
    if not target.is_dir():
        raise FileNotFoundError(target)
    return _workspace_relative(target, workspace_root)


def build_lock_payload(
    layout: Layout,
    *,
    git_metadata: Mapping[str, Any] | None = None,
    environment_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the in-memory lock without writing output or parsing any input."""

    workspace = Path(layout.workspace_root)
    if not workspace.is_dir():
        raise FileNotFoundError(workspace)

    project = Path(layout.project_root)
    work = Path(layout.work_root)
    gt_root = Path(layout.gt_cache_root)
    staged_root = Path(layout.staged_root)
    manifest_root = Path(layout.manifest_root)
    egobody_root = Path(layout.egobody_source_root)
    dust3r_root = Path(layout.dust3r_source_root)
    for root in (project, work, gt_root, staged_root, manifest_root, egobody_root, dust3r_root):
        _relative_root(root, workspace)

    gt_npz = sorted(gt_root.glob("*.gt.npz"), key=lambda path: path.name)
    gt_json = sorted(gt_root.glob("*.gt.json"), key=lambda path: path.name)
    if not gt_npz:
        raise FileNotFoundError(f"No *.gt.npz files in {gt_root}")
    if not gt_json:
        raise FileNotFoundError(f"No *.gt.json files in {gt_root}")

    direct_dependency_files = [
        _require_file(project / relative) for relative in DIRECT_DEPENDENCY_PATHS
    ]
    adaptive_joint = project / "src" / "dust3r" / "adaptive_joint.py"

    groups = {
        "checkpoints": inventory_group(
            [_require_file(project / relative) for relative in CHECKPOINT_PATHS],
            workspace,
            selection={
                "kind": "exact_project_paths",
                "paths": list(CHECKPOINT_PATHS),
            },
        ),
        "direct_project_dependencies": inventory_group(
            direct_dependency_files,
            workspace,
            selection={
                "kind": "exact_project_paths",
                "paths": list(DIRECT_DEPENDENCY_PATHS),
            },
        ),
        "dust3r_python_source_closure": inventory_group(
            _recursive_files(
                dust3r_root,
                suffixes=(".py",),
                exclude=(adaptive_joint,),
                excluded_directory_names=SOURCE_CACHE_DIRECTORIES,
                exclude_temporary=True,
            ),
            workspace,
            selection={
                "kind": "recursive_source_group",
                "root": _relative_root(dust3r_root, workspace),
                "include": ["**/*.py"],
                "exclude": [
                    _workspace_relative(adaptive_joint, workspace)
                    + " (already in direct_project_dependencies)"
                ],
            },
        ),
        "egobody_v20_implementation": inventory_group(
            _recursive_files(
                egobody_root,
                suffixes=V20_SOURCE_SUFFIXES,
                excluded_directory_names=SOURCE_CACHE_DIRECTORIES,
                exclude_temporary=True,
            ),
            workspace,
            selection={
                "kind": "recursive_source_and_config_suffixes",
                "root": _relative_root(egobody_root, workspace),
                "include_suffixes": list(V20_SOURCE_SUFFIXES),
                "excluded_directory_names": sorted(SOURCE_CACHE_DIRECTORIES),
                "temporary_file_policy": (
                    "exclude editor backups and names containing .tmp., .temp., "
                    "or .partial.; generated .pyc/.pyo are outside the suffix whitelist"
                ),
            },
        ),
        "gt_cache": inventory_group(
            [*gt_npz, *gt_json],
            workspace,
            selection={
                "kind": "non_recursive_globs",
                "root": _relative_root(gt_root, workspace),
                "patterns": ["*.gt.npz", "*.gt.json"],
            },
        ),
        "manifests": inventory_group(
            [_require_file(manifest_root / name) for name in MANIFEST_FILENAMES],
            workspace,
            selection={
                "kind": "exact_root_filenames",
                "root": _relative_root(manifest_root, workspace),
                "filenames": list(MANIFEST_FILENAMES),
            },
        ),
        "normative_staging_metadata": inventory_group(
            [_require_file(staged_root / name) for name in STAGING_FILENAMES],
            workspace,
            selection={
                "kind": "exact_root_filenames",
                "root": _relative_root(staged_root, workspace),
                "filenames": list(STAGING_FILENAMES),
                "note": (
                    "Only the completed filtered staging provenance and its "
                    "filtered staged runtime manifest are normative."
                ),
            },
        ),
        "topology_and_smpl_assets": inventory_group(
            [_require_file(project / relative) for relative in TOPOLOGY_ASSET_PATHS],
            workspace,
            selection={
                "kind": "exact_project_paths",
                "paths": list(TOPOLOGY_ASSET_PATHS),
            },
        ),
    }

    flattened: list[Mapping[str, Any]] = []
    for name in sorted(groups):
        flattened.extend(groups[name]["entries"])
    root_digest = tree_sha256(flattened)

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "lock_scope": "external-content-inventory",
        "entered_protocol_run_identity": False,
        "protocol_run_identity_statement": (
            "This external lock was created outside the protocol execution and "
            "was not included in protocol_state.run_identity."
        ),
        "content_handling_statement": (
            "Inputs were never parsed for metrics or semantic values; the builder "
            "only inventoried structural filesystem metadata and SHA-256 byte hashes."
        ),
        "provenance_timing": {
            "development": "retrospective",
            "holdout": "pre_run",
            "test": "pre_run",
        },
        "path_policy": {
            "workspace_root": ".",
            "encoding": "POSIX paths relative to the supplied workspace root",
            "absolute_workspace_root_embedded": False,
            "project_root": _relative_root(project, workspace),
            "external_work_root": _relative_root(work, workspace),
            "symlinks_allowed": False,
        },
        "canonicalization": {
            "json": (
                "UTF-8 JSON; sort_keys=true; separators=(',',':'); "
                "ensure_ascii=false; allow_nan=false; one trailing LF"
            ),
            "tree_tuple_fields": ["path", "bytes", "mtime_ns", "sha256"],
            "tree_order": "ascending Unicode code-point order of workspace-relative path",
        },
        "inventory_groups": groups,
        "tree": {
            "algorithm": "sha256",
            "entry_count": len(flattened),
            "total_bytes": sum(int(entry["bytes"]) for entry in flattened),
            "root_sha256": root_digest,
        },
        "git": dict(
            git_metadata
            if git_metadata is not None
            else collect_git_metadata(project, workspace)
        ),
        "environment": dict(
            environment_metadata
            if environment_metadata is not None
            else collect_environment_metadata()
        ),
    }


def _read_existing_regular(path: Path) -> bytes | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FileExistsError(f"Refusing non-regular existing output: {path}")
    return path.read_bytes()


def _create_or_verify(path: Path, data: bytes) -> None:
    existing = _read_existing_regular(path)
    if existing is not None:
        if existing != data:
            raise FileExistsError(f"Refusing to overwrite different bytes: {path}")
        return

    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = _read_existing_regular(path)
        if existing != data:
            raise FileExistsError(f"Refusing to overwrite different bytes: {path}")
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def write_immutable_lock(
    output_path: Path,
    payload: Mapping[str, Any],
    detached_sha256_path: Path | None = None,
) -> tuple[str, Path]:
    """Create canonical JSON and detached SHA without overwriting differences."""

    output = Path(output_path)
    detached = (
        Path(detached_sha256_path)
        if detached_sha256_path is not None
        else output.with_name(output.name + ".sha256")
    )
    if _lexical_absolute(output) == _lexical_absolute(detached):
        raise ValueError("Lock JSON and detached SHA-256 paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    detached.parent.mkdir(parents=True, exist_ok=True)

    canonical = canonical_json_bytes(dict(payload))
    digest = hashlib.sha256(canonical).hexdigest()
    detached_bytes = f"{digest}  {output.name}\n".encode("ascii")

    # Check both destinations before creating either one.  This guarantees that
    # an ordinary rerun cannot partially mutate the pair when one is different.
    existing_output = _read_existing_regular(output)
    if existing_output is not None and existing_output != canonical:
        raise FileExistsError(f"Refusing to overwrite different bytes: {output}")
    existing_detached = _read_existing_regular(detached)
    if existing_detached is not None and existing_detached != detached_bytes:
        raise FileExistsError(f"Refusing to overwrite different bytes: {detached}")

    _create_or_verify(output, canonical)
    _create_or_verify(detached, detached_bytes)
    return digest, detached


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    inferred_workspace = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=inferred_workspace)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--gt-cache-root", type=Path)
    parser.add_argument("--staged-root", type=Path)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--egobody-source-root", type=Path)
    parser.add_argument("--dust3r-source-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Canonical JSON output (default: WORK_ROOT/external_lock/"
            "egobody_cs150.external-lock.json)"
        ),
    )
    parser.add_argument(
        "--sha256-output",
        type=Path,
        help="Detached digest output (default: OUTPUT with .sha256 appended)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    layout = Layout.from_roots(
        args.workspace_root,
        project_root=args.project_root,
        work_root=args.work_root,
        gt_cache_root=args.gt_cache_root,
        staged_root=args.staged_root,
        manifest_root=args.manifest_root,
        egobody_source_root=args.egobody_source_root,
        dust3r_source_root=args.dust3r_source_root,
    )
    output = (
        args.output
        if args.output is not None
        else layout.work_root / "external_lock" / "egobody_cs150.external-lock.json"
    )
    payload = build_lock_payload(layout)
    lock_digest, detached_path = write_immutable_lock(
        output, payload, args.sha256_output
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "detached_sha256": str(detached_path),
                "lock_sha256": lock_digest,
                "tree_sha256": payload["tree"]["root_sha256"],
                "entry_count": payload["tree"]["entry_count"],
                "entered_protocol_run_identity": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
