from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import pytest

from versions.v20.egobody.build_external_lock import (
    CHECKPOINT_PATHS,
    DIRECT_DEPENDENCY_PATHS,
    MANIFEST_FILENAMES,
    STAGING_FILENAMES,
    TOPOLOGY_ASSET_PATHS,
    Layout,
    build_lock_payload,
    canonical_json_bytes,
    file_entry,
    tree_sha256,
    write_immutable_lock,
)


GIT_METADATA = {
    "repository_root": "Movie3R",
    "head_commit": "1" * 40,
    "head_tree": "2" * 40,
    "status_format": "synthetic-test",
    "status_porcelain": ["?? versions/v20/egobody/build_external_lock.py"],
    "worktree_clean": False,
    "tracked_clean": True,
    "untracked_count": 1,
}

ENVIRONMENT_METADATA = {
    "python": {"implementation": "CPython", "version": "test"},
    "platform": {"system": "test", "release": "test", "machine": "test"},
    "packages": {"torch": "test"},
    "torch_runtime": {"importable": True, "version": "test"},
}


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.write_bytes(data)
    return path


def _synthetic_layout(root: Path) -> Layout:
    project = root / "Movie3R"
    work = root / "data" / "EgoBody_work_v20"

    # These are opaque bytes on purpose: the lock builder must hash, not parse.
    _write(work / "gt_cache" / "case_a.gt.npz", b"not-an-npz\x00\xff")
    _write(work / "gt_cache" / "case_a.gt.json", b"not-json\xff")
    _write(work / "gt_cache" / "case_b.gt.npz", b"second-npz")
    _write(work / "gt_cache" / "case_b.gt.json", b"second-json")
    for name in STAGING_FILENAMES:
        _write(work / "staged_rgb" / name, f"opaque staging {name}\n")
    for name in MANIFEST_FILENAMES:
        _write(work / "manifests" / name, f"opaque manifest {name}\n")

    for relative in CHECKPOINT_PATHS:
        _write(project / relative, f"checkpoint {relative}\n")
    for relative in TOPOLOGY_ASSET_PATHS:
        _write(project / relative, f"topology {relative}\n")
    for relative in DIRECT_DEPENDENCY_PATHS:
        _write(project / relative, f"dependency {relative}\n")

    # adaptive_joint.py is already in the explicit dependencies.  This second
    # source verifies that the declared recursive dust3r closure is non-empty.
    _write(project / "src" / "dust3r" / "nested" / "closure.py", "VALUE = 1\n")
    _write(project / "src" / "dust3r" / "nested" / "ignored.pyc", b"generated")

    egobody = project / "versions" / "v20" / "egobody"
    _write(egobody / "implementation.py", "VALUE = 2\n")
    _write(egobody / "development_candidates.json", "opaque config\n")
    _write(egobody / "README.md", "source notes\n")
    _write(egobody / "tests" / "opaque.bin", b"\x00\x01\x02")
    _write(egobody / "__pycache__" / "implementation.cpython-310.pyc", b"pyc")
    _write(egobody / "__pycache__" / "cached.json", "cache\n")
    _write(egobody / ".pytest_cache" / "nodeids.json", "cache\n")
    _write(egobody / "generated.pyc", b"pyc")
    _write(egobody / "legacy.pyo", b"pyo")
    _write(egobody / "implementation.partial.py", "partial\n")
    _write(egobody / "implementation.tmp.py", "temporary\n")
    _write(egobody / "config.temp.yaml", "temporary\n")
    _write(egobody / "implementation.py.partial", "partial\n")
    _write(egobody / "implementation.py~", "backup\n")
    _write(egobody / ".partial" / "pending.py", "partial\n")
    _write(egobody / "tmp" / "generated.json", "temporary\n")

    return Layout.from_roots(root)


def _payload(layout: Layout) -> dict:
    return build_lock_payload(
        layout,
        git_metadata=GIT_METADATA,
        environment_metadata=ENVIRONMENT_METADATA,
    )


def test_payload_is_deterministic_complete_and_external(tmp_path: Path) -> None:
    layout = _synthetic_layout(tmp_path)

    first = _payload(layout)
    second = _payload(layout)

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["tree"]["root_sha256"] == second["tree"]["root_sha256"]
    assert first["entered_protocol_run_identity"] is False
    assert "not included in protocol_state.run_identity" in first[
        "protocol_run_identity_statement"
    ]
    assert first["provenance_timing"] == {
        "development": "retrospective",
        "holdout": "pre_run",
        "test": "pre_run",
    }

    groups = first["inventory_groups"]
    assert groups["gt_cache"]["file_count"] == 4
    assert groups["manifests"]["file_count"] == len(MANIFEST_FILENAMES)
    assert groups["normative_staging_metadata"]["file_count"] == 2
    assert groups["checkpoints"]["file_count"] == 3
    assert groups["topology_and_smpl_assets"]["file_count"] == 5
    assert groups["direct_project_dependencies"]["file_count"] == len(
        DIRECT_DEPENDENCY_PATHS
    )

    egobody_paths = {
        row["path"] for row in groups["egobody_v20_implementation"]["entries"]
    }
    assert egobody_paths == {
        "Movie3R/versions/v20/egobody/README.md",
        "Movie3R/versions/v20/egobody/development_candidates.json",
        "Movie3R/versions/v20/egobody/implementation.py",
    }
    closure_paths = {
        row["path"] for row in groups["dust3r_python_source_closure"]["entries"]
    }
    assert closure_paths == {"Movie3R/src/dust3r/nested/closure.py"}
    all_paths = [
        row["path"]
        for group in groups.values()
        for row in group["entries"]
    ]
    assert all(not Path(path).is_absolute() for path in all_paths)
    assert len(all_paths) == len(set(all_paths)) == first["tree"]["entry_count"]


def test_v20_source_inventory_excludes_generated_and_temporary_files(
    tmp_path: Path,
) -> None:
    layout = _synthetic_layout(tmp_path)
    group = _payload(layout)["inventory_groups"]["egobody_v20_implementation"]
    paths = {row["path"] for row in group["entries"]}

    excluded = {
        "Movie3R/versions/v20/egobody/tests/opaque.bin",
        "Movie3R/versions/v20/egobody/__pycache__/implementation.cpython-310.pyc",
        "Movie3R/versions/v20/egobody/__pycache__/cached.json",
        "Movie3R/versions/v20/egobody/.pytest_cache/nodeids.json",
        "Movie3R/versions/v20/egobody/generated.pyc",
        "Movie3R/versions/v20/egobody/legacy.pyo",
        "Movie3R/versions/v20/egobody/implementation.partial.py",
        "Movie3R/versions/v20/egobody/implementation.tmp.py",
        "Movie3R/versions/v20/egobody/config.temp.yaml",
        "Movie3R/versions/v20/egobody/implementation.py.partial",
        "Movie3R/versions/v20/egobody/implementation.py~",
        "Movie3R/versions/v20/egobody/.partial/pending.py",
        "Movie3R/versions/v20/egobody/tmp/generated.json",
    }
    assert paths.isdisjoint(excluded)
    assert group["selection"]["include_suffixes"] == [
        ".py",
        ".json",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
    ]
    assert group["selection"]["excluded_directory_names"] == [
        ".pytest_cache",
        "__pycache__",
    ]


def test_detached_sha_and_immutable_writes(tmp_path: Path) -> None:
    layout = _synthetic_layout(tmp_path / "fixture")
    payload = _payload(layout)
    output = tmp_path / "locks" / "lock.json"

    digest, detached = write_immutable_lock(output, payload)
    canonical = canonical_json_bytes(payload)
    assert output.read_bytes() == canonical
    assert digest == hashlib.sha256(canonical).hexdigest()
    assert detached == output.with_name("lock.json.sha256")
    assert detached.read_text(encoding="ascii") == f"{digest}  lock.json\n"

    json_mtime = output.stat().st_mtime_ns
    sha_mtime = detached.stat().st_mtime_ns
    assert write_immutable_lock(output, payload) == (digest, detached)
    assert output.stat().st_mtime_ns == json_mtime
    assert detached.stat().st_mtime_ns == sha_mtime

    different_output = tmp_path / "different.json"
    different_output.write_bytes(b"{}\n")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_immutable_lock(different_output, payload)
    assert different_output.read_bytes() == b"{}\n"
    assert not different_output.with_name("different.json.sha256").exists()

    changed_payload = copy.deepcopy(payload)
    changed_payload["schema_version"] = "different"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_immutable_lock(output, changed_payload)
    assert output.read_bytes() == canonical

    sha_mismatch_output = tmp_path / "sha-mismatch.json"
    sha_mismatch_output.write_bytes(canonical)
    sha_mismatch = sha_mismatch_output.with_name("sha-mismatch.json.sha256")
    sha_mismatch.write_text(f"{'0' * 64}  sha-mismatch.json\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_immutable_lock(sha_mismatch_output, payload)
    assert sha_mismatch.read_text(encoding="ascii").startswith("0" * 64)


def test_tree_root_changes_with_bytes_size_and_mtime(tmp_path: Path) -> None:
    target = _write(tmp_path / "tree" / "value.bin", b"abcd")

    first_entry = file_entry(target, tmp_path)
    first_root = tree_sha256([first_entry])

    target.write_bytes(b"wxyz")
    byte_root = tree_sha256([file_entry(target, tmp_path)])
    assert byte_root != first_root

    target.write_bytes(b"longer")
    size_root = tree_sha256([file_entry(target, tmp_path)])
    assert size_root != byte_root

    before = target.stat()
    os.utime(
        target,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )
    mtime_root = tree_sha256([file_entry(target, tmp_path)])
    assert mtime_root != size_root


def test_missing_required_input_fails_closed(tmp_path: Path) -> None:
    layout = _synthetic_layout(tmp_path)
    missing = layout.manifest_root / "egobody_cs150_test.spec.json"
    missing.unlink()

    with pytest.raises(FileNotFoundError) as error:
        _payload(layout)
    assert str(missing) in str(error.value)
