from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_spec_and_assets_exist():
    spec = json.loads((HERE / "FINAL_RUNTIME_SPEC.json").read_text(encoding="utf-8"))
    assert spec["release"] == "Movie3R-v15-final"
    assert spec["status"] == "frozen_for_batch_experiments"
    result = load("validate_release", HERE / "validate_release.py").validate(check_hash=False)
    assert result["valid"], result["errors"]


def test_manifest_template_has_runtime_fields():
    rows = [json.loads(line) for line in (HERE / "BATCH_MANIFEST_TEMPLATE.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert {"case_id", "sequence", "frame", "pre_camera", "post_camera"} <= row.keys()
        assert row["pre_frames"] >= 1
        assert row["post_frames"] >= 3


def test_case_id_is_path_safe():
    module = load("run_case", HERE / "run_case.py")
    assert module.safe_case_id("three t/1100") == "three_t_1100"
