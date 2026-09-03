from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("stage_egobody_assets.py")


def nested_zip(member: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


def build_outer(path: Path) -> None:
    calibration = nested_zip("calibrations/recording/cal_trans/value.json", b"{}")
    parameters = nested_zip("kinect_cam_params/kinect_master/Color.json", b"{}")
    empty_parameters = nested_zip("smplx_test/dummy.pkl", b"payload")
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        archive.writestr("release/data_info_release.csv", "recording_name\nrecording\n")
        archive.writestr("release/data_splits.csv", "train,val,test\n,,recording\n")
        archive.writestr("release/calibrations.zip", calibration)
        archive.writestr("release/kinect_cam_params.zip", parameters)
        archive.writestr("release/smplx_interactee_test.zip", empty_parameters)
        archive.writestr("release/smplx_camera_wearer_test.zip", empty_parameters)


def test_materializes_only_required_assets_and_expands_metadata(tmp_path: Path) -> None:
    outer = tmp_path / "EgoBody.zip"
    output = tmp_path / "work" / "outer"
    build_outer(outer)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--outer",
            str(outer),
            "--output-root",
            str(output),
            "--reserve-gib",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "complete"
    assert len(result["assets"]) == 6
    assert (output / "calibrations.zip").is_file()
    assert (output / "expanded/calibrations/recording/cal_trans/value.json").is_file()
    assert (output / "expanded/kinect_cam_params/kinect_master/Color.json").is_file()

    reused = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--outer",
            str(outer),
            "--output-root",
            str(output),
            "--reserve-gib",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(reused.stdout)["status"] == "complete"


def test_rejects_archive_without_central_directory(tmp_path: Path) -> None:
    outer = tmp_path / "EgoBody.zip"
    build_outer(outer)
    outer.write_bytes(outer.read_bytes()[:-40])
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--outer",
            str(outer),
            "--output-root",
            str(tmp_path / "output"),
            "--reserve-gib",
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert (
        "zip" in completed.stderr.lower()
        or "expected one outer member" in completed.stderr.lower()
    )
