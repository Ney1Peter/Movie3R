#!/usr/bin/env python3
"""Resume the complete five-protocol OnlineHMR extension campaign in order."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
MOVIE_PYTHON = WORKSPACE / "Movie3R/.venv/bin/python"
RUNNER = SCRIPT.with_name("run_onlinehmr_extensions_diskbounded.py")
ORDER = (
    "harmony4d_multicut",
    "aist_cs150",
    "mvhuman_mvh150",
    "aist_mc150_3",
    "aist_mc150_4",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("--mvhuman-audit-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    inventory = json.loads((root / "manifests/inventory.json").read_text(encoding="utf-8"))
    log_root = root / "campaign_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for name in ORDER:
        record = inventory["protocols"][name]
        command = [
            str(MOVIE_PYTHON), str(RUNNER),
            "--runtime-manifest", record["runtime_manifest"],
            "--evaluator-manifest", record["evaluator_manifest"],
            "--source-root", record["source_root"],
            "--work-root", str(root / "work" / name),
            "--run-root", str(root / "runs" / f"{name}_attempt01"),
            "--gpus", args.gpus,
            "--reserve-gib", str(args.reserve_gib),
        ]
        if name == "mvhuman_mvh150":
            command += ["--mvhuman-audit-root", str(args.mvhuman_audit_root.resolve())]
        path = log_root / f"{name}.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write("COMMAND " + json.dumps(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=WORKSPACE,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode:
            raise SystemExit(f"{name} stopped with code {completed.returncode}; see {path}")
        state = json.loads((root / "runs" / f"{name}_attempt01/protocol_state.json").read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            raise SystemExit(f"{name} is partial; see {path}")
        print(json.dumps({"protocol": name, "status": "complete", "cases": record["case_count"]}), flush=True)


if __name__ == "__main__":
    main()
