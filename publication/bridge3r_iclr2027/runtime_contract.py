"""Locked, evaluator-free Bridge3R boundary transaction primitives.

This module is deliberately small.  It makes the final paper contract
executable over already reconstructed Human3R/B0 arrays, while dataset adapters
remain responsible only for RGB staging, the causal detector, and backbone
inference.  The input arrays are required to originate from the clean-reset
post-cut branch; this module never reads ground truth, evaluator metadata, or
future frames to estimate a transaction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from versions.v16.harmony4d.causal_stabilization import (
    Candidate,
    apply_candidate,
    clone_arrays,
)


PUBLICATION_ROOT = Path(__file__).resolve().parent
LOCK_PATH = PUBLICATION_ROOT / "PAPER_METHOD_LOCK.json"


def load_method_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    """Read and minimally validate the immutable paper-method contract."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "Bridge3R-paper-method-lock-v1":
        raise ValueError("unexpected Bridge3R publication lock schema")
    fixed = payload.get("method", {}).get("fixed_parameters", {})
    expected = {
        "camera_alpha": 1.0,
        "boundary_kind": "translation",
        "boundary_blend": 0.5,
        "reliability_gate": False,
        "root_filter": False,
    }
    if fixed != expected:
        raise ValueError(f"publication method parameters differ from lock: {fixed}")
    return payload


def locked_candidate(lock: Mapping[str, Any] | None = None) -> Candidate:
    """Create exactly the one boundary candidate admitted by the paper."""

    payload = dict(lock) if lock is not None else load_method_lock()
    fixed = payload["method"]["fixed_parameters"]
    return Candidate(
        "bridge3r_publication_half_translation",
        camera_alpha=float(fixed["camera_alpha"]),
        boundary_kind=str(fixed["boundary_kind"]),
        boundary_blend=float(fixed["boundary_blend"]),
        use_velocity_target=False,
        root_alpha=None,
        root_beta=0.0,
        gate_max_boundary_residual_m=None,
        gate_min_matches=0,
        gate_min_match_fraction=0.0,
        gate_max_translation_m=None,
    )


def apply_locked_transaction(
    arrays: Mapping[str, Any],
    *,
    boundary: int | None,
    pairs: Sequence[tuple[int, int]],
    cut_detected: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one locked causal transaction or preserve an exact no-cut prefix.

    ``arrays`` must contain the clean-reset reconstruction for the complete
    currently observed prefix.  ``pairs`` are prediction-only associations at
    the proposed boundary.  Passing ``cut_detected=False`` is a strict no-op:
    it avoids even identity reindexing, which is the required no-cut behavior.
    """

    lock = load_method_lock()
    if not cut_detected:
        return clone_arrays(dict(arrays)), {
            "publication_method": "Bridge3R",
            "cut_applied": False,
            "state_lineage": {
                "pre_cut_state": "unchanged",
                "post_cut_state": "not_started",
                "shadow_state": "not_read",
            },
            "runtime_contract": {
                "gt_used": False,
                "future_frames_used": 0,
                "pre_frames_rewritten_after_emission": False,
                "no_cut_bit_exact": True,
            },
        }
    if boundary is None:
        raise ValueError("a detected cut requires an explicit boundary index")
    if boundary <= 0:
        raise ValueError("a cut boundary must have an observed pre-cut frame")
    output, diagnostics = apply_candidate(
        dict(arrays), int(boundary), [tuple(map(int, pair)) for pair in pairs], locked_candidate(lock)
    )
    if diagnostics["reliability_gate"]["enabled"] or diagnostics["root_filter"] is not None:
        raise AssertionError("inactive publication components were unexpectedly enabled")
    diagnostics.update(
        publication_method="Bridge3R",
        cut_applied=True,
        state_lineage={
            "pre_cut_state": "read_only_shadow",
            "post_cut_state": "clean_reset_input",
            "shadow_state": "read_only",
        },
    )
    return output, diagnostics


def apply_locked_multicut(
    arrays: Mapping[str, Any],
    transactions: Sequence[tuple[int, Sequence[tuple[int, int]]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compose sorted same-scene transactions without rewriting earlier prefixes."""

    current = clone_arrays(dict(arrays))
    previous = 0
    diagnostics = []
    for boundary, pairs in transactions:
        boundary = int(boundary)
        if boundary <= previous:
            raise ValueError("multi-cut boundaries must be strictly increasing")
        before = clone_arrays(current)
        current, debug = apply_locked_transaction(
            current, boundary=boundary, pairs=pairs, cut_detected=True
        )
        # Every earlier frame has already been emitted and is immutable.
        for key, value in before.items():
            if hasattr(value, "shape") and value.shape[0] >= boundary:
                if not (current[key][:boundary] == value[:boundary]).all():
                    raise AssertionError(f"multi-cut transaction rewrote prefix for {key}")
        diagnostics.append(debug)
        previous = boundary
    return current, diagnostics
