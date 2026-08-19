from __future__ import annotations

import numpy as np

from versions.v16.harmony4d.aggregate_multisequence import paired_sequence_test


def _row(sequence: str, value: float, accepted: bool) -> dict:
    return {
        "sequence": sequence,
        "metrics": {"W-MPJPE_mm": value},
        "diagnostics": {"reliability_gate": {"accepted": accepted}},
    }


def test_parent_significance_zeros_contractual_full_fallback_noise() -> None:
    primary = [
        _row("accepted_action", 9.0, True),
        _row("fallback_action", 9.99999, False),
    ]
    parent = [
        _row("accepted_action", 10.0, False),
        _row("fallback_action", 10.0, False),
    ]

    result = paired_sequence_test(
        primary,
        parent,
        "W-MPJPE_mm",
        higher=False,
        rng=np.random.default_rng(0),
        zero_all_fallback_sequences=True,
    )

    assert result["exact_fallback_zeroed_sequences"] == ["fallback_action"]
    assert result["primary_minus_baseline_mean"] == -0.5
    assert result["primary_better_fraction"] == 0.5
    assert result["p_two_sided"] == 1.0
