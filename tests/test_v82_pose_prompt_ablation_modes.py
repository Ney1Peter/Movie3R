"""Regression tests for the explicit publication token-ablation names."""

from __future__ import annotations

import pytest
import torch

from dust3r.v8_pose_prompt import V82PoseRelationPrompt


@pytest.mark.parametrize(
    ("mode", "expected_tokens"),
    [
        ("all", 3),
        ("semantic_only", 1),
        ("alignment_only", 1),
        ("semantic_alignment", 2),
        ("no_semantic", 2),
        ("no_alignment", 2),
        ("no_momentum", 2),
    ],
)
def test_relation_prompt_token_mode_has_declared_cardinality(mode: str, expected_tokens: int) -> None:
    torch.manual_seed(7)
    module = V82PoseRelationPrompt(enc_dim=8, dec_dim=12, num_heads=3, token_ablation=mode).eval()
    output = module(
        image_tokens=torch.randn(2, 5, 8),
        pose_token=torch.randn(2, 1, 12),
        state_tokens=torch.randn(2, 4, 12),
        pose_memory=torch.randn(2, 3, 24),
        prev_corr_token=torch.randn(2, 1, 12),
        prev_pose_token=torch.randn(2, 1, 12),
        prev_delta_token=torch.randn(2, 1, 12),
        prev_gate=torch.ones(2, 1, 1),
    )
    assert module.num_corr_tokens == expected_tokens
    assert output.corr_tokens.shape == (2, expected_tokens, 12)
    assert torch.isfinite(output.corr_tokens).all()


def test_only_modes_do_not_alias_historical_no_token_names() -> None:
    """``no_alignment`` retains momentum, unlike ``semantic_only``."""

    assert V82PoseRelationPrompt(enc_dim=8, dec_dim=12, num_heads=3, token_ablation="no_alignment").num_corr_tokens == 2
    assert V82PoseRelationPrompt(enc_dim=8, dec_dim=12, num_heads=3, token_ablation="semantic_only").num_corr_tokens == 1
