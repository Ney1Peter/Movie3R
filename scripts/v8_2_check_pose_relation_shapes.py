#!/usr/bin/env python3
"""Shape check for V8.2 pose-relation prompt modules."""

from __future__ import annotations

import torch

from dust3r.v8_pose_prompt import (
    V82PoseRelationPrompt,
    V82PoseRelationResidualHead,
    make_v8_corr_pos,
)


def main():
    batch_size = 2
    enc_dim = 1024
    dec_dim = 768
    num_img_tokens = 576
    num_humans = 3
    num_state_tokens = 324
    num_memory_tokens = 256

    prompt = V82PoseRelationPrompt(enc_dim=enc_dim, dec_dim=dec_dim)
    head = V82PoseRelationResidualHead(dec_dim=dec_dim)

    image_tokens = torch.randn(batch_size, num_img_tokens, enc_dim)
    pose_token = torch.randn(batch_size, 1, dec_dim)
    human_tokens = torch.randn(batch_size, num_humans, dec_dim)
    state_tokens = torch.randn(batch_size, num_state_tokens, dec_dim)
    pose_memory = torch.randn(batch_size, num_memory_tokens, dec_dim * 2)
    prev_corr = torch.randn(batch_size, 3, dec_dim)
    prev_pose = torch.randn(batch_size, 1, dec_dim)
    prev_delta = torch.randn(batch_size, 1, dec_dim)
    prev_gate = torch.rand(batch_size, 1, 1)

    out = prompt(
        image_tokens=image_tokens,
        pose_token=pose_token,
        human_tokens=human_tokens,
        state_tokens=state_tokens,
        pose_memory=pose_memory,
        prev_corr_token=prev_corr,
        prev_pose_token=prev_pose,
        prev_delta_token=prev_delta,
        prev_gate=prev_gate,
        return_attention=True,
    )
    delta_applied, gate, delta_raw, drift_logit = head(out.corr_tokens)
    corr_pos = make_v8_corr_pos(batch_size, out.corr_tokens.shape[1], image_tokens.device, torch.long)

    assert out.corr_tokens.shape == (batch_size, 3, dec_dim)
    assert out.body_tokens.shape == (batch_size, 1, dec_dim)
    assert out.history_token.shape == (batch_size, 1, dec_dim)
    assert out.camera_motion_token.shape == (batch_size, 1, dec_dim)
    assert delta_applied.shape == (batch_size, 1, dec_dim)
    assert delta_raw.shape == (batch_size, 1, dec_dim)
    assert gate.shape == (batch_size, 1, 1)
    assert drift_logit.shape == (batch_size, 1, 1)
    assert corr_pos.shape == (batch_size, 3, 2)
    assert out.body_attention is not None

    print("V8.2 pose relation prompt shape check passed")
    print(f"corr_tokens: {tuple(out.corr_tokens.shape)}")
    print(f"delta_applied: {tuple(delta_applied.shape)}")
    print(f"gate: {tuple(gate.shape)}, mean={float(gate.mean()):.6f}")
    print(f"drift_logit: {tuple(drift_logit.shape)}")
    print(f"corr_pos: {tuple(corr_pos.shape)}, dtype={corr_pos.dtype}")


if __name__ == "__main__":
    main()
