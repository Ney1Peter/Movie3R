import math

import torch
import torch.nn as nn


def standardize_quaternion(quaternions):
    quaternions = torch.nn.functional.normalize(quaternions, p=2, dim=-1)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def rotate_vector(q, v):
    q_vec = q[..., 1:]
    q_w = q[..., :1]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def rotvec_to_quaternion(rotvec, eps=1e-6):
    """Convert axis-angle rotation vectors to wxyz quaternions."""
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    angle2 = angle * angle
    small_scale = 0.5 - angle2 / 48.0 + angle2 * angle2 / 3840.0
    large_scale = torch.sin(half_angle) / angle.clamp_min(eps)
    scale = torch.where(angle < eps, small_scale, large_scale)
    quat = torch.cat([torch.cos(half_angle), rotvec * scale], dim=-1)
    return standardize_quaternion(quat)


def apply_left_se3_delta(camera_pose, delta_t, delta_rotvec, alpha):
    """Apply T_corr = exp(alpha * delta_xi) @ T_hat to absT_quaR poses."""
    pose_dtype = camera_pose.dtype
    delta_t = delta_t.to(dtype=pose_dtype)
    delta_rotvec = delta_rotvec.to(dtype=pose_dtype)
    alpha = alpha.to(dtype=pose_dtype).view(-1, 1)

    scaled_t = delta_t * alpha
    scaled_rotvec = delta_rotvec * alpha
    delta_q = rotvec_to_quaternion(scaled_rotvec)

    raw_t = camera_pose[:, :3]
    raw_q = standardize_quaternion(camera_pose[:, 3:7])
    corr_t = rotate_vector(delta_q, raw_t) + scaled_t
    corr_q = standardize_quaternion(quaternion_multiply(delta_q, raw_q))
    return torch.cat([corr_t, corr_q], dim=-1)


class HumanSceneTokenPoseAdapter(nn.Module):
    """Causal implicit human-scene token adapter for V7 pose correction.

    The adapter consumes only Human3R internal tokens plus the raw pose prior.
    It does not use decoded SMPL bodies, explicit background planes, future
    frames, or any post-hoc global optimization signal.
    """

    VALID_INPUT_MODES = {"human_scene", "all", "human", "scene", "pose"}

    def __init__(
        self,
        dec_dim,
        hidden_dim=512,
        input_mode="human_scene",
        max_delta_t=3.0,
        max_delta_r=0.75,
        dropout=0.0,
    ):
        super().__init__()
        if input_mode not in self.VALID_INPUT_MODES:
            raise ValueError(
                f"Unknown V7 adapter input_mode={input_mode!r}; "
                f"expected one of {sorted(self.VALID_INPUT_MODES)}"
            )
        self.dec_dim = int(dec_dim)
        self.input_mode = input_mode
        self.register_buffer(
            "max_delta_t",
            torch.tensor(float(max_delta_t)),
            persistent=False,
        )
        self.register_buffer(
            "max_delta_r",
            torch.tensor(float(max_delta_r)),
            persistent=False,
        )

        self.pose_norm = nn.LayerNorm(dec_dim)
        self.human_norm = nn.LayerNorm(dec_dim)
        self.scene_norm = nn.LayerNorm(dec_dim)
        self.memory_norm = nn.LayerNorm(dec_dim)
        self.pose_prior = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, dec_dim),
            nn.GELU(),
            nn.Linear(dec_dim, dec_dim),
        )

        in_dim = dec_dim * 5 + 3
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 9),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _mode_uses_human(self):
        return self.input_mode in {"human_scene", "all", "human"}

    def _mode_uses_scene(self):
        return self.input_mode in {"human_scene", "all", "scene"}

    def _pool_tokens(self, tokens, norm, batch_size, device, dtype):
        if tokens is None or tokens.shape[1] == 0:
            pooled = torch.zeros(batch_size, self.dec_dim, device=device, dtype=dtype)
            valid = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            return pooled, valid
        tokens = tokens.to(device=device, dtype=dtype)
        pooled = norm(tokens).mean(dim=1)
        valid = torch.ones(batch_size, 1, device=device, dtype=dtype)
        return pooled, valid

    def forward(
        self,
        pose_token,
        scene_tokens,
        human_tokens,
        memory_tokens,
        camera_pose,
    ):
        batch_size = camera_pose.shape[0]
        device = camera_pose.device
        dtype = pose_token.dtype

        pose_ctx = self.pose_norm(pose_token.to(device=device, dtype=dtype)).squeeze(1)
        scene_ctx, scene_valid = self._pool_tokens(
            scene_tokens, self.scene_norm, batch_size, device, dtype
        )
        human_ctx, human_valid = self._pool_tokens(
            human_tokens, self.human_norm, batch_size, device, dtype
        )
        memory_ctx, memory_valid = self._pool_tokens(
            memory_tokens, self.memory_norm, batch_size, device, dtype
        )

        if not self._mode_uses_scene():
            scene_ctx = torch.zeros_like(scene_ctx)
            scene_valid = torch.zeros_like(scene_valid)
        if not self._mode_uses_human():
            human_ctx = torch.zeros_like(human_ctx)
            human_valid = torch.zeros_like(human_valid)

        pose_prior = self.pose_prior(camera_pose.to(device=device, dtype=dtype))
        flags = torch.cat([human_valid, scene_valid, memory_valid], dim=-1)
        fused = torch.cat(
            [pose_ctx, pose_prior, human_ctx, scene_ctx, memory_ctx, flags], dim=-1
        )
        raw = self.head(fused)

        max_delta_t = self.max_delta_t.to(device=device, dtype=raw.dtype)
        max_delta_r = self.max_delta_r.to(device=device, dtype=raw.dtype)
        delta_t = torch.tanh(raw[:, 0:3]) * max_delta_t
        delta_rotvec = torch.tanh(raw[:, 3:6]) * max_delta_r
        alpha = torch.sigmoid(raw[:, 6])
        r_human = torch.sigmoid(raw[:, 7]) * human_valid.squeeze(-1)
        r_scene = torch.sigmoid(raw[:, 8]) * scene_valid.squeeze(-1)

        corrected_pose = apply_left_se3_delta(camera_pose, delta_t, delta_rotvec, alpha)
        delta_xi = torch.cat([delta_t, delta_rotvec], dim=-1)
        info = {
            "v7_pose_delta_xi": delta_xi,
            "v7_pose_delta_t": delta_t,
            "v7_pose_delta_rotvec": delta_rotvec,
            "v7_pose_alpha": alpha,
            "v7_pose_r_human": r_human,
            "v7_pose_r_scene": r_scene,
            "v7_pose_human_valid": human_valid.squeeze(-1),
            "v7_pose_scene_valid": scene_valid.squeeze(-1),
            "v7_pose_memory_valid": memory_valid.squeeze(-1),
            "v7_pose_delta_t_norm": torch.linalg.norm(delta_t, dim=-1),
            "v7_pose_delta_r_deg": torch.linalg.norm(delta_rotvec, dim=-1)
            * (180.0 / math.pi),
        }
        return corrected_pose, info
