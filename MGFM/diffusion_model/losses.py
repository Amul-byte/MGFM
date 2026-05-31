"""Motion regularizers for generated skeleton trajectories."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from diffusion_model.util import EPS, get_joint_index, get_skeleton_edges


CONTACT_JOINT_INDICES: tuple[int, ...] = (get_joint_index("ANKLE_LEFT"), get_joint_index("ANKLE_RIGHT"))
ANKLE_JOINT_INDICES: tuple[int, ...] = CONTACT_JOINT_INDICES
PELVIS_INDEX = get_joint_index("PELVIS")
ANGLE_TRIPLETS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("left_elbow", (get_joint_index("SHOULDER_LEFT"), get_joint_index("ELBOW_LEFT"), get_joint_index("WRIST_LEFT"))),
    ("right_elbow", (get_joint_index("SHOULDER_RIGHT"), get_joint_index("ELBOW_RIGHT"), get_joint_index("WRIST_RIGHT"))),
    ("left_knee", (get_joint_index("HIP_LEFT"), get_joint_index("KNEE_LEFT"), get_joint_index("ANKLE_LEFT"))),
    ("right_knee", (get_joint_index("HIP_RIGHT"), get_joint_index("KNEE_RIGHT"), get_joint_index("ANKLE_RIGHT"))),
    ("left_hip", (get_joint_index("PELVIS"), get_joint_index("HIP_LEFT"), get_joint_index("KNEE_LEFT"))),
    ("right_hip", (get_joint_index("PELVIS"), get_joint_index("HIP_RIGHT"), get_joint_index("KNEE_RIGHT"))),
    ("left_shoulder", (get_joint_index("SPINE_CHEST"), get_joint_index("SHOULDER_LEFT"), get_joint_index("ELBOW_LEFT"))),
    ("right_shoulder", (get_joint_index("SPINE_CHEST"), get_joint_index("SHOULDER_RIGHT"), get_joint_index("ELBOW_RIGHT"))),
    ("spine_tilt", (get_joint_index("PELVIS"), get_joint_index("SPINE_CHEST"), get_joint_index("NECK"))),
    ("neck_tilt", (get_joint_index("SPINE_CHEST"), get_joint_index("NECK"), get_joint_index("HEAD"))),
)
ANGLE_LIMITS_RAD: dict[str, tuple[float, float]] = {
    # Soft ranges: permissive enough for the current dataset, but still penalize
    # degenerate spider-like hyper-folded or locked poses.
    "left_knee": (math.radians(15.0), math.radians(175.0)),
    "right_knee": (math.radians(15.0), math.radians(175.0)),
    "left_elbow": (math.radians(10.0), math.radians(175.0)),
    "right_elbow": (math.radians(10.0), math.radians(175.0)),
}
JOINT_ANGLE_NAMES: tuple[str, ...] = tuple(name for name, _ in ANGLE_TRIPLETS)


def joint_angles(positions: torch.Tensor) -> torch.Tensor:
    """Per-frame interior hinge angles [B, T, 4], ported from SSDL compute_joint_angles."""
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError(f"joint_angles expects [B, T, J, 3], got {tuple(positions.shape)}")

    joint_pairs = torch.tensor([triplet for _, triplet in ANGLE_TRIPLETS], device=positions.device)
    _, num_frames, _, _ = positions.shape

    chunk_size = 100
    angles_chunks = []
    for i in range(0, num_frames, chunk_size):
        positions_chunk = positions[:, i:i + chunk_size]

        vectors1 = positions_chunk[:, :, joint_pairs[:, 1]] - positions_chunk[:, :, joint_pairs[:, 0]]
        vectors2 = positions_chunk[:, :, joint_pairs[:, 1]] - positions_chunk[:, :, joint_pairs[:, 2]]

        dot_product = torch.sum(vectors1 * vectors2, dim=-1)

        norm1 = torch.norm(vectors1, dim=-1)
        norm2 = torch.norm(vectors2, dim=-1)

        denominator = norm1 * norm2
        valid_denominator = denominator != 0

        cosine_angles = torch.zeros_like(dot_product)
        epsilon = 1e-6
        denominator = torch.clamp(denominator, min=epsilon)
        cosine_angles[valid_denominator] = dot_product[valid_denominator] / denominator[valid_denominator]

        cosine_angles = torch.clamp(cosine_angles, -1.0 + 1e-7, 1.0 - 1e-7)

        chunk_angles = torch.acos(cosine_angles)
        chunk_angles[~valid_denominator] = 0

        angles_chunks.append(chunk_angles)

    return torch.cat(angles_chunks, dim=1)


def temporal_difference(values: torch.Tensor) -> torch.Tensor:
    """Return causal finite differences with the first frame repeated to keep length T."""
    if values.ndim < 2:
        raise ValueError(f"temporal_difference expects at least 2 dims, got {tuple(values.shape)}")
    if values.shape[1] <= 1:
        return torch.zeros_like(values)
    diff_valid = values[:, 1:] - values[:, :-1]
    return torch.cat([diff_valid[:, :1], diff_valid], dim=1)


def joint_angle_velocities(x: torch.Tensor) -> torch.Tensor:
    """Return temporal changes of the key hinge-joint angles with shape [B, T, K]."""
    return temporal_difference(joint_angles(x))


def joint_angle_limit_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalize key joints when their angles leave permissive human ranges."""
    angles = joint_angles(x)
    constrained = [(i, name) for i, (name, _) in enumerate(ANGLE_TRIPLETS) if name in ANGLE_LIMITS_RAD]
    if not constrained:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    indices = [i for i, _ in constrained]
    names = [n for _, n in constrained]
    sub = angles[:, :, indices]
    lower = torch.tensor(
        [ANGLE_LIMITS_RAD[n][0] for n in names],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, -1)
    upper = torch.tensor(
        [ANGLE_LIMITS_RAD[n][1] for n in names],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, -1)
    below = F.relu(lower - sub)
    above = F.relu(sub - upper)
    return (below + above).mean()


def angular_reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Match generated and target key-joint angles with the SSDL-style raw Frobenius norm."""
    predicted_joint_angles = joint_angles(x_hat)
    real_joint_angles = joint_angles(x)
    difference = real_joint_angles - predicted_joint_angles
    return torch.linalg.vector_norm(difference.reshape(-1), ord=2)


def bone_length_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalize temporal bone-length drift within each generated sequence."""
    lengths = []
    for i, j in get_skeleton_edges():
        bone = torch.linalg.norm(x[:, :, i, :] - x[:, :, j, :], dim=-1)
        lengths.append(bone)
    bone_lengths = torch.stack(lengths, dim=-1)
    return bone_lengths.std(dim=1, unbiased=False).mean()


def foot_skating_loss(x: torch.Tensor, contact_threshold: float = 0.03) -> torch.Tensor:
    """Penalize horizontal ankle motion when ankle contacts are close to the ground."""
    contacts = x[:, :, CONTACT_JOINT_INDICES, :]
    ground = contacts[..., 2].amin(dim=1, keepdim=True)
    near_ground = (contacts[..., 2] - ground) < contact_threshold
    horizontal_velocity = torch.linalg.norm(contacts[:, 1:, :, :2] - contacts[:, :-1, :, :2], dim=-1)
    contact_mask = near_ground[:, 1:, :].to(x.dtype)
    denom = contact_mask.sum().clamp_min(1.0)
    return (horizontal_velocity * contact_mask).sum() / denom


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalize large temporal accelerations across all joints."""
    velocity = x[:, 1:, :, :] - x[:, :-1, :, :]
    acceleration = velocity[:, 1:, :, :] - velocity[:, :-1, :, :]
    return torch.mean(acceleration.pow(2))


def instability_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalize pelvis drift outside the ankle support width in the lateral axis."""
    pelvis_x = x[:, :, PELVIS_INDEX, 0]
    left_ankle_x = x[:, :, ANKLE_JOINT_INDICES[0], 0]
    right_ankle_x = x[:, :, ANKLE_JOINT_INDICES[1], 0]
    center = 0.5 * (left_ankle_x + right_ankle_x)
    support_half_width = 0.5 * (left_ankle_x - right_ankle_x).abs().clamp_min(EPS)
    overflow = torch.relu((pelvis_x - center).abs() - support_half_width)
    return overflow.mean()


def motion_losses(x: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Compute motion regularizers and optional anatomy-aware supervision."""
    loss_bone = bone_length_loss(x)
    loss_skate = foot_skating_loss(x)
    loss_smooth = smoothness_loss(x)
    loss_instab = instability_loss(x)
    loss_motion = loss_bone + loss_skate + loss_smooth + loss_instab
    loss_angle_limit = joint_angle_limit_loss(x)
    if target is not None:
        loss_angle_recon = angular_reconstruction_loss(x, target)
    else:
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        loss_angle_recon = zero
    return {
        "loss_bone": loss_bone,
        "loss_skate": loss_skate,
        "loss_smooth": loss_smooth,
        "loss_instab": loss_instab,
        "loss_motion": loss_motion,
        "loss_angle_limit": loss_angle_limit,
        "loss_angle_recon": loss_angle_recon,
    }
