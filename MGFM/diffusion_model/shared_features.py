"""Shared acceleration-derived motion features for skeleton and IMU inputs."""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_model.losses import JOINT_ANGLE_NAMES, joint_angles, joint_angle_velocities, temporal_difference
from diffusion_model.util import assert_shape


SHARED_FEATURE_NAMES: tuple[str, ...] = (
    "ax",
    "ay",
    "az",
    "magnitude",
    "pitch",
    "roll",
    "jerk_x",
    "jerk_y",
    "jerk_z",
    "jerk_magnitude",
)
SHARED_FEATURE_DIM: int = len(SHARED_FEATURE_NAMES)
ANGLE_FEATURE_NAMES: tuple[str, ...] = JOINT_ANGLE_NAMES
ANGULAR_VELOCITY_FEATURE_NAMES: tuple[str, ...] = tuple(f"d_{name}" for name in ANGLE_FEATURE_NAMES)
IMU_ANGLE_FEATURE_NAMES: tuple[str, ...] = JOINT_ANGLE_NAMES + tuple(f"d_{name}" for name in JOINT_ANGLE_NAMES)
SKELETON_SHARED_FEATURE_NAMES: tuple[str, ...] = (
    SHARED_FEATURE_NAMES + ANGLE_FEATURE_NAMES + ANGULAR_VELOCITY_FEATURE_NAMES
)
IMU_SHARED_FEATURE_NAMES: tuple[str, ...] = SHARED_FEATURE_NAMES + IMU_ANGLE_FEATURE_NAMES
ANGLE_AWARE_SHARED_FEATURE_DIM: int = len(SKELETON_SHARED_FEATURE_NAMES)

if len(IMU_SHARED_FEATURE_NAMES) != ANGLE_AWARE_SHARED_FEATURE_DIM:
    raise AssertionError("Skeleton and IMU shared feature schemas must have matching dimensionality.")


def build_shared_motion_features(accel_3d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build acceleration-derived motion features for [B, T, ..., 3] inputs."""
    if accel_3d.ndim < 3 or accel_3d.shape[-1] != 3:
        raise ValueError(
            f"build_shared_motion_features expects shape [B, T, ..., 3], got {tuple(accel_3d.shape)}"
        )

    ax = accel_3d[..., 0]
    ay = accel_3d[..., 1]
    az = accel_3d[..., 2]

    magnitude = torch.sqrt(torch.clamp(ax * ax + ay * ay + az * az, min=eps))
    pitch = torch.atan2(ax, torch.sqrt(torch.clamp(ay * ay + az * az, min=eps)))
    az_safe = torch.where(az.abs() < eps, torch.full_like(az, eps), az)
    roll = torch.atan2(ay, az_safe)

    if accel_3d.shape[1] > 1:
        jerk_valid = accel_3d[:, 1:] - accel_3d[:, :-1]
        jerk = torch.cat([jerk_valid[:, :1], jerk_valid], dim=1)
    else:
        jerk = torch.zeros_like(accel_3d)

    jerk_x = jerk[..., 0]
    jerk_y = jerk[..., 1]
    jerk_z = jerk[..., 2]
    jerk_magnitude = torch.sqrt(torch.clamp(jerk_x * jerk_x + jerk_y * jerk_y + jerk_z * jerk_z, min=eps))

    features = torch.stack(
        [ax, ay, az, magnitude, pitch, roll, jerk_x, jerk_y, jerk_z, jerk_magnitude],
        dim=-1,
    )

    if features.shape[:-1] != accel_3d.shape[:-1] or features.shape[-1] != SHARED_FEATURE_DIM:
        raise AssertionError(
            f"Shared feature shape mismatch: expected {accel_3d.shape[:-1]} + ({SHARED_FEATURE_DIM},), "
            f"got {tuple(features.shape)}"
        )
    return features


def _broadcast_per_frame_features(per_frame: torch.Tensor, num_joints: int) -> torch.Tensor:
    """Broadcast [B, T, C] features over joints to [B, T, J, C]."""
    assert_shape(per_frame, [None, None, None], "_broadcast_per_frame_features.per_frame")
    expanded = per_frame.unsqueeze(2).expand(-1, -1, num_joints, -1)
    assert_shape(
        expanded,
        [per_frame.shape[0], per_frame.shape[1], num_joints, per_frame.shape[2]],
        "_broadcast_per_frame_features.expanded",
    )
    return expanded


def compute_skeleton_acceleration(positions: torch.Tensor) -> torch.Tensor:
    """Compute a causal finite-difference acceleration tensor with exact [B, T, J, 3] output."""
    assert_shape(positions, [None, None, None, 3], "compute_skeleton_acceleration.positions")
    b, t, j, _ = positions.shape
    if t <= 1:
        accel = torch.zeros_like(positions)
        assert_shape(accel, [b, t, j, 3], "compute_skeleton_acceleration.accel")
        return accel

    vel_valid = positions[:, 1:] - positions[:, :-1]          # [B, T-1, J, 3]
    vel = torch.cat([vel_valid[:, :1], vel_valid], dim=1)     # [B, T,   J, 3]

    accel_valid = vel[:, 1:] - vel[:, :-1]                    # [B, T-1, J, 3]
    accel = torch.cat([accel_valid[:, :1], accel_valid], dim=1)  # [B, T, J, 3]

    assert_shape(accel, [b, t, j, 3], "compute_skeleton_acceleration.accel")
    return accel


def build_skeleton_shared_motion_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build the 18-dim skeleton shared-motion feature tensor [B, T, J, 18]."""
    assert_shape(x, [None, None, None, 3], "build_skeleton_shared_motion_features.x")
    accel = compute_skeleton_acceleration(x)
    accel_shared = build_shared_motion_features(accel, eps=eps)
    angle_features = _broadcast_per_frame_features(joint_angles(x), num_joints=x.shape[2])
    angular_velocity_features = _broadcast_per_frame_features(joint_angle_velocities(x), num_joints=x.shape[2])
    features = torch.cat([accel_shared, angle_features, angular_velocity_features], dim=-1)
    assert_shape(
        features,
        [x.shape[0], x.shape[1], x.shape[2], ANGLE_AWARE_SHARED_FEATURE_DIM],
        "build_skeleton_shared_motion_features.features",
    )
    return features


class IMUHingeAngleHead(nn.Module):
    """Predict SSDL hinge-angle attributes from phone/watch acceleration features."""

    def __init__(
        self,
        in_dim: int = SHARED_FEATURE_DIM * 2,
        hidden: int = 64,
        n_angles: int = len(JOINT_ANGLE_NAMES),
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_angles = int(n_angles)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_angles),
        )

    def forward(self, phone_shared: torch.Tensor, watch_shared: torch.Tensor) -> torch.Tensor:
        assert_shape(phone_shared, [None, None, SHARED_FEATURE_DIM], "IMUHingeAngleHead.phone_shared")
        assert_shape(
            watch_shared,
            [phone_shared.shape[0], phone_shared.shape[1], SHARED_FEATURE_DIM],
            "IMUHingeAngleHead.watch_shared",
        )
        x = torch.cat([phone_shared, watch_shared], dim=-1)
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"IMUHingeAngleHead expects last dim {self.in_dim}, got {x.shape[-1]}")
        hinge = self.net(x)
        assert_shape(hinge, [phone_shared.shape[0], phone_shared.shape[1], self.n_angles], "IMUHingeAngleHead.hinge")
        return hinge


def build_imu_shared_motion_features(
    phone_accel: torch.Tensor,
    watch_accel: torch.Tensor,
    hinge_head: IMUHingeAngleHead,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build the 18-dim phone+watch shared-motion feature tensor [B, T, 18]."""
    assert_shape(phone_accel, [None, None, 3], "build_imu_shared_motion_features.phone_accel")
    assert_shape(
        watch_accel,
        [phone_accel.shape[0], phone_accel.shape[1], 3],
        "build_imu_shared_motion_features.watch_accel",
    )
    phone_shared = build_shared_motion_features(phone_accel, eps=eps)
    watch_shared = build_shared_motion_features(watch_accel, eps=eps)
    accel_shared = 0.5 * (phone_shared + watch_shared)
    hinge = hinge_head(phone_shared, watch_shared)
    d_hinge = temporal_difference(hinge)
    features = torch.cat([accel_shared, hinge, d_hinge], dim=-1)
    assert_shape(
        features,
        [phone_accel.shape[0], phone_accel.shape[1], ANGLE_AWARE_SHARED_FEATURE_DIM],
        "build_imu_shared_motion_features.features",
    )
    return features


class SharedMotionLayer(nn.Module):
    """Shared projection applied to same-meaning motion features from both modalities."""

    def __init__(self, input_dim: int = ANGLE_AWARE_SHARED_FEATURE_DIM, d_shared: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_shared = d_shared
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_shared),
            nn.GELU(),
            nn.Linear(d_shared, d_shared),
        )

    def forward(self, x: torch.Tensor, modality: str = "imu") -> torch.Tensor:
        """Project the last dimension of x while preserving all leading dimensions."""
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"SharedMotionLayer expects last dim {self.input_dim}, got {x.shape[-1]}"
            )
        out = self.net(x)
        if out.shape[:-1] != x.shape[:-1] or out.shape[-1] != self.d_shared:
            raise AssertionError(
                f"SharedMotionLayer output shape mismatch: input={tuple(x.shape)} output={tuple(out.shape)}"
            )
        return out
