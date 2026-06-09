"""Utility functions and constants for joint-aware latent diffusion."""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch

DEFAULT_ODE_STEPS = 50
# Deprecated DDPM compatibility alias. New code uses DEFAULT_ODE_STEPS.
DEFAULT_TIMESTEPS = DEFAULT_ODE_STEPS
DEFAULT_LATENT_DIM = 256
DEFAULT_WINDOW = 32
DEFAULT_NUM_CLASSES = 14
DEFAULT_LAMBDA_CLS = 0.1
DEFAULT_FPS = 30.0
EPS = 1e-6

SKELETON_LAYOUT_VERSION = "21j_smartfall_bones_v1"
PHONE_WATCH_SENSOR_LAYOUT = "phone_watch_accel"
PHONE_WATCH_SENSOR_NAMES: Tuple[str, str] = ("phone", "watch")
INPUT_NORM_SCHEMA_NONE = "none"
INPUT_NORM_SCHEMA_ZSCORE = "imu_zscore_per_axis_v1"

# Legacy source joint order present in raw SmartFall skeleton CSVs and older
# checkpoints/datasets. This remains available only for input normalization.
SOURCE_JOINT_LABELS_32: Tuple[str, ...] = (
    "PELVIS",
    "SPINE_NAVAL",
    "SPINE_CHEST",
    "NECK",
    "CLAVICLE_LEFT",
    "SHOULDER_LEFT",
    "ELBOW_LEFT",
    "WRIST_LEFT",
    "HAND_LEFT",
    "HANDTIP_LEFT",
    "THUMB_LEFT",
    "CLAVICLE_RIGHT",
    "SHOULDER_RIGHT",
    "ELBOW_RIGHT",
    "WRIST_RIGHT",
    "HAND_RIGHT",
    "HANDTIP_RIGHT",
    "THUMB_RIGHT",
    "HIP_LEFT",
    "KNEE_LEFT",
    "ANKLE_LEFT",
    "FOOT_LEFT",
    "HIP_RIGHT",
    "KNEE_RIGHT",
    "ANKLE_RIGHT",
    "FOOT_RIGHT",
    "HEAD",
    "NOSE",
    "EYE_LEFT",
    "EAR_LEFT",
    "EYE_RIGHT",
    "EAR_RIGHT",
)
SOURCE_JOINTS_32 = len(SOURCE_JOINT_LABELS_32)

# Canonical internal 21-joint SmartFall skeleton used across the whole repo.
TARGET_INDICES_21: Tuple[int, ...] = (0, 1, 2, 3, 26, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25)
TARGET_INDICES: Tuple[int, ...] = TARGET_INDICES_21
JOINT_LABELS: Tuple[str, ...] = tuple(SOURCE_JOINT_LABELS_32[idx] for idx in TARGET_INDICES)
DEFAULT_JOINTS = len(JOINT_LABELS)

SOURCE_SKELETON_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),   # Pelvis -> Spine Navel
    (1, 2),   # Spine Navel -> Spine Chest
    (2, 3),   # Spine Chest -> Neck
    (3, 26),  # Neck -> Head
    (2, 4),   # Spine Chest -> Clavicle Left
    (4, 5),   # Clavicle Left -> Shoulder Left
    (5, 6),   # Shoulder Left -> Elbow Left
    (6, 7),   # Elbow Left -> Wrist Left
    (2, 11),  # Spine Chest -> Clavicle Right
    (11, 12), # Clavicle Right -> Shoulder Right
    (12, 13), # Shoulder Right -> Elbow Right
    (13, 14), # Elbow Right -> Wrist Right
    (0, 18),  # Pelvis -> Hip Left
    (18, 19), # Hip Left -> Knee Left
    (19, 20), # Knee Left -> Ankle Left
    (20, 21), # Ankle Left -> Foot Left
    (0, 22),  # Pelvis -> Hip Right
    (22, 23), # Hip Right -> Knee Right
    (23, 24), # Knee Right -> Ankle Right
    (24, 25), # Ankle Right -> Foot Right
)

_SKELETON_CONNECTIONS_BY_NAME: Tuple[Tuple[str, str], ...] = (
    ("PELVIS", "SPINE_NAVAL"),
    ("SPINE_NAVAL", "SPINE_CHEST"),
    ("SPINE_CHEST", "NECK"),
    ("NECK", "HEAD"),
    ("SPINE_CHEST", "CLAVICLE_LEFT"),
    ("CLAVICLE_LEFT", "SHOULDER_LEFT"),
    ("SHOULDER_LEFT", "ELBOW_LEFT"),
    ("ELBOW_LEFT", "WRIST_LEFT"),
    ("SPINE_CHEST", "CLAVICLE_RIGHT"),
    ("CLAVICLE_RIGHT", "SHOULDER_RIGHT"),
    ("SHOULDER_RIGHT", "ELBOW_RIGHT"),
    ("ELBOW_RIGHT", "WRIST_RIGHT"),
    ("PELVIS", "HIP_LEFT"),
    ("HIP_LEFT", "KNEE_LEFT"),
    ("KNEE_LEFT", "ANKLE_LEFT"),
    ("ANKLE_LEFT", "FOOT_LEFT"),
    ("PELVIS", "HIP_RIGHT"),
    ("HIP_RIGHT", "KNEE_RIGHT"),
    ("KNEE_RIGHT", "ANKLE_RIGHT"),
    ("ANKLE_RIGHT", "FOOT_RIGHT"),
)

_JOINT_TO_INDEX = {label: idx for idx, label in enumerate(JOINT_LABELS)}


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_joint_index(name: str) -> int:
    """Return the canonical joint index for a named 21-joint skeleton node."""
    try:
        return _JOINT_TO_INDEX[name]
    except KeyError as exc:
        raise KeyError(f"Unknown canonical joint name: {name}") from exc


def get_skeleton_edges() -> List[Tuple[int, int]]:
    """Return adjacency edges derived from the canonical 21-joint skeleton graph."""
    return [(get_joint_index(a), get_joint_index(b)) for a, b in _SKELETON_CONNECTIONS_BY_NAME]


def get_joint_labels() -> Tuple[str, ...]:
    """Return the exact expected canonical joint label order."""
    return JOINT_LABELS


def get_source_joint_labels_32() -> Tuple[str, ...]:
    """Return the legacy 32-joint source label order."""
    return SOURCE_JOINT_LABELS_32


def require_canonical_joint_count(num_joints: int, context: str = "skeleton") -> None:
    """Fail fast when code tries to instantiate a non-canonical internal layout."""
    if int(num_joints) != DEFAULT_JOINTS:
        raise ValueError(
            f"{context} must use the canonical {DEFAULT_JOINTS}-joint layout "
            f"({SKELETON_LAYOUT_VERSION}); got {num_joints}. Legacy 32-joint data is supported "
            "only as an input format and is projected at load time."
        )


def validate_joint_labels(labels: Sequence[str], allow_legacy_source: bool = False) -> None:
    """Validate canonical 21-joint labels, optionally allowing legacy 32-joint labels."""
    got = list(labels)
    expected = list(JOINT_LABELS)
    if got == expected:
        return
    if allow_legacy_source and got == list(SOURCE_JOINT_LABELS_32):
        return
    raise AssertionError(
        "Joint label order mismatch.\n"
        f"Expected canonical: {expected}\n"
        f"Got:                {got}"
    )


def _project_numpy_impl(pose: np.ndarray) -> np.ndarray:
    joints = pose.shape[-2]
    if joints == DEFAULT_JOINTS:
        return pose.astype(np.float32, copy=False)
    if joints == SOURCE_JOINTS_32:
        return pose[..., TARGET_INDICES, :].astype(np.float32, copy=False)
    raise ValueError(f"Expected {DEFAULT_JOINTS} or {SOURCE_JOINTS_32} joints, got {joints}")


def project_skeleton_to_canonical_numpy(pose: np.ndarray) -> np.ndarray:
    """Project a skeleton array with trailing shape [..., J, 3] to canonical 21 joints."""
    arr = np.asarray(pose, dtype=np.float32)
    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(f"Expected skeleton array ending in [J,3], got {arr.shape}")
    return _project_numpy_impl(arr)


def project_skeleton_to_canonical_torch(pose: torch.Tensor) -> torch.Tensor:
    """Project a skeleton tensor with trailing shape [..., J, 3] to canonical 21 joints."""
    if pose.ndim < 2 or pose.shape[-1] != 3:
        raise ValueError(f"Expected skeleton tensor ending in [J,3], got {tuple(pose.shape)}")
    joints = pose.shape[-2]
    if joints == DEFAULT_JOINTS:
        return pose
    if joints == SOURCE_JOINTS_32:
        index = torch.tensor(TARGET_INDICES, device=pose.device, dtype=torch.long)
        return pose.index_select(dim=pose.ndim - 2, index=index)
    raise ValueError(f"Expected {DEFAULT_JOINTS} or {SOURCE_JOINTS_32} joints, got {joints}")


def build_adjacency_matrix(num_joints: int, device: torch.device | None = None) -> torch.Tensor:
    """Build a binary adjacency matrix with self-loops for the canonical skeleton graph."""
    adj = torch.zeros((num_joints, num_joints), dtype=torch.float32, device=device)
    for i, j in get_skeleton_edges():
        if i < num_joints and j < num_joints:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    adj.fill_diagonal_(1.0)
    return adj


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal embeddings for flow time.

    Flow matching uses continuous t in [0, 1]. Legacy integer timesteps are
    still accepted during the migration; values in [0, 1] are scaled to retain
    high-frequency resolution.
    """
    assert timesteps.ndim == 1, "timesteps must have shape [B]"
    half_dim = dim // 2
    scale = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale)
    t_float = timesteps.float()
    if t_float.numel() == 0 or float(t_float.detach().max().item()) <= 1.0:
        t_float = t_float * 1000.0
    emb = t_float.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=emb.device)], dim=1)
    assert emb.shape == (timesteps.shape[0], dim), "invalid timestep embedding shape"
    return emb


def assert_shape(tensor: torch.Tensor, expected: Sequence[int | None], name: str) -> None:
    """Assert tensor rank and dimensions, where None means any size."""
    assert tensor.ndim == len(expected), f"{name} rank mismatch: got {tensor.ndim}, expected {len(expected)}"
    for idx, (actual, exp) in enumerate(zip(tensor.shape, expected)):
        if exp is not None:
            assert actual == exp, f"{name} dim {idx} mismatch: got {actual}, expected {exp}"
