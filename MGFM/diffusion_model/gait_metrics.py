"""Gait-metric utilities for caching, conditioning, and training losses."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from diffusion_model.util import (
    DEFAULT_FPS,
    DEFAULT_JOINTS,
    TARGET_INDICES,
    project_skeleton_to_canonical_numpy,
    project_skeleton_to_canonical_torch,
)

L_ANKLE = 15
R_ANKLE = 19
HEAD = 4
GAIT_METRIC_NAMES: tuple[str, ...] = (
    "Mean CoM Fore Aft",
    "StDev CoM Fore Aft",
    "Mean CoM Width",
    "StDev CoM Width",
    "Mean CoM Height",
    "StDev CoM Height",
    "Mean Walking Speed",
    "Mean Stride Width",
    "Mean Base of Support",
)
DEFAULT_GAIT_METRICS_DIM = len(GAIT_METRIC_NAMES)
GAIT_CACHE_VERSION = "v2_9metrics"


def gait_metrics_dim() -> int:
    """Return fixed gait-summary dimension."""
    return DEFAULT_GAIT_METRICS_DIM


def _zero_summary_np() -> np.ndarray:
    return np.zeros((DEFAULT_GAIT_METRICS_DIM,), dtype=np.float32)


def gait_vector_to_dict(vector: Sequence[float]) -> "OrderedDict[str, float]":
    """Convert ordered gait vector into a named mapping."""
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape[0] != DEFAULT_GAIT_METRICS_DIM:
        raise ValueError(f"Expected gait vector of size {DEFAULT_GAIT_METRICS_DIM}, got {arr.shape[0]}")
    return OrderedDict((name, float(value)) for name, value in zip(GAIT_METRIC_NAMES, arr.tolist()))


def save_gait_metrics_csv(path: str, metrics: Sequence[float] | Mapping[str, float]) -> None:
    """Save one gait-summary vector to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(metrics, Mapping):
        row = OrderedDict((name, float(metrics[name])) for name in GAIT_METRIC_NAMES)
    else:
        row = gait_vector_to_dict(metrics)
    pd.DataFrame([row]).to_csv(path, index=False, sep=";")


def load_gait_metrics_csv(path: str) -> np.ndarray:
    """Load one gait-summary vector from CSV saved by this module."""
    df = pd.read_csv(path, sep=";")
    if df.empty:
        raise ValueError(f"Empty gait metrics CSV: {path}")
    if all(name in df.columns for name in GAIT_METRIC_NAMES):
        row = df.iloc[0][list(GAIT_METRIC_NAMES)].to_numpy(dtype=np.float32)
    else:
        row = df.iloc[0].to_numpy(dtype=np.float32)
    if row.shape[0] != DEFAULT_GAIT_METRICS_DIM:
        raise ValueError(f"Expected gait metrics dim {DEFAULT_GAIT_METRICS_DIM}, got {row.shape[0]} from {path}")
    return row.astype(np.float32)


def fit_ground_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find floor plane based on the lowest points."""
    mean_pts = np.mean(points, axis=0)
    centered = points - mean_pts
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    return normal.astype(np.float32), mean_pts.astype(np.float32)


def compute_rotation_matrix_to_align_with_z(normal: np.ndarray) -> np.ndarray:
    """Calculate rotation needed to make floor flat (Z up)."""
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    normal = normal / max(np.linalg.norm(normal), 1e-8)
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3, dtype=np.float32)
    c = float(np.dot(normal, z_axis))
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float32,
    )
    return (np.eye(3, dtype=np.float32) + vx + vx @ vx * ((1.0 - c) / (s**2))).astype(np.float32)


def rotate_and_align(pose: np.ndarray) -> np.ndarray:
    """Align skeleton so it stands upright on a flat floor."""
    t, j, _ = pose.shape
    left_pts = pose[:, L_ANKLE, :]
    right_pts = pose[:, R_ANKLE, :]
    contact_points = np.concatenate([left_pts, right_pts], axis=0)
    normal, origin = fit_ground_plane(contact_points)
    rot = compute_rotation_matrix_to_align_with_z(normal)
    aligned = (rot @ (pose - origin).reshape(-1, 3).T).T.reshape(t, j, 3)
    aligned[:, :, 2] -= np.min(aligned[:, :, 2])
    return aligned.astype(np.float32)


def compute_weighted_com(pose: np.ndarray) -> np.ndarray:
    """Current notebook CoM implementation."""
    return np.mean(pose, axis=1).astype(np.float32)


def detect_gait_events(pose: np.ndarray) -> np.ndarray:
    """Detect gait events using smoothed head-height peaks."""
    z_signal = -gaussian_filter1d(pose[:, L_ANKLE, 2], sigma=1)
    peaks, _ = find_peaks(-z_signal, distance=15)
    return peaks.astype(np.int64)


def _extract_pose16_numpy(pose: np.ndarray) -> np.ndarray:
    if pose.ndim != 3 or pose.shape[-1] != 3:
        raise ValueError(f"Expected pose [T,J,3], got {pose.shape}")
    return project_skeleton_to_canonical_numpy(pose)


def compute_gait_metrics_numpy(pose: np.ndarray, fps: float = DEFAULT_FPS) -> tuple[np.ndarray, OrderedDict[str, float]]:
    """Compute reference gait metrics from one skeleton sequence."""
    pose16 = _extract_pose16_numpy(np.asarray(pose, dtype=np.float32))
    pose_aligned = rotate_and_align(pose16)
    com = compute_weighted_com(pose_aligned)
    peaks = detect_gait_events(pose_aligned)
    stride_segments = list(zip(peaks[:-1], peaks[1:]))
    if len(stride_segments) == 0:
        zero = _zero_summary_np()
        return zero, gait_vector_to_dict(zero)

    com_fore_aft: list[float] = []
    com_width: list[float] = []
    com_height: list[float] = []
    walking_speeds: list[float] = []
    stride_widths: list[float] = []
    bos_widths: list[float] = []
    for a, b in stride_segments:
        com_fore_aft.append(float(com[b, 1] - com[a, 1]))
        com_width.append(float(np.max(com[a:b, 0]) - np.min(com[a:b, 0])))
        com_height.append(float(np.max(com[a:b, 2]) - np.min(com[a:b, 2])))
        dist = float(np.abs(com[b, 1] - com[a, 1]))
        time = float((b - a) / max(fps, 1e-8))
        walking_speeds.append(dist / time if time > 0 else 0.0)
        stride_widths.append(float(np.abs(pose_aligned[a, L_ANKLE, 0] - pose_aligned[a, R_ANKLE, 0])))
        avg_bos = float(np.mean(np.abs(pose_aligned[a:b, L_ANKLE, 0] - pose_aligned[a:b, R_ANKLE, 0])))
        bos_widths.append(avg_bos)

    vector = np.array(
        [
            np.mean(com_fore_aft),
            np.std(com_fore_aft),
            np.mean(com_width),
            np.std(com_width),
            np.mean(com_height),
            np.std(com_height),
            np.mean(walking_speeds),
            np.mean(stride_widths),
            np.mean(bos_widths),
        ],
        dtype=np.float32,
    )
    return vector, gait_vector_to_dict(vector)


def _gaussian_kernel1d_torch(device: torch.device, dtype: torch.dtype, sigma: float = 1.0, radius: int = 3) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords**2) / max(2.0 * sigma * sigma, 1e-8))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.view(1, 1, -1)


def _extract_pose16_torch(pose: torch.Tensor) -> torch.Tensor:
    if pose.ndim != 3 or pose.shape[-1] != 3:
        raise ValueError(f"Expected pose [T,J,3], got {tuple(pose.shape)}")
    return project_skeleton_to_canonical_torch(pose)


def _rotation_matrix_to_z_torch(normal: torch.Tensor) -> torch.Tensor:
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=normal.device, dtype=normal.dtype)
    normal = normal / normal.norm().clamp_min(1e-8)
    v = torch.cross(normal, z_axis, dim=0)
    s = v.norm()
    if float(s.item()) == 0.0:
        return torch.eye(3, device=normal.device, dtype=normal.dtype)
    c = torch.dot(normal, z_axis)
    zero = torch.zeros((), device=normal.device, dtype=normal.dtype)
    vx = torch.stack(
        [
            torch.stack([zero, -v[2], v[1]]),
            torch.stack([v[2], zero, -v[0]]),
            torch.stack([-v[1], v[0], zero]),
        ]
    )
    return torch.eye(3, device=normal.device, dtype=normal.dtype) + vx + vx @ vx * ((1.0 - c) / (s * s).clamp_min(1e-8))


def rotate_and_align_torch(pose: torch.Tensor) -> torch.Tensor:
    """Torch approximation of the notebook ground-alignment pipeline."""
    pose16 = _extract_pose16_torch(pose)
    left_pts = pose16[:, L_ANKLE, :]
    right_pts = pose16[:, R_ANKLE, :]
    contact_points = torch.cat([left_pts, right_pts], dim=0)
    origin = contact_points.mean(dim=0)
    centered = contact_points - origin
    centered64 = centered.to(dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=pose.device, dtype=pose.dtype)
    if (not torch.isfinite(centered64).all()) or float(centered64.abs().max().item()) < 1e-8:
        normal = z_axis
    else:
        try:
            cov = centered64.transpose(0, 1) @ centered64
            eigvals, eigvecs = torch.linalg.eigh(cov)
            normal = eigvecs[:, 0].to(device=pose.device, dtype=pose.dtype)
            if not torch.isfinite(normal).all():
                normal = z_axis
        except RuntimeError:
            normal = z_axis
    rot = _rotation_matrix_to_z_torch(normal)
    aligned = ((pose16 - origin) @ rot.T).contiguous()
    z_aligned = aligned[..., 2:3] - aligned[..., 2:3].amin()
    return torch.cat([aligned[..., :2], z_aligned], dim=-1)


def _detect_gait_events_torch(pose16: torch.Tensor, distance: int = 15) -> torch.Tensor:
    head_z = pose16[:, L_ANKLE, 2].view(1, 1, -1)
    kernel = _gaussian_kernel1d_torch(device=pose16.device, dtype=pose16.dtype, sigma=1.0, radius=3)
    pad = kernel.shape[-1] // 2
    smooth = F.conv1d(F.pad(head_z, (pad, pad), mode="replicate"), kernel).view(-1)
    maxima = F.max_pool1d(smooth.view(1, 1, -1), kernel_size=3, stride=1, padding=1).view(-1)
    mask = (smooth >= maxima - 1e-6)
    candidates = torch.nonzero(mask, as_tuple=False).view(-1)
    if candidates.numel() < 2:
        return candidates
    kept: list[int] = [int(candidates[0].item())]
    for idx in candidates[1:].tolist():
        if idx - kept[-1] >= distance:
            kept.append(int(idx))
        elif float(smooth[idx].item()) > float(smooth[kept[-1]].item()):
            kept[-1] = int(idx)
    return torch.tensor(kept, device=pose16.device, dtype=torch.long)


def compute_gait_metrics_torch(pose: torch.Tensor, fps: float = DEFAULT_FPS) -> torch.Tensor:
    """Compute gait metrics from skeleton tensors for training-time losses."""
    if pose.ndim == 4:
        vectors = [compute_gait_metrics_torch(sample, fps=fps) for sample in pose]
        return torch.stack(vectors, dim=0)
    if pose.ndim != 3:
        raise ValueError(f"Expected pose [T,J,3] or [B,T,J,3], got {tuple(pose.shape)}")
    pose16 = rotate_and_align_torch(pose)
    com = pose16.mean(dim=1)
    peaks = _detect_gait_events_torch(pose16)
    if peaks.numel() < 2:
        return torch.zeros((DEFAULT_GAIT_METRICS_DIM,), device=pose.device, dtype=pose.dtype)

    com_fore_aft: list[torch.Tensor] = []
    com_width: list[torch.Tensor] = []
    com_height: list[torch.Tensor] = []
    walking_speeds: list[torch.Tensor] = []
    stride_widths: list[torch.Tensor] = []
    bos_widths: list[torch.Tensor] = []
    for idx in range(peaks.numel() - 1):
        a = int(peaks[idx].item())
        b = int(peaks[idx + 1].item())
        if b <= a:
            continue
        segment = com[a:b]
        com_fore_aft.append(com[b, 1] - com[a, 1])
        com_width.append(segment[:, 0].amax() - segment[:, 0].amin())
        com_height.append(segment[:, 2].amax() - segment[:, 2].amin())
        dist = (com[b, 1] - com[a, 1]).abs()
        delta_t = torch.tensor((b - a) / max(fps, 1e-8), device=pose.device, dtype=pose.dtype)
        walking_speeds.append(dist / delta_t.clamp_min(1e-8))
        stride_widths.append((pose16[a, L_ANKLE, 0] - pose16[a, R_ANKLE, 0]).abs())
        bos_widths.append((pose16[a:b, L_ANKLE, 0] - pose16[a:b, R_ANKLE, 0]).abs().mean())
    if len(com_fore_aft) == 0:
        return torch.zeros((DEFAULT_GAIT_METRICS_DIM,), device=pose.device, dtype=pose.dtype)

    def _mean(items: Iterable[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(list(items))
        return stack.mean()

    def _std(items: Iterable[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(list(items))
        return stack.std(unbiased=False)

    return torch.stack(
        [
            _mean(com_fore_aft),
            _std(com_fore_aft),
            _mean(com_width),
            _std(com_width),
            _mean(com_height),
            _std(com_height),
            _mean(walking_speeds),
            _mean(stride_widths),
            _mean(bos_widths),
        ],
        dim=0,
    )
