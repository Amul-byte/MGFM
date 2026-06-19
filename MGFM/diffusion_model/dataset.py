"""Dataset utilities with torch-file and paired CSV-folder modes."""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from diffusion_model.gait_metrics import (
    DEFAULT_GAIT_METRICS_DIM,
    GAIT_CACHE_VERSION,
    compute_gait_metrics_numpy,
    load_gait_metrics_csv,
    save_gait_metrics_csv,
)
from diffusion_model.util import (
    DEFAULT_FPS,
    DEFAULT_JOINTS,
    DEFAULT_NUM_CLASSES,
    SOURCE_JOINTS_32,
    DEFAULT_WINDOW,
    assert_shape,
    EPS,
    get_joint_labels,
    project_skeleton_to_canonical_numpy,
    require_canonical_joint_count,
)

ACTIVITY_RE = re.compile(r"A(\d{2})", re.IGNORECASE)
SUBJECT_RE = re.compile(r"S(\d+)", re.IGNORECASE)


def _parse_label_14(fname: str, num_classes: int = DEFAULT_NUM_CLASSES) -> int:
    """Parse activity code Axx from filename into zero-based label index."""
    match = ACTIVITY_RE.search(fname)
    if match is None:
        return 0
    activity = int(match.group(1))
    return max(0, min(num_classes - 1, activity - 1))


def _parse_subject_id(name: str) -> int:
    """Parse subject code Sxx from filename-like strings."""
    match = SUBJECT_RE.search(str(name))
    if match is None:
        raise ValueError(f"Could not parse subject id from: {name}")
    return int(match.group(1))


def parse_subject_list(raw: str) -> list[int]:
    """Parse a comma-separated subject list like '28,29,30'."""
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    return [int(item) for item in items]


def extract_subject_ids(dataset: Dataset) -> Optional[list[int]]:
    """Return per-sample subject ids when available on a dataset or subset."""
    if isinstance(dataset, Subset):
        base_subject_ids = extract_subject_ids(dataset.dataset)
        if base_subject_ids is None:
            return None
        return [base_subject_ids[idx] for idx in dataset.indices]
    subject_ids = getattr(dataset, "subject_ids", None)
    if subject_ids is None:
        return None
    if isinstance(subject_ids, torch.Tensor):
        return [int(x) for x in subject_ids.tolist()]
    return [int(x) for x in subject_ids]


# Hardcoded held-out test subjects: NEVER used for training or validation under
# any split mode. Reserved as a final unseen test set for generation/evaluation.
RESERVED_TEST_SUBJECTS: tuple[int, ...] = (62, 63)

# Only subjects with id strictly greater than this are eligible for training or
# validation. Subjects with id <= MIN_SUBJECT_ID are excluded from both train and
# val in every split mode (independent of RESERVED_TEST_SUBJECTS).
MIN_SUBJECT_ID: int = 0


def split_train_val_dataset(
    dataset: Dataset,
    val_split: float,
    seed: int,
    train_subjects: Optional[list[int]] = None,
    logger=None,
) -> tuple[Dataset, Optional[Dataset]]:
    """Split dataset into train/val subsets, using subject-wise partitioning when requested.

    Subjects in RESERVED_TEST_SUBJECTS, and subjects with id <= MIN_SUBJECT_ID, are
    excluded from both train and val in every branch, even if explicitly listed in
    ``train_subjects``.
    """
    subject_ids = extract_subject_ids(dataset)
    reserved = set(RESERVED_TEST_SUBJECTS)

    def _excluded(sid: int) -> bool:
        # Subjects in the reserved test set, or with id <= MIN_SUBJECT_ID, are
        # never used for train or val in any split mode.
        return sid in reserved or sid <= MIN_SUBJECT_ID

    if subject_ids is not None and logger is not None:
        excluded_present = sorted({sid for sid in set(subject_ids) if _excluded(sid)})
        if excluded_present:
            logger.info(
                "Subjects excluded from train/val (reserved %s or id<=%d): %s",
                sorted(reserved), MIN_SUBJECT_ID, excluded_present,
            )

    if train_subjects:
        if subject_ids is None:
            raise ValueError("Subject-wise split requested, but dataset does not expose per-sample subject_ids.")
        train_subject_set = {int(sid) for sid in train_subjects if not _excluded(int(sid))}
        all_subjects = sorted(set(subject_ids))
        train_idx = [idx for idx, sid in enumerate(subject_ids) if sid in train_subject_set]
        val_idx = [
            idx
            for idx, sid in enumerate(subject_ids)
            if sid not in train_subject_set and not _excluded(sid)
        ]
        missing_subjects = sorted(train_subject_set.difference(all_subjects))
        if missing_subjects and logger is not None:
            logger.warning("Requested train subjects not present in dataset: %s", missing_subjects)
        if not train_idx:
            raise ValueError("Subject-wise split produced an empty training set.")
        if not val_idx:
            raise ValueError("Subject-wise split produced an empty validation set.")
        if logger is not None:
            logger.info("Subject-wise split enabled.")
            logger.info("Train subjects: %s", sorted(set(subject_ids[idx] for idx in train_idx)))
            logger.info("Val subjects: %s", sorted(set(subject_ids[idx] for idx in val_idx)))
            logger.info("Train samples=%s Val samples=%s", len(train_idx), len(val_idx))
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    # Random split: restrict the eligible pool to eligible subjects first.
    if subject_ids is not None:
        eligible = [idx for idx, sid in enumerate(subject_ids) if not _excluded(sid)]
    else:
        eligible = list(range(len(dataset)))

    if val_split <= 0.0:
        return Subset(dataset, eligible), None
    total = len(eligible)
    if total < 2:
        raise ValueError("Validation split requires at least 2 samples.")
    n_val = max(1, int(round(total * val_split)))
    n_val = min(n_val, total - 1)
    g = torch.Generator().manual_seed(seed)
    perm = [eligible[i] for i in torch.randperm(total, generator=g).tolist()]
    train_idx = perm[:-n_val]
    val_idx = perm[-n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _subset_base_and_indices(dataset: Dataset) -> tuple[Dataset, list[int]]:
    """Return the raw parent dataset and absolute indices for a possibly nested Subset."""
    if not isinstance(dataset, Subset):
        return dataset, list(range(len(dataset)))
    base, parent_indices = _subset_base_and_indices(dataset.dataset)
    indices = [int(i) for i in dataset.indices]
    return base, [parent_indices[i] for i in indices]


def _stats_from_tensor(tensor: torch.Tensor, indices: list[int]) -> dict[str, list[float]]:
    if len(indices) == 0:
        raise ValueError("Cannot compute IMU input stats from an empty training subset.")
    values = tensor[indices].reshape(-1, 3).float()
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(EPS)
    return {
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
    }


def validate_imu_input_stats(input_stats: dict | None) -> dict | None:
    """Validate and normalize serialized phone/watch IMU z-score stats."""
    if input_stats is None:
        return None
    if not isinstance(input_stats, dict):
        raise ValueError("input_stats must be a dict or None.")
    normalized: dict[str, dict[str, list[float]]] = {}
    for key in ("phone_accel", "watch_accel"):
        entry = input_stats.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"input_stats must contain {key!r} with mean/std entries.")
        mean = torch.as_tensor(entry.get("mean"), dtype=torch.float32)
        std = torch.as_tensor(entry.get("std"), dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError(f"input_stats[{key!r}] mean/std must each have shape [3].")
        std = std.clamp_min(EPS)
        normalized[key] = {
            "mean": [float(x) for x in mean.tolist()],
            "std": [float(x) for x in std.tolist()],
        }
    return normalized


def compute_imu_input_stats(train_subset: Dataset) -> dict[str, dict[str, list[float]]]:
    """Compute per-axis phone/watch IMU z-score stats from train samples only."""
    base, indices = _subset_base_and_indices(train_subset)
    if not hasattr(base, "phone_accel") or not hasattr(base, "watch_accel"):
        raise ValueError("IMU input stats require a dataset with phone_accel and watch_accel tensors.")
    return validate_imu_input_stats(
        {
            "phone_accel": _stats_from_tensor(getattr(base, "phone_accel"), indices),
            "watch_accel": _stats_from_tensor(getattr(base, "watch_accel"), indices),
        }
    )


def normalize_imu_accel(accel: torch.Tensor, stats: dict[str, list[float]]) -> torch.Tensor:
    """Apply serialized per-axis z-score stats to a [T,3] or [B,T,3] IMU tensor."""
    mean = torch.as_tensor(stats["mean"], dtype=accel.dtype, device=accel.device)
    std = torch.as_tensor(stats["std"], dtype=accel.dtype, device=accel.device).clamp_min(EPS)
    return (accel - mean) / std


def _fill_nan_with_column_mean(arr: np.ndarray) -> np.ndarray:
    """Replace NaN values with per-column means (or 0 if a full column is NaN)."""
    out = arr.astype(np.float32, copy=True)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D array, got {out.shape}")
    nan_mask = np.isnan(out)
    if nan_mask.any():
        col_mean = np.nanmean(out, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        out[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return out


def read_csv_files(folder: Optional[str]) -> Dict[str, pd.DataFrame]:
    """Read all CSV files from a folder into a filename->DataFrame map."""
    result: Dict[str, pd.DataFrame] = {}
    if folder is None or not os.path.isdir(folder):
        return result
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(folder, fname)
        try:
            result[fname] = pd.read_csv(path, header=None)
        except Exception:
            print(f"[dataset] Skipped empty/unreadable file: {fname}")
            continue
    return result


def _skeleton_frame_to_joints(frame_block: np.ndarray, joints: int = DEFAULT_JOINTS) -> np.ndarray:
    """Convert raw skeleton CSV rows into canonical [T, 21, 3] joint tensors."""
    if frame_block.ndim != 2:
        raise ValueError(f"Skeleton input must be 2D, got {frame_block.shape}")
    if frame_block.shape[1] in {SOURCE_JOINTS_32 * 3 + 1, DEFAULT_JOINTS * 3 + 1}:
        frame_block = frame_block[:, 1:]
    if frame_block.shape[1] == SOURCE_JOINTS_32 * 3:
        pose = frame_block.reshape(frame_block.shape[0], SOURCE_JOINTS_32, 3).astype(np.float32) / 1000.0
    elif frame_block.shape[1] == DEFAULT_JOINTS * 3:
        pose = frame_block.reshape(frame_block.shape[0], DEFAULT_JOINTS, 3).astype(np.float32) / 1000.0
    else:
        raise ValueError(
            f"Expected {SOURCE_JOINTS_32 * 3}/{SOURCE_JOINTS_32 * 3 + 1} or "
            f"{DEFAULT_JOINTS * 3}/{DEFAULT_JOINTS * 3 + 1} skeleton columns, got {frame_block.shape[1]}"
        )
    pose = project_skeleton_to_canonical_numpy(pose)
    if joints != pose.shape[1]:
        raise ValueError(f"Canonical skeleton has {pose.shape[1]} joints, but dataset requested {joints}")
    return pose


def _extract_sensor_3axis(df: pd.DataFrame) -> np.ndarray:
    """Extract a tri-axial sensor stream from a phone/watch CSV."""
    # SmartFallMM phone CSV schema: [timestamp_ms, iso_datetime, elapsed_s, x, y, z].
    if df.shape[1] >= 6:
        sensor_df = df.iloc[:, 3:6].apply(pd.to_numeric, errors="coerce")
        return _fill_nan_with_column_mean(sensor_df.values.astype(np.float32))

    # Watch CSV schema: [datetime, x, y, z].
    if df.shape[1] == 4:
        sensor_df = df.iloc[:, 1:4].apply(pd.to_numeric, errors="coerce")
        return _fill_nan_with_column_mean(sensor_df.values.astype(np.float32))

    # Strict fallback: accept files containing only 3 numeric axes.
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.loc[:, numeric_df.notna().any(axis=0)]
    if numeric_df.shape[1] == 3:
        return _fill_nan_with_column_mean(numeric_df.values.astype(np.float32))
    raise ValueError(
        "Sensor CSV schema mismatch: expected >=6 columns with sensor axes at [3:6], "
        f"or 4 columns with axes at [1:4], or exactly 3 numeric columns; got {df.shape[1]}"
    )


def _extract_watch_3axis(df: pd.DataFrame) -> np.ndarray:
    """Backward-compatible wrapper for generic 3-axis extraction."""
    return _extract_sensor_3axis(df)


def _extract_sensor_accel3(df: pd.DataFrame) -> np.ndarray:
    """Backward-compatible wrapper for generic 3-axis extraction."""
    return _extract_sensor_3axis(df)


def _resample_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a [T, C] array to [target_len, C] via linear interpolation."""
    if arr.shape[0] == target_len:
        return arr
    src_x = np.linspace(0.0, 1.0, arr.shape[0])
    dst_x = np.linspace(0.0, 1.0, target_len)
    return np.stack([np.interp(dst_x, src_x, arr[:, c]) for c in range(arr.shape[1])], axis=-1).astype(np.float32)


def _windowed(arr: np.ndarray, window: int, stride: int) -> Sequence[np.ndarray]:
    """Create sliding windows from a [T,...] array."""
    if arr.shape[0] < window:
        return []
    return [arr[s : s + window] for s in range(0, arr.shape[0] - window + 1, stride)]


def _default_gait_cache_dir(skeleton_folder: Optional[str]) -> str:
    if skeleton_folder:
        return os.path.join(os.path.abspath(skeleton_folder), f"_gait_cache_{GAIT_CACHE_VERSION}")
    return os.path.abspath(f"gait_cache_{GAIT_CACHE_VERSION}")


def _cached_or_compute_gait_metrics(
    skeleton: np.ndarray,
    cache_path: Optional[str],
    fps: float = DEFAULT_FPS,
    disable_cache: bool = False,
) -> np.ndarray:
    if cache_path and not disable_cache and os.path.isfile(cache_path):
        try:
            return load_gait_metrics_csv(cache_path)
        except Exception:
            pass
    vector, named = compute_gait_metrics_numpy(skeleton, fps=fps)
    if cache_path and not disable_cache:
        save_gait_metrics_csv(cache_path, named)
    return vector.astype(np.float32)


class CSVPairedGaitDataset(Dataset):
    """Dataset built from skeleton CSVs plus phone/watch accelerometer CSVs."""

    def __init__(
        self,
        skeleton_folder: str,
        phone_accel_folder: Optional[str] = None,
        watch_accel_folder: Optional[str] = None,
        window: int = DEFAULT_WINDOW,
        joints: int = DEFAULT_JOINTS,
        stride: int = 30,
        num_classes: int = DEFAULT_NUM_CLASSES,
        gait_cache_dir: Optional[str] = None,
        disable_gait_cache: bool = False,
        input_stats: dict | None = None,
    ) -> None:
        super().__init__()
        require_canonical_joint_count(joints, "CSVPairedGaitDataset")
        if not (phone_accel_folder and watch_accel_folder):
            raise ValueError("CSVPairedGaitDataset requires phone_accel_folder and watch_accel_folder.")
        self.window = window
        self.joints = joints
        self.fps = 30
        self.gait_cache_dir = gait_cache_dir or _default_gait_cache_dir(skeleton_folder=skeleton_folder)
        self.disable_gait_cache = disable_gait_cache
        self.input_stats = validate_imu_input_stats(input_stats)
        os.makedirs(self.gait_cache_dir, exist_ok=True)

        skeleton_map = read_csv_files(skeleton_folder)
        phone_accel_map = read_csv_files(phone_accel_folder)
        watch_accel_map = read_csv_files(watch_accel_folder)
        common = sorted(
            set(skeleton_map)
            .intersection(phone_accel_map)
            .intersection(watch_accel_map)
        )
        if not common:
            raise ValueError("No common CSV filenames across skeleton/phone_accel/watch_accel folders")

        x_windows = []
        phone_accel_windows: list[np.ndarray] = []
        watch_accel_windows: list[np.ndarray] = []
        gait_metric_windows = []
        labels = []
        subject_ids = []
        skipped_parse = 0
        skipped_short = 0

        for fname in common:
            try:
                skel = _skeleton_frame_to_joints(_fill_nan_with_column_mean(skeleton_map[fname].values), joints=joints)
                phone_accel = _extract_sensor_3axis(phone_accel_map[fname])
                watch_accel = _extract_sensor_3axis(watch_accel_map[fname])
                # Resample both accelerometer streams onto the skeleton timeline so all
                # three streams share the skeleton's frame count. This handles phone/watch
                # IMU rates that differ from the skeleton rate, whereas naive truncation to
                # the shortest length misaligns streams sampled at different rates.
                t_skel = skel.shape[0]
                phone_accel = _resample_to_length(phone_accel, t_skel)
                watch_accel = _resample_to_length(watch_accel, t_skel)
                cache_path = os.path.join(self.gait_cache_dir, fname)
                gait_metrics = _cached_or_compute_gait_metrics(
                    skel,
                    cache_path=cache_path,
                    fps=float(self.fps),
                    disable_cache=disable_gait_cache,
                )
                if gait_metrics.shape[0] != DEFAULT_GAIT_METRICS_DIM:
                    raise ValueError(
                        f"Expected auto-computed gait_metrics dim {DEFAULT_GAIT_METRICS_DIM}, got {gait_metrics.shape[0]}"
                    )
            except Exception:
                print(f"[dataset] Skipped invalid paired sample: {fname}")
                skipped_parse += 1
                continue

            skel_w = _windowed(skel, window, stride)
            phone_accel_w = _windowed(phone_accel, window, stride)
            watch_accel_w = _windowed(watch_accel, window, stride)
            n = min(len(skel_w), len(phone_accel_w), len(watch_accel_w))
            if n == 0:
                skipped_short += 1
            for i in range(n):
                x_windows.append(skel_w[i])
                phone_accel_windows.append(phone_accel_w[i])
                watch_accel_windows.append(watch_accel_w[i])
                gait_metric_windows.append(gait_metrics)
                labels.append(_parse_label_14(fname, num_classes=num_classes))
                subject_ids.append(_parse_subject_id(fname))

        if len(x_windows) == 0:
            raise ValueError(
                "No valid paired windows found in CSV folders "
                f"(common_files={len(common)}, parse_failed={skipped_parse}, too_short_for_window={skipped_short}, "
                f"window={window}, stride={stride})"
            )

        self.skeleton = torch.tensor(np.stack(x_windows), dtype=torch.float32)
        self.phone_accel = torch.tensor(np.stack(phone_accel_windows), dtype=torch.float32)
        self.watch_accel = torch.tensor(np.stack(watch_accel_windows), dtype=torch.float32)
        self.gait_metrics = torch.tensor(np.stack(gait_metric_windows), dtype=torch.float32)
        self.label = torch.tensor(labels, dtype=torch.long)
        self.subject_ids = torch.tensor(subject_ids, dtype=torch.long)

        assert_shape(self.skeleton, [None, window, joints, 3], "CSVPairedGaitDataset.skeleton")
        assert_shape(self.phone_accel, [self.skeleton.shape[0], window, 3], "CSVPairedGaitDataset.phone_accel")
        assert_shape(self.watch_accel, [self.skeleton.shape[0], window, 3], "CSVPairedGaitDataset.watch_accel")
        assert_shape(self.gait_metrics, [self.skeleton.shape[0], None], "CSVPairedGaitDataset.gait_metrics")
        assert_shape(self.label, [self.skeleton.shape[0]], "CSVPairedGaitDataset.label")
        assert_shape(self.subject_ids, [self.skeleton.shape[0]], "CSVPairedGaitDataset.subject_ids")
        self._fps_tensor = torch.tensor(self.fps, dtype=torch.long)
        self._joint_labels = list(get_joint_labels())

    def __len__(self) -> int:
        """Return number of samples."""
        return self.skeleton.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one sample dictionary."""
        phone_accel = self.phone_accel[idx]
        watch_accel = self.watch_accel[idx]
        if self.input_stats is not None:
            phone_accel = normalize_imu_accel(phone_accel, self.input_stats["phone_accel"])
            watch_accel = normalize_imu_accel(watch_accel, self.input_stats["watch_accel"])
        sample: Dict[str, torch.Tensor] = {
            "skeleton": self.skeleton[idx],
            "phone_accel": phone_accel,
            "watch_accel": watch_accel,
            "gait_metrics": self.gait_metrics[idx],
            "label": self.label[idx],
            "fps": self._fps_tensor,
            "joint_labels": self._joint_labels,
        }
        return sample


def create_dataset(
    window: int = DEFAULT_WINDOW,
    joints: int = DEFAULT_JOINTS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    skeleton_folder: Optional[str] = None,
    phone_accel_folder: Optional[str] = None,
    watch_accel_folder: Optional[str] = None,
    stride: int = 30,
    gait_cache_dir: Optional[str] = None,
    disable_gait_cache: bool = False,
    input_stats: dict | None = None,
) -> Dataset:
    """Create dataset object from paired CSV folders."""
    if skeleton_folder and phone_accel_folder and watch_accel_folder:
        return CSVPairedGaitDataset(
            skeleton_folder=skeleton_folder,
            phone_accel_folder=phone_accel_folder,
            watch_accel_folder=watch_accel_folder,
            window=window,
            joints=joints,
            stride=stride,
            num_classes=num_classes,
            gait_cache_dir=gait_cache_dir,
            disable_gait_cache=disable_gait_cache,
            input_stats=input_stats,
        )
    raise ValueError(
        "Requires --skeleton_folder with (--phone_accel_folder + --watch_accel_folder)."
    )


def create_dataloader(
    batch_size: int,
    shuffle: bool = True,
    window: int = DEFAULT_WINDOW,
    joints: int = DEFAULT_JOINTS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    skeleton_folder: Optional[str] = None,
    phone_accel_folder: Optional[str] = None,
    watch_accel_folder: Optional[str] = None,
    stride: int = 30,
    num_workers: int = 0,
    pin_memory: bool = True,
    sampler: Optional[Sampler] = None,
    dataset: Optional[Dataset] = None,
    drop_last: bool = True,
    gait_cache_dir: Optional[str] = None,
    disable_gait_cache: bool = False,
    input_stats: dict | None = None,
) -> DataLoader:
    """Create dataloader from paired CSV folders."""
    if dataset is None:
        dataset = create_dataset(
            window=window,
            joints=joints,
            num_classes=num_classes,
            skeleton_folder=skeleton_folder,
            phone_accel_folder=phone_accel_folder,
            watch_accel_folder=watch_accel_folder,
            stride=stride,
            gait_cache_dir=gait_cache_dir,
            disable_gait_cache=disable_gait_cache,
            input_stats=input_stats,
        )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "sampler": sampler,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)
