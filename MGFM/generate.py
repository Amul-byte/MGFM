"""Flow-matching IMU-to-skeleton generation with 3D GIF visualization."""

from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:  # pragma: no cover - optional visualization dependency
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:  # pragma: no cover - optional visualization dependency
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:  # pragma: no cover - fallback only when imageio is unavailable
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from diffusion_model.dataset import (
    RESERVED_TEST_SUBJECTS,
    _fill_nan_with_column_mean,
    _parse_label_14,
    _skeleton_frame_to_joints,
    _windowed,
    create_dataloader,
    create_dataset,
    extract_subject_ids,
    parse_subject_list,
    read_csv_files,
    validate_imu_input_stats,
)
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM, compute_gait_metrics_torch, save_gait_metrics_csv
from diffusion_model.model import IMUConditionedSkeletonFlowModel
from diffusion_model.model_loader import load_checkpoint
from diffusion_model.util import (
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_ODE_STEPS,
    DEFAULT_WINDOW,
    INPUT_NORM_SCHEMA_NONE,
    INPUT_NORM_SCHEMA_ZSCORE,
    JOINT_LABELS,
    get_skeleton_edges,
)


CANONICAL_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(get_skeleton_edges())


def _input_norm_schema_from_arg(input_norm: str) -> str:
    if input_norm == "none":
        return INPUT_NORM_SCHEMA_NONE
    if input_norm == "zscore":
        return INPUT_NORM_SCHEMA_ZSCORE
    raise ValueError(f"Unsupported --input_norm={input_norm!r}")


def activity_name(label: int | torch.Tensor | np.integer) -> str:
    """Return canonical activity name for a zero-based class label."""
    label_int = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
    return f"activity_A{label_int + 1:02d}"


def set_seed(seed: int) -> None:
    """Set all random seeds used by this inference script."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_for_nans(tensor: torch.Tensor, name: str) -> None:
    """Print a compact finite-value check for generated tensors."""
    if torch.isnan(tensor).any():
        print(f"NaNs detected in {name}")
    elif torch.isinf(tensor).any():
        print(f"Infs detected in {name}")
    else:
        print(f"No NaNs/Infs in {name}")


def _as_skeleton_array(positions: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert [B,T,J,3], [T,J,3], [B,T,63], or [T,63] to [B,T,21,3]."""
    if isinstance(positions, torch.Tensor):
        arr = positions.detach().cpu().numpy()
    else:
        arr = np.asarray(positions)
    if arr.ndim == 2 and arr.shape[-1] == DEFAULT_JOINTS * 3:
        arr = arr.reshape(1, arr.shape[0], DEFAULT_JOINTS, 3)
    elif arr.ndim == 3 and arr.shape[-1] == DEFAULT_JOINTS * 3:
        arr = arr.reshape(arr.shape[0], arr.shape[1], DEFAULT_JOINTS, 3)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[None, ...]
    if arr.ndim != 4 or arr.shape[2] != DEFAULT_JOINTS or arr.shape[3] != 3:
        raise ValueError(
            f"Expected skeleton shape [B,T,{DEFAULT_JOINTS},3] or flattened "
            f"[B,T,{DEFAULT_JOINTS * 3}], got {arr.shape}."
        )
    return arr.astype(np.float32, copy=False)


def _axis_limits(sequence: np.ndarray) -> tuple[np.ndarray, float]:
    finite = sequence[np.isfinite(sequence).all(axis=-1)]
    if finite.size == 0:
        raise ValueError("Sequence has no finite XYZ coordinates to render.")
    min_xyz = np.percentile(finite, 2.0, axis=0)
    max_xyz = np.percentile(finite, 98.0, axis=0)
    center = (min_xyz + max_xyz) * 0.5
    radius = float(np.max(max_xyz - min_xyz) * 0.6)
    radius = max(radius, 1e-3)
    return center, radius


def save_skeleton_csv(
    positions: torch.Tensor | np.ndarray,
    save_path: str,
    sample_idx: int = 0,
    label: int | None = None,
    activity: str | None = None,
) -> None:
    """Save one generated [T,21,3] skeleton as raw frame rows.

    Each row is 63 comma-separated numbers in the canonical 21-joint
    SmartFall layout.
    """
    skeletons = _as_skeleton_array(positions)
    if sample_idx < 0 or sample_idx >= skeletons.shape[0]:
        raise IndexError(f"sample_idx={sample_idx} out of range for batch size {skeletons.shape[0]}.")
    sequence = skeletons[sample_idx]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for frame in sequence:
            writer.writerow(frame.reshape(DEFAULT_JOINTS * 3).tolist())
    print(f"CSV saved as {save_path}")


def save_labels_csv(
    labels: torch.Tensor | np.ndarray,
    save_path: str,
) -> None:
    """Save a summary CSV mapping sample index -> label -> activity name."""
    if isinstance(labels, torch.Tensor):
        labels_arr = labels.detach().cpu().numpy()
    else:
        labels_arr = np.asarray(labels)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_idx", "label", "activity"])
        for sample_idx, label in enumerate(labels_arr.tolist()):
            writer.writerow([sample_idx, int(label), activity_name(int(label))])
    print(f"labels CSV saved as {save_path}")


def visualize_skeleton(
    positions: torch.Tensor | np.ndarray,
    save_path: str = "skeleton_animation.gif",
    sample_idx: int = 0,
    fps: int = 30,
    elev: float = -90.0,
    azim: float = -90.0,
    canvas_inches: float = 8.0,
    label_joints: bool = True,
    title: str | None = None,
) -> None:
    """Render one generated [T,21,3] skeleton as a 3D GIF."""
    if plt is None:
        raise RuntimeError("matplotlib is required for 3D skeleton GIF rendering.")
    skeletons = _as_skeleton_array(positions)
    if sample_idx < 0 or sample_idx >= skeletons.shape[0]:
        raise IndexError(f"sample_idx={sample_idx} out of range for batch size {skeletons.shape[0]}.")

    sequence = skeletons[sample_idx]
    center, radius = _axis_limits(sequence)
    frames: list[np.ndarray] = []

    for frame_idx in range(sequence.shape[0]):
        fig = plt.figure(figsize=(canvas_inches, canvas_inches))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")
        ax.grid(False)
        ax.set_axis_off()

        frame = sequence[frame_idx]
        for joint1, joint2 in CANONICAL_CONNECTIONS:
            joint1_coords = frame[joint1]
            joint2_coords = frame[joint2]
            if not np.isfinite(joint1_coords).all() or not np.isfinite(joint2_coords).all():
                continue
            xs = [joint1_coords[0], joint2_coords[0]]
            ys = [joint1_coords[1], joint2_coords[1]]
            zs = [joint1_coords[2], joint2_coords[2]]
            ax.plot(xs, ys, zs, marker="o", color="darkblue", linewidth=2.0)
            ax.scatter(joint1_coords[0], joint1_coords[1], joint1_coords[2], color="red", s=45)
            ax.scatter(joint2_coords[0], joint2_coords[1], joint2_coords[2], color="red", s=45)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        if label_joints:
            for joint_idx, coords in enumerate(frame):
                if not np.isfinite(coords).all():
                    continue
                label = JOINT_LABELS[joint_idx] if joint_idx < len(JOINT_LABELS) else f"J{joint_idx}"
                ax.text(
                    coords[0],
                    coords[1],
                    coords[2],
                    f"{joint_idx}:{label}",
                    color="black",
                    fontsize=7,
                )
        if title:
            fig.suptitle(f"{title}  |  frame {frame_idx}", fontsize=12)

        plt.tight_layout()
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(image)
        plt.close(fig)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    duration = 1.0 / max(int(fps), 1)
    if imageio is not None:
        imageio.mimsave(save_path, frames, duration=duration)
    elif Image is not None:
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(duration * 1000),
            loop=0,
        )
    else:
        raise RuntimeError("Either imageio or PIL is required to save GIF files.")
    print(f"GIF saved as {save_path}")


def save_skeleton_panel(
    sequence: torch.Tensor | np.ndarray,
    save_path: str,
    num_frames: int = 5,
    elev: float = -90.0,
    azim: float = -90.0,
    label_joints: bool = True,
    title: str | None = None,
) -> None:
    """Save a static 3D panel for quick inspection."""
    if plt is None:
        raise RuntimeError("matplotlib is required for skeleton panel rendering.")
    skeleton = _as_skeleton_array(sequence)[0]
    num_frames = max(1, min(int(num_frames), skeleton.shape[0]))
    frame_ids = np.linspace(0, skeleton.shape[0] - 1, num=num_frames, dtype=int)
    center, radius = _axis_limits(skeleton)
    fig = plt.figure(figsize=(4.0 * num_frames, 4.0))

    for panel_idx, frame_idx in enumerate(frame_ids.tolist(), start=1):
        ax = fig.add_subplot(1, num_frames, panel_idx, projection="3d")
        ax.set_facecolor("white")
        ax.grid(False)
        ax.set_axis_off()
        frame = skeleton[frame_idx]
        for joint1, joint2 in CANONICAL_CONNECTIONS:
            p1 = frame[joint1]
            p2 = frame[joint2]
            if not np.isfinite(p1).all() or not np.isfinite(p2).all():
                continue
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], marker="o", color="darkblue")
            ax.scatter(p1[0], p1[1], p1[2], color="red", s=25)
            ax.scatter(p2[0], p2[1], p2[2], color="red", s=25)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        if label_joints:
            for joint_idx, coords in enumerate(frame):
                if not np.isfinite(coords).all():
                    continue
                label = JOINT_LABELS[joint_idx] if joint_idx < len(JOINT_LABELS) else f"J{joint_idx}"
                ax.text(
                    coords[0],
                    coords[1],
                    coords[2],
                    f"{joint_idx}:{label}",
                    color="black",
                    fontsize=6,
                )
        ax.set_title(f"t={frame_idx}")

    if title:
        fig.suptitle(title, fontsize=14)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"panel saved as {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and visualize skeleton samples from phone/watch IMU inputs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--imu_flow_ckpt", type=str, default="", help="Stage-5 IMU-conditioned skeleton-space flow checkpoint.")
    parser.add_argument("--skeleton_folder", type=str, default="")
    parser.add_argument("--phone_accel_folder", type=str, default="", help="Phone accelerometer CSV folder.")
    parser.add_argument("--watch_accel_folder", type=str, default="", help="Watch accelerometer CSV folder.")
    parser.add_argument(
        "--eval_subjects",
        type=str,
        default="",
        help="Comma-separated subjects to run inference on. Defaults to the reserved "
        f"held-out test subjects {','.join(str(s) for s in RESERVED_TEST_SUBJECTS)}. "
        "Pass 'all' to disable filtering and run on every subject.",
    )
    parser.add_argument("--gait_cache_dir", type=str, default="")
    parser.add_argument("--disable_gait_cache", action="store_true")
    parser.add_argument(
        "--input_norm",
        choices=["none", "zscore"],
        default="none",
        help="IMU-only input normalization expected by the Stage-5 checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--gait_metrics_dim", type=int, default=DEFAULT_GAIT_METRICS_DIM)
    parser.add_argument("--ode_steps", type=int, default=DEFAULT_ODE_STEPS, help="Euler ODE sampling steps for flow matching.")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale (1 disables CFG).")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--imu_graph", choices=["chain", "multiscale"], default=None,
                        help="IMU temporal graph type (GCN encoder only). Default 'multiscale' when --imu_encoder_type=gcn; "
                             "passing this with --imu_encoder_type=cnn raises an error.")
    parser.add_argument("--imu_encoder_type", choices=["gcn", "cnn"], default="gcn",
                        help="IMU encoder architecture used to build the model for checkpoint loading.")
    cnn_attn_group = parser.add_mutually_exclusive_group()
    cnn_attn_group.add_argument("--imu_cnn_attention", dest="imu_cnn_attention", action="store_true")
    cnn_attn_group.add_argument("--no_imu_cnn_attention", dest="imu_cnn_attention", action="store_false")
    parser.set_defaults(imu_cnn_attention=True)
    parser.add_argument("--stage2_dropout", type=float, default=0.25, help="Dropout rate used by the IMU encoder.")
    parser.add_argument("--skeleton_graph_op", choices=["gat", "gcn"], default="gcn")
    parser.add_argument("--temporal_block_type", choices=["conv", "attention"], default="conv")
    parser.add_argument("--denoiser_depth", type=int, default=6)
    parser.add_argument("--denoiser_num_heads", type=int, default=8)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs/results_new")
    parser.add_argument("--save_tensor", action="store_true", help="Save generated and reference tensors to output_dir.")
    parser.add_argument("--ground_truth_only", action="store_true", help="Render real dataset skeletons without loading model checkpoints.")
    parser.add_argument("--save_gif", action="store_true", help="Save 3D skeleton GIFs.")
    parser.add_argument("--save_panel", action="store_true", help="Save 3D skeleton panel PNGs.")
    parser.add_argument("--save_csv", action="store_true", help="Save per-sample 21x3 skeleton CSVs and a labels.csv summary.")
    parser.add_argument(
        "--per_activity_target",
        type=int,
        default=3,
        help="Keep iterating batches until each observed activity has saved this many GIFs/CSVs (0 disables per-activity mode).",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=200,
        help="Safety cap on how many dataloader batches to consume when searching for activity coverage.",
    )
    parser.add_argument("--gif_dir", type=str, default="", help="Directory for generated GIF and panel files. Defaults to output_dir/gifs.")
    parser.add_argument("--csv_dir", type=str, default="", help="Directory for generated skeleton CSV files. Defaults to output_dir/csv.")
    parser.add_argument("--gif_prefix", type=str, default="sample")
    parser.add_argument("--gif_fps", type=int, default=30)
    parser.add_argument("--panel_frames", type=int, default=5)
    parser.add_argument("--max_gifs", type=int, default=1)
    parser.add_argument("--view_elev", type=float, default=-90.0)
    parser.add_argument("--view_azim", type=float, default=-90.0)
    parser.add_argument("--no_joint_labels", action="store_true", help="Disable joint index/name labels in GIFs and panels.")
    return parser.parse_args()



def generate_batch_imu_flow(
    args: argparse.Namespace,
    model: IMUConditionedSkeletonFlowModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run one batch through the Stage-5 flow model and return generated + reference tensors."""
    phone_accel = batch["phone_accel"].to(device, non_blocking=True)
    watch_accel = batch["watch_accel"].to(device, non_blocking=True)
    real = batch["skeleton"].detach().cpu()

    generated = model.sample(
        phone_accel=phone_accel,
        watch_accel=watch_accel,
        n_steps=args.ode_steps,
        cfg_scale=args.cfg_scale,
    ).detach().cpu()

    return {
        "generated": generated,
        "real": real,
        "gait_target": batch["gait_metrics"].detach().cpu(),
        "labels": batch["label"].detach().cpu(),
    }


def load_imu_flow_model(args: argparse.Namespace, device: torch.device) -> IMUConditionedSkeletonFlowModel:
    if args.imu_encoder_type == "cnn" and args.imu_graph is not None:
        raise ValueError(
            "--imu_graph is only meaningful with --imu_encoder_type=gcn; drop one of the two flags."
        )
    imu_graph_resolved = args.imu_graph if args.imu_graph is not None else "multiscale"
    model = IMUConditionedSkeletonFlowModel(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
        imu_graph_type=imu_graph_resolved,
        stage2_dropout=args.stage2_dropout,
        skeleton_graph_op=args.skeleton_graph_op,
        temporal_block_type=args.temporal_block_type,
        denoiser_depth=args.denoiser_depth,
        denoiser_num_heads=args.denoiser_num_heads,
        cfg_dropout=0.0,
        imu_encoder_type=args.imu_encoder_type,
        imu_cnn_attention=args.imu_cnn_attention,
        input_norm_schema=_input_norm_schema_from_arg(args.input_norm),
    ).to(device)
    checkpoint = load_checkpoint(args.imu_flow_ckpt, model, strict=True)
    extra = checkpoint.get("extra", {}) if isinstance(checkpoint, dict) else {}
    if _input_norm_schema_from_arg(args.input_norm) == INPUT_NORM_SCHEMA_ZSCORE:
        args.input_stats = validate_imu_input_stats(extra.get("input_stats"))
        if args.input_stats is None:
            raise ValueError(f"Checkpoint {args.imu_flow_ckpt!r} is z-score normalized but lacks input_stats.")
    else:
        args.input_stats = None
    model.eval()
    return model


def build_dataloader(args: argparse.Namespace):
    dataset = create_dataset(
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        phone_accel_folder=args.phone_accel_folder or None,
        watch_accel_folder=args.watch_accel_folder or None,
        stride=args.stride,
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
        input_stats=getattr(args, "input_stats", None),
    )

    # Restrict inference to the held-out test subjects (default: 62, 63) so the
    # reported reconstruction error is measured on never-trained subjects only.
    # Pass --eval_subjects all to disable filtering entirely.
    if args.eval_subjects.strip().lower() == "all":
        print(f"Inference on ALL subjects (no held-out filter): {len(dataset)} windows.")
    else:
        eval_subjects = parse_subject_list(args.eval_subjects) if args.eval_subjects else list(RESERVED_TEST_SUBJECTS)
        subject_ids = extract_subject_ids(dataset)
        if subject_ids is None:
            raise ValueError("Subject-filtered inference requested, but the dataset does not expose per-sample subject_ids.")
        keep = {int(s) for s in eval_subjects}
        indices = [i for i, sid in enumerate(subject_ids) if sid in keep]
        if not indices:
            raise ValueError(
                f"No samples found for eval subjects {sorted(keep)}. Present in dataset: {sorted(set(subject_ids))}."
            )
        print(f"Inference restricted to subjects {sorted(keep)}: {len(indices)} windows.")
        dataset = Subset(dataset, indices)

    return create_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        shuffle=True,
        drop_last=False,
        dataset=dataset,
        input_stats=getattr(args, "input_stats", None),
    )


class GroundTruthSkeletonDataset(Dataset):
    """Ground-truth skeleton windows for visualization without sensor pairing."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.samples: list[torch.Tensor] = []
        self.labels: list[int] = []

        if not args.skeleton_folder:
            raise ValueError("--ground_truth_only requires --skeleton_folder.")
        skeleton_map = read_csv_files(args.skeleton_folder)
        if not skeleton_map:
            raise ValueError(f"No skeleton CSV files found in {args.skeleton_folder!r}")
        for fname, df in sorted(skeleton_map.items()):
            skel = _skeleton_frame_to_joints(_clean_skeleton_csv_frame(df), joints=args.joints)
            windows = _windowed(skel, args.window, args.stride)
            label = _parse_label_14(fname, args.num_classes)
            for window in windows:
                self.samples.append(torch.tensor(window, dtype=torch.float32))
                self.labels.append(label)

        if not self.samples:
            raise ValueError(
                "No ground-truth skeleton windows were created. Check --window/--stride and confirm skeleton files are long enough."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        skeleton = self.samples[idx]
        return {
            "skeleton": skeleton,
            "gait_metrics": torch.zeros(DEFAULT_GAIT_METRICS_DIM, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _clean_skeleton_csv_frame(df: Any) -> np.ndarray:
    """Convert header/index-bearing skeleton CSV frames into numeric arrays."""
    cleaned = df.apply(lambda col: col.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan}))
    numeric = cleaned.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError("Skeleton CSV contains no numeric coordinate data after dropping headers/index columns.")
    return _fill_nan_with_column_mean(numeric.to_numpy())


def build_ground_truth_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = GroundTruthSkeletonDataset(args)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        drop_last=False,
    )



def _samples_from_ground_truth_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    skeleton = batch["skeleton"].detach().cpu()
    return {
        "generated": skeleton,
        "real": skeleton,
        "gait_target": batch["gait_metrics"].detach().cpu(),
        "labels": batch["label"].detach().cpu(),
    }


def main() -> None:
    args = parse_args()
    if args.gait_metrics_dim != DEFAULT_GAIT_METRICS_DIM:
        raise ValueError(
            f"--gait_metrics_dim must equal the fixed auto-computed gait-summary size ({DEFAULT_GAIT_METRICS_DIM})."
        )
    if args.ode_steps <= 0:
        raise ValueError("--ode_steps must be positive.")
    if args.cfg_scale <= 0.0:
        raise ValueError("--cfg_scale must be positive.")
    if not args.ground_truth_only and not args.imu_flow_ckpt:
        raise ValueError("--imu_flow_ckpt is required unless --ground_truth_only is set.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    imu_flow_model = None
    if not args.ground_truth_only:
        imu_flow_model = load_imu_flow_model(args, device)
        loader = build_dataloader(args)
    else:
        loader = build_ground_truth_dataloader(args)

    run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    gen_skel_dir = output_dir / "generated_skeleton"
    real_skel_dir = output_dir / "real_skeleton"
    gen_gait_dir = output_dir / "generated_gait"
    real_gait_dir = output_dir / "real_gait"
    gen_gif_dir = output_dir / "generated_gifs"
    real_gif_dir = output_dir / "real_gifs"
    for _d in [output_dir, gen_skel_dir, real_skel_dir, gen_gait_dir, real_gait_dir, gen_gif_dir, real_gif_dir]:
        _d.mkdir(parents=True, exist_ok=True)

    target = max(int(args.per_activity_target), 0)
    num_classes = int(args.num_classes)
    gif_counts: dict[int, int] = {}
    csv_counts: dict[int, int] = {}
    panel_counts: dict[int, int] = {}
    labels_seen: list[int] = []
    labels_order: list[int] = []
    recon_errors: list[float] = []
    gait_errors: list[float] = []
    first_batch_saved_tensor = False

    if args.ground_truth_only:
        print("Rendering ground-truth skeletons from the dataset...")
    else:
        print("Generating samples based on phone/watch accelerometer inputs...")
    want_any = args.save_gif or args.save_panel or args.save_csv

    def _coverage_complete() -> bool:
        if target <= 0 or not want_any:
            return True
        observed = set(labels_seen)
        if len(observed) < num_classes:
            return False
        if args.save_gif and any(gif_counts.get(c, 0) < target for c in observed):
            return False
        if args.save_panel and any(panel_counts.get(c, 0) < target for c in observed):
            return False
        if args.save_csv and any(csv_counts.get(c, 0) < target for c in observed):
            return False
        return True

    max_batches = max(int(args.max_batches), 1)
    batch_idx = 0
    epoch = 0
    stop = False
    while batch_idx < max_batches and not stop:
        epoch += 1
        produced_any = False
        for batch in loader:
            produced_any = True
            if batch_idx >= max_batches:
                print(f"reached --max_batches={max_batches}; stopping.")
                stop = True
                break

            if args.ground_truth_only:
                samples = _samples_from_ground_truth_batch(batch)
            else:
                assert imu_flow_model is not None
                samples = generate_batch_imu_flow(args, imu_flow_model, batch, device)
            generated = samples["generated"]
            real = samples["real"]
            labels = samples["labels"]

            gait_pred = compute_gait_metrics_torch(generated.to(device), fps=30.0).cpu()
            real_gait_window = compute_gait_metrics_torch(real.to(device), fps=30.0).cpu()
            if not args.ground_truth_only:
                recon_errors.append(torch.linalg.norm(generated - real, dim=-1).mean().item())
                gait_errors.append(torch.mean((gait_pred - real_gait_window) ** 2).item())

            if args.save_tensor and not first_batch_saved_tensor:
                first_activity = activity_name(labels[0]) if len(labels) > 0 else "activity_unknown"
                tensor_path = output_dir / f"{args.gif_prefix}_{first_activity}_{run_tag}_generated.pt"
                torch.save(samples, tensor_path)
                print(f"saved generated tensors: {tensor_path}")
                first_batch_saved_tensor = True

            for sample_idx in range(generated.shape[0]):
                label_int = int(labels[sample_idx].item())
                if label_int not in labels_seen:
                    labels_seen.append(label_int)
                activity = activity_name(label_int)
                title_prefix = "ground truth" if args.ground_truth_only else "generated"
                title = f"{activity} {title_prefix} (label={label_int})"

                if args.save_csv and (target <= 0 or csv_counts.get(label_int, 0) < target):
                    slot = csv_counts.get(label_int, 0)
                    stem = f"{args.gif_prefix}_{activity}_{run_tag}_take{slot:02d}"
                    save_skeleton_csv(
                        generated,
                        save_path=str(gen_skel_dir / f"{stem}.csv"),
                        sample_idx=sample_idx,
                        label=label_int,
                        activity=activity,
                    )
                    if not args.ground_truth_only:
                        save_skeleton_csv(
                            real,
                            save_path=str(real_skel_dir / f"{stem}.csv"),
                            sample_idx=sample_idx,
                            label=label_int,
                            activity=activity,
                        )
                    save_gait_metrics_csv(str(real_gait_dir / f"{stem}.csv"), real_gait_window[sample_idx].tolist())
                    if not args.ground_truth_only:
                        save_gait_metrics_csv(str(gen_gait_dir / f"{stem}.csv"), gait_pred[sample_idx].tolist())
                    csv_counts[label_int] = slot + 1
                    labels_order.append(label_int)

                if args.save_gif and (target <= 0 or gif_counts.get(label_int, 0) < target):
                    slot = gif_counts.get(label_int, 0)
                    stem = f"{args.gif_prefix}_{activity}_{run_tag}_take{slot:02d}"
                    visualize_skeleton(
                        generated,
                        save_path=str(gen_gif_dir / f"{stem}.gif"),
                        sample_idx=sample_idx,
                        fps=args.gif_fps,
                        elev=args.view_elev,
                        azim=args.view_azim,
                        label_joints=not args.no_joint_labels,
                        title=title,
                    )
                    if not args.ground_truth_only:
                        visualize_skeleton(
                            real,
                            save_path=str(real_gif_dir / f"{stem}.gif"),
                            sample_idx=sample_idx,
                            fps=args.gif_fps,
                            elev=args.view_elev,
                            azim=args.view_azim,
                            label_joints=not args.no_joint_labels,
                            title=f"{activity} real (label={label_int})",
                        )
                    gif_counts[label_int] = slot + 1

                if args.save_panel and (target <= 0 or panel_counts.get(label_int, 0) < target):
                    slot = panel_counts.get(label_int, 0)
                    stem = f"{args.gif_prefix}_{activity}_{run_tag}_take{slot:02d}"
                    save_skeleton_panel(
                        generated[sample_idx],
                        save_path=str(gen_gif_dir / f"{stem}_panel.png"),
                        num_frames=args.panel_frames,
                        elev=args.view_elev,
                        azim=args.view_azim,
                        label_joints=not args.no_joint_labels,
                        title=title,
                    )
                    if not args.ground_truth_only:
                        save_skeleton_panel(
                            real[sample_idx],
                            save_path=str(real_gif_dir / f"{stem}_panel.png"),
                            num_frames=args.panel_frames,
                            elev=args.view_elev,
                            azim=args.view_azim,
                            label_joints=not args.no_joint_labels,
                            title=f"{activity} real (label={label_int})",
                        )
                    panel_counts[label_int] = slot + 1

            batch_idx += 1
            if _coverage_complete():
                print(f"coverage target reached after {batch_idx} batches ({epoch} loader passes).")
                stop = True
                break

        if not produced_any:
            print("dataloader produced no batches; stopping.")
            break

    if recon_errors:
        print(f"mean reconstruction joint error: {float(np.mean(recon_errors)):.6f}  (over {len(recon_errors)} batches)")
        print(f"mean gait MSE: {float(np.mean(gait_errors)):.6f}")

    if args.save_csv and labels_order:
        summary_path = output_dir / f"labels_{run_tag}.csv"
        save_labels_csv(torch.tensor(labels_order, dtype=torch.long), str(summary_path))

    if want_any:
        print("per-activity saved counts:")
        for label_int in sorted(set(labels_seen)):
            print(
                f"  {activity_name(label_int):>14}  gif={gif_counts.get(label_int, 0)}  "
                f"panel={panel_counts.get(label_int, 0)}  csv={csv_counts.get(label_int, 0)}"
            )
        missing = [c for c in range(num_classes) if c not in labels_seen]
        if missing:
            print(f"never observed in loader: {[activity_name(c) for c in missing]}")


if __name__ == "__main__":
    main()
