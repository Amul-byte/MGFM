"""Training entrypoint for IMU contrastive pretraining and skeleton-space flow."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from diffusion_model.dataset import (
    compute_imu_input_stats,
    create_dataloader,
    create_dataset,
    parse_subject_list,
    split_train_val_dataset,
    validate_imu_input_stats,
)
from diffusion_model.gait_metrics import DEFAULT_GAIT_METRICS_DIM, GAIT_METRIC_NAMES
from diffusion_model.losses import JOINT_ANGLE_NAMES
from diffusion_model.model import (
    IMUConditionedSkeletonFlowModel,
    IMUSkeletonContrastivePretrainModel,
)
from diffusion_model.model_loader import save_checkpoint
from diffusion_model.sensor_model import IMU_FEATURE_NAMES
from diffusion_model.training_eval import (
    save_run_manifest,
    write_history,
)
from diffusion_model.util import (
    DEFAULT_FPS,
    DEFAULT_JOINTS,
    DEFAULT_LATENT_DIM,
    DEFAULT_NUM_CLASSES,
    DEFAULT_ODE_STEPS,
    DEFAULT_WINDOW,
    INPUT_NORM_SCHEMA_NONE,
    INPUT_NORM_SCHEMA_ZSCORE,
    PHONE_WATCH_SENSOR_LAYOUT,
    set_seed,
)


LOGGER = logging.getLogger("train")
DEFAULT_TRAIN_SUBJECTS = [
    28, 29, 30, 31, 33, 35, 38, 39, 32, 36, 37, 43, 44, 45, 46, 49, 51, 56, 57, 58, 59, 61, 62
]


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _is_main_process() -> bool:
    return _get_rank() == 0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _sync_mean(value: float, device: torch.device) -> float:
    """Average scalar across ranks when DDP is active."""
    ''''Suppose GPU 0 gets train loss 0.9 and GPU 1 gets 1.1.
    Then this function averages them to 1.0.'''
    if not _is_distributed():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _sync_sum(value: float, device: torch.device) -> float:
    if not _is_distributed():
        return float(value)
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _sync_ratio(numerator: float, denominator: float, device: torch.device) -> float:
    total_num = _sync_sum(numerator, device)
    total_den = _sync_sum(denominator, device)
    return total_num / max(total_den, 1.0)


def _iter_with_progress(loader: torch.utils.data.DataLoader, desc: str, enabled: bool = True):
    """Wrap loader with tqdm when available, otherwise return plain iterator."""
    if tqdm is None or not enabled:
        return loader
    return tqdm(loader, total=len(loader), desc=desc, leave=False)


def _augment_stage4_imu(
    phone_accel: torch.Tensor,
    watch_accel: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Stage-4-only IMU augmentations used for contrastive pretraining."""
    noise_std = float(getattr(args, "stage4_imu_noise_std", 0.0))
    scale_range = float(getattr(args, "stage4_imu_scale_range", 0.0))
    mask_prob = float(getattr(args, "stage4_imu_time_mask_prob", 0.0))
    mask_width = int(getattr(args, "stage4_imu_time_mask_width", 0))
    if noise_std <= 0.0 and scale_range <= 0.0 and (mask_prob <= 0.0 or mask_width <= 0):
        return phone_accel, watch_accel

    phone_aug = phone_accel
    watch_aug = watch_accel

    if scale_range > 0.0:
        batch_size = phone_accel.shape[0]
        scale = 1.0 + (torch.rand(batch_size, 1, 1, device=phone_accel.device, dtype=phone_accel.dtype) * 2.0 - 1.0) * scale_range
        phone_aug = phone_aug * scale
        watch_aug = watch_aug * scale

    if noise_std > 0.0:
        phone_aug = phone_aug + torch.randn_like(phone_aug) * noise_std
        watch_aug = watch_aug + torch.randn_like(watch_aug) * noise_std

    if mask_prob > 0.0 and mask_width > 0:
        batch_size, num_frames, _ = phone_aug.shape
        width = min(mask_width, num_frames)
        apply_mask = torch.rand(batch_size, device=phone_aug.device) < mask_prob
        if bool(apply_mask.any()):
            phone_aug = phone_aug.clone()
            watch_aug = watch_aug.clone()
            max_start = max(num_frames - width + 1, 1)
            starts = torch.randint(max_start, (batch_size,), device=phone_aug.device)
            phone_fill = phone_aug.mean(dim=1, keepdim=True)
            watch_fill = watch_aug.mean(dim=1, keepdim=True)
            for sample_idx in apply_mask.nonzero(as_tuple=False).flatten().tolist():
                start = int(starts[sample_idx].item())
                end = start + width
                phone_aug[sample_idx, start:end] = phone_fill[sample_idx]
                watch_aug[sample_idx, start:end] = watch_fill[sample_idx]

    return phone_aug, watch_aug


def _log_step_progress(
    stage_name: str,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    metric_name: str,
    metric_value: float,
    epoch_start_time: float,
) -> None:
    """Print plain step progress for non-interactive environments (e.g. notebooks/nohup)."""
    if not _is_main_process():
        return
    elapsed = max(time.time() - epoch_start_time, 1e-6)
    steps_per_sec = step / elapsed
    eta_sec = max(0.0, (total_steps - step) / max(steps_per_sec, 1e-6))
    LOGGER.info(
        "[%s] epoch=%s/%s step=%s/%s %s=%.6f eta=%.1fs",
        stage_name,
        epoch,
        total_epochs,
        step,
        total_steps,
        metric_name,
        metric_value,
        eta_sec,
    )


def _setup_logging() -> None:
    """Configure timestamped logging format."""
    logging.basicConfig(
        level=logging.INFO if _is_main_process() else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _init_distributed(args: argparse.Namespace) -> tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise ValueError("DDP requires CUDA/NCCL in this script.")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        torch.cuda.set_device(local_rank)

        if args.ddp or world_size > 1:
            print(f"[train] DDP initialized rank={rank} local_rank={local_rank} world_size={world_size}")

        return True, local_rank, world_size

    if args.ddp:
        print("[train] --ddp was passed, but WORLD_SIZE=1 so DDP is disabled.")
    return False, 0, 1



def _resolve_data_mode(args: argparse.Namespace) -> str:
    """Return active input mode: phone+watch csv."""
    if args.skeleton_folder and args.phone_accel_folder and args.watch_accel_folder:
        return "phone-watch-csv"
    raise ValueError(
        "Requires --skeleton_folder with (--phone_accel_folder + --watch_accel_folder)."
    )


def _sensor_modality_from_args(args: argparse.Namespace) -> str:
    del args
    return "phone + watch accelerometer"


def _print_run_summary(args: argparse.Namespace, device: torch.device) -> None:
    """Print a compact, readable training run summary."""
    if not _is_main_process():
        return
    data_mode = _resolve_data_mode(args)
    LOGGER.info("Run dir: %s", args.save_dir)
    LOGGER.info("Device: %s", device)
    LOGGER.info(
        "Config: stage=%s epochs=%s batch_size=%s lr=%s window=%s stride=%s joints=%s latent_dim=%s ode_steps=%s val_split=%s",
        args.stage,
        args.epochs,
        args.batch_size,
        args.lr,
        args.window,
        args.stride,
        args.joints,
        args.latent_dim,
        args.ode_steps,
        args.val_split,
    )
    LOGGER.info("Data mode: %s", data_mode)
    LOGGER.info("skeleton_folder=%s", args.skeleton_folder)
    LOGGER.info("phone_accel_folder=%s", args.phone_accel_folder)
    LOGGER.info("watch_accel_folder=%s", args.watch_accel_folder)
    LOGGER.info("gait_cache_dir=%s", args.gait_cache_dir)
    LOGGER.info("Gait metrics dim: %s", args.gait_metrics_dim)
    LOGGER.info("Input normalization: %s", args.input_norm)
    LOGGER.info("Gait metric names: %s", ", ".join(GAIT_METRIC_NAMES))
    LOGGER.info("IMU feature names: %s", ", ".join(IMU_FEATURE_NAMES))
    LOGGER.info(
        "Skeleton graph ops: encoder=%s full_skeleton=%s",
        getattr(args, "encoder_graph_op_resolved", args.skeleton_graph_op or "gcn"),
        getattr(args, "skeleton_graph_op_resolved", args.skeleton_graph_op or "gat"),
    )
    LOGGER.info(
        "Denoiser capacity: depth=%s num_heads=%s temporal_block_type=%s",
        args.denoiser_depth,
        args.denoiser_num_heads,
        args.temporal_block_type,
    )
    LOGGER.info("One-to-one mode: %s", args.one_to_one)
    if args.stage == 4:
        LOGGER.info(
            "Stage4 objective: lambda_contrast=%s lambda_imu_cls=%s lambda_imu_gait=%s lambda_imu_angle=%s lambda_imu_angvel=%s lambda_imu_skel=%s lambda_imu_skel_vel=%s lambda_imu_skel_motion=%s contrast_dim=%s temperature=%s imu_dropout=%s activity_multipositive=%s",
            args.lambda_contrast,
            args.lambda_imu_cls,
            args.lambda_imu_gait,
            args.lambda_imu_angle,
            args.lambda_imu_angvel,
            args.lambda_imu_skel,
            args.lambda_imu_skel_vel,
            args.lambda_imu_skel_motion,
            args.contrast_dim,
            args.contrast_temperature,
            args.stage2_dropout,
            args.stage4_activity_multipositive,
        )
        LOGGER.info(
            "Stage4 IMU augmentation: noise_std=%s scale_range=%s time_mask_prob=%s time_mask_width=%s",
            args.stage4_imu_noise_std,
            args.stage4_imu_scale_range,
            args.stage4_imu_time_mask_prob,
            args.stage4_imu_time_mask_width,
        )
    if args.stage == 5:
        LOGGER.info(
            "Stage5 objective: lambda_pose=%s lambda_vel=%s lambda_angle=%s lambda_angle_limit=%s lambda_gait=%s lambda_motion=%s fps=%s ode_steps=%s cfg_dropout=%s cfg_scale=%s imu_lr=%s flow_lr=%s freeze_imu_epochs=%s",
            args.lambda_pose,
            args.lambda_vel,
            args.lambda_angle,
            args.lambda_angle_limit,
            args.lambda_gait,
            args.lambda_motion,
            args.fps,
            args.ode_steps,
            args.cfg_dropout,
            args.cfg_scale,
            args.imu_lr,
            args.flow_lr,
            args.freeze_imu_epochs,
        )


def _print_loader_summary(loader: torch.utils.data.DataLoader) -> None:
    """Print dataset and dataloader shape summary once at startup."""
    if not _is_main_process():
        return
    dataset = loader.dataset
    LOGGER.info(
        "Dataset: %s samples=%s batches_per_epoch=%s drop_last=%s",
        dataset.__class__.__name__,
        len(dataset),
        len(loader),
        loader.drop_last,
    )


def _make_run_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = args.run_name or f"stage{args.stage}_{stamp}"
    run_dir = Path(args.report_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _normalize_graph_op_arg(value: str | None, default: str | None = None) -> str:
    graph_op = default if value in {None, ""} else str(value).lower()
    if graph_op not in {"gat", "gcn"}:
        raise ValueError(f"Unsupported graph op: {value}")
    return graph_op


def _resolve_graph_ops(args: argparse.Namespace) -> tuple[str, str]:
    skeleton_graph_op = _normalize_graph_op_arg(args.skeleton_graph_op, default="gcn")
    encoder_graph_op = skeleton_graph_op
    return encoder_graph_op, skeleton_graph_op


def _sensor_locations_from_args(args: argparse.Namespace) -> list[str]:
    del args
    return ["phone", "watch"]


def _sensor_sources_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "phone_accel": Path(args.phone_accel_folder).name if args.phone_accel_folder else "",
        "watch_accel": Path(args.watch_accel_folder).name if args.watch_accel_folder else "",
    }


def _input_norm_schema_from_arg(input_norm: str) -> str:
    if input_norm == "none":
        return INPUT_NORM_SCHEMA_NONE
    if input_norm == "zscore":
        return INPUT_NORM_SCHEMA_ZSCORE
    raise ValueError(f"Unsupported --input_norm={input_norm!r}")


def _checkpoint_input_norm_schema(extra: dict) -> str:
    return str(extra.get("input_norm_schema") or INPUT_NORM_SCHEMA_NONE)


def _require_stage4_input_norm(
    checkpoint: Dict[str, object],
    expected_schema: str,
    path: str,
) -> dict | None:
    extra = checkpoint.get("extra", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(extra, dict):
        extra = {}
    ckpt_schema = _checkpoint_input_norm_schema(extra)
    if ckpt_schema != expected_schema:
        raise ValueError(
            f"Stage-4 checkpoint {path!r} uses input_norm_schema={ckpt_schema!r}, "
            f"but this run expects {expected_schema!r}. Pass matching --input_norm or retrain."
        )
    if expected_schema == INPUT_NORM_SCHEMA_ZSCORE:
        input_stats = validate_imu_input_stats(extra.get("input_stats"))
        if input_stats is None:
            raise ValueError(f"Stage-4 checkpoint {path!r} is z-score normalized but lacks input_stats.")
        return input_stats
    return None


def _build_train_val_loaders(
    args: argparse.Namespace,
    dataset: Dataset,
    distributed: bool,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader], dict | None]:
    """Create train/val dataloaders from one dataset using deterministic split."""
    train_subjects = parse_subject_list(args.train_subjects) if args.train_subjects else None
    train_dataset, val_dataset = split_train_val_dataset(
        dataset,
        val_split=args.val_split,
        seed=args.seed,
        train_subjects=train_subjects,
        logger=LOGGER if _is_main_process() else None,
    )
    input_stats = None
    input_norm_schema = _input_norm_schema_from_arg(args.input_norm)
    if input_norm_schema == INPUT_NORM_SCHEMA_ZSCORE:
        input_stats = getattr(args, "input_stats", None)
        if input_stats is None:
            input_stats = compute_imu_input_stats(train_dataset)
        input_stats = validate_imu_input_stats(input_stats)
        setattr(dataset, "input_stats", input_stats)

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
    train_loader_sampler_kwargs = {"sampler": train_sampler}
    train_loader = create_dataloader(
        batch_size=args.batch_size,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        phone_accel_folder=args.phone_accel_folder or None,
        watch_accel_folder=args.watch_accel_folder or None,
        stride=args.stride,
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        dataset=train_dataset,
        shuffle=not distributed,
        drop_last=True,
        **train_loader_sampler_kwargs,
    )

    if val_dataset is None:
        return train_loader, None, input_stats

    val_sampler = None
    if distributed:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        )
    val_loader_sampler_kwargs = {"sampler": val_sampler}
    val_loader = create_dataloader(
        batch_size=args.batch_size,
        window=args.window,
        joints=args.joints,
        num_classes=args.num_classes,
        skeleton_folder=args.skeleton_folder or None,
        phone_accel_folder=args.phone_accel_folder or None,
        watch_accel_folder=args.watch_accel_folder or None,
        stride=args.stride,
        gait_cache_dir=args.gait_cache_dir or None,
        disable_gait_cache=args.disable_gait_cache,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        dataset=val_dataset,
        shuffle=False,
        drop_last=False,
        **val_loader_sampler_kwargs,
    )
    return train_loader, val_loader, input_stats


def _print_epoch_summary(stage_name: str, epoch: int, total_epochs: int, metrics: Dict[str, float], sec: float) -> None:
    """Print one formatted epoch summary line."""
    if not _is_main_process():
        return
    items = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
    LOGGER.info("[%s] epoch=%s/%s %s epoch_time=%.1fs", stage_name, epoch, total_epochs, items, sec)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train IMU contrastive pretraining or IMU-conditioned skeleton-space flow")
    parser.add_argument("--stage", type=int, required=True, choices=[4, 5])
    parser.add_argument("--skeleton_folder", type=str, default="")
    parser.add_argument("--phone_accel_folder", type=str, default="", help="Phone accelerometer CSV folder.")
    parser.add_argument("--watch_accel_folder", type=str, default="", help="Watch accelerometer CSV folder.")
    parser.add_argument("--gait_cache_dir", type=str, default="")
    parser.add_argument("--disable_gait_cache", action="store_true")
    parser.add_argument(
        "--input_norm",
        choices=["none", "zscore"],
        default="none",
        help="Optional IMU-only input normalization. 'zscore' fits phone/watch stats on Stage-4 train subjects and reuses them downstream.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel (or auto-enabled under torchrun).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Stop training if the monitored loss (val if available, else train) does not "
        "improve for this many consecutive epochs. Set 0 to disable.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--report_dir", type=str, default="outputs/training_reports")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--imu_pretrain_ckpt", type=str, default="", help="Stage-4 IMU contrastive checkpoint for Stage-5 skeleton flow.")
    parser.add_argument("--pretrain_imu_clip", action="store_true", help="Compatibility flag for Stage-4 IMU-skeleton contrastive pretraining.")
    parser.add_argument("--ode_steps", type=int, default=DEFAULT_ODE_STEPS, help="Euler ODE sampling steps for flow matching.")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale for generation/evaluation (1 disables CFG).")
    parser.add_argument("--cfg_dropout", type=float, default=0.1, help="Stage-5 conditioning dropout probability for CFG training.")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--joints", type=int, default=DEFAULT_JOINTS)
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument(
        "--gait_metrics_dim",
        type=int,
        default=DEFAULT_GAIT_METRICS_DIM,
        help="Number of scalar gait metrics provided per sample. Defaults to the fixed auto-computed gait-summary size.",
    )
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--stride", type=int, default=45)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--one_to_one", dest="one_to_one", action="store_true", help="Enable one-to-one IMU->skeleton mode.")
    parser.add_argument("--no_one_to_one", dest="one_to_one", action="store_false", help="Use legacy conditioning behavior.")
    parser.set_defaults(one_to_one=True)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data used for validation (0 disables validation).")
    parser.add_argument(
        "--train_subjects",
        type=str,
        default="",
        help="Comma-separated subject ids used for training; all other subjects go to validation. Leave empty to use random val_split instead.",
    )
    parser.add_argument("--log_every", type=int, default=20, help="Plain progress print interval (steps) when tqdm is not interactive.")
    parser.add_argument("--skeleton_graph_op", choices=["gat", "gcn"], default="gcn",
                        help="Skeleton graph family for the Stage-4 skeleton encoder and Stage-5 flow denoiser.")
    parser.add_argument("--temporal_block_type", choices=["conv", "attention"], default="conv",
                        help="Temporal block type for skeleton denoiser: 'conv' (kernel=3, local) or 'attention' (self-attention, global receptive field).")
    parser.add_argument("--denoiser_depth", type=int, default=6,
                        help="Number of stacked graph + cross-attn + temporal blocks in the Stage-5 denoiser.")
    parser.add_argument("--denoiser_num_heads", type=int, default=8,
                        help="Attention head count for the denoiser cross-attention and temporal attention blocks (must divide --latent_dim).")
    parser.add_argument("--imu_graph", choices=["chain", "multiscale"], default=None,
                        help="IMU temporal graph type for the GCN encoder: 'chain' (j→j+1 only) or 'multiscale' (gaps 1,5,15,30). "
                             "Default 'multiscale' when --imu_encoder_type=gcn; ignored when --imu_encoder_type=cnn (passing this flag with cnn raises an error).")
    parser.add_argument("--imu_encoder_type", choices=["gcn", "cnn"], default="gcn",
                        help="IMU encoder architecture: 'gcn' (default, temporal graph) or 'cnn' (stride-1 dilated Conv1d stack on 10-D shared features).")
    cnn_attn_group = parser.add_mutually_exclusive_group()
    cnn_attn_group.add_argument("--imu_cnn_attention", dest="imu_cnn_attention", action="store_true",
                                help="Enable per-branch self-attention refinement after the CNN conv stack (default).")
    cnn_attn_group.add_argument("--no_imu_cnn_attention", dest="imu_cnn_attention", action="store_false",
                                help="Disable the CNN-encoder per-branch self-attention block.")
    parser.set_defaults(imu_cnn_attention=True)
    parser.add_argument("--stage2_dropout", type=float, default=0.25, help="Dropout rate used by the IMU encoder.")
    parser.add_argument("--contrast_dim", type=int, default=256, help="Projection dimension for IMU-skeleton contrastive pretraining.")
    parser.add_argument("--contrast_temperature", type=float, default=0.07, help="Temperature for symmetric IMU-skeleton contrastive loss.")
    parser.add_argument("--lambda_contrast", type=float, default=1.0, help="Weight for Stage-4 IMU-skeleton contrastive loss.")
    parser.add_argument("--lambda_imu_cls", type=float, default=0.1, help="Weight for Stage-4 IMU activity-classification auxiliary loss.")
    parser.add_argument("--lambda_imu_gait", type=float, default=0.0, help="Weight for Stage-4 IMU gait-prediction auxiliary loss (0 = off; pass >0 to enable).")
    parser.add_argument("--lambda_imu_angle", type=float, default=0.1, help="Weight for Stage-4 IMU-to-joint-angle trajectory loss.")
    parser.add_argument("--lambda_imu_angvel", type=float, default=0.1, help="Weight for Stage-4 IMU-to-joint-angular-velocity trajectory loss.")
    parser.add_argument("--lambda_imu_skel", type=float, default=0.05, help="Weight for Stage-4 IMU-to-root-relative-skeleton auxiliary loss.")
    parser.add_argument("--lambda_imu_skel_vel", type=float, default=0.05, help="Weight for Stage-4 IMU-to-root-relative-skeleton-velocity auxiliary loss.")
    parser.add_argument("--lambda_imu_skel_motion", type=float, default=0.0,
                        help="Weight for Stage-4 anti-collapse term that penalizes when predicted per-frame variance "
                             "falls below ground-truth per-frame variance. Use >0 to prevent the encoder from "
                             "learning the trivial 'predict mean pose' solution. Recommended: 0.5-1.0.")
    parser.add_argument("--stage4_activity_multipositive", action="store_true", help="Treat same-activity samples in each Stage-4 batch as additional contrastive positives.")
    parser.add_argument("--stage4_imu_noise_std", type=float, default=0.0, help="Gaussian noise std added to phone/watch IMU during Stage-4 training only.")
    parser.add_argument("--stage4_imu_scale_range", type=float, default=0.0, help="Symmetric random IMU scale range during Stage-4 training only; 0.1 samples scales in [0.9, 1.1].")
    parser.add_argument("--stage4_imu_time_mask_prob", type=float, default=0.0, help="Per-window probability of replacing one IMU time span with the sequence mean during Stage-4 training only.")
    parser.add_argument("--stage4_imu_time_mask_width", type=int, default=0, help="Width in frames for Stage-4 IMU time masking.")
    parser.add_argument("--freeze_imu_epochs", type=int, default=5, help="Number of Stage-5 epochs to keep the pretrained IMU encoder frozen.")
    parser.add_argument("--imu_lr", type=float, default=1e-5, help="Learning rate for the pretrained IMU encoder during Stage-5 fine-tuning.")
    parser.add_argument("--flow_lr", type=float, default=0.0, help="Stage-5 flow learning rate. Uses --lr when 0.")
    parser.add_argument("--lambda_pose", type=float, default=0.0, help="Weight for paired skeleton reconstruction loss in Stage 5.")
    parser.add_argument("--lambda_vel", type=float, default=0.0, help="Weight for velocity reconstruction loss in Stage 5.")
    parser.add_argument("--lambda_mos", type=float, default=0.0, help="Weight for Hof (2005) Margin-of-Stability (extrapolated-CoM vs base-of-support) matching loss in Stage 5.")
    parser.add_argument("--lambda_com", type=float, default=0.0, help="Weight for simple CoM-vs-BOS stability-margin matching loss in Stage 5 (no momentum term).")
    parser.add_argument("--lambda_angle", type=float, default=0.05, help="Weight for SSDL-style per-frame joint-angle Frobenius loss in Stage 5.")
    parser.add_argument("--lambda_angle_limit", type=float, default=1.0, help="Weight for soft key-joint angle-limit penalty in Stage 5.")
    parser.add_argument("--lambda_motion", type=float, default=1.0, help="Weight for motion regularization loss in Stage 5.")
    parser.add_argument("--lambda-gait", type=float, default=0.0, help="Weight for generated-vs-real gait-summary MSE loss in Stage 5 (0 = off; pass >0 to enable).")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Skeleton frame-rate for Stage-5 temporal losses.")
    parser.add_argument("--sample_seed", type=int, default=0, help="Deterministic sampling seed used for evaluation/generation.")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Cap batches per training epoch for smoke runs (0 disables).")
    parser.add_argument("--max_val_batches", type=int, default=0, help="Cap batches per validation epoch for smoke runs (0 disables).")
    return parser.parse_args(argv)



def _build_imu_pretrain_model(args: argparse.Namespace) -> IMUSkeletonContrastivePretrainModel:
    return IMUSkeletonContrastivePretrainModel(
        latent_dim=args.latent_dim,
        contrast_dim=args.contrast_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
        num_classes=args.num_classes,
        temperature=args.contrast_temperature,
        skeleton_graph_op=args.encoder_graph_op_resolved,
        imu_graph_type=args.imu_graph_resolved,
        stage2_dropout=args.stage2_dropout,
        activity_multipositive=args.stage4_activity_multipositive,
        imu_encoder_type=args.imu_encoder_type,
        imu_cnn_attention=args.imu_cnn_attention,
        input_norm_schema=_input_norm_schema_from_arg(args.input_norm),
        temporal_block_type=args.temporal_block_type,
        num_heads=args.denoiser_num_heads,
    )


def _build_imu_flow_model(args: argparse.Namespace) -> IMUConditionedSkeletonFlowModel:
    return IMUConditionedSkeletonFlowModel(
        latent_dim=args.latent_dim,
        num_joints=args.joints,
        gait_metrics_dim=args.gait_metrics_dim,
        imu_graph_type=args.imu_graph_resolved,
        stage2_dropout=args.stage2_dropout,
        skeleton_graph_op=args.skeleton_graph_op_resolved,
        temporal_block_type=args.temporal_block_type,
        denoiser_depth=args.denoiser_depth,
        denoiser_num_heads=args.denoiser_num_heads,
        cfg_dropout=args.cfg_dropout,
        imu_encoder_type=args.imu_encoder_type,
        imu_cnn_attention=args.imu_cnn_attention,
        input_norm_schema=_input_norm_schema_from_arg(args.input_norm),
    )


def _load_imu_encoder_from_pretrain(path: str, model: IMUConditionedSkeletonFlowModel) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    extra = checkpoint.get("extra", {}) if isinstance(checkpoint, dict) else {}
    expected_type = getattr(model, "imu_encoder_type", "gcn")
    ckpt_type = extra.get("imu_encoder_type")
    if ckpt_type and ckpt_type != expected_type:
        raise ValueError(
            f"Stage-4 checkpoint {path!r} was trained with imu_encoder_type={ckpt_type!r}, "
            f"but Stage-5 model was built with imu_encoder_type={expected_type!r}. "
            f"Pass --imu_encoder_type {ckpt_type} or retrain Stage 4 under the requested encoder."
        )
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    imu_state = {
        key.removeprefix("imu_encoder."): value
        for key, value in state_dict.items()
        if key.startswith("imu_encoder.")
    }
    if not imu_state:
        raise ValueError(f"Checkpoint {path!r} does not contain imu_encoder.* weights.")
    missing, unexpected = model.imu_encoder.load_state_dict(imu_state, strict=True)
    if missing or unexpected:
        raise ValueError(f"Failed to load IMU encoder from {path}: missing={missing}, unexpected={unexpected}")
    return checkpoint


def _set_imu_encoder_trainable(model: IMUConditionedSkeletonFlowModel, trainable: bool) -> None:
    for p in model.imu_encoder.parameters():
        p.requires_grad = bool(trainable)


def _set_stage5_train_mode(model: torch.nn.Module) -> None:
    model.train()
    _unwrap_model(model).imu_encoder.eval()


def _build_stage5_optimizer(model: IMUConditionedSkeletonFlowModel, args: argparse.Namespace) -> optim.Optimizer:
    flow_lr = args.flow_lr if args.flow_lr > 0 else args.lr
    imu_params = list(model.imu_encoder.parameters())
    imu_param_ids = {id(p) for p in imu_params}
    flow_params = [p for p in model.parameters() if id(p) not in imu_param_ids]
    return optim.AdamW(
        [
            {"params": flow_params, "lr": flow_lr, "name": "flow"},
            {"params": imu_params, "lr": args.imu_lr, "name": "imu_encoder"},
        ],
        weight_decay=1e-3,
    )


def train_stage4_imu_pretrain(args: argparse.Namespace, device: torch.device, distributed: bool, local_rank: int) -> None:
    """Pretrain IMU conditioning with IMU-skeleton contrastive learning."""
    model = _build_imu_pretrain_model(args).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
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
    )
    train_loader, val_loader, input_stats = _build_train_val_loaders(args, dataset=dataset, distributed=distributed)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    if _is_main_process():
        save_run_manifest(
            Path(args.run_dir),
            args,
            device,
            runtime={
                "optimizer": optimizer.__class__.__name__,
                "scheduler": "none",
                "model_type": "imu_skeleton_contrastive_pretrain",
                "sensor_layout": PHONE_WATCH_SENSOR_LAYOUT,
                "sensor_modality": _sensor_modality_from_args(args),
                "sensor_locations": _sensor_locations_from_args(args),
                "sensor_sources": _sensor_sources_from_args(args),
                "input_norm_schema": _input_norm_schema_from_arg(args.input_norm),
                "skeleton_feature_schema": _unwrap_model(model).skeleton_feature_schema,
                "imu_fusion_schema": _unwrap_model(model).imu_fusion_schema,
            },
        )
    model.train()
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")
    epochs_since_improvement = 0
    history: list[dict[str, float]] = []
    run_dir = Path(args.run_dir)

    for epoch in range(args.epochs):
        t0 = time.time()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        sums = {
            "total": 0.0,
            "contrast": 0.0,
            "cls": 0.0,
            "gait": 0.0,
            "angle": 0.0,
            "angvel": 0.0,
            "skel": 0.0,
            "skel_vel": 0.0,
            "skel_motion": 0.0,
        }
        n_batches = 0
        pbar = _iter_with_progress(train_loader, f"Stage4 Epoch {epoch + 1}/{args.epochs}", enabled=_is_main_process() and sys.stdout.isatty())
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            x = batch["skeleton"].to(device)
            phone_accel = batch["phone_accel"].to(device)
            watch_accel = batch["watch_accel"].to(device)
            gait_metrics = batch["gait_metrics"].to(device)
            y = batch["label"].to(device)
            phone_accel, watch_accel = _augment_stage4_imu(phone_accel, watch_accel, args)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x=x, phone_accel=phone_accel, watch_accel=watch_accel, gait_metrics=gait_metrics, y=y)
                loss = (
                    args.lambda_contrast * out["loss_contrast"]
                    + args.lambda_imu_cls * out["loss_cls"]
                    + args.lambda_imu_gait * out["loss_gait_pred"]
                    + args.lambda_imu_angle * out["loss_imu_angle"]
                    + args.lambda_imu_angvel * out["loss_imu_angvel"]
                    + args.lambda_imu_skel * out["loss_imu_skel"]
                    + args.lambda_imu_skel_vel * out["loss_imu_skel_vel"]
                    + args.lambda_imu_skel_motion * out["loss_imu_skel_motion"]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            sums["total"] += float(loss.item())
            sums["contrast"] += float(out["loss_contrast"].item())
            sums["cls"] += float(out["loss_cls"].item())
            sums["gait"] += float(out["loss_gait_pred"].item())
            sums["angle"] += float(out["loss_imu_angle"].item())
            sums["angvel"] += float(out["loss_imu_angvel"].item())
            sums["skel"] += float(out["loss_imu_skel"].item())
            sums["skel_vel"] += float(out["loss_imu_skel_vel"].item())
            sums["skel_motion"] += float(out["loss_imu_skel_motion"].item())
            n_batches += 1
        metrics = {
            "train_loss_total": _sync_mean(sums["total"] / max(n_batches, 1), device),
            "train_loss_contrast": _sync_mean(sums["contrast"] / max(n_batches, 1), device),
            "train_loss_cls": _sync_mean(sums["cls"] / max(n_batches, 1), device),
            "train_loss_gait_pred": _sync_mean(sums["gait"] / max(n_batches, 1), device),
            "train_loss_imu_angle": _sync_mean(sums["angle"] / max(n_batches, 1), device),
            "train_loss_imu_angvel": _sync_mean(sums["angvel"] / max(n_batches, 1), device),
            "train_loss_imu_skel": _sync_mean(sums["skel"] / max(n_batches, 1), device),
            "train_loss_imu_skel_vel": _sync_mean(sums["skel_vel"] / max(n_batches, 1), device),
            "train_loss_imu_skel_motion": _sync_mean(sums["skel_motion"] / max(n_batches, 1), device),
        }

        avg_val_total = None
        if val_loader is not None:
            model.eval()
            val_sums = {
                "total": 0.0,
                "contrast": 0.0,
                "cls": 0.0,
                "gait": 0.0,
                "angle": 0.0,
                "angvel": 0.0,
                "skel": 0.0,
                "skel_vel": 0.0,
                "skel_motion": 0.0,
            }
            val_n = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break
                    x = batch["skeleton"].to(device)
                    phone_accel = batch["phone_accel"].to(device)
                    watch_accel = batch["watch_accel"].to(device)
                    gait_metrics = batch["gait_metrics"].to(device)
                    y = batch["label"].to(device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(x=x, phone_accel=phone_accel, watch_accel=watch_accel, gait_metrics=gait_metrics, y=y)
                        val_loss = (
                            args.lambda_contrast * out["loss_contrast"]
                            + args.lambda_imu_cls * out["loss_cls"]
                            + args.lambda_imu_gait * out["loss_gait_pred"]
                            + args.lambda_imu_angle * out["loss_imu_angle"]
                            + args.lambda_imu_angvel * out["loss_imu_angvel"]
                            + args.lambda_imu_skel * out["loss_imu_skel"]
                            + args.lambda_imu_skel_vel * out["loss_imu_skel_vel"]
                            + args.lambda_imu_skel_motion * out["loss_imu_skel_motion"]
                        )
                    val_sums["total"] += float(val_loss.item())
                    val_sums["contrast"] += float(out["loss_contrast"].item())
                    val_sums["cls"] += float(out["loss_cls"].item())
                    val_sums["gait"] += float(out["loss_gait_pred"].item())
                    val_sums["angle"] += float(out["loss_imu_angle"].item())
                    val_sums["angvel"] += float(out["loss_imu_angvel"].item())
                    val_sums["skel"] += float(out["loss_imu_skel"].item())
                    val_sums["skel_vel"] += float(out["loss_imu_skel_vel"].item())
                    val_sums["skel_motion"] += float(out["loss_imu_skel_motion"].item())
                    val_n += 1
            model.train()
            avg_val_total = _sync_mean(val_sums["total"] / max(val_n, 1), device)
            metrics["val_loss_total"] = avg_val_total
            metrics["val_loss_contrast"] = _sync_mean(val_sums["contrast"] / max(val_n, 1), device)
            metrics["val_loss_cls"] = _sync_mean(val_sums["cls"] / max(val_n, 1), device)
            metrics["val_loss_gait_pred"] = _sync_mean(val_sums["gait"] / max(val_n, 1), device)
            metrics["val_loss_imu_angle"] = _sync_mean(val_sums["angle"] / max(val_n, 1), device)
            metrics["val_loss_imu_angvel"] = _sync_mean(val_sums["angvel"] / max(val_n, 1), device)
            metrics["val_loss_imu_skel"] = _sync_mean(val_sums["skel"] / max(val_n, 1), device)
            metrics["val_loss_imu_skel_vel"] = _sync_mean(val_sums["skel_vel"] / max(val_n, 1), device)
            metrics["val_loss_imu_skel_motion"] = _sync_mean(val_sums["skel_motion"] / max(val_n, 1), device)

        history.append({"epoch": float(epoch + 1), **metrics})
        if _is_main_process():
            write_history(run_dir, "stage4_imu_pretrain", history)
        _print_epoch_summary("Stage4", epoch + 1, args.epochs, metrics, time.time() - t0)
        # monitor_loss is a synced mean, so it is identical on every rank; compute the
        # improvement/early-stop bookkeeping on all ranks so DDP processes stop together.
        monitor_loss = avg_val_total if avg_val_total is not None else metrics["train_loss_total"]
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            epochs_since_improvement = 0
            if _is_main_process():
                save_checkpoint(
                    os.path.join(args.save_dir, "imu_pretrain_best.pt"),
                    _unwrap_model(model),
                    extra={
                        "best_monitor_loss": best_loss,
                        "best_monitor_name": "val_loss_total" if avg_val_total is not None else "train_loss_total",
                        "best_epoch": epoch + 1,
                        "encoder_graph_op": args.encoder_graph_op_resolved,
                        "skeleton_graph_op": args.skeleton_graph_op_resolved,
                        "gait_metric_names": list(GAIT_METRIC_NAMES),
                        "imu_feature_names": list(IMU_FEATURE_NAMES),
                        "imu_angle_target_names": list(JOINT_ANGLE_NAMES),
                        "imu_angle_auxiliary": "angles_and_angular_velocities_v1",
                        "imu_skeleton_auxiliary": _unwrap_model(model).imu_skeleton_auxiliary_schema,
                        "sensor_locations": _sensor_locations_from_args(args),
                        "sensor_sources": _sensor_sources_from_args(args),
                        "input_stats": input_stats,
                        "run_dir": args.run_dir,
                    },
                )
        else:
            epochs_since_improvement += 1
        if args.early_stopping_patience > 0 and epochs_since_improvement >= args.early_stopping_patience:
            if _is_main_process():
                LOGGER.info(
                    "[early-stop] Stage4: no improvement in %s for %d epochs (patience=%d); stopping at epoch %d/%d.",
                    "val_loss_total" if avg_val_total is not None else "train_loss_total",
                    epochs_since_improvement,
                    args.early_stopping_patience,
                    epoch + 1,
                    args.epochs,
                )
            break
    if _is_main_process():
        save_checkpoint(
            os.path.join(args.save_dir, "imu_pretrain.pt"),
            _unwrap_model(model),
            extra={
                "encoder_graph_op": args.encoder_graph_op_resolved,
                "skeleton_graph_op": args.skeleton_graph_op_resolved,
                "gait_metric_names": list(GAIT_METRIC_NAMES),
                "imu_feature_names": list(IMU_FEATURE_NAMES),
                "imu_angle_target_names": list(JOINT_ANGLE_NAMES),
                "imu_angle_auxiliary": "angles_and_angular_velocities_v1",
                "imu_skeleton_auxiliary": _unwrap_model(model).imu_skeleton_auxiliary_schema,
                "sensor_locations": _sensor_locations_from_args(args),
                "sensor_sources": _sensor_sources_from_args(args),
                "input_stats": input_stats,
                "run_dir": args.run_dir,
            },
        )


def train_stage5_imu_flow(args: argparse.Namespace, device: torch.device, distributed: bool, local_rank: int) -> None:
    """Train IMU-conditioned skeleton-space flow matching."""
    if not args.imu_pretrain_ckpt:
        raise ValueError("Stage 5 requires --imu_pretrain_ckpt from Stage 4.")
    model = _build_imu_flow_model(args).to(device)
    pretrain_checkpoint = _load_imu_encoder_from_pretrain(args.imu_pretrain_ckpt, model)
    args.input_stats = _require_stage4_input_norm(
        pretrain_checkpoint,
        expected_schema=_input_norm_schema_from_arg(args.input_norm),
        path=args.imu_pretrain_ckpt,
    )
    _set_imu_encoder_trainable(model, trainable=args.freeze_imu_epochs <= 0)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
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
    )
    train_loader, val_loader, input_stats = _build_train_val_loaders(args, dataset=dataset, distributed=distributed)
    optimizer = _build_stage5_optimizer(_unwrap_model(model), args)
    if _is_main_process():
        save_run_manifest(
            Path(args.run_dir),
            args,
            device,
            runtime={
                "optimizer": optimizer.__class__.__name__,
                "scheduler": "none",
                "model_type": "imu_conditioned_skeleton_flow",
                "pretrain_checkpoint_extra": pretrain_checkpoint.get("extra", {}) if isinstance(pretrain_checkpoint, dict) else {},
                "sensor_layout": PHONE_WATCH_SENSOR_LAYOUT,
                "sensor_modality": _sensor_modality_from_args(args),
                "sensor_locations": _sensor_locations_from_args(args),
                "sensor_sources": _sensor_sources_from_args(args),
                "input_norm_schema": _input_norm_schema_from_arg(args.input_norm),
                "skeleton_feature_schema": _unwrap_model(model).skeleton_feature_schema,
                "imu_fusion_schema": _unwrap_model(model).imu_fusion_schema,
                "diffusion_schema": _unwrap_model(model).diffusion_schema,
            },
        )
    _set_stage5_train_mode(model)
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")
    epochs_since_improvement = 0
    history: list[dict[str, float]] = []
    run_dir = Path(args.run_dir)

    for epoch in range(args.epochs):
        if epoch == args.freeze_imu_epochs:
            _set_imu_encoder_trainable(_unwrap_model(model), trainable=True)
        t0 = time.time()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        sums = {key: 0.0 for key in ("total", "flow", "pose", "vel", "com", "com_simple", "gait", "angle", "angle_limit", "motion")}
        n_batches = 0
        pbar = _iter_with_progress(train_loader, f"Stage5 Epoch {epoch + 1}/{args.epochs}", enabled=_is_main_process() and sys.stdout.isatty())
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            x = batch["skeleton"].to(device)
            phone_accel = batch["phone_accel"].to(device)
            watch_accel = batch["watch_accel"].to(device)
            gait_metrics = batch["gait_metrics"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    x=x,
                    phone_accel=phone_accel,
                    watch_accel=watch_accel,
                    gait_target=gait_metrics,
                    fps=args.fps,
                )
                loss = (
                    out["loss_diff"]
                    + args.lambda_pose * out["loss_pose"]
                    + args.lambda_vel * out["loss_vel"]
                    + args.lambda_mos * out["loss_com"]
                    + args.lambda_com * out["loss_com_simple"]
                    + args.lambda_angle * out["loss_angle_recon"]
                    + args.lambda_angle_limit * out["loss_angle_limit"]
                    + args.lambda_motion * out["loss_motion"]
                    + args.lambda_gait * out["loss_gait"]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in _unwrap_model(model).parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            sums["total"] += float(loss.item())
            sums["flow"] += float(out["loss_diff"].item())
            sums["pose"] += float(out["loss_pose"].item())
            sums["vel"] += float(out["loss_vel"].item())
            sums["com"] += float(out["loss_com"].item())
            sums["com_simple"] += float(out["loss_com_simple"].item())
            sums["gait"] += float(out["loss_gait"].item())
            sums["angle"] += float(out["loss_angle_recon"].item())
            sums["angle_limit"] += float(out["loss_angle_limit"].item())
            sums["motion"] += float(out["loss_motion"].item())
            n_batches += 1
        metrics = {f"train_loss_{key}": _sync_mean(value / max(n_batches, 1), device) for key, value in sums.items()}

        avg_val_total = None
        if val_loader is not None:
            model.eval()
            val_sums = {key: 0.0 for key in sums}
            val_n = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                        break
                    x = batch["skeleton"].to(device)
                    phone_accel = batch["phone_accel"].to(device)
                    watch_accel = batch["watch_accel"].to(device)
                    gait_metrics = batch["gait_metrics"].to(device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(
                            x=x,
                            phone_accel=phone_accel,
                            watch_accel=watch_accel,
                            gait_target=gait_metrics,
                            fps=args.fps,
                        )
                        val_loss = (
                            out["loss_diff"]
                            + args.lambda_pose * out["loss_pose"]
                            + args.lambda_vel * out["loss_vel"]
                            + args.lambda_mos * out["loss_com"]
                            + args.lambda_com * out["loss_com_simple"]
                            + args.lambda_angle * out["loss_angle_recon"]
                            + args.lambda_angle_limit * out["loss_angle_limit"]
                            + args.lambda_motion * out["loss_motion"]
                            + args.lambda_gait * out["loss_gait"]
                        )
                    val_sums["total"] += float(val_loss.item())
                    val_sums["flow"] += float(out["loss_diff"].item())
                    val_sums["pose"] += float(out["loss_pose"].item())
                    val_sums["vel"] += float(out["loss_vel"].item())
                    val_sums["com"] += float(out["loss_com"].item())
                    val_sums["com_simple"] += float(out["loss_com_simple"].item())
                    val_sums["gait"] += float(out["loss_gait"].item())
                    val_sums["angle"] += float(out["loss_angle_recon"].item())
                    val_sums["angle_limit"] += float(out["loss_angle_limit"].item())
                    val_sums["motion"] += float(out["loss_motion"].item())
                    val_n += 1
            _set_stage5_train_mode(model)
            avg_val_total = _sync_mean(val_sums["total"] / max(val_n, 1), device)
            for key, value in val_sums.items():
                metrics[f"val_loss_{key}"] = _sync_mean(value / max(val_n, 1), device)

        history.append({"epoch": float(epoch + 1), **metrics})
        if _is_main_process():
            write_history(run_dir, "stage5_imu_flow", history)
        _print_epoch_summary("Stage5", epoch + 1, args.epochs, metrics, time.time() - t0)
        # monitor_loss is a synced mean, so it is identical on every rank; compute the
        # improvement/early-stop bookkeeping on all ranks so DDP processes stop together.
        monitor_loss = avg_val_total if avg_val_total is not None else metrics["train_loss_total"]
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            epochs_since_improvement = 0
            if _is_main_process():
                save_checkpoint(
                    os.path.join(args.save_dir, "imu_flow_best.pt"),
                    _unwrap_model(model),
                    extra={
                        "best_monitor_loss": best_loss,
                        "best_monitor_name": "val_loss_total" if avg_val_total is not None else "train_loss_total",
                        "best_epoch": epoch + 1,
                        "imu_pretrain_ckpt": args.imu_pretrain_ckpt,
                        "skeleton_graph_op": args.skeleton_graph_op_resolved,
                        "gait_metric_names": list(GAIT_METRIC_NAMES),
                        "imu_feature_names": list(IMU_FEATURE_NAMES),
                        "sensor_locations": _sensor_locations_from_args(args),
                        "sensor_sources": _sensor_sources_from_args(args),
                        "input_stats": input_stats,
                        "run_dir": args.run_dir,
                    },
                )
        else:
            epochs_since_improvement += 1
        if args.early_stopping_patience > 0 and epochs_since_improvement >= args.early_stopping_patience:
            if _is_main_process():
                LOGGER.info(
                    "[early-stop] Stage5: no improvement in %s for %d epochs (patience=%d); stopping at epoch %d/%d.",
                    "val_loss_total" if avg_val_total is not None else "train_loss_total",
                    epochs_since_improvement,
                    args.early_stopping_patience,
                    epoch + 1,
                    args.epochs,
                )
            break
    if _is_main_process():
        save_checkpoint(
            os.path.join(args.save_dir, "imu_flow.pt"),
            _unwrap_model(model),
            extra={
                "imu_pretrain_ckpt": args.imu_pretrain_ckpt,
                "skeleton_graph_op": args.skeleton_graph_op_resolved,
                "gait_metric_names": list(GAIT_METRIC_NAMES),
                "imu_feature_names": list(IMU_FEATURE_NAMES),
                "sensor_locations": _sensor_locations_from_args(args),
                "sensor_sources": _sensor_sources_from_args(args),
                "input_stats": input_stats,
                "run_dir": args.run_dir,
            },
        )


def main() -> None:
    """Run stage-specific training."""
    args = parse_args()
    if args.val_split < 0.0 or args.val_split >= 1.0:
        raise ValueError("--val_split must be in [0.0, 1.0).")
    if args.gait_metrics_dim != DEFAULT_GAIT_METRICS_DIM:
        raise ValueError(
            f"--gait_metrics_dim must equal the fixed auto-computed gait-summary size ({DEFAULT_GAIT_METRICS_DIM})."
        )
    if args.ode_steps <= 0:
        raise ValueError("--ode_steps must be positive.")
    if args.cfg_dropout < 0.0 or args.cfg_dropout > 1.0:
        raise ValueError("--cfg_dropout must be in [0, 1].")
    if args.cfg_scale <= 0.0:
        raise ValueError("--cfg_scale must be positive.")
    if args.stage == 5 and args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.contrast_temperature <= 0:
        raise ValueError("--contrast_temperature must be positive.")
    if args.stage4_imu_noise_std < 0.0:
        raise ValueError("--stage4_imu_noise_std must be non-negative.")
    if args.stage4_imu_scale_range < 0.0 or args.stage4_imu_scale_range >= 1.0:
        raise ValueError("--stage4_imu_scale_range must be in [0, 1).")
    if args.stage4_imu_time_mask_prob < 0.0 or args.stage4_imu_time_mask_prob > 1.0:
        raise ValueError("--stage4_imu_time_mask_prob must be in [0, 1].")
    if args.stage4_imu_time_mask_width < 0:
        raise ValueError("--stage4_imu_time_mask_width must be non-negative.")
    if args.freeze_imu_epochs < 0:
        raise ValueError("--freeze_imu_epochs must be non-negative.")
    if args.imu_encoder_type == "cnn" and args.imu_graph is not None:
        raise ValueError(
            "--imu_graph is only meaningful with --imu_encoder_type=gcn; pass it without "
            "--imu_encoder_type=cnn or drop --imu_graph for the CNN encoder."
        )
    args.imu_graph_resolved = args.imu_graph if args.imu_graph is not None else "multiscale"
    distributed, local_rank, world_size = _init_distributed(args)
    _setup_logging()
    set_seed(args.seed)
    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    if _is_main_process():
        LOGGER.info("DDP enabled: %s (world_size=%s)", distributed, world_size)
    args.encoder_graph_op_resolved, args.skeleton_graph_op_resolved = _resolve_graph_ops(args)
    args.run_dir = str(_make_run_dir(args))
    _print_run_summary(args, device)

    if args.stage == 4:
        train_stage4_imu_pretrain(args, device, distributed=distributed, local_rank=local_rank)
    elif args.stage == 5:
        train_stage5_imu_flow(args, device, distributed=distributed, local_rank=local_rank)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")
    if _is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
