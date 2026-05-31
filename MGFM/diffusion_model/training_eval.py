"""Training-time logging and manifest helpers for Stage 4 and 5."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
    from matplotlib import pyplot as plt
except Exception:
    plt = None

from diffusion_model.gait_metrics import GAIT_METRIC_NAMES
from diffusion_model.sensor_model import IMU_FEATURE_NAMES
from diffusion_model.util import JOINT_LABELS, SKELETON_LAYOUT_VERSION


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sensor_name(path: str) -> str:
    return Path(path).name if path else ""


def write_curve_plot(
    out_path: Path,
    title: str,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], str]],
    x_label: str,
    y_label: str,
) -> None:
    if plt is None:
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(12, 7))
    for label, values, color in series:
        plt.plot(x_values, values, label=label, linewidth=2.5, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


_LOSS_COLORS: dict[str, str] = {
    "train_loss_total":    "#111827",
    "val_loss_total":      "#7c3aed",
    "train_loss_flow":     "#2563eb",
    "val_loss_flow":       "#dc2626",
    "train_loss_pose":     "#dc2626",
    "val_loss_pose":       "#f97316",
    "train_loss_vel":      "#0891b2",
    "val_loss_vel":        "#06b6d4",
    "train_loss_gait":     "#059669",
    "val_loss_gait":       "#10b981",
    "train_loss_motion":   "#0f766e",
    "val_loss_motion":     "#14b8a6",
    "train_loss_contrast": "#2563eb",
    "val_loss_contrast":   "#dc2626",
    "train_loss_cls":      "#7c3aed",
    "val_loss_cls":        "#a855f7",
}


def write_history(run_dir: Path, stage_name: str, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = sorted({k for row in history for k in row.keys()})
    write_csv(run_dir / stage_name / "history.csv", history, fieldnames)
    epochs = [row["epoch"] for row in history]
    series = [
        (key, [row.get(key, float("nan")) for row in history], _LOSS_COLORS.get(key, "#374151"))
        for key in ("train_loss_total", "val_loss_total")
        if key in fieldnames
    ]
    write_curve_plot(
        run_dir / stage_name / "loss_curves.png",
        f"{stage_name} Loss Curves",
        epochs, series, "Epoch", "Loss",
    )


def build_run_manifest(
    args: object,
    device: torch.device,
    runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    runtime = runtime or {}
    a = args  # type: ignore[assignment]
    return {
        "device": str(device),
        "stage": a.stage,
        "seed": a.seed,
        "dataset": {
            "skeleton_folder": getattr(a, "skeleton_folder", ""),
            "phone_accel_folder": getattr(a, "phone_accel_folder", ""),
            "watch_accel_folder": getattr(a, "watch_accel_folder", ""),
            "window": a.window,
            "stride": a.stride,
            "sensor_layout": runtime.get("sensor_layout", "phone_watch_accel"),
            "sensor_modality": runtime.get("sensor_modality", "phone + watch accelerometer"),
            "sensor_locations": runtime.get("sensor_locations", ["phone", "watch"]),
            "sensor_sources": runtime.get("sensor_sources", {
                "phone_accel": _sensor_name(getattr(a, "phone_accel_folder", "")),
                "watch_accel": _sensor_name(getattr(a, "watch_accel_folder", "")),
            }),
            "imu_feature_names": list(IMU_FEATURE_NAMES),
            "gait_metric_names": list(GAIT_METRIC_NAMES),
            "joint_labels": list(JOINT_LABELS),
            "skeleton_layout_version": SKELETON_LAYOUT_VERSION,
            "fps": getattr(a, "fps", None),
        },
        "model": {
            "skeleton_graph_op": getattr(a, "skeleton_graph_op_resolved", getattr(a, "skeleton_graph_op", "gcn")),
            "skeleton_feature_schema": runtime.get("skeleton_feature_schema"),
            "imu_fusion_schema": runtime.get("imu_fusion_schema"),
            "diffusion_schema": runtime.get("diffusion_schema"),
            "model_type": runtime.get("model_type"),
        },
        "optimization": {
            "epochs": a.epochs,
            "batch_size": a.batch_size,
            "lr": a.lr,
            "optimizer": runtime.get("optimizer", "AdamW"),
            "scheduler": runtime.get("scheduler", "none"),
        },
        "flow_matching": {
            "ode_steps": getattr(a, "ode_steps", None),
            "cfg_scale": getattr(a, "cfg_scale", None),
            "cfg_dropout": getattr(a, "cfg_dropout", None),
        },
        "checkpoints": {
            "imu_pretrain_ckpt": getattr(a, "imu_pretrain_ckpt", ""),
            "save_dir": a.save_dir,
        },
    }


def save_run_manifest(
    run_dir: Path,
    args: object,
    device: torch.device,
    runtime: dict[str, object] | None = None,
) -> None:
    write_json(run_dir / "run_manifest.json", build_run_manifest(args, device, runtime=runtime))
