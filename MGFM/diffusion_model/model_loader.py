"""Checkpoint save/load helpers."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from diffusion_model.util import (
    DEFAULT_JOINTS,
    INPUT_NORM_SCHEMA_NONE,
    PHONE_WATCH_SENSOR_LAYOUT,
    SKELETON_LAYOUT_VERSION,
)


def _layout_metadata_from_model(model: nn.Module) -> Dict[str, Any]:
    num_joints = getattr(model, "num_joints", DEFAULT_JOINTS)
    metadata = {
        "skeleton_layout_version": SKELETON_LAYOUT_VERSION,
        "num_joints": int(num_joints),
    }
    sensor_layout = getattr(model, "sensor_layout", None)
    if sensor_layout:
        metadata["sensor_layout"] = str(sensor_layout)
    for key in ("skeleton_feature_schema", "shared_motion_schema", "imu_fusion_schema", "diffusion_schema", "input_norm_schema"):
        value = getattr(model, key, None)
        if value:
            metadata[key] = str(value)
    for key in ("denoiser_depth", "denoiser_num_heads"):
        value = getattr(model, key, None)
        if value is not None:
            metadata[key] = int(value)
    for key in ("model_type", "skeleton_flow_space", "imu_encoder_type"):
        value = getattr(model, key, None)
        if value:
            metadata[key] = str(value)
    contrast_dim = getattr(model, "contrast_dim", None)
    if contrast_dim is not None:
        metadata["contrast_dim"] = int(contrast_dim)
    return metadata


def _validate_checkpoint_layout(checkpoint: Dict[str, Any], path: str, model: nn.Module | None = None) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} is not a structured dict; retrain under the {DEFAULT_JOINTS}-joint layout.")
    extra = checkpoint.get("extra", {})
    layout_version = extra.get("skeleton_layout_version", None)
    num_joints = extra.get("num_joints", None)
    if layout_version != SKELETON_LAYOUT_VERSION or int(num_joints or -1) != DEFAULT_JOINTS:
        raise ValueError(
            f"Checkpoint {path} uses an incompatible skeleton layout "
            f"(version={layout_version!r}, num_joints={num_joints!r}). "
            f"Expected version={SKELETON_LAYOUT_VERSION!r}, num_joints={DEFAULT_JOINTS}. Retrain from scratch."
        )
    if model is not None and getattr(model, "num_joints", DEFAULT_JOINTS) != DEFAULT_JOINTS:
        raise ValueError(
            f"Model expects num_joints={getattr(model, 'num_joints', None)}, but canonical layout requires {DEFAULT_JOINTS}."
        )
    if model is not None:
        expected_sensor_layout = getattr(model, "sensor_layout", None)
        checkpoint_sensor_layout = extra.get("sensor_layout", None)
        if expected_sensor_layout:
            if checkpoint_sensor_layout != expected_sensor_layout:
                raise ValueError(
                    f"Checkpoint {path} uses incompatible sensor_layout={checkpoint_sensor_layout!r}. "
                    f"Expected {expected_sensor_layout!r}. Retrain under the canonical phone+watch accelerometer layout."
                )
        elif checkpoint_sensor_layout not in {None, "", PHONE_WATCH_SENSOR_LAYOUT}:
            raise ValueError(
                f"Checkpoint {path} carries unexpected sensor_layout={checkpoint_sensor_layout!r} for this model."
            )
        schema_retrain_messages = {
            "skeleton_feature_schema": "Retrain affected checkpoints under the current skeleton feature schema.",
            "shared_motion_schema": "Retrain affected checkpoints under the current angle-aware shared-motion schema.",
            "imu_fusion_schema": "Retrain affected checkpoints under the current accelerometer-only IMU schema.",
            "diffusion_schema": "Retrain under the current flow-matching generative schema.",
            "imu_encoder_type": "Pass --imu_encoder_type that matches the checkpoint, or retrain under the requested encoder type.",
            "input_norm_schema": "Pass --input_norm that matches the checkpoint, or retrain with the requested input normalization.",
        }
        for schema_key, retrain_message in schema_retrain_messages.items():
            expected_schema = getattr(model, schema_key, None)
            if not expected_schema:
                continue
            checkpoint_schema = extra.get(schema_key, None)
            if schema_key == "input_norm_schema":
                checkpoint_schema = checkpoint_schema or INPUT_NORM_SCHEMA_NONE
            if checkpoint_schema != expected_schema:
                raise ValueError(
                    f"Checkpoint {path} uses incompatible {schema_key}={checkpoint_schema!r}. "
                    f"Expected {expected_schema!r}. {retrain_message}"
                )


def save_checkpoint(path: str, model: nn.Module, extra: Dict[str, Any] | None = None) -> None:
    """Save model state dict and optional metadata to checkpoint path."""
    payload: Dict[str, Any] = {"state_dict": model.state_dict()}
    merged_extra = {**_layout_metadata_from_model(model), **(extra or {})}
    payload["extra"] = merged_extra
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def _is_legacy_cnn_norm_buffer(key: str) -> bool:
    """Return True for obsolete BatchNorm buffers from the old IMU CNN encoder."""
    parts = key.split(".")
    return (
        len(parts) == 5
        and parts[0] == "imu_encoder"
        and parts[1] in {"phone_encoder", "watch_encoder", "diff_encoder"}
        and parts[2] in {"block1", "block2", "block3", "block4"}
        and parts[3] == "1"
        and parts[4] in {"running_mean", "running_var", "num_batches_tracked"}
    )


def _drop_legacy_cnn_norm_buffers(state_dict: Dict[str, Any], model: nn.Module) -> tuple[Dict[str, Any], list[str]]:
    """Remove legacy BatchNorm running-stat buffers that current GroupNorm models do not own."""
    model_keys = set(model.state_dict().keys())
    legacy_keys = sorted(
        key for key in state_dict.keys()
        if key not in model_keys and _is_legacy_cnn_norm_buffer(key)
    )
    if not legacy_keys:
        return state_dict, []
    dropped = set(legacy_keys)
    filtered = {key: value for key, value in state_dict.items() if key not in dropped}
    return filtered, legacy_keys


def load_checkpoint(path: str, model: nn.Module, strict: bool = True) -> Dict[str, Any]:
    """Load checkpoint into model and print missing/unexpected keys."""
    checkpoint = torch.load(path, map_location="cpu")
    _validate_checkpoint_layout(checkpoint, path, model=model)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    dropped_legacy_keys: list[str] = []
    if strict:
        state_dict, dropped_legacy_keys = _drop_legacy_cnn_norm_buffers(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded checkpoint: {path}")
    if dropped_legacy_keys:
        print(f"dropped legacy IMU CNN BatchNorm buffers ({len(dropped_legacy_keys)}): {dropped_legacy_keys}")
    print(f"missing keys ({len(missing)}): {missing}")
    print(f"unexpected keys ({len(unexpected)}): {unexpected}")
    return checkpoint
