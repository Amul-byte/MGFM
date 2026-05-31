"""Paper-aligned flow state for Stage 5 (root-relative jp + jv + cf + pelvis).

Encodes a [B, T, J, 3] world-frame skeleton into a per-joint F=10 feature
tensor whose channels are:

    [0:3]   jp                  -- joint position relative to current-frame pelvis
    [3:6]   jv                  -- temporal derivative of jp (zero-padded at t=0)
    [6]     cf                  -- foot contact (binary, only at FOOT/ANKLE joints)
    [7:10]  pelvis_xyz_centered -- pelvis world position with frame-0 subtracted,
                                   broadcast identically across all joints

This representation gives the flow root/pose decoupling and explicit velocity
and contact channels (the meaningful pieces of the Motion Flow Matching paper's
HumanML3D representation), while skipping `jr` (6D rotations -- ill-defined
without SMPL-style twist info) and yaw alignment (skipped for first cut).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusion_model.util import DEFAULT_JOINTS, get_joint_index


FLOW_PER_JOINT_FEATURE_DIM: int = 10
FLOW_FEATURE_SCHEMA: str = "root_relative_jp_jv_cf_pelvis_v1"

PELVIS_INDEX: int = get_joint_index("PELVIS")
FOOT_JOINT_NAMES: tuple[str, ...] = ("ANKLE_LEFT", "FOOT_LEFT", "ANKLE_RIGHT", "FOOT_RIGHT")
FOOT_JOINT_INDICES: tuple[int, ...] = tuple(get_joint_index(name) for name in FOOT_JOINT_NAMES)

# Slice layout (kept in sync with FLOW_PER_JOINT_FEATURE_DIM = 10).
JP_SLICE = slice(0, 3)
JV_SLICE = slice(3, 6)
CF_INDEX = 6
PELVIS_SLICE = slice(7, 10)

# Foot velocity threshold (m/frame) below which we declare contact.
# Matches the contact_threshold used by foot_skating_loss in losses.py.
DEFAULT_FOOT_CONTACT_THRESHOLD: float = 0.03

# Combined |phone_accel| + |watch_accel| (m/s^2) above which the cf supervision
# is masked out -- a coarse fall-window detector. 9.81 ~ 1 g, so ~25 leaves room
# for normal walking peaks (typically <15) and triggers on impacts.
DEFAULT_IMU_MAGNITUDE_MASK_THRESHOLD: float = 25.0


def _causal_first_difference(x: torch.Tensor) -> torch.Tensor:
    """Return temporal first differences, zero-padded at t=0 to keep length T."""
    if x.shape[1] <= 1:
        return torch.zeros_like(x)
    diff_valid = x[:, 1:] - x[:, :-1]
    pad = torch.zeros_like(diff_valid[:, :1])
    return torch.cat([pad, diff_valid], dim=1)


def compute_foot_contacts(
    x_world: torch.Tensor,
    threshold: float = DEFAULT_FOOT_CONTACT_THRESHOLD,
) -> torch.Tensor:
    """Per-joint binary contact channel [B, T, J, 1].

    Nonzero only at FOOT/ANKLE joints; 1 when the foot's instantaneous speed is
    below `threshold`, else 0. The derived signal is a heuristic; we mask its
    loss in IMU high-magnitude windows at training time.
    """
    if x_world.ndim != 4 or x_world.shape[-1] != 3:
        raise ValueError(f"compute_foot_contacts expects [B, T, J, 3], got {tuple(x_world.shape)}")
    b, t, j, _ = x_world.shape
    foot_idx = torch.tensor(FOOT_JOINT_INDICES, device=x_world.device, dtype=torch.long)
    foot_pos = x_world.index_select(dim=2, index=foot_idx)              # [B,T,4,3]
    foot_vel = _causal_first_difference(foot_pos)                       # [B,T,4,3]
    foot_speed = torch.linalg.norm(foot_vel, dim=-1)                    # [B,T,4]
    contact = (foot_speed < float(threshold)).to(dtype=x_world.dtype)   # [B,T,4]
    cf = x_world.new_zeros((b, t, j, 1))
    cf[:, :, foot_idx, 0] = contact
    return cf


def window_center(x_world: torch.Tensor) -> torch.Tensor:
    """Subtract the frame-0 pelvis xyz from every joint/frame in the window.

    The flow state is translation-equivariant by design: it stores pelvis xyz
    relative to frame 0 in `pelvis_xyz_centered` and joint positions relative
    to the current-frame pelvis in `jp`. Encoding the world-frame ground truth
    directly therefore loses the absolute frame-0 origin. Centering the ground
    truth before encoding makes encode/decode an exact round-trip and is the
    canonical comparison frame for downstream losses.
    """
    if x_world.ndim != 4 or x_world.shape[-1] != 3:
        raise ValueError(f"window_center expects [B, T, J, 3], got {tuple(x_world.shape)}")
    pelvis_frame0 = x_world[:, 0:1, PELVIS_INDEX:PELVIS_INDEX + 1, :]
    return x_world - pelvis_frame0


def encode_flow_state(x_world: torch.Tensor) -> torch.Tensor:
    """Encode world-frame xyz [B, T, J, 3] into the F=10 flow state.

    Round-trips exactly through `decode_flow_state` only on window-centered
    input (frame-0 pelvis at the origin); use `window_center` first if needed.
    """
    if x_world.ndim != 4 or x_world.shape[-1] != 3:
        raise ValueError(f"encode_flow_state expects [B, T, J, 3], got {tuple(x_world.shape)}")
    if x_world.shape[2] != DEFAULT_JOINTS:
        raise ValueError(
            f"encode_flow_state expects {DEFAULT_JOINTS} joints, got {x_world.shape[2]}"
        )
    pelvis = x_world[:, :, PELVIS_INDEX:PELVIS_INDEX + 1, :]            # [B,T,1,3]
    pelvis_centered = pelvis - pelvis[:, 0:1, :, :]                     # frame-0 origin
    jp = x_world - pelvis                                               # [B,T,J,3]
    jv = _causal_first_difference(jp)                                   # [B,T,J,3]
    cf = compute_foot_contacts(x_world)                                 # [B,T,J,1]
    pelvis_broadcast = pelvis_centered.expand(-1, -1, x_world.shape[2], -1)  # [B,T,J,3]
    state = torch.cat([jp, jv, cf, pelvis_broadcast], dim=-1)
    if state.shape[-1] != FLOW_PER_JOINT_FEATURE_DIM:
        raise RuntimeError(
            f"encode_flow_state produced F={state.shape[-1]}, expected {FLOW_PER_JOINT_FEATURE_DIM}"
        )
    return state


def decode_flow_state(state: torch.Tensor) -> torch.Tensor:
    """Decode the F=10 flow state back to world-frame xyz [B, T, J, 3].

    Pelvis is read from the PELVIS joint's broadcast channels (any joint would
    work in principle since they should agree, but pinning it avoids cross-joint
    averaging drift when the model violates the broadcast constraint).
    """
    if state.ndim != 4 or state.shape[-1] != FLOW_PER_JOINT_FEATURE_DIM:
        raise ValueError(
            f"decode_flow_state expects [B, T, J, {FLOW_PER_JOINT_FEATURE_DIM}], got {tuple(state.shape)}"
        )
    jp = state[..., JP_SLICE]
    pelvis_centered = state[:, :, PELVIS_INDEX:PELVIS_INDEX + 1, PELVIS_SLICE]  # [B,T,1,3]
    return jp + pelvis_centered


def imu_high_magnitude_mask(
    phone_accel: torch.Tensor,
    watch_accel: torch.Tensor,
    threshold: float = DEFAULT_IMU_MAGNITUDE_MASK_THRESHOLD,
) -> torch.Tensor:
    """Per-frame mask [B, T] = 1.0 where the contact loss should be applied.

    Returns 0.0 in frames where either sensor's magnitude exceeds `threshold`,
    e.g. impact or free-fall windows where the foot-velocity heuristic for cf
    is unreliable.
    """
    if phone_accel.ndim != 3 or phone_accel.shape[-1] != 3:
        raise ValueError(f"imu_high_magnitude_mask phone_accel expects [B,T,3], got {tuple(phone_accel.shape)}")
    if watch_accel.shape != phone_accel.shape:
        raise ValueError("phone_accel and watch_accel must share shape")
    p_mag = torch.linalg.norm(phone_accel, dim=-1)
    w_mag = torch.linalg.norm(watch_accel, dim=-1)
    high = (p_mag > float(threshold)) | (w_mag > float(threshold))
    return (~high).to(dtype=phone_accel.dtype)


def contact_loss_masked(
    cf_pred: torch.Tensor,
    cf_target: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 contact loss over foot joints, masked by frame_mask [B, T]."""
    if cf_pred.shape != cf_target.shape:
        raise ValueError(f"cf_pred {tuple(cf_pred.shape)} vs cf_target {tuple(cf_target.shape)} shape mismatch")
    if cf_pred.ndim != 4 or cf_pred.shape[-1] != 1:
        raise ValueError(f"cf_pred expects [B,T,J,1], got {tuple(cf_pred.shape)}")
    if frame_mask.ndim != 2 or frame_mask.shape != cf_pred.shape[:2]:
        raise ValueError(f"frame_mask expects [B,T] matching cf, got {tuple(frame_mask.shape)}")
    foot_idx = torch.tensor(FOOT_JOINT_INDICES, device=cf_pred.device, dtype=torch.long)
    pred_foot = cf_pred.index_select(dim=2, index=foot_idx).squeeze(-1)      # [B,T,4]
    target_foot = cf_target.index_select(dim=2, index=foot_idx).squeeze(-1)  # [B,T,4]
    per_elem = F.smooth_l1_loss(pred_foot, target_foot, reduction="none")    # [B,T,4]
    weighted = per_elem * frame_mask.unsqueeze(-1)
    denom = frame_mask.sum().clamp_min(1.0) * pred_foot.shape[-1]
    return weighted.sum() / denom
