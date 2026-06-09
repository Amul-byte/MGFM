"""Motion regularizers for generated skeleton trajectories."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from diffusion_model.util import DEFAULT_JOINTS, EPS, get_joint_index, get_skeleton_edges


CONTACT_JOINT_INDICES: tuple[int, ...] = (get_joint_index("ANKLE_LEFT"), get_joint_index("ANKLE_RIGHT"))
ANKLE_JOINT_INDICES: tuple[int, ...] = CONTACT_JOINT_INDICES
HIP_LEFT_INDEX: int = get_joint_index("HIP_LEFT")
GRAVITY: float = 9.81  # m/s^2, for the inverted-pendulum natural frequency in MoS
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

# De Leva (1996) whole-body CoM. Each segment CoM = p_prox + l*(p_dist - p_prox);
# whole-body CoM = sum(r*seg_com)/sum(r). Both steps are linear in joint positions,
# so the CoM collapses to a single fixed per-joint weight vector (sums to 1.0).
# Columns: (name, proximal, distal, l_male, l_female, r_male, r_female).
_DELEVA_SEGMENTS: tuple[tuple[str, str, str, float, float, float, float], ...] = (
    ("head", "NECK", "HEAD", 0.5002, 0.5831, 0.0694, 0.0668),
    ("trunk", "PELVIS", "NECK", 0.5138, 0.4920, 0.4346, 0.4257),
    ("uarm_l", "SHOULDER_LEFT", "ELBOW_LEFT", 0.5772, 0.5754, 0.0271, 0.0255),
    ("farm_l", "ELBOW_LEFT", "WRIST_LEFT", 0.4574, 0.4592, 0.0162, 0.0138),
    ("hand_l", "WRIST_LEFT", "WRIST_LEFT", 0.7900, 0.7474, 0.0061, 0.0056),
    ("uarm_r", "SHOULDER_RIGHT", "ELBOW_RIGHT", 0.5772, 0.5754, 0.0271, 0.0255),
    ("farm_r", "ELBOW_RIGHT", "WRIST_RIGHT", 0.4574, 0.4592, 0.0162, 0.0138),
    ("hand_r", "WRIST_RIGHT", "WRIST_RIGHT", 0.7900, 0.7474, 0.0061, 0.0056),
    ("thigh_l", "HIP_LEFT", "KNEE_LEFT", 0.4095, 0.3612, 0.1416, 0.1478),
    ("shank_l", "KNEE_LEFT", "ANKLE_LEFT", 0.4459, 0.4416, 0.0433, 0.0481),
    ("foot_l", "ANKLE_LEFT", "FOOT_LEFT", 0.4014, 0.4014, 0.0137, 0.0129),
    ("thigh_r", "HIP_RIGHT", "KNEE_RIGHT", 0.4095, 0.3612, 0.1416, 0.1478),
    ("shank_r", "KNEE_RIGHT", "ANKLE_RIGHT", 0.4459, 0.4416, 0.0433, 0.0481),
    ("foot_r", "ANKLE_RIGHT", "FOOT_RIGHT", 0.4014, 0.4014, 0.0137, 0.0129),
)


def _build_com_weights(gender: str = "male") -> tuple[float, ...]:
    """Collapse the De Leva segment table into a per-joint CoM weight vector."""
    male = gender == "male"
    w = [0.0] * DEFAULT_JOINTS
    total = 0.0
    for _, prox, dist, l_m, l_f, r_m, r_f in _DELEVA_SEGMENTS:
        l = l_m if male else l_f
        r = r_m if male else r_f
        w[get_joint_index(prox)] += r * (1.0 - l)
        w[get_joint_index(dist)] += r * l
        total += r
    return tuple(x / total for x in w)


COM_WEIGHTS_MALE: tuple[float, ...] = _build_com_weights("male")
COM_WEIGHTS_FEMALE: tuple[float, ...] = _build_com_weights("female")
# Male/female weights differ by <0.01 per joint, so one default vector is fine for
# training where per-sample gender is unavailable.
COM_WEIGHTS_DEFAULT: tuple[float, ...] = COM_WEIGHTS_MALE


def center_of_mass(x: torch.Tensor, weights: tuple[float, ...] | None = None) -> torch.Tensor:
    """De Leva whole-body CoM. x: [B, T, J, 3] -> [B, T, 3]."""
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"center_of_mass expects [B, T, J, 3], got {tuple(x.shape)}")
    w = weights if weights is not None else COM_WEIGHTS_DEFAULT
    wt = torch.tensor(w, device=x.device, dtype=x.dtype)
    return torch.einsum("btjc,j->btc", x, wt)


def margin_of_stability(x: torch.Tensor, fps: float) -> torch.Tensor:
    """Hof (2005) Margin of Stability in the lateral (ML/X) axis. x: [B,T,J,3] -> [B,T].

    Uses the *extrapolated* CoM, XcoM = CoM + CoM_velocity / omega0, where
    omega0 = sqrt(g / leg_length) is the inverted-pendulum natural frequency. Unlike a
    static CoM-vs-base-of-support check, XcoM accounts for momentum, so fast motion
    toward the support edge is correctly scored as less stable. The margin is the
    minimum signed distance from XcoM to either ankle (the base of support):
    margin > 0 -> momentum-corrected CoM is inside the support (stable); < 0 -> falling.

    Vertical axis is Y (index 1); lateral axis is X (index 0). Built only from joint
    differences, so it is translation-invariant and valid on window-centered input.
    """
    com_ml = center_of_mass(x)[..., 0]                                   # [B, T]
    com_ml_vel = torch.gradient(com_ml, spacing=1.0 / float(fps), dim=1)[0]
    hip_y = x[:, :, HIP_LEFT_INDEX, 1]
    ankle_y = x[:, :, ANKLE_JOINT_INDICES[0], 1]
    leg_length = (hip_y - ankle_y).abs().mean(dim=1).clamp_min(EPS)      # [B]
    omega0 = torch.sqrt(GRAVITY / leg_length)                           # [B]
    xcom_ml = com_ml + com_ml_vel / omega0[:, None]                     # [B, T]
    left_ankle_x = x[:, :, ANKLE_JOINT_INDICES[0], 0]
    right_ankle_x = x[:, :, ANKLE_JOINT_INDICES[1], 0]
    bos_outer_right = torch.maximum(right_ankle_x, left_ankle_x)
    bos_outer_left = torch.minimum(right_ankle_x, left_ankle_x)
    mos_right = bos_outer_right - xcom_ml
    mos_left = xcom_ml - bos_outer_left
    return torch.minimum(mos_right, mos_left)                           # [B, T]


def mos_loss(x_hat: torch.Tensor, x: torch.Tensor, fps: float) -> torch.Tensor:
    """SmoothL1 between generated and ground-truth Margin-of-Stability trajectories.

    Matching the GT margin -- rather than a one-sided relu(-margin) penalty (cf.
    ``instability_loss``) -- teaches the model to be unstable exactly when, and as much
    as, the real motion is (e.g. during a genuine fall), so it never suppresses falls.
    Needs only the paired ground-truth skeleton, never a fall/ADL label.
    """
    return F.smooth_l1_loss(margin_of_stability(x_hat, fps), margin_of_stability(x, fps))


def com_stability_margin(x: torch.Tensor) -> torch.Tensor:
    """Simple CoM-vs-BOS stability margin in the lateral (ML/X) axis. x:[B,T,J,3]->[B,T].

    Direct port of the notebook 'Simple CoM-based Stability Margin' cell: signed distance
    from the whole-body CoM to the nearer ankle (base-of-support edge), no momentum term.
    margin > 0 -> CoM inside the feet (stable); < 0 -> CoM crossed a foot.
    Translation-invariant, so valid on window-centered input.
    """
    com_ml = center_of_mass(x)[..., 0]                              # [B, T]
    left_ankle_x = x[:, :, ANKLE_JOINT_INDICES[0], 0]
    right_ankle_x = x[:, :, ANKLE_JOINT_INDICES[1], 0]
    bos_outer_right = torch.maximum(right_ankle_x, left_ankle_x)
    bos_outer_left = torch.minimum(right_ankle_x, left_ankle_x)
    dist_to_right = bos_outer_right - com_ml
    dist_to_left = com_ml - bos_outer_left
    return torch.minimum(dist_to_right, dist_to_left)              # [B, T]


def com_stability_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """SmoothL1 between generated and GT simple CoM stability-margin trajectories."""
    return F.smooth_l1_loss(com_stability_margin(x_hat), com_stability_margin(x))


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
