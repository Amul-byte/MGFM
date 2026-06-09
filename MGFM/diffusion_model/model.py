"""IMU-conditioned skeleton-space flow matching models (Stage 4 and 5)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.diffusion import FlowMatchingProcess
from diffusion_model.flow_features import (
    FLOW_FEATURE_SCHEMA,
    FLOW_PER_JOINT_FEATURE_DIM,
    PELVIS_SLICE,
    CF_INDEX,
    contact_loss_masked,
    compute_foot_contacts,
    decode_flow_state,
    encode_flow_state,
    imu_high_magnitude_mask,
    window_center,
)
from diffusion_model.gait_metrics import compute_gait_metrics_torch
from diffusion_model.losses import JOINT_ANGLE_NAMES, com_stability_loss, joint_angle_velocities, joint_angles, motion_losses, mos_loss
from diffusion_model.sensor_model import IMU_FUSION_SCHEMA, IMU_FUSION_SCHEMA_CNN, IMU_FUSION_SCHEMA_CNN_DIFF, IMULatentAligner
from diffusion_model.skeleton_model import (
    GraphDenoiserMasked,
    GraphDenoiserMaskedGCN,
    GraphEncoder,
    GraphEncoderGCN,
    SKELETON_FEATURE_DIM,
    SKELETON_FEATURE_SCHEMA,
)
from diffusion_model.util import (
    DEFAULT_JOINTS,
    INPUT_NORM_SCHEMA_NONE,
    PHONE_WATCH_SENSOR_LAYOUT,
    assert_shape,
    require_canonical_joint_count,
)


def _normalize_graph_op_name(graph_op: str | None, default: str = "gat") -> str:
    name = default if graph_op in {None, ""} else str(graph_op).lower()
    if name not in {"gat", "gcn"}:
        raise ValueError(f"Unsupported graph op: {graph_op!r}")
    return name


class IMUSkeletonContrastivePretrainModel(nn.Module):
    """Stage 4: IMU2CLIP-style contrastive pretraining aligning IMU to skeleton windows."""

    def __init__(
        self,
        latent_dim: int = 256,
        contrast_dim: int = 256,
        num_joints: int = DEFAULT_JOINTS,
        gait_metrics_dim: int = 0,
        num_classes: int = 14,
        temperature: float = 0.07,
        skeleton_graph_op: str = "gcn",
        imu_graph_type: str = "multiscale",
        stage2_dropout: float = 0.25,
        activity_multipositive: bool = False,
        imu_encoder_type: str = "gcn",
        imu_cnn_attention: bool = True,
        input_norm_schema: str = INPUT_NORM_SCHEMA_NONE,
        temporal_block_type: str = "conv",
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        require_canonical_joint_count(num_joints, "IMUSkeletonContrastivePretrainModel")
        self.latent_dim = latent_dim
        self.contrast_dim = contrast_dim
        self.num_joints = num_joints
        self.gait_metrics_dim = gait_metrics_dim
        self.num_classes = num_classes
        self.temperature = float(temperature)
        self.activity_multipositive = bool(activity_multipositive)
        self.sensor_layout = PHONE_WATCH_SENSOR_LAYOUT
        self.input_norm_schema = str(input_norm_schema)
        self.skeleton_feature_schema = SKELETON_FEATURE_SCHEMA
        self.imu_encoder_type = str(imu_encoder_type).lower()
        self.imu_fusion_schema = (
            IMU_FUSION_SCHEMA if self.imu_encoder_type == "gcn" else IMU_FUSION_SCHEMA_CNN_DIFF
        )
        self.imu_skeleton_auxiliary_schema = "root_relative_21j_xyz_velocity_v1"
        self.model_type = "imu_skeleton_contrastive_pretrain"

        self.imu_encoder = IMULatentAligner(
            latent_dim=latent_dim,
            graph_type=imu_graph_type,
            dropout=stage2_dropout,
            encoder_type=self.imu_encoder_type,
            cnn_attention=imu_cnn_attention,
        )
        encoder_cls = GraphEncoderGCN if _normalize_graph_op_name(skeleton_graph_op) == "gcn" else GraphEncoder
        self.skeleton_encoder = encoder_cls(
            input_dim=SKELETON_FEATURE_DIM,
            latent_dim=latent_dim,
            num_joints=num_joints,
            temporal_block_type=temporal_block_type,
            num_heads=num_heads,
        )
        self.imu_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, contrast_dim),
        )
        self.skel_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, contrast_dim),
        )
        self.cls_head = nn.Linear(latent_dim, num_classes)
        self.imu_angle_dim = len(JOINT_ANGLE_NAMES)
        self.imu_angle_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(p=float(stage2_dropout)),
            nn.Linear(latent_dim // 2, self.imu_angle_dim * 2),
        )
        self.imu_skeleton_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, num_joints * 3),
        )

        self.gait_pred_head = (
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.GELU(),
                nn.Linear(latent_dim // 2, gait_metrics_dim),
            )
            if gait_metrics_dim > 0
            else None
        )

    def encode_imu(
        self, phone_accel: torch.Tensor, watch_accel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_tokens, h_global = self.imu_encoder(phone_accel=phone_accel, watch_accel=watch_accel)
        with torch.autocast(device_type=h_global.device.type, enabled=False):
            imu_embed = F.normalize(self.imu_proj(h_global.float()), dim=-1, eps=1e-6)
        return h_tokens, h_global, imu_embed

    def encode_skeleton(self, x: torch.Tensor) -> torch.Tensor:
        assert_shape(x, [None, None, self.num_joints, 3], "IMUSkeletonContrastivePretrainModel.x")
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.skeleton_encoder(x.float())
            return F.normalize(self.skel_proj(z.mean(dim=(1, 2)).float()), dim=-1, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        phone_accel: torch.Tensor,
        watch_accel: torch.Tensor,
        gait_metrics: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert_shape(x, [None, None, self.num_joints, 3], "IMUSkeletonContrastivePretrainModel.x")
        assert_shape(phone_accel, [x.shape[0], x.shape[1], 3], "IMUSkeletonContrastivePretrainModel.phone_accel")
        assert_shape(watch_accel, [x.shape[0], x.shape[1], 3], "IMUSkeletonContrastivePretrainModel.watch_accel")

        h_tokens, h_global, imu_embed = self.encode_imu(phone_accel=phone_accel, watch_accel=watch_accel)
        skel_embed = self.encode_skeleton(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = imu_embed.float() @ skel_embed.float().T / max(self.temperature, 1e-6)
            labels = torch.arange(x.shape[0], device=x.device)
            if self.activity_multipositive and y is not None:
                y_flat = y.long().view(-1)
                positive_mask = y_flat[:, None].eq(y_flat[None, :])
                positive_mask[labels, labels] = True
                targets = positive_mask.float()
                targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1.0)
                loss_i2s = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                loss_s2i = -(targets.T * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
                loss_contrast = 0.5 * (loss_i2s + loss_s2i)
            else:
                loss_contrast = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

        loss_cls = (
            F.cross_entropy(self.cls_head(h_global.float()), y.long())
            if y is not None
            else torch.zeros((), device=x.device, dtype=x.dtype)
        )
        loss_gait_pred = (
            F.mse_loss(self.gait_pred_head(h_global.float()), gait_metrics.float())
            if self.gait_pred_head is not None and gait_metrics is not None
            else torch.zeros((), device=x.device, dtype=x.dtype)
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            b, t = x.shape[:2]
            imu_angle_pred = self.imu_angle_head(h_tokens.float())
            angle_target = joint_angles(x.float())
            angvel_target = joint_angle_velocities(x.float())
            target_rel = x.float() - x[:, :, :1, :].float()
            imu_skeleton_pred = self.imu_skeleton_head(h_tokens.float()).reshape(b, t, self.num_joints, 3)
            assert_shape(
                imu_angle_pred,
                [x.shape[0], x.shape[1], self.imu_angle_dim * 2],
                "IMUSkeletonContrastivePretrainModel.imu_angle_pred",
            )
            assert_shape(
                imu_skeleton_pred,
                [x.shape[0], x.shape[1], self.num_joints, 3],
                "IMUSkeletonContrastivePretrainModel.imu_skeleton_pred",
            )
            loss_imu_angle = F.smooth_l1_loss(imu_angle_pred[..., : self.imu_angle_dim], angle_target)
            loss_imu_angvel = F.smooth_l1_loss(imu_angle_pred[..., self.imu_angle_dim :], angvel_target)
            loss_imu_skel = F.smooth_l1_loss(imu_skeleton_pred, target_rel)
            loss_imu_skel_vel = F.smooth_l1_loss(
                imu_skeleton_pred[:, 1:] - imu_skeleton_pred[:, :-1],
                target_rel[:, 1:] - target_rel[:, :-1],
            )
            # Anti-collapse: penalize when per-frame predicted variance is
            # *below* ground-truth variance. Predicting a static mean pose
            # makes pred_time_std ~ 0 while gt_time_std > 0, so this term
            # gives a strong gradient that pure pose/velocity SmoothL1 does
            # not. Asymmetric on purpose — over-prediction of motion is
            # tolerated, under-prediction is penalized.
            pred_time_std = imu_skeleton_pred.std(dim=1, unbiased=False)
            gt_time_std = target_rel.std(dim=1, unbiased=False)
            loss_imu_skel_motion = F.relu(gt_time_std - pred_time_std).mean()

        return {
            "loss_contrast": loss_contrast,
            "loss_cls": loss_cls,
            "loss_gait_pred": loss_gait_pred,
            "loss_imu_angle": loss_imu_angle,
            "loss_imu_angvel": loss_imu_angvel,
            "loss_imu_skel": loss_imu_skel,
            "loss_imu_skel_vel": loss_imu_skel_vel,
            "loss_imu_skel_motion": loss_imu_skel_motion,
            "imu_embed": imu_embed,
            "skel_embed": skel_embed,
            "imu_angle_pred": imu_angle_pred,
            "imu_skeleton_pred": imu_skeleton_pred,
            "logits": logits,
            "h_global": h_global,
        }


class SkeletonSpaceFlowDenoiser(nn.Module):
    """Flow denoiser over the paper-aligned per-joint F-channel state.

    Per-joint state layout is defined by `flow_features.FLOW_PER_JOINT_FEATURE_DIM`
    and consists of jp + jv + cf + pelvis_xyz_centered (broadcast). The denoiser
    is shape-agnostic: it linearly projects F → latent_dim, runs the existing
    graph + temporal stack, and projects back to F per joint.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_joints: int = DEFAULT_JOINTS,
        depth: int = 6,
        skeleton_graph_op: str = "gcn",
        temporal_block_type: str = "conv",
        num_heads: int = 8,
        feature_dim: int = FLOW_PER_JOINT_FEATURE_DIM,
    ) -> None:
        super().__init__()
        require_canonical_joint_count(num_joints, "SkeletonSpaceFlowDenoiser")
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.feature_dim = int(feature_dim)
        self.skeleton_graph_op = _normalize_graph_op_name(skeleton_graph_op)
        self.in_proj = nn.Linear(self.feature_dim, latent_dim)
        denoiser_cls = GraphDenoiserMaskedGCN if self.skeleton_graph_op == "gcn" else GraphDenoiserMasked
        self.denoiser = denoiser_cls(
            latent_dim=latent_dim,
            num_joints=num_joints,
            depth=self.depth,
            temporal_block_type=temporal_block_type,
            num_heads=self.num_heads,
        )
        self.out_proj = nn.Linear(latent_dim, self.feature_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h_tokens: torch.Tensor | None = None,
        h_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert_shape(x_t, [None, None, self.num_joints, self.feature_dim], "SkeletonSpaceFlowDenoiser.x_t")
        z_t = self.in_proj(x_t)
        z_v = self.denoiser(z_t, t, h_tokens=h_tokens, h_global=h_global)
        velocity = self.out_proj(z_v)
        assert_shape(
            velocity,
            [x_t.shape[0], x_t.shape[1], self.num_joints, self.feature_dim],
            "SkeletonSpaceFlowDenoiser.velocity",
        )
        return velocity


class IMUConditionedSkeletonFlowModel(nn.Module):
    """Stage 5: one-stage IMU-conditioned skeleton-space flow matching model."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_joints: int = DEFAULT_JOINTS,
        gait_metrics_dim: int = 0,
        imu_graph_type: str = "multiscale",
        stage2_dropout: float = 0.25,
        skeleton_graph_op: str = "gcn",
        temporal_block_type: str = "conv",
        denoiser_depth: int = 6,
        denoiser_num_heads: int = 8,
        cfg_dropout: float = 0.1,
        imu_encoder_type: str = "gcn",
        imu_cnn_attention: bool = True,
        input_norm_schema: str = INPUT_NORM_SCHEMA_NONE,
    ) -> None:
        super().__init__()
        require_canonical_joint_count(num_joints, "IMUConditionedSkeletonFlowModel")
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.gait_metrics_dim = gait_metrics_dim
        self.cfg_dropout = float(cfg_dropout)
        self.sensor_layout = PHONE_WATCH_SENSOR_LAYOUT
        self.input_norm_schema = str(input_norm_schema)
        self.skeleton_feature_schema = SKELETON_FEATURE_SCHEMA
        self.imu_encoder_type = str(imu_encoder_type).lower()
        self.imu_fusion_schema = (
            IMU_FUSION_SCHEMA if self.imu_encoder_type == "gcn" else IMU_FUSION_SCHEMA_CNN_DIFF
        )
        self.diffusion_schema = "flow_matching_ot_jp_jv_cf_pelvis_v1"
        self.skeleton_flow_space = FLOW_FEATURE_SCHEMA
        self.flow_feature_dim = FLOW_PER_JOINT_FEATURE_DIM
        self.model_type = "imu_conditioned_skeleton_flow"
        self.denoiser_depth = int(denoiser_depth)
        self.denoiser_num_heads = int(denoiser_num_heads)

        self.imu_encoder = IMULatentAligner(
            latent_dim=latent_dim,
            graph_type=imu_graph_type,
            dropout=stage2_dropout,
            encoder_type=self.imu_encoder_type,
            cnn_attention=imu_cnn_attention,
        )
        self.denoiser = SkeletonSpaceFlowDenoiser(
            latent_dim=latent_dim,
            num_joints=num_joints,
            depth=self.denoiser_depth,
            skeleton_graph_op=skeleton_graph_op,
            temporal_block_type=temporal_block_type,
            num_heads=self.denoiser_num_heads,
            feature_dim=FLOW_PER_JOINT_FEATURE_DIM,
        )
        self.diffusion = FlowMatchingProcess()

    def encode_conditioning(
        self, phone_accel: torch.Tensor, watch_accel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.imu_encoder(phone_accel=phone_accel, watch_accel=watch_accel)

    def forward(
        self,
        x: torch.Tensor,
        phone_accel: torch.Tensor,
        watch_accel: torch.Tensor,
        gait_target: torch.Tensor | None = None,
        fps: float = 30.0,
    ) -> dict[str, torch.Tensor]:
        assert_shape(x, [None, None, self.num_joints, 3], "IMUConditionedSkeletonFlowModel.x")
        assert_shape(phone_accel, [x.shape[0], x.shape[1], 3], "IMUConditionedSkeletonFlowModel.phone_accel")
        assert_shape(watch_accel, [x.shape[0], x.shape[1], 3], "IMUConditionedSkeletonFlowModel.watch_accel")

        h_tokens, h_global = self.encode_conditioning(phone_accel=phone_accel, watch_accel=watch_accel)
        if self.training and self.cfg_dropout > 0.0:
            keep = (torch.rand(x.shape[0], device=x.device) >= self.cfg_dropout).to(dtype=h_tokens.dtype)
            h_tokens = h_tokens * keep.reshape(-1, 1, 1)
            h_global = h_global * keep.reshape(-1, 1)

        # Window-center the ground truth (frame-0 pelvis -> origin) so that
        # encode/decode is an exact identity and downstream losses compare
        # centered-to-centered. Existing motion losses (bone, skate, smooth,
        # instab, angles, gait) are translation-invariant so this is a no-op
        # for them numerically.
        x_centered = window_center(x)
        x_state = encode_flow_state(x_centered)

        t = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
        loss_flow, pred_v, x_t, noise = self.diffusion.flow_matching_loss(
            self.denoiser, x_state, t,
            h_tokens=h_tokens,
            h_global=h_global,
        )
        x_pred_state = self.diffusion.predict_clean_from_velocity(x_t, t, pred_v)
        x_pred = decode_flow_state(x_pred_state)

        # IMU-magnitude-gated foot-contact loss (heuristic foot-velocity contact
        # is unreliable in fall windows; we silence its supervision there).
        cf_target = compute_foot_contacts(x_centered)
        cf_pred = x_pred_state[..., CF_INDEX:CF_INDEX + 1]
        cf_mask = imu_high_magnitude_mask(phone_accel, watch_accel)
        loss_contact = contact_loss_masked(cf_pred, cf_target, cf_mask)

        gait_gen = compute_gait_metrics_torch(x_pred.float(), fps=fps)
        loss_pose = F.smooth_l1_loss(x_pred, x_centered)
        loss_vel = F.smooth_l1_loss(
            x_pred[:, 1:] - x_pred[:, :-1], x_centered[:, 1:] - x_centered[:, :-1]
        )
        # --lambda_mos weights the Hof (2005) Margin-of-Stability matching loss
        # (GT-matched, so it does not suppress falls), not raw CoM-trajectory matching.
        loss_com = mos_loss(x_pred.float(), x_centered.float(), fps=fps)
        # --lambda_com weights the simple CoM-vs-BOS stability-margin matching loss
        # (no momentum term; notebook 'Simple CoM-based Stability Margin' port).
        loss_com_simple = com_stability_loss(x_pred.float(), x_centered.float())
        loss_gait = (
            F.mse_loss(gait_gen, gait_target.float())
            if gait_target is not None
            else torch.zeros((), device=x.device, dtype=x.dtype)
        )
        loss_terms = motion_losses(x_pred.float(), target=x_centered.float())
        # Fold contact loss into loss_motion so train.py's existing aggregation
        # picks it up under args.lambda_motion without a CLI/schema bump.
        loss_terms["loss_contact"] = loss_contact
        loss_terms["loss_motion"] = loss_terms["loss_motion"] + loss_contact

        return {
            "x_pred": x_pred,
            "x_centered": x_centered,
            "x_pred_state": x_pred_state,
            "x_state": x_state,
            "pred_velocity": pred_v,
            "target_velocity": x_state - noise,
            "noise": noise,
            "x_t": x_t,
            "gait_gen": gait_gen,
            "loss_flow": loss_flow,
            "loss_diff": loss_flow,
            "loss_pose": loss_pose,
            "loss_vel": loss_vel,
            "loss_com": loss_com,
            "loss_com_simple": loss_com_simple,
            "loss_gait": loss_gait,
            **loss_terms,
        }

    @torch.no_grad()
    def sample(
        self,
        phone_accel: torch.Tensor,
        watch_accel: torch.Tensor,
        n_steps: int | None = None,
        cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        assert_shape(phone_accel, [None, None, 3], "IMUConditionedSkeletonFlowModel.sample.phone_accel")
        assert_shape(watch_accel, [phone_accel.shape[0], phone_accel.shape[1], 3], "IMUConditionedSkeletonFlowModel.sample.watch_accel")
        was_training = self.training
        self.eval()
        try:
            h_tokens, h_global = self.encode_conditioning(phone_accel=phone_accel, watch_accel=watch_accel)
            shape = torch.Size(
                (phone_accel.shape[0], phone_accel.shape[1], self.num_joints, FLOW_PER_JOINT_FEATURE_DIM)
            )
            sampled_state = self.diffusion.euler_sample_loop(
                self.denoiser,
                shape=shape,
                device=phone_accel.device,
                n_steps=n_steps,
                h_tokens=h_tokens,
                h_global=h_global,
                cfg_scale=cfg_scale,
                generator=generator,
            )
            return decode_flow_state(sampled_state)
        finally:
            self.train(was_training)
