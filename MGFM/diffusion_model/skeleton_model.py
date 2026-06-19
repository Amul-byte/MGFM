"""Skeleton graph encoder and denoiser for Stage 4 and 5."""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_model.graph_modules import (
    CrossAttentionBlock,
    GraphBlock,
    GraphBlockGCN,
    HAS_TORCH_GEOMETRIC,
    TemporalAttentionBlock,
    TemporalConvBlock,
    build_edge_index,
)
from diffusion_model.losses import base_of_support, center_of_mass, com_stability_margin
from diffusion_model.shared_features import (
    SKELETON_SHARED_FEATURE_NAMES,
    build_skeleton_shared_motion_features,
)
from diffusion_model.util import (
    DEFAULT_JOINTS,
    assert_shape,
    build_adjacency_matrix,
    require_canonical_joint_count,
    sinusoidal_timestep_embedding,
)


SKELETON_FEATURE_SCHEMA: str = "coords_accel_angle10_angvel10_comstab1_com3_bos2_v3"
SKELETON_FEATURE_NAMES: tuple[str, ...] = (
    ("x", "y", "z")
    + SKELETON_SHARED_FEATURE_NAMES
    + ("com_stability_margin",)
    + ("com_x", "com_y", "com_z")
    + ("bos_left", "bos_right")
)
SKELETON_FEATURE_DIM: int = len(SKELETON_FEATURE_NAMES)  # 39


def build_skeleton_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Expand raw skeleton coords [B, T, J, 3] into [B, T, J, 39].

    The trailing six channels are whole-body balance quantities (notebook cells 1-3),
    each a per-frame value broadcast identically to every joint:
      - ``com_stability_margin`` (1): simple CoM-vs-BOS margin (lateral/X only).
      - ``com_x/y/z`` (3): De Leva whole-body center of mass.
      - ``bos_left/right`` (2): outer mediolateral edges of the ankle base of support.
    """
    assert_shape(x, [None, None, None, 3], "build_skeleton_features.x")
    num_joints = x.shape[2]
    margin = com_stability_margin(x)                                          # [B, T]
    margin_feat = margin[:, :, None, None].expand(-1, -1, num_joints, 1)      # [B, T, J, 1]
    com = center_of_mass(x)                                                   # [B, T, 3]
    com_feat = com[:, :, None, :].expand(-1, -1, num_joints, -1)              # [B, T, J, 3]
    bos_left, bos_right = base_of_support(x)                                  # [B, T] each
    bos_feat = torch.stack([bos_left, bos_right], dim=-1)[:, :, None, :].expand(
        -1, -1, num_joints, -1
    )                                                                         # [B, T, J, 2]
    features = torch.cat(
        [x, build_skeleton_shared_motion_features(x, eps=eps), margin_feat, com_feat, bos_feat],
        dim=-1,
    )
    assert_shape(features, [x.shape[0], x.shape[1], x.shape[2], SKELETON_FEATURE_DIM], "build_skeleton_features.features")
    return features


def _normalize_skeleton_graph_op(graph_op: str) -> str:
    name = str(graph_op).lower()
    if name not in {"gat", "gcn"}:
        raise ValueError(f"Unsupported skeleton graph op: {graph_op!r}")
    return name


def _build_skeleton_graph_block(graph_op: str, dim: int, num_joints: int) -> nn.Module:
    if _normalize_skeleton_graph_op(graph_op) == "gcn":
        return GraphBlockGCN(dim=dim, num_joints=num_joints)
    return GraphBlock(dim=dim, num_heads=8, num_joints=num_joints, use_pyg=True)


def _register_graph_buffers(module: nn.Module, num_joints: int) -> None:
    """Register adjacency matrix and edge_index as non-persistent buffers."""
    adj = build_adjacency_matrix(num_joints=num_joints, device=torch.device("cpu"))
    module.register_buffer("_skel_adjacency", adj, persistent=False)
    ei = build_edge_index(num_joints, torch.device("cpu")) if HAS_TORCH_GEOMETRIC else None
    if ei is not None:
        module.register_buffer("_skel_edge_index", ei, persistent=False)
    else:
        module._skel_edge_index = None  # type: ignore[assignment]


class GraphEncoder(nn.Module):
    """Graph encoder mapping skeleton coordinates to joint-aware latent tokens."""

    def __init__(
        self,
        input_dim: int = SKELETON_FEATURE_DIM,
        latent_dim: int = 256,
        num_joints: int = DEFAULT_JOINTS,
        depth: int = 4,
        graph_op: str = "gat",
        temporal_block_type: str = "conv",
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        require_canonical_joint_count(num_joints, "GraphEncoder")
        if temporal_block_type == "attention" and latent_dim % num_heads != 0:
            raise ValueError(
                f"GraphEncoder with temporal_block_type='attention' requires latent_dim "
                f"divisible by num_heads; got latent_dim={latent_dim}, num_heads={num_heads}."
            )
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.temporal_block_type = temporal_block_type
        self.skeleton_feature_schema = SKELETON_FEATURE_SCHEMA
        self.in_proj = nn.Linear(input_dim, latent_dim)
        self.graph_blocks = nn.ModuleList(
            [_build_skeleton_graph_block(graph_op, dim=latent_dim, num_joints=num_joints) for _ in range(depth)]
        )
        if temporal_block_type == "attention":
            self.temporal_blocks = nn.ModuleList(
                [TemporalAttentionBlock(dim=latent_dim, num_heads=num_heads) for _ in range(depth)]
            )
        else:
            self.temporal_blocks = nn.ModuleList([TemporalConvBlock(dim=latent_dim) for _ in range(depth)])
        _register_graph_buffers(self, num_joints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [B, T, J, 3] → [B, T, J, latent_dim]."""
        x = build_skeleton_features(x)
        assert_shape(x, [None, None, self.num_joints, self.input_dim], "GraphEncoder.x")
        z = self.in_proj(x)
        for g_block, t_block in zip(self.graph_blocks, self.temporal_blocks):
            z = g_block(z, adjacency=self._skel_adjacency, edge_index=self._skel_edge_index)
            z = t_block(z)
        assert_shape(z, [x.shape[0], x.shape[1], self.num_joints, self.latent_dim], "GraphEncoder.z")
        return z


class GraphEncoderGCN(GraphEncoder):
    """GCN-based skeleton encoder — drop-in replacement for GraphEncoder."""

    def __init__(
        self,
        input_dim: int = SKELETON_FEATURE_DIM,
        latent_dim: int = 256,
        num_joints: int = DEFAULT_JOINTS,
        depth: int = 4,
        temporal_block_type: str = "conv",
        num_heads: int = 8,
    ) -> None:
        require_canonical_joint_count(num_joints, "GraphEncoderGCN")
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_joints=num_joints,
            depth=depth,
            graph_op="gcn",
            temporal_block_type=temporal_block_type,
            num_heads=num_heads,
        )


class GraphDenoiserMasked(nn.Module):
    """Adjacency-masked graph vector field with cross-attention IMU conditioning."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_joints: int = DEFAULT_JOINTS,
        depth: int = 6,
        temporal_block_type: str = "conv",
        num_heads: int = 8,
        graph_op: str = "gat",
    ) -> None:
        super().__init__()
        require_canonical_joint_count(num_joints, "GraphDenoiserMasked")
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"GraphDenoiserMasked requires latent_dim divisible by num_heads; "
                f"got latent_dim={latent_dim}, num_heads={num_heads}."
            )
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.depth = depth
        self.num_heads = num_heads
        self.temporal_block_type = temporal_block_type
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.global_cond_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.blocks = nn.ModuleList(
            [_build_skeleton_graph_block(graph_op, dim=latent_dim, num_joints=num_joints) for _ in range(depth)]
        )
        self.cross_attn_blocks = nn.ModuleList(
            [CrossAttentionBlock(dim=latent_dim, num_heads=num_heads) for _ in range(depth)]
        )
        if temporal_block_type == "attention":
            self.temporal_blocks = nn.ModuleList(
                [TemporalAttentionBlock(dim=latent_dim, num_heads=num_heads) for _ in range(depth)]
            )
        else:
            self.temporal_blocks = nn.ModuleList([TemporalConvBlock(dim=latent_dim) for _ in range(depth)])
        self.out = nn.Linear(latent_dim, latent_dim)
        _register_graph_buffers(self, num_joints)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        h_tokens: torch.Tensor | None = None,
        h_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict flow velocity for z_t with optional IMU conditioning tokens."""
        assert_shape(z_t, [None, None, self.num_joints, self.latent_dim], "GraphDenoiserMasked.z_t")
        assert_shape(t, [z_t.shape[0]], "GraphDenoiserMasked.t")
        if h_tokens is not None:
            assert_shape(h_tokens, [z_t.shape[0], z_t.shape[1], self.latent_dim], "GraphDenoiserMasked.h_tokens")
        if h_global is not None:
            assert_shape(h_global, [z_t.shape[0], self.latent_dim], "GraphDenoiserMasked.h_global")

        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.latent_dim)).unsqueeze(1).unsqueeze(1)
        x = z_t + t_emb

        if h_global is not None:
            x = x + self.global_cond_proj(h_global).unsqueeze(1).unsqueeze(1)

        for g_block, c_block, t_block in zip(self.blocks, self.cross_attn_blocks, self.temporal_blocks):
            x = g_block(x, adjacency=self._skel_adjacency, edge_index=self._skel_edge_index)
            if h_tokens is not None:
                x = c_block(x, h_tokens)
            x = t_block(x)

        velocity = self.out(x)
        assert_shape(velocity, [z_t.shape[0], z_t.shape[1], self.num_joints, self.latent_dim], "GraphDenoiserMasked.velocity")
        return velocity


class GraphDenoiserMaskedGCN(GraphDenoiserMasked):
    """GCN-based skeleton denoiser — same conditioning and temporal path."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_joints: int = DEFAULT_JOINTS,
        depth: int = 6,
        temporal_block_type: str = "conv",
        num_heads: int = 8,
    ) -> None:
        require_canonical_joint_count(num_joints, "GraphDenoiserMaskedGCN")
        super().__init__(
            latent_dim=latent_dim,
            num_joints=num_joints,
            depth=depth,
            temporal_block_type=temporal_block_type,
            num_heads=num_heads,
            graph_op="gcn",
        )
