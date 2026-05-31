"""Sensor models for IMU-to-latent alignment using temporal GNN branches."""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_model.graph_modules import HAS_TORCH_GEOMETRIC, TemporalGCNBlock, build_edge_index_from_adjacency
from diffusion_model.shared_features import SHARED_FEATURE_DIM, SHARED_FEATURE_NAMES, build_shared_motion_features
if HAS_TORCH_GEOMETRIC:
    from torch_geometric.nn import GCNConv
from diffusion_model.util import PHONE_WATCH_SENSOR_LAYOUT, assert_shape


IMU_FEATURE_NAMES: tuple[str, ...] = SHARED_FEATURE_NAMES
IMU_FUSION_SCHEMA: str = "accel_only_v1"
IMU_FUSION_SCHEMA_CNN: str = "accel_shared10_cnn_v1"
IMU_FUSION_SCHEMA_CNN_DIFF: str = "accel_shared10_diff_cnn_v1"
SUPPORTED_IMU_ENCODER_TYPES: tuple[str, ...] = ("gcn", "cnn")


def build_imu_features(accel: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build raw per-sensor accelerometer motion features."""
    assert_shape(accel, [None, None, 3], "build_imu_features.accel")
    features = build_shared_motion_features(accel, eps=eps)
    assert_shape(features, [accel.shape[0], accel.shape[1], len(IMU_FEATURE_NAMES)], "build_imu_features.features")
    return features

_IMU_GRAPH_SCALES: tuple[int, ...] = (1, 5, 15, 30)
"""Multi-scale temporal hop distances for the IMU graph.

With 3 stacked GCNConv layers and max distance 30, the receptive field is
3 × 30 + 1 = 91 frames — enough to cover a full gait cycle at 30 fps.
Distances chosen to capture:
  1  → local motion continuity (consecutive frames)
  5  → within-step oscillation (~0.17 s at 30 fps)
  15 → half a stride (~0.5 s)
  30 → full stride (~1 s, matches walking cadence at ~1 Hz)
"""


def build_chain_imu_graph_adjacency(window_len: int, device: torch.device) -> torch.Tensor:
    """Build [2T, 2T] simple chain adjacency for combined phone/watch streams.

    Node layout:
      - Nodes 0 .. T-1      : phone timesteps
      - Nodes T .. 2T-1     : watch timesteps

    Edge types:
      - Self-loops on all 2T nodes
      - Sequential edges (gap=1 only) within each stream: node[t] ↔ node[t+1]
      - Cross-sensor edges between aligned timesteps: (i, T+i) and (T+i, i)

    Receptive field with 3 stacked TemporalGCNBlock (3 internal GCN layers each):
    3 × 3 × 1 + 1 = ~19 frames — simpler than multiscale but still captures
    short-range motion patterns.
    """
    n = 2 * window_len
    adj = torch.zeros((n, n), dtype=torch.float32, device=device)

    # Self-loops
    idx_all = torch.arange(n, device=device)
    adj[idx_all, idx_all] = 1.0

    # Sequential edges (gap=1 only) within each stream
    idx = torch.arange(window_len - 1, device=device)
    adj[idx, idx + 1] = 1.0
    adj[idx + 1, idx] = 1.0
    adj[window_len + idx, window_len + idx + 1] = 1.0
    adj[window_len + idx + 1, window_len + idx] = 1.0

    # Cross-sensor edges (aligned timesteps)
    idx_t = torch.arange(window_len, device=device)
    adj[idx_t, window_len + idx_t] = 1.0
    adj[window_len + idx_t, idx_t] = 1.0

    assert_shape(adj, [n, n], "build_chain_imu_graph_adjacency.adj")
    return adj


def build_imu_graph_adjacency(window_len: int, device: torch.device) -> torch.Tensor:
    """Build [2T, 2T] multi-scale adjacency for combined phone/watch streams.

    Node layout:
      - Nodes 0 .. T-1      : phone timesteps
      - Nodes T .. 2T-1     : watch timesteps

    Edge types:
      - Self-loops on all 2T nodes
      - Multi-scale temporal edges within each stream at distances in _IMU_GRAPH_SCALES
      - Cross-sensor edges between aligned timesteps: (i, T+i) and (T+i, i)

    The original ±1-only graph limited GCN receptive field to 7 frames with
    3 layers — structurally blind to gait patterns (~30 frames).  Multi-scale
    edges give a 91-frame receptive field with the same 3 layers.
    """
    n = 2 * window_len
    adj = torch.zeros((n, n), dtype=torch.float32, device=device)

    # Self-loops
    idx_all = torch.arange(n, device=device)
    adj[idx_all, idx_all] = 1.0

    # Multi-scale intra-stream temporal edges
    for scale in _IMU_GRAPH_SCALES:
        if scale >= window_len:
            continue
        idx = torch.arange(window_len - scale, device=device)
        # Phone stream
        adj[idx, idx + scale] = 1.0
        adj[idx + scale, idx] = 1.0
        # Watch stream
        adj[window_len + idx, window_len + idx + scale] = 1.0
        adj[window_len + idx + scale, window_len + idx] = 1.0

    # Cross-sensor edges (aligned timesteps)
    idx_t = torch.arange(window_len, device=device)
    adj[idx_t, window_len + idx_t] = 1.0
    adj[window_len + idx_t, idx_t] = 1.0

    assert_shape(adj, [n, n], "build_imu_graph_adjacency.adj")
    return adj


class SensorGCNEncoder(nn.Module):
    """Per-sensor temporal GCN encoder.

    Parameter names: conv1/conv2/conv3, out_proj.
    Each conv is a GCNConv over a chain temporal graph on T nodes.
    """

    def __init__(
        self,
        input_dim: int = len(IMU_FEATURE_NAMES),
        latent_dim: int = 256,
        graph_type: str = "chain",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.graph_type = graph_type
        self.dropout = float(dropout)
        # Hidden dims match the saved Stage-2 branch family: input_dim→32→32→64→latent_dim
        self.conv1 = GCNConv(input_dim, 10, add_self_loops=False, normalize=False)
        self.conv2 = GCNConv(10, 64, add_self_loops=False, normalize=False)
        self.conv3 = GCNConv(64, 128, add_self_loops=False, normalize=False)
        self.drop1 = nn.Dropout(p=self.dropout)
        self.drop2 = nn.Dropout(p=self.dropout)
        self.drop3 = nn.Dropout(p=self.dropout)
        self.out_proj = nn.Linear(128, latent_dim)
        self._edge_cache: dict = {}

    def _get_batched_edge_index(self, b: int, t: int, device: torch.device) -> torch.Tensor:
        cache_key = (b, t, self.graph_type)
        if cache_key not in self._edge_cache:
            self_idx = torch.arange(t, device=device)
            srcs = [self_idx]
            dsts = [self_idx]
            scales = [1, 5, 15, 30] if self.graph_type == "multiscale" else [1]
            for scale in scales:
                if scale >= t:
                    continue
                idx = torch.arange(t - scale, device=device)
                srcs += [idx, idx + scale]
                dsts += [idx + scale, idx]
            src = torch.cat(srcs)
            dst = torch.cat(dsts)
            single_ei = torch.stack([src, dst], dim=0)
            batched = torch.cat([single_ei + i * t for i in range(b)], dim=1)
            self._edge_cache[cache_key] = batched
        ei = self._edge_cache[cache_key]
        if ei.device != device:
            ei = ei.to(device)
            self._edge_cache[cache_key] = ei
        return ei

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim] -> [B, T, latent_dim]"""
        b, t, _ = x.shape
        ei = self._get_batched_edge_index(b, t, x.device)
        h = x.reshape(b * t, -1)
        h = self.drop1(torch.relu(self.conv1(h, ei))).reshape(b * t, -1)
        h = self.drop2(torch.relu(self.conv2(h, ei))).reshape(b * t, -1)
        h = self.drop3(torch.relu(self.conv3(h, ei))).reshape(b * t, -1)
        return self.out_proj(h).reshape(b, t, -1)


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention with residual connection.

    Copied from cnn/cnn.py to keep the standalone classifier package isolated;
    see CLAUDE.md (the cnn/ directory is not imported by the flow pipeline).
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        return x + attn_out


class SensorCNNEncoder(nn.Module):
    """Per-sensor temporal CNN encoder (parallels SensorGCNEncoder).

    Stride-1 dilated Conv1d stack that preserves the input window length T.
    With kernel=3 and dilations 1/2/4/8, the receptive field at the final conv
    is 31 frames — sized for the canonical 32-frame window.

    Channel widths at latent_dim=D: input_dim → D/4 → D/2 → D → D.
    """

    def __init__(
        self,
        input_dim: int = len(IMU_FEATURE_NAMES),
        latent_dim: int = 256,
        dropout: float = 0.25,
        attention: bool = True,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim % 4 != 0:
            raise ValueError(f"latent_dim must be divisible by 4 for the CNN width schedule; got {latent_dim}.")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.attention = bool(attention)
        self.attention_heads = int(attention_heads)

        c1 = latent_dim // 4
        c2 = latent_dim // 2
        c3 = latent_dim
        c4 = latent_dim

        def block(in_ch: int, out_ch: int, dilation: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(p=self.dropout),
            )

        self.block1 = block(self.input_dim, c1, dilation=1)
        self.block2 = block(c1, c2, dilation=2)
        self.block3 = block(c2, c3, dilation=4)
        self.block4 = block(c3, c4, dilation=8)
        self.attn = (
            SelfAttentionBlock(d_model=latent_dim, num_heads=self.attention_heads, dropout=self.dropout)
            if self.attention
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim] -> [B, T, latent_dim]"""
        assert_shape(x, [None, None, self.input_dim], "SensorCNNEncoder.x")
        b, t, _ = x.shape
        h = x.transpose(1, 2)  # [B, input_dim, T]
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = h.transpose(1, 2)  # [B, T, latent_dim]
        if self.attn is not None:
            h = self.attn(h)
        assert_shape(h, [b, t, self.latent_dim], "SensorCNNEncoder.h")
        return h


class IMUGraphEncoder(nn.Module):
    """Encode phone and watch streams jointly as a single bipartite temporal graph.

    Nodes 0..T-1 represent phone timesteps, nodes T..2T-1 represent watch timesteps.
    Cross-sensor edges allow information to flow between the two streams at each
    aligned timestep, so the TGNN can capture phone-watch correlations explicitly.

    Args:
        input_dim:  Feature dimension of each IMU stream after build_imu_features (default 10).
        latent_dim: Output embedding dimension per node (default 256).
        depth:      Number of TemporalGraphBlock layers applied to the combined graph.
    """

    def __init__(self, input_dim: int = len(IMU_FEATURE_NAMES), latent_dim: int = 256, depth: int = 3,
                 graph_type: str = "chain") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.depth = depth
        self.graph_type = graph_type
        # Separate projections for each sensor so the model can learn sensor-specific
        # initial embeddings before the shared graph processing.
        self.phone_proj = nn.Linear(input_dim, latent_dim)
        self.watch_proj = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [TemporalGCNBlock(dim=latent_dim, num_layers=3) for _ in range(depth)]
        )
        # Lazy cache for IMU graph structures keyed by window length.
        # The IMU graph topology (adjacency + edge_index) is fixed for a given T,
        # so rebuilding it every forward call wastes ~2ms per step.
        self._imu_graph_cache: dict = {}

    def forward(self, phone_feat: torch.Tensor, watch_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run TGNN on the combined phone/watch graph and split outputs back by sensor.

        Args:
            phone_feat: [B, T, input_dim]
            watch_feat: [B, T, input_dim]

        Returns:
            phone_tokens: [B, T, latent_dim]
            watch_tokens: [B, T, latent_dim]
        """
        assert_shape(phone_feat, [None, None, self.input_dim], "IMUGraphEncoder.phone_feat")
        assert_shape(watch_feat, [None, None, self.input_dim], "IMUGraphEncoder.watch_feat")
        assert phone_feat.shape[:2] == watch_feat.shape[:2], "phone and watch must share [B, T]"

        b, t, _ = phone_feat.shape

        # Project each stream to latent dim
        phone_latent = self.phone_proj(phone_feat)  # [B, T, D]
        watch_latent = self.watch_proj(watch_feat)  # [B, T, D]

        # Concatenate along the time axis to form a single [B, 2T, D] node sequence.
        # Node i < T corresponds to phone timestep i; node T+i to watch timestep i.
        combined = torch.cat([phone_latent, watch_latent], dim=1)  # [B, 2T, D]

        # Retrieve or build the [2T, 2T] adjacency and edge_index for this window length.
        if t not in self._imu_graph_cache:
            if self.graph_type == "chain":
                adj = build_chain_imu_graph_adjacency(window_len=t, device=phone_feat.device)
            else:
                adj = build_imu_graph_adjacency(window_len=t, device=phone_feat.device)
            edge_index = build_edge_index_from_adjacency(adj) if HAS_TORCH_GEOMETRIC else None
            self._imu_graph_cache[t] = (adj, edge_index)
        adj, edge_index = self._imu_graph_cache[t]
        # Move cached tensors to current device if needed (e.g. after model.to(device))
        if adj.device != phone_feat.device:
            adj = adj.to(phone_feat.device)
            edge_index = edge_index.to(phone_feat.device) if edge_index is not None else None
            self._imu_graph_cache[t] = (adj, edge_index)

        for block in self.blocks:
            combined = block(combined, adjacency=adj, edge_index=edge_index)

        # Split back into per-sensor outputs
        phone_out = combined[:, :t, :]  # [B, T, D]
        watch_out = combined[:, t:, :]  # [B, T, D]

        assert_shape(phone_out, [b, t, self.latent_dim], "IMUGraphEncoder.phone_out")
        assert_shape(watch_out, [b, t, self.latent_dim], "IMUGraphEncoder.watch_out")
        return phone_out, watch_out


class IMULatentAligner(nn.Module):
    """Phone+watch accelerometer aligner with one encoder branch per sensor.

    `encoder_type` selects the per-branch architecture:
      - "gcn": existing temporal GCN encoder (default; preserves all current behavior)
      - "cnn": stride-1 dilated Conv1d stack (input from build_imu_features, T preserved)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        graph_type: str = "chain",
        dropout: float = 0.25,
        encoder_type: str = "gcn",
        cnn_attention: bool = True,
    ) -> None:
        super().__init__()
        encoder_type = str(encoder_type).lower()
        if encoder_type not in SUPPORTED_IMU_ENCODER_TYPES:
            raise ValueError(
                f"Unsupported encoder_type {encoder_type!r}; expected one of {SUPPORTED_IMU_ENCODER_TYPES}."
            )
        self.latent_dim = latent_dim
        self.sensor_layout = PHONE_WATCH_SENSOR_LAYOUT
        self.encoder_type = encoder_type
        self.graph_type = graph_type if encoder_type == "gcn" else ""
        self.cnn_attention = bool(cnn_attention) if encoder_type == "cnn" else False
        self.dropout = float(dropout)
        self.imu_fusion_schema = (
            IMU_FUSION_SCHEMA if encoder_type == "gcn" else IMU_FUSION_SCHEMA_CNN_DIFF
        )
        self.imu_feature_dim = SHARED_FEATURE_DIM
        if encoder_type == "gcn":
            self.phone_encoder = SensorGCNEncoder(
                input_dim=self.imu_feature_dim,
                latent_dim=latent_dim,
                graph_type=graph_type,
                dropout=self.dropout,
            )
            self.watch_encoder = SensorGCNEncoder(
                input_dim=self.imu_feature_dim,
                latent_dim=latent_dim,
                graph_type=graph_type,
                dropout=self.dropout,
            )
        else:
            self.phone_encoder = SensorCNNEncoder(
                input_dim=self.imu_feature_dim,
                latent_dim=latent_dim,
                dropout=self.dropout,
                attention=self.cnn_attention,
            )
            self.watch_encoder = SensorCNNEncoder(
                input_dim=self.imu_feature_dim,
                latent_dim=latent_dim,
                dropout=self.dropout,
                attention=self.cnn_attention,
            )
            self.diff_encoder = SensorCNNEncoder(
                input_dim=self.imu_feature_dim,
                latent_dim=latent_dim,
                dropout=self.dropout,
                attention=self.cnn_attention,
            )
        fuse_in_dim = latent_dim * 3 if encoder_type == "cnn" else latent_dim * 2
        self.fuse_tokens = nn.Sequential(
            nn.Linear(fuse_in_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.global_pool_attn = nn.Linear(latent_dim, 1)

    def forward(
        self,
        phone_accel: torch.Tensor,
        watch_accel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temporal/global embeddings from phone and watch accelerometer streams."""
        assert_shape(phone_accel, [None, None, 3], "IMULatentAligner.phone_accel")
        assert_shape(watch_accel, [phone_accel.shape[0], phone_accel.shape[1], 3], "IMULatentAligner.watch_accel")

        phone_features = build_imu_features(phone_accel)
        watch_features = build_imu_features(watch_accel)

        phone_tokens = self.phone_encoder(phone_features)
        watch_tokens = self.watch_encoder(watch_features)

        if self.encoder_type == "cnn":
            diff_features = build_imu_features(phone_accel - watch_accel)
            diff_tokens = self.diff_encoder(diff_features)
            fused = torch.cat([phone_tokens, watch_tokens, diff_tokens], dim=-1)
        else:
            fused = torch.cat([phone_tokens, watch_tokens], dim=-1)

        sensor_tokens = self.fuse_tokens(fused)
        attn_w = self.global_pool_attn(sensor_tokens).softmax(dim=1)  # [B, T, 1]
        h_global = (sensor_tokens * attn_w).sum(dim=1)                # [B, latent_dim]

        assert_shape(
            sensor_tokens,
            [phone_accel.shape[0], phone_accel.shape[1], self.latent_dim],
            "IMULatentAligner.sensor_tokens",
        )
        assert_shape(h_global, [phone_accel.shape[0], self.latent_dim], "IMULatentAligner.h_global")
        return sensor_tokens, h_global
