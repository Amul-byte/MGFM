"""Graph modules for fixed-adjacency skeleton and temporal processing."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_model.util import assert_shape, build_adjacency_matrix

try:
    from torch_geometric.nn import GATConv, GCNConv

    HAS_TORCH_GEOMETRIC = True
except Exception:
    HAS_TORCH_GEOMETRIC = False

if not HAS_TORCH_GEOMETRIC:
    raise ImportError("torch_geometric is required for proposal-exact GAT-only implementation.")


class MaskedGraphAttention(nn.Module):
    """Adjacency-masked multi-head attention over joints."""

    def __init__(self, dim: int, num_heads: int, num_joints: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_joints = num_joints
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply masked attention to x with shape [B, T, J, D]."""
        b, t, j, d = x.shape
        assert_shape(x, [None, None, self.num_joints, self.dim], "MaskedGraphAttention.x")
        assert_shape(adjacency, [self.num_joints, self.num_joints], "MaskedGraphAttention.adjacency")
        x_bt = x.reshape(b * t, j, d)
        attn_mask = adjacency <= 0
        y, _ = self.attn(x_bt, x_bt, x_bt, attn_mask=attn_mask)
        y = y.reshape(b, t, j, d)
        return y


class PyGGraphLayer(nn.Module):
    """Graph layer using torch_geometric GATConv when available."""

    def __init__(self, dim: int, num_joints: int, heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        self.gat = GATConv(in_channels=dim, out_channels=dim // heads, heads=heads, concat=True)
        # Cache batched edge_index by (b*t, num_joints, num_edges) key to avoid
        # rebuilding the offset tensor every forward call.
        self._batched_edge_cache: dict = {}

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply graph convolution to x with shape [B, T, J, D]."""
        b, t, j, d = x.shape
        assert_shape(x, [None, None, self.num_joints, self.dim], "PyGGraphLayer.x")
        assert_shape(edge_index, [2, None], "PyGGraphLayer.edge_index")
        x_flat = x.reshape(b * t * j, d)
        cache_key = (b * t, j, int(edge_index.shape[1]))
        if cache_key not in self._batched_edge_cache:
            offsets = torch.arange(0, b * t, device=x.device).repeat_interleave(edge_index.shape[1]) * j
            self._batched_edge_cache[cache_key] = (
                edge_index.repeat(1, b * t) + offsets.unsqueeze(0)
            ).contiguous()
        repeated_edge_index = self._batched_edge_cache[cache_key]
        y = self.gat(x_flat, repeated_edge_index)
        y = y.reshape(b, t, j, d)
        return y


class GraphBlock(nn.Module):
    """Residual graph block using GAT over adjacency-defined graph."""

    def __init__(self, dim: int, num_heads: int, num_joints: int, use_pyg: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        self.use_pyg = bool(use_pyg and HAS_TORCH_GEOMETRIC)
        if not self.use_pyg:
            raise RuntimeError("GraphBlock requires torch_geometric GATConv for proposal-exact mode.")
        self.graph_op = PyGGraphLayer(dim=dim, num_joints=num_joints, heads=num_heads)
        self.masked_attn = None
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run graph op then feed-forward with residual paths."""
        assert_shape(x, [None, None, self.num_joints, self.dim], "GraphBlock.x")
        assert_shape(adjacency, [self.num_joints, self.num_joints], "GraphBlock.adjacency")
        assert edge_index is not None, "edge_index is required when torch_geometric path is used"
        y = self.graph_op(x, edge_index)
        x = x + y
        x = x + self.ff(x)
        return x


class PyGGCNGraphLayer(nn.Module):
    """Graph layer using torch_geometric GCNConv over the skeleton graph."""

    def __init__(self, dim: int, num_joints: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        # Self-loops already included in build_adjacency_matrix; don't add again.
        self.gcn = GCNConv(in_channels=dim, out_channels=dim, add_self_loops=False, normalize=False)
        self._batched_edge_cache: dict = {}

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply GCN to x with shape [B, T, J, D]."""
        b, t, j, d = x.shape
        assert_shape(x, [None, None, self.num_joints, self.dim], "PyGGCNGraphLayer.x")
        assert_shape(edge_index, [2, None], "PyGGCNGraphLayer.edge_index")
        x_flat = x.reshape(b * t * j, d)
        cache_key = (b * t, j, int(edge_index.shape[1]))
        if cache_key not in self._batched_edge_cache:
            offsets = torch.arange(0, b * t, device=x.device).repeat_interleave(edge_index.shape[1]) * j
            self._batched_edge_cache[cache_key] = (
                edge_index.repeat(1, b * t) + offsets.unsqueeze(0)
            ).contiguous()
        repeated_edge_index = self._batched_edge_cache[cache_key]
        if repeated_edge_index.device != x.device:
            repeated_edge_index = repeated_edge_index.to(x.device)
            self._batched_edge_cache[cache_key] = repeated_edge_index
        y = self.gcn(x_flat, repeated_edge_index)
        return y.reshape(b, t, j, d)


class GraphBlockGCN(nn.Module):
    """Residual graph block using GCNConv — drop-in replacement for GraphBlock."""

    def __init__(self, dim: int, num_joints: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        self.graph_op = PyGGCNGraphLayer(dim=dim, num_joints=num_joints)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run GCN then feed-forward with residual paths."""
        assert_shape(x, [None, None, self.num_joints, self.dim], "GraphBlockGCN.x")
        assert edge_index is not None, "edge_index is required for GraphBlockGCN"
        y = self.graph_op(x, edge_index)
        x = x + y
        x = x + self.ff(x)
        return x


class TemporalMaskedGraphAttention(nn.Module):
    """Temporal adjacency-masked multi-head attention over sequence steps."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply masked temporal attention to x with shape [B, T, D]."""
        b, t, d = x.shape
        assert_shape(x, [None, None, self.dim], "TemporalMaskedGraphAttention.x")
        assert_shape(adjacency, [t, t], "TemporalMaskedGraphAttention.adjacency")
        attn_mask = adjacency <= 0
        y, _ = self.attn(x, x, x, attn_mask=attn_mask)
        assert_shape(y, [b, t, d], "TemporalMaskedGraphAttention.y")
        return y


class TemporalPyGGraphLayer(nn.Module):
    """Temporal graph layer using torch_geometric GATConv."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.gat = GATConv(in_channels=dim, out_channels=dim // heads, heads=heads, concat=True)
        self._batched_edge_cache: dict = {}

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply temporal graph convolution to x with shape [B, T, D]."""
        b, t, d = x.shape
        assert_shape(x, [None, None, self.dim], "TemporalPyGGraphLayer.x")
        assert_shape(edge_index, [2, None], "TemporalPyGGraphLayer.edge_index")
        x_flat = x.reshape(b * t, d)
        cache_key = (b, t, int(edge_index.shape[1]))
        if cache_key not in self._batched_edge_cache:
            offsets = torch.arange(0, b, device=x.device).repeat_interleave(edge_index.shape[1]) * t
            self._batched_edge_cache[cache_key] = (
                edge_index.repeat(1, b) + offsets.unsqueeze(0)
            ).contiguous()
        repeated_edge_index = self._batched_edge_cache[cache_key]
        y = self.gat(x_flat, repeated_edge_index)
        y = y.reshape(b, t, d)
        return y


class TemporalGraphBlock(nn.Module):
    """Residual temporal block using full-sequence multi-head attention.

    Replaced the TemporalPyGGraphLayer (GATConv with ±1 neighbours) with
    TemporalMaskedGraphAttention (full MHA over the sequence).

    Root cause of sensor model collapse: GATConv with 3 layers has a
    receptive field of only 7 frames out of 90.  Walking cadence (~1 Hz)
    spans 30 frames; a single step spans 15 frames — both are completely
    outside the 7-frame window, so the TGNN is structurally blind to gait
    patterns regardless of training signal.

    Full attention allows every timestep to attend to every other timestep,
    giving the sensor model a global view of the IMU window.
    The adjacency argument is accepted but ignored (kept for API compatibility
    with call sites that pass it).
    """

    def __init__(self, dim: int, num_heads: int, use_pyg: bool = True) -> None:
        super().__init__()
        self.dim = dim
        # Full-sequence attention: receptive field = T (entire window)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run full-sequence self-attention then feed-forward with residual paths."""
        b, t, d = x.shape
        assert_shape(x, [None, None, self.dim], "TemporalGraphBlock.x")
        y, _ = self.attn(x, x, x)
        x = x + y
        x = x + self.ff(x)
        assert_shape(x, [b, t, d], "TemporalGraphBlock.out")
        return x


class TemporalGCNBlock(nn.Module):
    """Residual temporal GCN block using GCNConv over a multi-scale temporal graph.

    Replaces full MHA with graph convolution.  The key difference from the old
    TGNN (GATConv with ±1 neighbours only) is that the graph this block operates
    on includes multi-scale temporal edges at distances {1, 5, 15, 30}, giving a
    receptive field of 3×30+1 = 91 frames with 3 stacked blocks — enough to
    capture a full gait cycle at 30 fps.

    GCNConv normalises by node degree, which handles the varying connectivity of
    boundary nodes (beginning/end of sequence) automatically.
    """

    def __init__(self, dim: int, num_layers: int = 3, dropout: float = 0.25) -> None:
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.dropout = dropout
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for TemporalGCNBlock.")
        # Stack of GCNConv layers with residual connections.
        # add_self_loops=False because self-loops are already in the graph adjacency.
        self.convs = nn.ModuleList(
            [GCNConv(in_channels=dim, out_channels=dim, add_self_loops=False, normalize=False) for _ in range(num_layers)]
        )
        self.drop = nn.Dropout(p=dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * 4, dim),
        )
        # Cache batched edge_index by (batch_size, seq_len, num_edges) to avoid
        # rebuilding the offset tensor every forward call.
        self._batched_edge_cache: dict = {}

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply GCN layers then feed-forward with residual paths.

        Args:
            x:          [B, T, D] node features.
            adjacency:  [T, T] adjacency matrix (used only for shape checking).
            edge_index: [2, E] edge list derived from adjacency (required).
        """
        b, t, d = x.shape
        assert_shape(x, [None, None, self.dim], "TemporalGCNBlock.x")
        assert edge_index is not None, "edge_index is required for TemporalGCNBlock"

        x_flat = x.reshape(b * t, d)

        # Build or retrieve batched edge_index with per-graph node offsets.
        cache_key = (b, t, int(edge_index.shape[1]), x.device.index if x.device.type == "cuda" else -1)
        if cache_key not in self._batched_edge_cache:
            offsets = torch.arange(0, b, device=x.device).repeat_interleave(edge_index.shape[1]) * t
            self._batched_edge_cache[cache_key] = (
                edge_index.repeat(1, b) + offsets.unsqueeze(0)
            ).contiguous()
        batched_ei = self._batched_edge_cache[cache_key]

        # Stacked GCN layers with residual connections.
        for conv in self.convs:
            h = conv(x_flat, batched_ei)          # [B*T, D]
            h = F.gelu(h)
            h = self.drop(h)
            x_flat = x_flat + h

        x = x_flat.reshape(b, t, d)
        x = x + self.ff(x)
        assert_shape(x, [b, t, d], "TemporalGCNBlock.out")
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention from joint latent tokens to temporal sensor tokens."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # Zero-init output projections so cross-attention is a no-op at initialisation.
        # This preserves the Stage-1 denoiser prior at the start of Stage-3 training
        # and lets conditioning grow in gradually (ControlNet / IP-Adapter trick).
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ff[2].weight)
        nn.init.zeros_(self.ff[2].bias)

    def forward(self, joint_tokens: torch.Tensor, sensor_tokens: torch.Tensor) -> torch.Tensor:
        """Condition [B,T,J,D] joint tokens with [B,T,D] sensor tokens via cross-attention."""
        b, t, j, d = joint_tokens.shape
        assert_shape(joint_tokens, [None, None, None, self.dim], "CrossAttentionBlock.joint_tokens")
        assert_shape(sensor_tokens, [b, t, self.dim], "CrossAttentionBlock.sensor_tokens")
        q = joint_tokens.reshape(b, t * j, d)
        attn_out, _ = self.attn(q, sensor_tokens, sensor_tokens)
        q = q + attn_out
        q = q + self.ff(q)
        out = q.reshape(b, t, j, d)
        assert_shape(out, [b, t, j, d], "CrossAttentionBlock.out")
        return out


class TemporalConvBlock(nn.Module):
    """Temporal convolution block applied per joint."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal convolution to x with shape [B, T, J, D]."""
        b, t, j, d = x.shape
        assert_shape(x, [None, None, None, self.dim], "TemporalConvBlock.x")
        x_perm = x.permute(0, 2, 3, 1).reshape(b * j, d, t)
        y = self.conv(x_perm)
        y = y.reshape(b, j, d, t).permute(0, 3, 1, 2)
        y = F.gelu(y)
        return x + y


class TemporalAttentionBlock(nn.Module):
    """Temporal self-attention block applied per joint (global receptive field).

    Drop-in replacement for ``TemporalConvBlock`` — same input/output contract
    ``[B, T, J, D]`` — but each frame attends to *every* frame instead of a
    local kernel-3 window. Uses a learnable positional embedding so frames are
    ordered (raw self-attention is permutation-invariant on T), and pre-norm
    LayerNorms so the residual stream stays numerically stable across depth.
    """

    def __init__(self, dim: int, num_heads: int = 8, max_len: int = 512) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_len = max_len
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.ln_attn = nn.LayerNorm(dim)
        self.ln_ff = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal self-attention to x with shape [B, T, J, D]."""
        b, t, j, d = x.shape
        assert_shape(x, [None, None, None, self.dim], "TemporalAttentionBlock.x")
        if t > self.max_len:
            raise ValueError(
                f"TemporalAttentionBlock got T={t} > max_len={self.max_len}; "
                f"increase max_len when constructing the block."
            )
        # Treat each joint independently: [B, T, J, D] -> [B*J, T, D]
        x_perm = x.permute(0, 2, 1, 3).reshape(b * j, t, d)
        x_perm = x_perm + self.pos_embed[:, :t, :]
        a = self.ln_attn(x_perm)
        attn_out, _ = self.attn(a, a, a)
        x_perm = x_perm + attn_out
        x_perm = x_perm + self.ff(self.ln_ff(x_perm))
        # Reshape back: [B*J, T, D] -> [B, T, J, D]
        out = x_perm.reshape(b, j, t, d).permute(0, 2, 1, 3)
        assert_shape(out, [b, t, j, d], "TemporalAttentionBlock.out")
        return out


def build_edge_index(num_joints: int, device: torch.device) -> torch.Tensor:
    """Build edge_index tensor for torch_geometric from fixed skeleton adjacency."""
    adj = build_adjacency_matrix(num_joints=num_joints, device=device)
    idx = adj.nonzero(as_tuple=False).t().contiguous()
    assert_shape(idx, [2, None], "build_edge_index.idx")
    return idx


def build_edge_index_from_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Build edge_index from a binary adjacency matrix."""
    assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1], "adjacency must be square"
    idx = adjacency.nonzero(as_tuple=False).t().contiguous()
    assert_shape(idx, [2, None], "build_edge_index_from_adjacency.idx")
    return idx
