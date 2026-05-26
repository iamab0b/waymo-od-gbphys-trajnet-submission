"""
Encoder modules for GB-Phys TrajNet.

Implements:
  - AgentHistoryEncoder  : 1D CNN (or Transformer) over agent history
  - RoadContextEncoder   : PointNet-style MLP for map features
  - TrafficLightEncoder  : encodes traffic light states and positions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentHistoryEncoder(nn.Module):
    """
    Encodes per-agent history sequences to a fixed-size embedding.

    Two variants:
      1. 1D CNN  (default) – 3 Conv1d layers with BN + ReLU, global avg pool.
      2. Transformer      – 2-layer TransformerEncoder with positional encoding.

    Args:
        input_dim:       Number of input features per timestep.
        embedding_dim:   Output embedding size (default 128).
        use_transformer: If True, use the Transformer variant.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 128,
                 use_transformer: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_transformer = use_transformer

        if not use_transformer:
            # ---------------------------------------------------------------
            # Option A: 1D CNN encoder
            # Input [B, T, C] transposed to [B, C, T] for Conv1d
            # ---------------------------------------------------------------
            self.conv_layers = nn.Sequential(
                # Layer 1: input_dim -> 64
                nn.Conv1d(input_dim, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                # Layer 2: 64 -> 128
                nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                # Layer 3: 128 -> embedding_dim
                nn.Conv1d(128, embedding_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
            )
            # Global average pooling is applied in forward()
            self.output_proj = nn.Identity()
        else:
            # ---------------------------------------------------------------
            # Option B: Transformer encoder
            # ---------------------------------------------------------------
            self.input_proj = nn.Linear(input_dim, embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            # Positional encoding buffer
            self._build_pos_encoding(max_len=128, d_model=embedding_dim)
            self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def _build_pos_encoding(self, max_len: int, d_model: int):
        """Pre-compute sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_encoding', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, T, input_dim]  agent history features.

        Returns:
            embedding: [batch, embedding_dim]
        """
        if not self.use_transformer:
            # Conv1d expects [B, C, T]
            x = x.transpose(1, 2)          # [B, input_dim, T]
            x = self.conv_layers(x)         # [B, embedding_dim, T]
            x = x.mean(dim=2)              # Global average pooling -> [B, embedding_dim]
        else:
            T = x.shape[1]
            x = self.input_proj(x)          # [B, T, embedding_dim]
            x = x + self.pos_encoding[:, :T, :]
            x = self.transformer(x)         # [B, T, embedding_dim]
            x = x.mean(dim=1)              # [B, embedding_dim]
            x = self.output_proj(x)
        return x


class RoadContextEncoder(nn.Module):
    """
    PointNet-style encoder for road-graph context.

    Each point is described by 6 features (xyz + dir_xyz).  A shared MLP
    lifts each point to a high-dimensional feature; global max-pooling
    aggregates across all points to produce a scene-level descriptor.

    Args:
        embedding_dim: Output embedding size (default 128).
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Shared MLP: 6 -> 64 -> 128 -> embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, road_pts: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            road_pts: [batch, N_pts, 6]  xyz + dir_xyz.
            mask:     [batch, N_pts]     boolean, True = valid point.
                      If None, all points are treated as valid.

        Returns:
            embedding: [batch, embedding_dim]
        """
        feats = self.mlp(road_pts)  # [B, N, embedding_dim]

        if mask is not None:
            # Set invalid-point features to -inf before max-pooling
            mask_expanded = mask.unsqueeze(-1).float()          # [B, N, 1]
            feats = feats * mask_expanded + (1.0 - mask_expanded) * (-1e9)

        # Global max-pooling over points
        embedding, _ = feats.max(dim=1)  # [B, embedding_dim]
        return embedding


class TrafficLightEncoder(nn.Module):
    """
    Encodes traffic-light states and positions into a context embedding.

    Input combines:
      - Traffic-light state (one-hot over 9 states, WOMD convention)
      - (x, y, z) position of each light

    Args:
        embedding_dim: Output embedding size (default 64).
        num_tl_states: Number of possible traffic-light states (default 9).
    """

    def __init__(self, embedding_dim: int = 64, num_tl_states: int = 9):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_tl_states = num_tl_states

        # Each light: one-hot state (num_tl_states) + xyz (3) = num_tl_states+3
        per_light_dim = num_tl_states + 3
        self.mlp = nn.Sequential(
            nn.Linear(per_light_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.pool_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, tl_state: torch.Tensor,
                tl_xyz: torch.Tensor,
                tl_valid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tl_state: [batch, T_tl, N_lights]  integer state indices (0..num_tl_states-1)
                      T_tl is typically 1 (current) or 11 (past+current).
            tl_xyz:   [batch, T_tl, N_lights, 3]  positions.
            tl_valid: [batch, T_tl, N_lights]  validity mask (integer).

        Returns:
            embedding: [batch, embedding_dim]
        """
        B, T_tl, N_lights = tl_state.shape

        # One-hot encode states
        state_clamped = tl_state.clamp(0, self.num_tl_states - 1)
        state_oh = F.one_hot(state_clamped, num_classes=self.num_tl_states).float()
        # [B, T_tl, N_lights, num_tl_states]

        # Concatenate with xyz
        # tl_xyz: [B, T_tl, N_lights, 3]
        per_light = torch.cat([state_oh, tl_xyz], dim=-1)  # [B, T_tl, N, num_tl_states+3]

        # Flatten time and lights
        per_light_flat = per_light.view(B, T_tl * N_lights, -1)    # [B, T*N, D]
        valid_flat = (tl_valid.view(B, T_tl * N_lights) > 0).float()  # [B, T*N]

        feats = self.mlp(per_light_flat)  # [B, T*N, embedding_dim]

        # Mask invalid lights and max-pool
        valid_exp = valid_flat.unsqueeze(-1)
        feats = feats * valid_exp + (1.0 - valid_exp) * (-1e9)
        embedding, _ = feats.max(dim=1)   # [B, embedding_dim]
        embedding = self.pool_proj(embedding)
        return embedding
