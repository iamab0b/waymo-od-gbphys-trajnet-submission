"""
GRU-based trajectory decoder for GB-Phys TrajNet (Stage 2).

Autoregressively unrolls a GRU conditioned on agent context, a goal offset,
and an initial kinematic state.  At each step the GRU output drives the
bicycle kinematic model, ensuring physical plausibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bicycle_model import BicycleStep


class GRUDecoder(nn.Module):
    """
    Autoregressive GRU decoder that outputs physically-plausible trajectories.

    Conditioning:
      h_0  = Linear(agent_emb || goal_offset || initial_state) -> [num_layers, B, hidden_dim]

    Per-step:
      input_t  = Linear(current_state[:5] || goal_offset[:2]) -> [B, hidden_dim]
      h_{t+1}  = GRU(input_t, h_t)
      (a, d)   = tanh(Linear(h_out, 2))    -- scaled to [-1, 1]; bicycle scales by a_max/delta_max
      new_state = BicycleStep(state_t, control_t)

    Args:
        agent_dim:   Dimension of agent history embedding (default 128).
        state_dim:   Kinematic state dimension (default 5: x,y,theta,v,phi).
        goal_dim:    Goal offset dimension (default 2: rel_x, rel_y).
        hidden_dim:  GRU hidden size (default 256).
        num_layers:  Number of GRU layers (default 2).
        dropout:     Dropout between GRU layers (default 0.1).
        dt:          Integration timestep (seconds, default 0.1).
        wheelbase:   Bicycle model wheelbase (metres).
        a_max:       Max acceleration magnitude.
        delta_max:   Max steering-rate magnitude.
        T:           Prediction horizon (timesteps, default 80).
    """

    def __init__(self,
                 agent_dim:  int   = 128,
                 state_dim:  int   = 5,
                 goal_dim:   int   = 2,
                 hidden_dim: int   = 256,
                 num_layers: int   = 2,
                 dropout:    float = 0.1,
                 dt:         float = 0.1,
                 wheelbase:  float = 2.7,
                 a_max:      float = 5.0,
                 delta_max:  float = 0.5,
                 T:          int   = 80):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.T          = T
        self.state_dim  = state_dim
        self.goal_dim   = goal_dim

        # ------------------------------------------------------------------
        # Condition encoder: maps initial context to GRU h_0
        # input = concat(agent_emb, goal_offset, initial_state)
        # size  = agent_dim + goal_dim + state_dim = 128 + 2 + 5 = 135
        # ------------------------------------------------------------------
        condition_dim = agent_dim + goal_dim + state_dim
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Project to [num_layers, B, hidden_dim] via a stack of linears
        self.h0_proj = nn.Linear(hidden_dim, num_layers * hidden_dim)

        # ------------------------------------------------------------------
        # Per-step input projection: concat(current_state, goal_offset) -> hidden_dim
        # size = state_dim + goal_dim = 5 + 2 = 7
        # ------------------------------------------------------------------
        step_input_dim = state_dim + goal_dim
        self.step_input_proj = nn.Linear(step_input_dim, hidden_dim)

        # ------------------------------------------------------------------
        # GRU cell (batch_first=True)
        # ------------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output head: hidden -> 2 controls (a_t, delta_t) in [-1, 1]
        self.output_head = nn.Linear(hidden_dim, 2)

        # Bicycle step (physics, no learnable params)
        self.bike_step = BicycleStep(dt=dt, wheelbase=wheelbase,
                                     a_max=a_max, delta_max=delta_max)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_h0(self, agent_embedding: torch.Tensor,
                  goal_offset: torch.Tensor,
                  initial_state: torch.Tensor) -> torch.Tensor:
        """
        Build initial hidden state from conditioning inputs.

        Returns:
            h0: [num_layers, batch, hidden_dim]
        """
        B = agent_embedding.shape[0]
        condition = torch.cat([agent_embedding, goal_offset, initial_state], dim=-1)
        h = self.condition_encoder(condition)              # [B, hidden_dim]
        h = self.h0_proj(h)                                # [B, num_layers*hidden_dim]
        h = h.view(B, self.num_layers, self.hidden_dim)    # [B, nl, H]
        h = h.permute(1, 0, 2).contiguous()                # [nl, B, H]
        return h

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self,
                agent_embedding: torch.Tensor,
                goal_position:   torch.Tensor,
                initial_state:   torch.Tensor) -> dict:
        """
        Autoregressively decode a trajectory from the given conditioning.

        Args:
            agent_embedding: [batch, agent_dim]   from AgentHistoryEncoder.
            goal_position:   [batch, 2]           goal (relative to current pos).
            initial_state:   [batch, 5]           (x, y, theta, v, phi).

        Returns:
            dict with keys:
                trajectories : [batch, T, 2]   (x, y) positions
                controls     : [batch, T, 2]   (a_t, delta_t) in [-1, 1]
                full_states  : [batch, T, 5]   full kinematic state per step
        """
        B = agent_embedding.shape[0]
        device = agent_embedding.device

        # Goal offset relative to initial position
        goal_offset = goal_position - initial_state[:, :2]  # [B, 2]

        # Build GRU h_0
        h = self._build_h0(agent_embedding, goal_offset, initial_state)

        state = initial_state.clone()    # [B, 5]
        traj_list   = []
        control_list = []
        state_list   = []

        for t in range(self.T):
            # Current goal offset (dynamic: towards goal from current pos)
            goal_offset_t = goal_position - state[:, :2]  # [B, 2]

            # Step input
            step_in = torch.cat([state, goal_offset_t], dim=-1)  # [B, 7]
            step_in = self.step_input_proj(step_in)               # [B, H]
            step_in = step_in.unsqueeze(1)                        # [B, 1, H]

            # GRU step
            gru_out, h = self.gru(step_in, h)                     # [B, 1, H]
            gru_out = gru_out.squeeze(1)                          # [B, H]

            # Control output in [-1, 1]
            control = torch.tanh(self.output_head(gru_out))       # [B, 2]

            # Bicycle step
            state = self.bike_step(state, control)                # [B, 5]

            traj_list.append(state[:, :2])
            control_list.append(control)
            state_list.append(state)

        trajectories = torch.stack(traj_list,    dim=1)   # [B, T, 2]
        controls     = torch.stack(control_list, dim=1)   # [B, T, 2]
        full_states  = torch.stack(state_list,   dim=1)   # [B, T, 5]

        return {
            'trajectories': trajectories,
            'controls':     controls,
            'full_states':  full_states,
        }
