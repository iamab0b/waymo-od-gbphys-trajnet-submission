"""
Baseline trajectory prediction models for GB-Phys TrajNet.

Implements:
  - ConstantVelocityBaseline : no learning, uses last observed velocity
  - LSTMBaseline             : LSTM encoder-decoder multi-modal prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstantVelocityBaseline:
    """
    Pure physics baseline: propagate agents at constant velocity.

    No learnable parameters.  Uses the last observed velocity from the
    input_states tensor to roll out T=80 steps at dt=0.1 s.
    """

    def __init__(self, T: int = 80, dt: float = 0.1, K: int = 6):
        """
        Args:
            T:   Number of future timesteps to predict.
            dt:  Integration timestep (seconds).
            K:   Number of trajectory modes (all identical for CV).
        """
        self.T  = T
        self.dt = dt
        self.K  = K

    def predict(self, batch: dict) -> dict:
        """
        Generate constant-velocity predictions.

        Args:
            batch: Dict of numpy arrays OR torch tensors with keys:
                   - input_states   : [B, 128, 11, 7]
                   - sample_is_valid: [B, 128]

        Returns:
            dict with keys:
                trajectories : [B, 128, K, 80, 2]  (x, y) – all K copies identical
                confidences  : [B, 128, K]          (uniform = 1/K)
        """
        is_torch = isinstance(batch['input_states'], torch.Tensor)

        if is_torch:
            inp  = batch['input_states'].detach().cpu().numpy()
            valid = batch['sample_is_valid'].detach().cpu().numpy()
        else:
            inp  = batch['input_states']
            valid = batch['sample_is_valid']

        B, A, _, _ = inp.shape
        T = self.T
        K = self.K
        dt = self.dt

        # Current position and velocity (last timestep = index 10)
        cur_pos = inp[:, :, -1, 0:2]   # [B, 128, 2]
        cur_vel = inp[:, :, -1, 5:7]   # [B, 128, 2]

        # Roll out: pos_t = cur_pos + cur_vel * t * dt
        steps = np.arange(1, T + 1, dtype=np.float32)  # [T]
        # [B, A, 1, 2] + [B, A, 1, 2] * [T] -> [B, A, T, 2]
        trajs = cur_pos[:, :, np.newaxis, :] + \
                cur_vel[:, :, np.newaxis, :] * steps[np.newaxis, np.newaxis, :, np.newaxis] * dt

        # Replicate K times
        trajs_k = np.broadcast_to(trajs[:, :, np.newaxis, :, :],
                                   (B, A, K, T, 2)).copy()  # [B, 128, K, T, 2]

        # Zero out invalid agents
        mask = valid[:, :, np.newaxis, np.newaxis, np.newaxis]  # [B, 128, 1, 1, 1]
        trajs_k = trajs_k * mask.astype(np.float32)

        # Uniform confidences
        confs = np.full((B, A, K), 1.0 / K, dtype=np.float32)

        if is_torch:
            device = batch['input_states'].device
            return {
                'trajectories': torch.from_numpy(trajs_k).to(device),
                'confidences':  torch.from_numpy(confs).to(device),
            }
        return {
            'trajectories': trajs_k,
            'confidences':  confs,
        }


# ---------------------------------------------------------------------------
# LSTM Baseline
# ---------------------------------------------------------------------------

class _LSTMEncoder(nn.Module):
    """LSTM encoder for agent history."""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, T, input_dim]
        Returns:
            h_n: [num_layers, batch, hidden_dim]
            c_n: [num_layers, batch, hidden_dim]
        """
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class _LSTMDecoder(nn.Module):
    """Autoregressive LSTM decoder for trajectory prediction."""

    def __init__(self, hidden_dim: int = 256, output_dim: int = 2,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, h0: torch.Tensor, c0: torch.Tensor,
                T: int, start_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h0, c0:    encoder hidden states [num_layers, B, H]
            T:         number of steps to decode
            start_pos: [B, 2] starting position

        Returns:
            positions: [B, T, 2]
        """
        B = start_pos.shape[0]
        device = start_pos.device
        h, c = h0, c0

        pos = start_pos          # [B, 2]
        preds = []
        for _ in range(T):
            inp = pos.unsqueeze(1)               # [B, 1, 2]
            out, (h, c) = self.lstm(inp, (h, c)) # [B, 1, H]
            delta = self.output_proj(out.squeeze(1))  # [B, 2]
            pos   = pos + delta                  # residual position
            preds.append(pos)

        return torch.stack(preds, dim=1)  # [B, T, 2]


class LSTMBaseline(nn.Module):
    """
    Multi-modal LSTM encoder-decoder trajectory predictor.

    Encodes agent history with an LSTM, then decodes K diverse trajectories
    by perturbing the latent state.  A learned confidence head scores
    each mode.

    Args:
        input_dim:  Number of input features per timestep.
        hidden_dim: LSTM hidden size (default 256).
        num_layers: Number of LSTM layers (default 2).
        K:          Number of predicted trajectory modes (default 6).
        T:          Prediction horizon (default 80).
        dropout:    Dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, K: int = 6, T: int = 80,
                 dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = _LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)

        # K decoders sharing architecture but separate parameters
        self.decoders = nn.ModuleList([
            _LSTMDecoder(hidden_dim, 2, num_layers, dropout)
            for _ in range(K)
        ])

        # K-way latent perturbation (mode embeddings)
        self.mode_embeddings = nn.Embedding(K, hidden_dim)

        # Confidence head: attends to hidden state -> K scores
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, K),
        )

    def forward(self, agent_history: torch.Tensor,
                current_pos: torch.Tensor = None) -> dict:
        """
        Args:
            agent_history: [batch, 128, T_hist, input_dim]  agent histories.
            current_pos:   [batch, 128, 2]  current positions (optional).
                           If None, last position from history is used.

        Returns:
            dict with keys:
                trajectories: [batch, 128, K, T, 2]
                confidences:  [batch, 128, K]
        """
        B, A, T_hist, feat_dim = agent_history.shape

        # Flatten agents into batch dimension
        hist_flat = agent_history.view(B * A, T_hist, feat_dim)

        if current_pos is None:
            # Use last observed x, y from history
            pos_flat = hist_flat[:, -1, :2]   # [B*A, 2]
        else:
            pos_flat = current_pos.view(B * A, 2)

        # Encode
        h_n, c_n = self.encoder(hist_flat)  # [nl, B*A, H]

        # Global hidden state for confidence scoring (top layer)
        global_h = h_n[-1]  # [B*A, H]

        # Confidence scores
        conf_logits = self.confidence_head(global_h)  # [B*A, K]
        confidences = F.softmax(conf_logits, dim=-1)   # [B*A, K]

        # Decode K trajectories (add mode embedding perturbation to h)
        trajs = []
        mode_emb = self.mode_embeddings.weight  # [K, H]

        for k in range(self.K):
            # Perturb top-layer hidden state with mode embedding
            h_k = h_n.clone()
            h_k[-1] = h_k[-1] + mode_emb[k:k+1, :]  # broadcast
            traj_k = self.decoders[k](h_k, c_n, self.T, pos_flat)  # [B*A, T, 2]
            trajs.append(traj_k)

        trajs = torch.stack(trajs, dim=1)   # [B*A, K, T, 2]

        trajs       = trajs.view(B, A, self.K, self.T, 2)
        confidences = confidences.view(B, A, self.K)

        return {
            'trajectories': trajs,
            'confidences':  confidences,
        }
