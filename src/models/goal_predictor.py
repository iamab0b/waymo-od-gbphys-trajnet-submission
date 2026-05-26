"""
Goal prediction module for GB-Phys TrajNet (Stage 1).

Components:
  - GoalCandidateSampler   : samples N goal candidates from the roadgraph
  - GoalScoringNetwork     : scores candidates given agent + road context
  - GoalPredictor          : combines the above into a single Stage-1 module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalCandidateSampler(nn.Module):
    """
    Sample N goal candidates from roadgraph points for each agent.

    Candidates are drawn within a speed-dependent radius around the agent's
    current position.  For efficiency the sampling is purely spatial (no
    learning) and operates on the full batch simultaneously.
    """

    def __init__(self, N: int = 64, base_radius: float = 50.0,
                 speed_scale: float = 5.0):
        """
        Args:
            N:            Number of goal candidates to sample.
            base_radius:  Minimum search radius in metres.
            speed_scale:  Multiplies speed to extend search radius.
        """
        super().__init__()
        self.N = N
        self.base_radius = base_radius
        self.speed_scale = speed_scale

    @torch.no_grad()
    def sample_candidates(self,
                          current_pos: torch.Tensor,
                          current_vel: torch.Tensor,
                          roadgraph_xyz: torch.Tensor,
                          roadgraph_dir: torch.Tensor,
                          roadgraph_valid: torch.Tensor,
                          N: int = None) -> torch.Tensor:
        """
        Sample N candidate goal points from the roadgraph for each agent.

        Args:
            current_pos:    [batch, 2]       current (x, y) of the agent.
            current_vel:    [batch, 2]       current (vx, vy).
            roadgraph_xyz:  [batch, M, 3]    roadgraph sample positions.
            roadgraph_dir:  [batch, M, 3]    roadgraph direction vectors.
            roadgraph_valid:[batch, M, 1]    validity flags (int64 or float).
            N:              Override default number of candidates.

        Returns:
            candidates: [batch, N, 4]  (rel_x, rel_y, dir_x, dir_y)
                        positions are RELATIVE to current_pos.
        """
        if N is None:
            N = self.N

        B, M, _ = roadgraph_xyz.shape
        device   = current_pos.device

        # Compute speed-dependent radius
        speed  = torch.norm(current_vel, dim=-1, keepdim=True)  # [B, 1]
        radius = self.base_radius + speed * self.speed_scale     # [B, 1]

        # Distances from agent to each road point
        road_xy = roadgraph_xyz[..., :2]                          # [B, M, 2]
        delta   = road_xy - current_pos.unsqueeze(1)              # [B, M, 2]
        dist    = torch.norm(delta, dim=-1)                       # [B, M]

        # Build valid mask (within radius & road valid)
        valid_flag = (roadgraph_valid[..., 0] > 0).float()        # [B, M]
        in_radius  = (dist <= radius).float()                     # [B, M]
        score      = valid_flag * in_radius + 1e-10               # add eps

        # Weighted sampling without replacement
        # Use top-N by distance-weighted score for deterministic differentiable
        # selection at eval time; stochastic during training via multinomial.
        if self.training:
            indices = torch.multinomial(score, num_samples=N, replacement=True)
        else:
            _, indices = torch.topk(score, k=N, dim=-1)

        # Gather positions and directions
        idx_exp   = indices.unsqueeze(-1).expand(B, N, 3)
        cand_xyz  = torch.gather(roadgraph_xyz, 1, idx_exp)       # [B, N, 3]
        cand_dir  = torch.gather(roadgraph_dir, 1, idx_exp)       # [B, N, 3]

        # Convert to relative coordinates
        rel_xy = cand_xyz[..., :2] - current_pos.unsqueeze(1)     # [B, N, 2]
        dir_xy = cand_dir[..., :2]                                 # [B, N, 2]

        candidates = torch.cat([rel_xy, dir_xy], dim=-1)           # [B, N, 4]
        return candidates


class GoalScoringNetwork(nn.Module):
    """
    Score goal candidates using agent + road context via dot-product attention.

    Architecture:
      1. Encode each candidate [4] -> [hidden_dim] via a small MLP.
      2. Build context from concat(agent_emb, road_emb) -> [hidden_dim].
      3. Score via dot product between context and each candidate embedding,
         then return raw logits (softmax applied outside for flexibility).
    """

    def __init__(self, agent_dim: int = 128, road_dim: int = 128,
                 candidate_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Candidate encoder: 4 -> 64 -> hidden_dim
        self.cand_encoder = nn.Sequential(
            nn.Linear(candidate_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_dim),
        )

        # Context encoder: concat(agent, road) -> hidden_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(agent_dim + road_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, hidden_dim),
        )

    def forward(self,
                agent_embedding: torch.Tensor,
                road_embedding: torch.Tensor,
                candidates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_embedding: [batch, agent_dim]
            road_embedding:  [batch, road_dim]
            candidates:      [batch, N_candidates, candidate_dim]

        Returns:
            logits: [batch, N_candidates]  (unnormalized scores)
        """
        B, N, _ = candidates.shape

        # Encode candidates: [B, N, hidden_dim]
        cand_emb = self.cand_encoder(candidates)

        # Encode context: [B, hidden_dim]
        context = torch.cat([agent_embedding, road_embedding], dim=-1)
        ctx_emb = self.context_encoder(context)   # [B, hidden_dim]

        # Dot-product attention: [B, 1, H] x [B, N, H]^T -> [B, N]
        # Scaled by sqrt(hidden_dim)
        scale   = math.sqrt(self.hidden_dim) if hasattr(self, '_scale') else (self.hidden_dim ** 0.5)
        logits  = (cand_emb * ctx_emb.unsqueeze(1)).sum(dim=-1) / scale  # [B, N]
        return logits


# Need math for GoalScoringNetwork
import math


class GoalPredictor(nn.Module):
    """
    Stage-1 module combining encoders, candidate sampling, and scoring.

    Given a full scene batch, produces K goal predictions per agent.

    Args:
        agent_encoder: AgentHistoryEncoder instance.
        road_encoder:  RoadContextEncoder instance.
        goal_sampler:  GoalCandidateSampler instance.
        scoring_net:   GoalScoringNetwork instance.
        K:             Number of top-K goals to return (default 6).
    """

    def __init__(self, agent_encoder, road_encoder, goal_sampler,
                 scoring_net, K: int = 6):
        super().__init__()
        self.agent_encoder = agent_encoder
        self.road_encoder   = road_encoder
        self.goal_sampler   = goal_sampler
        self.scoring_net    = scoring_net
        self.K = K

    def forward(self,
                agent_history: torch.Tensor,
                road_pts: torch.Tensor,
                road_mask: torch.Tensor,
                current_pos: torch.Tensor,
                current_vel: torch.Tensor,
                roadgraph_xyz: torch.Tensor,
                roadgraph_dir: torch.Tensor,
                roadgraph_valid: torch.Tensor):
        """
        Args:
            agent_history:   [batch, T, feature_dim]  history for ONE agent.
            road_pts:        [batch, N_pts, 6]         xyz+dir road features.
            road_mask:       [batch, N_pts]            validity mask.
            current_pos:     [batch, 2]                current position.
            current_vel:     [batch, 2]                current velocity.
            roadgraph_xyz:   [batch, M, 3]             for candidate sampling.
            roadgraph_dir:   [batch, M, 3]
            roadgraph_valid: [batch, M, 1]

        Returns:
            dict with keys:
                goal_positions    : [batch, K, 2]          relative to current
                goal_confidences  : [batch, K]             softmax scores
                all_logits        : [batch, N_candidates]
                all_candidates    : [batch, N_candidates, 4]
                agent_embedding   : [batch, agent_dim]
                road_embedding    : [batch, road_dim]
        """
        # Encode agent history -> [batch, agent_dim]
        agent_emb = self.agent_encoder(agent_history)

        # Encode road context -> [batch, road_dim]
        road_emb = self.road_encoder(road_pts, mask=road_mask)

        # Sample candidates -> [batch, N, 4]
        candidates = self.goal_sampler.sample_candidates(
            current_pos, current_vel,
            roadgraph_xyz, roadgraph_dir, roadgraph_valid,
        )

        # Score candidates -> [batch, N]
        logits = self.scoring_net(agent_emb, road_emb, candidates)

        # Top-K goals
        topk_logits, topk_idx = torch.topk(logits, k=self.K, dim=-1)  # [B, K]

        # Gather candidate positions for top-K
        # candidates: [B, N, 4]; topk_idx: [B, K]
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, self.K, 4)
        goal_positions = torch.gather(candidates, 1, idx_exp)[..., :2]  # [B, K, 2]

        goal_confidences = F.softmax(topk_logits, dim=-1)  # [B, K]

        return {
            'goal_positions':   goal_positions,
            'goal_confidences': goal_confidences,
            'all_logits':       logits,
            'all_candidates':   candidates,
            'agent_embedding':  agent_emb,
            'road_embedding':   road_emb,
        }
