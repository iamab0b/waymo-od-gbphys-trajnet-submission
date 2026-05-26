"""
Winner-Takes-All (WTA) trajectory loss for multi-modal prediction.

During training, only the single closest predicted trajectory to the ground
truth is penalised, encouraging mode diversity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WTALoss(nn.Module):
    """
    Winner-Takes-All L2 trajectory loss.

    For each agent, the predicted trajectory closest (in L2 distance at
    final timestep) to the ground truth is selected as the "winner" and
    receives the regression loss.  All other modes receive zero gradient.

    This encourages the model to produce diverse, plausible hypotheses.

    Args:
        reduce_time:  Reduction strategy over time dimension ('mean' or 'sum').
        normalise:    If True, divide by number of valid agents.
    """

    def __init__(self, reduce_time: str = 'mean', normalise: bool = True):
        super().__init__()
        assert reduce_time in ('mean', 'sum')
        self.reduce_time = reduce_time
        self.normalise   = normalise

    def forward(self,
                predicted_trajectories: torch.Tensor,
                gt_trajectory:          torch.Tensor,
                gt_valid_mask:          torch.Tensor) -> torch.Tensor:
        """
        Compute WTA loss.

        Args:
            predicted_trajectories: [batch, K, 80, 2]
                                    K predicted modes per agent.
            gt_trajectory:          [batch, 80, 2]
                                    Ground-truth future positions.
            gt_valid_mask:          [batch, 80]
                                    Boolean / float mask (1 = valid step).

        Returns:
            loss: Scalar tensor.
        """
        B, K, T, _ = predicted_trajectories.shape
        device = predicted_trajectories.device

        # Cast mask to float
        mask = gt_valid_mask.float()  # [B, T]

        # ----------------------------------------------------------------
        # Winner selection: find mode closest to GT at final valid timestep
        # We use the sum of L2 distances over all valid timesteps to be robust
        # ----------------------------------------------------------------
        gt_exp = gt_trajectory.unsqueeze(1).expand(B, K, T, 2)  # [B, K, T, 2]

        # Per-step squared distances
        sq_dist = ((predicted_trajectories - gt_exp) ** 2).sum(dim=-1)  # [B, K, T]

        # Mask out invalid steps, then sum
        mask_exp = mask.unsqueeze(1).expand(B, K, T)
        sq_dist_masked = sq_dist * mask_exp                              # [B, K, T]
        sum_dist = sq_dist_masked.sum(dim=-1)                            # [B, K]

        # Winner index per sample
        winner_idx = sum_dist.argmin(dim=1)   # [B]  in [0, K-1]

        # Gather winner trajectories  [B, T, 2]
        winner_exp = winner_idx.view(B, 1, 1, 1).expand(B, 1, T, 2)
        winner_traj = predicted_trajectories.gather(1, winner_exp).squeeze(1)  # [B, T, 2]

        # ----------------------------------------------------------------
        # Regression loss on winner only
        # ----------------------------------------------------------------
        l2 = ((winner_traj - gt_trajectory) ** 2).sum(dim=-1)  # [B, T]  L2 per step
        l2_masked = l2 * mask                                    # [B, T]

        if self.reduce_time == 'mean':
            # Average over valid steps per agent, then over batch
            num_valid = mask.sum(dim=-1).clamp(min=1.0)   # [B]
            per_agent_loss = l2_masked.sum(dim=-1) / num_valid  # [B]
        else:
            per_agent_loss = l2_masked.sum(dim=-1)

        # Determine valid agents (those with at least one valid future step)
        has_gt = (mask.sum(dim=-1) > 0).float()   # [B]

        if self.normalise:
            n_valid = has_gt.sum().clamp(min=1.0)
            loss = (per_agent_loss * has_gt).sum() / n_valid
        else:
            loss = (per_agent_loss * has_gt).sum()

        return loss

    def forward_with_indices(self,
                              predicted_trajectories: torch.Tensor,
                              gt_trajectory:          torch.Tensor,
                              gt_valid_mask:          torch.Tensor):
        """
        Same as forward() but also returns winner indices for diagnostics.

        Returns:
            loss:        Scalar.
            winner_idx:  [batch]  int64 winner mode indices.
        """
        B, K, T, _ = predicted_trajectories.shape
        mask = gt_valid_mask.float()

        gt_exp = gt_trajectory.unsqueeze(1).expand(B, K, T, 2)
        sq_dist = ((predicted_trajectories - gt_exp) ** 2).sum(dim=-1)
        mask_exp = mask.unsqueeze(1).expand(B, K, T)
        sum_dist = (sq_dist * mask_exp).sum(dim=-1)
        winner_idx = sum_dist.argmin(dim=1)

        winner_exp  = winner_idx.view(B, 1, 1, 1).expand(B, 1, T, 2)
        winner_traj = predicted_trajectories.gather(1, winner_exp).squeeze(1)
        l2 = ((winner_traj - gt_trajectory) ** 2).sum(dim=-1)
        l2_masked   = l2 * mask

        num_valid = mask.sum(dim=-1).clamp(min=1.0)
        per_agent = l2_masked.sum(dim=-1) / num_valid
        has_gt    = (mask.sum(dim=-1) > 0).float()
        n_valid   = has_gt.sum().clamp(min=1.0)
        loss      = (per_agent * has_gt).sum() / n_valid

        return loss, winner_idx
