"""
Goal cross-entropy loss for Stage-1 training.

The loss treats the nearest candidate to the ground-truth final position as
the target class, then applies standard cross-entropy over the candidate
scoring logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalCELoss(nn.Module):
    """
    Cross-entropy loss for goal candidate selection.

    The model predicts logits over N_candidates candidate positions.
    We define the "correct" candidate as the one with smallest L2 distance
    to the ground-truth final position, then apply cross-entropy.

    Args:
        label_smoothing: Optional label smoothing factor.
        normalise:       Average over valid agents when True.
    """

    def __init__(self, label_smoothing: float = 0.0, normalise: bool = True):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.normalise = normalise

    def forward(self,
                predicted_logits:   torch.Tensor,
                gt_final_position:  torch.Tensor,
                candidate_positions: torch.Tensor,
                agent_valid_mask:   torch.Tensor = None) -> torch.Tensor:
        """
        Compute the goal cross-entropy loss.

        Args:
            predicted_logits:    [batch, N_candidates]  raw unnormalised scores.
            gt_final_position:   [batch, 2]             ground-truth endpoint (x, y).
            candidate_positions: [batch, N_candidates, 2]  (rel_x, rel_y) of candidates.
            agent_valid_mask:    [batch]  boolean / float; skip invalid agents.

        Returns:
            loss: Scalar tensor.
        """
        B, N = predicted_logits.shape
        device = predicted_logits.device

        # ------------------------------------------------------------------
        # Find nearest candidate to GT endpoint
        # ------------------------------------------------------------------
        # gt_final_position: [B, 2]  -> broadcast to [B, N, 2]
        gt_exp   = gt_final_position.unsqueeze(1).expand(B, N, 2)
        dist_sq  = ((candidate_positions - gt_exp) ** 2).sum(dim=-1)  # [B, N]
        target   = dist_sq.argmin(dim=-1)  # [B]  long

        # ------------------------------------------------------------------
        # Cross-entropy
        # ------------------------------------------------------------------
        per_agent_ce = F.cross_entropy(
            predicted_logits,   # [B, N]
            target,             # [B]
            reduction='none',
            label_smoothing=self.label_smoothing,
        )  # [B]

        # ------------------------------------------------------------------
        # Mask and reduce
        # ------------------------------------------------------------------
        if agent_valid_mask is not None:
            has_gt = agent_valid_mask.float()
        else:
            has_gt = torch.ones(B, device=device)

        if self.normalise:
            n_valid = has_gt.sum().clamp(min=1.0)
            loss = (per_agent_ce * has_gt).sum() / n_valid
        else:
            loss = (per_agent_ce * has_gt).sum()

        return loss

    def forward_with_accuracy(self,
                               predicted_logits:   torch.Tensor,
                               gt_final_position:  torch.Tensor,
                               candidate_positions: torch.Tensor,
                               agent_valid_mask:   torch.Tensor = None,
                               top_k: int = 1) -> tuple:
        """
        Same as forward() but also returns top-k accuracy for diagnostics.

        Returns:
            loss:     Scalar.
            accuracy: Fraction of agents where true candidate is in top-k predictions.
        """
        B, N = predicted_logits.shape

        gt_exp   = gt_final_position.unsqueeze(1).expand(B, N, 2)
        dist_sq  = ((candidate_positions - gt_exp) ** 2).sum(dim=-1)
        target   = dist_sq.argmin(dim=-1)

        per_agent_ce = F.cross_entropy(predicted_logits, target, reduction='none')

        if agent_valid_mask is not None:
            has_gt = agent_valid_mask.float()
        else:
            has_gt = torch.ones(B, device=predicted_logits.device)

        n_valid = has_gt.sum().clamp(min=1.0)
        loss    = (per_agent_ce * has_gt).sum() / n_valid

        # Top-k accuracy
        _, topk_preds = torch.topk(predicted_logits, k=min(top_k, N), dim=-1)
        correct = (topk_preds == target.unsqueeze(1)).any(dim=-1).float()
        accuracy = (correct * has_gt).sum() / n_valid

        return loss, accuracy
