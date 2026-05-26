"""
Physics-based regularisation and boundary violation losses.

These losses penalise trajectories that violate physical plausibility:
  - PhysicsRegularizationLoss : jerk, steering rate, lateral acceleration
  - BoundaryViolationLoss     : distance from road boundary
  - CombinedLoss              : weighted sum of all loss components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsRegularizationLoss(nn.Module):
    """
    Penalise physically implausible control sequences.

    Three components:
      1. Jerk            = rate of change of acceleration (|da/dt|)
      2. Steering rate   = |ddelta/dt|  (already the control input, so
                           we penalise |delta_t| directly too)
      3. Lateral accel   = v^2 * |tan(phi)| / L  (centripetal acceleration)

    Args:
        L:            Bicycle model wheelbase (metres).
        w_jerk:       Weight for jerk penalty.
        w_steer:      Weight for steering rate penalty.
        w_lat:        Weight for lateral acceleration penalty.
        dt:           Time step for finite differences.
    """

    def __init__(self, L: float = 2.7, w_jerk: float = 0.1,
                 w_steer: float = 0.1, w_lat: float = 0.1, dt: float = 0.1):
        super().__init__()
        self.L = L
        self.w_jerk  = w_jerk
        self.w_steer = w_steer
        self.w_lat   = w_lat
        self.dt      = dt

    def forward(self, controls: torch.Tensor,
                full_states: torch.Tensor) -> torch.Tensor:
        """
        Compute physics regularisation loss.

        Args:
            controls:    [batch, T, 2]   (a_t, delta_t) control signals.
            full_states: [batch, T, 5]   (x, y, theta, v, phi) states.

        Returns:
            loss: Scalar tensor.
        """
        a_t     = controls[..., 0]       # [B, T]  accel
        delta_t = controls[..., 1]       # [B, T]  steering rate

        v   = full_states[..., 3]        # [B, T]  speed
        phi = full_states[..., 4]        # [B, T]  steering angle

        # 1) Jerk: finite diff of acceleration over time
        jerk = torch.diff(a_t, dim=-1)              # [B, T-1]
        jerk_loss = (jerk ** 2).mean()

        # 2) Steering rate: penalise large |delta_t|
        steer_loss = (delta_t ** 2).mean()

        # 3) Lateral acceleration: a_lat = v^2 * |tan(phi)| / L
        # Clamp tan(phi) to avoid explosion near phi=±π/2 with random init.
        a_lat = (v ** 2) * torch.abs(torch.tan(phi).clamp(-5.0, 5.0)) / (self.L + 1e-6)
        # Clamp the final value too — early training can produce implausible states
        a_lat = a_lat.clamp(max=50.0)
        lat_loss = (a_lat ** 2).mean()

        return self.w_jerk * jerk_loss + self.w_steer * steer_loss + self.w_lat * lat_loss


class BoundaryViolationLoss(nn.Module):
    """
    Penalise trajectory points far from the nearest road centerline sample.

    For each predicted position, find the minimum distance to any road
    sample point and penalise when that distance exceeds a threshold.

    Args:
        threshold:   Distance threshold (metres) beyond which violation begins.
        normalise:   Average over agents when True.
    """

    def __init__(self, threshold: float = 5.0, normalise: bool = True,
                 max_road_pts: int = 512):
        super().__init__()
        self.threshold    = threshold
        self.normalise    = normalise
        # Maximum road points to use per agent.  The full 30,000-point roadgraph
        # creates a [B*A, T, N] = [4096, 80, 30000] tensor (~78 GB) — far beyond
        # any GPU.  Subsampling to 512 keeps peak allocation under 300 MB while
        # still providing a meaningful road boundary signal.
        self.max_road_pts = max_road_pts

    def forward(self, trajectory: torch.Tensor,
                road_points: torch.Tensor,
                road_valid: torch.Tensor = None) -> torch.Tensor:
        """
        Compute boundary violation loss.

        Args:
            trajectory:  [batch, T, 2]          predicted (x, y) positions.
            road_points: [batch, N_pts, 2]       road sample positions (x, y).
            road_valid:  [batch, N_pts]          validity mask (optional).

        Returns:
            loss: Scalar tensor.
        """
        B, T, _ = trajectory.shape
        N = road_points.shape[1]

        # Subsample road points to avoid GPU OOM.
        # Peak memory for pairwise distance = B * T * N_sub * 2 * 4 bytes.
        # With N_sub=512, B=4096, T=80: ~2.7 GB — safe on a 40 GB A100.
        N_sub = min(N, self.max_road_pts)
        if N_sub < N:
            if road_valid is not None:
                # Prefer valid points in the subsample
                valid_mask = road_valid[0] > 0   # use first batch item as proxy
                valid_idx  = torch.where(valid_mask)[0]
                if len(valid_idx) >= N_sub:
                    perm = valid_idx[torch.randperm(len(valid_idx),
                                                    device=trajectory.device)[:N_sub]]
                else:
                    perm = torch.randperm(N, device=trajectory.device)[:N_sub]
            else:
                perm = torch.randperm(N, device=trajectory.device)[:N_sub]
            road_points = road_points[:, perm, :]
            if road_valid is not None:
                road_valid = road_valid[:, perm]

        # Pairwise distance: [B, T, 1, 2] vs [B, 1, N_sub, 2] → [B, T, N_sub]
        traj_exp = trajectory.unsqueeze(2)    # [B, T, 1,     2]
        road_exp = road_points.unsqueeze(1)   # [B, 1, N_sub, 2]
        dist     = torch.norm(traj_exp - road_exp, dim=-1)  # [B, T, N_sub]

        if road_valid is not None:
            valid_exp = road_valid.unsqueeze(1).float()  # [B, 1, N_sub]
            dist = dist + (1.0 - valid_exp) * 1e6

        # Min distance to any road point
        min_dist, _ = dist.min(dim=-1)   # [B, T]

        # Soft hinge: penalise when min_dist > threshold
        violation = F.relu(min_dist - self.threshold)   # [B, T]
        loss = (violation ** 2).mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Weighted combination of all GB-Phys TrajNet loss components.

    L_total = lambda_goal     * L_goal
            + lambda_wta      * L_wta
            + lambda_physics  * L_physics
            + lambda_boundary * L_boundary

    Args:
        lambda_goal:     Weight for goal cross-entropy loss.
        lambda_wta:      Weight for WTA regression loss.
        lambda_physics:  Weight for physics regularisation.
        lambda_boundary: Weight for boundary violation loss.
    """

    def __init__(self,
                 lambda_goal:     float = 1.0,
                 lambda_wta:      float = 1.0,
                 lambda_physics:  float = 0.1,
                 lambda_boundary: float = 0.5):
        super().__init__()
        self.lambda_goal     = lambda_goal
        self.lambda_wta      = lambda_wta
        self.lambda_physics  = lambda_physics
        self.lambda_boundary = lambda_boundary

        # Import individual loss modules here to avoid circular imports
        from src.losses.wta_loss  import WTALoss
        from src.losses.goal_loss import GoalCELoss

        self.wta_loss     = WTALoss()
        self.goal_ce_loss = GoalCELoss()
        self.phys_loss    = PhysicsRegularizationLoss()
        self.bound_loss   = BoundaryViolationLoss()

    def forward(self, model_output: dict, batch: dict,
                phase: int = 3) -> dict:
        """
        Compute all loss components.

        Args:
            model_output: Dict returned by GBPhysTrajNet.forward().
                Required keys: trajectories, confidences, goal_logits,
                all_candidates, controls, full_states.
            batch: Dict of ground-truth tensors.
                Required keys: gt_future_states [B, 128, 91, 7],
                gt_future_is_valid [B, 128, 91], tracks_to_predict [B, 128],
                roadgraph_xyz [B, N, 3].
            phase: Training phase (1, 2, or 3) controls which losses apply.

        Returns:
            loss_dict: {'total', 'goal', 'wta', 'physics', 'boundary'}
        """
        device = model_output['trajectories'].device
        B, A, K, T, _ = model_output['trajectories'].shape

        # ------------------------------------------------------------------
        # Prepare ground-truth
        # GT future positions: indices 11..90 of gt_future_states (x,y)
        # gt_future_states: [B, A, 91, 7]; future is indices 11..90
        # ------------------------------------------------------------------
        gt_full   = batch['gt_future_states']         # [B, A, 91, 7]
        gt_future = gt_full[:, :, 11:, 0:2]           # [B, A, 80, 2]  future x,y
        gt_valid  = batch['gt_future_is_valid'][:, :, 11:].float()  # [B, A, 80]

        # tracks_to_predict mask
        ttp = batch['tracks_to_predict'].float()       # [B, A]

        # ------------------------------------------------------------------
        # Loss terms
        # ------------------------------------------------------------------
        total_loss    = torch.tensor(0.0, device=device)
        goal_loss_val = torch.tensor(0.0, device=device)
        wta_loss_val  = torch.tensor(0.0, device=device)
        phys_loss_val = torch.tensor(0.0, device=device)
        bound_loss_val = torch.tensor(0.0, device=device)

        # ---- Goal CE loss (Phase 1 and 3) --------------------------------
        if phase in (1, 3) and self.lambda_goal > 0:
            goal_logits    = model_output['goal_logits']    # [B, A, N]
            all_candidates = model_output['all_candidates'] # [B, A, N, 4]

            # Flatten agents
            gl_flat  = goal_logits.view(B * A, -1)
            cand_pos = all_candidates[..., :2].view(B * A, -1, 2)

            # GT endpoint in AGENT-RELATIVE coordinates.
            # gt_future is in absolute world coords; candidates are relative.
            # Subtract each agent's current position before nearest-candidate lookup.
            cur_pos_flat = batch['input_states'][:, :, -1, 0:2].reshape(B * A, 2)
            gt_end_abs   = self._get_last_valid_pos(gt_future.view(B*A, T, 2),
                                                    gt_valid.view(B*A, T))
            gt_end_rel   = gt_end_abs - cur_pos_flat   # [B*A, 2] relative

            valid_flat = ttp.view(B * A)
            goal_loss_val = self.goal_ce_loss(
                gl_flat, gt_end_rel, cand_pos, agent_valid_mask=valid_flat)
            total_loss = total_loss + self.lambda_goal * goal_loss_val

        # ---- WTA trajectory loss (Phase 2 and 3) -------------------------
        if phase in (2, 3) and self.lambda_wta > 0:
            pred_trajs = model_output['trajectories']  # [B, A, K, T, 2]

            # ----------------------------------------------------------------
            # Compute WTA loss ONLY for tracks_to_predict agents, centred on
            # each agent's own current position.
            #
            # Why filter to TTP agents:
            #   The batch contains 128 agent slots but only 2-8 are actually
            #   tracked.  Invalid/background agents have gt_future_states = 0
            #   while their cur_pos is non-zero (e.g. 11767m from origin).
            #   Subtracting cur_pos from zero GT gives gt_rel = -11767m.
            #   Even with the gt_valid mask, these outliers inflate the loss to
            #   ~30M and swamp the gradient signal from real agents.
            #
            # Why use relative coordinates:
            #   WOMD absolute positions range over ±12 km.  Random decoder
            #   outputs near (0,0) versus GT at (11767, …) give absurd L2
            #   values and tiny relative gradients after clipping.
            #   Centering on cur_pos reduces the scale to ±200m, making the
            #   loss O(100-1000) and gradients informative from step 1.
            # ----------------------------------------------------------------
            cur_pos  = batch['input_states'][:, :, -1, 0:2]  # [B, A, 2]
            ttp_bool = batch['tracks_to_predict'].bool()       # [B, A]
            ttp_flat = ttp_bool.reshape(B * A)                 # [B*A]

            if ttp_flat.any():
                # Centre all agents, then select only TTP rows
                cur_k = cur_pos.unsqueeze(2).unsqueeze(2)  # [B, A, 1, 1, 2]
                cur_t = cur_pos.unsqueeze(2)               # [B, A, 1,    2]

                pred_rel_all = pred_trajs - cur_k          # [B, A, K, T, 2]
                gt_rel_all   = gt_future  - cur_t          # [B, A,    T, 2]

                # Flatten and filter to TTP agents only
                pred_flat = pred_rel_all.reshape(B*A, K, T, 2)[ttp_flat]  # [n, K, T, 2]
                gt_flat   = gt_rel_all.reshape(B*A, T, 2)[ttp_flat]       # [n,    T, 2]
                gv_flat   = gt_valid.reshape(B*A, T)[ttp_flat]             # [n,       T]

                wta_loss_val = self.wta_loss(pred_flat, gt_flat, gv_flat)
                total_loss   = total_loss + self.lambda_wta * wta_loss_val

        # ---- Physics regularisation (Phase 2 and 3) ----------------------
        if phase in (2, 3) and self.lambda_physics > 0:
            controls    = model_output['controls']     # [B, A, K, T, 2]
            full_states = model_output['full_states']  # [B, A, K, T, 5]

            # Average over K modes, flatten B*A
            c_flat = controls.view(B * A * K, T, 2)
            s_flat = full_states.view(B * A * K, T, 5)

            phys_loss_val = self.phys_loss(c_flat, s_flat)
            total_loss    = total_loss + self.lambda_physics * phys_loss_val

        # ---- Boundary violation (Phase 2 and 3) --------------------------
        if phase in (2, 3) and self.lambda_boundary > 0:
            pred_trajs = model_output['trajectories']  # [B, A, K, T, 2]
            road_xyz   = batch['roadgraph_xyz']         # [B, N, 3]
            road_valid = batch['roadgraph_valid'][..., 0].float()  # [B, N]
            road_xy    = road_xyz[..., :2]              # [B, N, 2]
            cur_pos    = batch['input_states'][:, :, -1, 0:2]  # [B, A, 2]

            # Use agent-relative coordinates (matches WTA normalisation above)
            ttp_mask = batch.get('tracks_to_predict')   # [B, A] bool
            if ttp_mask is not None:
                ttp_indices = [torch.where(ttp_mask[b])[0][:8] for b in range(B)]
                bound_losses = []
                for b in range(B):
                    idx = ttp_indices[b]
                    if len(idx) == 0:
                        continue
                    agent_pos = cur_pos[b, idx, :]          # [n_ttp, 2]
                    # Relative predicted trajectory
                    pred_b = pred_trajs[b, idx, 0, :, :] \
                             - agent_pos.unsqueeze(1)        # [n_ttp, T, 2]
                    # Relative road points
                    road_b = (road_xy[b].unsqueeze(0).expand(len(idx), -1, -1)
                              - agent_pos.unsqueeze(1))      # [n_ttp, N, 2]
                    rv_b   = road_valid[b].unsqueeze(0).expand(len(idx), -1)
                    bound_losses.append(self.bound_loss(pred_b, road_b, road_valid=rv_b))
                if bound_losses:
                    bound_loss_val = torch.stack(bound_losses).mean()
                    total_loss     = total_loss + self.lambda_boundary * bound_loss_val
            else:
                # Fallback: first 4 agents, relative coordinates
                agent_pos = cur_pos[:, :4, :].reshape(B * 4, 1, 2)
                pred_b = (pred_trajs[:, :4, 0, :, :].reshape(B * 4, T, 2)
                          - agent_pos)
                road_b = (road_xy.unsqueeze(1).expand(B, 4, -1, 2).reshape(B*4, -1, 2)
                          - agent_pos)
                rv_b   = road_valid.unsqueeze(1).expand(B, 4, -1).reshape(B*4, -1)
                bound_loss_val = self.bound_loss(pred_b, road_b, road_valid=rv_b)
                total_loss     = total_loss + self.lambda_boundary * bound_loss_val

        return {
            'total':    total_loss,
            'goal':     goal_loss_val,
            'wta':      wta_loss_val,
            'physics':  phys_loss_val,
            'boundary': bound_loss_val,
        }

    @staticmethod
    def _get_last_valid_pos(positions: torch.Tensor,
                            valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Get the last valid position for each trajectory.

        Args:
            positions:  [B, T, 2]
            valid_mask: [B, T]  float (1 = valid)

        Returns:
            last_pos: [B, 2]
        """
        B, T, _ = positions.shape
        device = positions.device

        # Find last valid index
        # valid_mask: [B, T]; multiply by arange to get index
        indices = torch.arange(T, device=device, dtype=torch.float32)
        # [B, T] * [T] -> max index of valid step
        masked_idx = valid_mask * indices.unsqueeze(0)    # [B, T]
        last_idx   = masked_idx.argmax(dim=-1).long()     # [B]

        # If no valid steps, use last step
        has_valid = (valid_mask.sum(dim=-1) > 0)
        last_idx  = torch.where(has_valid, last_idx,
                                torch.full_like(last_idx, T - 1))

        # Gather
        last_pos = positions[torch.arange(B, device=device), last_idx, :]  # [B, 2]
        return last_pos
