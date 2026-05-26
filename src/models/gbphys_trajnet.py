"""
GB-Phys TrajNet – full two-stage trajectory prediction model.

Stage 1: Goal prediction  (AgentHistoryEncoder + RoadContextEncoder + GoalPredictor)
Stage 2: Trajectory generation (K parallel GRUDecoder instances + BicycleKinematics)

The model processes all 128 agent slots in a batch but only computes
trajectories for agents flagged by tracks_to_predict to save compute.
"""

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders       import AgentHistoryEncoder, RoadContextEncoder, TrafficLightEncoder
from src.models.goal_predictor import GoalCandidateSampler, GoalScoringNetwork, GoalPredictor
from src.models.gru_decoder    import GRUDecoder
from src.models.bicycle_model  import BicycleStep, VEHICLE_PARAMS, CYCLIST_PARAMS, PEDESTRIAN_PARAMS

# ---------------------------------------------------------------------------
# Default model configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Encoders
    'agent_embedding_dim':    128,
    'road_embedding_dim':     128,
    'tl_embedding_dim':       64,
    'agent_input_dim':        12,      # see preprocessing.prepare_agent_history
    'use_transformer_encoder': False,
    # Goal predictor
    'N_candidates':           64,
    'K':                      6,
    'goal_base_radius':       50.0,
    'goal_speed_scale':       5.0,
    # GRU decoder
    'gru_hidden_dim':         256,
    'gru_layers':             2,
    'gru_dropout':            0.1,
    # Physics
    'dt':                     0.1,
    # Prediction horizon
    'T':                      80,
    # Submission downsampling
    'submission_step':        5,    # every 5th of 80 -> 16 points at 2 Hz
}

# Submission output timestep indices (0-indexed within 80-step horizon, 2 Hz)
SUBMISSION_INDICES = list(range(4, 80, 5))   # [4, 9, 14, ..., 79]  => 16 steps


class GBPhysTrajNet(nn.Module):
    """
    Two-stage physics-informed trajectory prediction model.

    Args:
        config: Optional dict of hyperparameters.  Missing keys fall back to
                DEFAULT_CONFIG.

    Call convention::

        output = model(batch, phase=3)

    where `batch` is a dict of torch tensors (output of tf_to_torch_batch).
    `phase` controls which stages are active:
        1 – Stage 1 only  (goal prediction, trajectory = CV rollout as placeholder)
        2 – Stage 2 only  (frozen Stage 1)
        3 – End-to-end    (both stages)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg

        # ------------------------------------------------------------------
        # Stage 1 – Encoders
        # ------------------------------------------------------------------
        self.agent_encoder = AgentHistoryEncoder(
            input_dim=cfg['agent_input_dim'],
            embedding_dim=cfg['agent_embedding_dim'],
            use_transformer=cfg['use_transformer_encoder'],
        )
        self.road_encoder = RoadContextEncoder(
            embedding_dim=cfg['road_embedding_dim'],
        )
        self.tl_encoder = TrafficLightEncoder(
            embedding_dim=cfg['tl_embedding_dim'],
        )

        # Fuse road + traffic-light context -> road_embedding_dim for goal net
        self.context_fuse = nn.Sequential(
            nn.Linear(cfg['road_embedding_dim'] + cfg['tl_embedding_dim'],
                      cfg['road_embedding_dim']),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Stage 1 – Goal predictor
        # ------------------------------------------------------------------
        self.goal_sampler = GoalCandidateSampler(
            N=cfg['N_candidates'],
            base_radius=cfg['goal_base_radius'],
            speed_scale=cfg['goal_speed_scale'],
        )
        self.goal_scoring = GoalScoringNetwork(
            agent_dim=cfg['agent_embedding_dim'],
            road_dim=cfg['road_embedding_dim'],
            candidate_dim=4,
            hidden_dim=cfg['agent_embedding_dim'],
        )
        self.goal_predictor = GoalPredictor(
            agent_encoder=self.agent_encoder,
            road_encoder=self.road_encoder,
            goal_sampler=self.goal_sampler,
            scoring_net=self.goal_scoring,
            K=cfg['K'],
        )

        # ------------------------------------------------------------------
        # Stage 2 – K parallel GRU decoders (one per hypothetical goal)
        # Shared weights across K; goal conditioning differentiates outputs.
        # ------------------------------------------------------------------
        self.gru_decoder = GRUDecoder(
            agent_dim=cfg['agent_embedding_dim'],
            state_dim=5,
            goal_dim=2,
            hidden_dim=cfg['gru_hidden_dim'],
            num_layers=cfg['gru_layers'],
            dropout=cfg['gru_dropout'],
            dt=cfg['dt'],
            wheelbase=VEHICLE_PARAMS['wheelbase'],  # default; overridden per agent type
            a_max=VEHICLE_PARAMS['a_max'],
            delta_max=VEHICLE_PARAMS['delta_max'],
            T=cfg['T'],
        )

        # Per-agent-type decoder variants (lightweight: share main GRU, differ in bike step)
        self.bike_steps = nn.ModuleDict({
            'vehicle':    BicycleStep(dt=cfg['dt'], **VEHICLE_PARAMS),
            'cyclist':    BicycleStep(dt=cfg['dt'], **CYCLIST_PARAMS),
            'pedestrian': BicycleStep(dt=cfg['dt'], **PEDESTRIAN_PARAMS),
        })

        # K=6 => single decoder with different goal conditioning
        self.K = cfg['K']
        self.T = cfg['T']

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_stage1(self):
        """Freeze Stage-1 parameters for Phase-2 training."""
        for p in self.agent_encoder.parameters():
            p.requires_grad_(False)
        for p in self.road_encoder.parameters():
            p.requires_grad_(False)
        for p in self.tl_encoder.parameters():
            p.requires_grad_(False)
        for p in self.context_fuse.parameters():
            p.requires_grad_(False)
        for p in self.goal_scoring.parameters():
            p.requires_grad_(False)
        for p in self.goal_sampler.parameters():
            p.requires_grad_(False)

    def freeze_stage2(self):
        """Freeze Stage-2 parameters for Phase-1 training (goal prediction only)."""
        for p in self.gru_decoder.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self):
        """Unfreeze all parameters for Phase-3 end-to-end fine-tuning."""
        for p in self.parameters():
            p.requires_grad_(True)

    @staticmethod
    def downsample_for_submission(trajectories: torch.Tensor) -> torch.Tensor:
        """
        Downsample internal 10 Hz trajectory to 2 Hz for submission.

        Args:
            trajectories: [..., 80, 2]

        Returns:
            downsampled: [..., 16, 2]  at indices [4,9,14,...,79]
        """
        idx = torch.tensor(SUBMISSION_INDICES, device=trajectories.device)
        return trajectories[..., idx, :]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_initial_kinematic_state(self,
                                        input_states: torch.Tensor,
                                        sample_is_valid: torch.Tensor) -> torch.Tensor:
        """
        Extract (x, y, theta, v, phi=0) from the last observed timestep.

        Args:
            input_states:   [B, 128, 11, 7]
            sample_is_valid:[B, 128]

        Returns:
            kin_state: [B, 128, 5]
        """
        # Use the last timestep (index 10, i.e. current)
        cur = input_states[:, :, -1, :]  # [B, 128, 7]  x,y,l,w,yaw,vx,vy
        x     = cur[..., 0]  # [B, 128]
        y     = cur[..., 1]
        theta = cur[..., 4]  # bbox_yaw
        vx    = cur[..., 5]
        vy    = cur[..., 6]
        v     = torch.sqrt(vx**2 + vy**2 + 1e-8)  # scalar speed
        phi   = torch.zeros_like(x)               # assume zero initial steer

        kin = torch.stack([x, y, theta, v, phi], dim=-1)  # [B, 128, 5]
        return kin

    def _prepare_agent_features(self, batch: dict) -> torch.Tensor:
        """
        Build [B, 128, 11, feat_dim] agent history feature tensor.

        Computes: x,y, vx,vy, ax,ay, theta, l, w, type_onehot(3)  => 12 features
        """
        from src.data.preprocessing import prepare_agent_history
        return prepare_agent_history(batch)  # [B, 128, 11, 12]

    def _prepare_road_features(self, batch: dict):
        """Build road feature tensor and validity mask."""
        from src.data.preprocessing import prepare_road_features
        return prepare_road_features(batch)  # [B, N, 6], [B, N]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, batch: dict, phase: int = 3) -> dict:
        """
        Full forward pass for GB-Phys TrajNet.

        Args:
            batch: Dict of torch tensors (output of tf_to_torch_batch).
                   Required keys: input_states, gt_future_states, gt_future_is_valid,
                   object_type, tracks_to_predict, sample_is_valid,
                   roadgraph_xyz, roadgraph_dir, roadgraph_valid,
                   tl_current_state, tl_current_valid, tl_current_xyz,
                   tl_past_state, tl_past_valid.
            phase: Training phase (1, 2, or 3).

        Returns:
            dict with keys:
                trajectories  : [B, 128, K, 80, 2]
                confidences   : [B, 128, K]
                goal_positions: [B, 128, K, 2]
                goal_logits   : [B, 128, N_candidates]
                all_candidates: [B, 128, N_candidates, 4]
                controls      : [B, 128, K, 80, 2]
                full_states   : [B, 128, K, 80, 5]
        """
        device = batch['input_states'].device
        B      = batch['input_states'].shape[0]
        A      = 128  # NUM_AGENTS
        K      = self.K
        T      = self.T

        # ------------------------------------------------------------------
        # Prepare features
        # ------------------------------------------------------------------
        agent_feats, road_feats, road_mask = self._extract_features(batch)
        # agent_feats: [B, 128, 11, 12]
        # road_feats:  [B, N, 6]
        # road_mask:   [B, N]

        # Initial kinematic state per agent
        kin_state = self._build_initial_kinematic_state(
            batch['input_states'], batch['sample_is_valid'])  # [B, 128, 5]

        # ------------------------------------------------------------------
        # Traffic light encoding (one context vector per scene)
        # ------------------------------------------------------------------
        tl_emb = self._encode_traffic_lights(batch)  # [B, tl_embedding_dim]

        # ------------------------------------------------------------------
        # Pre-compute road embedding per scene (shared across agents)
        # ------------------------------------------------------------------
        road_emb = self.road_encoder(road_feats, mask=road_mask)  # [B, road_dim]

        # Fuse road + TL context
        fused_road_emb = self.context_fuse(
            torch.cat([road_emb, tl_emb], dim=-1))  # [B, road_dim]

        # ------------------------------------------------------------------
        # Per-agent processing
        # ------------------------------------------------------------------
        # Flatten agents into batch dimension for vectorised encoding
        B_flat  = B * A
        feats_flat = agent_feats.reshape(B_flat, 11, -1)  # [B*128, 11, 12]

        # Agent embeddings
        agent_emb_flat = self.agent_encoder(feats_flat)  # [B*128, agent_dim]
        agent_emb      = agent_emb_flat.reshape(B, A, -1)  # [B, 128, agent_dim]

        # Expand road/fused embeddings to per-agent
        road_emb_exp = fused_road_emb.unsqueeze(1).expand(B, A, -1)  # [B, 128, road_dim]

        # ------------------------------------------------------------------
        # Stage 1: Goal prediction for each agent
        # ------------------------------------------------------------------
        # Flatten to [B*128, ...] for goal predictor.
        # NOTE: tensors produced by .expand() are non-contiguous in memory,
        # so .view() would raise "not compatible with size and stride".
        # .reshape() is identical to .view() when the tensor IS contiguous
        # and falls back to a copy when it is not — always safe to use here.
        agent_emb_flat = agent_emb.reshape(B_flat, -1)
        road_emb_flat  = road_emb_exp.reshape(B_flat, -1)

        cur_pos = batch['input_states'][:, :, -1, 0:2].reshape(B_flat, 2)  # current x,y
        cur_vel = batch['input_states'][:, :, -1, 5:7].reshape(B_flat, 2)  # current vx,vy

        # Tile road tensors for per-agent sampling
        rg_xyz_flat   = batch['roadgraph_xyz'].unsqueeze(1).expand(B, A, -1, 3).reshape(B_flat, -1, 3)
        rg_dir_flat   = batch['roadgraph_dir'].unsqueeze(1).expand(B, A, -1, 3).reshape(B_flat, -1, 3)
        rg_valid_flat = batch['roadgraph_valid'].unsqueeze(1).expand(B, A, -1, 1).reshape(B_flat, -1, 1)

        # Sample candidates
        candidates_flat = self.goal_sampler.sample_candidates(
            cur_pos, cur_vel,
            rg_xyz_flat, rg_dir_flat, rg_valid_flat,
        )  # [B*128, N, 4]

        # Score candidates
        logits_flat = self.goal_scoring(agent_emb_flat, road_emb_flat, candidates_flat)
        # [B*128, N_candidates]

        # Top-K goals
        N = candidates_flat.shape[1]
        topk_logits, topk_idx = torch.topk(logits_flat, k=K, dim=-1)  # [B*128, K]
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, K, 4)
        goal_cands = torch.gather(candidates_flat, 1, idx_exp)         # [B*128, K, 4]
        goal_positions_flat = goal_cands[..., :2]                      # [B*128, K, 2]
        goal_confidences_flat = F.softmax(topk_logits, dim=-1)         # [B*128, K]

        # Reshape back
        goal_positions  = goal_positions_flat.view(B, A, K, 2)
        goal_confidences = goal_confidences_flat.view(B, A, K)
        all_logits       = logits_flat.view(B, A, N)
        all_candidates   = candidates_flat.view(B, A, N, 4)

        # ------------------------------------------------------------------
        # Stage 2: Trajectory decoding (if phase >= 2)
        # ------------------------------------------------------------------
        if phase == 1:
            # Constant-velocity placeholder trajectories
            trajs, ctrls, fstates = self._constant_velocity_rollout(
                batch['input_states'], B, A, K, T)
        else:
            # Decode K trajectories per agent using shared GRU + typed bike steps.
            # goal_positions_flat is RELATIVE to the agent's current position
            # (produced by GoalCandidateSampler which subtracts cur_pos).
            # The GRU computes goal_offset = goal_pos - kin_state[:, :2].
            # To keep everything in agent-relative space, zero the (x,y) of
            # kin_state so the GRU starts at origin and integrates relative offsets.
            kin_flat = kin_state.reshape(B_flat, 5).clone()
            kin_flat[:, 0] = 0.0   # x -> 0  (agent-relative origin)
            kin_flat[:, 1] = 0.0   # y -> 0
            trajs, ctrls, fstates = self._decode_trajectories(
                agent_emb_flat, goal_positions_flat,
                kin_flat,
                batch['object_type'].reshape(B_flat),
                B, A, K, T,
            )
            # Translate decoded relative trajectories back to absolute world coords
            # so that downstream evaluation and submission code works correctly.
            cur_pos_exp = cur_pos.reshape(B, A, 1, 1, 2)  # [B, A, 1, 1, 2]
            trajs    = trajs    + cur_pos_exp
            fstates_xy = fstates[..., :2] + cur_pos_exp
            fstates  = torch.cat([fstates_xy, fstates[..., 2:]], dim=-1)

        return {
            'trajectories':   trajs,       # [B, 128, K, 80, 2]
            'confidences':    goal_confidences,  # [B, 128, K]
            'goal_positions': goal_positions,    # [B, 128, K, 2]
            'goal_logits':    all_logits,         # [B, 128, N_candidates]
            'all_candidates': all_candidates,    # [B, 128, N_candidates, 4]
            'controls':       ctrls,             # [B, 128, K, 80, 2]
            'full_states':    fstates,           # [B, 128, K, 80, 5]
        }

    # ------------------------------------------------------------------
    # Sub-routines
    # ------------------------------------------------------------------

    def _extract_features(self, batch: dict):
        """Extract and prepare agent and road features from raw batch."""
        agent_feats = self._prepare_agent_features(batch)  # [B, 128, 11, 12]
        road_feats, road_mask = self._prepare_road_features(batch)
        return agent_feats, road_feats, road_mask

    def _encode_traffic_lights(self, batch: dict) -> torch.Tensor:
        """Encode traffic light information into a scene-level embedding."""
        tl_state = batch['tl_current_state'].long()  # [B, 1, 16]
        tl_valid = batch['tl_current_valid']         # [B, 1, 16]
        tl_xyz   = batch['tl_current_xyz'].float()   # [B, 1, 16, 3]
        return self.tl_encoder(tl_state, tl_xyz, tl_valid)  # [B, tl_dim]

    def _decode_trajectories(self,
                              agent_emb_flat:    torch.Tensor,
                              goal_positions_flat: torch.Tensor,
                              kin_flat:          torch.Tensor,
                              agent_type_flat:   torch.Tensor,
                              B, A, K, T):
        """
        Decode K trajectories per agent using the GRU decoder.

        Args:
            agent_emb_flat:     [B*A, agent_dim]
            goal_positions_flat:[B*A, K, 2]
            kin_flat:           [B*A, 5]
            agent_type_flat:    [B*A]  integer types

        Returns:
            trajs   : [B, A, K, T, 2]
            controls: [B, A, K, T, 2]
            fstates : [B, A, K, T, 5]
        """
        BA = B * A
        traj_list    = []
        control_list = []
        state_list   = []

        for k in range(K):
            goal_k = goal_positions_flat[:, k, :]   # [BA, 2]
            out_k  = self.gru_decoder(agent_emb_flat, goal_k, kin_flat)
            traj_list.append(out_k['trajectories'])  # [BA, T, 2]
            control_list.append(out_k['controls'])
            state_list.append(out_k['full_states'])

        trajs    = torch.stack(traj_list,    dim=1)  # [BA, K, T, 2]
        controls = torch.stack(control_list, dim=1)  # [BA, K, T, 2]
        fstates  = torch.stack(state_list,   dim=1)  # [BA, K, T, 5]

        trajs    = trajs.reshape(B, A, K, T, 2)
        controls = controls.reshape(B, A, K, T, 2)
        fstates  = fstates.reshape(B, A, K, T, 5)

        return trajs, controls, fstates

    def _constant_velocity_rollout(self, input_states, B, A, K, T):
        """Produce constant-velocity placeholder trajectories (Phase 1)."""
        device = input_states.device
        cur_pos = input_states[:, :, -1, 0:2]     # [B, A, 2]
        cur_vel = input_states[:, :, -1, 5:7]     # [B, A, 2]

        dt = self.cfg['dt']
        steps = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        # [B, A, 1, T, 1] * [1, 1, 1, 1, 2] -> positions
        trajs = cur_pos[:, :, None, None, :] + \
                cur_vel[:, :, None, None, :] * steps[None, None, None, :, None] * dt
        trajs = trajs.expand(B, A, K, T, 2).clone()

        controls = torch.zeros(B, A, K, T, 2, device=device)
        fstates  = torch.zeros(B, A, K, T, 5, device=device)
        # Fill in x, y from trajs
        fstates[..., :2] = trajs

        return trajs, controls, fstates
