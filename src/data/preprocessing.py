"""
Preprocessing utilities for GB-Phys TrajNet.

Implements position normalisation, acceleration computation, agent-type
encoding, augmentation, and batch preparation helpers.
"""

import math

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    np = None
    NP_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

from src.data.feature_defs import NUM_AGENTS, NUM_HISTORY_STEPS, num_map_samples

# ---------------------------------------------------------------------------
# Agent type constants (matching WOMD proto)
# ---------------------------------------------------------------------------
TYPE_VEHICLE    = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST    = 3
NUM_AGENT_TYPES = 3  # vehicle, pedestrian, cyclist


def normalize_positions(states, reference_pos):
    """
    Centre positions relative to a reference position (typically agent current pos).

    Works with both numpy arrays and PyTorch tensors.

    Args:
        states:        [..., 2]  x/y positions (last dim must be 2).
        reference_pos: [..., 2]  Reference positions (broadcastable).

    Returns:
        Normalised positions with same shape as `states`.
    """
    return states - reference_pos


def compute_accelerations(velocities, dt=0.1):
    """
    Compute finite-difference accelerations from a velocity sequence.

    Args:
        velocities: [..., T, 2]  vx/vy sequence.
        dt:         Time step in seconds.

    Returns:
        accelerations: [..., T, 2]  ax/ay  (first step = 0).
    """
    if TORCH_AVAILABLE and isinstance(velocities, torch.Tensor):
        # finite diff along time axis
        dv = torch.diff(velocities, dim=-2)          # [..., T-1, 2]
        ax = dv / dt
        # Pad first timestep with zero
        zeros = torch.zeros_like(ax[..., :1, :])
        return torch.cat([zeros, ax], dim=-2)
    else:
        import numpy as np
        dv = np.diff(velocities, axis=-2)
        ax = dv / dt
        zeros = np.zeros_like(ax[..., :1, :])
        return np.concatenate([zeros, ax], axis=-2)


def agent_type_to_onehot(agent_type):
    """
    One-hot encode agent types.

    Args:
        agent_type: [...] integer tensor/array with values {1,2,3} or {0,1,2}.

    Returns:
        one_hot: [..., NUM_AGENT_TYPES]  float32
    """
    if TORCH_AVAILABLE and isinstance(agent_type, torch.Tensor):
        # Shift to 0-indexed if needed
        idx = agent_type.long() - 1
        idx = torch.clamp(idx, 0, NUM_AGENT_TYPES - 1)
        return F.one_hot(idx, num_classes=NUM_AGENT_TYPES).float()
    else:
        import numpy as np
        idx = np.array(agent_type, dtype=np.int64) - 1
        idx = np.clip(idx, 0, NUM_AGENT_TYPES - 1)
        eye = np.eye(NUM_AGENT_TYPES, dtype=np.float32)
        return eye[idx]


def prepare_agent_history(batch):
    """
    Build agent history feature tensor from a raw parsed batch dict.

    Features per timestep per agent:
        x, y, vx, vy, ax, ay, theta, length, width, type_onehot(3)
        => feature_dim = 12

    Args:
        batch: Dict of torch tensors from tf_to_torch_batch.
               Expected keys: input_states [B, 128, 11, 7], object_type [B, 128]

    Returns:
        agent_features: [B, 128, 11, 12]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for prepare_agent_history.")

    input_states = batch['input_states']  # [B, 128, 11, 7]
    object_type  = batch['object_type']  # [B, 128]

    B, A, T, _ = input_states.shape

    # Unpack fields (x,y,l,w,yaw,vx,vy)
    xy    = input_states[..., 0:2]   # [B, A, T, 2]
    l     = input_states[..., 2:3]   # [B, A, T, 1]
    w     = input_states[..., 3:4]
    theta = input_states[..., 4:5]
    vel   = input_states[..., 5:7]   # [B, A, T, 2]  vx, vy

    # Accelerations
    acc = compute_accelerations(vel, dt=0.1)  # [B, A, T, 2]

    # One-hot type  [B, A, 3] -> [B, A, 1, 3] -> broadcast [B, A, T, 3]
    type_oh = agent_type_to_onehot(object_type)             # [B, A, 3]
    type_oh = type_oh.unsqueeze(2).expand(B, A, T, NUM_AGENT_TYPES)

    agent_features = torch.cat([xy, vel, acc, theta, l, w, type_oh], dim=-1)  # [B, A, T, 12]
    return agent_features


def prepare_road_features(batch, max_points=None):
    """
    Build road context feature tensor from parsed batch dict.

    Filters by validity, concatenates xyz + dir_xyz.

    Args:
        batch:      Dict of torch tensors.
                    Keys: roadgraph_xyz [B, 30000, 3], roadgraph_dir [B, 30000, 3],
                          roadgraph_valid [B, 30000, 1]
        max_points: If set, randomly subsample to this many valid points.

    Returns:
        road_features: [B, N_pts, 6] where N_pts <= 30000
        road_mask:     [B, 30000] boolean validity mask (before subsampling)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for prepare_road_features.")

    xyz   = batch['roadgraph_xyz']    # [B, N, 3]
    dirv  = batch['roadgraph_dir']    # [B, N, 3]
    valid = batch['roadgraph_valid']  # [B, N, 1]

    road_mask = (valid[..., 0] > 0)  # [B, N]
    road_features = torch.cat([xyz, dirv], dim=-1)  # [B, N, 6]

    # Zero out invalid points (mask is applied in the encoder via masking)
    road_features = road_features * valid.float()

    return road_features, road_mask


def random_rotation_augmentation(batch, angle_range=(-math.pi, math.pi)):
    """
    Apply random 2-D rotation augmentation to a batch (in-place copy).

    Rotates all positions and velocities/directions. Modifies:
        input_states  (x, y, vx, vy)
        gt_future_states (x, y, vx, vy)
        roadgraph_xyz (x, y)
        roadgraph_dir (x, y)

    Args:
        batch:       Dict of torch tensors (B, ...).
        angle_range: Tuple (min_angle, max_angle) in radians.

    Returns:
        Augmented batch dict (new dict, does NOT modify in-place).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")

    batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    B = batch['input_states'].shape[0]
    device = batch['input_states'].device

    lo, hi = angle_range
    angles = torch.empty(B, device=device).uniform_(lo, hi)  # [B]
    cos_a  = torch.cos(angles)  # [B]
    sin_a  = torch.sin(angles)

    def _rot2d(tensor, x_idx, y_idx):
        """Rotate x/y channels in-place on cloned tensor."""
        x = tensor[..., x_idx].clone()
        y = tensor[..., y_idx].clone()
        # Broadcast: expand cos/sin to match tensor shape
        shape = [B] + [1] * (tensor.dim() - 2)
        c = cos_a.view(*shape)
        s = sin_a.view(*shape)
        tensor[..., x_idx] = x * c - y * s
        tensor[..., y_idx] = x * s + y * c
        return tensor

    # input_states [B, 128, 11, 7]: x=0,y=1; vx=5,vy=6
    batch['input_states'] = _rot2d(batch['input_states'], 0, 1)
    batch['input_states'] = _rot2d(batch['input_states'], 5, 6)

    # gt_future_states [B, 128, 91, 7]: same indices
    batch['gt_future_states'] = _rot2d(batch['gt_future_states'], 0, 1)
    batch['gt_future_states'] = _rot2d(batch['gt_future_states'], 5, 6)

    # roadgraph_xyz [B, 30000, 3]: x=0, y=1
    batch['roadgraph_xyz'] = _rot2d(batch['roadgraph_xyz'], 0, 1)

    # roadgraph_dir [B, 30000, 3]: x=0, y=1
    batch['roadgraph_dir'] = _rot2d(batch['roadgraph_dir'], 0, 1)

    # Also rotate yaw angles (bbox_yaw at index 4)
    batch['input_states'][..., 4]      = batch['input_states'][..., 4] + angles[:, None, None]
    batch['gt_future_states'][..., 4]  = batch['gt_future_states'][..., 4] + angles[:, None, None]

    return batch


def random_translation_augmentation(batch, offset_std=5.0):
    """
    Apply random 2-D translation augmentation to positions in a batch.

    Shifts all x/y positions by the same random offset per scene.

    Args:
        batch:      Dict of torch tensors.
        offset_std: Standard deviation of the Gaussian offset in metres.

    Returns:
        Augmented batch dict.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")

    batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    B = batch['input_states'].shape[0]
    device = batch['input_states'].device

    offsets = torch.randn(B, 2, device=device) * offset_std  # [B, 2]

    def _shift(tensor, x_idx, y_idx):
        shape = [B] + [1] * (tensor.dim() - 2)
        ox = offsets[:, 0].view(*shape)
        oy = offsets[:, 1].view(*shape)
        tensor[..., x_idx] = tensor[..., x_idx] + ox
        tensor[..., y_idx] = tensor[..., y_idx] + oy
        return tensor

    batch['input_states']     = _shift(batch['input_states'], 0, 1)
    batch['gt_future_states'] = _shift(batch['gt_future_states'], 0, 1)
    batch['roadgraph_xyz']    = _shift(batch['roadgraph_xyz'], 0, 1)

    return batch
