"""
Differentiable bicycle kinematic model for GB-Phys TrajNet.

Implements Euler-integrated bicycle equations with agent-type-specific
parameters.  No learnable parameters – purely physics-based rollout.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Per-agent-type physical parameters
# ---------------------------------------------------------------------------
VEHICLE_PARAMS    = dict(wheelbase=2.7,  a_max=5.0, delta_max=0.5)
CYCLIST_PARAMS    = dict(wheelbase=1.0,  a_max=3.0, delta_max=1.0)
PEDESTRIAN_PARAMS = dict(wheelbase=0.5,  a_max=2.0, delta_max=2.0)

# WOMD agent type codes
TYPE_VEHICLE    = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST    = 3


class BicycleStep(nn.Module):
    """
    Single Euler-integration step of the bicycle kinematic model.

    State vector: [x, y, theta, v, phi]
      x, y   : position (m)
      theta  : heading angle (rad)
      v      : longitudinal speed (m/s)
      phi    : front-wheel steering angle (rad)

    Controls: [a_t, delta_t]
      a_t     : longitudinal acceleration (m/s^2)
      delta_t : steering-angle rate (rad/s)

    Euler integration:
      x     += v * cos(theta) * dt
      y     += v * sin(theta) * dt
      theta += (v / L) * tan(phi) * dt
      v     += a_t * dt
      phi   += delta_t * dt

    Args:
        dt:          Integration timestep (seconds).
        wheelbase:   Vehicle wheelbase L (metres).
        a_max:       Maximum acceleration magnitude (m/s^2).
        delta_max:   Maximum steering-rate magnitude (rad/s).
    """

    def __init__(self, dt: float = 0.1, wheelbase: float = 2.7,
                 a_max: float = 5.0, delta_max: float = 0.5):
        super().__init__()
        self.dt        = dt
        self.L         = wheelbase
        self.a_max     = a_max
        self.delta_max = delta_max

    def forward(self, state: torch.Tensor,
                control: torch.Tensor) -> torch.Tensor:
        """
        Advance state by one timestep.

        Args:
            state:   [..., 5]  (x, y, theta, v, phi)
            control: [..., 2]  (a_t, delta_t) – raw network outputs (will be
                     scaled via tanh clamping inside this function)

        Returns:
            new_state: [..., 5]
        """
        # Scale controls
        a_t     = control[..., 0] * self.a_max     # scaled accel
        delta_t = control[..., 1] * self.delta_max  # scaled steer rate

        x     = state[..., 0]
        y     = state[..., 1]
        theta = state[..., 2]
        v     = state[..., 3]
        phi   = state[..., 4]

        dt = self.dt
        L  = self.L

        # Euler update
        x_new     = x + v * torch.cos(theta) * dt
        y_new     = y + v * torch.sin(theta) * dt
        theta_new = theta + (v / (L + 1e-6)) * torch.tan(phi) * dt
        v_new     = v + a_t * dt
        phi_new   = phi + delta_t * dt

        # Clamp physical limits
        v_new   = torch.clamp(v_new,   -30.0, 30.0)
        phi_new = torch.clamp(phi_new, -0.7,   0.7)

        return torch.stack([x_new, y_new, theta_new, v_new, phi_new], dim=-1)


class BicycleKinematics(nn.Module):
    """
    Full bicycle kinematic rollout over T timesteps.

    Wraps BicycleStep for multi-step integration.  No learnable parameters.

    Args:
        dt:        Integration timestep (seconds).
        wheelbase: Vehicle wheelbase (metres).
        a_max:     Maximum acceleration (m/s^2).
        delta_max: Maximum steering rate (rad/s).
    """

    def __init__(self, dt: float = 0.1, wheelbase: float = 2.7,
                 a_max: float = 5.0, delta_max: float = 0.5):
        super().__init__()
        self.step = BicycleStep(dt=dt, wheelbase=wheelbase,
                                a_max=a_max, delta_max=delta_max)

    def forward(self, initial_state: torch.Tensor,
                controls: torch.Tensor) -> torch.Tensor:
        """
        Roll out T steps from an initial state.

        Args:
            initial_state: [batch, 5]       (x, y, theta, v, phi)
            controls:      [batch, T, 2]    (a_t, delta_t) per step

        Returns:
            states: [batch, T, 5]   states AFTER each control application
        """
        T = controls.shape[1]
        state = initial_state
        state_seq = []
        for t in range(T):
            state = self.step(state, controls[:, t, :])
            state_seq.append(state)
        return torch.stack(state_seq, dim=1)   # [batch, T, 5]


class AgentTypeKinematics(nn.Module):
    """
    Selector that returns a bicycle model configured for a given agent type.

    Supports batched, per-agent agent types by blending outputs from the
    three typed models via a soft mask (differentiable).

    Args:
        dt: Integration timestep (seconds).
    """

    def __init__(self, dt: float = 0.1):
        super().__init__()
        self.vehicle_model    = BicycleKinematics(dt=dt, **VEHICLE_PARAMS)
        self.cyclist_model    = BicycleKinematics(dt=dt, **CYCLIST_PARAMS)
        self.pedestrian_model = BicycleKinematics(dt=dt, **PEDESTRIAN_PARAMS)
        self.dt = dt

    @staticmethod
    def get_model_for_type(agent_type: int, dt: float = 0.1) -> BicycleKinematics:
        """
        Return a standalone BicycleKinematics for the given integer agent type.

        Args:
            agent_type: 1 = vehicle, 2 = pedestrian, 3 = cyclist.
            dt:         Integration timestep.
        """
        if agent_type == TYPE_VEHICLE:
            return BicycleKinematics(dt=dt, **VEHICLE_PARAMS)
        elif agent_type == TYPE_CYCLIST:
            return BicycleKinematics(dt=dt, **CYCLIST_PARAMS)
        else:
            return BicycleKinematics(dt=dt, **PEDESTRIAN_PARAMS)

    def forward(self, initial_state: torch.Tensor,
                controls: torch.Tensor,
                agent_type: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectories for a batch with mixed agent types.

        Runs all three typed models and blends results using agent type masks
        so the operation is differentiable with respect to controls.

        Args:
            initial_state: [batch, 5]
            controls:      [batch, T, 2]
            agent_type:    [batch]  integer type codes (1/2/3)

        Returns:
            states: [batch, T, 5]
        """
        sv = self.vehicle_model(initial_state, controls)     # [B, T, 5]
        sc = self.cyclist_model(initial_state, controls)     # [B, T, 5]
        sp = self.pedestrian_model(initial_state, controls)  # [B, T, 5]

        is_vehicle = (agent_type == TYPE_VEHICLE).float().view(-1, 1, 1)    # [B,1,1]
        is_cyclist = (agent_type == TYPE_CYCLIST).float().view(-1, 1, 1)
        is_ped     = (agent_type == TYPE_PEDESTRIAN).float().view(-1, 1, 1)

        # Fallback: treat unknown types as pedestrian
        total = is_vehicle + is_cyclist + is_ped
        total = torch.clamp(total, min=1.0)
        is_ped = is_ped + (1.0 - (is_vehicle + is_cyclist + is_ped).clamp(max=1.0))

        states = is_vehicle * sv + is_cyclist * sc + is_ped * sp
        return states
