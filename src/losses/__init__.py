"""
Losses package for GB-Phys TrajNet.
"""

from src.losses.wta_loss      import WTALoss
from src.losses.goal_loss     import GoalCELoss
from src.losses.physics_loss  import (
    PhysicsRegularizationLoss,
    BoundaryViolationLoss,
    CombinedLoss,
)

__all__ = [
    'WTALoss',
    'GoalCELoss',
    'PhysicsRegularizationLoss',
    'BoundaryViolationLoss',
    'CombinedLoss',
]
