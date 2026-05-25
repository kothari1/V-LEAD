"""V-LEAD network interface contract + a dummy implementation for smoke tests."""
from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class VLeadNetworkProtocol(Protocol):
    """Contract any trained V-LEAD network must satisfy.

    Forward inputs:
        rgb:           [B, T, 3, H, W]  normalized float (ImageNet stats)
        depth:         [B, T, 1, H, W]  metric depth (raw float), or None
        goal_heading:  [B, 3]           unit vector in world frame
        goal_distance: [B, 1]           positive scalar (meters)

    Forward output:
        [B, H, 4] receding-horizon velocity commands [vx, vy, vz, psi_dot].
        H defaults to 10 but is read from output shape at runtime.
    """

    def forward(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor],
        goal_heading: torch.Tensor,
        goal_distance: torch.Tensor,
    ) -> torch.Tensor: ...


class DummyVLeadNet(nn.Module):
    """Always returns zero velocity commands (hover).

    Used for smoke-testing the deployment pipeline before any real network
    has been trained. Useful sanity check that wiring (preprocessing,
    inner-loop controller, recorder, eval) is correct end-to-end.
    """

    def __init__(self, horizon: int = 10):
        super().__init__()
        self.horizon = horizon
        # Register a parameter so .to(device) / .to(dtype) propagate.
        self.register_buffer("_zero", torch.zeros(1))

    def forward(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor],
        goal_heading: torch.Tensor,
        goal_distance: torch.Tensor,
    ) -> torch.Tensor:
        B = rgb.shape[0]
        return torch.zeros(
            B, self.horizon, 4,
            device=rgb.device, dtype=rgb.dtype,
        )
