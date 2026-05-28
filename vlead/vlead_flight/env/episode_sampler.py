"""Start state and goal sampling for FigsDroneEnv episodes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class EpisodeSpec:
    """One sampled episode: where the drone starts, where it must go."""

    x0: np.ndarray            # (10,) full FiGS state
    target_xyz: np.ndarray    # (3,)


@dataclass
class EpisodeSampler:
    """Uniform-random start pose + goal in configurable boxes.

    World frame is NED (z down). Position z is typically negative when
    above the ground. Defaults sample a small box near the origin with the
    drone level and stationary.

    Curriculum hook: shrink/expand goal_radius_max as success rate climbs.
    """

    start_xyz_low: Sequence[float] = (-1.0, -1.0, -1.5)
    start_xyz_high: Sequence[float] = (1.0, 1.0, -1.0)
    goal_radius_min: float = 1.5
    goal_radius_max: float = 4.0
    goal_z_low: float = -2.0
    goal_z_high: float = -0.8
    yaw_jitter: float = 0.0   # radians; 0 disables
    velocity_jitter: float = 0.0
    bbox_xyz_low: Optional[Sequence[float]] = None
    bbox_xyz_high: Optional[Sequence[float]] = None
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def seed(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed)

    def _in_bbox(self, xyz: np.ndarray) -> bool:
        if self.bbox_xyz_low is None or self.bbox_xyz_high is None:
            return True
        lo = np.asarray(self.bbox_xyz_low)
        hi = np.asarray(self.bbox_xyz_high)
        return bool(np.all(xyz >= lo) and np.all(xyz <= hi))

    def sample(self) -> EpisodeSpec:
        rng = self._rng

        # Start position
        start_xyz = rng.uniform(
            low=np.asarray(self.start_xyz_low, dtype=np.float64),
            high=np.asarray(self.start_xyz_high, dtype=np.float64),
        )

        # Goal: sample a direction in the XY plane + a Z within configured range,
        # then scale to a radius drawn from [goal_radius_min, goal_radius_max].
        # Retry until the goal is inside the optional bbox.
        for _ in range(50):
            theta = rng.uniform(0.0, 2 * np.pi)
            radius_xy = rng.uniform(self.goal_radius_min, self.goal_radius_max)
            goal_xy = start_xyz[0:2] + radius_xy * np.array([np.cos(theta), np.sin(theta)])
            goal_z = rng.uniform(self.goal_z_low, self.goal_z_high)
            target = np.array([goal_xy[0], goal_xy[1], goal_z], dtype=np.float64)
            if self._in_bbox(target):
                break

        # Initial attitude: identity quaternion + optional yaw jitter
        yaw = rng.uniform(-self.yaw_jitter, self.yaw_jitter) if self.yaw_jitter > 0 else 0.0
        qx, qy, qz, qw = 0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)

        # Initial velocity: small jitter or zero
        if self.velocity_jitter > 0:
            v0 = rng.uniform(-self.velocity_jitter, self.velocity_jitter, size=3)
        else:
            v0 = np.zeros(3)

        x0 = np.array(
            [start_xyz[0], start_xyz[1], start_xyz[2],
             v0[0], v0[1], v0[2],
             qx, qy, qz, qw],
            dtype=np.float64,
        )
        return EpisodeSpec(x0=x0, target_xyz=target)
