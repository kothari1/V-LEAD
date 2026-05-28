"""Termination / truncation predicates for FigsDroneEnv."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class TerminationConfig:
    success_radius: float = 0.5            # m; goal reached when dist < radius
    max_episode_steps: int = 300           # truncation (15 s @ 20 Hz)
    bbox_xyz_low: Optional[Sequence[float]] = None
    bbox_xyz_high: Optional[Sequence[float]] = None
    ground_z: Optional[float] = None       # crash if pz > ground_z (NED: z increases downward)
    ceiling_z: Optional[float] = None      # crash if pz < ceiling_z
    speed_kill: Optional[float] = None     # crash if ||v|| exceeds this


def check_termination(
    xcr: np.ndarray,
    dist_to_goal: float,
    step_idx: int,
    cfg: TerminationConfig,
) -> Tuple[bool, bool, str]:
    """Returns (terminated, truncated, reason).

    `terminated` = episode ended for an in-task reason (success/crash).
    `truncated` = episode ended because of step budget (Gymnasium semantics).
    """
    if dist_to_goal < cfg.success_radius:
        return True, False, "success"

    pos = xcr[0:3]
    vel = xcr[3:6]

    if cfg.bbox_xyz_low is not None and cfg.bbox_xyz_high is not None:
        lo = np.asarray(cfg.bbox_xyz_low)
        hi = np.asarray(cfg.bbox_xyz_high)
        if np.any(pos < lo) or np.any(pos > hi):
            return True, False, "bbox_violation"

    if cfg.ground_z is not None and pos[2] > cfg.ground_z:
        return True, False, "ground_crash"
    if cfg.ceiling_z is not None and pos[2] < cfg.ceiling_z:
        return True, False, "ceiling_crash"

    if cfg.speed_kill is not None and float(np.linalg.norm(vel)) > cfg.speed_kill:
        return True, False, "overspeed"

    if step_idx + 1 >= cfg.max_episode_steps:
        return False, True, "timeout"

    return False, False, ""
