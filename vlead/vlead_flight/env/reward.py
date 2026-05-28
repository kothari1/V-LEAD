"""Composable reward for the goal-conditioned drone navigation task.

Default mix (all weights tunable via RewardConfig):
    r_progress  = -Δdist_to_goal              (positive when closing distance)
    r_success   = +R_goal                     (one-shot bonus on success terminal)
    r_alive     = -c_time                     (per-step cost; encourages finishing)
    r_crash     = -R_crash                    (one-shot penalty on crash terminal)
    r_smooth    = -c_smooth · ||a - a_prev||² (penalize jittery actions)
    r_yaw       = -c_yaw · |heading_error|    (encourage facing the goal)
    r_altitude  = -c_alt · (pz - alt_target)² (fixed-altitude flight)
    r_speedcap  = -c · max(0, ||v|| - v_cap)² (prevent runaway)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class RewardConfig:
    w_progress: float = 1.0
    w_success: float = 50.0
    w_alive: float = 0.01
    w_crash: float = 50.0
    w_smooth: float = 0.05
    w_yaw: float = 0.05
    w_altitude: float = 0.10
    w_speedcap: float = 0.10
    alt_target: float = -1.2          # NED: ~1.2 m above ground
    speed_cap: float = 3.0            # m/s
    success_radius: float = 0.5


class GoalReward:
    """Computes per-step reward. Stateful in the action-smoothness term only."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg
        self._prev_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_action = None

    def __call__(
        self,
        *,
        xcr: np.ndarray,
        prev_dist: float,
        new_dist: float,
        action: np.ndarray,
        goal_heading: np.ndarray,
        terminated: bool,
        term_reason: str,
    ) -> Dict[str, float]:
        cfg = self.cfg
        comp: Dict[str, float] = {}

        comp["progress"] = cfg.w_progress * (prev_dist - new_dist)
        comp["alive"] = -cfg.w_alive

        # Smoothness on the action (skip first step).
        if self._prev_action is not None and cfg.w_smooth > 0:
            diff = np.asarray(action) - self._prev_action
            comp["smooth"] = -cfg.w_smooth * float(np.dot(diff, diff))
        else:
            comp["smooth"] = 0.0
        self._prev_action = np.asarray(action, dtype=np.float64).copy()

        # Yaw alignment: heading error between drone forward axis and goal direction in XY.
        if cfg.w_yaw > 0:
            comp["yaw"] = -cfg.w_yaw * _yaw_alignment_error(xcr, goal_heading)
        else:
            comp["yaw"] = 0.0

        if cfg.w_altitude > 0:
            comp["altitude"] = -cfg.w_altitude * float((xcr[2] - cfg.alt_target) ** 2)
        else:
            comp["altitude"] = 0.0

        if cfg.w_speedcap > 0:
            speed = float(np.linalg.norm(xcr[3:6]))
            comp["speedcap"] = -cfg.w_speedcap * float(max(0.0, speed - cfg.speed_cap) ** 2)
        else:
            comp["speedcap"] = 0.0

        comp["success"] = cfg.w_success if (terminated and term_reason == "success") else 0.0
        comp["crash"] = (
            -cfg.w_crash
            if (terminated and term_reason in ("bbox_violation", "ground_crash",
                                               "ceiling_crash", "overspeed"))
            else 0.0
        )

        comp["total"] = sum(v for v in comp.values())
        return comp


def _yaw_alignment_error(xcr: np.ndarray, goal_heading: np.ndarray) -> float:
    """|wrap(yaw_drone - yaw_to_goal)| in radians, computed in world XY plane."""
    R = Rotation.from_quat(xcr[6:10]).as_matrix()  # scalar-last Hamilton
    fwd = R[:, 0]                                  # body-x in world frame
    fwd_xy = fwd[:2]
    n = float(np.linalg.norm(fwd_xy))
    if n < 1e-6:
        return 0.0
    fwd_xy = fwd_xy / n
    goal_xy = np.asarray(goal_heading[:2], dtype=np.float64)
    g = float(np.linalg.norm(goal_xy))
    if g < 1e-6:
        return 0.0
    goal_xy = goal_xy / g
    cos_err = float(np.clip(np.dot(fwd_xy, goal_xy), -1.0, 1.0))
    return float(np.arccos(cos_err))
