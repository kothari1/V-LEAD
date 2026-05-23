"""Per-step rollout recorder for offline analysis, DAgger labels, RL replay."""
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class RolloutRecorder:
    """Buffer per-step (observation, action) tuples during a deployment rollout.

    Schema per step (all numpy except scalars):
        t              float           sim time (s)
        x              (10,)           drone state
        rgb            (H, W, 3) u8    raw RGB at native resolution (None if not stored)
        depth          (H, W) f32      raw metric depth (None if depth disabled)
        goal_heading   (3,) f64        unit vector world frame
        goal_dist      float           meters
        vel_pred       (H, 4) f32      full receding-horizon network output
        u_cmd          (4,) f32        body-rate command sent to ACADOS
        expert_vel     (4,) or None    DAgger label, set by caller when applicable
    """

    def __init__(self):
        self.steps: list = []

    def add(
        self,
        tcr: float,
        xcr: np.ndarray,
        rgb: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        goal_heading: np.ndarray,
        goal_dist: float,
        vel_pred: np.ndarray,
        body_rate_cmd: np.ndarray,
        expert_vel: Optional[np.ndarray] = None,
    ):
        self.steps.append({
            "t": float(tcr),
            "x": np.asarray(xcr).copy(),
            "rgb": np.asarray(rgb).copy() if rgb is not None else None,
            "depth": np.asarray(depth).copy() if depth is not None else None,
            "goal_heading": np.asarray(goal_heading).copy(),
            "goal_dist": float(goal_dist),
            "vel_pred": np.asarray(vel_pred).copy(),
            "u_cmd": np.asarray(body_rate_cmd).copy(),
            "expert_vel": (
                np.asarray(expert_vel).copy() if expert_vel is not None else None
            ),
        })

    def __len__(self):
        return len(self.steps)

    def clear(self):
        self.steps.clear()

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"steps": self.steps}, path)
        return path

    @staticmethod
    def load(path) -> "RolloutRecorder":
        data = torch.load(Path(path), map_location="cpu")
        r = RolloutRecorder()
        r.steps = data["steps"]
        return r
