"""Post-rollout evaluation metrics for V-LEAD deployments."""
from typing import Dict

import numpy as np


def goal_reached(Xro: np.ndarray, target_xyz: np.ndarray, radius: float = 0.5) -> bool:
    """True iff drone enters `radius` of target at any timestep."""
    target = np.asarray(target_xyz).reshape(3, 1)
    dists = np.linalg.norm(Xro[0:3, :] - target, axis=0)
    return bool(np.any(dists < radius))


def time_to_goal(
    Tro: np.ndarray, Xro: np.ndarray, target_xyz: np.ndarray, radius: float = 0.5
) -> float:
    """First time at which drone enters goal radius. Returns inf if never reached."""
    target = np.asarray(target_xyz).reshape(3, 1)
    dists = np.linalg.norm(Xro[0:3, :] - target, axis=0)
    hits = np.where(dists < radius)[0]
    if len(hits) == 0:
        return float("inf")
    return float(Tro[hits[0]])


def final_distance(Xro: np.ndarray, target_xyz: np.ndarray) -> float:
    return float(np.linalg.norm(Xro[0:3, -1] - np.asarray(target_xyz)))


def trajectory_length(Xro: np.ndarray) -> float:
    """Cumulative arc length of the 3D position trajectory (meters)."""
    diffs = np.diff(Xro[0:3, :], axis=1)
    return float(np.sum(np.linalg.norm(diffs, axis=0)))


def control_bounds_violations(Uro: np.ndarray):
    """Count timesteps where uf or body rates exceed FiGS bounds."""
    uf_viol = int(np.sum((Uro[0] < -1.0) | (Uro[0] > 0.0)))
    w_viol = int(np.sum(np.abs(Uro[1:4]) > 5.0))
    return uf_viol, w_viol


def summarize(
    Tro: np.ndarray,
    Xro: np.ndarray,
    Uro: np.ndarray,
    target_xyz: np.ndarray,
    radius: float = 0.5,
) -> Dict:
    uf_viol, w_viol = control_bounds_violations(Uro)
    return {
        "success": goal_reached(Xro, target_xyz, radius),
        "final_dist": final_distance(Xro, target_xyz),
        "time_to_goal": time_to_goal(Tro, Xro, target_xyz, radius),
        "traj_length": trajectory_length(Xro),
        "uf_bound_violations": uf_viol,
        "rate_bound_violations": w_viol,
        "mean_uf": float(Uro[0].mean()),
    }


def print_summary(summary: Dict) -> None:
    print("─" * 56)
    print(" Rollout summary")
    print("─" * 56)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("─" * 56)
