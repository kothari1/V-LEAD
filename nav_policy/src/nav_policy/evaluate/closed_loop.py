"""
Closed-loop evaluation of a trained RGBVelocityPolicy inside the FiGS
simulator.

This script REQUIRES a 3DGS scene checkpoint that FiGS knows how to load via
``figs.simulator.Simulator(scene, rollout, frame)``. The processed cache and
the .pt trajectories alone are not enough -- they contain the recorded RGB
videos but not the splat needed to re-render new viewpoints once the policy
drives the drone off the expert's trajectory.

Outputs land under ``<output_dir>``:
    summary.json                aggregate metrics across all rollouts
    per_rollout.csv             one row per rollout
    rollout_<name>/             one folder per rollout with
        video.mp4                 policy-driven RGB rollout
        trajectory.npz            {Tro, Xro, Uro, Tsol, Adv} from sim.simulate
        expert_reference.npz      {Tro, Xro, Uro} loaded from setup_from
        metrics.json              per-rollout metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from nav_policy.deploy.policy_controller import RGBVelocityController


CMD_NAMES = ("vx", "vy", "vz", "psi_dot")


@dataclass
class ExpertRef:
    """Subset of a saved validation rollout used as an oracle for comparison."""
    Tro: np.ndarray   # (Nctl+1,)
    Xro: np.ndarray   # (10, Nctl+1)
    Uro: np.ndarray   # (4, Nctl)
    t0: float
    tf: float
    x0: np.ndarray    # (10,)
    course: str
    rollout_id: str
    setup_from: Path
    sub_idx: int
    goal_xy: Optional[np.ndarray] = None   # (2,) semantic target XY; None if unavailable


def load_expert_setup(setup_from: Path, sub_idx: int) -> ExpertRef:
    blob = torch.load(setup_from, weights_only=False, map_location="cpu")
    # Handle flat-list (V-LEAD/flightroom format) and dict-with-'data' (SINGER format)
    if isinstance(blob, list):
        trajs = blob
    elif isinstance(blob, dict) and "data" in blob:
        trajs = blob["data"]
    else:
        raise ValueError(
            f"{setup_from}: expected list or dict-with-'data', got {type(blob).__name__}"
        )
    if sub_idx >= len(trajs):
        raise IndexError(
            f"{setup_from}: sub_idx={sub_idx} out of range ({len(trajs)} trajectories)"
        )
    traj = trajs[sub_idx]
    Tro = np.asarray(traj["Tro"], dtype=np.float64)
    Xro = np.asarray(traj["Xro"], dtype=np.float64)
    Uro = np.asarray(traj["Uro"], dtype=np.float64)
    goal_xy = None
    if "goal_xy" in traj:
        goal_xy = np.asarray(traj["goal_xy"], dtype=np.float64).ravel()[:2]
    return ExpertRef(
        Tro=Tro, Xro=Xro, Uro=Uro,
        t0=float(Tro[0]),
        tf=float(Tro[-1]),
        x0=Xro[:, 0].copy(),
        course=str(traj.get("course", "")),
        rollout_id=str(traj.get("rollout_id", "")),
        setup_from=setup_from,
        sub_idx=sub_idx,
        goal_xy=goal_xy,
    )


def _yaw_series(quat_cols: np.ndarray) -> np.ndarray:
    """quat_cols: (4, N) Hamilton scalar-last -> unwrapped yaw (N,)."""
    yaw = Rotation.from_quat(quat_cols.T).as_euler("xyz", degrees=False)[:, 2]
    return np.unwrap(yaw)


def _path_length(positions: np.ndarray) -> float:
    """positions: (3, N) -> total path length in meters."""
    if positions.shape[1] < 2:
        return 0.0
    diffs = np.diff(positions, axis=1)
    return float(np.sum(np.linalg.norm(diffs, axis=0)))


def _bbox_violation(positions: np.ndarray,
                    ref_positions: np.ndarray,
                    margin: float = 2.0) -> Tuple[bool, int]:
    """Return (any_violation, first_violation_step) using a bbox grown from the expert."""
    lo = ref_positions.min(axis=1) - margin
    hi = ref_positions.max(axis=1) + margin
    out = (positions.T < lo) | (positions.T > hi)             # (N, 3)
    bad = out.any(axis=1)                                      # (N,)
    if bad.any():
        return True, int(np.argmax(bad))
    return False, -1


def compute_metrics(expert: ExpertRef,
                    Tpol: np.ndarray,
                    Xpol: np.ndarray,
                    Upol: np.ndarray,
                    Tsol: np.ndarray,
                    success_position_tol: float = 0.5,
                    success_tracking_tol: float = 1.0,
                    bbox_margin: float = 2.0,
                    success_goal_dist: float = 2.0) -> Dict[str, float]:
    """All quantities are in physical units (m, m/s, rad/s, seconds).

    Two success definitions are reported:
      success          — tracking-based: final position within success_position_tol of
                         the expert's end point AND tracking_rmse < success_tracking_tol
                         AND no bbox violation.
      goal_success     — task-based: drone ended within success_goal_dist of goal_xy
                         (the semantic target centroid) AND no bbox violation.
                         Falls back to `success` when goal_xy is unavailable.
    """
    n_pol = min(int(Tpol.shape[0]) - 1, int(expert.Tro.shape[0]) - 1, int(Upol.shape[1]))
    if n_pol <= 1:
        raise RuntimeError("policy rollout produced too few control steps")

    p_pol = Xpol[0:3, : n_pol + 1]
    p_exp = expert.Xro[0:3, : n_pol + 1]
    v_pol = Xpol[3:6, : n_pol + 1]
    v_exp = expert.Xro[3:6, : n_pol + 1]
    yaw_pol = _yaw_series(Xpol[6:10, : n_pol + 1])
    yaw_exp = _yaw_series(expert.Xro[6:10, : n_pol + 1])

    # Position tracking
    pos_err = np.linalg.norm(p_pol - p_exp, axis=0)            # (n+1,)
    tracking_rmse = float(np.sqrt(np.mean(pos_err ** 2)))
    final_pos_err = float(pos_err[-1])
    max_pos_err = float(pos_err.max())

    # Achieved-velocity tracking (drone-state vs expert-state)
    vel_err = v_pol - v_exp                                     # (3, n+1)
    vel_rmse_xyz = np.sqrt((vel_err ** 2).mean(axis=1))         # (3,)
    yaw_err = yaw_pol - yaw_exp
    yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi           # wrap
    yaw_rmse = float(np.sqrt(np.mean(yaw_err ** 2)))

    # Path length
    path_length_pol = _path_length(p_pol)
    path_length_exp = _path_length(p_exp)

    # Bounding-box safety proxy
    bbox_hit, bbox_step = _bbox_violation(p_pol, p_exp, margin=bbox_margin)

    # Tracking-based success (original criterion)
    success = (
        (not bbox_hit)
        and final_pos_err < success_position_tol
        and tracking_rmse < success_tracking_tol
    )

    # Goal-based success: reached within success_goal_dist of semantic target, no crash
    if expert.goal_xy is not None:
        goal_xy = expert.goal_xy                                 # (2,)
        # XY distance only — altitude is handled by the VelocityController
        dist_to_goal = np.linalg.norm(
            p_pol[:2, :] - goal_xy[:, None], axis=0
        )                                                        # (n+1,)
        dist_to_goal_final = float(dist_to_goal[-1])
        min_dist_to_goal = float(dist_to_goal.min())
        goal_success = (not bbox_hit) and (dist_to_goal_final < success_goal_dist)
    else:
        dist_to_goal_final = float("nan")
        min_dist_to_goal = float("nan")
        goal_success = success  # fall back to tracking-based when goal_xy not available

    # Inference latency (model forward time recorded by RGBVelocityController.tsol[1])
    model_latencies_ms = Tsol[1, :n_pol] * 1000.0
    inner_latencies_ms = Tsol[3, :n_pol] * 1000.0

    return {
        "n_steps": int(n_pol),
        "duration_s": float(Tpol[n_pol] - Tpol[0]),
        "success": bool(success),
        "goal_success": bool(goal_success),
        "bbox_violation": bool(bbox_hit),
        "bbox_violation_step": int(bbox_step),
        "dist_to_goal_final_m": dist_to_goal_final,
        "min_dist_to_goal_m": min_dist_to_goal,
        "tracking_rmse_m": tracking_rmse,
        "final_position_error_m": final_pos_err,
        "max_position_error_m": max_pos_err,
        "vel_rmse_x_mps": float(vel_rmse_xyz[0]),
        "vel_rmse_y_mps": float(vel_rmse_xyz[1]),
        "vel_rmse_z_mps": float(vel_rmse_xyz[2]),
        "vel_rmse_norm_mps": float(np.sqrt(np.mean((vel_err ** 2).sum(axis=0)))),
        "yaw_rmse_rad": yaw_rmse,
        "path_length_policy_m": path_length_pol,
        "path_length_expert_m": path_length_exp,
        "latency_model_ms_mean": float(model_latencies_ms.mean()),
        "latency_model_ms_p95": float(np.percentile(model_latencies_ms, 95)),
        "latency_inner_ms_mean": float(inner_latencies_ms.mean()),
    }


def _save_video(frames: np.ndarray, path: Path, fps: int = 20) -> None:
    import imageio.v3 as iio
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (N,H,W,3), got {frames.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), frames.astype(np.uint8), plugin="FFMPEG", fps=fps)


def run_one(rollout_cfg: dict,
            controller: RGBVelocityController,
            output_dir: Path) -> Dict[str, float]:
    """Run a single closed-loop FiGS rollout and write artifacts."""
    import gc
    import torch
    # Local import so this file can be imported on machines without the FiGS env.
    from figs.simulator import Simulator

    name = rollout_cfg["name"]
    scene = rollout_cfg["scene"]
    rollout = rollout_cfg.get("rollout", "baseline")
    frame = rollout_cfg.get("frame", "carl")
    setup_from = Path(rollout_cfg["setup_from"]).resolve()
    sub_idx = int(rollout_cfg.get("sub_idx", 0))

    expert = load_expert_setup(setup_from, sub_idx)
    rollout_dir = output_dir / f"rollout_{name}"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    sim = Simulator(scene, rollout, frame)

    # Use goal_xy (semantic target centroid) when available; fall back to the
    # expert's final position so the controller always has a valid goal vector.
    goal_pos_xy = (
        expert.goal_xy
        if expert.goal_xy is not None
        else expert.Xro[0:2, -1].astype(np.float64)
    )
    controller.reset(goal_pos_xy=goal_pos_xy)
    t_start = time.time()
    try:
        Tpol, Xpol, Upol, Imgs, Tsol, Adv = sim.simulate(
            controller, expert.t0, expert.tf, expert.x0,
            obj=None, query=None, vision_processor=None, validation=False,
        )
    finally:
        # Explicitly free the scene from GPU memory so the next rollout
        # (potentially a different, larger scene) can load without OOM.
        del sim
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    wall_time = time.time() - t_start

    metrics = compute_metrics(expert, Tpol, Xpol, Upol, Tsol, goal_xy=expert.goal_xy)
    metrics["wall_time_s"] = float(wall_time)
    metrics["scene"] = scene
    metrics["rollout"] = rollout
    metrics["frame"] = frame
    metrics["name"] = name
    metrics["course"] = expert.course

    np.savez_compressed(
        rollout_dir / "trajectory.npz",
        Tro=Tpol, Xro=Xpol, Uro=Upol, Tsol=Tsol, Adv=Adv,
    )
    np.savez_compressed(
        rollout_dir / "expert_reference.npz",
        Tro=expert.Tro, Xro=expert.Xro, Uro=expert.Uro,
    )
    if "rgb" in Imgs:
        _save_video(Imgs["rgb"], rollout_dir / "video.mp4", fps=int(controller.hz))
    with open(rollout_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    dist_str = (f"  dist_to_goal={metrics['dist_to_goal_final_m']:.2f}m"
                if not np.isnan(metrics.get("dist_to_goal_final_m", float("nan"))) else "")
    print(f"  [{name}] goal_success={metrics['goal_success']}  "
          f"tracking_rmse={metrics['tracking_rmse_m']:.3f}m{dist_str}  "
          f"latency={metrics['latency_model_ms_mean']:.1f}ms  "
          f"({metrics['n_steps']} steps in {wall_time:.1f}s)",
          flush=True)
    return metrics


def aggregate(per_rollout: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_rollout:
        return {}
    keys_to_avg = [
        "tracking_rmse_m", "final_position_error_m", "max_position_error_m",
        "dist_to_goal_final_m", "min_dist_to_goal_m",
        "vel_rmse_x_mps", "vel_rmse_y_mps", "vel_rmse_z_mps", "vel_rmse_norm_mps",
        "yaw_rmse_rad",
        "path_length_policy_m", "path_length_expert_m",
        "latency_model_ms_mean", "latency_model_ms_p95",
        "latency_inner_ms_mean", "duration_s",
    ]
    agg: Dict[str, float] = {}
    for k in keys_to_avg:
        vals = [r[k] for r in per_rollout if k in r and not np.isnan(r[k])]
        if vals:
            agg[f"mean_{k}"] = float(np.mean(vals))
            agg[f"std_{k}"] = float(np.std(vals))
    agg["n_rollouts"] = len(per_rollout)
    agg["success_rate"] = float(np.mean([1.0 if r["success"] else 0.0 for r in per_rollout]))
    agg["goal_success_rate"] = float(
        np.mean([1.0 if r.get("goal_success", r["success"]) else 0.0 for r in per_rollout])
    )
    agg["bbox_violation_rate"] = float(
        np.mean([1.0 if r["bbox_violation"] else 0.0 for r in per_rollout])
    )
    return agg


def evaluate(config_path: Path,
             checkpoint_override: Optional[Path] = None,
             output_dir_override: Optional[Path] = None,
             run_tag_override: Optional[str] = None) -> Dict[str, float]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides (ablation / per-round runs).  Each maps to the
    # corresponding YAML field.
    if checkpoint_override is not None:
        cfg["checkpoint"] = str(checkpoint_override)
    if output_dir_override is not None:
        cfg["output_dir"] = str(output_dir_override)
    if run_tag_override is not None:
        cfg["run_tag"] = str(run_tag_override)

    # Resolve all paths relative to the nav_policy root (config's grandparent)
    # BEFORE any FiGS/nerfstudio call, because Simulator.__init__ calls
    # os.chdir(DATA_PATH) which would corrupt subsequent relative .resolve() calls.
    base = config_path.resolve().parent.parent
    ckpt_path = (base / cfg["checkpoint"]).resolve()
    output_dir = (base / cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] {ckpt_path}")
    print(f"[output_dir] {output_dir}")

    # Pre-resolve all setup_from paths in the rollout configs.
    for rcfg in cfg.get("rollouts", []):
        if "setup_from" in rcfg:
            rcfg["setup_from"] = str((base / rcfg["setup_from"]).resolve())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = RGBVelocityController.from_checkpoint(
        ckpt_path,
        frame_name=cfg.get("frame", "carl"),
        Kv=float(cfg.get("Kv", 2.0)),
        Ka=float(cfg.get("Ka", 5.0)),
        device=device,
    )

    per_rollout: List[Dict[str, float]] = []
    for rcfg in cfg["rollouts"]:
        try:
            metrics = run_one(rcfg, controller, output_dir)
            per_rollout.append(metrics)
        except Exception as exc:                              # pragma: no cover
            print(f"  [{rcfg.get('name', '?')}] FAILED: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            per_rollout.append({
                "name": rcfg.get("name", "?"),
                "success": False, "bbox_violation": True,
                "error": str(exc),
            })

    summary = aggregate([r for r in per_rollout if "error" not in r])
    summary["checkpoint"] = str(ckpt_path)
    summary["run_tag"] = str(cfg.get("run_tag", output_dir.name))
    # Persist the zero-goal-heading flag stamped into the checkpoint so the
    # collector can attribute closed-loop numbers to the right ablation row.
    try:
        ckpt_blob = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        ckpt_train_cfg = ckpt_blob.get("config", {}).get("train", {})
        summary["zero_goal_heading"] = bool(ckpt_train_cfg.get("zero_goal_heading", False))
    except Exception:
        summary["zero_goal_heading"] = False
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if per_rollout:
        all_keys = sorted({k for r in per_rollout for k in r.keys()})
        with open(output_dir / "per_rollout.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in per_rollout:
                w.writerow(r)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop FiGS evaluation of a trained policy.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Override the YAML `checkpoint` field.  Useful for re-using one "
             "closed-loop config across BC and per-round DAgger checkpoints.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the YAML `output_dir` field so each ablation lands in "
             "its own directory and the collector can attribute the rows.",
    )
    p.add_argument(
        "--run-tag", type=str, default=None,
        help="Override the YAML `run_tag` field; persisted in summary.json.",
    )
    args = p.parse_args()
    evaluate(args.config,
             checkpoint_override=args.checkpoint,
             output_dir_override=args.output_dir,
             run_tag_override=args.run_tag)


if __name__ == "__main__":
    main()
