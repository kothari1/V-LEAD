"""
DAgger: roll out the current policy in FiGS, query the expert at policy-
visited states, append the new (rgb, expert-label) samples to the dataset,
and write an updated manifest so a downstream ``train_bc.py`` call can
fine-tune on the aggregated data.

Two expert-oracle modes are supported, selected by the config field
``oracle: reference | mpc`` (default: ``reference``):

* ``reference`` -- copy the recorded expert state-trajectory's velocity and
  yaw rate at the matching time index::

      vel(k)      = expert.Xro[3:6, k]
      psi_dot(k)  = (yaw(expert.Xro[6:10, k+1]) - yaw(expert.Xro[6:10, k])) * hz

  This is a fixed-trajectory oracle that does not adapt to the policy's
  drifted state, so it cannot tell the policy how to recover.

* ``mpc`` (proper DAgger) -- re-solve the FiGS VehicleRateMPC starting from
  the policy's actual state at each visited control step and read the
  planned next state as the label::

      vel(k)      = solver.get(1, 'x')[3:6]
      psi_dot(k)  = (yaw(solver.get(1, 'x')[6:10]) - yaw(x_policy[6:10])) * hz

  This is the canonical DAgger label (Ross et al. 2011) and provides drift
  correction at the cost of one OCP solve per step.

Outputs:
    data/processed/<output_run>/cache/dagger_r{round}_<name>.pt
    data/processed/manifest.json   (extended with new entries tagged "round")
    data/processed/<output_run>/dagger_summary.json   (run metadata for the
                                                       ablation collector)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

from nav_policy.data.build_dataset import CONTROL_HZ, write_cache
from nav_policy.data.normalization import CommandStats
from nav_policy.deploy.policy_controller import RGBVelocityController
from nav_policy.evaluate.closed_loop import load_expert_setup


def _resize_uint8(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected HxWx3, got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if _HAVE_CV2:
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    from PIL import Image
    return np.asarray(Image.fromarray(frame, mode="RGB").resize((size, size), Image.BILINEAR))


def _quat_to_yaw_series(quat_cols: np.ndarray) -> np.ndarray:
    """quat_cols: (4, N) -> unwrapped yaw (N,)."""
    yaw = Rotation.from_quat(quat_cols.T).as_euler("xyz", degrees=False)[:, 2]
    return np.unwrap(yaw)


def _relabel_one(rollout_cfg: dict,
                 controller: RGBVelocityController,
                 image_size: int,
                 cache_path: Path,
                 dagger_round: int,
                 oracle: str = "reference",
                 mpc_policy: str = "vrmpc_rrt",
                 frame: str = "carl") -> Dict[str, object]:
    """Run the policy on one expert-setup, build a DAgger cache, return its summary."""
    from figs.simulator import Simulator

    name = rollout_cfg["name"]
    expert = load_expert_setup(Path(rollout_cfg["setup_from"]).resolve(),
                               int(rollout_cfg.get("sub_idx", 0)))
    sim = Simulator(
        rollout_cfg["scene"],
        rollout_cfg.get("rollout", "baseline"),
        rollout_cfg.get("frame", "carl"),
    )

    goal_pos_xy = expert.Xro[0:2, -1].astype(np.float64)
    controller.reset(goal_pos_xy=goal_pos_xy)
    t0 = time.time()
    try:
        Tpol, Xpol, Upol, Imgs, Tsol, _ = sim.simulate(
            controller, expert.t0, expert.tf, expert.x0,
        )
    finally:
        # Free the 3DGS scene from GPU memory before the next rollout loads
        # its own scene, otherwise the second Simulator load triggers OOM.
        del sim
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    wall_time = time.time() - t0

    if "rgb" not in Imgs:
        raise RuntimeError(f"[{name}] no rgb frames returned from sim.simulate")
    rgb_frames = Imgs["rgb"]                                  # (n_pol, H, W, 3) uint8
    n_pol = int(rgb_frames.shape[0])

    # Align to whichever of (policy steps, expert steps) is shorter.
    n_align = min(n_pol, int(expert.Xro.shape[1]) - 1)
    if n_align <= 0:
        raise RuntimeError(f"[{name}] no aligned control steps to relabel")

    relabel_t0 = time.time()
    if oracle == "reference":
        # Fixed-reference oracle: copy expert state-traj velocity and yaw rate
        # at the matching time index, regardless of where the policy actually is.
        vel = expert.Xro[3:6, :n_align].T.astype(np.float32)        # (n, 3)
        quat_cols = expert.Xro[6:10, : n_align + 1]
        yaw = _quat_to_yaw_series(quat_cols)
        psi_dot = ((yaw[1:] - yaw[:-1]) * CONTROL_HZ).astype(np.float32)
    elif oracle == "mpc":
        # Proper-DAgger oracle: re-solve the MPC at each policy-visited state.
        from nav_policy.dagger.mpc_oracle import MPCRelabeler

        relabeler = MPCRelabeler.from_expert(
            Xro=expert.Xro,
            Uro=expert.Uro,
            frame=frame,
            policy=mpc_policy,
            control_hz=CONTROL_HZ,
            use_RTI=False,
        )
        vel = np.zeros((n_align, 3), dtype=np.float32)
        psi_dot = np.zeros((n_align,), dtype=np.float32)
        for k in range(n_align):
            v_k, p_k = relabeler.label(float(Tpol[k]), Xpol[:, k])
            vel[k] = v_k
            psi_dot[k] = p_k
    else:
        raise ValueError(f"unknown oracle: {oracle!r} (expected 'reference' or 'mpc')")
    relabel_wall = time.time() - relabel_t0

    # Goal heading + distance for the DAgger cache, both computed from the
    # POLICY's actual XY positions (not the recorded expert positions) so they
    # match what the next policy will see on its own rollouts.
    goal_xy = expert.Xro[0:2, -1]                              # fixed goal = traj end
    pos_xy = Xpol[0:2, :n_align]                               # policy-visited positions
    delta = goal_xy[:, None] - pos_xy                          # (2, n)
    raw_norms = np.linalg.norm(delta, axis=0)                  # (n,) raw meters
    safe_norms = np.maximum(raw_norms[None, :], 1e-6)          # (1, n)
    goal_heading_np = (delta / safe_norms).T.astype(np.float32)  # (n, 2)
    goal_dist_np = raw_norms.astype(np.float32)                # (n,)

    # Resize and reorder policy-rendered frames into the cache's [n, 3, S, S] uint8 tensor.
    resized = np.stack(
        [_resize_uint8(f, image_size) for f in rgb_frames[:n_align]], axis=0
    )
    rgb = torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous()

    write_cache(
        cache_path,
        rgb_uint8=rgb,
        vel=torch.from_numpy(vel),
        psi_dot=torch.from_numpy(psi_dot),
        goal_heading=torch.from_numpy(goal_heading_np),
        goal_dist=torch.from_numpy(goal_dist_np),
        meta={
            "run": cache_path.parent.parent.name,
            "stack_id": str(rollout_cfg.get("name", "")),
            "sub_idx": int(rollout_cfg.get("sub_idx", 0)),
            "rollout_id": expert.rollout_id,
            "course": expert.course,
            "n_frames": int(n_align),
            "control_hz": CONTROL_HZ,
            "source": "dagger",
            "dagger_round": int(dagger_round),
            "oracle": oracle,
            "mpc_policy": mpc_policy if oracle == "mpc" else None,
            "setup_from": str(rollout_cfg["setup_from"]),
            "scene": rollout_cfg["scene"],
        },
    )

    print(f"  [{name}] wrote {cache_path.name}  n={n_align}  "
          f"rollout={wall_time:.1f}s  relabel={relabel_wall:.1f}s  oracle={oracle}")
    return {
        "name": name,
        "cache": cache_path,
        "n_frames": int(n_align),
        "duration_s": float(Tpol[n_align] - Tpol[0]) if n_align < Tpol.shape[0] else float(Tpol[-1] - Tpol[0]),
        "oracle": oracle,
        "rollout_wall_s": float(wall_time),
        "relabel_wall_s": float(relabel_wall),
    }


def _append_to_manifest(processed_root: Path,
                        new_caches: List[Path],
                        T: int,
                        H: int,
                        dagger_round: int,
                        split_assignment: str = "train") -> None:
    """Extend manifest.json with windows from new caches."""
    manifest_path = processed_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing; run build_dataset.py first to seed the manifest"
        )
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if int(manifest["T"]) != T or int(manifest["H"]) != H:
        raise ValueError(
            f"manifest T,H ({manifest['T']},{manifest['H']}) != dagger T,H ({T},{H})"
        )

    new_entries = 0
    for cache_path in new_caches:
        blob = torch.load(cache_path, weights_only=False, map_location="cpu")
        n = int(blob["rgb"].shape[0])
        k_min, k_max = T - 1, n - H
        if k_max <= k_min:
            continue
        rel = cache_path.relative_to(processed_root).as_posix()
        for k in range(k_min, k_max):
            manifest["samples"].append({
                "cache": rel,
                "k": int(k),
                "split": split_assignment,
                "round": int(dagger_round),
            })
            new_entries += 1

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"[manifest] appended {new_entries} windows from {len(new_caches)} caches "
          f"(round={dagger_round}, split={split_assignment})")


def run(config_path: Path) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve all paths relative to the nav_policy root (config's grandparent)
    # BEFORE any FiGS/Simulator call, because Simulator.__init__ calls
    # os.chdir(DATA_PATH) which corrupts subsequent relative .resolve() calls.
    nav_root = config_path.resolve().parent.parent
    base_cfg_path = (nav_root / cfg["base_config"]).resolve()
    with open(base_cfg_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    processed_root = (base_cfg_path.parent.parent / base_cfg["data"]["processed_root"]).resolve()
    T = int(base_cfg["window"]["T"])
    H = int(base_cfg["window"]["H"])
    image_size = int(base_cfg["window"]["image_size"])

    dagger_round = int(cfg["round"])
    output_run = str(cfg.get("output_run", f"dagger_r{dagger_round}"))
    cache_dir = processed_root / output_run / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = (nav_root / cfg["checkpoint"]).resolve()

    # Oracle selection: 'reference' (legacy, fixed-trajectory) or 'mpc'
    # (proper DAgger via VehicleRateMPC re-solve at each policy state).  A
    # per-rollout 'oracle' override is honored; the global cfg value is the
    # default.
    default_oracle = str(cfg.get("oracle", "reference")).lower()
    if default_oracle not in ("reference", "mpc"):
        raise ValueError(f"oracle must be 'reference' or 'mpc'; got {default_oracle!r}")
    mpc_policy = str(cfg.get("mpc_policy", "vrmpc_rrt"))
    frame_name = str(cfg.get("frame", "carl"))

    # Pre-resolve all setup_from paths before the first Simulator call.
    for rcfg in cfg.get("rollouts", []):
        if "setup_from" in rcfg:
            rcfg["setup_from"] = str((nav_root / rcfg["setup_from"]).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = RGBVelocityController.from_checkpoint(
        ckpt_path,
        frame_name=frame_name,
        Kv=float(cfg.get("Kv", 2.0)),
        Ka=float(cfg.get("Ka", 5.0)),
        device=device,
    )

    new_caches: List[Path] = []
    per_rollout_summaries: List[Dict[str, object]] = []
    for rcfg in cfg["rollouts"]:
        cache_name = f"dagger_r{dagger_round}_{rcfg['name']}.pt"
        cache_path = cache_dir / cache_name
        per_oracle = str(rcfg.get("oracle", default_oracle)).lower()
        try:
            res = _relabel_one(
                rcfg, controller, image_size=image_size,
                cache_path=cache_path, dagger_round=dagger_round,
                oracle=per_oracle,
                mpc_policy=mpc_policy,
                frame=frame_name,
            )
            new_caches.append(res["cache"])
            per_rollout_summaries.append({
                "name": res["name"],
                "n_frames": res["n_frames"],
                "duration_s": res["duration_s"],
                "oracle": res["oracle"],
                "rollout_wall_s": res["rollout_wall_s"],
                "relabel_wall_s": res["relabel_wall_s"],
                "cache": str(res["cache"].relative_to(processed_root).as_posix()),
            })
        except Exception as exc:                              # pragma: no cover
            print(f"  [{rcfg.get('name', '?')}] FAILED: {exc}", file=sys.stderr)
            per_rollout_summaries.append({
                "name": rcfg.get("name", "?"),
                "oracle": per_oracle,
                "error": str(exc),
            })

    if not new_caches:
        print("[dagger] no caches written; manifest unchanged", file=sys.stderr)
        return

    _append_to_manifest(
        processed_root, new_caches, T=T, H=H,
        dagger_round=dagger_round,
        split_assignment=cfg.get("split_assignment", "train"),
    )

    summary_path = processed_root / output_run / "dagger_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "round": dagger_round,
            "output_run": output_run,
            "oracle_default": default_oracle,
            "mpc_policy": mpc_policy if default_oracle == "mpc" else None,
            "checkpoint": str(ckpt_path),
            "config": str(config_path),
            "run_tag": str(cfg.get("run_tag", output_run)),
            "rollouts": per_rollout_summaries,
        }, f, indent=2)
    print(f"[dagger r{dagger_round}] summary -> {summary_path}")
    print(f"[dagger r{dagger_round}] done. Re-train with:\n"
          f"  python scripts/train_bc.py --config {base_cfg_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run one DAgger round on a trained checkpoint.")
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
