"""Closed-loop evaluation of a saved SAC checkpoint in FigsDroneEnv.

Loads model + builds env from the same yaml, runs N episodes with a
deterministic policy, summarizes per-episode reward / length / termination
reason / final distance. Saves JSON + per-episode CSV.

Usage:
    python scripts/eval_sac.py \\
        --config configs/sac_default.yaml \\
        --checkpoint /project/kothari1/vlead_data/rl_runs/sac_v1_seed0/best/best_model.zip \\
        --n-episodes 20 \\
        --output-dir /project/kothari1/vlead_data/rl_runs/sac_v1_seed0/closed_loop_eval
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

# torch>=2.6 compatibility for nerfstudio/gsplat checkpoints.
from vlead_flight._torch_compat import enable_legacy_torch_load

enable_legacy_torch_load()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to SAC .zip (best_model.zip or sac_*.zip).")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=12345,
                        help="Sampler seed for eval (separate from train seed).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = args.seed
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from stable_baselines3 import SAC
    from nav_policy.rl.train.train_sac import _build_env

    print(f"[eval] building env (scene={cfg['env']['scene_name']})", flush=True)
    env = _build_env(cfg)

    print(f"[eval] loading SAC checkpoint: {args.checkpoint}", flush=True)
    model = SAC.load(args.checkpoint, env=env, device=device)

    rows = []
    reason_counts: Counter = Counter()
    print(f"[eval] running {args.n_episodes} episodes (deterministic={args.deterministic})", flush=True)
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        target = info["target_xyz"]
        x0 = info["x0"]
        ep_reward = 0.0
        ep_len = 0
        start_dist = float(np.linalg.norm(target - x0[0:3]))
        final_dist = start_dist
        reason = ""
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            final_dist = float(info["dist_to_goal"])
            reason = info["term_reason"]
        success = (reason == "success")
        reason_counts[reason or "timeout"] += 1
        rows.append({
            "episode": ep,
            "ep_reward": ep_reward,
            "ep_length": ep_len,
            "term_reason": reason or "timeout",
            "success": int(success),
            "start_dist": start_dist,
            "final_dist": final_dist,
            "dist_closed": start_dist - final_dist,
        })
        print(
            f"[eval] ep {ep:3d} | r={ep_reward:+8.2f} | len={ep_len:4d} | "
            f"final_dist={final_dist:.2f} (start {start_dist:.2f}) | {reason or 'timeout'}",
            flush=True,
        )

    csv_path = out_dir / "per_episode.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rewards = np.array([r["ep_reward"] for r in rows], dtype=np.float64)
    lengths = np.array([r["ep_length"] for r in rows], dtype=np.float64)
    final_d = np.array([r["final_dist"] for r in rows], dtype=np.float64)
    successes = np.array([r["success"] for r in rows], dtype=np.float64)

    summary = {
        "checkpoint": args.checkpoint,
        "n_episodes": int(args.n_episodes),
        "deterministic": bool(args.deterministic),
        "success_rate": float(successes.mean()),
        "ep_reward_mean": float(rewards.mean()),
        "ep_reward_std": float(rewards.std()),
        "ep_length_mean": float(lengths.mean()),
        "ep_length_std": float(lengths.std()),
        "final_dist_mean": float(final_d.mean()),
        "final_dist_std": float(final_d.std()),
        "term_reason_counts": dict(reason_counts),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 56)
    print(" closed-loop eval summary")
    print("=" * 56)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")
    print(f"\n[eval] CSV: {csv_path}")
    print(f"[eval] JSON: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
