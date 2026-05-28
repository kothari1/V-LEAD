#!/usr/bin/env python3
"""Smoke test for vlead_flight.env.FigsDroneEnv.

Runs one short episode with uniformly random actions and prints per-step
diagnostics. Verifies FiGS + gsplat wire-up, observation shapes, reward
+ termination signals.

Usage:
    python scripts/smoke_env.py \
        --scene "flightroom_ssv_exp/gemsplat/2026-02-28_205058" \
        --steps 20
"""
from __future__ import annotations

import argparse

import numpy as np

# torch>=2.6 compatibility for nerfstudio/gsplat checkpoints. Must run before
# any module that imports FiGS / triggers nerfstudio.eval_setup.
from vlead_flight._torch_compat import enable_legacy_torch_load

enable_legacy_torch_load()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=str)
    ap.add_argument("--rollout", default="baseline", type=str)
    ap.add_argument("--frame", default="carl", type=str)
    ap.add_argument("--steps", default=20, type=int)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()

    from vlead_flight.env import (
        EpisodeSampler,
        FigsDroneEnv,
        RewardConfig,
        TerminationConfig,
    )

    env = FigsDroneEnv(
        scene_name=args.scene,
        rollout_name=args.rollout,
        frame_name=args.frame,
        hz=20,
        frame_window=4,
        sampler=EpisodeSampler(
            start_xyz_low=(-0.3, -0.3, -1.3),
            start_xyz_high=(0.3, 0.3, -1.1),
            goal_radius_min=1.0,
            goal_radius_max=2.5,
            goal_z_low=-1.4,
            goal_z_high=-1.0,
        ),
        reward_cfg=RewardConfig(),
        term_cfg=TerminationConfig(
            max_episode_steps=args.steps,
            ground_z=0.10,
            ceiling_z=-3.0,
            speed_kill=5.0,
        ),
    )

    obs, info = env.reset(seed=args.seed)
    print(f"reset: rgb={obs['rgb'].shape} {obs['rgb'].dtype} | goal={obs['goal']} | target={info['target_xyz']}")

    rng = np.random.default_rng(args.seed)
    total_r = 0.0
    for i in range(args.steps):
        action = rng.uniform(env.action_space.low, env.action_space.high)
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        print(
            f"step {i:3d} | a={np.round(action, 2)} | r={r:+.3f} | "
            f"dist={info['dist_to_goal']:.3f} | term={terminated} trunc={truncated} | "
            f"reason={info['term_reason']}"
        )
        if terminated or truncated:
            break

    print(f"total reward: {total_r:+.3f}")
    env.close()


if __name__ == "__main__":
    main()
