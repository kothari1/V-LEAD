"""Gymnasium env wrapping FiGS for online RL training of the V-LEAD pilot.

Single-env design (one Simulator per env instance). The env owns:
    - figs.simulator.Simulator        (gsplat + ACADOS solver + drone configs)
    - figs.control.VelocityController (vel cmd -> body-rate cmd inner loop)
    - vlead_flight.observation         (RGB preprocessing + temporal frame buffer)
    - vlead_flight.env.episode_sampler (start/goal randomization)
    - vlead_flight.env.reward          (composable goal-conditioned reward)
    - vlead_flight.env.termination     (success/crash/timeout predicates)

Per env.step(): convert (vx,vy,vz,psi_dot) action -> body-rate cmd via the
inner VelocityController, run n_sim2ctl ACADOS substeps, render the new RGB
frame from gsplat at the new pose, push to the frame buffer, compute reward
and termination, return Gymnasium tuple.

Observation is a Dict (matches SB3 MultiInputPolicy):
    rgb:  uint8 [T, 3, H, W]   ImageNet stats applied at policy side, not here
                                (keeps replay buffer compact; cast+normalize in
                                feature extractor)
    goal: float32 [goal_input_dim]   [hx, hy, hz, d_normalized] or [hx, hy, d_norm]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from vlead_flight.env.episode_sampler import EpisodeSampler, EpisodeSpec
from vlead_flight.env.reward import GoalReward, RewardConfig
from vlead_flight.env.termination import TerminationConfig, check_termination
from vlead_flight.observation import (
    FrameBuffer,
    compute_goal,
    imagenet_norm_buffers,
    preprocess_rgb,
)


class FigsDroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scene_name: str,
        rollout_name: str = "baseline",
        frame_name: str = "carl",
        configs_path: Optional[Path] = None,
        gsplats_path: Optional[Path] = None,
        *,
        hz: int = 20,
        frame_window: int = 4,
        img_resolution: Tuple[int, int] = (224, 224),
        goal_distance_scale: float = 5.0,
        goal_input_dim: int = 3,
        action_low: Tuple[float, float, float, float] = (-3.0, -3.0, -1.5, -1.5),
        action_high: Tuple[float, float, float, float] = (3.0, 3.0, 1.5, 1.5),
        kv: float = 2.0,
        ka: float = 5.0,
        sampler: Optional[EpisodeSampler] = None,
        reward_cfg: Optional[RewardConfig] = None,
        term_cfg: Optional[TerminationConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Lazy FiGS imports so the package remains importable on stripped envs.
        from figs.simulator import Simulator
        from figs.control.velocity_controller import VelocityController
        from figs.utilities.trajectory_helper import xv_to_T  # noqa: F401  (used in _render)

        self._scene_name = scene_name
        self._frame_name = frame_name
        self._sim = Simulator(
            scene_name=scene_name,
            rollout_name=rollout_name,
            frame_name=frame_name,
            configs_path=configs_path,
            gsplats_path=gsplats_path,
        )
        self._vel_ctrl = VelocityController(
            hz=hz, Kv=kv, Ka=ka,
            frame_name=frame_name, configs_path=configs_path,
        )

        # Derived sim cadence.
        self._hz_ctrl = hz
        self._hz_sim = int(self._sim.conFiG["rollout"]["frequency"])
        if self._hz_sim % hz != 0:
            raise ValueError(
                f"hz_sim ({self._hz_sim}) must be a multiple of control hz ({hz})"
            )
        self._n_sim2ctl = self._hz_sim // hz
        self._dt_ctl = 1.0 / hz

        # Camera + extrinsics (for direct gsplat render).
        cam_cfg = self._sim.conFiG["drone"]["camera"]
        self._T_c2b = np.asarray(self._sim.conFiG["drone"]["T_c2b"], dtype=np.float64)
        self._gs_camera = self._sim.gsplat.generate_output_camera(cam_cfg)
        self._xv_to_T = xv_to_T

        # Device + dtype.
        self._device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = torch.float32
        self._mean, self._std = imagenet_norm_buffers(self._device, self._dtype)

        # Frame buffer (stores RAW uint8 → cheap replay; normalize in feature extractor)
        self.frame_window = frame_window
        self.img_h, self.img_w = img_resolution
        self._rgb_buf_np = np.zeros(
            (frame_window, 3, self.img_h, self.img_w), dtype=np.uint8
        )
        self._buf_idx = 0
        self._buf_filled = False

        # Sampling, reward, termination
        self._sampler = sampler if sampler is not None else EpisodeSampler()
        self._reward = GoalReward(reward_cfg if reward_cfg is not None else RewardConfig())
        self._term_cfg = term_cfg if term_cfg is not None else TerminationConfig()

        # Goal embedding
        if goal_input_dim not in (3, 4):
            raise ValueError(
                "FigsDroneEnv supports goal_input_dim 3 ([hx,hy,d/scale]) or "
                "4 ([hx,hy,hz,d/scale])."
            )
        self.goal_input_dim = goal_input_dim
        self.goal_distance_scale = float(goal_distance_scale)

        # Spaces
        self.action_space = spaces.Box(
            low=np.array(action_low, dtype=np.float32),
            high=np.array(action_high, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(
                low=0, high=255,
                shape=(frame_window, 3, self.img_h, self.img_w),
                dtype=np.uint8,
            ),
            "goal": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(goal_input_dim,), dtype=np.float32,
            ),
        })

        # Rollout state
        self._t = 0.0
        self._x = None  # (10,)
        self._target = None  # (3,)
        self._step_idx = 0
        self._prev_dist = 0.0

    # ── helpers ────────────────────────────────────────────────────────────

    def _render(self, x: np.ndarray) -> np.ndarray:
        """Render current-pose RGB via gsplat. Returns (H, W, 3) uint8."""
        T_b2w = self._xv_to_T(x)
        T_c2w = T_b2w @ self._T_c2b
        img_dict = self._sim.gsplat.render_rgb(self._gs_camera, T_c2w)
        return img_dict["rgb"]

    def _push_frame(self, rgb_raw_hwc: np.ndarray) -> None:
        # Resize via torch then store uint8 in the buffer (replay-friendly).
        # No normalization here.
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_raw_hwc)).permute(2, 0, 1)  # [3,H,W] uint8
        rgb_t = rgb_t.unsqueeze(0).float()
        if rgb_t.shape[-2:] != (self.img_h, self.img_w):
            rgb_t = torch.nn.functional.interpolate(
                rgb_t, size=(self.img_h, self.img_w),
                mode="bilinear", align_corners=False,
            )
        rgb_u8 = rgb_t.squeeze(0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        if not self._buf_filled:
            for k in range(self.frame_window):
                self._rgb_buf_np[k] = rgb_u8
            self._buf_idx = 1 % self.frame_window
            self._buf_filled = True
        else:
            self._rgb_buf_np[self._buf_idx] = rgb_u8
            self._buf_idx = (self._buf_idx + 1) % self.frame_window

    def _frame_stack(self) -> np.ndarray:
        order = [(self._buf_idx + k) % self.frame_window for k in range(self.frame_window)]
        return self._rgb_buf_np[order].copy()

    def _make_obs(self, heading: np.ndarray, dist: float) -> Dict[str, np.ndarray]:
        if self.goal_input_dim == 3:
            goal = np.array(
                [heading[0], heading[1], dist / self.goal_distance_scale],
                dtype=np.float32,
            )
        else:
            goal = np.array(
                [heading[0], heading[1], heading[2], dist / self.goal_distance_scale],
                dtype=np.float32,
            )
        return {"rgb": self._frame_stack(), "goal": goal}

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._sampler.seed(seed)

        spec: EpisodeSpec = self._sampler.sample()
        self._x = spec.x0.copy()
        self._target = spec.target_xyz.copy()
        self._t = 0.0
        self._step_idx = 0
        self._reward.reset()
        self._buf_idx = 0
        self._buf_filled = False

        rgb_raw = self._render(self._x)
        self._push_frame(rgb_raw)

        heading, dist = compute_goal(self._x, self._target)
        self._prev_dist = dist
        info = {"target_xyz": self._target.copy(), "x0": self._x.copy()}
        return self._make_obs(heading, dist), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # 1. Clip + interpret action as [vx, vy, vz, psi_dot].
        a = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low.astype(np.float64),
            self.action_space.high.astype(np.float64),
        )
        # 2. Inner velocity controller -> body-rate command.
        ucr, _, _, _ = self._vel_ctrl.control(
            self._t, self._x, None, a, None, None,
        )
        # 3. Advance ACADOS solver n_sim2ctl substeps with the held cmd.
        x_next = self._x
        for _ in range(self._n_sim2ctl):
            x_next = self._sim.solver.simulate(x=x_next, u=ucr)
        self._x = np.asarray(x_next)
        self._t += self._dt_ctl
        self._step_idx += 1

        # 4. Render new pose, push frame.
        rgb_raw = self._render(self._x)
        self._push_frame(rgb_raw)

        # 5. Goal/reward/termination.
        heading, new_dist = compute_goal(self._x, self._target)
        terminated, truncated, reason = check_termination(
            self._x, new_dist, self._step_idx - 1, self._term_cfg,
        )
        reward_comp = self._reward(
            xcr=self._x,
            prev_dist=self._prev_dist,
            new_dist=new_dist,
            action=a,
            goal_heading=heading,
            terminated=terminated,
            term_reason=reason,
        )
        self._prev_dist = new_dist

        obs = self._make_obs(heading, new_dist)
        info = {
            "term_reason": reason,
            "dist_to_goal": new_dist,
            "reward_components": reward_comp,
            "x": self._x.copy(),
            "ucr": ucr.copy(),
        }
        return obs, float(reward_comp["total"]), bool(terminated), bool(truncated), info

    def close(self) -> None:
        # gsplat / ACADOS solvers clean themselves up on GC; nothing explicit.
        return
