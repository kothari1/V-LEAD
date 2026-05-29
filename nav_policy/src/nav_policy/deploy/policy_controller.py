"""
FiGS-compatible controller that wraps a trained RGBVelocityPolicy and chains
the existing FiGS VelocityController as the inner loop.

Implements the duck-typed contract used by figs.simulator.Simulator.simulate():

    policy.hz                            -> int
    policy.nzcr                          -> Optional[int]
    policy.control(tcr, xcr, upr, obj, icr, zcr)
        returns (ucr, zcr, adv, tsol)

with ucr = [uf, wx, wy, wz] (consumed by the ACADOS integrator).

Receding-horizon execution:
    Every control step we predict a horizon of H velocity commands but execute
    only the FIRST step, then re-plan from the freshest frame on the next call.

Goal conditioning:
    At each step the controller reads the drone's current XY position from
    xcr[0:2] and computes:
        - a 2-D unit heading vector toward the stored goal_pos_xy, and
        - (optionally) a scale-normalized scalar distance to that goal.
    The resulting [hx, hy] or [hx, hy, d/scale] vector is passed to the policy
    alongside the RGB sequence so the policy can couple obstacle avoidance
    with explicit goal-seeking behaviour and approach-speed control.

    goal_pos_xy must be set before the first call to control(), either via
    the constructor or via set_goal().  The goal_input_dim and
    goal_distance_scale are loaded automatically from the checkpoint config so
    deployment matches training without any extra YAML.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from figs.control.velocity_controller import VelocityController

from nav_policy.data.normalization import CommandStats
from nav_policy.deploy.frame_buffer import FrameBuffer
from nav_policy.model.flow_matching_policy import FlowMatchingPolicy
from nav_policy.model.rgb_velocity_policy import RGBVelocityPolicy


class RGBVelocityController:
    """Wraps RGBVelocityPolicy + FiGS VelocityController for closed-loop deployment."""

    def __init__(self,
                 model: RGBVelocityPolicy,
                 stats: CommandStats,
                 inner: VelocityController,
                 image_size: int = 224,
                 goal_pos_xy: Optional[np.ndarray] = None,
                 zero_goal_heading: bool = False,
                 goal_distance_scale: float = 5.0,
                 device: Optional[torch.device] = None) -> None:
        if model.cmd_dim != 4:
            raise ValueError(f"expected cmd_dim=4, got {model.cmd_dim}")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.eval().to(self.device)
        self.stats = stats
        self.inner = inner
        self.buf = FrameBuffer(
            T=model.T,
            image_size=image_size,
            device=self.device,
        )
        # goal_pos_xy: 2-D world-frame position of the mission goal.
        # Set via set_goal() before the first control() call.
        self._goal_pos_xy: Optional[np.ndarray] = (
            np.asarray(goal_pos_xy, dtype=np.float64).ravel()[:2]
            if goal_pos_xy is not None else None
        )
        # No-goal-heading ablation: force the goal input to zero so the policy
        # must rely on vision alone.  Must match the training-time
        # train.zero_goal_heading flag; loaded automatically by
        # from_checkpoint() from the checkpoint's saved config.
        self._zero_goal_heading: bool = bool(zero_goal_heading)
        # Goal-input layout (2 = heading only, 3 = heading + normalized distance)
        # is inferred from the loaded model.  goal_distance_scale is loaded
        # from the checkpoint config so deployment matches training exactly.
        self._goal_input_dim: int = int(model.goal_input_dim)
        if goal_distance_scale <= 0.0:
            raise ValueError(f"goal_distance_scale must be > 0; got {goal_distance_scale}")
        self._goal_distance_scale: float = float(goal_distance_scale)
        # FiGS Simulator polls these.
        self.hz = inner.hz
        self.nzcr = None
        self.name = "RGBVelocityController"

    def set_goal(self, goal_pos_xy: np.ndarray) -> None:
        """Update the mission goal (world XY).  Call before each new episode."""
        self._goal_pos_xy = np.asarray(goal_pos_xy, dtype=np.float64).ravel()[:2]

    def _compute_goal_vector(self, xcr: np.ndarray) -> np.ndarray:
        """
        Assemble the goal-input vector for the policy.

        Layout:
            goal_input_dim=2 -> [hx, hy]                          (heading only)
            goal_input_dim=3 -> [hx, hy, d / goal_distance_scale] (heading + dist)
        """
        if self._zero_goal_heading:
            return np.zeros(self._goal_input_dim, dtype=np.float32)
        if self._goal_pos_xy is None:
            # Fallback when no goal is set: unit heading, zero distance.
            heading = np.array([1.0, 0.0], dtype=np.float32)
            d_norm = 0.0
        else:
            delta = self._goal_pos_xy - xcr[0:2].astype(np.float64)
            norm = float(np.linalg.norm(delta))
            if norm < 1e-6:
                heading = np.array([1.0, 0.0], dtype=np.float32)
                d_norm = 0.0
            else:
                heading = (delta / norm).astype(np.float32)
                d_norm = float(norm / self._goal_distance_scale)
        if self._goal_input_dim == 2:
            return heading
        return np.array([heading[0], heading[1], d_norm], dtype=np.float32)

    @classmethod
    def from_checkpoint(cls,
                        ckpt_path: Path,
                        frame_name: str = "carl",
                        Kv: float = 2.0,
                        Ka: float = 5.0,
                        goal_pos_xy: Optional[np.ndarray] = None,
                        device: Optional[torch.device] = None) -> "RGBVelocityController":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        cfg = ckpt["config"]
        model = RGBVelocityPolicy(
            T=int(cfg["window"]["T"]),
            H=int(cfg["window"]["H"]),
            cmd_dim=int(cfg["model"]["cmd_dim"]),
            gru_hidden=int(cfg["model"]["gru_hidden"]),
            gru_layers=int(cfg["model"]["gru_layers"]),
            mlp_hidden=tuple(cfg["model"]["mlp_hidden"]),
            mlp_dropout=float(cfg["model"].get("mlp_dropout", 0.1)),
            goal_emb_dim=int(cfg["model"].get("goal_emb_dim", 32)),
            goal_input_dim=int(cfg["model"].get("goal_input_dim", 2)),
            freeze_stem_and_layer1=bool(cfg["model"].get("freeze_stem_and_layer1", True)),
        )
        model.load_state_dict(ckpt["model"])
        stats = CommandStats.from_dict(ckpt["stats"])
        inner = VelocityController(hz=20, Kv=Kv, Ka=Ka, frame_name=frame_name)
        zero_goal = bool(cfg.get("train", {}).get("zero_goal_heading", False))
        goal_distance_scale = float(cfg.get("model", {}).get("goal_distance_scale", 5.0))
        return cls(
            model=model,
            stats=stats,
            inner=inner,
            image_size=int(cfg["window"].get("image_size", 224)),
            goal_pos_xy=goal_pos_xy,
            zero_goal_heading=zero_goal,
            goal_distance_scale=goal_distance_scale,
            device=device,
        )

    def reset(self, goal_pos_xy: Optional[np.ndarray] = None) -> None:
        """Reset frame buffer and optionally update the goal position."""
        self.buf.reset()
        if goal_pos_xy is not None:
            self.set_goal(goal_pos_xy)

    @torch.inference_mode()
    def _predict_first_command(self,
                               icr: np.ndarray,
                               goal_np: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return ([vx, vy, vz, psi_dot], inference_seconds) from the freshest frame."""
        t0 = time.time()
        self.buf.push(icr)
        rgb_seq = self.buf.tensor()                              # [1, T, 3, S, S]
        goal_t = torch.from_numpy(goal_np).unsqueeze(0).to(
            self.device, non_blocking=True
        )                                                        # [1, goal_input_dim]
        u_hat_z = self.model(rgb_seq, goal_t)                    # [1, H, 4] z-scored
        u_hat = self.stats.destandardize(u_hat_z)               # [1, H, 4] raw
        cmd0 = u_hat[0, 0].cpu().numpy().astype(np.float64)     # [4]
        return cmd0, time.time() - t0

    def control(self,
                tcr: float,
                xcr: np.ndarray,
                upr,
                obj,
                icr: np.ndarray,
                zcr) -> Tuple[np.ndarray, None, np.ndarray, np.ndarray]:
        """
        Args:
            tcr, xcr, upr, obj, zcr: see figs.control.base_controller.BaseController.
            icr: HxWx3 uint8 RGB frame from the FiGS renderer.

        Returns:
            ucr:  np.ndarray (4,) [uf, wx, wy, wz] consumed by the ACADOS integrator.
            zcr:  None.
            adv:  np.ndarray (4,) -- the de-standardized predicted velocity command (for logging).
            tsol: np.ndarray (4,) -- [_, model_seconds, _, inner_seconds].
        """
        goal_np = self._compute_goal_vector(xcr)                 # [goal_input_dim] float32
        cmd0, dt_model = self._predict_first_command(icr, goal_np)

        t1 = time.time()
        ucr, _, _, _ = self.inner.control(
            tcr=tcr,
            xcr=xcr,
            upr=upr,
            obj=cmd0,           # override with policy-predicted velocity
            icr=None,
            zcr=None,
        )
        dt_inner = time.time() - t1

        adv = cmd0.astype(np.float64)
        tsol = np.array([0.0, float(dt_model), 0.0, float(dt_inner)], dtype=np.float64)
        return ucr, None, adv, tsol


class FlowMatchingController(RGBVelocityController):
    """Wraps FlowMatchingPolicy + FiGS VelocityController for closed-loop deployment.

    Identical duck-typed interface to RGBVelocityController; the only differences
    are that the model is a FlowMatchingPolicy and prediction uses Euler ODE sampling
    instead of a direct forward pass.
    """

    def __init__(self, *args, n_sample_steps: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._n_sample_steps = n_sample_steps
        self.name = "FlowMatchingController"

    @classmethod
    def from_checkpoint(cls,
                        ckpt_path: Path,
                        frame_name: str = "carl",
                        Kv: float = 2.0,
                        Ka: float = 5.0,
                        goal_pos_xy: Optional[np.ndarray] = None,
                        device: Optional[torch.device] = None) -> "FlowMatchingController":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        cfg = ckpt["config"]
        fm_cfg = cfg.get("fm", {})
        model = FlowMatchingPolicy(
            T=int(cfg["window"]["T"]),
            H=int(cfg["window"]["H"]),
            cmd_dim=int(cfg["model"]["cmd_dim"]),
            gru_hidden=int(cfg["model"]["gru_hidden"]),
            gru_layers=int(cfg["model"]["gru_layers"]),
            goal_emb_dim=int(cfg["model"].get("goal_emb_dim", 32)),
            goal_input_dim=int(cfg["model"].get("goal_input_dim", 3)),
            freeze_stem_and_layer1=bool(cfg["model"].get("freeze_stem_and_layer1", True)),
            time_emb_dim=int(fm_cfg.get("time_emb_dim", 64)),
            vf_hidden=tuple(fm_cfg.get("vf_hidden", [512, 512, 512])),
            vf_use_skip=bool(fm_cfg.get("vf_use_skip", True)),
        )
        model.load_state_dict(ckpt["model"])
        stats = CommandStats.from_dict(ckpt["stats"])
        inner = VelocityController(hz=20, Kv=Kv, Ka=Ka, frame_name=frame_name)
        zero_goal = bool(cfg.get("train", {}).get("zero_goal_heading", False))
        goal_distance_scale = float(cfg.get("model", {}).get("goal_distance_scale", 5.0))
        n_sample_steps = int(fm_cfg.get("n_val_steps", 4))
        return cls(
            model=model,
            stats=stats,
            inner=inner,
            image_size=int(cfg["window"].get("image_size", 224)),
            goal_pos_xy=goal_pos_xy,
            zero_goal_heading=zero_goal,
            goal_distance_scale=goal_distance_scale,
            n_sample_steps=n_sample_steps,
            device=device,
        )

    @torch.inference_mode()
    def _predict_first_command(self,
                               icr: np.ndarray,
                               goal_np: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.time()
        self.buf.push(icr)
        rgb_seq = self.buf.tensor()
        goal_t = torch.from_numpy(goal_np).unsqueeze(0).to(
            self.device, non_blocking=True
        )
        u_hat_z = self.model.sample(rgb_seq, goal_t, n_steps=self._n_sample_steps)
        u_hat = self.stats.destandardize(u_hat_z)
        cmd0 = u_hat[0, 0].cpu().numpy().astype(np.float64)
        return cmd0, time.time() - t0
