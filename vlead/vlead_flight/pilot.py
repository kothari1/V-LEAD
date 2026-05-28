"""V-LEAD pilot: trained network → velocity commands → body rates → ACADOS.

Wraps a trained V-LEAD network (following VLeadNetworkProtocol) as a
FiGS-compatible duck-typed controller. The Simulator can pass this object as
`policy` to `simulate()` with no changes to FiGS code.

Architecture per control step:
    1. Compute goal_heading, goal_distance from (target_xyz - xcr[0:3]).
    2. Acquire observation:
         - RGB: use icr if matches resolution, else preprocess (resize+normalize).
         - Depth (if use_depth): own gsplat.render_rgb() call at current pose.
    3. Push frame(s) into circular T-frame buffer.
    4. Forward pass network → (B=1, H, 4) receding-horizon velocity tensor.
    5. Take first velocity command (standard receding-horizon convention).
    6. Delegate to internal VelocityController → body rates.
    7. Optionally record step (for offline analysis / DAgger / RL).
    8. Return (ucr, zcr, adv, tsol) — matches FiGS BaseController contract.
"""
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from vlead_flight.observation import (
    FrameBuffer,
    compute_goal,
    imagenet_norm_buffers,
    preprocess_depth,
    preprocess_rgb,
)


class VLeadPilot:
    """Duck-typed FiGS controller wrapping a trained V-LEAD network."""

    def __init__(
        self,
        network: torch.nn.Module,
        target_xyz: np.ndarray,
        gsplat,
        frame_name: str = "carl",
        hz: int = 20,
        use_depth: bool = False,
        frame_window: int = 8,
        img_resolution: Tuple[int, int] = (224, 224),
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        kv: float = 2.0,
        ka: float = 5.0,
        configs_path: Optional[Path] = None,
        recorder=None,
        compile_network: bool = False,
        autocast: bool = False,
    ):
        """
        Args:
            network:         torch.nn.Module following VLeadNetworkProtocol.
            target_xyz:      (3,) world-frame goal position.
            gsplat:          Reference to Simulator.gsplat (for depth render).
            frame_name:      Drone frame config (loads mass, thrust, camera).
            hz:              Control frequency, must match Simulator.
            use_depth:       If True, pilot renders its own depth via gsplat
                             at each step (~+1 GPU render cost).
            frame_window:    Temporal window T for spatiotemporal backbones.
            img_resolution:  (H, W) network input resolution.
            device:          'cuda' or 'cpu'. Auto-detected if None.
            dtype:           Tensor dtype. Use bf16 on Blackwell for speedup.
            kv, ka:          Inner-loop VelocityController gains.
            configs_path:    Override for figs configs/ root.
            recorder:        Optional RolloutRecorder to capture per-step data.
            compile_network: If True, wrap network with torch.compile (silently
                             falls back if unsupported).
            autocast:        If True, run network forward inside torch.autocast
                             on the chosen dtype. Recommended with bf16/fp16.
        """
        # Lazy imports keep `vlead` importable on systems without FiGS.
        from figs.control.velocity_controller import VelocityController
        from figs.control.base_controller import BaseController
        from figs.dynamics.model_specifications import generate_specifications

        # ── Duck-typing contract attrs ────────────────────────────────────
        self.hz = hz
        self.nzcr = None

        # ── Device + dtype (Blackwell-friendly) ───────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.autocast = autocast and dtype != torch.float32

        # ── Network ───────────────────────────────────────────────────────
        self.network = network.to(self.device).to(self.dtype).eval()
        if compile_network:
            try:
                self.network = torch.compile(self.network, mode="reduce-overhead")
            except Exception as e:
                print(f"[vlead] torch.compile unavailable, continuing without ({e})")

        # ── Inner velocity → body rate controller ─────────────────────────
        self._inner = VelocityController(
            hz=hz, Kv=kv, Ka=ka,
            frame_name=frame_name, configs_path=configs_path,
        )

        # ── Frame / camera config ─────────────────────────────────────────
        loader = _ConfigLoader(configs_path)
        frame_cfg = loader.load_json_config("frame", frame_name)
        drn_spec = generate_specifications(frame_cfg)
        self._T_c2b = np.asarray(drn_spec["T_c2b"], dtype=np.float64)
        self._cam_cfg = drn_spec["camera"]

        # ── GSplat for depth rendering ────────────────────────────────────
        self._gsplat = gsplat
        self._gs_camera = gsplat.generate_output_camera(self._cam_cfg) if gsplat is not None else None

        # ── Goal + observation config ─────────────────────────────────────
        self.target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        self.use_depth = use_depth
        self.frame_window = frame_window
        self.img_h, self.img_w = img_resolution

        # ImageNet normalization stats (consistent with SINGER preprocessing)
        self._mean, self._std = imagenet_norm_buffers(self.device, self.dtype)

        # ── Circular frame buffers ────────────────────────────────────────
        self._rgb_fb = FrameBuffer(
            frame_window, 3, (self.img_h, self.img_w),
            device=self.device, dtype=self.dtype,
        )
        self._depth_fb = (
            FrameBuffer(
                frame_window, 1, (self.img_h, self.img_w),
                device=self.device, dtype=self.dtype,
            )
            if use_depth else None
        )

        # ── Recorder hook ─────────────────────────────────────────────────
        self.recorder = recorder

    # ── Public API ────────────────────────────────────────────────────────

    def set_target(self, target_xyz: np.ndarray) -> None:
        """Update the goal position. Triggers immediate goal re-computation
        on next control() call. Useful for moving targets or curriculum tests."""
        self.target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)

    def reset_buffer(self) -> None:
        """Clear the temporal frame buffer. Call between independent rollouts."""
        self._rgb_fb.reset()
        if self._depth_fb is not None:
            self._depth_fb.reset()

    # ── FiGS controller interface ─────────────────────────────────────────

    @torch.no_grad()
    def control(self, tcr, xcr, upr, obj, icr, zcr):
        """Called by Simulator.simulate() at each control step.

        Returns: (ucr, zcr, adv, tsol) matching BaseController contract.
        """
        t_start = time.time()

        # 1. Goal vector in world frame
        goal_heading, goal_dist = self._compute_goal(xcr)
        gh = torch.tensor(
            goal_heading, device=self.device, dtype=self.dtype
        ).unsqueeze(0)  # [1, 3]
        gd = torch.tensor(
            [[goal_dist]], device=self.device, dtype=self.dtype
        )  # [1, 1]

        # 2. Observation acquisition
        rgb_proc, depth_proc, rgb_raw, depth_raw = self._acquire_observation(xcr, icr)

        # 3. Push into temporal buffer
        self._push_frame(rgb_proc, depth_proc)

        # 4. Network forward
        rgb_in, depth_in = self._get_temporal_input()
        if self.autocast:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                pred = self.network(rgb_in, depth_in, gh, gd)  # [1, H, 4]
        else:
            pred = self.network(rgb_in, depth_in, gh, gd)

        pred_np = pred[0].detach().to(torch.float32).cpu().numpy()  # [H, 4]

        # 5. Apply first command (receding-horizon convention)
        vel_cmd = pred_np[0]  # [4,]

        # 6. Inner loop: velocity → body rates
        ucr, _, adv, _ = self._inner.control(tcr, xcr, upr, vel_cmd, None, None)

        # 7. Record step if requested
        if self.recorder is not None:
            self.recorder.add(
                tcr=tcr, xcr=xcr,
                rgb=rgb_raw, depth=depth_raw,
                goal_heading=goal_heading, goal_dist=goal_dist,
                vel_pred=pred_np, body_rate_cmd=ucr,
                expert_vel=None,
            )

        tsol = np.array([0.0, time.time() - t_start, 0.0, 0.0])
        return ucr, zcr, adv, tsol

    # ── Internals ─────────────────────────────────────────────────────────

    def _compute_goal(self, xcr: np.ndarray):
        return compute_goal(xcr, self.target_xyz)

    def _acquire_observation(self, xcr, icr):
        """Returns (rgb_proc, depth_proc, rgb_raw, depth_raw).

        - rgb_raw: native-resolution uint8 array as numpy
        - rgb_proc: normalized tensor [3, H, W] on device
        - depth_*: similar for depth (None if use_depth=False)
        """
        rgb_raw = None
        depth_raw = None

        if self.use_depth:
            # Need depth → render ourselves from gsplat at current pose
            if self._gsplat is None:
                raise RuntimeError(
                    "VLeadPilot.use_depth=True but gsplat reference is None. "
                    "Pass gsplat=sim.gsplat at construction."
                )
            from figs.utilities.trajectory_helper import xv_to_T
            T_b2w = xv_to_T(xcr)
            T_c2w = T_b2w @ self._T_c2b
            img_dict = self._gsplat.render_rgb(self._gs_camera, T_c2w)
            rgb_raw = img_dict["rgb"]
            depth_raw = img_dict.get("depth_raw")
            if depth_raw is None:
                raise RuntimeError(
                    "gsplat.render_rgb did not return 'depth_raw'. "
                    "Check perception_mode.yml or gsplat version."
                )
        else:
            rgb_raw = icr
            if rgb_raw is None:
                raise RuntimeError(
                    "Simulator passed icr=None to VLeadPilot.control(). "
                    "Set perception_mode.yml visual_mode='rgb'."
                )

        rgb_proc = self._preprocess_rgb(rgb_raw)
        depth_proc = self._preprocess_depth(depth_raw) if depth_raw is not None else None
        return rgb_proc, depth_proc, rgb_raw, depth_raw

    def _preprocess_rgb(self, rgb_np: np.ndarray) -> torch.Tensor:
        return preprocess_rgb(
            rgb_np,
            device=self.device, dtype=self.dtype,
            mean=self._mean, std=self._std,
            target_hw=(self.img_h, self.img_w),
        )

    def _preprocess_depth(self, depth_np: np.ndarray) -> torch.Tensor:
        return preprocess_depth(
            depth_np,
            device=self.device, dtype=self.dtype,
            target_hw=(self.img_h, self.img_w),
        )

    def _push_frame(self, rgb_tensor: torch.Tensor, depth_tensor):
        self._rgb_fb.push(rgb_tensor)
        if self._depth_fb is not None and depth_tensor is not None:
            self._depth_fb.push(depth_tensor)

    def _get_temporal_input(self):
        """Return tensors in chronological order (oldest → newest).

        rgb:   [1, T, 3, H, W]
        depth: [1, T, 1, H, W] or None
        """
        rgb = self._rgb_fb.get_batched()
        depth = self._depth_fb.get_batched() if self._depth_fb is not None else None
        return rgb, depth


# ── Small helper to reuse BaseController.load_json_config without inheriting ──

class _ConfigLoader:
    """Stand-in for BaseController to access its load_json_config helper."""
    def __init__(self, configs_path: Optional[Path]):
        from figs.control.base_controller import BaseController
        # BaseController is abstract; we use its non-abstract config loader by
        # constructing a minimal subclass on the fly.
        class _Loader(BaseController):
            def control(self, tcr, xcr, upr, obj, icr, zcr): ...
        self._loader = _Loader(configs_path)

    def load_json_config(self, kind: str, name: str):
        return self._loader.load_json_config(kind, name)
