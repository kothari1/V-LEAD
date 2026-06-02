"""Custom SB3 callbacks for V-LEAD SAC training.

- RewardComponentsCallback: logs each reward term separately to TB so you can
  see which is dominating (progress vs smoothness vs altitude etc.).
- WandbSyncCallback: mirrors TB scalars to W&B if `wandb` is configured.
- build_callbacks(): one-stop assembly of Eval + Checkpoint + RewardComponents
  + Wandb based on the train_sac yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
except Exception:  # pragma: no cover
    BaseCallback = object  # type: ignore
    CallbackList = None  # type: ignore
    CheckpointCallback = None  # type: ignore
    EvalCallback = None  # type: ignore
    Monitor = None  # type: ignore


class RewardComponentsCallback(BaseCallback):
    """Logs the env's `info['reward_components']` dict to TB.

    Aggregates across the rollout vector (single env => one info per step) and
    flushes per-component mean every `log_freq` steps.
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)
        self._buf: Dict[str, List[float]] = {}
        self._term_counts: Dict[str, int] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            comp = info.get("reward_components") if isinstance(info, dict) else None
            if isinstance(comp, dict):
                for k, v in comp.items():
                    self._buf.setdefault(k, []).append(float(v))
            reason = info.get("term_reason") if isinstance(info, dict) else None
            if reason:
                self._term_counts[reason] = self._term_counts.get(reason, 0) + 1

        if self.num_timesteps % self.log_freq == 0 and self._buf:
            for k, vs in self._buf.items():
                if vs:
                    self.logger.record(f"reward_components/{k}", float(np.mean(vs)))
            for k, c in self._term_counts.items():
                self.logger.record(f"term_reason_count/{k}", int(c))
            self._buf.clear()
        return True


class WandbSyncCallback(BaseCallback):
    """Forwards SB3 logger scalars to W&B.

    Construct only after wandb.init has been called. Pulls the same scalars
    SB3's TB writer emits and re-logs them under the same keys.
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True
        try:
            import wandb
        except ImportError:
            return True
        # SB3's Logger.name_to_value holds the last-recorded scalars per key.
        if hasattr(self.logger, "name_to_value"):
            scalars = {k: float(v) for k, v in self.logger.name_to_value.items()
                       if isinstance(v, (int, float, np.floating, np.integer))}
            if scalars:
                wandb.log(scalars, step=self.num_timesteps)
        return True


class TrajectoryPlotCallback(BaseCallback):
    """Plots the drone trajectory at the end of every `plot_freq` episodes.

    For each plotted episode saves a PNG to `out_dir/traj/` and logs it to
    TensorBoard as an image so it appears in the Images tab.

    Each plot shows:
      - Top-down XY view: trajectory colored by step index, start (green circle),
        goal (red star), end marker (green ✓ = success, red ✗ = crash, grey ⏱ = timeout)
      - Altitude (pz) vs step number
      - Per-step reward vs step number
    """

    def __init__(self, out_dir: Path, plot_freq: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self._out_dir = Path(out_dir) / "traj"
        self._plot_freq = int(plot_freq)
        self._ep_count = 0
        self._reset()

    def _reset(self):
        self._pos: List[np.ndarray] = []   # (3,) each step
        self._rewards: List[float] = []
        self._target: Optional[np.ndarray] = None
        self._start: Optional[np.ndarray] = None
        self._term_reason: str = "unknown"

    def _on_step(self) -> bool:
        if not _MPL_OK:
            return True
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        x = info.get("x")
        if x is not None:
            pos = np.asarray(x[:3], dtype=np.float32)
            self._pos.append(pos)
            if self._start is None:
                self._start = pos.copy()
            target = info.get("target_xyz")
            if target is not None:
                self._target = np.asarray(target, dtype=np.float32)

        self._rewards.append(float(rewards[0]) if rewards else 0.0)
        self._term_reason = info.get("term_reason") or self._term_reason

        if dones[0]:
            self._ep_count += 1
            if self._ep_count % self._plot_freq == 0 and len(self._pos) > 1:
                self._save_plot()
            self._reset()

        return True

    def _save_plot(self):
        pos = np.array(self._pos)          # (N, 3) in NED
        rewards = np.array(self._rewards)  # (N,)
        target = self._target
        start = self._start
        reason = self._term_reason
        n = len(pos)

        # Marker + colour by outcome
        outcome_marker = {"success": ("✓", "green"), "timeout": ("⏱", "grey")}
        end_char, end_col = outcome_marker.get(reason, ("✗", "red"))
        title = f"ep {self._ep_count} | {reason} | R={rewards.sum():.1f}"

        colors = cm.plasma(np.linspace(0, 1, n))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(title, fontsize=11)

        # ── Plot 1: XY top-down ────────────────────────────────────────────
        ax = axes[0]
        for i in range(n - 1):
            ax.plot(pos[i:i+2, 1], pos[i:i+2, 0], color=colors[i], lw=1.5)
        if start is not None:
            ax.plot(start[1], start[0], "go", ms=8, label="start")
        if target is not None:
            ax.plot(target[1], target[0], "r*", ms=12, label="goal")
        ax.plot(pos[-1, 1], pos[-1, 0], marker="$" + end_char + "$",
                ms=14, color=end_col, label=reason)
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        ax.set_title("XY top-down")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # ── Plot 2: Altitude over time ─────────────────────────────────────
        ax = axes[1]
        ax.plot(pos[:, 2], color="steelblue", lw=1.5)   # pz (NED: negative = up)
        ax.axhline(-1.2, color="orange", ls="--", lw=1, label="alt target")
        ax.axhline(-1.95, color="red", ls=":", lw=1, label="ceiling")
        ax.axhline(-0.05, color="brown", ls=":", lw=1, label="ground")
        if target is not None:
            ax.axhline(target[2], color="green", ls="--", lw=1, label="goal z")
        ax.set_xlabel("step")
        ax.set_ylabel("pz NED (m)")
        ax.set_title("Altitude (pz)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ── Plot 3: Per-step reward ────────────────────────────────────────
        ax = axes[2]
        ax.plot(rewards, color="purple", lw=1, alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        cumulative = np.cumsum(rewards)
        ax2 = ax.twinx()
        ax2.plot(cumulative, color="darkorange", lw=1.5, alpha=0.8)
        ax2.set_ylabel("cumulative R", color="darkorange")
        ax.set_xlabel("step")
        ax.set_ylabel("step reward")
        ax.set_title("Reward")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save PNG
        self._out_dir.mkdir(parents=True, exist_ok=True)
        fpath = self._out_dir / f"ep{self._ep_count:05d}_t{self.num_timesteps}.png"
        fig.savefig(fpath, dpi=90, bbox_inches="tight")

        # Log to TensorBoard as image
        try:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
            buf.seek(0)
            import struct, zlib  # parse PNG to get HWC array via matplotlib
            img_arr = plt.imread(buf)  # (H, W, 4) RGBA float32
            img_chw = (img_arr[:, :, :3] * 255).astype(np.uint8).transpose(2, 0, 1)
            for fmt in getattr(self.logger, "output_formats", []):
                writer = getattr(fmt, "writer", None)
                if writer is not None:
                    writer.add_image("trajectory/episode", img_chw, self.num_timesteps)
                    break
        except Exception:
            pass

        plt.close(fig)


def build_eval_env(cfg: Dict[str, Any]):
    """Builds a separate FigsDroneEnv for evaluation rollouts.

    Same scene/sampler as train env, but wrapped in SB3's Monitor so episode
    returns get logged. Seeded with eval.seed for reproducible rollouts.
    """
    from nav_policy.rl.train.train_sac import _build_env  # local import to dodge cycles
    eval_cfg = dict(cfg)
    eval_overrides = cfg.get("eval", {}) or {}
    # Allow eval-specific sampler overrides
    if "sampler" in eval_overrides:
        eval_cfg["sampler"] = {**cfg.get("sampler", {}), **eval_overrides["sampler"]}
    env = _build_env(eval_cfg)
    if Monitor is not None:
        env = Monitor(env)
    return env


def build_callbacks(cfg: Dict[str, Any], out_dir: Path) -> Optional[Any]:
    """Assemble CallbackList from yaml flags.

    yaml schema:
        callbacks:
            eval:
                enabled: true
                freq: 5000
                n_episodes: 3
                seed: 42
            checkpoint:
                enabled: true
                freq: 5000
            reward_components:
                enabled: true
                log_freq: 200
            wandb:
                enabled: false
                project: vlead-sac
                run_name: null
                log_freq: 200
    """
    if CallbackList is None:
        return None

    cb_cfg = cfg.get("callbacks", {}) or {}
    cbs: List[Any] = []

    # Eval
    eval_cfg = cb_cfg.get("eval", {})
    if eval_cfg.get("enabled", False) and EvalCallback is not None:
        eval_env = build_eval_env(cfg)
        cbs.append(EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(out_dir / "best"),
            log_path=str(out_dir / "eval"),
            eval_freq=int(eval_cfg.get("freq", 5000)),
            n_eval_episodes=int(eval_cfg.get("n_episodes", 3)),
            deterministic=True,
            render=False,
        ))

    # Checkpointing
    ckpt_cfg = cb_cfg.get("checkpoint", {})
    if ckpt_cfg.get("enabled", True) and CheckpointCallback is not None:
        cbs.append(CheckpointCallback(
            save_freq=int(ckpt_cfg.get("freq", 5000)),
            save_path=str(out_dir / "ckpt"),
            name_prefix="sac",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ))

    # Per-component reward logging
    rc_cfg = cb_cfg.get("reward_components", {})
    if rc_cfg.get("enabled", True):
        cbs.append(RewardComponentsCallback(
            log_freq=int(rc_cfg.get("log_freq", 200)),
        ))

    # Trajectory plots
    traj_cfg = cb_cfg.get("trajectory", {})
    if traj_cfg.get("enabled", True) and _MPL_OK:
        cbs.append(TrajectoryPlotCallback(
            out_dir=out_dir,
            plot_freq=int(traj_cfg.get("plot_freq", 10)),
        ))

    # W&B
    wb_cfg = cb_cfg.get("wandb", {})
    if wb_cfg.get("enabled", False):
        try:
            import wandb
            wandb.init(
                project=wb_cfg.get("project", "vlead-sac"),
                name=wb_cfg.get("run_name"),
                config=cfg,
                dir=str(out_dir),
                sync_tensorboard=True,
            )
            cbs.append(WandbSyncCallback(log_freq=int(wb_cfg.get("log_freq", 200))))
        except ImportError:
            print("[callbacks] wandb requested but package not installed; skipping")

    if not cbs:
        return None
    return CallbackList(cbs)
