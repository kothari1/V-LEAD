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
