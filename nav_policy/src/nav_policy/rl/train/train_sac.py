"""SAC trainer for the V-LEAD pilot network.

Pipeline:
    1. Load YAML config.
    2. Build FigsDroneEnv (Gymnasium).
    3. Build SAC with MultiInputPolicy + BCEncoderFeatureExtractor.
    4. (Optional) Warm-start the feature extractor from a BC checkpoint.
    5. model.learn(total_timesteps=...).
    6. Save model + buffer + final eval rollout.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

# torch>=2.6 compatibility for nerfstudio/gsplat checkpoints. Must run before
# any module that imports FiGS / triggers nerfstudio.eval_setup.
from vlead_flight._torch_compat import enable_legacy_torch_load

enable_legacy_torch_load()

from nav_policy.rl.model.feature_extractor import BCEncoderFeatureExtractor
from nav_policy.rl.train.callbacks import build_callbacks
from nav_policy.rl.warm_start.bc_to_rl import (
    load_bc_into_feature_extractor,
    load_bc_into_sac_actor,
)


def _build_env(cfg: Dict[str, Any]):
    """Instantiate FigsDroneEnv from yaml-config dicts."""
    from vlead_flight.env import (
        EpisodeSampler,
        FigsDroneEnv,
        RewardConfig,
        TerminationConfig,
    )

    env_cfg = cfg["env"]
    sampler = EpisodeSampler(**cfg.get("sampler", {}))
    reward_cfg = RewardConfig(**cfg.get("reward", {}))
    term_cfg = TerminationConfig(**cfg.get("termination", {}))

    env = FigsDroneEnv(
        scene_name=env_cfg["scene_name"],
        rollout_name=env_cfg.get("rollout_name", "baseline"),
        frame_name=env_cfg.get("frame_name", "carl"),
        hz=env_cfg.get("hz", 20),
        frame_window=env_cfg.get("frame_window", 4),
        img_resolution=tuple(env_cfg.get("img_resolution", (224, 224))),
        goal_distance_scale=env_cfg.get("goal_distance_scale", 5.0),
        goal_input_dim=env_cfg.get("goal_input_dim", 4),
        action_low=tuple(env_cfg.get("action_low", (-3.0, -3.0, -1.5, -1.5))),
        action_high=tuple(env_cfg.get("action_high", (3.0, 3.0, 1.5, 1.5))),
        sampler=sampler,
        reward_cfg=reward_cfg,
        term_cfg=term_cfg,
    )
    return env


def _build_sac(env, cfg: Dict[str, Any], device: str):
    from stable_baselines3 import SAC

    model_cfg = cfg.get("model", {})
    sac_cfg = cfg.get("sac", {})

    policy_kwargs = dict(
        features_extractor_class=BCEncoderFeatureExtractor,
        features_extractor_kwargs=dict(
            T=model_cfg.get("T", 4),
            H_pred=model_cfg.get("H_pred", 10),
            cmd_dim=model_cfg.get("cmd_dim", 4),
            gru_hidden=model_cfg.get("gru_hidden", 256),
            gru_layers=model_cfg.get("gru_layers", 1),
            mlp_hidden=tuple(model_cfg.get("mlp_hidden", (256, 128))),
            goal_emb_dim=model_cfg.get("goal_emb_dim", 32),
            freeze_visual=model_cfg.get("freeze_visual", True),
        ),
        net_arch=dict(
            pi=list(model_cfg.get("actor_mlp", [256, 128])),
            qf=list(model_cfg.get("critic_mlp", [256, 128])),
        ),
        share_features_extractor=True,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=sac_cfg.get("learning_rate", 3.0e-4),
        buffer_size=sac_cfg.get("buffer_size", 100_000),
        learning_starts=sac_cfg.get("learning_starts", 1_000),
        batch_size=sac_cfg.get("batch_size", 64),
        tau=sac_cfg.get("tau", 0.005),
        gamma=sac_cfg.get("gamma", 0.99),
        train_freq=sac_cfg.get("train_freq", 1),
        gradient_steps=sac_cfg.get("gradient_steps", 1),
        ent_coef=sac_cfg.get("ent_coef", "auto"),
        target_update_interval=sac_cfg.get("target_update_interval", 1),
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=str(Path(cfg["output_dir"]) / "tb"),
        seed=cfg.get("seed", 0),
    )
    return model


def _find_bc_extractors(policy) -> list:
    """Collect every BCEncoderFeatureExtractor instance under the SAC policy.

    SB3 SAC quirk: policy.features_extractor can be None even with
    share_features_extractor=True; the live extractor(s) sit on
    policy.actor and policy.critic. When share=True, all three point to the
    same Module; when share=False, each is independent and we must warm-start
    every copy.
    """
    extractors: list = []
    seen: set = set()
    candidates = [
        getattr(policy, "features_extractor", None),
        getattr(getattr(policy, "actor", None), "features_extractor", None),
        getattr(getattr(policy, "critic", None), "features_extractor", None),
        getattr(getattr(policy, "critic_target", None), "features_extractor", None),
    ]
    for c in candidates:
        if isinstance(c, BCEncoderFeatureExtractor) and id(c) not in seen:
            extractors.append(c)
            seen.add(id(c))
    return extractors


def _warm_start(model, cfg: Dict[str, Any]) -> None:
    ws_cfg = cfg.get("warm_start", {}) or {}
    bc_ckpt = ws_cfg.get("bc_checkpoint")
    if not bc_ckpt:
        print("[sac] no BC checkpoint provided; training feature extractor from scratch")
        return

    # 1. Encoder warm-start (visual + goal embedding).
    extractors = _find_bc_extractors(model.policy)
    if not extractors:
        raise RuntimeError(
            "warm-start could not locate any BCEncoderFeatureExtractor on the SAC "
            "policy (policy.features_extractor / policy.actor.features_extractor / "
            "policy.critic.features_extractor were all unsuitable). Check "
            "policy_kwargs.features_extractor_class is wired."
        )
    for ext in extractors:
        info = load_bc_into_feature_extractor(ext, bc_ckpt)
        head_dropped = len([k for k in info.get('unexpected_keys', []) if k.startswith('head.')])
        print(
            f"[sac] warm-started encoder ({type(ext).__name__} id={id(ext)}) "
            f"from {bc_ckpt}; head keys dropped: {head_dropped}"
        )
    print(f"[sac] warm-started {len(extractors)} extractor instance(s)")

    # 2. Optional actor head warm-start (BC MLPHead -> SAC actor latent_pi + mu).
    if ws_cfg.get("init_actor_from_bc_head", False):
        action_dim = int(model.action_space.shape[0])
        try:
            stats = load_bc_into_sac_actor(
                model.policy.actor, bc_ckpt, action_dim=action_dim,
            )
            print(
                "[sac] warm-started actor latent_pi + mu from BC head: "
                f"{stats['latent_pi_layers_copied']} hidden layers + "
                f"{stats['mu_rows_copied']} mu rows (BC head output dim "
                f"{stats['bc_head_output_dim']})"
            )
        except Exception as e:
            print(f"[sac] WARNING actor warm-start failed: {e}")
            print("[sac] continuing with random actor head; encoder warm-start retained")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override cfg.output_dir.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    cfg.setdefault("output_dir", "data/rl_runs/sac_default")
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps
    cfg.setdefault("total_timesteps", 50_000)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[sac] building env (scene={cfg['env']['scene_name']})", flush=True)
    env = _build_env(cfg)
    print(f"[sac] env ready. obs={env.observation_space} act={env.action_space}", flush=True)

    print(f"[sac] building SAC (buffer={cfg.get('sac', {}).get('buffer_size')}, device={device})", flush=True)
    model = _build_sac(env, cfg, device=device)
    print("[sac] SAC ready, replay buffer allocated", flush=True)

    _warm_start(model, cfg)

    out_dir = Path(cfg["output_dir"])
    callbacks = build_callbacks(cfg, out_dir)
    if callbacks is not None:
        print(f"[sac] callbacks active: {[type(c).__name__ for c in callbacks.callbacks]}", flush=True)

    print(f"[sac] starting learn() for {cfg['total_timesteps']} timesteps", flush=True)
    model.learn(
        total_timesteps=int(cfg["total_timesteps"]),
        log_interval=cfg.get("log_interval", 1),
        progress_bar=True,
        callback=callbacks,
    )

    model.save(out_dir / "sac_final")
    print(f"[sac] saved final model to {out_dir/'sac_final'}.zip")


if __name__ == "__main__":
    main()
