"""BC (and FM) checkpoint -> RL warm-start helpers.

BC checkpoint format (produced by train_bc.py):
    {"model": <RGBVelocityPolicy state_dict>, "config": {...},
     "epoch": int, "val_loss": float, "val_mse_overall": float, "stats": {...}}

FM checkpoint format (produced by train_fm.py — identical structure):
    {"model": <FlowMatchingPolicy state_dict>, "config": {...},
     "epoch": int, "val_loss": float, "val_mse_overall": float, "stats": {...}}

Both share the same encoder keys (visual.*, gru.*, gru_norm.*, goal_embed.*),
so load_bc_into_feature_extractor() works for EITHER checkpoint type.

For FM checkpoints, the state dict contains extra vector_field.* keys that are
not present in RGBVelocityPolicy (the inner_policy type used by BCEncoderFeatureExtractor).
These appear as "unexpected_keys" and are silently ignored — they do not trigger
an error because only head.* and vector_field.* keys are non-encoder.

Two warm-start paths:
- load_bc_into_feature_extractor: copy visual+goal encoder from BC or FM
  checkpoint into the SB3 feature extractor (always safe for both types).
- load_bc_into_sac_actor: copy BC's MLP head into SAC actor.latent_pi + mu
  (BC checkpoints only — FM head is a conditional vector field, not a mean MLP).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from nav_policy.model.rgb_velocity_policy import RGBVelocityPolicy
from nav_policy.rl.model.feature_extractor import BCEncoderFeatureExtractor


def load_bc_into_feature_extractor(
    extractor: BCEncoderFeatureExtractor,
    bc_checkpoint_path: str | Path,
    *,
    strict: bool = False,
    map_location: str = "cpu",
) -> Dict[str, list]:
    """Loads BC weights into the extractor's inner RGBVelocityPolicy.

    Args:
        extractor:           Built BCEncoderFeatureExtractor instance (already
                             constructed with the right T/gru/goal dims).
        bc_checkpoint_path:  Path to a bc_best.pt or bc_latest.pt file.
        strict:              If True, every BC weight must match (including the
                             head). If False (default), the BC head weights may
                             be silently dropped since the RL actor builds its
                             own action head; only the encoder is required.
        map_location:        torch.load map_location.

    Returns:
        Dict with `missing_keys` and `unexpected_keys` from torch.load_state_dict.
    """
    ckpt = torch.load(Path(bc_checkpoint_path), map_location=map_location)
    if "model" not in ckpt:
        raise ValueError(
            f"BC checkpoint at {bc_checkpoint_path} has no 'model' key; "
            f"keys={list(ckpt.keys())}"
        )
    sd = ckpt["model"]

    missing, unexpected = extractor.inner_policy.load_state_dict(sd, strict=strict)

    if strict:
        return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}

    # Filter out non-encoder keys that are expected to differ between BC and FM:
    #   head.*          - BC MLP head (H*cmd_dim output); unused in RL feature extractor
    #   vector_field.*  - FM conditional vector field; unused in RL feature extractor
    non_encoder = ("head.", "vector_field.")
    real_missing = [k for k in missing if not any(k.startswith(p) for p in non_encoder)]
    real_unexpected = [k for k in unexpected if not any(k.startswith(p) for p in non_encoder)]
    if real_missing:
        raise RuntimeError(
            f"BC -> RL load missing required encoder keys: {real_missing}"
        )
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "encoder_missing_keys": real_missing,
        "encoder_unexpected_keys": real_unexpected,
    }


def load_bc_stats(bc_checkpoint_path: str | Path, map_location: str = "cpu") -> Optional[Dict]:
    """Returns the CommandStats dict from a BC ckpt, if present, else None."""
    ckpt = torch.load(Path(bc_checkpoint_path), map_location=map_location)
    stats = ckpt.get("stats")
    return stats


def _build_bc_model_from_ckpt(
    bc_checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
) -> RGBVelocityPolicy:
    """Reconstructs the BC RGBVelocityPolicy from a checkpoint (config + weights)."""
    ckpt = torch.load(Path(bc_checkpoint_path), map_location=map_location)
    if "model" not in ckpt:
        raise ValueError(f"BC checkpoint at {bc_checkpoint_path} has no 'model' key")
    sd = ckpt["model"]
    cfg = ckpt.get("config", {}) or {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    bc = RGBVelocityPolicy(
        T=int(model_cfg.get("T", 4)),
        H=int(model_cfg.get("H", 10)),
        cmd_dim=int(model_cfg.get("cmd_dim", 4)),
        gru_hidden=int(model_cfg.get("gru_hidden", 256)),
        gru_layers=int(model_cfg.get("gru_layers", 1)),
        mlp_hidden=tuple(model_cfg.get("mlp_hidden", (256, 128))),
        mlp_dropout=float(model_cfg.get("mlp_dropout", 0.1)),
        goal_emb_dim=int(model_cfg.get("goal_emb_dim", 32)),
        goal_input_dim=int(model_cfg.get("goal_input_dim", 3)),
        freeze_stem_and_layer1=True,
    )
    bc.load_state_dict(sd, strict=True)
    return bc


def load_bc_into_sac_actor(
    sac_actor: nn.Module,
    bc_checkpoint_path: str | Path,
    *,
    action_dim: int = 4,
    map_location: str = "cpu",
) -> Dict[str, int]:
    """Initialize a SAC `Actor`'s latent_pi + mu from BC head's MLP layers.

    Mapping (must hold for the copy to be valid):
        net_arch.pi == BC config's mlp_hidden  (default [256, 128])
        SAC actor.latent_pi has the same Linear sequence as BC.head.net
        (Dropout layers in BC.head.net are skipped — they carry no weights)
        SAC actor.mu = Linear(net_arch.pi[-1], action_dim)
        BC's final Linear emits H*cmd_dim values; row layout is
        [vx_0, vy_0, vz_0, psi_dot_0, vx_1, ...] -> first `cmd_dim` rows = t=0.

    Args:
        sac_actor:           model.policy.actor (SB3 SAC Actor module).
        bc_checkpoint_path:  Path to a BC checkpoint with config + weights.
        action_dim:          SAC action space dim; must equal BC's cmd_dim.

    Returns:
        Counts of layers copied.
    """
    bc = _build_bc_model_from_ckpt(bc_checkpoint_path, map_location=map_location)

    bc_linears = [m for m in bc.head.net if isinstance(m, nn.Linear)]
    if not bc_linears:
        raise RuntimeError("No Linear layers found in BC head.net")

    sac_latent_pi = getattr(sac_actor, "latent_pi", None)
    sac_mu = getattr(sac_actor, "mu", None)
    if sac_latent_pi is None or sac_mu is None:
        raise RuntimeError(
            "SAC actor has no latent_pi/mu — SB3 API may have changed; "
            "inspect type(sac_actor) and adapt the warm-start mapping."
        )
    sac_linears = [m for m in sac_latent_pi if isinstance(m, nn.Linear)]
    if len(bc_linears) - 1 != len(sac_linears):
        raise RuntimeError(
            f"net_arch mismatch: BC head has {len(bc_linears)} Linear layers "
            f"({len(bc_linears) - 1} hidden + 1 output), but SAC actor.latent_pi "
            f"has {len(sac_linears)} Linear layers. Set policy_kwargs.net_arch.pi "
            f"to match BC's mlp_hidden."
        )

    # Shape sanity for output mu (first action_dim rows of BC final linear).
    bc_final = bc_linears[-1]
    if bc_final.weight.shape[0] < action_dim:
        raise RuntimeError(
            f"BC head outputs {bc_final.weight.shape[0]} dims; need >= "
            f"{action_dim} to slice an action-shaped tensor."
        )
    if sac_mu.weight.shape != (action_dim, bc_final.weight.shape[1]):
        raise RuntimeError(
            f"sac_actor.mu shape {tuple(sac_mu.weight.shape)} does not match "
            f"(action_dim={action_dim}, BC final in_features="
            f"{bc_final.weight.shape[1]}). Check net_arch.pi[-1]."
        )

    with torch.no_grad():
        for sac_l, bc_l in zip(sac_linears, bc_linears[:-1]):
            if sac_l.weight.shape != bc_l.weight.shape:
                raise RuntimeError(
                    f"Layer shape mismatch: SAC {tuple(sac_l.weight.shape)} vs "
                    f"BC {tuple(bc_l.weight.shape)}. Adjust net_arch.pi."
                )
            sac_l.weight.copy_(bc_l.weight)
            sac_l.bias.copy_(bc_l.bias)
        sac_mu.weight.copy_(bc_final.weight[:action_dim])
        sac_mu.bias.copy_(bc_final.bias[:action_dim])

    return {
        "latent_pi_layers_copied": len(sac_linears),
        "mu_rows_copied": int(action_dim),
        "bc_head_output_dim": int(bc_final.weight.shape[0]),
    }
