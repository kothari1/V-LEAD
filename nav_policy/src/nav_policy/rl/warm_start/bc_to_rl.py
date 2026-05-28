"""BC checkpoint -> RL feature-extractor warm-start.

The BC trainer saves checkpoints as:
    {"model": <RGBVelocityPolicy state_dict>, "config": {...},
     "epoch": int, "val_loss": float, "val_mse_overall": float, "stats": {...}}

This loader copies the BC `RGBVelocityPolicy` weights into the
BCEncoderFeatureExtractor's inner policy so the visual+goal encoder used by
SAC starts from a trained checkpoint. SAC's actor / Q heads still train from
scratch — full actor warm-start requires custom SAC policy subclassing and
is deferred to v2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

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

    # Filter out the head: BC's head is wider (H * cmd_dim) and unused for RL,
    # so non-matching head keys are expected and not errors.
    head_prefix = "head."
    real_missing = [k for k in missing if not k.startswith(head_prefix)]
    real_unexpected = [k for k in unexpected if not k.startswith(head_prefix)]
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
