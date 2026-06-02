"""Build a policy module from a training/eval YAML config."""

from __future__ import annotations

from typing import Union

import torch.nn as nn

from nav_policy.model.rgb_da2_policy import RGBDA2VelocityPolicy
from nav_policy.model.rgb_velocity_policy import RGBVelocityPolicy


def build_model(cfg: dict) -> nn.Module:
    # Flow-matching checkpoints carry an "fm" config section and a model section
    # without an "arch"/"mlp_hidden" (the head is a conditional vector field, not an
    # MLP regressor). Build the FlowMatchingPolicy directly so SAC can fine-tune it.
    if "fm" in cfg:
        from nav_policy.model.flow_matching_policy import FlowMatchingPolicy
        mcfg = cfg["model"]
        fmcfg = cfg["fm"]
        wcfg = cfg["window"]
        return FlowMatchingPolicy(
            T=int(wcfg["T"]),
            H=int(wcfg["H"]),
            cmd_dim=int(mcfg["cmd_dim"]),
            gru_hidden=int(mcfg["gru_hidden"]),
            gru_layers=int(mcfg.get("gru_layers", 1)),
            goal_emb_dim=int(mcfg.get("goal_emb_dim", 32)),
            goal_input_dim=int(mcfg.get("goal_input_dim", 3)),
            freeze_stem_and_layer1=bool(mcfg.get("freeze_stem_and_layer1", True)),
            time_emb_dim=int(fmcfg.get("time_emb_dim", 64)),
            vf_hidden=tuple(fmcfg.get("vf_hidden", (512, 512, 512))),
            vf_use_skip=bool(fmcfg.get("vf_use_skip", True)),
            n_val_steps=int(fmcfg.get("n_val_steps", 4)),
        )

    mcfg = cfg["model"]
    arch = str(mcfg.get("arch", "rgb_resnet18"))
    common = dict(
        T=int(cfg["window"]["T"]),
        H=int(cfg["window"]["H"]),
        cmd_dim=int(mcfg["cmd_dim"]),
        gru_hidden=int(mcfg["gru_hidden"]),
        gru_layers=int(mcfg["gru_layers"]),
        mlp_hidden=tuple(mcfg["mlp_hidden"]),
        mlp_dropout=float(mcfg.get("mlp_dropout", 0.1)),
        goal_emb_dim=int(mcfg.get("goal_emb_dim", 32)),
        goal_input_dim=int(mcfg.get("goal_input_dim", 2)),
        freeze_stem_and_layer1=bool(mcfg.get("freeze_stem_and_layer1", True)),
    )

    if arch == "rgb_resnet18":
        return RGBVelocityPolicy(**common)

    if arch == "rgb_da2_crossattn_v1":
        return RGBDA2VelocityPolicy(
            **common,
            fusion="crossattn",
            depth_feat_dim=int(mcfg.get("depth_feat_dim", 256)),
            cross_attn_heads=int(mcfg.get("cross_attn_heads", 4)),
        )

    if arch == "rgb_da2_concat_v1":
        return RGBDA2VelocityPolicy(
            **common,
            fusion="concat",
            depth_feat_dim=int(mcfg.get("depth_feat_dim", 256)),
        )

    raise ValueError(
        f"unknown model.arch={arch!r}; expected rgb_resnet18, "
        "rgb_da2_crossattn_v1, or rgb_da2_concat_v1"
    )


def model_uses_depth(cfg: dict) -> bool:
    if "fm" in cfg:  # flow-matching policy is RGB-only
        return False
    arch = str(cfg.get("model", {}).get("arch", "rgb_resnet18"))
    return arch.startswith("rgb_da2_")
