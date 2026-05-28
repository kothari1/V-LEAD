"""SB3 features extractor wrapping the BC encoder.

Stable-Baselines3 SAC/PPO with MultiInputPolicy expects a Dict obs and a
BaseFeaturesExtractor that turns the dict into a flat feature tensor. The
extractor here reuses `RGBVelocityPolicy.encode()` so the visual+goal encoder
is identical to the BC model — making BC -> RL warm-start a state_dict copy.

Obs dict from FigsDroneEnv:
    rgb:  uint8 [T, 3, H, W]
    goal: float32 [goal_input_dim]   (3 or 4)

The env stores uint8 frames to keep the SAC replay buffer small. We cast
to float and apply ImageNet normalization here, inside the GPU forward
graph, so no per-sample CPU work is needed during training.
"""
from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import torch
import torch.nn as nn

from nav_policy.model.rgb_velocity_policy import RGBVelocityPolicy


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception:  # pragma: no cover - SB3 is an optional dep at import time
    BaseFeaturesExtractor = nn.Module  # type: ignore


class BCEncoderFeatureExtractor(BaseFeaturesExtractor):  # type: ignore[misc]
    """Encoder shared between SAC actor and twin Q-nets.

    Args:
        observation_space: Gymnasium Dict space (must contain rgb + goal).
        T:                 Frame window. Must match env's frame_window.
        H_pred:            BC head horizon (unused for RL; needed only so the
                           wrapped RGBVelocityPolicy constructs cleanly when
                           we load a BC checkpoint into it).
        cmd_dim:           BC head action dim (4 for warm-start compatibility).
        gru_hidden:        Must match BC checkpoint.
        gru_layers:        Must match BC checkpoint.
        goal_emb_dim:      Must match BC checkpoint.
        freeze_visual:     Freeze the ResNet backbone during RL (recommended
                           early; unfreeze later once Q-nets are warm).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        T: int = 4,
        H_pred: int = 10,
        cmd_dim: int = 4,
        gru_hidden: int = 256,
        gru_layers: int = 1,
        mlp_hidden: Sequence[int] = (256, 128),
        goal_emb_dim: int = 32,
        freeze_visual: bool = True,
    ) -> None:
        feature_dim = gru_hidden + goal_emb_dim
        super().__init__(observation_space, features_dim=feature_dim)

        goal_space = observation_space["goal"]
        if goal_space.shape[0] == 4:
            goal_input_dim = 3
            self._drop_goal_z = True
        elif goal_space.shape[0] in (2, 3):
            goal_input_dim = int(goal_space.shape[0])
            self._drop_goal_z = False
        else:
            raise ValueError(
                f"goal obs dim must be 2, 3, or 4; got {goal_space.shape[0]}"
            )

        # Pre-build the inner BC policy so weights can be loaded 1:1 from a BC ckpt.
        self._policy = RGBVelocityPolicy(
            T=T, H=H_pred, cmd_dim=cmd_dim,
            gru_hidden=gru_hidden, gru_layers=gru_layers,
            mlp_hidden=tuple(mlp_hidden),
            mlp_dropout=0.0,                  # no dropout during RL rollouts
            goal_emb_dim=goal_emb_dim,
            goal_input_dim=goal_input_dim,
            freeze_stem_and_layer1=True,
        )

        if feature_dim != self._policy.feature_dim:
            raise ValueError(
                "feature_dim mismatch: extractor was sized for "
                f"{feature_dim}, BC policy emits {self._policy.feature_dim}"
            )

        if freeze_visual:
            for p in self._policy.visual.parameters():
                p.requires_grad = False

        self.register_buffer(
            "_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False
        )

    @property
    def inner_policy(self) -> RGBVelocityPolicy:
        """Exposed for BC warm-start to load state_dict into."""
        return self._policy

    def forward(self, observations: dict) -> torch.Tensor:
        rgb = observations["rgb"]      # [B, T, 3, H, W] uint8 or float
        goal = observations["goal"]    # [B, goal_dim] float32

        if rgb.dtype != torch.float32:
            rgb = rgb.float()
        rgb = rgb / 255.0
        rgb = (rgb - self._mean) / self._std

        if self._drop_goal_z:
            # Env emits [hx, hy, hz, d/scale]; encoder wants [hx, hy, d/scale].
            goal = torch.cat([goal[:, :2], goal[:, 3:4]], dim=-1)

        return self._policy.encode(rgb, goal)
