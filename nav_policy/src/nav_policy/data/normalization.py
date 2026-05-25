"""Shared normalization utilities (image and command-space)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(rgb_uint8: torch.Tensor,
                       mean: Sequence[float] = IMAGENET_MEAN,
                       std: Sequence[float] = IMAGENET_STD) -> torch.Tensor:
    """
    Convert a uint8 RGB tensor of shape [..., 3, H, W] to a float ImageNet-
    normalized tensor of the same shape.
    """
    if rgb_uint8.dtype != torch.uint8:
        raise TypeError(f"expected uint8 input, got {rgb_uint8.dtype}")
    x = rgb_uint8.to(torch.float32).div_(255.0)
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - m) / s


@dataclass
class CommandStats:
    """Per-component mean/std for [vx, vy, vz, psi_dot]."""
    mean: np.ndarray  # shape (4,)
    std: np.ndarray   # shape (4,)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "CommandStats":
        return cls(mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"], dtype=np.float32))

    @classmethod
    def fit(cls, samples: np.ndarray, eps: float = 1e-6) -> "CommandStats":
        """`samples`: array of shape [N, 4] with raw command values."""
        if samples.ndim != 2 or samples.shape[1] != 4:
            raise ValueError(f"expected [N, 4], got {samples.shape}")
        mean = samples.mean(axis=0).astype(np.float32)
        std = samples.std(axis=0).astype(np.float32)
        std = np.maximum(std, eps).astype(np.float32)
        return cls(mean=mean, std=std)

    def standardize_np(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - m) / s

    def destandardize(self, z: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=z.dtype, device=z.device)
        s = torch.as_tensor(self.std, dtype=z.dtype, device=z.device)
        return z * s + m
