"""Reusable observation helpers shared by deployment pilot and RL env.

Extracted from VLeadPilot so the Gymnasium env wrapper can build identical
RGB preprocessing, frame buffering, and goal computation without duplicating
the pilot's state machine.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ImageNet stats (consistent with SINGER preprocessing).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def compute_goal(xcr: np.ndarray, target_xyz: np.ndarray) -> Tuple[np.ndarray, float]:
    """Returns (unit heading vector [3], distance scalar) from state position to target."""
    delta = np.asarray(target_xyz, dtype=np.float64).reshape(3) - np.asarray(xcr[0:3], dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    heading = delta / (dist + 1e-8)
    return heading, dist


def imagenet_norm_buffers(device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(3, 1, 1)
    return mean, std


def preprocess_rgb(
    rgb_np: np.ndarray,
    *,
    device,
    dtype,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """[H, W, 3] uint8 → [3, H_out, W_out] ImageNet-normalized tensor on device."""
    x = torch.from_numpy(np.ascontiguousarray(rgb_np)).to(device, non_blocking=True)
    x = x.permute(2, 0, 1).to(dtype) / 255.0
    if x.shape[-2:] != target_hw:
        x = F.interpolate(
            x.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False,
        ).squeeze(0)
    return (x - mean) / std


def preprocess_depth(
    depth_np: np.ndarray,
    *,
    device,
    dtype,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(depth_np)).to(device, non_blocking=True).to(dtype)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.shape[-2:] != target_hw:
        x = F.interpolate(
            x.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False,
        ).squeeze(0)
    return x


class FrameBuffer:
    """Circular T-frame buffer for spatiotemporal network inputs.

    On the first push after reset, every slot is filled with the same frame so
    the network sees a valid full sequence from t=0.
    """

    def __init__(
        self,
        T: int,
        channels: int,
        hw: Tuple[int, int],
        *,
        device,
        dtype,
    ) -> None:
        self.T = T
        self._buf = torch.zeros(T, channels, hw[0], hw[1], device=device, dtype=dtype)
        self._idx = 0
        self._filled = False

    def reset(self) -> None:
        self._buf.zero_()
        self._idx = 0
        self._filled = False

    def push(self, frame: torch.Tensor) -> None:
        if not self._filled:
            for k in range(self.T):
                self._buf[k] = frame
            self._idx = 1 % self.T
            self._filled = True
        else:
            self._buf[self._idx] = frame
            self._idx = (self._idx + 1) % self.T

    def get(self) -> torch.Tensor:
        """Returns [T, C, H, W] in chronological order (oldest → newest)."""
        if not self._filled:
            # Caller should push at least once before get(); return zeros otherwise.
            return self._buf.clone()
        order = [(self._idx + k) % self.T for k in range(self.T)]
        return self._buf[order]

    def get_batched(self) -> torch.Tensor:
        """Returns [1, T, C, H, W]."""
        return self.get().unsqueeze(0)
