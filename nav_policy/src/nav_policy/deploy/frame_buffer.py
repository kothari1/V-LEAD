"""
Rolling frame buffer for online inference.

Holds the last T resized + ImageNet-normalized RGB tensors. When the buffer
is partially full it replicates the most recent frame to fill the missing
positions; this is preferable to zero-padding because ImageNet statistics
make zeros far from any real frame in feature space.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from nav_policy.data.normalization import IMAGENET_MEAN, IMAGENET_STD, imagenet_normalize

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


def _resize_uint8(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 uint8, got {frame.shape}")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if _HAVE_CV2:
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    from PIL import Image
    return np.asarray(
        Image.fromarray(frame, mode="RGB").resize((size, size), Image.BILINEAR)
    )


class FrameBuffer:
    """Maintain a rolling window of T preprocessed frames."""

    def __init__(self,
                 T: int = 4,
                 image_size: int = 224,
                 mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD,
                 device: Optional[torch.device] = None) -> None:
        self.T = T
        self.image_size = image_size
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # buffer stores already-resized uint8 tensors in CHW order.
        self._buf_uint8: list[torch.Tensor] = []

    def reset(self) -> None:
        self._buf_uint8.clear()

    def push(self, frame: np.ndarray) -> None:
        """`frame` is an HxWx3 uint8 RGB image from the FiGS renderer."""
        resized = _resize_uint8(frame, self.image_size)
        chw = torch.from_numpy(resized).permute(2, 0, 1).contiguous()   # [3, S, S] uint8
        if len(self._buf_uint8) >= self.T:
            self._buf_uint8.pop(0)
        self._buf_uint8.append(chw)

    def is_ready(self) -> bool:
        return len(self._buf_uint8) > 0

    def tensor(self) -> torch.Tensor:
        """Return a [1, T, 3, S, S] float32 ImageNet-normalized tensor on self.device."""
        if not self._buf_uint8:
            raise RuntimeError("FrameBuffer is empty; push at least one frame before calling tensor()")
        n = len(self._buf_uint8)
        if n < self.T:
            # Pad at the FRONT with copies of the oldest frame (i.e. the warm-up uses
            # the earliest available frame for the missing past positions). This avoids
            # spurious motion features from zero-frames or duplicated newest frames.
            pad = [self._buf_uint8[0]] * (self.T - n)
            seq = pad + self._buf_uint8
        else:
            seq = self._buf_uint8
        stacked = torch.stack(seq, dim=0)                                # [T, 3, S, S] uint8
        normed = imagenet_normalize(stacked, mean=self.mean, std=self.std)
        return normed.unsqueeze(0).to(self.device, non_blocking=True)    # [1, T, 3, S, S]
