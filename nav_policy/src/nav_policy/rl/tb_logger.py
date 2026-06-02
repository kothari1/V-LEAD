"""Thin TensorBoard writer wrapper for RL training.

Keeps train_rl.py free of repetitive `writer.add_scalar` lines and provides
a no-op fallback so disabling TB doesn't require sprinkling `if writer:`
checks at every call site.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class TBLogger:
    """SummaryWriter wrapper. enabled=False -> no-op methods."""

    def __init__(self, log_dir: Path, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.log_dir = Path(log_dir)
        self._writer = None
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            print(f"[tb] tensorboard import failed ({e}); disabling TB logging")
            self.enabled = False
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is None:
            return
        try:
            self._writer.add_scalar(tag, float(value), int(step))
        except Exception:
            pass

    def log_dict(self, prefix: str, scalars: Mapping[str, float], step: int) -> None:
        """Bulk-log a dict of scalars under a shared prefix.

        Example: log_dict("reward", {"progress": 1.2, "bbox": -5.0}, step=42)
        writes tags ``reward/progress`` and ``reward/bbox`` at step 42.
        """
        if self._writer is None or not scalars:
            return
        for key, val in scalars.items():
            self.log_scalar(f"{prefix}/{key}" if prefix else str(key), val, step)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        if self._writer is None:
            return
        try:
            self._writer.add_text(tag, str(text), int(step))
        except Exception:
            pass

    def flush(self) -> None:
        if self._writer is None:
            return
        self._writer.flush()

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.flush()
        self._writer.close()
        self._writer = None
