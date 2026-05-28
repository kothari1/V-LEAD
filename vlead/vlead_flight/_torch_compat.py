"""torch>=2.6 compatibility shim for loading nerfstudio/gsplat checkpoints.

PyTorch 2.6 flipped `torch.load`'s `weights_only` default from False to True.
nerfstudio's gsplat checkpoints contain numpy scalars + dtypes inside their
config dumps and fail the new safe-unpickle path. Both gemsplat ckpts and
SINGER recorder ckpts are trusted local artifacts in this repo, so we either
(a) allowlist the numpy types they need, or (b) fall back to weights_only=False.

`enable_legacy_torch_load()` does (a) first and then patches torch.load to
default to weights_only=False as a belt-and-suspenders fallback for older
SINGER recordings. Call once at process start, before importing FiGS or any
module that triggers nerfstudio's eval_setup.
"""
from __future__ import annotations

import functools

import numpy as np
import torch


_PATCHED = False


def enable_legacy_torch_load() -> None:
    """Idempotent. Safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from torch.serialization import add_safe_globals
        # Numpy scalars + dtype objects nerfstudio's config dumps contain.
        safe = [
            np.core.multiarray.scalar,
            np.dtype,
            np.ndarray,
        ]
        # numpy scalar dtypes
        for name in (
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
            "float16", "float32", "float64",
            "bool_",
        ):
            t = getattr(np, name, None)
            if t is not None:
                safe.append(t)
        # numpy dtype metaclass entries
        for dt_name in ("Int64DType", "Float64DType", "Float32DType", "BoolDType"):
            dt = getattr(getattr(np, "dtypes", object()), dt_name, None)
            if dt is not None:
                safe.append(dt)
        add_safe_globals(safe)
    except Exception:
        # add_safe_globals not available or numpy layout differs; fall through.
        pass

    # Belt-and-suspenders: default weights_only=False unless caller specifies.
    _orig_load = torch.load

    @functools.wraps(_orig_load)
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load
    _PATCHED = True
