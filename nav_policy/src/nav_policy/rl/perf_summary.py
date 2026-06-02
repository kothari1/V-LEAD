"""Startup model + memory + inference-latency summary for RL training.

One-shot diagnostics meant to be printed once at the top of train_rl.train()
and persisted as JSON next to the run's checkpoints. Useful for documenting
"what kind of model are we training" in a single artifact per run.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from nav_policy.rl.stochastic_policy import StochasticVelocityPolicy


def _param_breakdown(policy: nn.Module) -> Dict[str, Any]:
    """Trainable / frozen param counts overall + per top-level submodule."""
    by_prefix = defaultdict(lambda: {"trainable": 0, "frozen": 0})
    total_trainable = total_frozen = 0
    total_bytes = 0
    for name, p in policy.named_parameters():
        prefix = name.split(".", 1)[0] if "." in name else name
        bucket = by_prefix[prefix]
        if p.requires_grad:
            bucket["trainable"] += p.numel()
            total_trainable += p.numel()
        else:
            bucket["frozen"] += p.numel()
            total_frozen += p.numel()
        total_bytes += p.numel() * p.element_size()
    return {
        "total_trainable": int(total_trainable),
        "total_frozen": int(total_frozen),
        "total_params": int(total_trainable + total_frozen),
        "approx_bytes": int(total_bytes),
        "approx_mb": float(total_bytes) / (1024.0 * 1024.0),
        "per_module": {k: dict(v) for k, v in by_prefix.items()},
    }


def _replay_buffer_footprint(*,
                             capacity: int,
                             frame_window: int,
                             image_size: int,
                             use_depth: bool,
                             fp16: bool) -> Dict[str, Any]:
    """Theoretical per-transition + total replay-buffer RAM."""
    bytes_per_value = 2 if fp16 else 4
    rgb_bytes = frame_window * 3 * image_size * image_size * bytes_per_value
    # Each transition stores rgb + next_rgb + goal + next_goal + small extras.
    goal_bytes = 4 * 4  # 4 float scalars
    extras = 4 * 4  # action_z (4 floats) + reward + done + log_prob
    depth_bytes = (frame_window * 1 * image_size * image_size * bytes_per_value) if use_depth else 0
    per_tr = 2 * rgb_bytes + 2 * goal_bytes + 2 * depth_bytes + extras
    total = capacity * per_tr
    return {
        "per_transition_bytes": int(per_tr),
        "total_bytes": int(total),
        "total_gb": float(total) / (1024.0 ** 3),
        "fp16": bool(fp16),
        "capacity": int(capacity),
    }


@torch.no_grad()
def _inference_latency(policy: StochasticVelocityPolicy,
                       *,
                       device: torch.device,
                       T: int,
                       image_size: int,
                       goal_dim: int,
                       batch_size: int = 1,
                       n_warmup: int = 5,
                       n_timed: int = 20) -> Dict[str, float]:
    """Time policy.act on a dummy batch. Reports mean / std ms.

    Restores the policy's training mode before returning so the caller's
    subsequent .backward() through the GRU is not blocked by cuDNN's
    "RNN backward only in training mode" rule.
    """
    was_training = policy.training
    policy.eval()
    rgb = torch.randn(batch_size, T, 3, image_size, image_size, device=device)
    goal = torch.randn(batch_size, goal_dim, device=device)
    depth = torch.randn(batch_size, T, 1, image_size, image_size, device=device) if policy.use_depth else None

    # Warmup
    for _ in range(n_warmup):
        policy.act(rgb, goal, depth)
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        policy.act(rgb, goal, depth)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    if was_training:
        policy.train()

    n = len(samples)
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / max(n - 1, 1)
    return {
        "batch_size": int(batch_size),
        "n_timed": int(n),
        "mean_ms": float(mean),
        "std_ms": float(var ** 0.5),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def summarize_perf(policy: StochasticVelocityPolicy,
                   *,
                   device: torch.device,
                   algorithm: str,
                   sac_kw: Optional[Dict[str, Any]] = None,
                   ppo_kw: Optional[Dict[str, Any]] = None,
                   compress_transitions: bool = True,
                   frame_window: int = 4,
                   image_size: int = 224,
                   goal_dim: int = 3,
                   batch_size_for_latency: int = 1) -> Dict[str, Any]:
    """Build the full startup perf-summary dict."""
    out: Dict[str, Any] = {
        "algorithm": algorithm,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "params": _param_breakdown(policy),
        "inference_latency_b1": _inference_latency(
            policy,
            device=device,
            T=int(policy.base.T),
            image_size=image_size,
            goal_dim=goal_dim,
            batch_size=batch_size_for_latency,
        ),
    }

    if algorithm == "sac" and sac_kw is not None:
        out["replay_buffer"] = _replay_buffer_footprint(
            capacity=int(sac_kw.get("replay_capacity", 0)),
            frame_window=frame_window,
            image_size=image_size,
            use_depth=bool(policy.use_depth),
            fp16=bool(compress_transitions),
        )
        # Time SAC's typical mini-batch latency too.
        out["inference_latency_sac_batch"] = _inference_latency(
            policy,
            device=device,
            T=int(policy.base.T),
            image_size=image_size,
            goal_dim=goal_dim,
            batch_size=int(sac_kw.get("batch_size", 256)),
        )
    elif algorithm == "ppo" and ppo_kw is not None:
        out["inference_latency_ppo_batch"] = _inference_latency(
            policy,
            device=device,
            T=int(policy.base.T),
            image_size=image_size,
            goal_dim=goal_dim,
            batch_size=int(ppo_kw.get("batch_size", 256)),
        )

    return out


def format_perf_block(summary: Dict[str, Any]) -> str:
    """Pretty stdout block. One-line entries; safe for `print(...)`."""
    lines = []
    lines.append("=" * 64)
    lines.append(f" perf summary  algorithm={summary['algorithm']} device={summary['device']}")
    lines.append("=" * 64)
    p = summary["params"]
    lines.append(f"  params     trainable={p['total_trainable']:>11,}  frozen={p['total_frozen']:>11,}  approx={p['approx_mb']:.1f} MB")
    for mod, counts in p["per_module"].items():
        lines.append(f"    {mod:>14}: trainable={counts['trainable']:>11,}  frozen={counts['frozen']:>11,}")
    lat = summary["inference_latency_b1"]
    lines.append(f"  latency b=1 mean={lat['mean_ms']:.2f} ms  std={lat['std_ms']:.2f}  min={lat['min_ms']:.2f}  max={lat['max_ms']:.2f}")
    if "inference_latency_sac_batch" in summary:
        latb = summary["inference_latency_sac_batch"]
        lines.append(f"  latency b={latb['batch_size']} mean={latb['mean_ms']:.2f} ms  std={latb['std_ms']:.2f}")
    if "inference_latency_ppo_batch" in summary:
        latb = summary["inference_latency_ppo_batch"]
        lines.append(f"  latency b={latb['batch_size']} mean={latb['mean_ms']:.2f} ms  std={latb['std_ms']:.2f}")
    if "replay_buffer" in summary:
        rb = summary["replay_buffer"]
        lines.append(
            f"  replay     capacity={rb['capacity']:,}  per_tr={rb['per_transition_bytes']/1024:.1f} KB  "
            f"total~={rb['total_gb']:.2f} GB  fp16={rb['fp16']}"
        )
    lines.append("=" * 64)
    return "\n".join(lines)


def persist_perf_summary(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
