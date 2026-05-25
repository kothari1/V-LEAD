"""
PyTorch Dataset over the processed manifest produced by build_dataset.py.

Yields per-sample:
    rgb:    float32 tensor [T, 3, S, S], ImageNet-normalized
    goal:   float32 tensor [goal_input_dim]
            goal_input_dim=2 -> [hx, hy]               (heading only)
            goal_input_dim=3 -> [hx, hy, d_normalized] (heading + distance)
    u_star: float32 tensor [H, 4], z-scored using CommandStats
    meta:   dict with un-normalized targets and index info:
            - "u_raw": [H, 4] float32 (vx, vy, vz, psi_dot)

The distance is loaded in raw meters from the cache and normalized to
roughly [0, 1] by dividing by ``goal_distance_scale`` (default 5 m).  The
scale is a config knob, not data-derived statistics, so it can be retuned
without rebuilding the cache.  Old caches that pre-date the distance
feature fall back to zero (a warning is emitted by inspect_pt.py).

Color jitter (training only):
    When use_color_jitter=True, a fresh random ColorJitter transform is applied
    independently to every frame in the T-frame window before ImageNet
    normalization.  Independent per-frame jitter ensures the model cannot use
    inter-frame color consistency as a shortcut proxy for motion estimation.
    The transform is applied to uint8 CHW tensors; torchvision handles this
    natively for brightness/contrast/saturation/hue adjustments.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import ColorJitter

from nav_policy.data.normalization import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CommandStats,
    imagenet_normalize,
)


class RGBHorizonDataset(Dataset):
    """Reads sub-trajectory caches lazily, builds windows on the fly."""

    def __init__(self,
                 processed_root: Path,
                 split: str = "train",
                 cache_blobs_in_memory: bool = False,
                 cache_lru_size: int = 64,
                 use_color_jitter: bool = False,
                 zero_goal_heading: bool = False,
                 goal_input_dim: int = 2,
                 goal_distance_scale: float = 5.0) -> None:
        self.processed_root = Path(processed_root)
        manifest_path = self.processed_root / "manifest.json"
        stats_path = self.processed_root / "stats.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing {manifest_path}; run build_dataset first")
        if not stats_path.exists():
            raise FileNotFoundError(f"missing {stats_path}; run build_dataset first")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        with open(stats_path, "r") as f:
            stats_dict = json.load(f)

        self.T: int = int(manifest["T"])
        self.H: int = int(manifest["H"])
        self.image_size: int = int(manifest["image_size"])
        self.imagenet_mean = tuple(manifest.get("imagenet_mean", IMAGENET_MEAN))
        self.imagenet_std = tuple(manifest.get("imagenet_std", IMAGENET_STD))
        self.stats = CommandStats.from_dict(stats_dict)
        self.samples: List[Dict] = [e for e in manifest["samples"] if e["split"] == split]
        if not self.samples:
            raise RuntimeError(f"no samples with split='{split}' in manifest")

        # Per-frame color jitter (training only).  Each frame in a window gets an
        # independently sampled transform so temporal colour consistency is not a
        # learnable cue.  Applied to uint8 CHW tensors before ImageNet normalisation.
        self._jitter: Optional[ColorJitter] = (
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
            if use_color_jitter else None
        )

        # 'No goal heading' ablation: force the goal input to zero so the
        # policy must rely on vision alone.  The cached unit vector is loaded
        # and discarded; matching closed-loop deployment must also set the
        # zero-goal-heading flag in the policy controller.  When True, the
        # distance channel (if present) is also zeroed so the policy truly has
        # no goal-related signal.
        self._zero_goal: bool = bool(zero_goal_heading)

        # Goal input assembly.  goal_input_dim=2 keeps backwards compatibility
        # with the heading-only checkpoints; goal_input_dim=3 appends a
        # scale-normalized scalar distance to the goal.  Caches built before
        # the distance feature do not contain a 'goal_dist' tensor; we
        # silently fall back to zero in that case so old checkpoints can still
        # be evaluated.
        if goal_input_dim not in (2, 3):
            raise ValueError(f"goal_input_dim must be 2 or 3; got {goal_input_dim}")
        self._goal_input_dim: int = int(goal_input_dim)
        if goal_distance_scale <= 0.0:
            raise ValueError(f"goal_distance_scale must be > 0; got {goal_distance_scale}")
        self._goal_distance_scale: float = float(goal_distance_scale)

        # Preloading every cache (rgb + labels) works for small datasets but will
        # OOM on flightroom-scale data (~2500 caches, tens of GB of uint8 RGB).
        # Default is lazy load with a small LRU so repeated windows from the same
        # trajectory do not re-read disk every step.
        self._cache_blobs: Optional[Dict[str, Dict]] = (
            {} if cache_blobs_in_memory else None
        )
        self._lru: Optional[OrderedDict[str, Dict]] = (
            None if cache_blobs_in_memory else OrderedDict()
        )
        self._lru_max = max(1, int(cache_lru_size))
        if cache_blobs_in_memory:
            seen: set[str] = set()
            unique = sorted({s["cache"] for s in self.samples})
            print(
                f"[RGBHorizonDataset] preloading {len(unique)} caches "
                f"for split='{split}' into RAM...",
                flush=True,
            )
            for rel in unique:
                if rel not in seen:
                    seen.add(rel)
                    self._cache_blobs[rel] = torch.load(
                        self.processed_root / rel,
                        weights_only=False,
                        map_location="cpu",
                    )
            print(f"[RGBHorizonDataset] split='{split}' preload done.", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_blob(self, rel: str) -> Dict:
        if self._cache_blobs is not None:
            return self._cache_blobs[rel]
        assert self._lru is not None
        if rel in self._lru:
            self._lru.move_to_end(rel)
            return self._lru[rel]
        blob = torch.load(
            self.processed_root / rel, weights_only=False, map_location="cpu"
        )
        self._lru[rel] = blob
        if len(self._lru) > self._lru_max:
            self._lru.popitem(last=False)
        return blob

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        s = self.samples[idx]
        blob = self._load_blob(s["cache"])
        k = int(s["k"])
        T, H = self.T, self.H

        rgb_window = blob["rgb"][k - T + 1 : k + 1]          # uint8 [T, 3, S, S]
        if rgb_window.shape[0] != T:
            raise RuntimeError(
                f"window {idx} cache={s['cache']} k={k} produced {rgb_window.shape[0]} "
                f"frames (expected T={T})"
            )
        vel = blob["vel"][k : k + H]                          # float32 [H, 3]
        psi_dot = blob["psi_dot"][k : k + H].unsqueeze(-1)    # float32 [H, 1]
        u_raw = torch.cat([vel, psi_dot], dim=-1)             # float32 [H, 4]

        # Goal input assembly.  Caches built before goal-heading support are
        # missing this key; fall back to a zero vector so old caches can still
        # be loaded (policy will receive no goal signal -- retrigger
        # build_dataset to get the proper signal).
        if self._zero_goal:
            goal = torch.zeros(self._goal_input_dim, dtype=torch.float32)
        else:
            if "goal_heading" in blob:
                heading = blob["goal_heading"][k].float()          # [2]
            else:
                heading = torch.zeros(2, dtype=torch.float32)
            if self._goal_input_dim == 2:
                goal = heading
            else:
                # Append a scale-normalized scalar distance.  Old caches lacking
                # the 'goal_dist' tensor get a zero distance (which simply
                # silences this channel; the policy will still get the heading).
                if "goal_dist" in blob:
                    d_raw = float(blob["goal_dist"][k].item())     # meters
                else:
                    d_raw = 0.0
                d_norm = d_raw / self._goal_distance_scale
                goal = torch.cat([heading, torch.tensor([d_norm], dtype=torch.float32)])

        # Apply per-frame color jitter independently before normalization.
        if self._jitter is not None:
            rgb_window = torch.stack(
                [self._jitter(rgb_window[t]) for t in range(T)], dim=0
            )  # still [T, 3, S, S] uint8

        rgb = imagenet_normalize(rgb_window, mean=self.imagenet_mean, std=self.imagenet_std)
        u_star = self.stats.standardize(u_raw)
        return rgb, goal, u_star, {"u_raw": u_raw, "k": k, "cache": s["cache"]}


class CacheBucketSampler(Sampler):
    """
    Memory-efficient sampler for large lazy-loaded datasets.

    Problem with random shuffling on disk-backed datasets:
        Each window (sample) may come from a different cache file.  With
        121k windows across 2640 caches, a standard RandomSampler causes
        ~121k cache loads per epoch — one per sample — even with an LRU,
        because the LRU is too small to cover the whole dataset.

    Solution — bucket by cache, shuffle at cache level:
        1. Shuffle the list of unique cache files each epoch.
        2. Within each cache, shuffle the order of its windows.
        3. Yield sample indices in that order.

        Result: each cache file is loaded exactly ONCE per epoch (≈2640
        loads) rather than once per sample (≈121k loads).  This gives a
        ~46x reduction in disk reads for the flightroom dataset.

    Batches formed from this order will draw from 2-4 consecutive caches,
    providing adequate within-batch diversity while keeping I/O minimal.

    Usage:
        sampler = CacheBucketSampler(train_ds, seed=cfg["train"]["seed"])
        sampler.set_epoch(epoch)   # call before each epoch for different ordering
        train_dl = DataLoader(train_ds, batch_size=128, sampler=sampler, ...)
    """

    def __init__(self, dataset: "RGBHorizonDataset", seed: int = 0) -> None:
        # Group dataset indices by their cache file.
        groups: Dict[str, List[int]] = {}
        for idx, s in enumerate(dataset.samples):
            key = s["cache"]
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        self._groups: List[List[int]] = list(groups.values())
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return sum(len(g) for g in self._groups)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed + self._epoch)
        # Shuffle cache order so every epoch sees a different trajectory sequence.
        order = rng.permutation(len(self._groups)).tolist()
        for gi in order:
            group = self._groups[gi]
            # Shuffle windows within this cache.
            perm = rng.permutation(len(group)).tolist()
            for wi in perm:
                yield group[wi]
