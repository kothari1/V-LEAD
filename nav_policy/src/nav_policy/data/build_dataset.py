"""
Walk SINGER-format validation rollouts and produce a per-sub-trajectory cache
plus a global manifest + command-space statistics.

Raw layout (per run):
    data/raw/<run>/trajectories_val{NNNNN}.pt
    data/raw/<run>/imgdata_val{NNNNN}.pt
    data/raw/<run>/video_val_rollout_images_rgb{NNNNN}.mp4

Processed layout:
    data/processed/<run>/cache/stack{NNNNN}_sub{i}.pt   # {rgb, vel, psi_dot, goal_heading, goal_dist, meta}
    data/processed/manifest.json                        # global train/val window list
    data/processed/stats.json                           # CommandStats over training set

A "window" is identified by (cache_relpath, k):
    rgb_frames     = cache.rgb[k - T + 1 : k + 1]                # [T, 3, S, S] uint8
    goal_heading   = cache.goal_heading[k]                        # [2] float32
    goal_dist      = cache.goal_dist[k]                           # [] float32 (meters)
    expert_command = stack([cache.vel[k:k+H], cache.psi_dot[k:k+H]], dim=-1)  # [H, 4] float32

Goal heading:  unit vector in the world XY-plane pointing from the drone's
current position toward the sub-trajectory's final position (the end of the
reference trajectory).  At each step k:
    delta_xy = Xro[0:2, -1] - Xro[0:2, k]
    goal_heading[k] = delta_xy / ||delta_xy||   (fallback = [1, 0] if at goal)

Goal distance: scalar Euclidean distance in world XY from the drone's current
position to the goal, in raw meters:
    goal_dist[k] = ||Xro[0:2, -1] - Xro[0:2, k]||
The dataset normalizes by a configurable characteristic scale at read time so
the model sees an input on the order of [0, 1] near the goal; raw meters are
kept in the cache so the scale can be retuned without rebuilding.

The video is 20 fps, matches the controller rate. Each cache file corresponds
to exactly one sub-trajectory (one of the 4 in a stack); the cache rgb already
has the per-frame trim and resize applied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import imageio.v3 as iio
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

from nav_policy.data.normalization import CommandStats


CONTROL_HZ = 20.0  # vrmpc_br.json -> "hz": 20

_STACK_RE = re.compile(r"trajectories_val(\d{5})\.pt$")


def write_cache(cache_path: Path,
                rgb_uint8: torch.Tensor,
                vel: torch.Tensor,
                psi_dot: torch.Tensor,
                goal_heading: torch.Tensor,
                meta: dict,
                goal_dist: Optional[torch.Tensor] = None) -> None:
    """
    Write a single sub-trajectory cache.

    rgb_uint8:    [N, 3, S, S] uint8
    vel:          [N, 3] float32 (world-frame [vx, vy, vz])
    psi_dot:      [N]    float32 (world-frame yaw rate, rad/s)
    goal_heading: [N, 2] float32 (unit vector toward sub-traj goal in world XY)
    goal_dist:    [N]    float32 (raw meters from current XY to sub-traj goal XY)
                  Optional only for backward compatibility with old callers;
                  the dataset will fall back to zeros if missing, but new
                  caches should always include it.
    meta:         arbitrary JSON-serializable dict
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if rgb_uint8.dtype != torch.uint8:
        raise TypeError(f"expected uint8 rgb, got {rgb_uint8.dtype}")
    if vel.dtype != torch.float32 or psi_dot.dtype != torch.float32:
        raise TypeError("vel and psi_dot must be float32")
    if goal_heading.dtype != torch.float32 or goal_heading.shape[-1] != 2:
        raise TypeError(f"goal_heading must be float32 [N,2], got {goal_heading.shape}")
    blob = {
        "rgb": rgb_uint8.contiguous(),
        "vel": vel.contiguous(),
        "psi_dot": psi_dot.contiguous(),
        "goal_heading": goal_heading.contiguous(),
        "meta": dict(meta),
    }
    if goal_dist is not None:
        if goal_dist.dtype != torch.float32 or goal_dist.ndim != 1:
            raise TypeError(f"goal_dist must be float32 [N], got {goal_dist.shape}")
        blob["goal_dist"] = goal_dist.contiguous()
    torch.save(blob, cache_path)


def _compute_goal_heading(xro: np.ndarray, n: int) -> np.ndarray:
    """
    Compute per-step goal heading unit vectors.

    xro:   (10, Nctl+1) state matrix; rows 0:2 are world-frame XY position.
    n:     number of steps to emit.
    Returns [n, 2] float32.
    """
    goal_xy = xro[0:2, -1]                          # (2,) final XY position
    pos_xy = xro[0:2, :n]                           # (2, n) current positions
    delta = goal_xy[:, None] - pos_xy               # (2, n)
    norms = np.linalg.norm(delta, axis=0, keepdims=True)   # (1, n)
    norms = np.maximum(norms, 1e-6)
    heading = (delta / norms).T.astype(np.float32)  # (n, 2)
    return heading


def _compute_goal_distance(xro: np.ndarray, n: int) -> np.ndarray:
    """
    Per-step Euclidean distance from current XY to the sub-trajectory's goal XY.

    Returns [n] float32 in raw meters.  The dataset reader normalizes by a
    characteristic scale at training time, so the raw value is kept on disk
    to allow retuning without rebuilding the cache.
    """
    goal_xy = xro[0:2, -1]                          # (2,)
    pos_xy = xro[0:2, :n]                           # (2, n)
    delta = goal_xy[:, None] - pos_xy               # (2, n)
    dist = np.linalg.norm(delta, axis=0).astype(np.float32)  # (n,)
    return dist


# ----------------------------- utilities -----------------------------


def _resize_uint8(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize HxWx3 uint8 to size x size x 3 uint8 (bilinear)."""
    if _HAVE_CV2:
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    from PIL import Image  # local import to keep the cv2 path zero-overhead
    img = Image.fromarray(frame, mode="RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img)


def _quat_to_yaw(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Compute yaw (rotation around world z) from a series of quaternions.
    Input shape: [N, 4] in scipy/Hamilton scalar-last order [qx, qy, qz, qw].
    Output: [N] yaw in radians, unwrapped.
    """
    eul = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
    yaw = eul[:, 2]
    return np.unwrap(yaw)


def _read_video(path: Path) -> np.ndarray:
    """Decode an mp4 into a uint8 array of shape [N, H, W, 3]."""
    frames = iio.imread(str(path), plugin="FFMPEG")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"unexpected video shape {frames.shape} from {path}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return frames


def _iter_stack_ids(run_dir: Path) -> Iterator[str]:
    for name in sorted(os.listdir(run_dir)):
        m = _STACK_RE.match(name)
        if m:
            yield m.group(1)


# ----------------------------- core logic -----------------------------


@dataclass
class StackPaths:
    traj: Path
    imgdata: Path
    rgb_video: Path


def _stack_paths(run_dir: Path, stack_id: str) -> StackPaths:
    return StackPaths(
        traj=run_dir / f"trajectories_val{stack_id}.pt",
        imgdata=run_dir / f"imgdata_val{stack_id}.pt",
        rgb_video=run_dir / f"video_val_rollout_images_rgb{stack_id}.mp4",
    )


def _build_one_stack(stack_id: str,
                     paths: StackPaths,
                     out_run_dir: Path,
                     image_size: int) -> List[Path]:
    """
    Cache the 4 sub-trajectories of one stack to disk.

    Returns the list of cache paths written for this stack (may be < 4 if the
    video frame range is missing or the sub-trajectory has zero Nctl).
    """
    out_run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    traj_stack = torch.load(paths.traj, weights_only=False, map_location="cpu")
    img_stack = torch.load(paths.imgdata, weights_only=False, map_location="cpu")
    frames = _read_video(paths.rgb_video)

    trajectories = traj_stack["data"]
    img_meta = img_stack["data"]
    if len(trajectories) != len(img_meta):
        raise ValueError(
            f"stack {stack_id}: trajectories ({len(trajectories)}) != "
            f"imgdata ({len(img_meta)}) sub-entries"
        )

    written: List[Path] = []
    for sub_idx, (traj, meta) in enumerate(zip(trajectories, img_meta)):
        xro = traj["Xro"]              # (10, Nctl+1)
        uro = traj["Uro"]              # (4, Nctl)
        nctl = int(traj["Ndata"])      # = Uro.shape[1]
        start = int(meta["start_id"])
        end = int(meta["end_id"])      # inclusive

        # Sanity checks tied to the FiGS controller frequency.
        if uro.shape[1] != nctl:
            raise ValueError(f"stack {stack_id} sub {sub_idx}: Uro Nctl mismatch")
        # 100 control steps -> 100 frames (frame index k corresponds to controller step k)
        n_frames_expected = nctl
        n_frames_video = end - start + 1
        if n_frames_video != n_frames_expected:
            # Tolerate off-by-one; rollout writers sometimes drop the terminal frame.
            n_frames_use = min(n_frames_expected, n_frames_video)
        else:
            n_frames_use = n_frames_expected
        if n_frames_use <= 0:
            continue

        # --- visual cache ---
        sub_frames = frames[start : start + n_frames_use]   # [n, H, W, 3]
        resized = np.stack(
            [_resize_uint8(f, image_size) for f in sub_frames], axis=0
        )                                                    # [n, S, S, 3]
        rgb = torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous()  # [n, 3, S, S]
        assert rgb.dtype == torch.uint8, rgb.dtype

        # --- command labels ---
        # velocity at controller step k is Xro[3:6, k] (world frame).
        # We use indices [0, n_frames_use) to align with the frames we kept.
        vel = xro[3:6, :n_frames_use].T.astype(np.float32)   # [n, 3]

        # yaw rate via finite-difference of unwrapped yaw across Xro[6:10].
        # Xro has Nctl+1 columns -> we need yaw at k and k+1 for psi_dot[k].
        quat_cols = xro[6:10, : n_frames_use + 1]            # (4, n+1) if available
        if quat_cols.shape[1] < n_frames_use + 1:
            # If for some reason we don't have the terminal state, repeat the last col.
            quat_cols = np.concatenate(
                [quat_cols, quat_cols[:, -1:]], axis=1
            )
        quat = quat_cols.T                                    # (n+1, 4)
        yaw = _quat_to_yaw(quat)                              # (n+1,)
        psi_dot = (yaw[1:] - yaw[:-1]) * CONTROL_HZ           # (n,)
        psi_dot = psi_dot.astype(np.float32)

        # --- goal heading (unit vector from current XY to trajectory end XY) ---
        goal_heading_np = _compute_goal_heading(xro, n_frames_use)  # [n, 2]
        # --- goal distance (scalar Euclidean distance to trajectory end XY) ---
        goal_dist_np = _compute_goal_distance(xro, n_frames_use)    # [n]

        cache_path = cache_dir / f"stack{stack_id}_sub{sub_idx}.pt"
        write_cache(
            cache_path,
            rgb_uint8=rgb,
            vel=torch.from_numpy(vel),
            psi_dot=torch.from_numpy(psi_dot),
            goal_heading=torch.from_numpy(goal_heading_np),
            goal_dist=torch.from_numpy(goal_dist_np),
            meta={
                "run": out_run_dir.name,
                "stack_id": stack_id,
                "sub_idx": sub_idx,
                "rollout_id": str(traj.get("rollout_id", "")),
                "course": str(traj.get("course", "")),
                "n_frames": int(n_frames_use),
                "control_hz": CONTROL_HZ,
                "source": "bc_expert",
            },
        )
        written.append(cache_path)

    return written


# ----------------------------- driver -----------------------------


def _enumerate_windows(cache_paths: List[Path],
                       T: int,
                       H: int,
                       processed_root: Path) -> List[dict]:
    """Build the global window list (one entry per training sample)."""
    entries: List[dict] = []
    for cache_path in cache_paths:
        blob = torch.load(cache_path, weights_only=False, map_location="cpu")
        n = int(blob["rgb"].shape[0])
        # valid k: rgb window [k-T+1, k] in-range AND command window [k, k+H-1] in-range
        k_min = T - 1
        k_max = n - H        # exclusive bound: k <= n - H
        if k_max <= k_min:
            continue
        rel = cache_path.relative_to(processed_root).as_posix()
        for k in range(k_min, k_max):
            entries.append({"cache": rel, "k": int(k)})
    return entries


def _split_train_val(entries: List[dict],
                     val_fraction: float,
                     seed: int) -> List[dict]:
    """Tag each window with split='train'|'val' grouped by cache (per sub-trajectory)."""
    rng = np.random.default_rng(seed)
    caches = sorted({e["cache"] for e in entries})
    rng.shuffle(caches)
    n_val = max(1, int(round(val_fraction * len(caches))))
    val_set = set(caches[:n_val])
    for e in entries:
        e["split"] = "val" if e["cache"] in val_set else "train"
    return entries


def _compute_stats_from_entries(entries: List[dict],
                                processed_root: Path,
                                H: int) -> CommandStats:
    """
    Fit CommandStats over all training windows using a streaming, one-cache-at-a-time
    pass that avoids loading RGB tensors (which can total many GB for large datasets).

    Algorithm: Welford online mean/variance over flattened [H, 4] windows so that
    only one cache file's vel+psi_dot arrays are resident at a time.
    """
    # Group training windows by cache file to minimise re-loads.
    from collections import defaultdict
    cache_to_ks: dict[str, List[int]] = defaultdict(list)
    for e in entries:
        if e.get("split") == "train":
            cache_to_ks[e["cache"]].append(int(e["k"]))

    if not cache_to_ks:
        raise RuntimeError("no training entries available to fit command stats")

    # Welford accumulator (per channel, across all H-step horizons)
    n_total = 0
    mean_acc = np.zeros(4, dtype=np.float64)
    M2_acc   = np.zeros(4, dtype=np.float64)

    for rel, ks in cache_to_ks.items():
        # Load only the lightweight tensors; skip RGB.
        raw = torch.load(processed_root / rel, weights_only=False, map_location="cpu")
        vel_np     = raw["vel"].numpy()      # [N, 3] float32
        psi_np     = raw["psi_dot"].numpy()  # [N]    float32
        del raw                              # release RGB immediately

        for k in ks:
            u = np.concatenate(
                [vel_np[k : k + H],
                 psi_np[k : k + H, None]], axis=1
            )                                # [H, 4] float32
            # Welford update for each of the H rows
            for row in u:
                n_total += 1
                delta = row.astype(np.float64) - mean_acc
                mean_acc += delta / n_total
                M2_acc   += delta * (row.astype(np.float64) - mean_acc)

    # Population std from Welford accumulator
    var  = M2_acc / max(n_total - 1, 1)
    std  = np.sqrt(np.maximum(var, 1e-8))
    return CommandStats(mean=mean_acc.astype(np.float32),
                        std=std.astype(np.float32))


def build(config_path: Path) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # All paths are resolved relative to the directory containing the config
    # (which is nav_policy/configs/) -> parent.parent == nav_policy/
    base = config_path.resolve().parent.parent
    raw_root = (base / cfg["data"]["raw_root"]).resolve()
    processed_root = (base / cfg["data"]["processed_root"]).resolve()
    runs: List[str] = cfg["data"]["runs"]
    drop_missing_rgb: bool = cfg["data"].get("drop_missing_rgb", True)
    val_fraction: float = cfg["data"].get("val_fraction", 0.10)
    val_seed: int = cfg["data"].get("val_split_seed", 0)
    T: int = int(cfg["window"]["T"])
    H: int = int(cfg["window"]["H"])
    image_size: int = int(cfg["window"]["image_size"])

    processed_root.mkdir(parents=True, exist_ok=True)
    all_cache_paths: List[Path] = []

    for run in runs:
        run_dir = raw_root / run
        if not run_dir.is_dir():
            print(f"[skip] {run}: directory missing at {run_dir}", file=sys.stderr)
            continue

        out_run_dir = processed_root / run
        stack_ids = list(_iter_stack_ids(run_dir))
        print(f"[{run}] {len(stack_ids)} stacks found, processing...")

        for stack_id in tqdm(stack_ids, unit="stack", leave=False):
            paths = _stack_paths(run_dir, stack_id)
            if not paths.rgb_video.exists():
                if drop_missing_rgb:
                    continue
                raise FileNotFoundError(f"missing rgb video: {paths.rgb_video}")
            if not paths.imgdata.exists():
                raise FileNotFoundError(f"missing imgdata: {paths.imgdata}")

            written = _build_one_stack(stack_id, paths, out_run_dir, image_size)
            all_cache_paths.extend(written)

    if not all_cache_paths:
        print("No caches were written. Check that data/raw/<run>/ is populated.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[manifest] enumerating windows over {len(all_cache_paths)} caches...")
    entries = _enumerate_windows(all_cache_paths, T=T, H=H, processed_root=processed_root)
    print(f"[manifest] {len(entries)} total windows")

    entries = _split_train_val(entries, val_fraction=val_fraction, seed=val_seed)
    n_train = sum(1 for e in entries if e["split"] == "train")
    n_val = len(entries) - n_train
    print(f"[manifest] split: train={n_train}  val={n_val}")

    print(f"[stats] fitting CommandStats over train split...")
    stats = _compute_stats_from_entries(entries, processed_root, H=H)
    print(f"[stats] mean={stats.mean.tolist()}  std={stats.std.tolist()}")

    manifest = {
        "T": T,
        "H": H,
        "image_size": image_size,
        "control_hz": CONTROL_HZ,
        "imagenet_mean": list(cfg["window"]["imagenet_mean"]),
        "imagenet_std": list(cfg["window"]["imagenet_std"]),
        "samples": entries,
    }
    with open(processed_root / "manifest.json", "w") as f:
        json.dump(manifest, f)
    with open(processed_root / "stats.json", "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    print(f"[done] wrote {processed_root / 'manifest.json'}")
    print(f"[done] wrote {processed_root / 'stats.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build nav_policy training cache + manifest.")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to YAML config (e.g. configs/default.yaml).")
    args = p.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
