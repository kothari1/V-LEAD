"""
OT-CFM training loop for FlowMatchingPolicy.

Reads:
    nav_policy/data/processed_flightroom/manifest.json
    nav_policy/data/processed_flightroom/stats.json

Writes (under cfg.train.checkpoint_dir):
    fm_best.pt    (lowest val_mse_overall; same format as bc_best.pt)
    fm_latest.pt  (last epoch)
    log.csv       (per-epoch metrics)

The checkpoint format is identical to BC:
    {"model": state_dict, "config": cfg, "epoch": n,
     "val_loss": fm_loss, "val_mse_overall": mse, "stats": stats_dict}

This means load_bc_into_feature_extractor(extractor, "fm_best.pt") works
unchanged for SAC RL warm-start.

Run:
    python scripts/train_fm.py --config configs/flightroom_fm.yaml
    python scripts/train_fm.py --config configs/flightroom_fm_modal.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from nav_policy.data.normalization import CommandStats
from nav_policy.data.rgb_horizon_dataset import CacheBucketSampler, RGBHorizonDataset
from nav_policy.model.flow_matching_policy import FlowMatchingPolicy
from nav_policy.model.losses import per_component_mse
from nav_policy.model.rgb_velocity_policy import count_parameters

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collate(batch):
    rgbs, goals, u_stars, metas = zip(*batch)
    rgb = torch.stack(rgbs)
    goal = torch.stack(goals)
    u_star = torch.stack(u_stars)
    u_raw = torch.stack([m["u_raw"] for m in metas])
    return rgb, goal, u_star, u_raw


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __call__(self): return self


# ── Data ──────────────────────────────────────────────────────────────────────

def _make_loaders(cfg: dict, processed_root: Path):
    use_jitter = bool(cfg["train"].get("color_jitter", True))
    zero_goal  = bool(cfg["train"].get("zero_goal_heading", False))
    goal_input_dim    = int(cfg["model"].get("goal_input_dim", 3))
    goal_distance_scale = float(cfg["model"].get("goal_distance_scale", 5.0))
    data_cfg = cfg.get("data", {})
    cache_in_mem = bool(data_cfg.get("cache_blobs_in_memory", False))
    cache_lru    = int(data_cfg.get("cache_lru_size", 64))
    ds_kw = dict(
        cache_blobs_in_memory=cache_in_mem,
        cache_lru_size=cache_lru,
        zero_goal_heading=zero_goal,
        goal_input_dim=goal_input_dim,
        goal_distance_scale=goal_distance_scale,
    )

    train_runs = data_cfg.get("train_runs") or None   # None = all tagged-train runs
    val_runs   = data_cfg.get("val_runs")             # None = all; [] = intentionally empty
    train_ds = RGBHorizonDataset(processed_root, split="train",
                                 use_color_jitter=use_jitter,
                                 run_filter=train_runs, **ds_kw)
    val_ds   = RGBHorizonDataset(processed_root, split="val",
                                 use_color_jitter=False,
                                 run_filter=val_runs, **ds_kw)

    pin = torch.cuda.is_available()
    nw_train = int(cfg["train"].get("num_workers", 0))
    nw_val   = min(2, nw_train)

    use_bucket = bool(data_cfg.get("bucket_sampling", True))
    if use_bucket:
        train_sampler = CacheBucketSampler(train_ds, seed=int(cfg["train"].get("seed", 0)))
        shuffle_arg   = None
    else:
        train_sampler = None
        shuffle_arg   = True

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=train_sampler,
        shuffle=shuffle_arg,
        num_workers=nw_train,
        pin_memory=pin and nw_train > 0,
        drop_last=True,
        collate_fn=_collate,
        persistent_workers=nw_train > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=nw_val,
        pin_memory=pin and nw_val > 0,
        drop_last=False,
        collate_fn=_collate,
    )
    return train_ds, val_ds, train_dl, val_dl, train_sampler


# ── Model ─────────────────────────────────────────────────────────────────────

def _build_model(cfg: dict) -> FlowMatchingPolicy:
    fm_cfg = cfg.get("fm", {})
    return FlowMatchingPolicy(
        T=int(cfg["window"]["T"]),
        H=int(cfg["window"]["H"]),
        cmd_dim=int(cfg["model"]["cmd_dim"]),
        gru_hidden=int(cfg["model"]["gru_hidden"]),
        gru_layers=int(cfg["model"]["gru_layers"]),
        goal_emb_dim=int(cfg["model"].get("goal_emb_dim", 32)),
        goal_input_dim=int(cfg["model"].get("goal_input_dim", 3)),
        freeze_stem_and_layer1=bool(cfg["model"].get("freeze_stem_and_layer1", True)),
        time_emb_dim=int(fm_cfg.get("time_emb_dim", 64)),
        vf_hidden=tuple(fm_cfg.get("vf_hidden", [512, 512, 512])),
        vf_use_skip=bool(fm_cfg.get("vf_use_skip", True)),
    )


# ── Training epoch ────────────────────────────────────────────────────────────

def _run_train_epoch(policy: FlowMatchingPolicy,
                     loader: DataLoader,
                     device: torch.device,
                     optimizer,
                     scaler,
                     grad_clip: float = 1.0,
                     log_every: int = 50,
                     header: str = "") -> Dict[str, float]:
    policy.train()
    totals: Dict[str, float] = defaultdict(float)
    n_batches = 0
    t0 = time.time()
    autocast_ctx = torch.cuda.amp.autocast if scaler is not None else _NullCtx

    for it, (rgb, goal, u_star, _) in enumerate(loader):
        rgb    = rgb.to(device, non_blocking=True)
        goal   = goal.to(device, non_blocking=True)
        u_star = u_star.to(device, non_blocking=True)

        with autocast_ctx():
            loss = policy.fm_loss(rgb, goal, u_star)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

        totals["fm_loss"] += float(loss.item())
        n_batches += 1

        if (it + 1) % log_every == 0:
            avg_loss = totals["fm_loss"] / n_batches
            print(f"  {header} it={it + 1:>5d}  fm_loss={avg_loss:.4f}", flush=True)

    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    avg["sec_per_epoch"] = time.time() - t0
    avg["n_batches"] = n_batches
    return avg


def _run_val_epoch(policy: FlowMatchingPolicy,
                   loader: DataLoader,
                   stats: CommandStats,
                   device: torch.device,
                   n_val_steps: int = 4) -> Dict[str, float]:
    """Evaluates FM loss AND action MSE (via sample) on the validation set."""
    policy.eval()
    totals: Dict[str, float] = defaultdict(float)
    n_batches = 0

    with torch.no_grad():
        for rgb, goal, u_star, u_raw in loader:
            rgb    = rgb.to(device, non_blocking=True)
            goal   = goal.to(device, non_blocking=True)
            u_star = u_star.to(device, non_blocking=True)
            u_raw  = u_raw.to(device, non_blocking=True)

            # 1. FM matching loss (fast, no ODE integration)
            loss = policy.fm_loss(rgb, goal, u_star)
            totals["fm_loss"] += float(loss.item())

            # 2. Sample and compute physical-unit MSE (n_val_steps Euler steps)
            u_hat = policy.sample(rgb, goal, n_steps=n_val_steps)
            metrics = per_component_mse(u_hat.float(), u_raw, stats)
            for k, v in metrics.items():
                totals[k] += float(v.item())

            n_batches += 1

    if n_batches == 0:
        nan = float("nan")
        return {"fm_loss": nan, "mse_overall": nan, "mse_vx": nan,
                "mse_vy": nan, "mse_vz": nan, "mse_psi_dot": nan, "mse_lin_vel": nan}
    return {k: v / n_batches for k, v in totals.items()}


# ── W&B helpers ───────────────────────────────────────────────────────────────

def _wb_init(wb_cfg: dict, cfg: dict, run_tag: str) -> bool:
    if not wb_cfg.get("enabled", False):
        return False
    if not _WANDB_AVAILABLE:
        print("[wandb] not installed — skipping W&B logging", flush=True)
        return False
    _wandb.init(
        project=wb_cfg.get("project", "vlead-fm"),
        name=run_tag or wb_cfg.get("run_name") or None,
        config=cfg,
    )
    return True


def _wb_log(metrics: dict, step: int) -> None:
    if _WANDB_AVAILABLE and _wandb.run is not None:
        _wandb.log(metrics, step=step)


# ── Main training function ────────────────────────────────────────────────────

def train(config_path: Path,
          checkpoint_dir_override: Optional[Path] = None,
          run_tag_override: Optional[str] = None,
          resume_from: Optional[Path] = None) -> None:

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    base = config_path.resolve().parent.parent

    if checkpoint_dir_override is not None:
        cfg.setdefault("train", {})["checkpoint_dir"] = str(checkpoint_dir_override)
    if run_tag_override is not None:
        cfg.setdefault("train", {})["run_tag"] = str(run_tag_override)

    processed_root = (base / cfg["data"]["processed_root"]).resolve()
    ckpt_dir = (base / cfg["train"]["checkpoint_dir"]).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_tag = cfg.get("train", {}).get("run_tag", "fm_run")
    print(f"[checkpoint_dir] {ckpt_dir}")
    print(f"[run_tag]        {run_tag}")

    _set_seed(int(cfg["train"].get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  cuda_available={torch.cuda.is_available()}")

    train_ds, val_ds, train_dl, val_dl, train_sampler = _make_loaders(cfg, processed_root)
    stats = train_ds.stats
    has_val = len(val_ds) > 0
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  "
          f"T={train_ds.T}  H={train_ds.H}  S={train_ds.image_size}")
    print(f"[stats] mean={stats.mean.tolist()}  std={stats.std.tolist()}")

    policy = _build_model(cfg).to(device)
    if resume_from is not None:
        resume_path = (base / resume_from).resolve()
        blob = torch.load(resume_path, weights_only=False, map_location=device)
        policy.load_state_dict(blob["model"])
        print(f"[resume] warm-started from {resume_path}")

    total_params = count_parameters(policy, trainable_only=False)
    train_params = count_parameters(policy, trainable_only=True)
    vf_params    = count_parameters(policy.vector_field, trainable_only=True)
    print(f"[model] total={total_params:,}  trainable={train_params:,}  "
          f"vector_field={vf_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    use_amp = bool(cfg["train"].get("amp", True)) and torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    fm_cfg      = cfg.get("fm", {})
    n_val_steps = int(fm_cfg.get("n_val_steps", 4))
    log_every   = int(cfg["train"].get("log_every", 50))
    grad_clip   = float(cfg["train"].get("grad_clip", 1.0))

    wb_enabled  = _wb_init(cfg.get("wandb", {}), cfg, run_tag)

    log_path   = ckpt_dir / "log.csv"
    log_fields = [
        "epoch",
        "train_fm_loss",
        "val_fm_loss",
        "val_mse_vx", "val_mse_vy", "val_mse_vz", "val_mse_psi_dot",
        "val_mse_lin_vel", "val_mse_overall",
        "sec",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(log_fields)

    best_val = math.inf
    patience  = int(cfg["train"].get("early_stopping_patience", 0))
    no_improve = 0

    for epoch in range(int(cfg["train"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        header = f"[epoch {epoch + 1:>3d}]"
        tr = _run_train_epoch(
            policy, train_dl, device, optimizer, scaler,
            grad_clip=grad_clip, log_every=log_every, header=header,
        )
        va = _run_val_epoch(policy, val_dl, stats, device, n_val_steps=n_val_steps)

        if has_val:
            print(
                f"{header}  train_fm={tr['fm_loss']:.4f}  "
                f"val_fm={va['fm_loss']:.4f}  val_mse_lin={va['mse_lin_vel']:.4f}  "
                f"val_mse_psi={va['mse_psi_dot']:.4f}  sec={tr['sec_per_epoch']:.1f}",
                flush=True,
            )
        else:
            print(
                f"{header}  train_fm={tr['fm_loss']:.4f}  [no val]  "
                f"sec={tr['sec_per_epoch']:.1f}",
                flush=True,
            )

        # ── W&B logging ──────────────────────────────────────────────────────
        if wb_enabled:
            wb_dict = {
                "train/fm_loss":       tr["fm_loss"],
                "train/sec_per_epoch": tr["sec_per_epoch"],
            }
            if has_val:
                wb_dict.update({
                    "val/fm_loss":     va["fm_loss"],
                    "val/mse_vx":      va["mse_vx"],
                    "val/mse_vy":      va["mse_vy"],
                    "val/mse_vz":      va["mse_vz"],
                    "val/mse_psi_dot": va["mse_psi_dot"],
                    "val/mse_lin_vel": va["mse_lin_vel"],
                    "val/mse_overall": va["mse_overall"],
                })
            _wb_log(wb_dict, step=epoch + 1)

        # ── CSV log ──────────────────────────────────────────────────────────
        nan = float("nan")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                tr["fm_loss"],
                va.get("fm_loss", nan),
                va.get("mse_vx", nan), va.get("mse_vy", nan),
                va.get("mse_vz", nan), va.get("mse_psi_dot", nan),
                va.get("mse_lin_vel", nan), va.get("mse_overall", nan),
                tr["sec_per_epoch"],
            ])

        # ── Checkpoint ───────────────────────────────────────────────────────
        state = {
            "model":           policy.state_dict(),
            "config":          cfg,
            "epoch":           epoch + 1,
            "val_loss":        va.get("fm_loss", nan),
            "val_mse_overall": va.get("mse_overall", nan),
            "stats":           stats.to_dict(),
        }
        torch.save(state, ckpt_dir / "fm_latest.pt")

        # Track by val_mse_overall when val exists; fall back to train_fm_loss.
        track = va["mse_overall"] if has_val else tr["fm_loss"]
        if track < best_val:
            best_val   = track
            no_improve = 0
            torch.save(state, ckpt_dir / "fm_best.pt")
            label = f"val_mse_overall={best_val:.4f}" if has_val else f"train_fm_loss={best_val:.4f}"
            print(f"  -> saved fm_best.pt ({label})", flush=True)
        else:
            no_improve += 1
            print(f"  -> no improvement ({no_improve}/{patience or '∞'})", flush=True)

        if patience > 0 and no_improve >= patience:
            print(f"[early stop] no improvement for {patience} epochs.", flush=True)
            break

    if wb_enabled and _WANDB_AVAILABLE:
        _wandb.finish()
    metric_label = "val_mse_overall" if has_val else "train_fm_loss"
    print(f"[done] best {metric_label}={best_val:.4f}  ckpt={ckpt_dir/'fm_best.pt'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train FlowMatchingPolicy via OT-CFM.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Override train.checkpoint_dir from YAML.")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Override train.run_tag; used as W&B run name.")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Path to checkpoint whose model weights are loaded before training.")
    args = p.parse_args()
    train(args.config,
          checkpoint_dir_override=args.checkpoint_dir,
          run_tag_override=args.run_tag,
          resume_from=args.resume_from)


if __name__ == "__main__":
    main()
