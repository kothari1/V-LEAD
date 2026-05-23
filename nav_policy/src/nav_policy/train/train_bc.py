"""
Behavior-cloning training loop for RGBVelocityPolicy.

Reads:
    nav_policy/data/processed/manifest.json
    nav_policy/data/processed/stats.json

Writes (under cfg.train.checkpoint_dir):
    bc_best.pt   (lowest val_mse_overall)
    bc_latest.pt (last epoch)
    log.csv      (per-epoch metrics)

Run inside the docker container:
    python -m nav_policy.train.train_bc --config configs/default.yaml
or via the thin wrapper:
    python scripts/train_bc.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
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
from nav_policy.data.rgb_horizon_dataset import RGBHorizonDataset
from nav_policy.model.losses import bc_loss, per_component_mse
from nav_policy.model.rgb_velocity_policy import RGBVelocityPolicy, count_parameters


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collate(batch):
    rgbs, goals, u_stars, metas = zip(*batch)
    rgb = torch.stack(rgbs, dim=0)
    goal = torch.stack(goals, dim=0)
    u_star = torch.stack(u_stars, dim=0)
    u_raw = torch.stack([m["u_raw"] for m in metas], dim=0)
    return rgb, goal, u_star, u_raw


def _make_loaders(cfg: dict, processed_root: Path):
    use_jitter = bool(cfg["train"].get("color_jitter", True))
    zero_goal = bool(cfg["train"].get("zero_goal_heading", False))
    goal_input_dim = int(cfg["model"].get("goal_input_dim", 2))
    goal_distance_scale = float(cfg["model"].get("goal_distance_scale", 5.0))
    data_cfg = cfg.get("data", {})
    cache_in_mem = bool(data_cfg.get("cache_blobs_in_memory", False))
    cache_lru = int(data_cfg.get("cache_lru_size", 64))
    ds_kw = dict(
        cache_blobs_in_memory=cache_in_mem,
        cache_lru_size=cache_lru,
        zero_goal_heading=zero_goal,
        goal_input_dim=goal_input_dim,
        goal_distance_scale=goal_distance_scale,
    )
    train_ds = RGBHorizonDataset(
        processed_root, split="train",
        use_color_jitter=use_jitter,
        **ds_kw,
    )
    val_ds = RGBHorizonDataset(
        processed_root, split="val",
        use_color_jitter=False,
        **ds_kw,
    )
    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=pin,
        drop_last=True,
        collate_fn=_collate,
        persistent_workers=cfg["train"].get("num_workers", 4) > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=min(2, cfg["train"].get("num_workers", 4)),
        pin_memory=pin,
        drop_last=False,
        collate_fn=_collate,
    )
    return train_ds, val_ds, train_dl, val_dl


def _build_model(cfg: dict) -> RGBVelocityPolicy:
    m = RGBVelocityPolicy(
        T=int(cfg["window"]["T"]),
        H=int(cfg["window"]["H"]),
        cmd_dim=int(cfg["model"]["cmd_dim"]),
        gru_hidden=int(cfg["model"]["gru_hidden"]),
        gru_layers=int(cfg["model"]["gru_layers"]),
        mlp_hidden=tuple(cfg["model"]["mlp_hidden"]),
        mlp_dropout=float(cfg["model"].get("mlp_dropout", 0.1)),
        goal_emb_dim=int(cfg["model"].get("goal_emb_dim", 32)),
        goal_input_dim=int(cfg["model"].get("goal_input_dim", 2)),
        freeze_stem_and_layer1=bool(cfg["model"].get("freeze_stem_and_layer1", True)),
    )
    return m


def _run_epoch(model: RGBVelocityPolicy,
               loader: DataLoader,
               stats: CommandStats,
               device: torch.device,
               optimizer=None,
               scaler=None,
               lambda_smooth: float = 0.05,
               grad_clip: float = 1.0,
               log_every: int = 50,
               header: str = "") -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals: Dict[str, float] = defaultdict(float)
    n_batches = 0
    t0 = time.time()
    for it, (rgb, goal, u_star, u_raw) in enumerate(loader):
        rgb = rgb.to(device, non_blocking=True)
        goal = goal.to(device, non_blocking=True)
        u_star = u_star.to(device, non_blocking=True)
        u_raw = u_raw.to(device, non_blocking=True)

        autocast_ctx = torch.cuda.amp.autocast if scaler is not None else _NullCtx
        with autocast_ctx():
            u_hat = model(rgb, goal)
            losses = bc_loss(u_hat, u_star, lambda_smooth=lambda_smooth)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(losses.total).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.total.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        with torch.no_grad():
            metrics = per_component_mse(u_hat.detach().float(), u_raw, stats)
        totals["loss_total"] += float(losses.total.item())
        totals["loss_cmd"] += float(losses.cmd.item())
        totals["loss_smooth"] += float(losses.smooth.item())
        for k, v in metrics.items():
            totals[k] += float(v.item())
        n_batches += 1

        if train_mode and (it + 1) % log_every == 0:
            avg = {k: v / n_batches for k, v in totals.items()}
            print(
                f"  {header} it={it + 1:>5d}  "
                f"loss={avg['loss_total']:.4f}  cmd={avg['loss_cmd']:.4f}  "
                f"smooth={avg['loss_smooth']:.4f}  "
                f"mse_lin={avg['mse_lin_vel']:.4f}  mse_psi={avg['mse_psi_dot']:.4f}",
                flush=True,
            )

    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    avg["sec_per_epoch"] = time.time() - t0
    avg["n_batches"] = n_batches
    return avg


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __call__(self): return self


def train(config_path: Path,
          checkpoint_dir_override: Optional[Path] = None,
          run_tag_override: Optional[str] = None,
          resume_from: Optional[Path] = None) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    base = config_path.resolve().parent.parent

    # CLI overrides for ablation / per-round runs.  These mutate the in-memory
    # config dict so they also get stamped into the saved checkpoint, which
    # means downstream eval scripts pick them up automatically.
    if checkpoint_dir_override is not None:
        cfg.setdefault("train", {})["checkpoint_dir"] = str(checkpoint_dir_override)
    if run_tag_override is not None:
        cfg.setdefault("train", {})["run_tag"] = str(run_tag_override)

    processed_root = (base / cfg["data"]["processed_root"]).resolve()
    ckpt_dir = (base / cfg["train"]["checkpoint_dir"]).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint_dir] {ckpt_dir}")
    if "run_tag" in cfg.get("train", {}):
        print(f"[run_tag]        {cfg['train']['run_tag']}")

    _set_seed(int(cfg["train"].get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  cuda_available={torch.cuda.is_available()}")

    train_ds, val_ds, train_dl, val_dl = _make_loaders(cfg, processed_root)
    stats = train_ds.stats
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  "
          f"T={train_ds.T}  H={train_ds.H}  S={train_ds.image_size}")
    print(f"[stats] mean={stats.mean.tolist()}  std={stats.std.tolist()}")

    model = _build_model(cfg).to(device)
    if resume_from is not None:
        resume_path = (base / resume_from).resolve()
        blob = torch.load(resume_path, weights_only=False, map_location=device)
        model.load_state_dict(blob["model"])
        print(f"[resume] warm-started from {resume_path}")
    print(f"[model] trainable_params={count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    use_amp = bool(cfg["train"].get("amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    log_path = ckpt_dir / "log.csv"
    log_fields = [
        "epoch", "train_loss_total", "train_loss_cmd", "train_loss_smooth",
        "train_mse_lin_vel", "train_mse_psi_dot",
        "val_loss_total", "val_loss_cmd", "val_mse_lin_vel", "val_mse_psi_dot",
        "val_mse_vx", "val_mse_vy", "val_mse_vz",
        "sec",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(log_fields)

    best_val = math.inf
    for epoch in range(int(cfg["train"]["epochs"])):
        header = f"[epoch {epoch + 1:>3d}]"
        tr = _run_epoch(
            model, train_dl, stats, device,
            optimizer=optimizer, scaler=scaler,
            lambda_smooth=float(cfg["train"]["lambda_smooth"]),
            grad_clip=float(cfg["train"].get("grad_clip", 1.0)),
            log_every=int(cfg["train"].get("log_every", 50)),
            header=header,
        )
        va = _run_epoch(
            model, val_dl, stats, device,
            optimizer=None, scaler=None,
            lambda_smooth=float(cfg["train"]["lambda_smooth"]),
            header=header + " [val]",
        )

        print(
            f"{header} "
            f"train_loss={tr['loss_total']:.4f}  val_loss={va['loss_total']:.4f}  "
            f"val_mse_lin={va['mse_lin_vel']:.4f}  val_mse_psi={va['mse_psi_dot']:.4f}  "
            f"sec={tr['sec_per_epoch']:.1f}",
            flush=True,
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                tr["loss_total"], tr["loss_cmd"], tr["loss_smooth"],
                tr["mse_lin_vel"], tr["mse_psi_dot"],
                va["loss_total"], va["loss_cmd"],
                va["mse_lin_vel"], va["mse_psi_dot"],
                va["mse_vx"], va["mse_vy"], va["mse_vz"],
                tr["sec_per_epoch"],
            ])

        state = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch + 1,
            "val_loss": va["loss_total"],
            "val_mse_overall": va["mse_overall"],
            "stats": stats.to_dict(),
        }
        torch.save(state, ckpt_dir / "bc_latest.pt")
        if va["mse_overall"] < best_val:
            best_val = va["mse_overall"]
            torch.save(state, ckpt_dir / "bc_best.pt")
            print(f"  -> saved bc_best.pt (val_mse_overall={best_val:.4f})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Train RGBVelocityPolicy via behavior cloning.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Override train.checkpoint_dir from the YAML.  Useful for keeping "
             "BC and per-round DAgger checkpoints in separate folders so the "
             "ablation table can later cite each one.",
    )
    p.add_argument(
        "--run-tag", type=str, default=None,
        help="Override train.run_tag from the YAML; stamped into the saved "
             "checkpoint config and propagated by downstream eval scripts to "
             "summary.json so the ablation collector can identify the run.",
    )
    p.add_argument(
        "--resume-from", type=Path, default=None,
        help="Path to a checkpoint (.pt) whose model weights are loaded before "
             "training starts.  Use this for DAgger fine-tuning so the model "
             "warm-starts from the previous round rather than training from "
             "scratch on the aggregated dataset.",
    )
    args = p.parse_args()
    train(args.config,
          checkpoint_dir_override=args.checkpoint_dir,
          run_tag_override=args.run_tag,
          resume_from=args.resume_from)


if __name__ == "__main__":
    main()
