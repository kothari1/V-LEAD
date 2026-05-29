"""
Plot offline evaluation results: predicted vs ground-truth velocity commands.

Reads the artifacts written by eval_offline.py:
    <eval_dir>/predictions.npz     { u_hat [N,H,4], u_raw [N,H,4] }
    <eval_dir>/per_horizon.csv     RMSE per horizon step per component
    <eval_dir>/summary.json        aggregate metrics

Writes to <eval_dir>/:
    scatter_h0.png          Pred vs GT scatter at horizon step 0 (first executed cmd)
    scatter_hmean.png       Pred vs GT scatter averaged over all H steps
    rmse_by_horizon.png     RMSE degradation as a function of horizon step
    error_hist.png          Error histograms per component at h=0

Usage:
    python scripts/plot_eval_offline.py --eval-dir data/eval/fm_offline
    python scripts/plot_eval_offline.py --eval-dir data/eval/fm_offline --show
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CMD_NAMES = ("vx [m/s]", "vy [m/s]", "vz [m/s]", "ψ̇ [rad/s]")
CMD_KEYS  = ("vx", "vy", "vz", "psi_dot")


def _load_artifacts(eval_dir: Path):
    pred_path = eval_dir / "predictions.npz"
    horizon_path = eval_dir / "per_horizon.csv"
    summary_path = eval_dir / "summary.json"

    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} not found — run eval_offline.py first")

    data = np.load(pred_path, allow_pickle=True)
    u_hat = data["u_hat"].astype(np.float32)   # [N, H, 4]
    u_raw = data["u_raw"].astype(np.float32)   # [N, H, 4]

    horizon_rows = []
    if horizon_path.exists():
        with open(horizon_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                horizon_rows.append({k: float(v) for k, v in row.items()})

    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return u_hat, u_raw, horizon_rows, summary


def plot_scatter(u_hat: np.ndarray, u_raw: np.ndarray, h: int,
                 eval_dir: Path, show: bool, tag: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Predicted vs Ground-Truth  (horizon step h={h})", fontsize=13)

    for c, (ax, cname) in enumerate(zip(axes, CMD_NAMES)):
        pred = u_hat[:, h, c]
        gt   = u_raw[:, h, c]
        lo = min(pred.min(), gt.min())
        hi = max(pred.max(), gt.max())
        pad = (hi - lo) * 0.05 + 1e-6

        ax.scatter(gt, pred, s=1, alpha=0.15, color="steelblue", rasterized=True)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "r--", linewidth=1, label="y=x")
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        ax.set_title(f"{cname}\nRMSE={rmse:.4f}", fontsize=10)
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Predicted")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    out = eval_dir / f"scatter_{tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out}")
    if show:
        plt.show()
    plt.close()


def plot_rmse_by_horizon(horizon_rows: list, eval_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    if not horizon_rows:
        print("[plot] per_horizon.csv empty — skipping RMSE-by-horizon plot")
        return

    H = len(horizon_rows)
    steps = [int(r["horizon_step"]) for r in horizon_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for c, (key, cname, col) in enumerate(zip(CMD_KEYS, CMD_NAMES, colors)):
        rmse_col = f"rmse_{key}"
        vals = [r[rmse_col] for r in horizon_rows]
        ax.plot(steps, vals, marker="o", markersize=4, label=cname, color=col)

    ax.set_xlabel("Horizon step h")
    ax.set_ylabel("RMSE (physical units)")
    ax.set_title("Prediction RMSE vs Horizon Step")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4)
    ax.set_xlim(-0.5, H - 0.5)

    out = eval_dir / "rmse_by_horizon.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out}")
    if show:
        plt.show()
    plt.close()


def plot_error_hist(u_hat: np.ndarray, u_raw: np.ndarray,
                    eval_dir: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    err = u_hat[:, 0, :] - u_raw[:, 0, :]   # [N, 4] errors at h=0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Prediction Error Distribution (horizon step h=0)", fontsize=13)

    for c, (ax, cname) in enumerate(zip(axes, CMD_NAMES)):
        e = err[:, c]
        ax.hist(e, bins=60, color="steelblue", alpha=0.8, edgecolor="none")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{cname}\nμ={e.mean():.4f}  σ={e.std():.4f}", fontsize=10)
        ax.set_xlabel("Error (pred − GT)")
        ax.set_ylabel("Count")
        ax.grid(True, linewidth=0.4, axis="y")

    plt.tight_layout()
    out = eval_dir / "error_hist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out}")
    if show:
        plt.show()
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot offline eval artifacts.")
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="Directory written by eval_offline.py (contains predictions.npz etc.)")
    p.add_argument("--show", action="store_true",
                   help="Also open the plots in a window (requires a display).")
    args = p.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")   # non-interactive backend when --show not requested

    eval_dir = args.eval_dir.resolve()
    u_hat, u_raw, horizon_rows, summary = _load_artifacts(eval_dir)

    print(f"[plot] loaded  u_hat={u_hat.shape}  u_raw={u_raw.shape}")
    if summary:
        rmse = summary.get("rmse_overall", {})
        print(f"[plot] RMSE: vx={rmse.get('vx','?'):.4f}  vy={rmse.get('vy','?'):.4f}"
              f"  vz={rmse.get('vz','?'):.4f}  psi_dot={rmse.get('psi_dot','?'):.4f}")

    # Scatter at h=0 (the step the policy actually executes)
    plot_scatter(u_hat, u_raw, h=0, eval_dir=eval_dir, show=args.show, tag="h0")

    # Scatter averaged over all H steps (overall prediction quality)
    u_hat_mean = u_hat.mean(axis=1, keepdims=True)   # [N,1,4]
    u_raw_mean = u_raw.mean(axis=1, keepdims=True)
    plot_scatter(u_hat_mean, u_raw_mean, h=0,
                 eval_dir=eval_dir, show=args.show, tag="hmean")

    # RMSE degradation over horizon
    plot_rmse_by_horizon(horizon_rows, eval_dir=eval_dir, show=args.show)

    # Error histograms at h=0
    plot_error_hist(u_hat, u_raw, eval_dir=eval_dir, show=args.show)

    print(f"[plot] all figures saved to {eval_dir}/")


if __name__ == "__main__":
    main()
