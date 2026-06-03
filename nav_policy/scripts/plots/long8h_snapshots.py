"""F5: SAC long_8h full-110 success vs training episode (regression story)."""

import matplotlib.pyplot as plt
import numpy as np

from _eval_data import load_long8h_snapshots
from _style import ALGO_COLORS, BC_SEED_FULL110, apply_style, save_fig


def main():
    apply_style()
    rows = load_long8h_snapshots()
    # Separate the snapshot trace from _best / _latest markers
    eps_snap = [r["ep"] for r in rows if r["ep"] <= 1600]
    rate_snap = [r["success"] * 100 for r in rows if r["ep"] <= 1600]
    lo_snap = [(r["ci_lo"] or r["success"]) * 100 for r in rows if r["ep"] <= 1600]
    hi_snap = [(r["ci_hi"] or r["success"]) * 100 for r in rows if r["ep"] <= 1600]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    color = ALGO_COLORS["SAC"]
    ax.plot(eps_snap, rate_snap, "-o", color=color, lw=1.6,
            markersize=5, markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.2, label="iter snapshots (every 400 ep)")
    ax.fill_between(eps_snap, lo_snap, hi_snap, color=color, alpha=0.13)

    # _best / _latest annotations
    best = next((r for r in rows if r["label"] == "best"), None)
    latest = next((r for r in rows if r["label"] == "latest"), None)
    if best:
        ax.scatter([1700], [best["success"] * 100], marker="*", s=180,
                   color="#2ca02c", zorder=5, edgecolors="black", linewidth=0.7,
                   label=f"_best.pt = {best['success']*100:.1f}%")
        ax.annotate("_best", (1700, best["success"] * 100),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=7.5, color="#2ca02c")
    if latest:
        ax.scatter([1820], [latest["success"] * 100], marker="X", s=90,
                   color="#d62728", zorder=5, edgecolors="black", linewidth=0.7,
                   label=f"_latest.pt = {latest['success']*100:.1f}%")
        ax.annotate("_latest", (1820, latest["success"] * 100),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=7.5, color="#d62728")

    # BC seed reference
    ax.axhline(BC_SEED_FULL110 * 100, color="#7f7f7f", linestyle="--",
               linewidth=1.0, alpha=0.8)
    ax.text(50, BC_SEED_FULL110 * 100 + 1.5,
            f"BC seed baseline = {BC_SEED_FULL110*100:.1f}%",
            fontsize=8, color="#555")

    # annotate each snapshot point
    for ep, rate in zip(eps_snap[1:], rate_snap[1:]):  # skip seed
        ax.annotate(f"{rate:.1f}%", (ep, rate),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color="#222")

    ax.set_xlabel("training episode")
    ax.set_ylabel("full-110 success rate (%)")
    ax.set_title("vanilla SAC long-8h: full-110 over training (regression)")
    ax.set_xlim(-50, 1900)
    ax.set_ylim(0, 75)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    save_fig(fig, "long8h_snapshots")


if __name__ == "__main__":
    main()
