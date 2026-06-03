"""F4: Algorithm-grouped bar comparing SAC variants vs PPO vs TD3+BC vs BC seed."""

import matplotlib.pyplot as plt
import numpy as np

from _eval_data import load_canonical
from _style import ALGO_COLORS, BC_SEED_FULL110, apply_style, save_fig


def main():
    apply_style()
    rows = load_canonical()
    # group order
    order = ["BC", "SAC", "PPO", "TD3+BC"]
    groups = {a: [] for a in order}
    for r in rows:
        groups[r["algo"]].append(r)

    # build x positions: per-group cluster
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x_pos, x_labels, bar_colors, rates, lo_err, hi_err, ns = [], [], [], [], [], [], []
    cursor = 0
    group_centers = []
    for algo in order:
        grp = groups[algo]
        if not grp:
            continue
        start = cursor
        for r in grp:
            x_pos.append(cursor)
            x_labels.append(r["label"].replace("\n", " "))
            bar_colors.append(ALGO_COLORS[algo])
            rates.append(r["success"] * 100)
            lo_err.append((r["success"] - (r["ci_lo"] or r["success"])) * 100)
            hi_err.append(((r["ci_hi"] or r["success"]) - r["success"]) * 100)
            ns.append(r["n"])
            cursor += 1
        group_centers.append((algo, (start + cursor - 1) / 2.0))
        cursor += 0.8  # gap

    ax.bar(x_pos, rates, color=bar_colors, edgecolor="black", linewidth=0.4,
           yerr=[lo_err, hi_err], capsize=3,
           error_kw={"elinewidth": 0.8, "ecolor": "#444"})
    ax.set_xticks(x_pos, x_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("full-110 success rate (%)")
    ax.set_title("Algorithm comparison on identical BC seed + env")
    ax.set_ylim(0, 80)

    # BC seed dashed reference
    ax.axhline(BC_SEED_FULL110 * 100, color="#7f7f7f", linestyle="--",
               linewidth=1.0, alpha=0.7, zorder=0)

    # annotate bars
    for xp, r in zip(x_pos, rates):
        ax.text(xp, r + 2.0, f"{r:.1f}%", ha="center", fontsize=8.5)

    # group labels below x-axis
    for algo, center in group_centers:
        ax.text(center, -16.5, algo, ha="center", fontsize=10,
                fontweight="bold", color=ALGO_COLORS[algo],
                transform=ax.transData)

    ax.set_xlim(-0.7, cursor)
    fig.subplots_adjust(bottom=0.2)
    save_fig(fig, "algo_comparison")


if __name__ == "__main__":
    main()
