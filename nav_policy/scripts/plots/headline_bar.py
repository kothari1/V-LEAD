"""F2: Headline horizontal bar of full-110 success rate across ckpts."""

import matplotlib.pyplot as plt
import numpy as np

from _eval_data import load_canonical
from _style import ALGO_COLORS, BC_SEED_FULL110, apply_style, save_fig


def main():
    apply_style()
    rows = load_canonical()
    # sort descending by success
    rows.sort(key=lambda r: r["success"], reverse=True)

    labels = [r["label"] for r in rows]
    rates = np.array([r["success"] for r in rows])
    lo = np.array([(r["ci_lo"] or r["success"]) for r in rows])
    hi = np.array([(r["ci_hi"] or r["success"]) for r in rows])
    err_low = rates - lo
    err_high = hi - rates
    colors = [ALGO_COLORS[r["algo"]] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(rows) + 1.2))
    y = np.arange(len(rows))
    bars = ax.barh(y, rates * 100, color=colors, edgecolor="black", linewidth=0.4,
                   xerr=[err_low * 100, err_high * 100], capsize=3,
                   error_kw={"elinewidth": 0.8, "ecolor": "#444"})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Goal success rate (%)")
    ax.set_xlim(0, 92)
    ax.set_title("Held-out test set (full-110): success rate by checkpoint",
                 pad=12)

    # BC seed reference
    ax.axvline(BC_SEED_FULL110 * 100, color="#7f7f7f", linestyle="--",
               linewidth=1.0, alpha=0.7, zorder=0)
    ax.text(BC_SEED_FULL110 * 100 + 0.5, -0.85,
            "BC seed", fontsize=8, color="#555", va="top", ha="left")

    # annotate each bar
    for yi, r in enumerate(rows):
        x = r["success"] * 100
        ax.text(x + 1.0 + (err_high[yi] * 100), yi,
                f"{x:.1f}% ({int(round(x/100*r['n']))}/{r['n']})",
                va="center", fontsize=8, color="#222")

    # legend OUTSIDE plot to avoid overlap with bars
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.4)
               for c in ALGO_COLORS.values()]
    ax.legend(handles, list(ALGO_COLORS.keys()),
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              title="algorithm", framealpha=0.9, fontsize=8)

    save_fig(fig, "headline_bar")


if __name__ == "__main__":
    main()
