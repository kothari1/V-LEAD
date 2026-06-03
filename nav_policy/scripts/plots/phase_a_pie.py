"""F8: Phase-A BC failure-mining bucket pie. Uses bucket_failures.classify."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# import classify() from bucket_failures.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bucket_failures import classify  # noqa: E402

from _paths import PHASE_A_DIR
from _style import apply_style, save_fig

CSV = PHASE_A_DIR / "bc_seed.per_rollout.csv"


def main():
    apply_style()
    counts = {}
    with CSV.open() as f:
        for row in csv.DictReader(f):
            mode = classify(row)
            counts[mode] = counts.get(mode, 0) + 1
    print("  mode counts:", counts)

    # focus on failure modes only (drop success)
    fail_counts = {k: v for k, v in counts.items()
                   if k not in ("success", "warmup_error")}
    order = ["yaw_only", "both", "pos_only", "collision", "near_miss", "other"]
    labels = [k for k in order if fail_counts.get(k, 0) > 0]
    vals = [fail_counts[k] for k in labels]
    pretty = {"yaw_only": "yaw only",
              "both": "pos + yaw",
              "pos_only": "pos only",
              "collision": "collision",
              "near_miss": "near miss",
              "other": "other"}
    pretty_labels = [f"{pretty[k]}\n({v})" for k, v in zip(labels, vals)]
    colors = {"yaw_only": "#1f77b4", "both": "#9467bd", "pos_only": "#ff9f4a",
              "collision": "#d62728", "near_miss": "#bcbd22", "other": "#7f7f7f"}
    pie_colors = [colors[k] for k in labels]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    wedges, txts, autotxts = ax.pie(
        vals, labels=pretty_labels, colors=pie_colors,
        autopct=lambda p: f"{p:.0f}%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9})
    for t in autotxts:
        t.set_color("white"); t.set_fontweight("bold"); t.set_fontsize(9)
    total = sum(vals)
    ax.set_title(f"Phase-A BC failure-mining: {total} failures by mode\n"
                 "(071353 train pool, shuffled w/ early-stop at 50 fails)",
                 fontsize=10)
    ax.grid(False)
    save_fig(fig, "phase_a_pie")


if __name__ == "__main__":
    main()
