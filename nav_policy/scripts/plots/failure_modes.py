"""F6: BC seed failure-mode pie + timeout final-pos-err histogram."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _paths import FULL110_DIR
from _style import apply_style, save_fig

CSV = FULL110_DIR / "bc_seed.per_rollout.csv"


def main():
    apply_style()
    rows = []
    with CSV.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)

    fails = [r for r in rows if r.get("goal_success") != "True"]
    # mode counts
    counts = {"timeout": 0, "collision": 0, "exception": 0}
    timeout_pos_err = []
    for r in fails:
        term = (r.get("termination") or "").lower()
        if term == "collision":
            counts["collision"] += 1
        elif r.get("error") and r.get("error").strip():
            counts["exception"] += 1
        else:
            counts["timeout"] += 1
            try:
                timeout_pos_err.append(float(r["final_goal_position_error_m"]))
            except (KeyError, ValueError):
                pass

    print(f"  {len(fails)} failures: {counts}")
    print(f"  timeout pos err: median={np.median(timeout_pos_err):.2f} m,"
          f" max={max(timeout_pos_err):.2f} m,"
          f" <0.75m: {sum(1 for x in timeout_pos_err if x < 0.75)}")

    fig, (ax_pie, ax_hist) = plt.subplots(1, 2, figsize=(8.5, 3.4),
                                          gridspec_kw={"width_ratios": [1, 1.4]})

    # ---- LEFT: pie ----
    pie_labels = [f"timeout\n({counts['timeout']})",
                  f"collision\n({counts['collision']})",
                  f"exception\n({counts['exception']})"]
    pie_vals = [counts["timeout"], counts["collision"], counts["exception"]]
    pie_colors = ["#ff9f4a", "#d62728", "#9467bd"]
    wedges, txts, autotxts = ax_pie.pie(
        pie_vals, labels=pie_labels, colors=pie_colors,
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9})
    for t in autotxts:
        t.set_color("white"); t.set_fontweight("bold")
    ax_pie.set_title(f"BC seed failure modes (n={len(fails)})", fontsize=10)
    ax_pie.grid(False)

    # ---- RIGHT: histogram of timeout final pos err ----
    bins = np.arange(0, 1.6, 0.1)
    ax_hist.hist(timeout_pos_err, bins=bins, color="#ff9f4a",
                 edgecolor="black", linewidth=0.4, alpha=0.85)
    ax_hist.axvline(0.5, color="#2ca02c", linestyle="--", linewidth=1.5,
                    label="success threshold (0.5 m)")
    median = float(np.median(timeout_pos_err))
    ax_hist.axvline(median, color="#1f77b4", linestyle=":", linewidth=1.5,
                    label=f"median = {median:.2f} m")
    ax_hist.set_xlabel("final goal position error (m)")
    ax_hist.set_ylabel("# timeouts")
    n_near = sum(1 for x in timeout_pos_err if x < 0.75)
    ax_hist.set_title(f"Timeout terminal pos error: drone parks near goal\n"
                      f"{n_near}/{len(timeout_pos_err)} within 0.75 m",
                      fontsize=10)
    ax_hist.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_hist.set_xlim(0, 1.5)

    fig.suptitle("Where the BC seed loses points on full-110",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "failure_modes")


if __name__ == "__main__":
    main()
