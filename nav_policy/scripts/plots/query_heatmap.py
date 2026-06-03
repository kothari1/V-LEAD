"""F7: 110-query x ckpt success heatmap. Reveals shared-pass, regression-
risk, and always-fail bands."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from _paths import FULL110_DIR
from _style import apply_style, save_fig

SOURCES = [
    ("BC seed", FULL110_DIR / "bc_seed.per_rollout.csv"),
    ("SAC v7",  FULL110_DIR / "sac_v7_best.per_rollout.csv"),
    ("SAC v8",  FULL110_DIR / "sac_v8_best.per_rollout.csv"),
    ("PPO v6",  FULL110_DIR / "ppo_v6_best.per_rollout.csv"),
]


def load_success(csv_path):
    out = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            name = row.get("name") or ""
            if not name:
                continue
            out[name] = row.get("goal_success") == "True"
    return out


def main():
    apply_style()
    success_maps = []
    for label, path in SOURCES:
        if not path.exists():
            continue
        success_maps.append((label, load_success(path)))

    # universe of queries = union of names
    all_names = sorted({n for _, m in success_maps for n in m})
    matrix = np.zeros((len(all_names), len(success_maps)), dtype=int)
    for j, (_, m) in enumerate(success_maps):
        for i, n in enumerate(all_names):
            matrix[i, j] = 1 if m.get(n, False) else 0

    # sort queries by row-sum descending (top: always-pass, bottom: always-fail)
    row_sum = matrix.sum(axis=1)
    order = np.argsort(-row_sum, kind="stable")
    matrix = matrix[order]
    names_sorted = [all_names[i] for i in order]

    # classify regions for annotation bands
    n_ckpt = matrix.shape[1]
    always_pass = int(np.sum(row_sum == n_ckpt))
    always_fail = int(np.sum(row_sum == 0))
    mixed = matrix.shape[0] - always_pass - always_fail
    print(f"  queries: total={matrix.shape[0]}  always_pass={always_pass}"
          f"  mixed={mixed}  always_fail={always_fail}")

    seed_idx = 0  # BC seed column
    seed_passes = matrix[:, seed_idx].sum()
    seed_pass_but_rl_breaks = int(np.sum(
        (matrix[:, seed_idx] == 1) & (matrix[:, 1:].min(axis=1) == 0)))
    seed_fail_but_rl_solves = int(np.sum(
        (matrix[:, seed_idx] == 0) & (matrix[:, 1:].max(axis=1) == 1)))
    print(f"  seed_passes={seed_passes}, regression-risk={seed_pass_but_rl_breaks},"
          f" winnable={seed_fail_but_rl_solves}")

    fig, ax = plt.subplots(figsize=(4.5, 8.0))
    cmap = ListedColormap(["#d62728", "#2ca02c"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   norm=BoundaryNorm([0, 0.5, 1], 2),
                   interpolation="nearest")

    ax.set_xticks(range(len(success_maps)),
                  [lab for lab, _ in success_maps],
                  rotation=30, ha="right", fontsize=9)
    ax.set_yticks([])  # 110 rows; too dense
    ax.set_ylabel(f"queries (n={matrix.shape[0]}, "
                  f"sorted by # ckpts passing)")
    ax.set_title("Per-query success matrix (full-110)\n"
                 "green = pass, red = fail")

    # vertical lines between columns
    for x in range(1, len(success_maps)):
        ax.axvline(x - 0.5, color="white", linewidth=1.0)

    # annotate bands
    ax.text(len(success_maps) + 0.15, always_pass / 2,
            f"shared pass\n{always_pass}",
            fontsize=8, color="#2ca02c", va="center", ha="left",
            fontweight="bold")
    if mixed > 0:
        ax.text(len(success_maps) + 0.15,
                always_pass + mixed / 2,
                f"mixed\n{mixed}",
                fontsize=8, color="#bb8800", va="center", ha="left",
                fontweight="bold")
    ax.text(len(success_maps) + 0.15,
            always_pass + mixed + always_fail / 2,
            f"shared fail\n{always_fail}",
            fontsize=8, color="#d62728", va="center", ha="left",
            fontweight="bold")

    # caption box
    caption = (f"BC seed: {seed_passes} passes  •  "
               f"RL breaks {seed_pass_but_rl_breaks} seed passes  •  "
               f"RL solves {seed_fail_but_rl_solves} new fail")
    fig.text(0.5, -0.01, caption, ha="center", fontsize=8, color="#333")

    ax.grid(False)
    fig.tight_layout()
    save_fig(fig, "query_heatmap")


if __name__ == "__main__":
    main()
