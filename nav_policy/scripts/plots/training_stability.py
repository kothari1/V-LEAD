"""F3: training stability comparison: vanilla SAC long-8h vs TD3+BC 80ep.

Reads TB scalar tags from each run's tb/ directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from _paths import TRAINING_DIR
from _style import ALGO_COLORS, apply_style, save_fig

TB_SAC = TRAINING_DIR / "rl_sac_dagger_r12_long_8h_v1" / "tb"
TB_TD3 = TRAINING_DIR / "rl_td3bc_dagger_r12_80ep" / "tb"


def load_scalar(tb_dir, tag):
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    return steps, vals


def main():
    apply_style()
    fig, (ax_eval, ax_ret) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # ---- LEFT: in-loop eval success rate over training episode ----
    s, v = load_scalar(TB_SAC, "eval/goal_success_rate")
    ax_eval.plot(s, v * 100, "-o", color=ALGO_COLORS["SAC"], lw=1.8,
                 markersize=5, label="SAC long-8h (holdout-30)")
    s, v = load_scalar(TB_TD3, "eval/goal_success_rate")
    ax_eval.plot(s, v * 100, "-s", color=ALGO_COLORS["TD3+BC"], lw=1.8,
                 markersize=5, label="TD3+BC 80ep (holdout-30)")

    sf_s, sf_v = load_scalar(TB_TD3, "eval/seed_floor")
    if len(sf_v):
        ax_eval.axhline(sf_v[0] * 100, color="#7f7f7f", linestyle="--",
                        linewidth=1.0, alpha=0.6)
        ax_eval.text(50, sf_v[0] * 100 + 1.8,
                     f"BC seed floor on holdout-30 = {sf_v[0]*100:.0f}%",
                     fontsize=8, color="#555")

    ax_eval.set_xlabel("training episode")
    ax_eval.set_ylabel("in-loop eval success (%)")
    ax_eval.set_title("In-loop hold-out eval: SAC drops below seed, TD3+BC holds")
    ax_eval.set_ylim(0, 100)
    ax_eval.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # ---- RIGHT: rollout mean_return over training (collapse signal) ----
    s, v = load_scalar(TB_SAC, "rollout/mean_return")
    ax_ret.plot(s, v, "-", color=ALGO_COLORS["SAC"], lw=1.2, alpha=0.85,
                label=f"SAC long-8h (min={v.min():.0f})")

    s, v = load_scalar(TB_TD3, "rollout/mean_return")
    ax_ret.plot(s, v, "-s", color=ALGO_COLORS["TD3+BC"], lw=1.6,
                markersize=4, label=f"TD3+BC 80ep (min={v.min():.0f})")

    ax_ret.axhline(0, color="#bbb", linewidth=0.8, alpha=0.6)
    ax_ret.set_xlabel("training iteration")
    ax_ret.set_ylabel("rollout mean return")
    ax_ret.set_title("Rollout return: vanilla SAC dives below -100, TD3+BC bounded")
    ax_ret.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.suptitle("Why we switched from vanilla SAC to residual TD3+BC",
                 fontsize=12, y=1.01, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "training_stability")


if __name__ == "__main__":
    main()
