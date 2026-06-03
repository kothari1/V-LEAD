"""Aggregate full-110 eval results from the snapshot under results/logs/eval/.

Yields dicts with: label, algo, success, ci_lo, ci_hi, n, source.
"""

import json
from pathlib import Path

from _paths import FULL110_DIR

# canonical full-110 evals — (label, algo, snapshot summary.json tag)
SOURCES = [
    ("BC seed", "BC", "bc_seed"),
    ("SAC v7", "SAC", "sac_v7_best"),
    ("SAC v8", "SAC", "sac_v8_best"),
    ("PPO v6", "PPO", "ppo_v6_best"),
    ("SAC long-8h\n(_best)", "SAC", "sac_long8h_best"),
]


def load_summary(path):
    d = json.loads(Path(path).read_text())
    return {
        "success": d["goal_success_rate"],
        "ci_lo": d.get("goal_success_rate_ci95_lo"),
        "ci_hi": d.get("goal_success_rate_ci95_hi"),
        "n": d["n_rollouts"],
        "timeout": d.get("timeout_rate"),
        "collision": d.get("collision_rate"),
        "checkpoint": d.get("checkpoint"),
    }


def find_td3bc_full110_evals():
    """Glob the snapshot for any td3bc / failure_focus tag drop-ins.

    Returns list of (label, algo, summary_path).
    """
    found = []
    for sj in FULL110_DIR.glob("td3bc*.summary.json"):
        tag = sj.stem.replace(".summary", "")
        found.append((f"TD3+BC\n({tag[:24]})", "TD3+BC", sj))
    for sj in FULL110_DIR.glob("failure_focus*.summary.json"):
        tag = sj.stem.replace(".summary", "")
        found.append((f"TD3+BC\n({tag[:24]})", "TD3+BC", sj))
    return found


def load_canonical():
    """Return list of dicts for the canonical comparison set."""
    rows = []
    for label, algo, tag in SOURCES:
        path = FULL110_DIR / f"{tag}.summary.json"
        if not path.exists():
            continue
        row = load_summary(path)
        row.update(label=label, algo=algo, source=str(path))
        rows.append(row)
    for label, algo, path in find_td3bc_full110_evals():
        row = load_summary(path)
        row.update(label=label, algo=algo, source=str(path))
        rows.append(row)
    return rows


# Snapshot tags for the long_8h regression trajectory.  Episode count is
# parsed from the tag name where possible; _best / _latest go to the end.
LONG8H_TAGS = [
    (0,    "seed",    "sac_long8h_seed_baseline"),
    (400,  "ep 400",  "sac_long8h_iter0100_ep00400"),
    (800,  "ep 800",  "sac_long8h_iter0200_ep00800"),
    (1200, "ep 1200", "sac_long8h_iter0300_ep01200"),
    (1600, "ep 1600", "sac_long8h_iter0400_ep01600"),
    (1601, "best",    "sac_long8h_best"),
    (1602, "latest",  "sac_long8h_latest"),
]


def load_long8h_snapshots():
    rows = []
    for ep, label, tag in LONG8H_TAGS:
        path = FULL110_DIR / f"{tag}.summary.json"
        if not path.exists():
            continue
        row = load_summary(path)
        row.update(ep=ep, label=label, source=str(path))
        rows.append(row)
    rows.sort(key=lambda r: r["ep"])
    return rows


if __name__ == "__main__":
    print("=== canonical ===")
    for r in load_canonical():
        print(f"  {r['algo']:7s} {r['label']:30s} {r['success']*100:5.1f}% n={r['n']}")
    print("\n=== long_8h snapshots ===")
    for r in load_long8h_snapshots():
        print(f"  ep={r['ep']:5d} {r['label']:8s} {r['success']*100:5.1f}%")
