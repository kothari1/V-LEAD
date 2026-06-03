"""Classify BC failures from a Phase-A per_rollout.csv and emit a focused
TD3+BC rollouts list.

Modes:
  success      : goal_success == True
  yaw_only     : !success, pos_err <= POS_TOL, yaw_err >  YAW_TOL
  pos_only     : !success, pos_err >  POS_TOL, yaw_err <= YAW_TOL
  both         : !success, pos_err >  POS_TOL, yaw_err >  YAW_TOL
  collision    : termination == 'collision'
  near_miss    : !success, pos_err <= POS_TOL, yaw_err <= YAW_TOL
  warmup_error : empty name row from RuntimeError (skipped from focus pool)

Selection priority for failures: yaw_only -> both -> pos_only -> collision.
"""

import argparse
import csv
import random
import re
from pathlib import Path

POS_TOL = 0.5
YAW_TOL = 0.5

TRAIN_DIRS = {
    "071353": "data/raw/flightroom_ssv_exp_2026-05-22_071353",
    "071718": "data/raw/flightroom_ssv_exp_2026-05-22_071718",
}


def classify(row):
    name = row.get("name", "")
    if not name:
        return "warmup_error"
    if row.get("termination") == "collision":
        return "collision"
    gs = row.get("goal_success", "")
    if gs == "True":
        return "success"
    try:
        pos = float(row.get("final_goal_position_error_m", "nan"))
        yaw = float(row.get("final_goal_yaw_error_rad", "nan"))
    except ValueError:
        return "other"
    if pos <= POS_TOL and yaw > YAW_TOL:
        return "yaw_only"
    if pos > POS_TOL and yaw <= YAW_TOL:
        return "pos_only"
    if pos > POS_TOL and yaw > YAW_TOL:
        return "both"
    return "near_miss"


def name_to_yaml_entry(name, src_dir_key):
    m = re.match(r"flightroom_ssv_exp_q(\d{5})", name)
    if not m:
        raise ValueError(f"unparseable name: {name}")
    qidx = m.group(1)
    setup = f"{TRAIN_DIRS[src_dir_key]}/trajectories_val{qidx}.pt"
    return (
        f"  - {{name: fr_{src_dir_key}_q{qidx}, scene: flightroom_ssv_exp, "
        f"rollout: baseline,\n"
        f"     setup_from: {setup}, sub_idx: 0}}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="per_rollout.csv from Phase A")
    ap.add_argument("--src-dir-key", default="071353",
                    choices=list(TRAIN_DIRS.keys()),
                    help="which train timestamp the CSV was generated against")
    ap.add_argument("--n-failures", type=int, default=30)
    ap.add_argument("--n-success", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-yaml-snippet",
                    help="optional path to write the rollouts: yaml block")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            row["__mode"] = classify(row)
            rows.append(row)

    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["__mode"], []).append(r)

    print("=== mode breakdown ===")
    for m in ("success", "yaw_only", "pos_only", "both", "collision",
              "near_miss", "warmup_error", "other"):
        print(f"  {m:14s} {len(by_mode.get(m, []))}")
    print(f"  TOTAL          {len(rows)}")

    # rank within each failure bucket by failure severity (worse first)
    def yaw_severity(r):
        try:
            return float(r["final_goal_yaw_error_rad"])
        except Exception:
            return 0.0

    def pos_severity(r):
        try:
            return float(r["final_goal_position_error_m"])
        except Exception:
            return 0.0

    by_mode.get("yaw_only", []).sort(key=yaw_severity, reverse=True)
    by_mode.get("both", []).sort(key=lambda r: yaw_severity(r) + pos_severity(r),
                                  reverse=True)
    by_mode.get("pos_only", []).sort(key=pos_severity, reverse=True)
    by_mode.get("collision", []).sort(key=pos_severity, reverse=True)

    picked = []
    for mode in ("yaw_only", "both", "pos_only", "collision"):
        if len(picked) >= args.n_failures:
            break
        need = args.n_failures - len(picked)
        take = by_mode.get(mode, [])[:need]
        for r in take:
            picked.append((mode, r))

    # successes: random sample for diversity
    succ_pool = by_mode.get("success", []).copy()
    rng.shuffle(succ_pool)
    successes = succ_pool[:args.n_success]
    for r in successes:
        picked.append(("success", r))

    print(f"\n=== picked {len(picked)} ({args.n_failures} failures + "
          f"{args.n_success} successes) ===")
    summary = {}
    for mode, _ in picked:
        summary[mode] = summary.get(mode, 0) + 1
    for m, n in sorted(summary.items()):
        print(f"  {m:14s} {n}")

    snippet_lines = ["rollouts:"]
    for mode, r in picked:
        snippet_lines.append(name_to_yaml_entry(r["name"], args.src_dir_key))
        snippet_lines[-1] += f"  # {mode}"
    snippet = "\n".join(snippet_lines) + "\n"

    print("\n=== rollouts yaml snippet ===")
    print(snippet)

    if args.out_yaml_snippet:
        Path(args.out_yaml_snippet).write_text(snippet)
        print(f"wrote -> {args.out_yaml_snippet}")


if __name__ == "__main__":
    main()
