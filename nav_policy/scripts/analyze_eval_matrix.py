#!/usr/bin/env python3
"""Build a per-query success matrix across all eval ckpts in a queue batch.

Reads `per_rollout.csv` from each `<run_name>/` subdir under a queue
batch root and emits:
  - {batch_root}/query_matrix.csv  rows = query name, columns = ckpts,
                                   values = 1/0 (goal_success).
  - stdout: per-ckpt totals; queries that every ckpt succeeds or every
    ckpt fails; pairwise overlap stats vs a chosen baseline (BC seed).

Usage:
    python scripts/analyze_eval_matrix.py \
        --batch-root /project/.../_eval_batch_20260602 \
        --baseline checkpoints__bc_best_balanced_dagger_r12_new
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Set


def load_run_successes(run_dir: Path) -> Dict[str, int]:
    """Return {query_name: 1 if goal_success else 0} for one ckpt's eval."""
    csv_path = run_dir / "per_rollout.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    out: Dict[str, int] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            qname = row["name"]
            ok = str(row.get("goal_success", "")).strip().lower() == "true"
            out[qname] = int(ok)
    return out


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-root", type=Path, required=True,
                   help="The {output-root} from eval_queue, holding one "
                        "subfolder per ckpt with its per_rollout.csv.")
    p.add_argument("--baseline", type=str, default=None,
                   help="Run name to treat as the baseline for overlap "
                        "comparison (default: the first ckpt alphabetically "
                        "whose name contains 'bc_').")
    p.add_argument("--out", type=Path, default=None,
                   help="Override the output matrix CSV path.")
    args = p.parse_args()

    batch_root = args.batch_root.resolve()
    run_dirs = sorted(
        d for d in batch_root.iterdir()
        if d.is_dir() and (d / "per_rollout.csv").is_file()
    )
    if not run_dirs:
        raise SystemExit(f"No per_rollout.csv found under {batch_root}")

    # Load each run's per-query success.
    per_run: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for d in run_dirs:
        try:
            per_run[d.name] = load_run_successes(d)
        except FileNotFoundError as e:
            print(f"  skipping {d.name}: {e}")

    if not per_run:
        raise SystemExit("No usable per_rollout.csv files; nothing to analyze.")

    # Union of query names.
    all_queries = sorted({q for s in per_run.values() for q in s.keys()})
    print(f"[matrix] {len(per_run)} ckpts × {len(all_queries)} queries")

    # Pick baseline.
    if args.baseline:
        baseline = args.baseline
    else:
        baseline = next(
            (k for k in per_run if "bc_" in k.lower()),
            list(per_run.keys())[0],
        )
    print(f"[matrix] baseline = {baseline}\n")

    # Write CSV: query, then 1/0 per ckpt.
    out_path = args.out or (batch_root / "query_matrix.csv")
    ckpt_names = list(per_run.keys())
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query"] + ckpt_names)
        for q in all_queries:
            w.writerow([q] + [per_run[c].get(q, "") for c in ckpt_names])
    print(f"[matrix] wrote {out_path}\n")

    # Per-ckpt totals.
    print("=" * 64)
    print(" Per-ckpt successes / n_queries")
    print("=" * 64)
    for c in ckpt_names:
        s = sum(per_run[c].get(q, 0) for q in all_queries)
        n = len(per_run[c])
        marker = "  <- baseline" if c == baseline else ""
        print(f"  {c[:56]:<56} {s:>3}/{n:<3} = {100 * s / max(n, 1):5.1f}%{marker}")
    print()

    # Always-success and always-fail buckets.
    always_pass: List[str] = []
    always_fail: List[str] = []
    mixed: List[str] = []
    for q in all_queries:
        vals = [per_run[c].get(q, None) for c in ckpt_names]
        only = {v for v in vals if v is not None}
        if only == {1}:
            always_pass.append(q)
        elif only == {0}:
            always_fail.append(q)
        else:
            mixed.append(q)

    print("=" * 64)
    print(" Query difficulty buckets")
    print("=" * 64)
    print(f"  always pass  : {len(always_pass):>3}   (every ckpt succeeded)")
    print(f"  always fail  : {len(always_fail):>3}   (no ckpt ever succeeded)")
    print(f"  mixed        : {len(mixed):>3}   (at least one ckpt differs)")
    if always_fail:
        print(f"  always-fail queries (sample 10): {always_fail[:10]}")
    print()

    # Baseline overlap with every other ckpt.
    print("=" * 64)
    print(f" Overlap vs baseline ({baseline})")
    print("=" * 64)
    base_pass: Set[str] = {q for q in all_queries if per_run[baseline].get(q, 0) == 1}
    base_pass_n = len(base_pass)
    for c in ckpt_names:
        if c == baseline:
            continue
        c_pass = {q for q in all_queries if per_run[c].get(q, 0) == 1}
        inter = base_pass & c_pass
        only_base = base_pass - c_pass
        only_c = c_pass - base_pass
        jac = jaccard(base_pass, c_pass)
        print(
            f"  vs {c[:48]:<48}"
        )
        print(
            f"    both pass: {len(inter):>3}   "
            f"only baseline: {len(only_base):>3}   "
            f"only this:     {len(only_c):>3}   "
            f"jaccard: {jac:.3f}"
        )
        # Net change vs baseline.
        net = len(only_c) - len(only_base)
        sign = "+" if net >= 0 else ""
        print(
            f"    net delta: {sign}{net} successes "
            f"(BC pass-rate: {100 * base_pass_n / len(all_queries):.1f}% -> "
            f"this ckpt: {100 * len(c_pass) / len(all_queries):.1f}%)"
        )
    print()


if __name__ == "__main__":
    main()
