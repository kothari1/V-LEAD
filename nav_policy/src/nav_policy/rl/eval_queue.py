"""Queue multiple closed-loop evals serially.

Designed for overnight batches: hand it a list of checkpoint paths + the
eval config + the rollouts dir, and it runs `evaluate()` on each one in
sequence, writes a per-ckpt summary, and prints a final roll-up table.

Usage examples (inside container, cwd = /workspace/nav_policy):

    # Multiple --ckpt flags
    python -m nav_policy.rl.eval_queue \\
        --config configs/eval_closed_loop_flightroom_holdout_14.yaml \\
        --output-root /project/kothari1/vlead_data/rl_runs/dagger_r12/_eval_batch \\
        --rollouts-from-dir data/raw/flightroom_ssv_exp_2026-05-22_071733_trajs-110 \\
        --ckpt /project/.../rl_sac_dagger_r12_v7/rl_sac_dagger_r12_v7_best.pt \\
        --ckpt /project/.../rl_sac_dagger_r12_v8/rl_sac_dagger_r12_v8_best.pt \\
        --ckpt data/checkpoints/bc_best_balanced_dagger_r12_new.pt

    # Or from a text file (one path per line; blanks + #-comments ignored)
    python -m nav_policy.rl.eval_queue \\
        --config configs/eval_closed_loop_flightroom_holdout_14.yaml \\
        --output-root /project/.../_eval_batch \\
        --rollouts-from-dir data/raw/flightroom_ssv_exp_2026-05-22_071733_trajs-110 \\
        --ckpt-list /tmp/eval_targets.txt
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from nav_policy.evaluate.closed_loop import evaluate


def _read_ckpt_list_file(path: Path) -> List[Path]:
    out: List[Path] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(Path(line))
    return out


def _derive_run_name(ckpt: Path) -> str:
    """Build a unique output-subdir name for this ckpt.

    Combines the parent dir name (which is usually the run_tag) with the
    file stem so a single batch can mix multiple runs without collisions.
    """
    parent = ckpt.parent.name or "ckpt"
    stem = ckpt.stem
    if stem in parent:
        return parent
    return f"{parent}__{stem}"


def _success_rate_from_summary(summary: Dict[str, Any]) -> Optional[float]:
    """Try a few summary fields to extract a single success-rate number."""
    for key in ("goal_settled_rate", "goal_reached_rate", "success_rate"):
        val = summary.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict) and "mean" in val:
            return float(val["mean"])
    return None


def run_queue(
    config_path: Path,
    ckpts: List[Path],
    output_root: Path,
    *,
    rollouts_from_dir: Optional[Path] = None,
    rollouts_limit: Optional[int] = None,
    sub_idx: int = 0,
    scene_name: Optional[str] = None,
    continue_on_error: bool = True,
) -> List[Dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    queue_log_path = output_root / "queue_log.json"
    table_path = output_root / "queue_table.csv"

    results: List[Dict[str, Any]] = []
    queue_start = time.time()

    for idx, ckpt in enumerate(ckpts):
        ckpt = ckpt.resolve()
        run_name = _derive_run_name(ckpt)
        out_dir = output_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n{'=' * 64}\n"
            f"[queue] ({idx + 1}/{len(ckpts)}) {ckpt.name}\n"
            f"[queue]   ckpt:    {ckpt}\n"
            f"[queue]   output:  {out_dir}\n"
            f"{'=' * 64}",
            flush=True,
        )

        if not ckpt.is_file():
            print(f"[queue] MISSING ckpt {ckpt}; skipping", flush=True)
            results.append({
                "ckpt": str(ckpt),
                "run_name": run_name,
                "output_dir": str(out_dir),
                "status": "missing",
                "elapsed_s": 0.0,
            })
            continue

        started = time.time()
        try:
            summary = evaluate(
                config_path,
                checkpoint_override=ckpt,
                output_dir_override=out_dir,
                run_tag_override=run_name,
                rollouts_from_dir=rollouts_from_dir,
                rollouts_limit=rollouts_limit,
                sub_idx=sub_idx,
                scene_name=scene_name,
            )
        except KeyboardInterrupt:
            print("[queue] KeyboardInterrupt; aborting queue.", flush=True)
            raise
        except Exception as exc:
            print(f"[queue] FAILED on {ckpt.name}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            results.append({
                "ckpt": str(ckpt),
                "run_name": run_name,
                "output_dir": str(out_dir),
                "status": "error",
                "error": repr(exc),
                "elapsed_s": time.time() - started,
            })
            if continue_on_error:
                continue
            break
        else:
            elapsed = time.time() - started
            results.append({
                "ckpt": str(ckpt),
                "run_name": run_name,
                "output_dir": str(out_dir),
                "status": "ok",
                "elapsed_s": elapsed,
                "success_rate": _success_rate_from_summary(summary),
                "n_rollouts": int(summary.get("n_rollouts", 0))
                              if isinstance(summary.get("n_rollouts"), (int, float))
                              else None,
            })

    queue_elapsed = time.time() - queue_start

    # Persist a roll-up JSON + CSV alongside per-ckpt summaries.
    payload = {
        "config": str(config_path.resolve()),
        "output_root": str(output_root.resolve()),
        "rollouts_from_dir": str(rollouts_from_dir) if rollouts_from_dir else None,
        "rollouts_limit": rollouts_limit,
        "started_iso": _dt.datetime.fromtimestamp(queue_start).isoformat(),
        "finished_iso": _dt.datetime.now().isoformat(),
        "queue_elapsed_s": queue_elapsed,
        "n_ckpts": len(ckpts),
        "results": results,
    }
    with open(queue_log_path, "w") as f:
        json.dump(payload, f, indent=2)

    table_fields = ["run_name", "ckpt", "status", "success_rate", "n_rollouts", "elapsed_s"]
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table_fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in table_fields})

    print(
        f"\n{'=' * 64}\n"
        f"[queue] DONE  n={len(results)}  elapsed={queue_elapsed/60:.1f} min\n"
        f"[queue] roll-up JSON -> {queue_log_path}\n"
        f"[queue] roll-up CSV  -> {table_path}\n"
        f"{'=' * 64}",
        flush=True,
    )

    # Pretty stdout table.
    print(f"{'run_name':<60} {'status':<8} {'rate':>8} {'rolls':>6} {'min':>6}")
    for r in results:
        rate = r.get("success_rate")
        rate_str = f"{rate * 100:.1f}%" if isinstance(rate, (int, float)) else "?"
        nr = r.get("n_rollouts")
        nr_str = str(nr) if isinstance(nr, int) else "?"
        elapsed_min = (r.get("elapsed_s") or 0.0) / 60.0
        print(
            f"{r['run_name'][:60]:<60} {r['status']:<8} {rate_str:>8} "
            f"{nr_str:>6} {elapsed_min:>6.1f}"
        )

    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Queue multiple closed-loop evals serially.",
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Eval YAML to use for every ckpt (rollouts list can be "
                        "overridden by --rollouts-from-dir).")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Parent directory; each ckpt gets a subfolder named "
                        "after its run_tag/file stem.")
    p.add_argument("--ckpt", type=Path, action="append", default=[],
                   help="Checkpoint to evaluate. Repeatable.")
    p.add_argument("--ckpt-list", type=Path, default=None,
                   help="Optional text file with one ckpt path per line "
                        "(blanks and #-comments are ignored).")
    p.add_argument("--rollouts-from-dir", type=Path, default=None,
                   help="Override the YAML rollouts list with all "
                        "trajectories_val*.pt under this directory (relative "
                        "paths resolve against nav_policy/).")
    p.add_argument("--rollouts-limit", type=int, default=None,
                   help="Cap the number of trajectories loaded by "
                        "--rollouts-from-dir.")
    p.add_argument("--sub-idx", type=int, default=0,
                   help="sub_idx applied to every auto-generated rollout.")
    p.add_argument("--scene-name", type=str, default=None,
                   help="scene name applied to every auto-generated rollout.")
    p.add_argument("--no-continue-on-error", action="store_true",
                   help="Stop the queue at the first failing ckpt instead of "
                        "skipping and continuing.")

    args = p.parse_args()

    ckpts: List[Path] = list(args.ckpt)
    if args.ckpt_list is not None:
        ckpts.extend(_read_ckpt_list_file(args.ckpt_list))

    if not ckpts:
        sys.exit("No checkpoints given. Pass --ckpt one or more times, or --ckpt-list FILE.")

    run_queue(
        config_path=args.config,
        ckpts=ckpts,
        output_root=args.output_root,
        rollouts_from_dir=args.rollouts_from_dir,
        rollouts_limit=args.rollouts_limit,
        sub_idx=args.sub_idx,
        scene_name=args.scene_name,
        continue_on_error=not args.no_continue_on_error,
    )


if __name__ == "__main__":
    main()
