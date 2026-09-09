# Ablations

The ablation table in the report is populated by running each variant below
and then aggregating the per-run summaries with
`scripts/collect_ablations.py`, which writes `data/eval/ablation_summary.csv`
and `.md`.  All commands assume the docker container CWD
`/workspace/nav_policy/`.

## Naming convention

| Concept            | Source                                | Where it shows up               |
| ------------------ | ------------------------------------- | ------------------------------- |
| Ablation name      | `--run-tag` CLI or YAML               | All summary.json files          |
| Where checkpoints go | `--checkpoint-dir` CLI or YAML      | `data/checkpoints_<tag>/`       |
| Where eval lands   | `--output-dir` CLI or YAML            | `data/eval/<tag>_offline/` etc. |

`train_bc.py` and `eval_in_figs.py` both accept `--checkpoint-dir` /
`--checkpoint` / `--output-dir` / `--run-tag` overrides so a single base config
can drive every ablation by changing flags rather than copying YAMLs.  The
overrides are stamped into the saved checkpoint and into the summary.json so
the collector picks them up automatically.

## Variants and how to run them

### Full pipeline (BC + 2 DAgger rounds) -- each round in its own folder

```bash
# 1. BC baseline
python scripts/train_bc.py --config configs/default.yaml \
  --checkpoint-dir data/checkpoints_bc --run-tag bc_round0
python scripts/eval_offline.py --config configs/default.yaml \
  --checkpoint data/checkpoints_bc/bc_best.pt \
  --output-dir data/eval/bc_round0_offline
python scripts/eval_in_figs.py --config configs/eval_closed_loop.yaml \
  --checkpoint data/checkpoints_bc/bc_best.pt \
  --output-dir data/eval/bc_round0_closed_loop \
  --run-tag bc_round0_closed_loop

# 2. DAgger round 1 (reference oracle, rollouts the BC checkpoint above)
python scripts/run_dagger.py --config configs/dagger_round1.yaml
python scripts/train_bc.py --config configs/default.yaml \
  --checkpoint-dir data/checkpoints_dagger_r1 --run-tag dagger_r1
python scripts/eval_offline.py --config configs/default.yaml \
  --checkpoint data/checkpoints_dagger_r1/bc_best.pt \
  --output-dir data/eval/dagger_r1_offline
python scripts/eval_in_figs.py --config configs/eval_closed_loop.yaml \
  --checkpoint data/checkpoints_dagger_r1/bc_best.pt \
  --output-dir data/eval/dagger_r1_closed_loop \
  --run-tag dagger_r1_closed_loop

# 3. DAgger round 2 (rollouts the dagger_r1 checkpoint -- already wired in
#    configs/dagger_round2.yaml)
python scripts/run_dagger.py --config configs/dagger_round2.yaml
python scripts/train_bc.py --config configs/default.yaml \
  --checkpoint-dir data/checkpoints_dagger_r2 --run-tag dagger_r2
python scripts/eval_offline.py --config configs/default.yaml \
  --checkpoint data/checkpoints_dagger_r2/bc_best.pt \
  --output-dir data/eval/dagger_r2_offline
python scripts/eval_in_figs.py --config configs/eval_closed_loop.yaml \
  --checkpoint data/checkpoints_dagger_r2/bc_best.pt \
  --output-dir data/eval/dagger_r2_closed_loop \
  --run-tag dagger_r2_closed_loop
```

After this completes, every round's checkpoint, training log, offline summary,
and closed-loop summary live in its own folder.  No file ever gets overwritten
by a later round.

### DAgger: MPC oracle (proper DAgger, re-solves OCP at each policy state)

Alternative round-1 oracle.  Run after step 1 (BC) of the full pipeline:

```bash
python scripts/run_dagger.py --config configs/dagger_round1_mpc.yaml
python scripts/train_bc.py --config configs/default.yaml \
  --checkpoint-dir data/checkpoints_dagger_r1_mpc --run-tag dagger_r1_mpc
python scripts/eval_offline.py --config configs/default.yaml \
  --checkpoint data/checkpoints_dagger_r1_mpc/bc_best.pt \
  --output-dir data/eval/dagger_r1_mpc_offline
python scripts/eval_in_figs.py --config configs/eval_closed_loop.yaml \
  --checkpoint data/checkpoints_dagger_r1_mpc/bc_best.pt \
  --output-dir data/eval/dagger_r1_mpc_closed_loop \
  --run-tag dagger_r1_mpc_closed_loop
```

The cache meta records `oracle="mpc"` and `mpc_policy="vrmpc_rrt"`, so the
collector and inspector can distinguish proper-DAgger entries from
reference-DAgger entries.  Each label costs one ACADOS SQP solve (~30-50 ms
on the laptop), so a 100-step rollout adds ~3-5 s of wall-time vs. the
reference oracle.

### No goal vector (heading + distance both zeroed)

```bash
python scripts/train_bc.py --config configs/ablation_no_goal.yaml
python scripts/eval_offline.py --config configs/ablation_no_goal.yaml \
  --checkpoint data/checkpoints_no_goal/bc_best.pt \
  --output-dir data/eval/no_goal_offline
python scripts/eval_in_figs.py --config configs/eval_closed_loop_no_goal.yaml
```

The training-time flag `train.zero_goal_heading: true` zeros the entire goal
input vector (heading **and** normalized distance), is stamped into the
checkpoint config, and is read by the policy controller at deployment so the
closed-loop eval matches training without an extra flag.

### History length T and horizon H sweeps

Copy `configs/default.yaml` to `configs/ablation_T<k>.yaml` (resp.
`ablation_H<k>.yaml`), change `window.T` (resp. `window.H`), set
`train.checkpoint_dir: data/checkpoints_T<k>/`, set
`train.run_tag: ablation_T<k>`, then run training and offline eval as above.
Closed-loop eval also requires a matching `eval_closed_loop_T<k>.yaml`.

## Aggregating results

```bash
python scripts/collect_ablations.py
```

Outputs:
* `data/eval/ablation_summary.csv` -- one row per run, all fields.
* `data/eval/ablation_summary.md`  -- human-readable markdown tables.

Drop the markdown tables into the report or paste the relevant rows into
the project write-up (ablation table).
