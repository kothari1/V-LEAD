# V-LEAD — Snapshot of all data needed to reproduce poster figures

Self-contained snapshot copied from `/project/kothari1/vlead_data/rl_runs/`
on **2026-06-03**. Anyone with this folder can regenerate every plot in
`results/*.{pdf,png}` without further access to the project NFS.

See `../../TEST_RESULTS.md` for the source-of-truth eval index +
checkpoint paths.

---

## Quick start (regenerate every figure from scratch)

After cloning the V-LEAD repo on any machine:

```bash
cd nav_policy/scripts/plots
pip install -r requirements.txt          # one-time: matplotlib, numpy, tensorboard
bash regenerate.sh                        # runs all 7 plot scripts
```

Every figure drops into `V-LEAD/results/{*.pdf, *.png}`. No project NFS
access required — all scripts read from `V-LEAD/results/logs/` (this
directory).

To regenerate one figure:

```bash
cd nav_policy/scripts/plots
python3 headline_bar.py           # or algo_comparison / failure_modes / long8h_snapshots /
                                  #    phase_a_pie / query_heatmap / training_stability
```

Plot scripts use **only** the snapshot files documented below. If you
want to add a new figure that reads a new tag, drop a new
`<tag>.per_rollout.csv` + `<tag>.summary.json` into the appropriate
`eval/<suite>/` directory and either add it to `_eval_data.py:SOURCES`
or write a new plot script.

To view TensorBoard for any training run:

```bash
tensorboard --logdir V-LEAD/results/logs/training/
```

---

## Layout

```
logs/
├── eval/                                    closed-loop eval outputs
│   ├── full-110/                            n=109, TEST pool 071733 (the ladder)
│   ├── holdout-14/                          n=14, TEST subset
│   └── phase-A-175/                         n=171, TRAIN pool 071353 (BC failure mining)
└── training/                                per-run training artifacts
    ├── rl_sac_dagger_r12_v6/
    ├── rl_sac_dagger_r12_v7/
    ├── rl_sac_dagger_r12_v8/
    ├── rl_sac_dagger_r12_long_8h_v1/        the regression run
    ├── rl_ppo_dagger_r12_v6/
    ├── rl_ppo_dagger_r12_v7/
    ├── rl_td3bc_dagger_r12_80ep/            validation run
    ├── rl_td3bc_dagger_r12_max/             running (snapshot at copy time)
    └── rl_td3bc_failure_focus_v1/           running (snapshot at copy time)
```

Each `training/<run>/` directory contains:

| file | what |
|---|---|
| `episodes.csv` | one row per training episode (return, success, steps, term) |
| `log.csv` | one row per training iteration (losses, lr, alpha, etc.) |
| `perf.json` | wall-clock + iteration-rate metadata |
| `summary.json` | top-level run summary (init_checkpoint, total_episodes, ...) — present for completed runs only |
| `tb/events.out.tfevents.*` | TensorBoard scalar tags (see below) |

---

## File naming convention (eval/)

Each evaluated checkpoint produced **two** files in its suite folder:

- `<tag>.per_rollout.csv` — one row per trajectory, every metric (final_pos_err, yaw_err, termination, n_steps, etc.)
- `<tag>.summary.json` — aggregate over all rollouts (goal_success_rate, CI95, timeout_rate, collision_rate, metrics_config)

`<tag>` is the short label used in TEST_RESULTS.md.

### full-110/ (canonical held-out test)

| tag | n | success | source ckpt | local path |
|---|---|---|---|---|
| `bc_seed` | 109 | 62.4% | `bc_best_balanced_dagger_r12_new.pt` | `eval/full-110/bc_seed.per_rollout.csv` |
| `bc_legacy` | 109 | 2.8% | `bc_best.pt` (older, cross-env) | `eval/full-110/bc_legacy.per_rollout.csv` |
| `sac_v7_best` | 109 | 63.3% | `rl_sac_dagger_r12_v7_best.pt` | `eval/full-110/sac_v7_best.per_rollout.csv` |
| `sac_v8_best` | 109 | 63.3% | `rl_sac_dagger_r12_v8_best.pt` | `eval/full-110/sac_v8_best.per_rollout.csv` |
| `ppo_v6_best` | 109 | 40.4% | `rl_ppo_dagger_r12_v6_best.pt` | `eval/full-110/ppo_v6_best.per_rollout.csv` |
| `sac_long8h_seed_baseline` | 109 | 62.4% | bc_seed, re-run in long_8h batch (sanity) | `eval/full-110/sac_long8h_seed_baseline.per_rollout.csv` |
| `sac_long8h_iter0100_ep00400` | 109 | 16.5% | long_8h snapshot @ ep 400 | `eval/full-110/sac_long8h_iter0100_ep00400.per_rollout.csv` |
| `sac_long8h_iter0200_ep00800` | 109 | 45.9% | long_8h snapshot @ ep 800 | `eval/full-110/sac_long8h_iter0200_ep00800.per_rollout.csv` |
| `sac_long8h_iter0300_ep01200` | 109 | 4.6% | long_8h snapshot @ ep 1200 | `eval/full-110/sac_long8h_iter0300_ep01200.per_rollout.csv` |
| `sac_long8h_iter0400_ep01600` | 109 | 32.1% | long_8h snapshot @ ep 1600 | `eval/full-110/sac_long8h_iter0400_ep01600.per_rollout.csv` |
| `sac_long8h_best` | 109 | 55.0% | `rl_sac_dagger_r12_long_8h_v1_best.pt` | `eval/full-110/sac_long8h_best.per_rollout.csv` |
| `sac_long8h_latest` | 109 | 32.1% | `rl_sac_dagger_r12_long_8h_v1_latest.pt` | `eval/full-110/sac_long8h_latest.per_rollout.csv` |

Plus orchestration roll-ups (same dir):
- `_queue_log.batch1.json` + `_queue_table.batch1.csv` — covers the first 5 ckpts (bc_seed, bc_legacy, sac_v7, sac_v8, ppo_v6)
- `_queue_log.long8h.json` — covers the long_8h batch (7 ckpts)

### holdout-14/

| tag | n | success | source ckpt | local path |
|---|---|---|---|---|
| `bc_seed` | 14 | 50.0% | bc_best_balanced_dagger_r12_new.pt | `eval/holdout-14/bc_seed.per_rollout.csv` |
| `sac_v7_best` | 14 | 57.1% | rl_sac_dagger_r12_v7_best.pt | `eval/holdout-14/sac_v7_best.per_rollout.csv` |

### phase-A-175/ (BC failure mining on the TRAIN pool)

| tag | n | success | notes | local path |
|---|---|---|---|---|
| `bc_seed` | 171 | 73.1% | `--shuffle --stop-at-failures 50` triggered at row 175; 4 errored | `eval/phase-A-175/bc_seed.per_rollout.csv` |

Plus `_queue_log.json` + `_queue_table.csv`.

---

## Training artifacts — local paths

| Run | Algorithm | episodes.csv | log.csv | TB events | summary.json |
|---|---|---|---|---|---|
| rl_sac_dagger_r12_v6 | SAC | `training/rl_sac_dagger_r12_v6/episodes.csv` | `.../log.csv` | `.../tb/events.out.tfevents.*` (2 files, second from resume) | `.../summary.json` |
| rl_sac_dagger_r12_v7 | SAC | `training/rl_sac_dagger_r12_v7/episodes.csv` | `.../log.csv` | `.../tb/events.out.tfevents.*` | `.../summary.json` |
| rl_sac_dagger_r12_v8 | SAC | `training/rl_sac_dagger_r12_v8/episodes.csv` | `.../log.csv` | `.../tb/events.out.tfevents.*` | `.../summary.json` |
| rl_sac_dagger_r12_long_8h_v1 | SAC | `training/rl_sac_dagger_r12_long_8h_v1/episodes.csv` | `.../log.csv` | `.../tb/events.out.tfevents.*` | `.../summary.json` |
| rl_ppo_dagger_r12_v6 | PPO | `training/rl_ppo_dagger_r12_v6/episodes.csv` | `.../log.csv` | `.../tb/events.out.tfevents.*` | `.../summary.json` |
| rl_ppo_dagger_r12_v7 | PPO | `training/rl_ppo_dagger_r12_v7/episodes.csv` | `.../log.csv` | `.../tb/events.out.tfevents.*` | — (no summary) |
| rl_td3bc_dagger_r12_80ep | TD3+BC | `training/rl_td3bc_dagger_r12_80ep/episodes.csv` | `.../log.csv` | `.../tb/events.out.tfevents.*` (2 files) | `.../summary.json` |
| rl_td3bc_dagger_r12_max | TD3+BC | `training/rl_td3bc_dagger_r12_max/episodes.csv` | `.../log.csv` | `.../tb/events.out.tfevents.*` (2 files) | — (running) |
| rl_td3bc_failure_focus_v1 | TD3+BC | `training/rl_td3bc_failure_focus_v1/episodes.csv` | `.../log.csv` | `.../tb/events.out.tfevents.*` | — (running) |

---

## TensorBoard event files

Each `training/<run>/tb/events.out.tfevents.*` carries the scalar tags
emitted by `nav_policy.rl.train_rl`.  Tags present across runs:

- `train/q1_loss`, `train/q2_loss`, `train/policy_loss`, `train/alpha`, `train/policy_log_std_mean`, `train/ref_kl`
- TD3+BC adds: `train/bc_mse`, `eval/seed_floor`
- `rollout/mean_return`, `rollout/success_rate`, `rollout/mean_steps`
- `episode/return`, `episode/success`, `episode/collision`, `episode/final_pos_err_m`, `episode/goal_settled`, `episode/steps`
- `eval/goal_success_rate`, `eval/n_rollouts`, `eval/n_success`
- `eval_per_query/flightroom_holdout_qNNN` (one tag per eval query)
- `reward/*_mean` for each reward component

Quick view (point TB at the snapshot, NOT the project NFS):

```bash
tensorboard --logdir results/logs/training/
```

---

## Plot-script ↔ data-file map

| Script | Reads from |
|---|---|
| `headline_bar.py` | `eval/full-110/{bc_seed, sac_v7_best, sac_v8_best, ppo_v6_best, sac_long8h_best, td3bc_*}.summary.json` (the `td3bc_*` rows appear automatically when their summary.json files are added) |
| `algo_comparison.py` | same as `headline_bar.py` |
| `long8h_snapshots.py` | `eval/full-110/sac_long8h_{seed_baseline, iter0100_ep00400, iter0200_ep00800, iter0300_ep01200, iter0400_ep01600, best, latest}.summary.json` |
| `failure_modes.py` | `eval/full-110/bc_seed.per_rollout.csv` |
| `phase_a_pie.py` | `eval/phase-A-175/bc_seed.per_rollout.csv` |
| `query_heatmap.py` | `eval/full-110/{bc_seed, sac_v7_best, sac_v8_best, ppo_v6_best}.per_rollout.csv` |
| `training_stability.py` | `training/rl_sac_dagger_r12_long_8h_v1/tb/events.out.tfevents.*` + `training/rl_td3bc_dagger_r12_80ep/tb/events.out.tfevents.*` |

Shared helpers (no plot output of their own):
- `_paths.py` resolves all snapshot paths from this README's directory.
- `_eval_data.py` aggregates summary.json files for bar/curve plots.
- `_style.py` shared matplotlib rcParams + algorithm color map.
- `requirements.txt` minimum pip deps.
- `regenerate.sh` runs all 7 plot scripts in sequence.

---

## File count + size

80 files, 3.3 MB total.  TB events dominate: `rl_sac_dagger_r12_long_8h_v1/tb`
alone is 940 KB (400 iterations × ~40 scalar tags).

---

## Appendix — original NFS source paths

Mapping for anyone who needs to trace a snapshot file back to its
authoritative copy on `/project/kothari1/vlead_data/rl_runs/`.

### Eval source paths

| local path | NFS source |
|---|---|
| `eval/full-110/bc_seed.per_rollout.csv` | `dagger_r12/_eval_batch_20260602/checkpoints__bc_best_balanced_dagger_r12_new/per_rollout.csv` |
| `eval/full-110/bc_legacy.per_rollout.csv` | `dagger_r12/_eval_batch_20260602/checkpoints__bc_best/per_rollout.csv` |
| `eval/full-110/sac_v7_best.per_rollout.csv` | `dagger_r12/_eval_batch_20260602/rl_sac_dagger_r12_v7__rl_sac_dagger_r12_v7_best/per_rollout.csv` |
| `eval/full-110/sac_v8_best.per_rollout.csv` | `dagger_r12/_eval_batch_20260602/rl_sac_dagger_r12_v8__rl_sac_dagger_r12_v8_best/per_rollout.csv` |
| `eval/full-110/ppo_v6_best.per_rollout.csv` | `dagger_r12/_eval_batch_20260602/rl_ppo_dagger_r12_v6__rl_ppo_dagger_r12_v6_best/per_rollout.csv` |
| `eval/full-110/_queue_log.batch1.json` | `dagger_r12/_eval_batch_20260602/queue_log.json` |
| `eval/full-110/_queue_table.batch1.csv` | `dagger_r12/_eval_batch_20260602/queue_table.csv` |
| `eval/full-110/sac_long8h_seed_baseline.per_rollout.csv` | `dagger_r12/rl_sac_dagger_r12_long_8h_v1/eval_110_20260602/checkpoints__bc_best_balanced_dagger_r12_new/per_rollout.csv` |
| `eval/full-110/sac_long8h_best.per_rollout.csv` | `dagger_r12/rl_sac_dagger_r12_long_8h_v1/eval_110_20260602/rl_sac_dagger_r12_long_8h_v1__rl_sac_dagger_r12_long_8h_v1_best/per_rollout.csv` |
| `eval/full-110/sac_long8h_latest.per_rollout.csv` | `.../rl_sac_dagger_r12_long_8h_v1__rl_sac_dagger_r12_long_8h_v1_latest/per_rollout.csv` |
| `eval/full-110/sac_long8h_iter0100_ep00400.per_rollout.csv` | `.../snapshots__rl_sac_dagger_r12_long_8h_v1_iter0100_ep00400/per_rollout.csv` |
| `eval/full-110/sac_long8h_iter0200_ep00800.per_rollout.csv` | `.../snapshots__rl_sac_dagger_r12_long_8h_v1_iter0200_ep00800/per_rollout.csv` |
| `eval/full-110/sac_long8h_iter0300_ep01200.per_rollout.csv` | `.../snapshots__rl_sac_dagger_r12_long_8h_v1_iter0300_ep01200/per_rollout.csv` |
| `eval/full-110/sac_long8h_iter0400_ep01600.per_rollout.csv` | `.../snapshots__rl_sac_dagger_r12_long_8h_v1_iter0400_ep01600/per_rollout.csv` |
| `eval/full-110/_queue_log.long8h.json` | `dagger_r12/rl_sac_dagger_r12_long_8h_v1/eval_110_20260602/queue_log.json` |
| `eval/holdout-14/sac_v7_best.per_rollout.csv` | `dagger_r12/rl_sac_dagger_r12_v7_holdout14_eval/per_rollout.csv` |
| `eval/holdout-14/bc_seed.per_rollout.csv` | `rahul_best/bc_best_rahul_holdout14_eval/per_rollout.csv` |
| `eval/phase-A-175/bc_seed.per_rollout.csv` | `bc_train_eval/071353/checkpoints__bc_best_balanced_dagger_r12_new/per_rollout.csv` |
| `eval/phase-A-175/_queue_log.json` | `bc_train_eval/071353/queue_log.json` |
| `eval/phase-A-175/_queue_table.csv` | `bc_train_eval/071353/queue_table.csv` |

(`.summary.json` companions live next to each `.per_rollout.csv` in the
NFS source dirs, copied identically.)

### Training source paths

NFS root: `/project/kothari1/vlead_data/rl_runs/dagger_r12/<run>/`

For run `<run>`:
- `training/<run>/episodes.csv` ← `<run>_episodes.csv`
- `training/<run>/log.csv` ← `<run>_log.csv`
- `training/<run>/perf.json` ← `<run>_perf.json`
- `training/<run>/summary.json` ← `<run>_summary.json` (if present)
- `training/<run>/tb/` ← `tb/` directory (all events files preserved)

### Source checkpoints (not copied; large)

| ckpt | NFS path |
|---|---|
| BC seed `bc_best_balanced_dagger_r12_new.pt` (51 MB) | `/project/kothari1/vlead_data/checkpoints/bc_best_balanced_dagger_r12_new.pt` |
| BC legacy `bc_best.pt` (46 MB) | `/project/kothari1/vlead_data/checkpoints/bc_best.pt` |
| Each RL `<run>_best.pt` / `_latest.pt` | `/project/kothari1/vlead_data/rl_runs/dagger_r12/<run>/<run>_{best,latest}.pt` |
| Snapshots | `/project/kothari1/vlead_data/rl_runs/dagger_r12/<run>/snapshots/<run>_iter*_ep*.pt` |

Checkpoint binaries intentionally excluded from snapshot (size). Re-run
evals via `nav_policy/scripts/eval_queue.py` if a ckpt's per-rollout
file is needed but missing.
