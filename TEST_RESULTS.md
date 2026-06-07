# V-LEAD Pilot — Eval Results Index

One row per (checkpoint, eval suite) pair. Sources: `summary.json` and
`queue_log.json` in each eval directory. Update with a new row whenever
a new eval completes.

---

## Conventions

- **Eval tolerances (uniform across every run below).** `success_yaw_tol_rad: 0.5`
  (~28.6°), `goal_position_tol_m: 0.5`, `success_tracking_tol_m: 1.0`. No
  loose-tolerance variants kept.
- **Pools.** Train = `flightroom_ssv_exp_2026-05-22_071353` (220 trajs) +
  `..._071718` (220 trajs). Test = `..._071733_trajs-110` (110 trajs, ladder).
  No leakage: 071733 is **never** seen during training.
- **⚠ Train goal ≠ test goal (cross-object).** Goals differ by pool:
  071353 = **green clock**, 071718 = **yellow drill**, 071733 = **ladder**.
  So **full-110 measures cross-object generalization**, not same-goal
  improvement. RL fine-tuned on clock/drill cannot be expected to raise the
  *ladder* score — which is exactly why every RL run sits ≤ seed on full-110.
  To measure same-goal improvement, select/score on **clockdrill-40** (held-out
  spawns of the trained goals); treat full-110 as a generalization / non-
  regression check only.
- **Success rate format.** `XX.X%` from `goal_success_rate` field;
  `n_rollouts` reported separately. Full-110 evals report 109 because one
  trajectory is dropped due to a known warmup error.
- **Model architecture** is `rgb_da2_crossattn_v1` everywhere except `bc_best.pt`
  which is the older `rgb_resnet18`.
- **All eval configs use `--rollouts-from-dir`** to override the rollouts list
  in the yaml. The yaml supplies only the metrics_config block. So the
  "full-110" runs use `eval_closed_loop_flightroom_holdout_14.yaml` as the
  config file but evaluate the 110-traj 071733 dir.

---

## Eval suites (legend)

| Suite | Config yaml | N trajs | Source pool | Purpose |
|---|---|---|---|---|
| **full-110** | `eval_closed_loop_flightroom_holdout_14.yaml` + `--rollouts-from-dir ...071733_trajs-110` | 110 (109 ok) | TEST | canonical hold-out, scored ladder |
| **holdout-30** | `eval_closed_loop_flightroom_holdout_30.yaml` | 30 | TEST subset | fast hold-out gate for `_80ep`. ⚠ **Weak gate: the BC seed already scores 27/30 = 90% here** (near-ceiling, ladder subset), so it can't show improvement and is dominated by regression-risk queries. Superseded by clockdrill-40 for selection. |
| **holdout-14** | `eval_closed_loop_flightroom_holdout_14.yaml` | 14 | TEST subset | early-iter screen |
| **clockdrill-40** | `eval_closed_loop_clockdrill_heldout_40.yaml` | 40 | held-out spawns of trained goals | in-loop eval gate for `rl_td3bc_dagger_r12_max` |
| **phase-A-175** | shuffle of 071353 + `--stop-at-failures 50` | 175 | TRAIN subset (071353) | BC failure mining for `failure_focus_v1` |

---

## BC seeds

### `bc_best_balanced_dagger_r12_new.pt` (51 MB, mtime 2026-06-01 13:51)

Path: `/project/kothari1/vlead_data/checkpoints/bc_best_balanced_dagger_r12_new.pt`
Arch: `rgb_da2_crossattn_v1`. Training: flightroom-only BC + DAgger r12.
**This is the seed for every recent RL run.**

| Eval suite | n | success | CI95 | timeout | collision | eval dir | mtime |
|---|---|---|---|---|---|---|---|
| full-110 | 109 | **62.4%** | 53.2–70.7% | 31.2% | 6.4% | `rl_runs/dagger_r12/_eval_batch_20260602/checkpoints__bc_best_balanced_dagger_r12_new/` | Jun 2 02:12 |
| full-110 | 109 | 62.4% | 53.2–70.7% | 31.2% | 6.4% | `rl_runs/dagger_r12/rl_sac_dagger_r12_long_8h_v1/eval_110_20260602/checkpoints__bc_best_balanced_dagger_r12_new/` | Jun 2 11:13 (duplicate run, same ckpt; sanity for long_8h regression analysis) |
| holdout-14 | 14 | 50.0% | — | 35.7% | 14.3% | `rl_runs/rahul_best/bc_best_rahul_holdout14_eval/` | Jun 1 23:48 |
| phase-A-175 | 171 | **73.1%** | 66.7–78.9% | 31.6% | 1.2% | `rl_runs/bc_train_eval/071353/checkpoints__bc_best_balanced_dagger_r12_new/` | Jun 2 20:47 (175 rollouts attempted; `--stop-at-failures 50` triggered at row 175; 4 errored) |

**Note.** Higher rate on phase-A-175 (73.1%) than full-110 (62.4%) is expected:
phase-A samples from the train pool 071353, which the BC seed was trained on;
full-110 is the held-out test pool 071733.

### `bc_best.pt` (46 MB, mtime 2026-05-25 16:26)

Path: `/project/kothari1/vlead_data/checkpoints/bc_best.pt`. Arch:
`rgb_resnet18` (older). Trained on backroom + packardpark (cross-env).

| Eval suite | n | success | timeout | collision | eval dir | mtime |
|---|---|---|---|---|---|---|
| full-110 | 109 | **2.8%** | 72.5% | 27.5% | `rl_runs/dagger_r12/_eval_batch_20260602/checkpoints__bc_best/` | Jun 2 01:23 |

Cross-env transfer: expected to be bad. Listed only for context.

---

## Seed failure-mode analysis (full-110, ladder)

Source: `_eval_batch_20260602/` per_rollout.csv + query_matrix.csv across 5
checkpoints (bc_best, dagger_r12 seed, ppo_v6, sac_v7, sac_v8). Computed via
`analyze_eval_matrix.py` logic.

Cross-checkpoint query matrix (110 ladder queries):

| Category | Count |
|---|---|
| Seed passes | 68 |
| Seed fails, cracked by ≥1 other ckpt (**winnable**) | **1** |
| Seed fails, never solved by ANY of the 5 ckpts | **41** |
| Seed passes that ≥1 RL ckpt **broke** (regression-risk) | **25** |

**Read:** RL's net effect on full-110 so far = break ~25 seed passes, crack ~1
new fail → this *is* why every RL run lands ≤ seed. The 41 never-solved fails
are the apparent ceiling for the current approach.

Seed's 42 failures by mode (seed per_rollout):
- **34 timeout, 7 collision, 1 exception.**
- The timeouts are **near-misses**: median final pos err **0.94 m**, all
  < 1.45 m, 15 within 0.75 m, **0 reached the goal**, progress ratio 0.81. The
  drone gets ~90% of the way, then can't settle inside the 0.5 m radius before
  the clock runs out.

**Implication:** real headroom = terminal convergence (decelerate/settle the
last metre), which is learnable — **but only if the eval/train goal matches**
(see cross-object caveat). This is why `rl_td3bc_dagger_r12_max` selects on
clockdrill-40 rather than the ladder suite.

---

## RL runs

Ordered by recency (newest first). For each run: config, training
metadata, every checkpoint emitted, and every eval row associated with
those checkpoints.

---

### 🏆 `rl_td3bc_dagger_r12_max` (completed Jun 3 01:26 — **NEW HEADLINE**)

- **Config:** `nav_policy/configs/train_rl_flightroom_td3bc_dagger_r12_max.yaml`
- **Run dir:** `/project/kothari1/vlead_data/rl_runs/dagger_r12/rl_td3bc_dagger_r12_max/`
- **TB:** `.../rl_td3bc_dagger_r12_max/tb/events.out.tfevents.{1780456335,1780461176}.*` (2 events files; second from resume)
- **Algorithm:** TD3+BC (residual_scale=1.5, bc_weight=0.5, critic_warmup=200)
- **Pool:** 50 trajs (25 spawns × 2 dirs spread across 0..216)
- **Episodes:** 586 (target 600; stopped slightly early)
- **In-loop eval gate:** clockdrill-40 (`eval_closed_loop_clockdrill_heldout_40.yaml`); `best_eval_goal_success_rate=0.5`
- Checkpoints: `_best.pt`, `_latest.pt` + 7 snapshots (iter0010..iter0070).

Held-out full-110 evals (`eval_110_20260603/`):

| Ckpt | Suite | n | success | CI95 | timeout | collision | eval dir | mtime |
|---|---|---|---|---|---|---|---|---|
| `rl_td3bc_dagger_r12_max_best.pt` | full-110 | 109 | **65.1%** | 56.0–74.3% | 28.4% | 7.3% | `eval_110_20260603/rl_td3bc_dagger_r12_max__rl_td3bc_dagger_r12_max_best/` | Jun 3 02:54 |
| `rl_td3bc_dagger_r12_max_latest.pt` | full-110 | 109 | 56.9% | 47.7–66.1% | 37.6% | 6.4% | `eval_110_20260603/rl_td3bc_dagger_r12_max__rl_td3bc_dagger_r12_max_latest/` | Jun 3 03:43 |

**The result we wanted.** `_best.pt` clears BC seed (62.4%) by +2.7 pts and ties/edges SAC v7/v8 (63.3%) on the cross-object ladder. Timeout rate drops from 31.2% (seed) → 28.4%, i.e. the residual head learned terminal convergence on a few previously-timing-out queries. `_latest.pt` (586 ep) drifted ~6 pts below `_best`, validating the `eval_floor_from_seed` + in-loop selection: the gate caught the better mid-training ckpt before late-iter drift.

---

### `rl_td3bc_failure_focus_v1` (completed Jun 3 04:26)

- **Config:** `nav_policy/configs/train_rl_flightroom_td3bc_failure_focus_v1.yaml`
- **Run dir:** `/project/kothari1/vlead_data/rl_runs/dagger_r12/rl_td3bc_failure_focus_v1/`
- **TB:** `.../rl_td3bc_failure_focus_v1/tb/events.out.tfevents.1780465425.*`
- **Algorithm:** TD3+BC (residual_actor, freeze_base_full, critic_warmup=500, residual_scale=1.0, bc_weight=1.0)
- **Pool:** 40 trajs (28 yaw_only + 2 both + 10 success), mined from phase-A-175 (071353 clock only)
- **Episodes:** 600 (150 iters × 4)
- **In-loop eval gate:** holdout-30; `best_eval_goal_success_rate=0.9` = seed floor itself (nothing beat the seed on cross-object hold-out — expected; train pool is clock-only, hold-out is ladder).
- Checkpoints: `_best.pt` (≈ BC seed; floor mechanism froze it at step 0), `_latest.pt`, `snapshots/iter0010_ep00040.pt`.

Held-out full-110 evals (`_eval_failure_focus_v1/`):

| Ckpt | Suite | n | success | CI95 | timeout | collision | eval dir | mtime |
|---|---|---|---|---|---|---|---|---|
| `rl_td3bc_failure_focus_v1_latest.pt` | full-110 | 109 | **45.9%** | 36.7–56.0% | 47.7% | 6.4% | `_eval_failure_focus_v1/rl_td3bc_failure_focus_v1__rl_td3bc_failure_focus_v1_latest/` | Jun 3 10:51 |

`_best.pt` not full-110 evaluated — would be identical to BC seed (62.4%) since the floor mechanism froze it at the seed. `_latest.pt` regressed on cross-object ladder (-16.5 pts vs seed), expected: training pool was clock-only, hold-out is ladder. The narrow-pool design optimized same-goal performance at the cost of cross-object generalization — opposite trade-off from `_max`'s broader pool.

**Compare with `_max`:** same algorithm (TD3+BC), same seed, same eval. `_max` (50-traj broad pool, clock+drill) → 65.1%. `_failure_focus_v1` (40-traj narrow pool, clock-only failures) → 45.9%. The broader, balanced pool wins on full-110 ladder.

---

### `rl_td3bc_dagger_r12_80ep` (completed, validation run)

- **Config:** `nav_policy/configs/train_rl_flightroom_td3bc_dagger_r12_80ep.yaml`
- **Run dir:** `/project/kothari1/vlead_data/rl_runs/dagger_r12/rl_td3bc_dagger_r12_80ep/`
- **TB:** `.../rl_td3bc_dagger_r12_80ep/tb/events.out.tfevents.{1780451402,1780451734}.*`
- **Algorithm:** TD3+BC (critic_warmup=40, short fast-iter validation)
- **Pool:** 20 trajs (inherited from `dagger_r12` base config)
- **Episodes:** 80 (20 iters × 4) — purpose: confirm TD3+BC plumbing not broken
- **In-loop eval gate:** holdout-30

| Ckpt | In-loop eval gate | success | source |
|---|---|---|---|
| `rl_td3bc_dagger_r12_80ep_best.pt` | holdout-30 | **90.0%** | ⚠ **= the seed floor, NOT a trained gain.** `best_eval_goal_success_rate: 0.9` is the seed itself (residual zero-init ⇒ policy == BC at ep 0); `eval_floor_from_seed` saved it as the initial best and nothing beat it. |
| `rl_td3bc_dagger_r12_80ep_latest.pt` | (none) | — | |
| Snapshots: iter0005 / 0010 / 0015 / 0020 | (none) | — | |

**Per-eval trajectory on holdout-30 (from TB):** seed floor (ep0) = **27/30
(90%)** → ep40 = **22/30 (73%)** → ep80 = **26/30 (87%)**. The trained
checkpoints *regressed* vs the seed (dipped on early exploration, recovering by
ep80 but still below). `_best.pt` is therefore the seed. **Takeaways:** (1) no
collapse — `q1_loss` stayed bounded (one 9.6 spike, recovered; vs the 8h run's
150+), the eval-floor held; (2) holdout-30 has no headroom for this seed, which
is why `_max` switched to clockdrill-40. **No standalone full-110 eval run.**

---

### `rl_sac_dagger_r12_long_8h_v1` (completed; this is the regression)

- **Config:** `nav_policy/configs/train_rl_flightroom_sac_dagger_r12_long_8h.yaml`
- **Run dir:** `/project/kothari1/vlead_data/rl_runs/dagger_r12/rl_sac_dagger_r12_long_8h_v1/`
- **TB:** `.../rl_sac_dagger_r12_long_8h_v1/tb/events.out.tfevents.1780384914.*`
- **Algorithm:** SAC (classic, no residual; this is what we diagnosed regressed)
- **Pool:** 20 trajs
- **Episodes:** 1600 (400 iters × 4) — completed Jun 2 06:44
- **In-loop eval gate:** holdout-30

Checkpoints emitted: 2 top-level (`_best.pt`, `_latest.pt`) + 40 snapshots
(every 10 iters: `iter0010_ep00040.pt` ... `iter0400_ep01600.pt`).

Held-out evals (`eval_110_20260602/`, queue elapsed 5.4 h):

| Ckpt | Suite | n | success | CI95 | timeout | collision | eval dir | mtime |
|---|---|---|---|---|---|---|---|---|
| `bc_best_balanced_dagger_r12_new.pt` (baseline) | full-110 | 109 | 62.4% | 53.2–70.7% | 31.2% | 6.4% | `eval_110_20260602/checkpoints__bc_best_balanced_dagger_r12_new/` | Jun 2 11:13 |
| `iter0100_ep00400` | full-110 | 109 | **16.5%** | 10.1–23.9% | 79.8% | 4.6% | `.../snapshots__..._iter0100_ep00400/` | Jun 2 11:59 |
| `iter0200_ep00800` | full-110 | 109 | **45.9%** | 36.7–55.0% | 47.7% | 6.4% | `.../snapshots__..._iter0200_ep00800/` | Jun 2 12:45 |
| `iter0300_ep01200` | full-110 | 109 | **4.6%** | 0.9–9.2% | 89.0% | 6.4% | `.../snapshots__..._iter0300_ep01200/` | Jun 2 13:32 |
| `iter0400_ep01600` | full-110 | 109 | **32.1%** | 23.9–41.3% | 57.8% | 11.0% | `.../snapshots__..._iter0400_ep01600/` | Jun 2 14:18 |
| `_best.pt` (selected on in-loop eval) | full-110 | 109 | **55.0%** | 45.9–64.2% | 40.4% | 4.6% | `.../rl_sac_dagger_r12_long_8h_v1__rl_sac_dagger_r12_long_8h_v1_best/` | Jun 2 15:04 |
| `_latest.pt` | full-110 | 109 | 32.1% | 23.9–41.3% | 57.8% | 11.0% | `.../rl_sac_dagger_r12_long_8h_v1__rl_sac_dagger_r12_long_8h_v1_latest/` | Jun 2 15:51 |

**Pattern.** SAC drifted below BC seed (55% < 62.4%) on the held-out test set,
even though the in-loop holdout-30 gate had selected `_best.pt`. Diagnosed
causes: FIFO replay (50k cap vs 384k transitions = 87% eviction), narrow
20-traj train pool, sparse eval cadence. Triggered the TD3+BC switch.

---

### `rl_sac_dagger_r12_v8` (completed)

- **Config:** `nav_policy/configs/train_rl_flightroom_sac_dagger_r12.yaml` (canonical SAC + dagger_r12 seed)
- **Run dir:** `/project/kothari1/vlead_data/rl_runs/dagger_r12/rl_sac_dagger_r12_v8/`
- **TB:** `.../rl_sac_dagger_r12_v8/tb/events.out.tfevents.1780381358.*`
- **Algorithm:** SAC. **Iterations:** completed at iter 285 (Jun 2 00:07).
- Checkpoints: `_best.pt`, `_latest.pt` (no snapshots).

| Ckpt | Suite | n | success | CI95 | timeout | collision | eval dir | mtime |
|---|---|---|---|---|---|---|---|---|
| `rl_sac_dagger_r12_v8_best.pt` | full-110 | 109 | **63.3%** | 54.1–72.5% | 32.1% | 4.6% | `rl_runs/dagger_r12/_eval_batch_20260602/rl_sac_dagger_r12_v8__rl_sac_dagger_r12_v8_best/` | Jun 2 03:48 |

---

### `rl_sac_dagger_r12_v7` (completed)

- **Config:** `nav_policy/configs/train_rl_flightroom_sac_dagger_r12.yaml`
- **Run dir:** `/project/kothari1/vlead_data/rl_runs/dagger_r12/rl_sac_dagger_r12_v7/`
- **TB:** `.../rl_sac_dagger_r12_v7/tb/events.out.tfevents.1780378652.*`
- **Algorithm:** SAC. **Completed:** Jun 1 23:20.
- Checkpoints: `_best.pt`, `_latest.pt`.

| Ckpt | Suite | n | success | CI95 | timeout | collision | eval dir | mtime |
|---|---|---|---|---|---|---|---|---|
| `rl_sac_dagger_r12_v7_best.pt` | full-110 | 109 | **63.3%** | 54.1–72.5% | 33.9% | 5.5% | `rl_runs/dagger_r12/_eval_batch_20260602/rl_sac_dagger_r12_v7__rl_sac_dagger_r12_v7_best/` | Jun 2 03:00 |
| `rl_sac_dagger_r12_v7_best.pt` | holdout-14 | 14 | 57.1% | — | 35.7% | 14.3% | `rl_runs/dagger_r12/rl_sac_dagger_r12_v7_holdout14_eval/` | Jun 1 23:31 |

v7 and v8 tie at 63.3% — both clear BC seed by ~1 pt.

---

### `rl_sac_dagger_r12_v6` (completed)

- **Run dir:** `/project/kothari1/vlead_data/rl_runs/dagger_r12/rl_sac_dagger_r12_v6/`
- **TB:** `.../rl_sac_dagger_r12_v6/tb/events.out.tfevents.{1780375898,1780376752}.*`
- Checkpoints: `_best.pt`, `_latest.pt`.
- **No full-110 eval has been run on v6.**

---

### `rl_ppo_dagger_r12_v7` (completed)

- **Run dir:** `/project/kothari1/vlead_data/rl_runs/dagger_r12/rl_ppo_dagger_r12_v7/`
- **TB:** `.../rl_ppo_dagger_r12_v7/tb/events.out.tfevents.1780378961.*`
- Checkpoints: `_best.pt`, `_latest.pt`.
- **No full-110 eval has been run on v7.**

---

### `rl_ppo_dagger_r12_v6` (completed)

- **Run dir:** `/project/kothari1/vlead_data/rl_runs/dagger_r12/rl_ppo_dagger_r12_v6/`
- **TB:** `.../rl_ppo_dagger_r12_v6/tb/events.out.tfevents.1780375864.*`
- Checkpoints: `_best.pt`, `_latest.pt`.

| Ckpt | Suite | n | success | CI95 | timeout | collision | eval dir | mtime |
|---|---|---|---|---|---|---|---|---|
| `rl_ppo_dagger_r12_v6_best.pt` | full-110 | 109 | **40.4%** | 31.2–49.5% | 53.2% | 10.1% | `rl_runs/dagger_r12/_eval_batch_20260602/rl_ppo_dagger_r12_v6__rl_ppo_dagger_r12_v6_best/` | Jun 2 04:36 |

PPO underperformed SAC on the same seed.

---

### `rl_sac_dagger_r12_v4` and `v5` (loose files, no run dir)

These are floating outputs (no enclosing dir) under
`/project/kothari1/vlead_data/rl_runs/dagger_r12/`:

- `rl_sac_dagger_r12_v4_{best,latest}.pt`, `_episodes.csv`, `_log.csv`, `_summary.json`
- `rl_sac_dagger_r12_v5_{best,latest}.pt`, `_episodes.csv`, `_log.csv`
  (no summary.json)

`v4` summary reports: 80 episodes, `best_mean_return: 22.05`, no held-out
eval. Both are exploratory pre-v6 SAC iterations. No closed-loop eval
was ever run on these checkpoints.

---

### `td3bc_smoke` (completed, plumbing test only)

- **Config:** `nav_policy/configs/train_rl_flightroom_td3bc_smoke.yaml` (critic_warmup=0)
- **Run dir:** `/project/kothari1/vlead_data/rl_runs/td3bc_smoke/rl_td3bc_smoke/`
- **TB:** `.../rl_td3bc_smoke/tb/events.out.tfevents.1780450098.*`
- 2 iters / 4 episodes total. Purpose: exercise actor-update code path
  from step 1.
- Checkpoints: `_best.pt`, `_latest.pt`, snapshots `iter0001`, `iter0002`.
- **No closed-loop eval intended.**

---

### Legacy: `sac_v1` and `sac_v1_seed0` (May, pre-rewrite SB3 SAC)

- `/project/kothari1/vlead_data/rl_runs/sac_v1/` — TB at `tb/SAC_1/events.out.tfevents.1780003981.*`
- `/project/kothari1/vlead_data/rl_runs/sac_v1_seed0/` — TB at `tb/SAC_1/events.out.tfevents.1780005163.*`

Both use the legacy SB3 stack (pre-canonical-trainer rewrite). Each has:
`ckpt/`, `best/`, `eval/evaluations.npz` (SB3 internal eval format,
not summary.json). Numbers not directly comparable to the dagger_r12 family.

**No closed-loop summary.json exists** for these; the only eval artifact
is the SB3 `evaluations.npz` inside `eval/`.

---

## Per-rollout CSVs (for matrix analysis / plotting)

`nav_policy/scripts/analyze_eval_matrix.py` consumes these to build the
per-query × per-ckpt success matrix (used to identify always-fail trajs,
Jaccard overlap between ckpts, etc.).

| Eval dir | per_rollout.csv |
|---|---|
| `rl_runs/dagger_r12/_eval_batch_20260602/checkpoints__bc_best_balanced_dagger_r12_new/` | ✓ |
| `rl_runs/dagger_r12/_eval_batch_20260602/checkpoints__bc_best/` | ✓ |
| `rl_runs/dagger_r12/_eval_batch_20260602/rl_sac_dagger_r12_v7__..._best/` | ✓ |
| `rl_runs/dagger_r12/_eval_batch_20260602/rl_sac_dagger_r12_v8__..._best/` | ✓ |
| `rl_runs/dagger_r12/_eval_batch_20260602/rl_ppo_dagger_r12_v6__..._best/` | ✓ |
| `rl_runs/dagger_r12/rl_sac_dagger_r12_long_8h_v1/eval_110_20260602/*/` (8 dirs) | ✓ |
| `rl_runs/dagger_r12/rl_sac_dagger_r12_v7_holdout14_eval/` | ✓ |
| `rl_runs/rahul_best/bc_best_rahul_holdout14_eval/` | ✓ |
| `rl_runs/bc_train_eval/071353/checkpoints__bc_best_balanced_dagger_r12_new/` | ✓ (175 rows, phase-A failure mining) |
| `rl_runs/dagger_r12/rl_td3bc_dagger_r12_max/eval_110_20260603/*_best/` | ✓ (the 65.1% headline) |
| `rl_runs/dagger_r12/rl_td3bc_dagger_r12_max/eval_110_20260603/*_latest/` | ✓ |
| `rl_runs/dagger_r12/_eval_failure_focus_v1/*_latest/` | ✓ |

---

## TensorBoard event-file index

Single `tensorboard --logdir /project/kothari1/vlead_data/rl_runs/` will
recursively pick up every event file below. Per-run paths for reference:

| Run | TB events |
|---|---|
| `rl_ppo_dagger_r12_v6` | `tb/events.out.tfevents.1780375864.*` |
| `rl_ppo_dagger_r12_v7` | `tb/events.out.tfevents.1780378961.*` |
| `rl_sac_dagger_r12_v6` | `tb/events.out.tfevents.{1780375898,1780376752}.*` (2; second is resume) |
| `rl_sac_dagger_r12_v7` | `tb/events.out.tfevents.1780378652.*` |
| `rl_sac_dagger_r12_v8` | `tb/events.out.tfevents.1780381358.*` |
| `rl_sac_dagger_r12_long_8h_v1` | `tb/events.out.tfevents.1780384914.*` |
| `rl_td3bc_dagger_r12_80ep` | `tb/events.out.tfevents.{1780451402,1780451734}.*` |
| `rl_td3bc_dagger_r12_max` | `tb/events.out.tfevents.{1780456335,1780461176}.*` |
| `rl_td3bc_failure_focus_v1` | `tb/events.out.tfevents.1780465425.*` |
| `td3bc_smoke/rl_td3bc_smoke` | `tb/events.out.tfevents.1780450098.*` |
| `sac_v1` (legacy) | `tb/SAC_1/events.out.tfevents.1780003981.*` |
| `sac_v1_seed0` (legacy) | `tb/SAC_1/events.out.tfevents.1780005163.*` |

Key scalar tags emitted by the canonical trainer (`nav_policy.rl.train_rl`):
- `train/actor_loss`, `train/critic_loss`, `train/bc_anchor_loss`, `train/q_mean`
- `eval/<suite>/goal_success_rate`, `eval/<suite>/timeout_rate`, `eval/<suite>/collision_rate`
- `episodes/return_mean`, `episodes/success_rate`

---

## Headline summary (TL;DR ranking, full-110 only)

| Ckpt | full-110 success | Δ vs BC seed | notes |
|---|---|---|---|
| 🏆 `rl_td3bc_dagger_r12_max_best` | **65.1%** | **+2.7 pt** | **new best**; TD3+BC, 50-traj clock+drill pool |
| `rl_sac_dagger_r12_v7_best` | 63.3% | +0.9 pt | tied with v8 |
| `rl_sac_dagger_r12_v8_best` | 63.3% | +0.9 pt | tied with v7 |
| `bc_best_balanced_dagger_r12_new` (seed) | 62.4% | (baseline) | |
| `rl_td3bc_dagger_r12_max_latest` | 56.9% | −5.5 pt | late-iter drift; floor caught _best earlier |
| `rl_sac_dagger_r12_long_8h_v1_best` | 55.0% | **−7.4 pt** | regression; triggered TD3+BC pivot |
| `rl_td3bc_failure_focus_v1_latest` | 45.9% | −16.5 pt | narrow clock-only pool ⇒ cross-object regression |
| `rl_ppo_dagger_r12_v6_best` | 40.4% | −22.0 pt | PPO worse than SAC |
| `rl_sac_dagger_r12_long_8h_v1_latest` | 32.1% | −30.3 pt | |
| `bc_best.pt` (legacy cross-env) | 2.8% | −59.6 pt | not comparable |

**Headline result.** TD3+BC `_max_best` beats every other RL run AND the BC seed
on the cross-object full-110, **even though it was selected on clockdrill-40
(a different suite)**. The residual head + frozen base + eval-floor combination
worked as designed: improved on the trained goals AND generalized to the held-out
object. SAC v7/v8 (+0.9 pt) are within noise of seed; `_max` (+2.7 pt) is the
first run with a meaningful gap.

**Caveat on this ranking:** full-110 is the *ladder* goal, never in RL training
(clock/drill) — see the cross-object caveat in Conventions. So full-110 ≈ seed
is the *expected best case* for any RL run; it measures generalization, not the
gain RL is optimizing. Same-goal improvement must be read off **clockdrill-40**.

**Pool-design contrast:** `_max` (50-traj broad, clock+drill stratified) and
`_failure_focus_v1` (40-traj narrow, clock-only mined failures) used the same
algorithm + seed + eval. Result: broad pool +2.7 pt, narrow-mined pool −16.5 pt.
The narrow pool over-fit a single object's failure modes at the cost of ladder
generalization. **Implication for future runs:** for cross-object hold-out
improvement, breadth and balance beat targeted failure curation.

---

## Maintenance

Whenever a new eval completes, append a row to the matching RL run
section. Source of truth for cell values:

- `goal_success_rate`, `n_rollouts`, `checkpoint`, `metrics_config` →
  `{eval_dir}/summary.json`
- `mtime` → `ls -la --time-style=long-iso {eval_dir}/summary.json`
- Path of `per_rollout.csv` → next to `summary.json` in the same dir
- Eval suite name → infer from the config in `{batch_parent}/queue_log.json`
  `config` field

When a brand-new RL run finishes, add a new RL-run subsection mirroring
the layout above: config path, run dir, TB events path, algorithm,
episodes, ckpt list, and an empty eval table to be populated as evals
run.
