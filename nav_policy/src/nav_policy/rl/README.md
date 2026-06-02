# V-LEAD RL — quick reference

Canonical PPO + SAC fine-tuning for the V-LEAD pilot network, with BC KL
anchor, held-out deterministic eval, TensorBoard, and per-run perf summary.

> **All commands at the [end of this README](#commands).** Skip there for
> the cheat-sheet.

## What's in this folder

| File | Role |
|---|---|
| `train_rl.py` | trainer entry; routes PPO vs SAC via yaml `rl.algorithm` |
| `stochastic_policy.py` | Gaussian-actor wrapper around BC policy + critic + KL anchor |
| `ppo.py` | clipped-objective PPO update + BC anchor term |
| `sac.py` | twin-Q SAC update + BC anchor term |
| `buffer.py` | `RolloutBuffer` (PPO) and `ReplayBuffer` (SAC) |
| `rollout.py` | `RLTrainingController` (FiGS-compatible) + episode collection |
| `rewards.py` | `compute_episode_rewards`; returns per-step list + per-term components |
| `train_eval.py` | deterministic eval-on-suite for in-training checkpoint selection |
| `tb_logger.py` | thin SummaryWriter wrapper with no-op fallback |
| `perf_summary.py` | startup model + memory + latency summary |
| `paths.py` | rollout-video path helpers |

## Run layout (per-run subfolders)

```
{checkpoint_dir}/
  {run_tag}/
    {run_tag}_log.csv            # per-iter scalars
    {run_tag}_episodes.csv       # per-episode rows
    {run_tag}_summary.json
    {run_tag}_perf.json          # startup model+memory+latency snapshot
    {run_tag}_best.pt            # best by eval/goal_success_rate
    {run_tag}_latest.pt
    tb/                          # TensorBoard event files
    videos/iter*_*.mp4           # one per episode if --save-videos
```

## TensorBoard tags

| Group | Tags |
|---|---|
| `rollout/` | `mean_return`, `mean_steps`, `success_rate` (per-iter, on training rollouts) |
| `train/` | `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `ref_kl`, `q1_loss`, `q2_loss`, `alpha`, `policy_log_std_mean` |
| `reward/` | per-iter mean of each reward term: `progress_mean`, `heading_mean`, `step_mean`, `action_smooth_mean`, `bbox_mean`, `collision_mean`, `timeout_mean`, `success_mean`, `total_mean` |
| `episode/` | `return`, `steps`, `final_pos_err_m`, `success`, `collision`, `goal_settled` (per-episode, indexed by global_episode) |
| `eval/` | `goal_success_rate`, `n_success`, `n_rollouts` (per eval event, every `eval_every_episodes` eps) |
| `eval_per_query/` | one 0/1 per held-out query name (per eval event) |
| `perf/summary` | startup JSON dump (text panel) |

## Train rollouts vs held-out eval rollouts

| Set | Source | Used by |
|---|---|---|
| TRAIN | `data/raw/flightroom_ssv_exp_2026-05-22_071353/` + `..._071718/` | RL on-policy collection (yaml's `rollouts:` list, 4 queries) |
| HELDOUT VAL | `data/raw/flightroom_ssv_exp_2026-05-22_071733_trajs-110/` (q12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103) | `configs/eval_closed_loop_flightroom_holdout_14.yaml` — in-training deterministic eval + `_best.pt` selection |
| FULL 110 (TEST) | same dir, all 110 | One-shot post-training eval via `--rollouts-from-dir` |

`/project/.../2026-05-22_071733/...` is symlinked at
`nav_policy/data/raw/flightroom_ssv_exp_2026-05-22_071733_trajs-110`.

## Reward function

File: `rewards.py::compute_episode_rewards`. Per step `i`:

- `progress_weight * (prev_dist_xy - dist_xy)`
- `heading_weight * cos(angle(vel_xy, goal_dir_xy))`
- `step_penalty`
- `bbox_penalty` if outside expert bbox + margin
- `collision_penalty` if FiGS reports a collision
- `- action_smooth_weight * ||a_t − a_{t-1}||²`

Terminal one-shot:
- `success_bonus` only if `goal_settled` AND no bbox / collision in episode
- `timeout_penalty` only if `termination == "timeout"` AND not settled / collided

XY-plane only for progress and heading. Returns `(rewards, components)`
where the dict has each term's total contribution across the episode —
fed into TB `reward/*_mean`.

## BC KL anchor

Both PPO and SAC add `ref_kl_coef × KL(BC ‖ current)` to their policy loss
(Round 4 work). Yaml block (works for either algorithm):

```yaml
rl:
  bc_anchor:
    kl_coef: 0.02
  sac:
    ref_kl_coef: 0.02   # SAC reads
  ppo:
    ref_kl_coef: 0.02   # PPO reads
```

`StochasticVelocityPolicy.frozen_reference_copy()` produces the lean
reference (no critic deepcopy). `policy.kl_to(other, rgb, goal, depth)`
computes the per-sample KL.

## Known gotchas

- **`policy.eval()` blocks RNN backward.** Anything that calls
  `policy.eval()` (perf summary; closed-loop eval) must restore the prior
  mode or the next SAC/PPO backward crashes with
  `RuntimeError: cudnn RNN backward can only be called in training mode`.
  Both `perf_summary._inference_latency` and `SACTrainer.update` are now
  guarded.
- **`compress_transitions=True` stores rgb in float16.** SAC casts to
  float32 at `SACTrainer.update`'s top; PPO casts at `ppo_update` batch
  build.
- **Replay theoretical footprint = capacity × per-tr bytes.** Perf summary
  prints it. At `replay_capacity: 100000`, `T=4`, `224²`, fp16 with depth
  off it's ~300 GB. Container has 251 GB RAM; actual usage caps at the
  episodes you actually collect (~16k transitions for a 20-iter run = ~48
  GB). Drop `sac.replay_capacity` if you bump iters.
- **`policy.act` and `policy.q_input` share `_encode`** (one backbone
  forward instead of two). SAC.update uses `policy.act_full(...)` to also
  reuse the encoder latent for the Q-net forward.

---

# Commands

All commands assume you're **inside** the `vlead` Docker container at
`/workspace/nav_policy` unless noted otherwise.

## 0. Enter the container

```bash
# from host
cd ~/autonomy_projects/V-LEAD
CUDA_VISIBLE_DEVICES=<gpu_id> docker compose run --rm vlead
# then inside:
cd /workspace/nav_policy
```

Wrap the launch in `tmux` if you want the run to survive logging out:

```bash
# host
tmux new -s sac_run
# inside tmux:
cd ~/autonomy_projects/V-LEAD
CUDA_VISIBLE_DEVICES=0 docker compose run --rm vlead bash -c "
cd /workspace/nav_policy && \
python -m nav_policy.rl.train_rl \
    --config configs/train_rl_flightroom_sac_dagger_r12.yaml \
    --run-tag rl_sac_dagger_r12_v8 --save-videos
"
# detach: Ctrl-B then D
# reattach: tmux attach -t sac_run
```

## 1. Train SAC

```bash
python -m nav_policy.rl.train_rl \
    --config configs/train_rl_flightroom_sac_dagger_r12.yaml \
    --run-tag rl_sac_dagger_r12_v8 \
    --save-videos
```

Optional flags:
- `--seed 42` — override `rl.seed` for multi-seed runs
- `--n-iterations 1` — smoke test (1 iter only)
- `--rollouts-per-iteration 2` — smoke test
- `--resume-from /path/to/run/{tag}_latest.pt` — continue a run

## 2. Train PPO

Same trainer, different yaml; `rl.algorithm: ppo` is set in the canonical
PPO config.

```bash
python -m nav_policy.rl.train_rl \
    --config configs/train_rl_flightroom_ppo_dagger_r12.yaml \
    --run-tag rl_ppo_dagger_r12_v8 \
    --save-videos
```

## 3. Watch progress

Per-iter CSV tail:
```bash
# host
cd ~/autonomy_projects/V-LEAD
tail -F /project/kothari1/vlead_data/rl_runs/dagger_r12/<run_tag>/<run_tag>_log.csv
```

Per-episode CSV pretty-print:
```bash
column -s, -t \
    < /project/kothari1/vlead_data/rl_runs/dagger_r12/<run_tag>/<run_tag>_episodes.csv \
    | tail -20
```

TensorBoard (host or container; host preferred):
```bash
# host — point at the per-run-tag root to see all runs side by side
~/.local/bin/tensorboard \
    --logdir /project/kothari1/vlead_data/rl_runs/dagger_r12 \
    --port 6006
# open http://coruscant:6006
```

Inspect the startup perf snapshot:
```bash
cat /project/kothari1/vlead_data/rl_runs/dagger_r12/<run_tag>/<run_tag>_perf.json | jq
```

## 4. Closed-loop eval against a fixed yaml suite

Standard pattern: use the held-out 14-query suite that's also used during
in-training eval.

```bash
python scripts/eval_in_figs.py \
    --config configs/eval_closed_loop_flightroom_holdout_14.yaml \
    --checkpoint /project/kothari1/vlead_data/rl_runs/dagger_r12/<run_tag>/<run_tag>_best.pt \
    --output-dir /project/kothari1/vlead_data/rl_runs/dagger_r12/<run_tag>/holdout14_eval
```

Output: `summary.json` + per-rollout artifacts under `--output-dir`.

## 5. Closed-loop eval against the full 110-traj pool (publishable test number)

```bash
python scripts/eval_in_figs.py \
    --config configs/eval_closed_loop_flightroom_holdout_14.yaml \
    --checkpoint /project/kothari1/vlead_data/rl_runs/dagger_r12/<run_tag>/<run_tag>_best.pt \
    --output-dir /project/kothari1/vlead_data/rl_runs/dagger_r12/<run_tag>/test110_eval \
    --rollouts-from-dir data/raw/flightroom_ssv_exp_2026-05-22_071733_trajs-110
```

`--rollouts-from-dir` overrides the yaml's `rollouts:` block with **every**
`trajectories_val*.pt` in the directory (sorted by index). The yaml's
`metrics:` block still drives tolerances + warmup.

Cap for smoke testing:
```bash
... --rollouts-from-dir data/raw/flightroom_ssv_exp_2026-05-22_071733_trajs-110 \
    --rollouts-limit 5
```

## 6. Eval the BC baseline (un-fine-tuned)

Same script, point at the BC ckpt directly:

```bash
python scripts/eval_in_figs.py \
    --config configs/eval_closed_loop_flightroom_holdout_14.yaml \
    --checkpoint data/checkpoints/bc_best_balanced_dagger_r12_new.pt \
    --output-dir /project/kothari1/vlead_data/rl_runs/bc_baseline/holdout14_eval \
    --rollouts-from-dir data/raw/flightroom_ssv_exp_2026-05-22_071733_trajs-110
```

Use the same `--rollouts-from-dir` arg to get the "BC over the full 110"
baseline for the paper.

## 7. Inspect a checkpoint's eval meta

The RL checkpoint pickle includes the meta dict written at save time
(includes `eval_goal_success_rate`, `eval_per_query`, `eval_suite`,
`global_episode`, etc.).

```bash
python - <<'PY'
import torch
ckpt = torch.load(
    "/project/kothari1/vlead_data/rl_runs/dagger_r12/<run_tag>/<run_tag>_best.pt",
    weights_only=False, map_location="cpu",
)
m = ckpt.get("rl_meta", {})
print(f"global_episode={m.get('global_episode')}")
print(f"eval_suite={m.get('eval_suite')}")
print(f"eval_goal_success_rate={m.get('eval_goal_success_rate')}")
for q, ok in (m.get("eval_per_query") or {}).items():
    print(f"  {q}: {'ok' if ok else 'FAIL'}")
PY
```

## 8. Quick scalars from TB

```bash
python - <<'PY'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
tb_dirs = sorted(glob.glob("/project/kothari1/vlead_data/rl_runs/dagger_r12/<run_tag>/tb"))
ea = EventAccumulator(tb_dirs[-1], size_guidance={"scalars": 0}); ea.Reload()
for k in ("rollout/mean_return", "rollout/success_rate", "eval/goal_success_rate",
          "train/ref_kl", "train/policy_log_std_mean"):
    if k in ea.Tags()["scalars"]:
        evs = ea.Scalars(k)
        print(f"{k}: start={evs[0].value:+.3f} end={evs[-1].value:+.3f}")
PY
```

## 9. BC stack (legacy, for ref)

```bash
# build dataset (one-shot, takes ~30 min on first run)
python scripts/build_dataset.py --config configs/default.yaml

# train BC baseline (no RL)
python scripts/train_bc.py --config configs/default.yaml
```

Output BC ckpt under `<checkpoint_dir>/bc_best.pt`. Point
`rl.checkpoint: data/checkpoints/<your_bc>.pt` at it in the RL yaml.
