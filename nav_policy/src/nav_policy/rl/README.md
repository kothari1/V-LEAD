# V-LEAD Online RL

Gymnasium-compatible Reinforcement Learning scaffold for the V-LEAD pilot
network. Wraps the FiGS drone simulator + 3DGS renderer in a `gym.Env`,
plugs Stable-Baselines3 SAC on top with the BC visual encoder reused as
the actor-critic feature extractor, and supports warm-starting both the
encoder and the actor head from a BC checkpoint.

> **All commands at the [end of this README](#quick-command-reference).**

---

## Table of contents

1. [What this gives you](#what-this-gives-you)
2. [Repo layout](#repo-layout)
3. [Environment (`FigsDroneEnv`)](#environment-figsdroneenv)
4. [Reward](#reward) — **most-tuned thing, where to change it**
5. [Termination](#termination)
6. [Episode sampler](#episode-sampler)
7. [Network + warm-start](#network--warm-start)
8. [Training: config + callbacks](#training-config--callbacks)
9. [Evaluation](#evaluation)
10. [Tuning recipes](#tuning-recipes)
11. [Known gotchas](#known-gotchas)
12. [Quick command reference](#quick-command-reference)

---

## What this gives you

- **One pilot network, three training modes**: offline Behavior Cloning
  (existing `nav_policy.train.train_bc`), online RL via SAC
  (`nav_policy.rl.train.train_sac`), and DAgger relabeling (existing).
- **Reuse of the BC visual encoder** in the RL actor-critic so an RL run
  doesn't have to learn vision from scratch.
- **Two-level warm-start** from BC: encoder weights (always), and
  optionally the actor latent-MLP + mean head from BC's MLPHead.
- **Composable per-step reward** that is logged term-by-term to
  TensorBoard so you can see exactly which signal drives the policy.
- **Drop-in SB3 callbacks**: periodic eval, checkpointing,
  per-component reward logging, optional W&B.
- **Closed-loop evaluation** script that loads a saved SAC `.zip` and
  rolls out deterministic episodes in FiGS.

---

## Repo layout

The RL code lives in two packages — `vlead_flight` owns everything that
touches the FiGS simulator; `nav_policy` owns model / algorithm / config.

```
V-LEAD/
├── vlead/vlead_flight/
│   ├── observation.py             # compute_goal, preprocess_rgb/depth, FrameBuffer
│   ├── _torch_compat.py           # torch>=2.6 weights_only fix for nerfstudio ckpts
│   └── env/
│       ├── figs_drone_env.py      # FigsDroneEnv (Gymnasium)
│       ├── episode_sampler.py     # EpisodeSampler (start + goal randomization)
│       ├── reward.py              # GoalReward + RewardConfig            ← edit weights here
│       └── termination.py         # TerminationConfig + check_termination
│
└── nav_policy/
    ├── configs/
    │   └── sac_default.yaml       # ← the one yaml that drives everything
    ├── scripts/
    │   ├── train_sac.py           # CLI: train
    │   ├── eval_sac.py            # CLI: evaluate saved ckpt
    │   └── smoke_env.py           # CLI: random-action env smoke
    └── src/nav_policy/rl/
        ├── model/
        │   └── feature_extractor.py   # BCEncoderFeatureExtractor (SB3)
        ├── warm_start/
        │   └── bc_to_rl.py            # BC ckpt → SAC encoder + actor
        ├── train/
        │   ├── train_sac.py           # main trainer
        │   └── callbacks.py           # eval / ckpt / reward-components / W&B
        └── eval/
            └── eval_sac.py            # closed-loop evaluation
```

---

## Environment (`FigsDroneEnv`)

Class: `vlead_flight.env.FigsDroneEnv` ([figs_drone_env.py](../../../../../vlead/vlead_flight/env/figs_drone_env.py))

**Action space** (`Box`, 4-dim):
`[vx, vy, vz, psi_dot]` in world frame (NED). Defaults from yaml:
`low = [-3, -3, -0.3, -1.5]`, `high = [3, 3, 0.3, 1.5]` (m/s, m/s, m/s, rad/s).
Actions are converted to body-rate commands `[uf, ωx, ωy, ωz]` by a
P-cascaded `figs.control.VelocityController`, which is then integrated
by ACADOS for `hz_sim / hz_ctrl` substeps per env step.

**Observation space** (`Dict`):
- `rgb`: `Box(0, 255, (T, 3, H, W), uint8)` — T=4 frames temporal stack,
  224×224 each. Stored as `uint8` to keep replay buffer compact;
  ImageNet-normalized on GPU inside the feature extractor.
- `goal`: `Box(-inf, inf, (4,), float32)` — `[hx, hy, hz, d/scale]`
  (unit heading vector + scale-normalized distance). Set
  `env.goal_input_dim=3` to drop `hz`.

**Per-step flow:**
1. Clip the action to `action_space`.
2. `VelocityController.control(action)` → body-rate command.
3. Loop `n_sim2ctl` ACADOS substeps with held cmd.
4. `gsplat.render_rgb(camera, T_c2w)` at the new pose → new RGB frame.
5. Compute `goal_heading`, `dist_to_goal`, reward components,
   termination.
6. Return Gymnasium 5-tuple. `info` contains
   `reward_components` (dict), `term_reason` (str), `dist_to_goal`,
   `x` (full state), `ucr` (body-rate cmd).

The env **does not** drive `Simulator.simulate()` — it owns
`Simulator.solver`, `Simulator.gsplat`, and `VelocityController`
directly so it has clean per-step control.

---

## Reward

File: [vlead/vlead_flight/env/reward.py](../../../../../vlead/vlead_flight/env/reward.py)
Class: `GoalReward(cfg: RewardConfig)`
Configured in: `nav_policy/configs/sac_default.yaml` under `reward:`

Each step, `GoalReward.__call__` computes:

| Term | Formula | Default weight | What it does |
|---|---|---|---|
| `progress` | `(prev_dist - new_dist)` | `w_progress = 1.0` | Reward for closing distance to goal. Positive when drone is moving toward target. |
| `success` | `+1` at terminal if `dist < success_radius`, else `0` | `w_success = 50.0` | One-shot bonus when goal reached. |
| `alive` | `-1` per step | `w_alive = 0.01` | Constant per-step cost (encourages finishing). |
| `crash` | `-1` at terminal if `term_reason ∈ {bbox_violation, ground_crash, ceiling_crash, overspeed}`, else `0` | `w_crash = 50.0` | One-shot penalty for any crash mode. |
| `smooth` | `-‖a_t − a_{t-1}‖²` (zero on first step) | `w_smooth = 0.05` | Penalizes jittery action sequences. |
| `yaw` | `−` arc-cos angle between drone body-x axis (XY) and unit goal heading (XY) | `w_yaw = 0.05` | Encourages drone to face the goal. |
| `altitude` | `−(pz − alt_target)²` | `w_altitude = 1.0` | Holds drone at `alt_target = -1.2` m (NED). |
| `speedcap` | `−max(0, ‖v‖ − speed_cap)²` | `w_speedcap = 0.1`, `speed_cap = 3.0` | Penalizes runaway speeds. |

Total reward is the weighted sum: `sum(w_i * term_i)`. The per-term and
total values are stored in `info["reward_components"]` so the
`RewardComponentsCallback` can log each to TensorBoard:

```
reward_components/progress
reward_components/success
reward_components/alive
reward_components/crash
reward_components/smooth
reward_components/yaw
reward_components/altitude
reward_components/speedcap
reward_components/total
```

### Editing the reward

- **Reweight an existing term**: change `reward.w_*` in
  `sac_default.yaml`. Restart training.
- **Add a new term**: add it to `GoalReward.__call__` in `reward.py`,
  include it in the `comp` dict, and add a `w_*` field to
  `RewardConfig`. It will automatically appear in TensorBoard via the
  reward-components callback (no callback change needed).
- **Change `alt_target`, `speed_cap`, `success_radius`**: these are
  fields of `RewardConfig` and live in the same `reward:` yaml block.

### Practical heuristics

- The `crash` term dominates total reward at episode boundaries because
  it fires once with weight 50. If your TB shows `reward_components/total`
  hovering around `-1` per step but you also see `reward_components/crash`
  in the same magnitude, almost all the "total" is crash penalty —
  the policy needs to learn to not crash before anything else matters.
- `progress` going positive is the first sign of real learning. Until
  then the policy is just reducing crash rate, not navigating.
- The yaw term computes the angle between the drone's body-x axis and
  the goal heading **in the XY plane only**, so vertical heading is
  ignored.

---

## Termination

File: [vlead/vlead_flight/env/termination.py](../../../../../vlead/vlead_flight/env/termination.py)
Function: `check_termination(xcr, dist_to_goal, step_idx, cfg)`

Returns `(terminated, truncated, reason)` per Gymnasium semantics:
**terminated** = the task ended in-domain (success/crash), **truncated** =
ran out of steps.

| Reason | Trigger | Setting |
|---|---|---|
| `success` | `dist_to_goal < success_radius` | `success_radius = 0.5` m |
| `bbox_violation` | drone position outside `[bbox_xyz_low, bbox_xyz_high]` | scene-shaped box |
| `ground_crash` | `pz > ground_z` (NED: positive pz = closer to floor) | `ground_z = -0.05` m |
| `ceiling_crash` | `pz < ceiling_z` | `ceiling_z = -1.95` m |
| `overspeed` | `‖vel‖ > speed_kill` | `speed_kill = 5` m/s |
| `timeout` | step count ≥ `max_episode_steps` | `max_episode_steps = 300` (=15s @ 20Hz) |

`term_reason` is published to `info` and logged as cumulative counters
to TB: `term_reason_count/success`, `term_reason_count/ceiling_crash`,
etc.

---

## Episode sampler

File: [vlead/vlead_flight/env/episode_sampler.py](../../../../../vlead/vlead_flight/env/episode_sampler.py)
Class: `EpisodeSampler`

`reset()` calls `sample()` to get an `EpisodeSpec(x0, target_xyz)`:

1. Start position: uniform in `[start_xyz_low, start_xyz_high]`.
2. Goal direction: random angle θ in XY plane.
3. Goal radius: uniform in `[goal_radius_min, goal_radius_max]`.
4. Goal Z: uniform in `[goal_z_low, goal_z_high]`.
5. Reject + retry (50x) if goal outside `[bbox_xyz_low, bbox_xyz_high]`.
6. Initial attitude: identity quaternion (+ `yaw_jitter` if > 0).
7. Initial velocity: zero (+ `velocity_jitter` if > 0).

All knobs live under `sampler:` in `sac_default.yaml`. Curriculum
trick: shrink `goal_radius_max` early in training (easier task → more
success signal) and bump it back later. Currently a manual yaml edit;
auto-curriculum callback is a future addition.

---

## Network + warm-start

### Shared encoder

`BCEncoderFeatureExtractor`
([feature_extractor.py](model/feature_extractor.py)) wraps a
`RGBVelocityPolicy.encode(rgb, goal)` call so the 288-dim feature
(`gru_hidden + goal_emb_dim`) feeds SB3 SAC's actor and twin critics.

```
rgb [B, T, 3, H, W] uint8     goal [B, 4 or 3]
       │                              │
       │ /255  ImageNet stats         │
       ▼                              │
   _PerFrameResNet18 (T frames)       │
       │ → [B, T, 512]                │
       ▼                              │
       GRU (256) → LayerNorm          │
       │ → [B, 256]                   │
       └──────────────┬───────────────┘
                      ▼
            concat → [B, 288]   ← SB3 features
```

### Two-level warm-start

File: [warm_start/bc_to_rl.py](warm_start/bc_to_rl.py)

1. **Encoder warm-start** (`load_bc_into_feature_extractor`):
   copies BC's `_PerFrameResNet18` + GRU + LayerNorm + goal-embed
   weights into the extractor. BC's MLPHead keys are dropped. Always
   safe.

2. **Actor head warm-start** (`load_bc_into_sac_actor`):
   copies BC's MLPHead Linear layers into SAC actor's `latent_pi` MLP,
   and the first `cmd_dim` (=4) rows of BC's final Linear into
   SAC `actor.mu`. Result: the SAC actor's initial mean action is
   BC's t=0 velocity command. Requires
   `policy_kwargs.net_arch.pi == BC config's mlp_hidden` (default
   `[256, 128]`).

Toggle the actor warm-start via `warm_start.init_actor_from_bc_head: true`
in yaml. Failure logs a WARNING and falls back to encoder-only.

### SB3 quirk: share_features_extractor

In SB3 2.2.1, `policy.features_extractor` can be `None` even with
`share_features_extractor=True`. The shared extractor lives on
`policy.actor.features_extractor` and `policy.critic.features_extractor`.
`_find_bc_extractors` in `train_sac.py` collects every distinct
extractor instance and warm-starts each. When sharing fails silently,
this prints `warm-started 2 extractor instance(s)` instead of 1.

---

## Training: config + callbacks

### `sac_default.yaml` structure

```yaml
seed:               # int
output_dir:         # path
total_timesteps:    # int

env:                # FigsDroneEnv constructor args (scene_name, frame_name, hz, ...)
sampler:            # EpisodeSampler fields
reward:             # RewardConfig fields  ← see Reward section
termination:        # TerminationConfig fields
model:              # feature extractor dims (T, gru_hidden, mlp_hidden, etc.) + actor/critic net_arch
sac:                # SB3 SAC hyperparams (lr, buffer_size, batch_size, tau, gamma, ent_coef, ...)
warm_start:
  bc_checkpoint:    # path to bc_best.pt, or null
  init_actor_from_bc_head: true

callbacks:
  eval:             # {enabled, freq, n_episodes, seed}
  checkpoint:       # {enabled, freq}
  reward_components:# {enabled, log_freq}
  wandb:            # {enabled, project, run_name, log_freq}

eval:               # optional eval-env sampler overrides
```

### Callbacks

File: [train/callbacks.py](train/callbacks.py)

- `EvalCallback` (SB3): every `freq` env steps, run `n_episodes`
  deterministic eval rollouts in a separate `Monitor`-wrapped env. Saves
  `best_model.zip` whenever `mean_reward` improves and writes
  `eval/evaluations.npz`.
- `CheckpointCallback` (SB3): saves `sac_<step>_steps.zip` every `freq` steps.
- `RewardComponentsCallback` (ours): pulls `info["reward_components"]`
  + `info["term_reason"]`, logs per-term means and per-reason
  cumulative counts to TB every `log_freq` steps.
- `WandbSyncCallback` (ours): re-emits SB3's scalars to W&B if `wandb`
  is enabled.

### Outputs of a training run

`<output_dir>/`
```
tb/SAC_1/                events.out.tfevents.*  # TB logs
best/best_model.zip      # checkpoint with the highest eval mean_reward so far
ckpt/sac_<step>_steps.zip  # periodic checkpoints
eval/evaluations.npz     # timesteps + per-eval rewards/lengths
sac_final.zip            # last model
```

**Use `best/best_model.zip`** for deployment / eval — not `sac_final.zip`.
SAC commonly degrades late if ent_coef collapses; the best checkpoint
captures the high-water mark.

---

## Evaluation

File: [eval/eval_sac.py](eval/eval_sac.py)
CLI: `scripts/eval_sac.py`

Loads a saved SAC `.zip`, builds `FigsDroneEnv` from the same yaml,
runs N deterministic episodes, prints per-episode rows and writes:

- `per_episode.csv`: `episode, ep_reward, ep_length, term_reason, success, start_dist, final_dist, dist_closed`
- `summary.json`: `success_rate, ep_reward_{mean,std}, ep_length_{mean,std}, final_dist_{mean,std}, term_reason_counts`

You can also use SB3's `evaluate_policy` directly, but this script
preserves V-LEAD-specific metrics (distance closed, terminal reason)
that SB3 doesn't track.

---

## Tuning recipes

| Symptom | Most likely cause | First thing to try |
|---|---|---|
| `term_reason_count/ceiling_crash` dominant, linear growth | Random actor exploration crashes upward | Tighten `env.action_low/high` `vz` range; bump `reward.w_altitude` |
| `term_reason_count/bbox_violation` dominant | Goals or starts too close to scene wall | Shrink `sampler.bbox_xyz_*` further from scene bounds; check `sampler.start_xyz_*` margin from walls |
| `reward_components/progress` stays ~0 forever | Actor not learning to approach goal | Verify BC warm-start fired (look for `[sac] warm-started actor latent_pi`); consider lower `goal_radius_max` curriculum |
| `eval/mean_reward` peaks then regresses | `ent_coef` collapsed too early, policy overfits to narrow region | Fix `sac.ent_coef: 0.1` (disable auto) or raise `sac.target_entropy` |
| `train/critic_loss` exploding | Reward scale too big (huge crash penalty + huge progress) | Lower `reward.w_crash` and/or `reward.w_success`, or apply reward clipping |
| Eval reward swings ±50 between adjacent evals | `callbacks.eval.n_episodes` too small | Bump to ≥10 |
| `train/actor_loss` stays positive +30 | Entropy bonus dominating Q-value; policy over-confident | Lower `model.freeze_visual: false` to give actor more capacity, or fix ent_coef as above |
| Training fps < 2 it/s | cuDNN nvrtc fallback + grad-step compute | `pip install nvidia-cuda-nvrtc-cu11` in container; consider `sac.train_freq: 4` for 4× rollout/train ratio |
| OOM during SAC build | Replay buffer too large (`buffer_size * T * 3 * H * W` bytes) | Drop `sac.buffer_size` (5000 ≈ 3 GB at 4×224²) |

---

## Known gotchas

- **PyTorch ≥ 2.6 + nerfstudio**: nerfstudio's gsplat checkpoint
  contains numpy scalars; torch 2.6+ defaults `weights_only=True` and
  blocks the load.
  [`vlead_flight/_torch_compat.py`](../../../../../vlead/vlead_flight/_torch_compat.py)
  fixes this with `add_safe_globals` + `weights_only=False` fallback.
  Already wired into every entry point that touches FiGS.

- **`stable-baselines3 ≥ 2.3` upgrades torch**: requires torch ≥ 2.3
  and silently breaks the image's pinned `torch==2.1.2+cu118` →
  `gsplat==1.5.3+pt21cu118` mismatch → CUDA backend stops loading.
  `vlead/pyproject.toml` pins `stable-baselines3==2.2.1`, and the
  docker compose entrypoint also installs that exact version.

- **`vlead-site-packages` Docker volume**: the V-LEAD compose file used
  to mount a named volume over `/usr/local/lib/python3.10/dist-packages`,
  which captured a stale snapshot of the image's Python packages. After
  any image rebuild the container still saw the old gsplat. The volume
  was removed; if you re-add it, expect very confusing behavior.

- **GPU compatibility**: `figs:latest` is built against CUDA 11.8.
  Blackwell GPUs (sm_120) are **incompatible**. Stick to L40S / A100 /
  earlier Ampere.

- **First-time runs download a 1.26 GB CLIP model** and a ResNet-18 + AlexNet
  bundle. These cache to `~/.cache` (= the `vlead-model-cache` named
  volume in compose) and persist across runs.

- **`info["reward_components"]` keys are dynamic** — if you add a new
  reward term to `GoalReward`, the TB tag `reward_components/<your_key>`
  will appear automatically without touching the callback.

- **CUDA_VISIBLE_DEVICES** is set by the compose `environment:` field
  (defaults to GPU 0). Override per-run from the host:
  `CUDA_VISIBLE_DEVICES=1 docker compose run --rm vlead`.

---

## Quick command reference

All commands run **inside the `vlead` Docker container** unless noted.

### 0. Enter the container

```bash
cd ~/autonomy_projects/V-LEAD
CUDA_VISIBLE_DEVICES=1 docker compose run --rm vlead
# inside container:
cd /workspace/nav_policy
```

### 1. Env smoke (no model, random actions)

```bash
python scripts/smoke_env.py \
    --scene flightroom_ssv_exp/gemsplat/2026-02-28_205058 \
    --steps 20
```

Verifies env reset/step, observation shapes, reward components, and
termination.

### 2. SAC training

```bash
python scripts/train_sac.py \
    --config configs/sac_default.yaml \
    --total-timesteps 100000 \
    --seed 0 \
    --output-dir /project/kothari1/vlead_data/rl_runs/sac_v1_seed0
```

`--seed` overrides yaml `seed:`. Edit `configs/sac_default.yaml` for
everything else (reward weights, action range, warm-start path, ...).

### 3. TensorBoard (on host)

```bash
# host, not container
~/.local/bin/tensorboard \
    --logdir /project/kothari1/vlead_data/rl_runs/sac_v1_seed0/tb \
    --port 6006
```

Or `python -m tensorboard.main --logdir ... --port 6006 --host 0.0.0.0`
from inside the container (compose uses `network_mode: host`).

### 4. Closed-loop evaluation of a saved checkpoint

```bash
python scripts/eval_sac.py \
    --config configs/sac_default.yaml \
    --checkpoint /project/kothari1/vlead_data/rl_runs/sac_v1_seed0/best/best_model.zip \
    --n-episodes 20 \
    --output-dir /project/kothari1/vlead_data/rl_runs/sac_v1_seed0/closed_loop_eval
```

Writes `per_episode.csv` and `summary.json`. **Use `best/best_model.zip`,
not `sac_final.zip`.**

### 5. BC dataset / BC baseline (existing scripts, for warm-start ckpt)

```bash
python scripts/build_dataset.py --config configs/default.yaml
python scripts/train_bc.py     --config configs/default.yaml
```

Output: `<checkpoint_dir>/bc_best.pt` — point `warm_start.bc_checkpoint`
at this in `sac_default.yaml`.

### 6. Quick scalars dump from a TB run

```bash
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
tb = glob.glob('/project/kothari1/vlead_data/rl_runs/sac_v1_seed0/tb/SAC_*')[-1]
ea = EventAccumulator(tb, size_guidance={'scalars': 0}); ea.Reload()
for k in ['rollout/ep_rew_mean', 'eval/mean_reward', 'train/ent_coef',
          'reward_components/progress', 'reward_components/total']:
    if k in ea.Tags()['scalars']:
        evs = ea.Scalars(k)
        print(f'{k}: start={evs[0].value:+.3f} end={evs[-1].value:+.3f}')
for k in ea.Tags()['scalars']:
    if k.startswith('term_reason_count/'):
        print(f'{k}: end={ea.Scalars(k)[-1].value:.0f}')
"
```

### 7. Eval results from `evaluations.npz`

```bash
python -c "
import numpy as np
d = np.load('/project/kothari1/vlead_data/rl_runs/sac_v1_seed0/eval/evaluations.npz')
print('best eval reward:', d['results'].mean(axis=1).max())
print('final eval reward:', d['results'].mean(axis=1)[-1])
"
```
