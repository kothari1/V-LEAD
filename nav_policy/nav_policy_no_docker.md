# nav_policy (no-Docker / local venv)

> This is a Docker-free version of [README.md](README.md). Instead of the FiGS
> Docker container and Modal cloud training, every step runs inside a local
> Python 3.10 virtualenv. The pipeline scripts and configs are identical — only
> the *environment* changes (no `docker compose run`, no `modal run`).
>
> See the repo-root `yash_readme.md` for the full machine-specific setup notes.

Goal-conditioned visual navigation for the FiGS quadrotor simulator. Maps a short
history of onboard RGB frames (+ optional DA2-S depth) and a goal vector to a
receding-horizon sequence of velocity commands `[vx, vy, vz, psi_dot]`, tracked
by FiGS's `VelocityController`.

**Training stack:** behavior cloning (BC) → DAgger → optional **RL fine-tuning**
(PPO default, SAC optional). **No relightable 3DGS** — robustness via 2D
augmentations and observation-latency simulation.

## Architecture (A2 primary)

```text
FiGS RGB (+ live DA2-S depth at deploy)
      |
      v
rolling frame buffer (T=4, optional latency)
      |
      +-- ResNet-18 per-frame encoder --+
      |                                  |
      +-- DA2-S depth CNN encoder -------+--> cross-attention fusion --> GRU
      |
      v
goal vector [hx, hy, d_norm] --> embedding
      |
      v
MLP head -> [vx, vy, vz, psi_dot] x H=10  (first step executed @ 20 Hz)
      |
      v
FiGS VelocityController -> body rates -> ACADOS integrator
```

## Repository layout

```text
nav_policy/
|-- configs/                 BC, DAgger, eval, RL yaml files
|-- data/
|   |-- raw/<run>/           SINGER-format rollouts
|   |-- processed/           cache blobs + manifest (gitignored)
|   |-- weights/             depth_anything_v2_vits.pth
|   |-- checkpoints_*/       BC / RL outputs (gitignored)
|-- src/nav_policy/
|   |-- data/                dataset, augmentations, normalization
|   |-- model/               RGB / RGB+DA2 policies, depth estimator
|   |-- train/               train_bc.py
|   |-- rl/                  PPO / SAC fine-tuning
|   |-- deploy/              policy_controller, frame_buffer
|   |-- dagger/              DAgger rollouts + MPC oracle
|   |-- evaluate/            offline + closed-loop eval
|   |-- vendor/              vendored Depth Anything V2 (no PyPI dep)
|-- scripts/                 CLI entry points
\-- modal_train.py           cloud BC on Modal (NOT used in this no-Docker flow)
```

## 0. Environment setup (one-time, replaces Docker)

Requires Python 3.10. On Ubuntu 22.04/24.04, install it via the deadsnakes PPA
first (see `yash_readme.md`), then create the venv from the **repo root**:

```bash
cd /path/to/V-LEAD

python3.10 -m venv .venv
source .venv/bin/activate

# Upgrade pip first — the bundled pip is too old and rejects the torch wheels.
pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.8 build)
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# nav_policy + RL extras (gymnasium, stable-baselines3==2.2.1, tensorboard)
pip install -e 'nav_policy[rl]'

# FiGS (editable) — needed for closed-loop eval, DAgger, and RL
pip install -e FiGS-Standalone/

# Full 3DGS rendering stack (nerfstudio, gsplat, timm, einops, pandas, matplotlib).
# NOTE: pins numpy==1.26.4, so it DOWNGRADES numpy 2.x -> 1.x (required by gsplat).
# gsplat is pinned to 1.4.0 to match nerfstudio 1.1.5's hard pin; opencv-python is
# pinned to 4.10.x (numpy-1.x compatible) so grad-cam doesn't pull in opencv 4.13 (numpy>=2).
pip install -r requirements.txt
```

> **gsplat / CUDA toolkit:** gsplat JIT-compiles its CUDA kernels on the first
> render (not at `pip install`), so a CUDA toolkit with `nvcc` must be on `PATH`
> for closed-loop eval and RL. If `which nvcc` is empty, export it before running
> (this machine has CUDA 12.9 under `/usr/local/cuda`):
>
> ```bash
> export CUDA_HOME=/usr/local/cuda
> export PATH=$CUDA_HOME/bin:$PATH
> ```
>
> Add these two lines to `.venv/bin/activate` (or your shell profile) to make it
> stick. `import gsplat` works without `nvcc`; only the first kernel compile needs it.

> **ROS users:** if `source /opt/ros/.../setup.bash` runs in your `~/.bashrc`, it
> sets `PYTHONPATH` to ROS's site-packages, which leaks into the venv and breaks
> imports. Run `unset PYTHONPATH` before activating, or don't auto-source ROS.

For **every** command below: activate the venv first and run from inside
`nav_policy/`:

```bash
cd /path/to/V-LEAD/nav_policy
source ../.venv/bin/activate
```

## Quick smoke test (verify the environment)

Before running the pipeline, confirm the venv is wired up correctly. This needs
**no data, checkpoints, or 3DGS scenes** — it only checks imports, CUDA, the
augmentations module, a model forward pass, and the FiGS simulator:

```bash
python scripts/smoke_check.py
```

Expected: a PASS/FAIL/SKIP table ending in `ALL CHECKS PASSED` (exit 0). The
`nerfstudio` / `gsplat` rows show **SKIP** until you install the root
`requirements.txt` — that's fine for BC / offline work, but closed-loop eval and
RL rollouts (sections 3–4) need them.

```text
  [PASS] python 3.10                        3.10.x
  [PASS] CUDA available                     12.8 / NVIDIA GeForce RTX 4090
  [PASS] data.augmentations                 window+rgb+depth aug OK
  [PASS] model forward (RGBVelocityPolicy)  rgb[2,4,3,224,224]+goal[2,2] -> (2, 10, 4)
  [PASS] figs (simulator)                   figs + VelocityController
  [SKIP] nerfstudio (3DGS)                  install root requirements.txt for RL / closed-loop
  ...
```

> If you see a `SKIP` on **no ROS PYTHONPATH leak**, run `unset PYTHONPATH` and
> re-activate the venv (see the ROS note above) before training.

Once the smoke test is green, run the end-to-end RL smoke test from section 4
(`python scripts/train_rl.py --config configs/train_rl_smoke.yaml --save-videos`)
to exercise the full rollout + training loop.

## Full pipeline

### 1. Dataset + depth

```bash
python scripts/build_dataset_flightroom.py --config configs/flightroom.yaml
python scripts/precompute_da2_depth.py --processed-root data/processed_flightroom
```

Place DA2 weights at `data/weights/depth_anything_v2_vits.pth`.

### 2. BC training (local)

The Modal cloud path is skipped in this no-Docker flow — train locally instead:

```bash
python scripts/train_bc.py --config configs/arch_rgb_da2_crossattn.yaml
```

Best weights land at `data/checkpoints_a2_da2_crossattn/bc_best.pt`.

### 3. DAgger + eval

```bash
python scripts/run_dagger.py --config configs/dagger_round1_mpc.yaml

# Fine-tune locally on the new DAgger caches:
python scripts/train_bc.py --config configs/flightroom.yaml \
  --run-tag dagger_r1_mpc \
  --resume-from data/checkpoints_a2_da2_crossattn/bc_best.pt \
  --checkpoint-dir data/checkpoints_dagger_r1_mpc

python scripts/eval_offline.py \
  --config configs/eval_offline_flightroom.yaml \
  --checkpoint data/checkpoints_a2_da2_crossattn/bc_best.pt \
  --output-dir data/eval/flightroom_bc_offline

python scripts/eval_in_figs.py --config configs/eval_closed_loop_flightroom_suite.yaml
python scripts/eval_in_figs.py --config configs/eval_closed_loop_backroom_val.yaml
python scripts/eval_in_figs.py --config configs/eval_closed_loop_ood_suite.yaml
```

### 4. RL fine-tuning (requires 3DGS scenes)

Warm-start from a BC or DAgger checkpoint. **PPO is the default**; switch
algorithm in the yaml.

**Smoke test** (2 rollouts, 1 iteration, videos + TensorBoard):

```bash
python scripts/train_rl.py --config configs/train_rl_smoke.yaml --save-videos
```

Equivalently, the module entry point used during local bring-up:

```bash
python -m nav_policy.rl.train_rl --config configs/train_rl_local.yaml \
  --n-iterations 1 --rollouts-per-iteration 2
```

**Full training:**

```bash
# PPO (default)
python scripts/train_rl.py --config configs/train_rl_flightroom.yaml

# SAC
python scripts/train_rl.py --config configs/train_rl_flightroom_sac.yaml
```

**Logs:** CSV files are written alongside checkpoints (`*_episodes.csv`, `*_log.csv`, `*_summary.json`).

**Videos** (optional): pass `--save-videos` or set `save_videos: true` in yaml.
Files go to `video_dir` (default `data/checkpoints_<tag>/videos/<run_tag>/`).

Key config fields (`configs/train_rl_flightroom.yaml`):

```yaml
rl:
  algorithm: ppo          # or sac
  checkpoint_dir: data/checkpoints_rl_a2
  n_iterations: 20
  rollouts_per_iteration: 4
  rewards:
    progress_weight: 1.0
    collision_penalty: -10.0
    success_bonus: 5.0
```

Outputs: `data/checkpoints_rl_a2/rl_ppo_a2_best.pt` (compatible with
`eval_in_figs.py` via `--checkpoint`).

Resume RL training:

```bash
python scripts/train_rl.py --config configs/train_rl_flightroom.yaml \
  --resume-from data/checkpoints_rl_a2/rl_ppo_a2_latest.pt
```

Monitor with TensorBoard:

```bash
tensorboard --logdir data/checkpoints_rl_a2/tensorboard
```

## Augmentations (training)

Configured under `train:` in yaml (`configs/flightroom.yaml`):

| Augmentation | Config keys |
|---|---|
| Color jitter | `color_jitter: true` |
| Gaussian blur | `photometric_aug.blur_prob`, `blur_sigma_range` |
| Brightness / gamma | `photometric_aug.brightness_*`, `gamma_*` |
| Observation latency | `observation_latency` (train + stress eval) |

Implementation: `src/nav_policy/data/augmentations.py`,
`src/nav_policy/data/rgb_horizon_dataset.py`.

## RL design (no relightable renderer)

| Component | Location |
|---|---|
| Gaussian actor + critic | `src/nav_policy/rl/stochastic_policy.py` |
| FiGS rollout collector | `src/nav_policy/rl/rollout.py` |
| Rewards (progress, heading, bbox, success) | `src/nav_policy/rl/rewards.py` |
| PPO | `src/nav_policy/rl/ppo.py` |
| SAC | `src/nav_policy/rl/sac.py` |
| Training loop | `src/nav_policy/rl/train_rl.py` |

RL rollouts are restricted to **flightroom training scenes** (same as DAgger).
Rewards use goal progress, velocity-heading alignment, expert-bbox violation
penalty, and sparse terminal success — **not** relighting.

## Data assumptions

Each `data/raw/<run>/` directory is SINGER validation-rollout format:

- `trajectories_val{NNNNN}.pt` — state/control logs @ 20 Hz
- `video_val_rollout_images_rgb{NNNNN}.mp4` — 20 fps RGB
- `imgdata_val{NNNNN}.pt` — sub-trajectory frame ranges

Goal supervision: expert sub-trajectory endpoint `Xro[0:2,-1]`.

**Splits (flightroom branch):**

| Split | Scenes |
|---|---|
| Train | flightroom 064652, 071718, 071353 |
| Val | flightroom 071733 + backroom |
| OOD test | packardpark (closed-loop only) |

## Dependencies

- Python 3.10 virtualenv (see section 0) — **no Docker, no Modal**
- PyTorch 2.7.0+cu128 + torchvision 0.22.0+cu128
- FiGS / ACADOS / 3DGS (closed-loop, DAgger, RL) — `pip install -e FiGS-Standalone/`
- nerfstudio + gsplat (RL / closed-loop rendering) — via root `requirements.txt`
- DA2-S vendored under `src/nav_policy/vendor/` (no `pip install depth-anything-v2`)

See `docs/CROSS_SCENE_DA2_RL_PLAN.md` for the full experiment plan.
