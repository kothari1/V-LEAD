# LEAD Framework: User Instructions

> **Last updated:** 2026-03-02
> **Audience:** Developers and researchers using the LEAD repo for quadcopter autonomy research.
> For AI agent context (architecture, gotchas, file maps), see `AGENT_CONTEXT.md`.
> For step-by-step smoke tests, see `SMOKE_TESTS.md`.

---

## What is LEAD?

LEAD integrates two systems:

- **FiGS-Standalone** (*Flying in Gaussian Splats*): Physics-accurate quadcopter simulator flying through 3D Gaussian Splat environments. Provides dynamics, MPC expert controller, RRT* trajectory planning, and rendering.

- **SINGER** (*Scene Understanding via Synthesized Visual Inertial Data from Experts*): Learned autonomy layer on top of FiGS. Trains vision-language navigation policies via DAgger-style imitation learning from expert MPC demonstrations.

- **Semantic_HSM**: Semantic Hierarchical State Machine — trained on SINGER validation rollout data to learn high-level scene understanding.

The full pipeline: capture real environment → train 3DGS → generate expert rollouts → train neural policy → deploy in simulation.

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Host | `coruscant`, user `kothari1` |
| GPU | Available via Docker (no `runtime: nvidia`; use `deploy.resources.reservations.devices`) |
| Docker | No sudo needed — `kothari1` is in `docker` group |
| Python | 3.10 (inside containers only — do not run pipeline scripts on host) |
| Data drive | `/data/kothari1/singer_figs_data` — **all large outputs must go here** |
| Home disk | `/home/kothari1` is near capacity — never write large files there |

### Critical Directory Symlinks
These symlinks redirect large data to the data drive automatically:
- `FiGS-Standalone/3dgs` → `/data/kothari1/singer_figs_data/3dgs`
- `SINGER/cohorts` → `/data/kothari1/singer_figs_data/rollouts_singer`

**Do not delete these symlinks.** They work inside Docker because `DATA_PATH` is mounted at the same absolute path.

---

## Environment Setup

### 1. Build the FiGS Docker Image (one-time, ~20 min)
Only needed once. Already built on coruscant as `figs:latest`.
```bash
cd /home/kothari1/autonomy_projects/LEAD/FiGS-Standalone
git submodule update --init gemsplat
CUDA_ARCHITECTURES=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.') \
  docker compose build
```

### 2. Install Python Dependencies (one-time per Docker volume)
The `site-packages` Docker volume persists across runs. Run this once:
```bash
cd /home/kothari1/autonomy_projects/LEAD/SINGER
docker compose run --rm singer bash -lc "
  python3 -m pip install typer 'transformers==4.40.0' 'huggingface_hub==0.23.0' shapely scikit-image imageio
"
```
> **Version pinning is critical:** `transformers==4.40.0` is the highest version compatible with PyTorch 2.1.2+cu118 in the container.

### 3. Verify `.env` Files
Both repos have `.env` files that must point to the data drive:

**`FiGS-Standalone/.env`:**
```
DATA_PATH=/data/kothari1/singer_figs_data
```

**`SINGER/.env`:**
```
DATA_PATH=/data/kothari1/singer_figs_data
FIGS_PATH=../FiGS-Standalone
```

---

## Container Entry Points

### FiGS Container
```bash
cd /home/kothari1/autonomy_projects/LEAD/FiGS-Standalone
docker compose -f docker-compose.base.yml run --rm figs
# Working directory inside: /workspace/FiGS-Standalone
```

### SINGER Container (primary development environment)
```bash
cd /home/kothari1/autonomy_projects/LEAD/SINGER
docker compose run --rm singer
# Working directory inside: /workspace/SINGER
# Editable installs: figs, gemsplat, sousvide
# Bind mounts: FiGS-Standalone, Semantic_HSM
```

---

## Full Pipeline: Training a Neural Pilot

All steps run inside the SINGER container at `/workspace/SINGER`. Use `smoke_test.yml` for testing (33 batches, 5 epochs) or `ssv_multi3dgs.yml` for production.

### Step 1: Generate Training Rollouts
MPC expert flies RRT* paths toward semantic targets. Applies domain randomization (mass, force perturbations).
```bash
python3 notebooks/ssv_multi3dgs_campaign.py generate-rollouts \
    --config-file configs/experiment/smoke_test.yml
```
**Output:** `trajectories{id:05d}.pt`, `imgdata{id:05d}.pt`, `video{id:05d}.mp4` in `.../rollout_data/<scene>/`

### Step 2: Generate Validation Rollouts
Same as Step 1 but without domain randomization. Also renders multi-channel video (RGB + depth + semantic) needed for `Semantic_HSM` training.
```bash
python3 notebooks/ssv_multi3dgs_campaign.py generate-rollouts \
    --config-file configs/experiment/smoke_test.yml --validation-mode
```
**Output:** `trajectories_val{id:05d}.pt`, `video_val_rollout_images_{semantic,rgb,depth}{id:05d}.mp4` in `.../rollout_data/<scene>/`

> **Perception mode required:** `FiGS-Standalone/configs/perception/perception_mode.yml` must have:
> ```yaml
> visual_mode: "semantic_depth"
> perception_type: "similarity"
> ```

### Step 3: Generate Observations
Replays expert trajectories through the Pilot's observation pipeline. Produces `(state, image, expert_action)` training tuples.
```bash
python3 notebooks/ssv_multi3dgs_campaign.py generate-observations \
    --config-file configs/experiment/smoke_test.yml
```
**Output:** `observations_{id:05d}.pt` in `.../observation_data/<pilot_name>/`

### Step 4: Train History Encoder
Trains `HistoryEncoder` (Parameter module) to predict drone model parameters from delta-state history.
```bash
python3 notebooks/ssv_multi3dgs_campaign.py train-history \
    --config-file configs/experiment/smoke_test.yml
```
**Output:** History encoder checkpoint in `.../roster/<pilot_name>/`

### Step 5: Train Commander Module
Locks the trained HistoryEncoder. Trains `VisionMLP` + `CommanderSV` jointly.
```bash
python3 notebooks/ssv_multi3dgs_campaign.py train-command \
    --config-file configs/experiment/smoke_test.yml
```
**Output:** Full policy checkpoint in `.../roster/<pilot_name>/`

### Step 6: Evaluate
Deploy trained pilot in the gsplat scene and evaluate performance.
```bash
python3 notebooks/ssv_multi3dgs_campaign.py simulate \
    --config-file configs/experiment/smoke_test.yml
```

---

## Training Semantic_HSM

`Semantic_HSM/scripts/train_03.py` trains `VisualNavPolicy_Sequence` on the **validation rollouts** from Step 2. It has no FiGS/SINGER dependencies — runs directly on the host using the `lead_ml` conda env (no Docker needed).

### Python Environment

The `lead_ml` env lives on the data drive (keeps `/home` free). Activate it with:
```bash
conda activate /data/kothari1/singer_figs_data/conda_envs/lead_ml
```
Packages: `torch 2.6.0+cu124`, `torchvision`, `tqdm`, `matplotlib`, `scipy`, `opencv-python`.

To add new packages (always route caches to data drive):
```bash
TMPDIR=/data/kothari1/singer_figs_data/.pip_tmp \
  pip install --cache-dir /data/kothari1/singer_figs_data/.pip_cache <package>
```

### Running Training

**Before running, verify CONFIG settings in `train_03.py`:**
```python
CONFIG = {
    'data_root': '/data/kothari1/singer_figs_data/rollouts_singer/smoke_test/rollout_data/flightroom_ssv_exp',
    'start_idx': 0,
    'end_idx': 32,   # inclusive; covers indices 00000-00032 (33 validation batches)
    ...
}
```

**Run directly on the host** (interactive matplotlib loss plot included):
```bash
conda activate /data/kothari1/singer_figs_data/conda_envs/lead_ml
cd ~/autonomy_projects/LEAD/Semantic_HSM/scripts
TORCH_HOME=/data/kothari1/singer_figs_data/.torch_hub \
CUDA_VISIBLE_DEVICES=1 \
  python3 train_03.py
```

> **GPU note:** GPU 0 is typically occupied by other users. `CUDA_VISIBLE_DEVICES=1` targets GPU 1. Verify with `nvidia-smi` before running.

> **`TORCH_HOME`:** Redirects MobileNetV3 weight downloads to the data drive. Required on first run; cached after that.

**Stopping training early** (all options finish the current epoch then save the model):
| Method | How |
|--------|-----|
| Ctrl+C in the terminal | Most convenient — sends SIGINT, handled gracefully |
| `q` or `Esc` in the plot window | Requires the matplotlib window to have keyboard focus |
| Sentinel file | `touch ~/autonomy_projects/LEAD/Semantic_HSM/scripts/STOP_TRAINING` from any terminal |

**Checkpoints** → `Semantic_HSM/scripts/checkpoints_v3/` (every 5 epochs + `final_model_seq.pth` on exit).

**Expected input files per index `id`:**
- `trajectories_val{id:05d}.pt` — trajectory state/action sequence
- `video_val_rollout_images_rgb{id:05d}.mp4` — RGB video
- `video_val_rollout_images_semantic{id:05d}.mp4` — semantic heatmap video
- `video_val_rollout_images_depth{id:05d}.mp4` — depth map video

---

## V2C State-Machine Simulation (`Semantic_HSM/sim/`)

An alternative to `SINGER/notebooks/simulate_v2c_adi.py` that warm-starts the V2C
policy with a rule-based state machine before handing control to the neural policy.
This prevents the immediate `COLLISION_BOUNDS` failure that occurs when the policy
fires from a cold start at a random position.

**Run inside the SINGER container:**
```bash
python3 /workspace/Semantic_HSM/sim/simulate.py \
    --model /workspace/Semantic_HSM/scripts/trained_models/t5_altframes_seq_final_model.pth \
    --object "green clock"
```

**State machine sequence:**
```
STABILIZE → SCAN → CENTER → NAVIGATE → SUCCESS / OOB / TIMEOUT
```

| Phase | What it does |
|-------|--------------|
| STABILIZE | P-control altitude to -1.0 m NED; holds x/y |
| SCAN | Rotates 360° at 0.05 rad/step; records semantic peak yaw; fills history buffer |
| CENTER | P-control yaw toward best semantic hit; waits until object centred in FOV |
| NAVIGATE | Runs `VisualNavPolicy_Sequence` with warm history buffer |

**CLI options:**
```
--scene     STR    3DGS scene name (default: flightroom_ssv_exp)
--object    STR    Language query  (default: green clock)
--model     STR    Path to .pth checkpoint [REQUIRED]
--renderer  STR    gemsplat | splatfacto   (default: gemsplat)
--timeout   FLOAT  Hard time limit in s    (default: 180.0)
--max-speed FLOAT  Max drone speed m/s     (default: 2.0)
```

Output videos → `Semantic_HSM/sim/outputs/` (small files, local storage OK).
For full documentation see `Semantic_HSM/sim/README.md`.

---

## Configuration Reference

### Experiment Config (`configs/experiment/*.yml`)
Controls the top-level pipeline: which scenes, which pilot, how many epochs.
```yaml
cohort: "smoke_test"       # Output directory name under cohorts/
method: "rrt"              # Trajectory generation method (configs/method/rrt.json)
Nep_his: 5                 # History encoder training epochs
Nep_com: 5                 # Commander training epochs
flights:
  - ["flightroom_ssv_exp", "flightroom_ssv_exp"]   # [scene_3dgs, scene_config]
roster:
  - "InstinctJester"       # Pilot architecture (configs/pilots/InstinctJester.json)
```

### Perception Config (`FiGS-Standalone/configs/perception/perception_mode.yml`)
Controls what the simulator renders at each timestep.
```yaml
visual_mode: "semantic_depth"   # "rgb" for fast single-channel, "semantic_depth" for all 3
perception_type: "similarity"   # Always use "similarity" — "clipseg" is slow and buggy
extra_channels: []
```

### Method Config (`configs/method/rrt.json`)
Controls trajectory generation, domain randomization, and frame sets.
- `trajectory_set.initial` — perturbation bounds for initial drone state (defines n_domain_rand_perturbations)
- `frame_set` — list of drone frame configs for domain randomization
- `sample_set` — rollout simulation parameters (duration, rate, noise profile)

### Scene Config (`configs/scenes/<scene>.yml`)
Per-scene parameters: semantic target queries, flight altitude/radius, RRT* obstacle params.

### Pilot Config (`configs/pilots/InstinctJester.json`)
Neural network architecture: HistoryEncoder hidden dims, VisionMLP backbone (SqueezeNet1_1), CommanderSV hidden dims, state/observation indices.

---

## Available 3DGS Models

For the `flightroom_ssv_exp` scene (used by all smoke tests):

| Model | Path | Checkpoint |
|-------|------|------------|
| gemsplat (preferred, user-trained) | `3dgs/workspace/outputs/flightroom_ssv_exp/gemsplat/2026-02-28_205058/` | step-000029999.ckpt |
| gemsplat (older) | `trained_gsplats/flightroom_ssv_exp/gemsplat/2026-02-03_115017/` | step-000028000.ckpt |
| splatfacto | `3dgs/workspace/outputs/flightroom/splatfacto/2024-07-12_145513/` | step-000029999.ckpt |

Paths are relative to `/data/kothari1/singer_figs_data/`. The Simulator searches `FiGS-Standalone/3dgs/workspace/outputs/<scene>` first, then falls back to `$DATA_PATH/trained_gsplats/<scene>`.

---

## Output File Nomenclature

All rollout data goes to `.../rollouts_singer/<cohort>/rollout_data/<scene>/`.

Each file uses a zero-padded 5-digit batch index. The index maps to one rollout: one RRT branch toward one target object, with one domain-randomized drone configuration.

**Training files** (330 files for smoke_test: 3 targets × 11 branches × 10 perturbations):

| File | Content |
|------|---------|
| `trajectories{id:05d}.pt` | Expert MPC trajectory: state `x` (10-dim), control `u` (4-dim), timestamps `t`, metadata |
| `imgdata{id:05d}.pt` | Per-frame semantic similarity maps (tensors) from gemsplat rendering |
| `video{id:05d}.mp4` | Semantic heatmap video — CLIP similarity scores as colored overlay |

**Validation files** (33 files for smoke_test: 3 targets × 11 branches):

| File | Content |
|------|---------|
| `trajectories_val{id:05d}.pt` | Validation trajectory (same format, no domain randomization) |
| `imgdata_val{id:05d}.pt` | Validation image tensors |
| `video_val_rollout_images_semantic{id:05d}.mp4` | Semantic heatmap — target object highlighted |
| `video_val_rollout_images_rgb{id:05d}.mp4` | Raw RGB from drone forward camera |
| `video_val_rollout_images_depth{id:05d}.mp4` | Depth map (JET colormap: near=blue, far=red) |

---

## Common Issues

### "python not found"
Use `python3` explicitly inside SINGER container. Never use bare `python`.

### "pip not in PATH"
Use `python3 -m pip` instead of `pip`.

### CLIP / AlexNet downloads on first run
CLIP (1.26 GB) and AlexNet (233 MB) download to `/root/.cache/` on first run. Cached in the `model-cache` Docker named volume — subsequent runs are fast.

### `transformers` version conflict
Always pin: `transformers==4.40.0`. Higher versions require PyTorch ≥2.2 but the container has 2.1.2.

### `site-packages` volume stale after image rebuild
If the base `figs:latest` image changes, clear the cached packages: `docker compose down -v` in the SINGER directory, then re-run the one-time dependency setup.

### Storage full on `/home/kothari1`
All large outputs must go to `/data/kothari1/singer_figs_data/`. The symlinks (`SINGER/cohorts`, `FiGS-Standalone/3dgs`) ensure this automatically. Never run pipeline scripts that write to `/home/kothari1`.

### Old `video_val{id}.mp4` files in output dir
These are artifacts from earlier pipeline iterations with different naming conventions. Safe to ignore or delete. The correct validation videos are the `video_val_rollout_images_{channel}{id}.mp4` set.

---

## Quick Reference Commands

```bash
# Enter FiGS container
cd FiGS-Standalone && docker compose -f docker-compose.base.yml run --rm figs

# Enter SINGER container
cd SINGER && docker compose run --rm singer

# Run full smoke test pipeline (inside SINGER container)
python3 notebooks/ssv_multi3dgs_campaign.py generate-rollouts --config-file configs/experiment/smoke_test.yml
python3 notebooks/ssv_multi3dgs_campaign.py generate-rollouts --config-file configs/experiment/smoke_test.yml --validation-mode
python3 notebooks/ssv_multi3dgs_campaign.py generate-observations --config-file configs/experiment/smoke_test.yml
python3 notebooks/ssv_multi3dgs_campaign.py train-history --config-file configs/experiment/smoke_test.yml
python3 notebooks/ssv_multi3dgs_campaign.py train-command --config-file configs/experiment/smoke_test.yml

# Check output files
ls /data/kothari1/singer_figs_data/rollouts_singer/smoke_test/rollout_data/flightroom_ssv_exp/

# Train Semantic_HSM (inside SINGER container, after validation rollouts)
python3 /workspace/Semantic_HSM/scripts/train_03.py
```
