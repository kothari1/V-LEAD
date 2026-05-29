# nav_policy

RGB-only visuomotor policy for the FiGS quadrotor simulator.  Maps a rolling
buffer of onboard RGB frames to a short-horizon sequence of velocity commands
`[vx, vy, vz, psi_dot]`, tracked by FiGS's `VelocityController`.

**For the full pipeline (data → training → RL), see [`DATA_SETUP.md`](DATA_SETUP.md).**

---

## Architecture

Two policy variants share the same visual encoder backbone:

### FlowMatchingPolicy (primary — CS224R)
```
RGB frame buffer (T=4)
      |
ResNet-18 per frame  (shared weights)
      |
GRU (hidden=256) + LayerNorm
      |
goal embedding [hx, hy, dist]
      |
context vector (288-dim)
      |
ConditionalVectorField  ← OT-CFM training target
  sinusoidal time embedding (64-dim)
  3-layer SiLU MLP (512-wide, skip connection)
      |
ODE integrate (Euler, NFE=10 at deploy)
      |
[vx, vy, vz, psi_dot] × H=10
```

### RGBVelocityPolicy (BC baseline)
Same encoder, MLP head replacing the vector field.

---

## Quick start (Modal — no local GPU needed)

```bash
# 1. One-time setup
pip install modal
modal setup
modal volume create vlead-data
modal volume create vlead-raw-data
modal secret create wandb WANDB_API_KEY=<key>

# 2. Upload raw data and build processed dataset
modal volume put vlead-raw-data data/raw /raw
modal run --detach modal_train.py::build_dataset_remote_local

# 3. Train (wait for build to finish first)
modal run --detach modal_train.py::main_fm              # combined
modal run --detach modal_train.py::main_fm_fulltrajs    # ablation A
modal run --detach modal_train.py::main_fm_shuffled     # ablation B

# 4. Download checkpoint
modal volume get vlead-data checkpoints_flightroom_fm/fm_best.pt \
    data/checkpoints_modal/fm_best.pt
```

See [`DATA_SETUP.md`](DATA_SETUP.md) for the full guide including Google Drive
download, data layout, and Sherlock HPC alternative.

---

## Training configs

| Config | Description |
|---|---|
| `flightroom_fm_modal.yaml` | All 3 train folders + val, 10 epochs |
| `flightroom_fm_modal_fulltrajs.yaml` | Full spawn-to-goal trajs only, 15 epochs |
| `flightroom_fm_modal_shuffled.yaml` | Shuffled 2-sec clips only, 15 epochs |
| `sac_fm.yaml` | SAC fine-tuning warm-started from FM checkpoint |

---

## RL pipeline

1. Train FM policy → `fm_best.pt`
2. `bc_to_rl.py` loads FM visual encoder into `BCEncoderFeatureExtractor`
3. `train_sac.py` runs SAC against `FigsDroneEnv` (Gymnasium wrapper in `vlead/`)

---

## Repository layout

```
nav_policy/
├── configs/                    YAML training configs
├── data/
│   ├── raw/<run>/              Raw SINGER rollouts (.pt + .mp4) — gitignored
│   └── processed_flightroom/   Processed caches + manifest + stats — gitignored
├── modal_train.py              Modal cloud entrypoints
├── scripts/                    CLI entry points (build_dataset, train_bc, train_fm, train_sac)
└── src/nav_policy/
    ├── data/                   build_dataset_flightroom.py, rgb_horizon_dataset.py
    ├── model/                  flow_matching_policy.py, rgb_velocity_policy.py
    ├── train/                  train_fm.py, train_bc.py, metrics.py
    ├── deploy/                 policy_controller.py, frame_buffer.py
    └── rl/                     train_sac.py, bc_to_rl.py, callbacks.py
```
