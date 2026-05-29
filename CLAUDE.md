# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

V-LEAD is a quadcopter autonomy research framework. It integrates a physics-accurate
3D Gaussian Splatting simulator (FiGS) with learned visuomotor policies trained via
imitation learning and online RL.

**Current CS224R work:** flow-matching policy (BC seed) → SAC fine-tuning in FiGS.

---

## Repo Layout

```
V-LEAD/
├── FiGS-Standalone/    # Git submodule: physics simulator + GS renderer + MPC expert
├── SINGER/             # Git submodule: DAgger data pipeline + SINGER policy (SVNet)
├── vlead/              # Local package: vlead_flight — Gymnasium env, pilot, recorder
│   ├── vlead_flight/
│   │   ├── env/        # FigsDroneEnv, EpisodeSampler, RewardConfig, TerminationConfig
│   │   ├── pilot.py    # VLeadPilot (duck-typed controller wrapping trained network)
│   │   ├── recorder.py # RolloutRecorder
│   │   └── observation.py, eval.py, deploy.py
│   └── tests/test_pilot_smoke.py
├── nav_policy/         # Local package: RGB-only policy (ResNet-18 + GRU + MLP)
│   ├── src/nav_policy/
│   │   ├── data/       # build_dataset, rgb_horizon_dataset, normalization
│   │   ├── model/      # RGBVelocityPolicy, losses
│   │   ├── train/      # train_bc, metrics
│   │   ├── deploy/     # PolicyController, frame_buffer
│   │   └── rl/         # train_sac.py, bc_to_rl.py, BCEncoderFeatureExtractor, callbacks
│   ├── scripts/        # CLI entry points: build_dataset, train_bc, train_sac, eval_*
│   └── configs/        # default.yaml, SAC YAML configs
├── AGENT_CONTEXT.md    # Deep architecture reference (read before touching FiGS/SINGER internals)
├── V-LEAD_instructions.md  # Operator guide: containers, data gen pipeline, gotchas
└── cs224r_tasks.md     # Active task list for CS224R project
```

---

## Environment: Docker-Only

All pipeline scripts run inside Docker containers. **Never run FiGS/SINGER scripts on the host.**

```bash
# FiGS container (physics sim + renderer)
cd FiGS-Standalone && docker compose -f docker-compose.base.yml run --rm figs

# SINGER container (primary dev env — includes figs, gemsplat, sousvide editable installs)
cd SINGER && docker compose run --rm singer

# V-LEAD master container (all 4 packages)
cd V-LEAD && docker compose run --rm vlead

# nav_policy container
cd nav_policy && docker compose run --rm nav_policy
```

Use `python3` (not `python`) inside containers.

GPU note: Only L40S GPUs (sm_89) work with the current `figs:latest` image.
Blackwell GPUs (sm_120) are incompatible. Use `CUDA_VISIBLE_DEVICES=1`.

---

## Key Commands

### Tests (smoke tests — no trained network needed)
```bash
# Inside vlead container or with vlead_flight installed:
python /workspace/vlead/tests/test_pilot_smoke.py
```

### nav_policy: Build dataset → Train BC → Train SAC
```bash
# Inside nav_policy container:
python scripts/build_dataset.py --config configs/default.yaml
python scripts/train_bc.py --config configs/default.yaml
python scripts/train_sac.py --config configs/sac_default.yaml [--seed N] [--output-dir PATH]
```

### SINGER DAgger pipeline (inside SINGER container)
```bash
python3 notebooks/ssv_multi3dgs_campaign.py generate-rollouts \
    --config-file configs/experiment/smoke_test.yml
python3 notebooks/ssv_multi3dgs_campaign.py generate-observations \
    --config-file configs/experiment/smoke_test.yml
python3 notebooks/ssv_multi3dgs_campaign.py train-history \
    --config-file configs/experiment/smoke_test.yml
python3 notebooks/ssv_multi3dgs_campaign.py train-command \
    --config-file configs/experiment/smoke_test.yml
```

### V-LEAD data generation (inside SINGER container)
```bash
# Dry run (~2 min):
python3 notebooks/generate_training_data.py \
    --config-file configs/experiment/vlead_dryrun.yml --validation-mode

# Full run, parallelized by target object across containers:
python3 notebooks/generate_training_data.py \
    --config-file configs/experiment/vlead_flightroom.yml --validation-mode \
    --query-indices 0,1
```

---

## Architecture: How the Pieces Connect

### Controller duck-typing contract
Neither `VLeadPilot` nor SINGER's `Pilot` inherits from `BaseController`.
`Simulator.simulate()` is polymorphic: any object with
`.control(tcr, xcr, upr, obj, icr, zcr)` and `.hz` attribute works.
**Match this exact signature when adding controllers.**

### State & control conventions
- State x (10-dim): `[px, py, pz, vx, vy, vz, qx, qy, qz, qw]`
  - Quaternion: Hamilton convention, scalar-last
  - World z is **down** (NED-like); `vz < 0` = upward motion
- Control u (4-dim): `[thrust, ωx, ωy, ωz]`
  - Thrust `uf ∈ [-1, 0]`; hover ≈ -0.41
  - Body rates ±5 rad/s

### nav_policy network (RGBVelocityPolicy)
```
RGB frame buffer (T=4) → ResNet-18 per frame → GRU → MLP → [vx,vy,vz,psi_dot] × H=10
```
Only the first step of the H=10 horizon is executed each control cycle.
Output fed to FiGS `VelocityController` (P-cascade inner loop).

### RL pipeline (CS224R)
1. Train BC policy (`train_bc.py`) → checkpoint: `{"model": state_dict, "config": ..., "stats": ...}`
2. Warm-start SAC from BC: `BCEncoderFeatureExtractor` loads BC visual encoder; optionally `load_bc_into_sac_actor` copies MLP head to SAC actor.
3. `train_sac.py` runs stable-baselines3 SAC against `FigsDroneEnv` (Gymnasium wrapper).
4. `FigsDroneEnv` is in `vlead/vlead_flight/env/`; reward in `reward.py`, termination in `termination.py`.

### Checkpoint format
BC checkpoints: `torch.save({"model": state_dict, "config": cfg, "epoch": n, "stats": ...}, path)`.
Do **not** use raw `state_dict` files or pickled `nn.Module`.

---

## Critical Gotchas

1. **ACADOS solver lifecycle:** `VehicleRateMPC.__init__()` compiles C code into a temp dir. Each container call is isolated (race-safe). `del ctl` after use to avoid re-init errors.

2. **Perception config must be set before data gen:**
   `FiGS-Standalone/configs/perception/perception_mode.yml` → `visual_mode: semantic_depth`, `perception_type: similarity`. Never use `perception_type: clipseg` (slow, buggy).

3. **Module aliasing for legacy checkpoints:** Old `.pth` files reference `controller.policies.*`. `generate_networks.py::_alias_controller_to_control()` patches `sys.modules` at load time.

4. **Coordinate frames:** World (w), GSplat/nerfstudio (g), Camera (c), Body (b). `T_w2g = diag([1,-1,-1,1])` flips Y and Z. All control happens in World frame.

5. **`site-packages` Docker volume:** If `figs:latest` image changes, run `docker compose down -v` in SINGER to clear cached packages before re-running the one-time pip install.

6. **Storage:** `/home/kothari1` is near capacity. All large outputs go to `/data/kothari1/singer_figs_data/` (legacy) or `/project/kothari1/vlead_data/` (V-LEAD outputs). Symlinks handle this automatically — do not delete them.

7. **`running_min/max` bootstrap (gemsplat):** Pre-set `running_min=-1.0, running_max=1.0` before the first render call or you get NaN in the scaled semantic output.

8. **Training stages are order-dependent (SINGER DAgger):** rollouts → observations → train-history → train-command. Running out of order produces a misconfigured policy.

---

## Data Format Reference

### Trajectory dict (`.pt` files)
| Field | Shape | Description |
|---|---|---|
| `Tro` | `(N+1,)` | Timestamps |
| `Xro` | `(10, N)` | State: `[px,py,pz, vx,vy,vz, qx,qy,qz,qw]` |
| `Uro` | `(4, N)` | Controls: `[thrust, ωx, ωy, ωz]` |
| `goal_xy` | `(2,)` | XY centroid of target object (world frame) |
| `heading_vec` | `(2, N)` | Unit vector drone→goal per timestep |
| `dist` | `(N,)` | XY scalar distance to goal |

### Output directories
- SINGER rollouts: `/data/kothari1/singer_figs_data/rollouts_singer/<cohort>/rollout_data/<scene>/`
- V-LEAD rollouts: `/project/kothari1/vlead_data/rollouts_vlead/<cohort>/rollout_data/<YYYY-MM-DD_HHMMSS>/<scene>/`
- nav_policy data: `nav_policy/data/raw/<run>/` (SINGER validation-rollout format)
