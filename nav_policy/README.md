# nav_policy

RGB-only visual navigation policy for the FiGS quadrotor simulator. Maps a
short history of onboard RGB frames to a short-horizon sequence of velocity
commands `[vx, vy, vz, psi_dot]`, which are tracked by FiGS's existing
`VelocityController` (see `FiGS-Standalone/src/figs/control/velocity_controller.py`).

## Architecture

```text
FiGS RGB camera
      |
      v
rolling frame buffer (T=4)
      |
      v
ResNet-18 visual encoder (shared per frame)
      |
      v
GRU temporal module
      |
      v
MLP command head
      |
      v
[vx, vy, vz, psi_dot] x H=10  (only first step is executed each control step)
      |
      v
FiGS VelocityController (P-cascaded inner loop)
      |
      v
quadrotor body-rate commands [uf, wx, wy, wz]
```

## Repository layout

```text
nav_policy/
|-- configs/                 default.yaml etc.
|-- data/
|   |-- raw/<run>/           SINGER-format rollouts (trajectories_val*.pt, video_val*.mp4, imgdata_val*.pt)
|   |-- processed/<run>/     per-sub-trajectory caches + manifest + stats (gitignored)
|   |-- checkpoints/         training outputs (gitignored)
|-- docker-compose.yml       reuses figs:latest, mounts nav_policy + FiGS
|-- src/nav_policy/
|   |-- data/                build_dataset.py, rgb_horizon_dataset.py, normalization.py
|   |-- model/               rgb_velocity_policy.py, losses.py
|   |-- train/               train_bc.py, metrics.py
|   |-- deploy/              policy_controller.py, frame_buffer.py
|-- scripts/                 thin CLI entry points
|-- pyproject.toml
\-- README.md
```

## Quick start

```bash
# 1. Build dataset (host: WSL or container)
cd nav_policy
docker compose run --rm nav_policy
# inside container:
python scripts/build_dataset.py --config configs/default.yaml

# 2. Train BC
python scripts/train_bc.py --config configs/default.yaml

# 3. Deploy in FiGS (TBD)
python scripts/eval_in_figs.py --checkpoint data/checkpoints/bc_latest.pt --scene <scene>
```

## Data assumptions

Each `data/raw/<run>/` directory is in SINGER validation-rollout format:

- `trajectories_val{NNNNN}.pt` -- dict with `data: List[Trajectory]`, where each
  Trajectory has `Tro (Nctl+1,)`, `Xro (10, Nctl+1)`, `Uro (4, Nctl)` at 20 Hz.
- `imgdata_val{NNNNN}.pt` -- dict with `data: List[{rollout_id, start_id, end_id}]`
  specifying the frame range of each sub-trajectory in the concatenated rgb video.
- `video_val_rollout_images_rgb{NNNNN}.mp4` -- 20 fps, 640x360, concatenation
  of all sub-trajectories in the stack.

State convention (matches `FiGS-Standalone/src/figs/dynamics/model_equations.py`):
position `Xro[0:3]`, velocity `Xro[3:6]`, quaternion (Hamilton, scalar-last
`[qx, qy, qz, qw]`) `Xro[6:10]`. World z is down (NED-like).

The training target `[vx, vy, vz, psi_dot]` is reconstructed per controller
step from:
- `vx, vy, vz = Xro[3:6, k]` (achieved velocity in world frame)
- `psi_dot   = unwrap(yaw(Xro[6:10, k+1])) - unwrap(yaw(Xro[6:10, k])) * fs`
  with `fs = 20 Hz`.
