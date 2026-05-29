# V-LEAD Rollout Dataset

Expert MPC drone demonstrations in a 3D Gaussian Splatting scene, generated for goal-conditioned visuomotor navigation training.

---

## Directory Structure

```
rollouts_vlead/
├── readme_data.md                          ← this file
└── vlead_flightroom/
    └── rollout_data/
        └── flightroom_ssv_exp/
            ├── <YYYY-MM-DD_HHMMSS>/        ← one dir per generation run
            │   ├── trajectories0.pt        ← trajectory tensor bundle for query 0
            │   ├── trajectories1.pt
            │   ├── ...
            │   ├── video0.mp4              ← rendered video for query 0
            │   ├── video1.mp4
            │   └── ...
            └── <another_run_timestamp>/
                └── ...
```

Each `trajectories{i}.pt` corresponds to query index `i` (see Semantic Queries below).

---

## Dataset Variants

Two modes were generated:

| Mode | Description | Traj length | Reps per branch | Domain randomization |
|------|-------------|-------------|-----------------|----------------------|
| **Validation** | Full spawn-to-goal trajectories | ~13 s (N≈260 steps) | 1 | None |
| **Training** | 2-second segments starting from random branch points | 2 s (N=40 steps) | 4 | Yes (mass ±0.3, disturbance force ±0.3) |

Training trajectories start from waypoints distributed along RRT* branches. Domain randomization varies simulated drone mass and applies random persistent body forces, improving policy robustness to model mismatch.

---

## Semantic Queries

Query indices map to physical objects in the flight room scene:

| Index | Query string |
|-------|-------------|
| 0 | `"green clock"` |
| 1 | `"green and pink leafblower"` |
| 2 | `"yellow handheld cordless drill on two boxes"` |
| 3 | `"human mannequin"` |
| 4 | `"ladder"` |

Goals are determined by CLIP semantic similarity: the scene point cloud is queried with the text string to find the 3D centroid of the matching region. This centroid is `goal_xy` in the trajectory tensors.

---

## Trajectory File Format

Each `trajectories{i}.pt` is a Python list of dicts. Load with:

```python
import torch
trajs = torch.load("trajectories0.pt")
traj = trajs[0]   # first trajectory in the list
```

### Fields per trajectory

#### `Tro` — time vector
- Shape: `(N+1,)`
- Units: seconds
- Description: timestamps of each state. `Tro[0] = 0.0`, `Tro[-1] ≈ 13.0` (val) or `2.0` (train). Step size is 1/20 s (20 Hz simulation).

#### `Xro` — state trajectory
- Shape: `(10, N+1)`
- Description: full drone state at each timestep. **Column `i` corresponds to time `Tro[i]`.**
- Row layout:

| Row | Symbol | Units | Description |
|-----|--------|-------|-------------|
| 0 | `px` | m | X position (world frame) |
| 1 | `py` | m | Y position (world frame) |
| 2 | `pz` | m | Z position (world frame, NED: negative = upward) |
| 3 | `vx` | m/s | X velocity (world frame) |
| 4 | `vy` | m/s | Y velocity (world frame) |
| 5 | `vz` | m/s | Z velocity (world frame) |
| 6 | `qx` | — | Quaternion x-component |
| 7 | `qy` | — | Quaternion y-component |
| 8 | `qz` | — | Quaternion z-component |
| 9 | `qw` | — | Quaternion w-component (scalar, Hamilton convention) |

**Coordinate frame:** NED-like. `pz < 0` means the drone is above the ground. The flight altitude is approximately `pz = -1.0 m`.

**Quaternion convention:** scalar-last Hamilton: `[qx, qy, qz, qw]`. Represents rotation from world to body frame.

#### `Uro` — control inputs (expert MPC commands)
- Shape: `(4, N)`
- Description: body-rate commands applied by the NMPC expert at each step. **Column `i` is applied during `[Tro[i], Tro[i+1]]`**, resulting in state `Xro[:,i+1]`.
- Row layout:

| Row | Symbol | Units | Range | Description |
|-----|--------|-------|-------|-------------|
| 0 | `uf` | — | [-1, 0] | Normalized collective thrust. -1 = full throttle, 0 = no thrust |
| 1 | `ωx` | rad/s | ±5 | Roll rate command (body x-axis) |
| 2 | `ωy` | rad/s | ±5 | Pitch rate command (body y-axis) |
| 3 | `ωz` | rad/s | ±5 | Yaw rate command (body z-axis) |

**Thrust sign:** Negative because the body +Z axis points upward; the motor thrust acts in the +Z body direction, countering +Z gravity in the NED world frame.

#### `heading_vec` — goal heading supervision signal
- Shape: `(2, N+1)`
- Description: unit vector pointing from the drone's current XY position toward `goal_xy`. Aligned with `Xro` columns (column `i` corresponds to time `Tro[i]`).
- Computation:
  ```
  diff = goal_xy - [Xro[0,i], Xro[1,i]]
  heading_vec[:,i] = diff / ||diff||   (or [0,0] if drone is on goal)
  ```
- Use this as the directional supervision signal for the goal-conditioned policy (analogous to a "compass" pointing toward the target).

#### `dist` — distance to goal
- Shape: `(N+1,)`
- Units: meters
- Description: Euclidean XY distance from drone to goal at each timestep. Decreases as the drone approaches the object. Aligned with `Xro` columns.
  ```
  dist[i] = ||goal_xy - [Xro[0,i], Xro[1,i]]||
  ```

#### `goal_xy` — goal position (static)
- Shape: `(2,)`
- Units: meters
- Description: 2D world-frame XY coordinates of the semantic target centroid (the object the drone is flying toward). Constant for all timesteps within a trajectory. Used to compute `heading_vec` and `dist`.

#### `frame` — metadata dict
Contains per-trajectory metadata. Key fields:

| Key | Type | Description |
|-----|------|-------------|
| `"scene_name"` | str | Name of the 3DGS scene (`"flightroom_ssv_exp"`) |
| `"course_name"` | str | Name of the RRT* course/config used |
| `"query"` | str | Text query used for CLIP semantic targeting |
| `"goal_pose"` | array (3,) | 3D XYZ point sampled on the 2m-radius orbit around the object centroid (the RRT* goal point, not the object itself) |
| `"obj_centroid"` | array (3,) | 3D XYZ centroid of the CLIP-identified object region (= `[goal_xy[0], goal_xy[1], centroid_z]`) |

**Note:** `goal_xy` equals `[obj_centroid[0], obj_centroid[1]]`. The RRT* planner navigates toward `goal_pose` (on a 2m orbit), but supervision uses `obj_centroid` directly as the semantic target.

#### `Tsol`, `Adv` — solver diagnostics (not needed for training)
- `Tsol`: shape `(4, N)` — ACADOS solver timing stats per step
- `Adv`: shape `(4, N)` — adversarial perturbation buffer (all NaN in this dataset; not used)

---

## Video Files

Each `video{i}.mp4` is a rendered first-person view:
- Resolution: **640 × 360** pixels, RGB
- Frame rate: **20 fps**
- Duration matches trajectory length (val ≈ 13 s, train = 2 s)
- **Frame `k` of the video corresponds to state `Xro[:,k]` and control `Uro[:,k]`** (0-indexed, before the control is applied)
- Rendered from the drone's forward-facing camera in the 3DGS scene using gemsplat (photorealistic Gaussian Splatting renderer with CLIP semantic features)

---

## RRT* Planning Parameters (per query)

| Parameter | Value |
|-----------|-------|
| Orbit radius (outer) | 2.0 m |
| Orbit radius (inner) | 0.4 m |
| Flight altitude | -1.0 m (NED) |
| CLIP similarity threshold | 0.90 |
| CLIP filter radius | 0.025 |
| RRT* nodes per branch | 2500 |
| Branches per query | 110 |

---

## Quick Usage Example

```python
import torch
import numpy as np

trajs = torch.load("trajectories3.pt")   # query 3: "human mannequin"
traj = trajs[0]

Tro         = traj["Tro"]           # (N+1,)
Xro         = traj["Xro"]           # (10, N+1)
Uro         = traj["Uro"]           # (4, N)
heading_vec = traj["heading_vec"]   # (2, N+1)
dist        = traj["dist"]          # (N+1,)
goal_xy     = traj["goal_xy"]       # (2,)

# Position at step 0
pos0 = Xro[:3, 0]   # [px, py, pz]

# Supervised heading at step 0 (goal direction in XY)
goal_dir = heading_vec[:, 0]   # unit vector [dx, dy]

# Expert control at step 0
u0 = Uro[:, 0]   # [uf, wx, wy, wz]
```

---

## Scene

Scene: **flightroom_ssv_exp** — a motion capture flight room containing various objects (clock, leafblower, drill, mannequin, ladder). Reconstructed from ~1200 iPhone 15 Pro images using COLMAP + GemSplat (3D Gaussian Splatting with CLIP semantic features).

3DGS checkpoint: `step-000029999.ckpt` (30k training steps, gemsplat format)