# vlead_flight

V-LEAD goal-conditioned visuomotor navigation deployment for quadrotors (CS231N project).

Wraps a trained PyTorch network as a duck-typed FiGS controller so the FiGS Simulator can fly a drone in a Gaussian-Splat scene using the network's velocity-command outputs — with no changes to FiGS itself.

> **Naming**: outer repo dir is `V-LEAD/vlead/` (project root for this package). Importable Python module is `vlead_flight` (mirrors SINGER's `sousvide.flight` convention). All imports: `from vlead_flight import ...`. CLI: `python -m vlead_flight.deploy ...`.

---

## Pipeline (one control step)

```
RGB (+optional Depth) at current camera pose
        │
        ▼  preprocess + push to circular T-frame buffer
[B=1, T, 3, H, W]  (rgb)
[B=1, T, 1, H, W]  (depth, optional)
goal_heading [B,3]   <-- (target_xyz - drone_pos) / ||·||
goal_distance [B,1]  <-- ||target_xyz - drone_pos||
        │
        ▼  network.forward(...)
[B, H=10, 4]   receding-horizon velocity commands [vx, vy, vz, ψ̇]
        │
        ▼  take first command (standard MPC convention)
vel_cmd [4]    in world frame
        │
        ▼  inner-loop VelocityController (P-control vel → P-control attitude)
body-rate cmd [uf, ωx, ωy, ωz]
        │
        ▼  FiGS ACADOS integrator
next drone state
```

---

## Prerequisites

| What | Where / How |
|------|-------------|
| V-LEAD repo cloned with submodules | `git clone --recursive https://github.com/kothari1/V-LEAD.git` |
| `figs:latest` Docker image built | One-time `docker compose build` in `FiGS-Standalone/` |
| `DATA_PATH` env var set | Defaults to `/data/kothari1/singer_figs_data`; override via `SINGER/.env` |
| Trained 3DGS scene checkpoint | Lives at `$DATA_PATH/3dgs/workspace/outputs/<scene>/...` |
| Drone frame config | Default `carl` in `FiGS-Standalone/configs/frame/carl.json` |

Tested scene: `flightroom_ssv_exp/gemsplat/2026-02-28_205058`. Tested frame: `carl`.

---

## Install (inside SINGER Docker container)

The SINGER `docker-compose.yml` mounts `../vlead` at `/workspace/vlead` and runs `pip install -e` automatically on container start. To enter:

```bash
cd /home/kothari1/autonomy_projects/V-LEAD/SINGER
docker compose run --rm singer
```

To override the mount path: `VLEAD_PATH=/some/other/path docker compose run --rm singer`.

---

## CLI — `python -m vlead_flight.deploy`

### `smoke` — wiring sanity check, no checkpoint needed
```bash
python -m vlead_flight.deploy smoke
```
Loads `DummyVLeadNet` (always outputs zero velocities). Drone should hover. Exit 0 means wiring works end-to-end.

### `rollout` — eval a trained checkpoint
```bash
python -m vlead_flight.deploy rollout \
    --checkpoint /path/to/model.pth \
    --scene "flightroom_ssv_exp/gemsplat/2026-02-28_205058" \
    --target "5.0,0.0,-1.5" \
    --duration 15.0 \
    --use-depth \
    --record \
    --output-dir /data/kothari1/singer_figs_data/vlead_runs/eval_001 \
    --dtype bf16 \
    --compile-network
```
- `--target` is comma-separated `"x,y,z"` (typer parses single-string; negatives like `-1.5` work)
- `--record` writes `rollout.pt` (per-step obs+action) to `--output-dir`
- Always writes `trajectory.npz` (Tro, Xro, Uro) to `--output-dir`
- Exits 0 if goal reached, 1 otherwise (pipeable)

### `dagger` — STUB. v1 path is offline; see "DAgger workaround" below.

---

## Programmatic use

```python
import numpy as np
import torch
from figs.simulator import Simulator
from vlead_flight.pilot import VLeadPilot
from vlead_flight.recorder import RolloutRecorder
from vlead_flight.eval import summarize, print_summary

# 1. Build sim
sim = Simulator(
    "flightroom_ssv_exp/gemsplat/2026-02-28_205058",
    "baseline", "carl",
    gsplats_path="/data/kothari1/singer_figs_data/3dgs",
)

# 2. Load trained network (must be pickled nn.Module, NOT state_dict)
net = torch.load("/path/to/model.pth", map_location="cuda")

# 3. Build pilot
rec = RolloutRecorder()
pilot = VLeadPilot(
    network=net,
    target_xyz=np.array([5.0, 0.0, -1.5]),
    gsplat=sim.gsplat,           # required for depth render
    frame_name="carl",
    hz=20,
    use_depth=True,
    frame_window=8,
    img_resolution=(224, 224),
    device="cuda",
    dtype=torch.bfloat16,
    kv=2.0, ka=5.0,
    recorder=rec,
    compile_network=True,
    autocast=True,
)

# 4. Run rollout (single-scene, single-target)
x0 = np.array([0., 0., -1.,  0., 0., 0.,  0., 0., 0., 1.])  # see "State vector" below
Tro, Xro, Uro, _, _, _ = sim.simulate(pilot, 0.0, 15.0, x0)

# 5. Eval + save
print_summary(summarize(Tro, Xro, Uro, pilot.target_xyz))
rec.save("rollout.pt")

# 6. Move to a new target without rebuilding
pilot.set_target(np.array([2., 3., -2.]))
pilot.reset_buffer()             # clears T-frame history between independent runs
```

---

## `VLeadPilot.__init__` arguments

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `network` | `nn.Module` | required | Must satisfy `VLeadNetworkProtocol` (see below) |
| `target_xyz` | `np.ndarray (3,)` | required | World-frame goal in NED (z down, z<0 is above ground) |
| `gsplat` | `figs.render.gsplat_semantic.GSplat` | required | Pass `sim.gsplat`. Needed for depth render even if `use_depth=False` (cheap retain) |
| `frame_name` | `str` | `"carl"` | Loads `FiGS-Standalone/configs/frame/<name>.json` for mass, thrust, camera |
| `hz` | `int` | `20` | Must match `Simulator` control rate |
| `use_depth` | `bool` | `False` | If True, pilot re-renders via `gsplat.render_rgb()` to get `depth_raw`; +1 GPU render/step |
| `frame_window` | `int` | `8` | Temporal T for spatiotemporal backbones (3D-ResNet, etc.) |
| `img_resolution` | `(int, int)` | `(224, 224)` | Network input HW. Resize+normalize done internally |
| `device` | `str` | auto | `"cuda"` or `"cpu"`; auto-detects |
| `dtype` | `torch.dtype` | `float32` | Use `bfloat16` on Blackwell for speedup |
| `kv` | `float` | `2.0` | Inner-loop velocity P-gain |
| `ka` | `float` | `5.0` | Inner-loop attitude P-gain (must satisfy `ka >> kv` for stability) |
| `configs_path` | `Path` | None | Override for `FiGS-Standalone/configs/` root |
| `recorder` | `RolloutRecorder` | None | If set, captures every control step |
| `compile_network` | `bool` | False | Wrap network with `torch.compile(mode='reduce-overhead')`; silently skips if unsupported |
| `autocast` | `bool` | False | Run forward inside `torch.autocast` — recommended with bf16/fp16 |

Methods: `.set_target(xyz)`, `.reset_buffer()`.

---

## Network contract

Your trained `nn.Module` must implement this `forward`:

```python
def forward(
    rgb:           torch.Tensor,             # [B, T, 3, H, W]  ImageNet-normalized float
    depth:         Optional[torch.Tensor],   # [B, T, 1, H, W]  raw metric depth, or None
    goal_heading:  torch.Tensor,             # [B, 3]   unit vector in world frame
    goal_distance: torch.Tensor,             # [B, 1]   positive scalar (meters)
) -> torch.Tensor:                            # [B, H, 4]  receding-horizon velocity commands
```

Output is `[B, H, 4]` where `H` is the receding horizon length (default 10, read from output shape at runtime). Each row is `[vx, vy, vz, ψ̇]` in the world frame.

See `vlead_flight/network_protocol.py` for the runtime-checkable `Protocol` and `DummyVLeadNet` reference.

### Checkpoint format

`--checkpoint` must point to a **pickled `nn.Module` instance**, not a state-dict. Save with:
```python
torch.save(model, "model.pth")   # whole-module pickle ← USE THIS
# NOT torch.save(model.state_dict(), ...)
```
If you have a state_dict, instantiate the architecture first, `load_state_dict()`, then `torch.save(model, ...)`.

---

## Conventions

### State vector (10-dim)

`xcr = [px, py, pz, vx, vy, vz, qx, qy, qz, qw]`
- Position in world frame (meters)
- Velocity in world frame (m/s)
- Quaternion Hamilton convention, scalar-last `[qx, qy, qz, qw]`. Identity = `[0,0,0,1]`.

### Coordinate frame

World frame is **NED-like**: z points DOWN, gravity = `[0, 0, +9.81]`. So `target_xyz = [5, 0, -1.5]` is 5 m forward, 0 m sideways, 1.5 m ABOVE ground.

### Control output (4-dim, from inner loop)

`u = [uf, ωx, ωy, ωz]`
- `uf ∈ [-1, 0]` normalized thrust (negative; hover ≈ -0.41 for `carl`)
- Body rates `±5` rad/s

### Velocity command (4-dim, what the network outputs)

`[vx, vy, vz, ψ̇]` in world frame. `vz < 0` = upward.

---

## Recorder output schema

`RolloutRecorder.save(path)` writes a torch pickle: `{"steps": [step_dict, ...]}` where each `step_dict` has:

| Key | Type | Shape | Notes |
|-----|------|-------|-------|
| `t` | float | scalar | Sim time (s) |
| `x` | np.float64 | `(10,)` | Drone state at this step |
| `rgb` | np.uint8 or None | `(H, W, 3)` | Raw RGB at native resolution |
| `depth` | np.float32 or None | `(H, W)` | Raw metric depth (None if `use_depth=False`) |
| `goal_heading` | np.float64 | `(3,)` | Unit vector world frame |
| `goal_dist` | float | scalar | Meters to target |
| `vel_pred` | np.float32 | `(H, 4)` | Full receding-horizon prediction |
| `u_cmd` | np.float32 | `(4,)` | Body-rate cmd sent to ACADOS |
| `expert_vel` | np.ndarray or None | `(4,)` | DAgger label, set externally |

Load back:
```python
from vlead_flight.recorder import RolloutRecorder
rec = RolloutRecorder.load("rollout.pt")
print(len(rec), "steps")
print(rec.steps[0]["vel_pred"].shape)   # (10, 4)
```

---

## Eval metrics (`vlead_flight/eval.py`)

`summarize(Tro, Xro, Uro, target_xyz, radius=0.5)` returns:

| Key | Meaning |
|-----|---------|
| `success` | bool — did drone enter `radius` of target at any timestep? |
| `final_dist` | meters from target at end of rollout |
| `time_to_goal` | first time inside `radius` (inf if never) |
| `traj_length` | cumulative arc length (m) |
| `uf_bound_violations` | count of steps with `uf` outside `[-1, 0]` |
| `rate_bound_violations` | count of steps with `‖ω‖ > 5` |
| `mean_uf` | average thrust command |

Individual functions also exposed: `goal_reached`, `time_to_goal`, `final_distance`, `trajectory_length`, `control_bounds_violations`.

---

## Receding-horizon strategy

Network outputs 10 velocity commands per step but **only the first is applied** (standard MPC convention). The remaining 9 supervise training and could be applied open-loop in future for inference speedup, but v1 always re-plans every step at the controller rate.

---

## DAgger workaround (until CLI stub is real)

The `dagger` CLI is a documented stub. v1 path:

```bash
# 1. Capture student trajectory with full state log
python -m vlead_flight.deploy rollout --checkpoint student.pth --record \
    --output-dir runs/dagger_iter_3

# 2. Offline: re-run expert MPC at each recorded state, append (state, expert_action) to training set
python scripts/relabel_with_expert.py runs/dagger_iter_3/rollout.pt   # YOUR script

# 3. Retrain student on aggregated dataset
```

Full per-step online DAgger (expert queried *during* simulation) needs more design — pilot-callable expert MPC, state injection, intervention logic. Out of scope for v1.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'figs'` | Outside container, or container started without entrypoint | Enter via `docker compose run --rm singer` (NOT `... python ...` directly — overrides command) |
| `The search path ... did not return any configurations` | Symlink broken or `DATA_PATH` not mounted | Check `FiGS-Standalone/.env` sets `DATA_PATH=/data/kothari1/singer_figs_data`; restart container |
| `Simulator passed icr=None to VLeadPilot.control()` | `perception_mode.yml` is set to a mode the pilot doesn't handle | Set `FiGS-Standalone/configs/perception/perception_mode.yml` to `visual_mode: rgb` |
| `gsplat.render_rgb did not return 'depth_raw'` | Old gsplat or wrong perception_type | Use the current gemsplat fork; check `gsplat_semantic.py` returns `depth_raw` |
| `Checkpoint must be a pickled nn.Module instance` | You saved a state_dict | Re-save with `torch.save(model, path)` after instantiating + `load_state_dict` |
| `search path returned multiple configurations` | Multiple `config.yml` under scene name | Pass exact sub-path like `flightroom_ssv_exp/gemsplat/2026-02-28_205058` |
| Drone drifts/oscillates | Inner-loop gains off for this frame | Increase `ka`, lower `kv`; check `frame_name` matches drone |
| `torch.compile` warnings | Backend unsupported on this torch+CUDA combo | Drop `--compile-network` (logs warning, continues without) |

---

## Blackwell GPUs

Code is device-agnostic — no `compute_capability` checks, no inline PTX, no SM-arch in `pyproject.toml`.

For Blackwell-class hardware:
- `--dtype bf16` → native bf16 throughput
- `--compile-network` → `torch.compile(mode='reduce-overhead')` (huge gain on Blackwell)
- `autocast=True` constructor arg in programmatic use

**Blocker for Blackwell deployment today**: the FiGS Docker base image ships with PyTorch 2.1.2+cu118, which does NOT run on Blackwell. Need to rebuild that image with PyTorch ≥2.6 + CUDA ≥12.8. One-time infra task, separate from this package.

---

## Tests

```bash
python /workspace/vlead/tests/test_pilot_smoke.py
```
4 tests:
1. `test_dummy_pilot_hovers` — DummyVLeadNet holds position
2. `test_recorder_collects_steps` — recorder captures every step at right rate
3. `test_depth_flag_renders_depth` — depth tensors have positive metric values
4. `test_set_target_updates_goal` — target setter works

---

## Limitations / out of scope (v1)

- Single scene, single target per rollout (no multi-scene campaign mode like SINGER's `ssv_multi3dgs_campaign.py`)
- DAgger CLI is a stub (offline workaround above works)
- No RL implementation (recorder provides data hook; algo lives in separate module)
- No real-world hardware (ZED, ROS) deployment
- Network architecture not provided (write your own following `VLeadNetworkProtocol`)
- FiGS Docker image rebuild for Blackwell is a separate task

---

## File map

| File | Role |
|------|------|
| `vlead_flight/pilot.py` | `VLeadPilot` — duck-typed FiGS controller |
| `vlead_flight/network_protocol.py` | Forward signature + `DummyVLeadNet` reference |
| `vlead_flight/recorder.py` | `RolloutRecorder` for offline / DAgger / RL data |
| `vlead_flight/eval.py` | Post-rollout metrics |
| `vlead_flight/deploy.py` | Typer CLI (`smoke`, `rollout`, `dagger` stub) |
| `tests/test_pilot_smoke.py` | Wiring sanity checks (4 tests, no trained net needed) |
| `pyproject.toml` | Editable install |

---

## Related code (read-only deps in V-LEAD)

| File | What vlead uses |
|------|-----------------|
| `FiGS-Standalone/src/figs/simulator.py` | `Simulator.simulate()` — duck-types `VLeadPilot` |
| `FiGS-Standalone/src/figs/control/velocity_controller.py` | Inner-loop vel→body rate (already tested) |
| `FiGS-Standalone/src/figs/control/base_controller.py` | Config loader helper |
| `FiGS-Standalone/src/figs/dynamics/model_specifications.py` | `generate_specifications()` for camera + drone params |
| `FiGS-Standalone/src/figs/utilities/trajectory_helper.py` | `xv_to_T()` body-to-world transform |
| `FiGS-Standalone/src/figs/render/gsplat_semantic.py` | `GSplat.render_rgb()` for depth queries |
