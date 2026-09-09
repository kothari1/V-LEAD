# Architecture

How V-LEAD fits together: the simulator layer, the controller contract that lets a
neural policy stand in for an MPC, and the training/deployment packages built on top.

- [Layers](#layers)
- [The controller contract](#the-controller-contract)
- [State and control vectors](#state-and-control-vectors)
- [Control loop](#control-loop)
- [Policy architecture](#policy-architecture)
- [Repository layout](#repository-layout)
- [Container setup](#container-setup)
- [Gotchas](#gotchas)

---

## Layers

| Layer | Package | Role |
|---|---|---|
| Simulation + rendering | `figs` (`FiGS-Standalone/`) | Quadrotor dynamics (CasADi ODE, ACADOS integrator), body-rate MPC expert, min-snap / RRT\* trajectory generation, 3D Gaussian-Splatting renderer (RGB, depth, CLIP-semantic) |
| Expert data synthesis | `sousvide` (`SINGER/`) | Campaign orchestration: fly the MPC expert through 3DGS scenes, record synchronized video and state/control logs |
| Policy training | `nav_policy/` | Dataset builder, ResNet-18 + DA2 cross-attention policy, BC / DAgger / RL trainers, offline + closed-loop evaluators |
| Policy deployment | `vlead/` (`vlead_flight`) | Wraps a trained network as a FiGS-compatible controller; rollout CLI, recorder, Gym-style RL environment |

`FiGS-Standalone` and `SINGER` are git submodules. `nav_policy` and `vlead` are the
code written for this project.

The dependency direction is one-way: `sousvide` and `nav_policy` import `figs`; `figs`
knows nothing about them. Integration is through plain Python imports, not linked
native libraries.

---

## The controller contract

This is the design decision that makes the rest of the system possible.

`figs.simulator.Simulator.simulate()` does not require a controller to inherit from any
base class. It accepts **any object** exposing:

```python
.control(tcr, xcr, upr, obj, icr, zcr) -> (u, ...)   # one control step
.hz                                                   # control rate attribute
```

This is a **duck-typing contract**, not an inheritance hierarchy. Consequences:

- The privileged MPC expert (`figs.control.vehicle_rate_mpc.VehicleRateMPC`), SINGER's
  `Pilot`, and this project's `vlead_flight.pilot.VLeadPilot` are interchangeable at the
  simulator boundary.
- A learned policy can be dropped in with **zero changes to FiGS**.
- Expert and student run through an identical code path, so behavioral differences are
  attributable to the policy rather than to harness differences — which is what makes
  the DAgger oracle re-solve and closed-loop eval trustworthy.

`VLeadPilot` therefore sits between the network and the simulator: it maintains the
rolling frame buffer, computes the goal vector, runs the forward pass, de-standardizes
the predicted command, and hands the first step to the inner velocity controller.

---

## State and control vectors

Defined by the quadrotor ODE in `figs.dynamics.model_equations`.

**State** `x ∈ R^10`:

| Indices | Symbol | Meaning |
|---|---|---|
| `[0:3]` | `px, py, pz` | Position, world frame (NED-like: **z points down**) |
| `[3:6]` | `vx, vy, vz` | Velocity, world frame |
| `[6:10]` | `qx, qy, qz, qw` | Orientation quaternion (Hamilton, **scalar-last**; identity = `[0,0,0,1]`) |

**Low-level control** `u ∈ R^4` — what the ACADOS integrator consumes:

| Index | Symbol | Range |
|---|---|---|
| `[0]` | `uf` | Normalized collective thrust, `[-1, 0]` (negative; hover ≈ `-0.41` for the `carl` frame) |
| `[1:4]` | `ωx, ωy, ωz` | Body rates, rad/s, `±5.0` |

Dynamics: `ṗ = v`, `v̇ = g + (tn·uf/m)·R(q)·e₃`, `q̇ = ½·q⊗ω`.

**Policy output** — deliberately *not* `u`. The network emits velocity commands
`[vx, vy, vz, ψ̇]` in the world frame, and a cascaded P velocity controller already
present in FiGS converts them to `u`. The policy never models attitude or thrust
dynamics.

> Because the world frame is z-down, a goal at `[5, 0, -1.5]` is 5 m forward and
> 1.5 m **above** ground, and a commanded `vz < 0` means climb.

---

## Control loop

One iteration at 20 Hz:

```
ACADOS integrator          x_{k+1} = f(x_k, u_k)
        |
        v  state x_k
camera transform           T_c2w = body_to_cam(x_k)
        |
        v
3DGS renderer              render_rgb(camera, T_c2w) -> RGB (+ depth_raw)
        |
        v
        +--> MPC expert     VehicleRateMPC.control()   [data generation, DAgger oracle]
        |
        +--> neural policy  VLeadPilot.control()       [training, deployment]
                              rolling T-frame buffer
                              frozen DA2-S depth
                              goal vector (heading, distance)
                              network forward -> [H,4]
                              take first command
                              velocity controller -> u
        |
        v  u_k = [uf, wx, wy, wz]
back to integrator
```

Every control step triggers a fresh 3DGS render. That render is the dominant cost and
the binding constraint on how many RL episodes and eval rollouts are affordable.

---

## Policy architecture

```
RGB history [B,4,3,224,224]
   |                    \
   |                     +--> frozen DA2-S (ViT-S) --> depth maps --> CNN --> 256-D
   |                                                                            |
   +--> ResNet-18 (ImageNet init; stem + layer1 frozen) --> 512-D/frame         |
                                    |                                           |
                                    v                                           v
                          LayerNorm + cross-attention (4 heads; RGB queries depth)
                                    |
                                    v
                          GRU (1 layer, hidden 256) + LayerNorm
                                    |
   goal vector [hx, hy, d_norm] --> Linear(3->32) + ReLU --+
                                    |                       |
                                    v                       v
                          concat -> 288 -> MLP(256, 128, dropout 0.1) -> 40
                                    |
                                    v
                          command horizon [B, 10, 4]  (z-scored)
                                    |
                                    v
                          execute first step; re-plan next tick
```

Two details that matter:

- **LayerNorm before concatenation.** GRU hidden states have unbounded magnitude while
  the goal embedding is naturally bounded (unit vector through Linear+ReLU). Without
  normalizing, the 256-D visual stream swamps the 32-D goal stream by sheer scale and
  goal conditioning stops mattering.
- **Horizon supervision.** Only the first of the 10 predicted steps is ever executed.
  The remaining nine are training-time supervision that regularize the recurrent state.

The network predicts in z-scored command space; per-component means and standard
deviations are fit once on the round-0 training split and stored in the checkpoint, so
the runtime wrapper can de-standardize.

---

## Repository layout

```
V-LEAD/
├── FiGS-Standalone/            submodule — simulator, MPC, 3DGS rendering
├── SINGER/                     submodule — expert rollout campaigns
├── nav_policy/
│   ├── configs/                BC, DAgger, eval, RL yaml
│   ├── scripts/                CLI entry points + plot scripts
│   └── src/nav_policy/
│       ├── data/               dataset, augmentations, normalization
│       ├── model/              RGB and RGB+DA2 policies, depth estimator
│       ├── train/              BC trainer
│       ├── dagger/             DAgger rollouts + MPC re-solve oracle
│       ├── rl/                 PPO, SAC, residual TD3+BC, rollout collection
│       ├── evaluate/           offline + closed-loop evaluators
│       └── vendor/             vendored Depth Anything V2
├── vlead/vlead_flight/         deployment: pilot, recorder, Gym-style env
├── results/                    figures + committed run logs (self-contained)
├── docs/figures/               write-up figures
├── ARCHITECTURE.md             this file
├── TEST_RESULTS.md             per-(checkpoint, suite) eval index
├── V-LEAD_instructions.md      pipeline operator guide
└── FiGS_instructions.md        simulator / 3DGS operator guide
```

---

## Container setup

Everything runs inside one image, `figs:latest`, built from
`FiGS-Standalone/Dockerfile.FiGS`. It ships Python 3.10, PyTorch 2.1.2+cu118,
nerfstudio, gsplat, ACADOS, CasADi, and numpy 1.26.4.

`docker-compose.yml` at the repo root bind-mounts all four packages and
editable-installs them on first entry:

```bash
DATA_PATH=/path/to/your/data docker compose run --rm vlead
# lands in /workspace/vlead with figs, gemsplat, sousvide, vlead_flight, nav_policy installed
```

`DATA_PATH` is mounted **at the same absolute path inside the container as outside**,
because scene configs and checkpoints store absolute paths. It is a required variable —
compose fails fast with a message if unset.

GPU access uses the modern compose syntax
(`deploy.resources.reservations.devices`). The legacy `runtime: nvidia` key is not
supported and will error.

---

## Gotchas

Collected while building this; each cost real debugging time.

- **`perception_mode.yml` must match what the pilot expects.** If it is set to a mode
  `VLeadPilot` does not handle, the simulator passes `icr=None` into `.control()` and the
  rollout dies. For RGB policies set `visual_mode: rgb` in
  `FiGS-Standalone/configs/perception/perception_mode.yml`.

- **Checkpoints for `--checkpoint` must be pickled `nn.Module` instances, not
  state-dicts.** Save with `torch.save(model, path)`. If you have a state-dict,
  instantiate the architecture, `load_state_dict()`, then re-save the module.

- **`stable-baselines3` is pinned to 2.2.1.** Versions ≥2.3 require torch ≥2.3 and will
  silently upgrade the image's torch 2.1.2+cu118 to a CUDA 13 wheel, which breaks
  `gsplat`.

- **Train/val splits are assigned by sub-trajectory, never by frame.** A frame-level
  random split leaks temporally adjacent frames across the boundary and inflates
  validation numbers.

- **Scene paths must be specific enough to be unambiguous.** If multiple `config.yml`
  files live under a scene name, pass the full sub-path
  (e.g. `flightroom_ssv_exp/gemsplat/2026-02-28_205058`) or discovery raises on multiple
  matches.

- **A yaw-correct arrival is required for success.** The settle detector needs position
  *and* yaw tolerance plus consecutive low-velocity steps. A policy can sit exactly on
  the goal and still be scored a failure for facing the wrong way — this is the dominant
  residual failure mode (see [TEST_RESULTS.md](TEST_RESULTS.md)).

- **Inner-loop gains must satisfy `ka >> kv`.** With the defaults `kv=2.0, ka=5.0`;
  drifting or oscillating flight usually means these are wrong for the frame, or that
  `frame_name` does not match the drone being simulated.
