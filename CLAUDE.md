# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**V-LEAD** — drone autonomy in photorealistic 3D Gaussian Splat (3DGS) environments (Stanford CS231N / CS224R project). A physics-accurate quadrotor simulator is coupled with learned visuomotor navigation policies trained from MPC expert demonstrations.

The tree contains **two first-party Python packages** plus **two vendored git submodules**:

| Dir | Package | Role | First-party? |
|---|---|---|---|
| `nav_policy/` | `nav_policy` | Goal-conditioned RGB(+depth)→velocity navigation. BC → DAgger → RL (PPO/SAC) training stack. **The main active project.** | yes |
| `vlead/` | `vlead_flight` | Thin deployment shim: wraps a trained `nn.Module` as a duck-typed FiGS controller (`VLeadPilot`) so FiGS can fly it. CLI: `python -m vlead_flight.deploy`. | yes |
| `FiGS-Standalone/` | `figs` | Submodule. The simulator: CasADi/ACADOS quadrotor dynamics, body-rate **MPC expert**, RRT*/min-snap planning, nerfstudio/gsplat rendering (RGB + depth + CLIP-semantic). | submodule |
| `SINGER/` | `sousvide` | Submodule. The original DAgger imitation pipeline (rollout → observation → train-history → train-command). nav_policy consumes its rollout data format. | submodule |

`AGENT_CONTEXT.md` is a detailed cognitive map of the FiGS+SINGER submodules and references a `Semantic_HSM/` package — **note that `Semantic_HSM/` is not present in this working tree**; treat those sections as background on the submodules only.

## Critical conventions (shared across all packages)

These are duck-typing/numerical contracts. Violating them fails silently or produces wrong-frame results — verify before changing anything touching control, state, or rendering.

- **State vector (10-dim):** `[px,py,pz, vx,vy,vz, qx,qy,qz,qw]`. Position & velocity in world frame; quaternion **Hamilton, scalar-last**, identity `[0,0,0,1]`.
- **World frame is NED-like:** z points **DOWN**, gravity `[0,0,+9.81]`. `target=[5,0,-1.5]` means 5 m forward, 1.5 m **above** ground.
- **Control output (4-dim):** `u=[uf, ωx,ωy,ωz]`. Thrust `uf ∈ [-1,0]` (negative; hover ≈ -0.41 for `carl` frame). Body rates ±5 rad/s.
- **Policy output:** `[B, H, 4]` receding-horizon velocity commands `[vx,vy,vz,ψ̇]` in world frame (H≈10). **Only the first step is applied** (standard MPC); the rest supervise training.
- **Controller contract is duck-typed, not inherited.** `Simulator.simulate()` accepts any object with `.control(tcr, xcr, upr, obj, icr, zcr)` and a `.hz` attribute. `VLeadPilot` and SINGER's `Pilot` do **not** subclass FiGS's `BaseController`. Match the exact signature or the sim loop breaks silently.
- **Coordinate transform `T_w2g` = diag([1,-1,-1,1])** (world→gsplat flips Y,Z). All controller math is world-frame; only rendering converts.

## Everything runs in Docker (Python 3.10 only)

The `figs:latest` image (built once from `FiGS-Standalone/Dockerfile.FiGS`, ~12.5 GB, PyTorch 2.1.2+cu118) is the only supported environment. `figs`, `gemsplat`, and the relevant packages are editable-installed on first container entry (slow once, instant after). There is no host-side test/run path — always go through a container.

```bash
# nav_policy work (dataset, BC/DAgger/RL training, eval) — from nav_policy/
docker compose -f nav_policy/docker-compose.yml run --rm nav_policy bash   # from repo root
# or:  cd nav_policy && docker compose run --rm nav_policy

# Full V-LEAD container (mounts all 4 packages: figs, sousvide, vlead_flight, nav_policy)
docker compose run --rm vlead          # from repo root; lands in /workspace/vlead

# SINGER-only
cd SINGER && docker compose run --rm singer
```

**Do not** invoke `docker compose run ... python ...` directly — that overrides the entrypoint that installs packages and creates the host-UID user. Always get a shell first, then run.

### sb3 / torch pin (do not bump)
`stable-baselines3` is pinned to **2.2.1** everywhere (`pyproject.toml` extras, compose files). sb3 ≥2.3 requires torch ≥2.3 and silently upgrades the image's torch 2.1.2+cu118 → a CUDA-13 wheel that **breaks gsplat**. Same reason: don't change the base torch.

### Docker volumes
Changing dependencies in the base image requires `docker compose down -v` to clear the persisted `site-packages` named volume.

## nav_policy — common commands (inside container, CWD `/workspace/nav_policy`)

```bash
# Build dataset + precompute depth (DA2 weights at data/weights/depth_anything_v2_vits.pth)
python scripts/build_dataset_flightroom.py --config configs/flightroom.yaml
python scripts/precompute_da2_depth.py --processed-root data/processed_flightroom

# BC training (local) — or via Modal: `modal run nav_policy/modal_train.py`
python scripts/train_bc.py --config configs/arch_rgb_da2_crossattn.yaml

# DAgger
python scripts/run_dagger.py --config configs/dagger_round1_mpc.yaml

# RL fine-tuning (warm-started from BC/DAgger ckpt; PPO is default, SAC optional)
python scripts/train_rl.py --config configs/train_rl_smoke.yaml --save-videos   # smoke
python scripts/train_rl.py --config configs/train_rl_flightroom.yaml            # PPO
python scripts/train_rl.py --config configs/train_rl_flightroom_sac.yaml        # SAC
python scripts/train_rl.py --config <cfg> --resume-from data/checkpoints_rl_a2/rl_ppo_a2_latest.pt

# Eval
python scripts/eval_offline.py --config configs/eval_offline_flightroom.yaml --checkpoint <ckpt> --output-dir <dir>
python scripts/eval_in_figs.py --config configs/eval_closed_loop_flightroom_suite.yaml   # closed-loop
```

Config selects everything (algorithm, rewards, augmentations, splits). RL logs are CSVs next to checkpoints (`*_episodes.csv`, `*_log.csv`, `*_summary.json`). The architecture/training/RL source map lives in `nav_policy/README.md`; the experiment plan in `nav_policy/docs/CROSS_SCENE_DA2_RL_PLAN.md`; ablation configs in `nav_policy/configs/ABLATIONS.md`.

**Scene splits (flightroom):** train = 064652/071718/071353; val = 071733 + backroom; OOD test = packardpark (closed-loop only).

## vlead_flight — deployment

```bash
# inside the `vlead` container, CWD /workspace/vlead
python -m vlead_flight.deploy smoke                      # wiring check, no checkpoint (drone hovers)
python -m vlead_flight.deploy rollout --checkpoint <m.pth> --scene <scene> --target "5.0,0.0,-1.5" --record --use-depth
python tests/test_pilot_smoke.py                         # 4 wiring tests (needs the figs container + a scene)
```

- **Checkpoint must be a pickled `nn.Module` instance**, not a state_dict: `torch.save(model, path)`. A state_dict is rejected at load.
- Your network must implement `forward(rgb[B,T,3,H,W], depth[B,T,1,H,W]|None, goal_heading[B,3], goal_distance[B,1]) -> [B,H,4]`. See `vlead/vlead_flight/network_protocol.py` (runtime-checkable `Protocol` + `DummyVLeadNet`).
- The deploy doc in `vlead/README.md` has the full programmatic API, recorder schema, eval metrics, and a troubleshooting table.

## ACADOS gotcha

`VehicleRateMPC.__init__` generates and compiles C code (`c_generated_code/` in CWD) at construction. **`del ctl` after use** or ACADOS re-init errors on the next construction. Relevant when writing the DAgger MPC oracle or any expert-relabel script (`nav_policy/src/nav_policy/dagger/mpc_oracle.py`).

## Storage note (from AGENT_CONTEXT.md)

On the original `coruscant` host, `/home/kothari1` is at 100% capacity — large artifacts, 3DGS scenes, and checkpoints live under `DATA_PATH` (default `/data/kothari1/singer_figs_data`), with `FiGS-Standalone/3dgs` and `SINGER/cohorts` symlinked there. Paths and `DATA_PATH` differ on this host; check the `.env` files (`FiGS-Standalone/.env`, `SINGER/.env`, root `.env`) for the live values before assuming any absolute path.
