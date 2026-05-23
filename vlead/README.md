# vlead

V-LEAD goal-conditioned visuomotor navigation for quadrotors (CS231N project).

## What it does

Wraps a trained V-LEAD network as a duck-typed FiGS controller, so the FiGS Simulator can fly the drone using the network's velocity-command outputs without any FiGS code changes.

Pipeline at every control step:
```
RGB (+optional Depth) → temporal frame buffer →
  Network forward → 10 velocity commands [vx, vy, vz, ψ̇] →
    take first command → VelocityController → body rates → ACADOS
```

## Install (inside SINGER Docker container)

The SINGER `docker-compose.yml` automatically mounts and editable-installs this package alongside `figs`, `gemsplat`, and `sousvide`. Just enter the container:

```bash
cd /home/kothari1/autonomy_projects/V-LEAD/SINGER
docker compose run --rm singer
```

## Quick start

```bash
# Smoke test — no checkpoint needed, uses DummyVLeadNet (hover)
python -m vlead.deploy smoke

# Full rollout with trained net
python -m vlead.deploy rollout \
    --checkpoint /path/to/model.pth \
    --target 5.0 0.0 -1.5 \
    --duration 15.0 \
    --record \
    --output-dir /data/kothari1/singer_figs_data/vlead_runs/eval_001

# With depth observations
python -m vlead.deploy rollout --checkpoint ... --use-depth

# Run smoke tests
python /workspace/vlead/tests/test_pilot_smoke.py
```

## Network protocol

Your trained `nn.Module` must implement:

```python
def forward(
    rgb:           Tensor,          # [B, T, 3, H, W]
    depth:         Tensor | None,   # [B, T, 1, H, W]
    goal_heading:  Tensor,          # [B, 3]   unit vector world frame
    goal_distance: Tensor,          # [B, 1]   meters
) -> Tensor:                         # [B, H, 4]  receding-horizon velocities
```

See `vlead/network_protocol.py` for the runtime-checkable `Protocol` and a `DummyVLeadNet` reference impl.

## Blackwell GPUs

Code is device-agnostic — no compute_capability checks, no inline PTX.

For Blackwell-class hardware:
- Run with `--dtype bf16` for native bf16 throughput.
- Add `--compile-network` to enable `torch.compile(mode='reduce-overhead')`.
- The FiGS Docker base image ships with PyTorch 2.1.2+cu118 which does not run on Blackwell. Rebuild that image with PyTorch ≥2.6 + CUDA ≥12.8 (one-time infra task, not part of this package).

## Files

| File | Role |
|------|------|
| `vlead/pilot.py` | `VLeadPilot` — duck-typed FiGS controller |
| `vlead/network_protocol.py` | Forward signature + `DummyVLeadNet` |
| `vlead/recorder.py` | `RolloutRecorder` for offline / DAgger / RL |
| `vlead/eval.py` | Post-rollout metrics |
| `vlead/deploy.py` | Typer CLI: `smoke`, `rollout`, `dagger` |
| `tests/test_pilot_smoke.py` | Wiring sanity checks |
