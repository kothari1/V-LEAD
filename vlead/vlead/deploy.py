"""CLI entry point for V-LEAD pilot deployment in FiGS.

Run inside the SINGER (or FiGS) Docker container:

    # Smoke test (no checkpoint, uses DummyVLeadNet → hover)
    python -m vlead.deploy smoke

    # Full rollout with a trained checkpoint
    python -m vlead.deploy rollout --checkpoint /path/to/model.pth \\
        --target 5.0 0.0 -1.5 --duration 15.0 --record

    # With depth observations
    python -m vlead.deploy rollout --checkpoint ... --use-depth
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

from vlead.pilot import VLeadPilot
from vlead.recorder import RolloutRecorder
from vlead.network_protocol import DummyVLeadNet
from vlead.eval import summarize, print_summary

app = typer.Typer(no_args_is_help=True)

DEFAULT_DATA = os.environ.get("DATA_PATH", "/data/kothari1/singer_figs_data")
DEFAULT_SCENE = "flightroom_ssv_exp/gemsplat/2026-02-28_205058"
DEFAULT_X0 = np.array([0., 0., -1.,  0., 0., 0.,  0., 0., 0., 1.])


def _parse_target(target: str) -> np.ndarray:
    """Parse 'x,y,z' string into a (3,) float array."""
    parts = [p.strip() for p in target.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter(
            f"--target must be 'x,y,z' (3 comma-separated floats), got {target!r}"
        )
    try:
        return np.array([float(p) for p in parts])
    except ValueError as e:
        raise typer.BadParameter(f"--target parse error: {e}")


def _load_network(checkpoint: Optional[Path], device: str) -> torch.nn.Module:
    if checkpoint is None:
        print("[vlead] No --checkpoint provided. Using DummyVLeadNet (hover).")
        return DummyVLeadNet(horizon=10)

    print(f"[vlead] Loading checkpoint from {checkpoint}")
    net = torch.load(str(checkpoint), map_location=device)
    if not isinstance(net, torch.nn.Module):
        raise ValueError(
            "Checkpoint must be a pickled nn.Module instance. "
            "If you have a state_dict, instantiate the architecture first "
            "and load_state_dict() before saving as a module."
        )
    return net


def _build_simulator(scene: str, data_path: str):
    from figs.simulator import Simulator
    print(f"[vlead] Building Simulator (scene={scene})")
    return Simulator(
        scene, "baseline", "carl",
        gsplats_path=Path(data_path) / "3dgs",
    )


@app.command()
def smoke(
    scene: str = DEFAULT_SCENE,
    duration: float = 5.0,
    data_path: str = DEFAULT_DATA,
    use_depth: bool = False,
):
    """Smoke test with DummyVLeadNet — drone should hover at start position."""
    sim = _build_simulator(scene, data_path)
    pilot = VLeadPilot(
        network=DummyVLeadNet(horizon=10),
        target_xyz=np.array([2., 0., -1.]),
        gsplat=sim.gsplat,
        frame_name="carl",
        use_depth=use_depth,
    )
    Tro, Xro, Uro, _, _, _ = sim.simulate(pilot, 0.0, duration, DEFAULT_X0.copy())
    print_summary(summarize(Tro, Xro, Uro, pilot.target_xyz))


@app.command()
def rollout(
    checkpoint: Optional[Path] = typer.Option(None, help="Pickled nn.Module checkpoint"),
    scene: str = DEFAULT_SCENE,
    target: str = typer.Option("5.0,0.0,-1.5", help="x,y,z world frame (comma-separated)"),
    duration: float = 15.0,
    use_depth: bool = False,
    record: bool = False,
    output_dir: Path = Path("vlead_runs/eval"),
    data_path: str = DEFAULT_DATA,
    compile_network: bool = False,
    dtype: str = typer.Option("fp32", help="fp32 | bf16 | fp16"),
):
    """Run a single rollout with the given trained checkpoint."""
    target_xyz = _parse_target(target)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    if dtype not in dtype_map:
        raise typer.BadParameter(f"--dtype must be one of {list(dtype_map)}")
    torch_dtype = dtype_map[dtype]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _load_network(checkpoint, device)

    sim = _build_simulator(scene, data_path)
    rec = RolloutRecorder() if record else None

    pilot = VLeadPilot(
        network=net,
        target_xyz=target_xyz,
        gsplat=sim.gsplat,
        frame_name="carl",
        use_depth=use_depth,
        device=device,
        dtype=torch_dtype,
        compile_network=compile_network,
        autocast=(torch_dtype != torch.float32),
        recorder=rec,
    )

    Tro, Xro, Uro, _, _, _ = sim.simulate(pilot, 0.0, duration, DEFAULT_X0.copy())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(Tro, Xro, Uro, target_xyz)
    print_summary(summary)

    if rec is not None:
        rec_path = rec.save(output_dir / "rollout.pt")
        print(f"[vlead] Recorded {len(rec)} steps to {rec_path}")

    np.savez(output_dir / "trajectory.npz", Tro=Tro, Xro=Xro, Uro=Uro)
    print(f"[vlead] Trajectory saved to {output_dir / 'trajectory.npz'}")

    raise typer.Exit(code=0 if summary["success"] else 1)


@app.command()
def dagger(
    checkpoint: Path,
    expert_trajectory: Path,
    scene: str = DEFAULT_SCENE,
    target: str = typer.Option("5.0,0.0,-1.5", help="x,y,z (comma-separated)"),
    duration: float = 15.0,
    output_dir: Path = Path("vlead_runs/dagger"),
    data_path: str = DEFAULT_DATA,
):
    """DAgger v1 stub. Real implementation needs an expert pilot queryable
    at arbitrary (state, time) — typically VehicleRateMPC with the student's
    visited state injected. For now use:

        python -m vlead.deploy rollout --checkpoint ... --record

    and post-process the recorded states against expert MPC offline.
    """
    raise NotImplementedError(
        "DAgger CLI is a stub. v1 path:\n"
        "  1. Run `rollout --record` to capture student trajectory + states.\n"
        "  2. Re-run expert MPC at each recorded state offline.\n"
        "  3. Append (state, expert_action) tuples to training set.\n"
        "  4. Retrain.\n"
        "Full per-step DAgger requires more design — see plan."
    )


if __name__ == "__main__":
    app()
