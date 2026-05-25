"""Smoke tests for VLeadPilot — wiring sanity check, no trained net needed.

Run inside SINGER Docker container:
    python3 /workspace/vlead/tests/test_pilot_smoke.py
"""
import os
import sys
from pathlib import Path

import numpy as np

DATA_PATH = Path(os.environ.get("DATA_PATH", "/data/kothari1/singer_figs_data"))
SCENE = "flightroom_ssv_exp/gemsplat/2026-02-28_205058"
X0 = np.array([0., 0., -1.,  0., 0., 0.,  0., 0., 0., 1.])


def _build_sim_once(_cache={}):
    if "sim" not in _cache:
        from figs.simulator import Simulator
        print("[smoke] Building Simulator (one-time, ~30 s)...")
        _cache["sim"] = Simulator(
            SCENE, "baseline", "carl", gsplats_path=DATA_PATH / "3dgs"
        )
    return _cache["sim"]


def test_dummy_pilot_hovers():
    """DummyVLeadNet → zero velocities → drone holds its initial position."""
    from vlead_flight.pilot import VLeadPilot
    from vlead_flight.network_protocol import DummyVLeadNet

    sim = _build_sim_once()
    pilot = VLeadPilot(
        network=DummyVLeadNet(horizon=10),
        target_xyz=np.array([5., 0., -1.5]),
        gsplat=sim.gsplat,
        frame_name="carl",
    )
    Tro, Xro, Uro, _, _, _ = sim.simulate(pilot, 0.0, 5.0, X0.copy())

    z_drift = abs(Xro[2, -1] - X0[2])
    final_vel = np.abs(Xro[3:6, -10:]).mean(axis=1)
    print(f"  z drift:   {z_drift:.4f} m  (expect < 0.2)")
    print(f"  final vel: {final_vel}  (expect each < 0.2)")
    assert z_drift < 0.2, f"z drift {z_drift:.3f} exceeds 0.2 m"
    assert np.all(final_vel < 0.2), f"settled vel {final_vel}"
    print("  PASS")


def test_recorder_collects_steps():
    """RolloutRecorder should hold one entry per control step."""
    from vlead_flight.pilot import VLeadPilot
    from vlead_flight.recorder import RolloutRecorder
    from vlead_flight.network_protocol import DummyVLeadNet

    sim = _build_sim_once()
    rec = RolloutRecorder()
    pilot = VLeadPilot(
        network=DummyVLeadNet(horizon=10),
        target_xyz=np.array([5., 0., -1.5]),
        gsplat=sim.gsplat,
        frame_name="carl",
        recorder=rec,
    )
    sim.simulate(pilot, 0.0, 2.0, X0.copy())

    expected = int(2.0 * 20)  # duration_s * hz
    print(f"  steps recorded: {len(rec)}  (expect {expected})")
    assert len(rec) == expected, f"got {len(rec)} steps"
    assert rec.steps[0]["vel_pred"].shape == (10, 4), rec.steps[0]["vel_pred"].shape
    assert rec.steps[0]["u_cmd"].shape == (4,)
    print("  PASS")


def test_depth_flag_renders_depth():
    """use_depth=True should populate depth in recorder."""
    from vlead_flight.pilot import VLeadPilot
    from vlead_flight.recorder import RolloutRecorder
    from vlead_flight.network_protocol import DummyVLeadNet

    sim = _build_sim_once()
    rec = RolloutRecorder()
    pilot = VLeadPilot(
        network=DummyVLeadNet(horizon=10),
        target_xyz=np.array([5., 0., -1.5]),
        gsplat=sim.gsplat,
        frame_name="carl",
        use_depth=True,
        recorder=rec,
    )
    sim.simulate(pilot, 0.0, 1.0, X0.copy())

    assert rec.steps[0]["depth"] is not None
    depth = rec.steps[0]["depth"]
    print(f"  depth shape: {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}]")
    assert depth.min() > 0, "depth should be positive metric values"
    print("  PASS")


def test_set_target_updates_goal():
    from vlead_flight.pilot import VLeadPilot
    from vlead_flight.network_protocol import DummyVLeadNet

    pilot = VLeadPilot(
        network=DummyVLeadNet(),
        target_xyz=np.array([1., 2., 3.]),
        gsplat=_build_sim_once().gsplat,
        frame_name="carl",
    )
    assert np.allclose(pilot.target_xyz, [1., 2., 3.])
    pilot.set_target(np.array([7., 8., 9.]))
    assert np.allclose(pilot.target_xyz, [7., 8., 9.])
    print("  PASS")


if __name__ == "__main__":
    tests = [
        ("test_dummy_pilot_hovers",      test_dummy_pilot_hovers),
        ("test_recorder_collects_steps", test_recorder_collects_steps),
        ("test_depth_flag_renders_depth", test_depth_flag_renders_depth),
        ("test_set_target_updates_goal", test_set_target_updates_goal),
    ]
    failures = []
    for name, fn in tests:
        print(f"\n── {name} ──")
        try:
            fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            failures.append(name)

    print("\n" + "=" * 56)
    if failures:
        print(f"  {len(failures)} test(s) FAILED: {failures}")
        sys.exit(1)
    print("  All tests passed.")
