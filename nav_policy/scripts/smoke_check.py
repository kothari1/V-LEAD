#!/usr/bin/env python3
"""Quick environment smoke test for the no-Docker / local-venv setup.

Verifies the local virtualenv is wired up correctly WITHOUT needing any data,
checkpoints, or 3DGS scenes. Prints a PASS/FAIL/SKIP table and exits non-zero
if any REQUIRED check fails (OPTIONAL checks only warn).

Usage:
    cd nav_policy
    python scripts/smoke_check.py
"""
from __future__ import annotations

import importlib
import sys
import traceback

# ANSI colors (fall back to plain text if not a tty)
_TTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s
GREEN = lambda s: _c("32", s)
RED = lambda s: _c("31", s)
YEL = lambda s: _c("33", s)

results: list[tuple[str, str, str]] = []  # (status, name, detail)
_required_failed = False


def check(name: str, fn, *, required: bool = True) -> None:
    """Run `fn`; record PASS / FAIL (required) / SKIP (optional)."""
    global _required_failed
    try:
        detail = fn() or ""
        results.append(("PASS", name, str(detail)))
    except Exception as e:  # noqa: BLE001 - we want to catch everything
        detail = f"{type(e).__name__}: {e}"
        if required:
            _required_failed = True
            results.append(("FAIL", name, detail))
        else:
            results.append(("SKIP", name, detail))


def _import(mod: str):
    def _fn():
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "") or getattr(m, "__file__", "")
    return _fn


# ---- Required: interpreter + core stack -----------------------------------
def _py_version():
    v = sys.version_info
    if v[:2] != (3, 10):
        raise RuntimeError(f"expected Python 3.10, got {v.major}.{v.minor}.{v.micro}")
    return f"{v.major}.{v.minor}.{v.micro}"
check("python 3.10", _py_version)

def _no_ros_leak():
    leaked = [p for p in sys.path if "/opt/ros/" in p]
    if leaked:
        raise RuntimeError(
            f"ROS path on sys.path (run `unset PYTHONPATH`): {leaked[0]}"
        )
    return "clean"
# Warning-only: a ROS leak is a latent footgun (py3.12 ext modules on a 3.10
# path) but doesn't necessarily break nav_policy, so don't hard-fail on it.
check("no ROS PYTHONPATH leak", _no_ros_leak, required=False)

check("torch", _import("torch"))
check("torchvision", _import("torchvision"))

def _cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() == False")
    return f"{torch.version.cuda} / {torch.cuda.get_device_name(0)}"
check("CUDA available", _cuda)

# ---- Required: nav_policy package -----------------------------------------
check("nav_policy", _import("nav_policy"))

def _augmentations():
    import torch
    from nav_policy.data.augmentations import (
        apply_depth_blur, apply_rgb_augmentations, window_with_latency,
    )
    frames = torch.randint(0, 256, (10, 3, 32, 32), dtype=torch.uint8)
    w = window_with_latency(frames, k=5, T=4, latency=1)
    assert w.shape == (4, 3, 32, 32) and w.dtype == torch.uint8, (w.shape, w.dtype)
    aug = apply_rgb_augmentations(w, blur_prob=1.0, gamma_prob=1.0, brightness_prob=1.0)
    assert aug.shape == w.shape and aug.dtype == torch.uint8
    d = apply_depth_blur(torch.rand(4, 1, 32, 32), 1.0)
    assert d.shape == (4, 1, 32, 32)
    return "window+rgb+depth aug OK"
check("data.augmentations", _augmentations)

def _model_forward():
    import torch
    from nav_policy.model.rgb_velocity_policy import RGBVelocityPolicy, count_parameters
    m = RGBVelocityPolicy(T=4, H=10, cmd_dim=4, goal_input_dim=2)
    m.eval()
    rgb = torch.zeros(2, 4, 3, 224, 224)        # [B, T, 3, S, S]
    goal = torch.zeros(2, 2)                     # [B, goal_input_dim]
    with torch.no_grad():
        out = m(rgb, goal)
    assert tuple(out.shape) == (2, 10, 4), out.shape
    return f"rgb[2,4,3,224,224]+goal[2,2] -> {tuple(out.shape)}, {count_parameters(m):,} params"
check("model forward (RGBVelocityPolicy)", _model_forward)

# ---- Required: RL stack ----------------------------------------------------
check("gymnasium", _import("gymnasium"))
check("stable_baselines3", _import("stable_baselines3"))

def _rl_policy():
    import nav_policy.rl.stochastic_policy  # noqa: F401
    return "rl.stochastic_policy imports"
check("nav_policy.rl", _rl_policy)

# ---- Required: media -------------------------------------------------------
check("opencv (cv2)", _import("cv2"))
check("imageio", _import("imageio"))
check("av (PyAV)", _import("av"))

# ---- Required-for-sim: FiGS ------------------------------------------------
def _figs():
    import figs  # noqa: F401
    from figs.control.velocity_controller import VelocityController  # noqa: F401
    return "figs + VelocityController"
check("figs (simulator)", _figs)

# ---- Optional: 3DGS rendering stack (needed for closed-loop / RL rollouts) -
check("nerfstudio (3DGS)", _import("nerfstudio"), required=False)
check("gsplat (3DGS)", _import("gsplat"), required=False)

# ---- Report ----------------------------------------------------------------
print()
name_w = max(len(n) for _, n, _ in results)
for status, name, detail in results:
    tag = {"PASS": GREEN("PASS"), "FAIL": RED("FAIL"), "SKIP": YEL("SKIP")}[status]
    print(f"  [{tag}] {name.ljust(name_w)}  {detail}")

n_pass = sum(s == "PASS" for s, _, _ in results)
n_fail = sum(s == "FAIL" for s, _, _ in results)
n_skip = sum(s == "SKIP" for s, _, _ in results)
print()
print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped (optional)")

if _required_failed:
    print(RED("\n  SMOKE TEST FAILED — fix the FAIL rows above before running the pipeline."))
    sys.exit(1)

if n_skip:
    print(YEL("\n  Core OK. Optional 3DGS deps missing — fine for BC/offline work; "
              "install root requirements.txt for closed-loop eval / RL rollouts."))
else:
    print(GREEN("\n  ALL CHECKS PASSED — environment is ready."))
sys.exit(0)
