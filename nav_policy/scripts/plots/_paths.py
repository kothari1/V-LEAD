"""Shared paths for plot scripts. Resolved relative to repo root so the
snapshot is drop-in usable on any machine that has a clone of V-LEAD/."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
EVAL_DIR = LOGS_DIR / "eval"
TRAINING_DIR = LOGS_DIR / "training"

FULL110_DIR = EVAL_DIR / "full-110"
HOLDOUT14_DIR = EVAL_DIR / "holdout-14"
PHASE_A_DIR = EVAL_DIR / "phase-A-175"
