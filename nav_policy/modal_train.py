"""
Modal training app for V-LEAD nav_policy.

WHAT IS MODAL?
--------------
Modal is a cloud platform where you write ordinary Python functions decorated
with @app.function().  When you run `modal run modal_train.py`, Modal:
  1. Builds a container image with your specified dependencies
  2. Uploads your local code to that container
  3. Allocates a GPU machine in the cloud
  4. Runs your function on it
  5. Returns results / saves files to a persistent Volume

There is no Kubernetes, no Dockerfile to manage, no SSH.  Everything is code.

KEY MODAL CONCEPTS (each is one Python object):
  modal.App    - the application; groups related functions together
  modal.Image  - the container environment (OS packages + pip installs)
  modal.Volume - persistent cloud storage; survives between runs
  modal.Mount  - mounts local files/dirs into the container at runtime
  @app.function(...) - turns a Python function into a cloud function

SETUP (one-time, on your local Windows machine, NOT inside Docker):
  pip install modal
  modal setup              # opens browser to authenticate

DATA UPLOAD (one-time, after building the processed dataset):
  modal volume create vlead-data
  # From the directory containing nav_policy/:
  modal volume put vlead-data nav_policy/data/processed_flightroom /processed_flightroom

RUNNING TRAINING:
  modal run nav_policy/modal_train.py
  modal run nav_policy/modal_train.py::train_bc --run-tag my_experiment

DOWNLOADING THE CHECKPOINT AFTER TRAINING:
  modal volume get vlead-data checkpoints_flightroom/bc_best.pt \
      nav_policy/data/checkpoints_modal/bc_best.pt

MONITORING:
  modal.com/dashboard  ->  see live logs, GPU utilization, cost
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# ── 1. App ────────────────────────────────────────────────────────────────────
# An App groups all Modal objects (functions, volumes, images) for this project.
# The name appears on the Modal dashboard.
app = modal.App("vlead-bc-training")

# ── 2. Container Image ────────────────────────────────────────────────────────
# Modal builds this image once and caches it.  Subsequent runs reuse the cache
# unless you change the image definition.
#
# We start from the official PyTorch + CUDA 12.4 image (has torch pre-installed
# with GPU support) and layer our additional dependencies on top.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        add_python="3.10",
    )
    # System packages needed by OpenCV and imageio/ffmpeg
    .apt_install(["libgl1", "libglib2.0-0", "ffmpeg"])
    # Python packages nav_policy needs (torch/torchvision are already in the base)
    .pip_install(
        "torchvision==0.19.0",
        "imageio[ffmpeg]>=2.34",
        "scipy>=1.13",
        "tqdm",
        "pyyaml",
        "numpy",
        "opencv-python-headless",
        "Pillow",
    )
)

# ── 3. Persistent Volume ───────────────────────────────────────────────────────
# A Volume is cloud disk that persists between runs (unlike the container
# filesystem which is wiped after each run).
#
# We use one volume for everything: processed dataset + output checkpoints.
# The volume is mounted at /data inside the container.
#
# create_if_missing=True means `modal run` will create it automatically if it
# doesn't exist yet.  After that, use the CLI to put/get files.
DATA_VOLUME_NAME = "vlead-data"
VOLUME_MOUNT_PATH = "/data"

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

# ── 4. Code Mount ─────────────────────────────────────────────────────────────
# A Mount copies local files into the container at runtime.  This means you
# don't need to rebuild the image every time you edit your Python code.
#
# We mount the entire nav_policy/ directory to /workspace/nav_policy.
# The heavy data/ subdirectory is excluded (it lives in the Volume).
NAV_POLICY_DIR = Path(__file__).parent  # the nav_policy/ directory

code_mount = modal.Mount.from_local_dir(
    local_path=NAV_POLICY_DIR,
    remote_path="/workspace/nav_policy",
    # Exclude directories that should not be copied (data is in the Volume;
    # __pycache__ and .git are noise).
    condition=lambda rel_path: not any(
        part in rel_path
        for part in ["/data/", "__pycache__", ".git", ".egg-info", "*.pyc"]
    ),
)

# ── 5. Training Function ───────────────────────────────────────────────────────
# @app.function turns an ordinary Python function into a cloud function.
# Every keyword argument here is Modal configuration:
#
#   image=        which container to run in
#   gpu=          which GPU to allocate (A100 = 40 GB VRAM, ~80 GB RAM, 40 vCPUs)
#   volumes=      {mount_path: volume} — mounts the Volume inside the container
#   mounts=       local code to copy in at runtime
#   timeout=      hard wall-clock limit in seconds (here: 4 hours)
#
# The function body runs INSIDE the cloud container, not on your machine.
@app.function(
    image=image,
    gpu="A100",          # change to "H100" for ~2x faster training at higher cost
    volumes={VOLUME_MOUNT_PATH: data_volume},
    mounts=[code_mount],
    timeout=60 * 60 * 4,  # 4-hour hard limit; 10 epochs should finish in ~30 min
)
def train_bc(run_tag: str = "flightroom_bc_v1") -> str:
    """
    Run one complete BC training job in the cloud.

    Args:
        run_tag: Label stamped into the saved checkpoint and summary.json.
                 Use a different tag for each experiment.

    Returns:
        Path (inside the Volume) where bc_best.pt was saved.
    """
    import os
    import subprocess

    # ── a. Make nav_policy importable ──────────────────────────────────────
    # We didn't bake the package into the image (it changes with every code
    # edit), so we add its src/ directory to PYTHONPATH at runtime instead.
    # This is instant (no pip install needed).
    env = {
        **os.environ,
        "PYTHONPATH": "/workspace/nav_policy/src",
    }

    # ── b. Confirm GPU is visible ───────────────────────────────────────────
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"[modal] GPU: {result.stdout.strip()}", flush=True)

    # ── c. Run training ────────────────────────────────────────────────────
    # This calls the exact same train_bc.py script as the local workflow.
    # The only difference is the config file, which points paths to /data/.
    cmd = [
        sys.executable, "scripts/train_bc.py",
        "--config", "configs/flightroom_modal.yaml",
        "--run-tag", run_tag,
    ]
    print(f"[modal] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/workspace/nav_policy", env=env)

    # ── d. Persist Volume writes ───────────────────────────────────────────
    # Modal Volumes need an explicit .commit() call to guarantee that writes
    # made during the function are visible to future runs.
    data_volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"Training exited with code {proc.returncode}")

    ckpt_path = f"{VOLUME_MOUNT_PATH}/checkpoints_flightroom/bc_best.pt"
    print(f"[modal] Done. Checkpoint at {ckpt_path}", flush=True)
    return ckpt_path


# ── 6. Local entrypoint ────────────────────────────────────────────────────────
# @app.local_entrypoint runs on your local machine when you do `modal run`.
# It is the glue between the CLI and the remote function.
@app.local_entrypoint()
def main(run_tag: str = "flightroom_bc_v1"):
    """
    Trigger remote training and print the checkpoint location.

    Usage:
        modal run nav_policy/modal_train.py
        modal run nav_policy/modal_train.py --run-tag my_run_v2

    After training completes, download the checkpoint with:
        modal volume get vlead-data checkpoints_flightroom/bc_best.pt \\
            nav_policy/data/checkpoints_modal/bc_best.pt
    """
    print(f"[local] Submitting training job with run_tag='{run_tag}' ...")
    ckpt = train_bc.remote(run_tag=run_tag)
    print(f"[local] Training complete. Checkpoint saved at: {ckpt}")
    print()
    print("To download the checkpoint, run:")
    print(
        f"  modal volume get {DATA_VOLUME_NAME} "
        f"checkpoints_flightroom/bc_best.pt "
        f"nav_policy/data/checkpoints_modal/bc_best.pt"
    )
