"""
Modal training app for V-LEAD nav_policy.

HOW IT WORKS (Modal 1.x API):
  Code is bundled into the Image using image.add_local_dir() instead of the
  deprecated modal.Mount.  The image is rebuilt when code changes (fast --
  only the add_local_dir layer is invalidated, not the pip-install layers).

SETUP (one-time, on your local Windows machine, NOT inside Docker):
  pip install modal
  modal setup                    # opens browser to authenticate
  modal volume create vlead-data
  modal volume create vlead-raw-data

DATA UPLOAD — Option A: upload already-processed data (if you ran build_dataset locally):
  modal volume put vlead-data "C:\\path\\to\\nav_policy\\data\\processed_flightroom" /processed_flightroom

DATA UPLOAD — Option B: upload raw SINGER data and process in the cloud:
  # 1. Download raw folders from Google Drive (use gdown or browser):
  #    pip install gdown
  #    gdown --folder <FOLDER_ID> -O data/raw/<run_name>
  # 2. Upload raw data to the raw volume:
  #    modal volume put vlead-raw-data "C:\\path\\to\\nav_policy\\data\\raw" /raw
  # 3. Process in cloud (CPU job, no GPU needed):
  #    modal run nav_policy/modal_train.py::build_dataset_remote
  # The processed output lands in vlead-data at /processed_flightroom.

RUNNING TRAINING:
  # BC policy:
  modal run nav_policy/modal_train.py
  modal run nav_policy/modal_train.py --run-tag my_bc_v2

  # Flow Matching policy:
  modal run nav_policy/modal_train.py::main_fm
  modal run nav_policy/modal_train.py::main_fm --run-tag my_fm_v2

DOWNLOADING CHECKPOINTS:
  # BC:
  modal volume get vlead-data checkpoints_flightroom/bc_best.pt nav_policy/data/checkpoints_modal/bc_best.pt
  # FM:
  modal volume get vlead-data checkpoints_flightroom_fm/fm_best.pt nav_policy/data/checkpoints_modal/fm_best.pt

MONITORING:
  modal.com/apps  ->  live logs, GPU utilization, cost per second
  Weights & Biases: https://wandb.ai  (W&B enabled by default in FM config)
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# ── 1. App ─────────────────────────────────────────────────────────────────────
# Groups all Modal objects for this project.  Name appears in the dashboard.
app = modal.App("vlead-bc-training")

NAV_POLICY_DIR = Path(__file__).parent   # the nav_policy/ directory

# ── 2. Container Image ──────────────────────────────────────────────────────────
# Modal builds this once and caches each layer.  Only layers that change are
# rebuilt on subsequent runs.
#
# Layer order matters for caching:
#   1. Base image  (never changes)
#   2. apt_install (rarely changes)
#   3. pip_install (changes only when dependencies change)
#   4. add_local_dir (changes whenever your Python code changes -- fast layer)
#
# add_local_dir(copy=False) is the default: files are injected at container
# startup without baking them into the image layer, which is fastest for
# iterative development.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        add_python="3.10",
    )
    .apt_install(["libgl1", "libglib2.0-0", "ffmpeg"])
    .pip_install(
        "torchvision==0.19.0",
        "imageio[ffmpeg]>=2.34",
        "scipy>=1.13",
        "tqdm",
        "pyyaml",
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "wandb",
    )
    # Inject the nav_policy source code into the container.
    # Excludes the data/ directory (that lives in the Volume).
    .add_local_dir(
        str(NAV_POLICY_DIR),
        remote_path="/workspace/nav_policy",
        ignore=["data/", "__pycache__", "*.egg-info", ".git", "*.pyc"],
    )
)

# ── 3. Persistent Volume ────────────────────────────────────────────────────────
# Cloud disk that survives between runs.  Holds the processed dataset +
# checkpoints written during training.
DATA_VOLUME_NAME = "vlead-data"
VOLUME_MOUNT_PATH = "/data"

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

RAW_VOLUME_NAME = "vlead-raw-data"
raw_volume = modal.Volume.from_name(RAW_VOLUME_NAME, create_if_missing=True)

# ── 4. Training Function ────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A100",               # 40 GB VRAM, ~80 GB RAM, 40 vCPUs
    volumes={VOLUME_MOUNT_PATH: data_volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 4,     # 4-hour hard limit (10 epochs finishes in ~30 min)
)
def train_bc(
    run_tag: str = "flightroom_bc_v1",
    resume_from: str = "",
    checkpoint_dir: str = "",
) -> str:
    """
    Run one complete BC training job in the cloud.

    The function body runs INSIDE the Modal container, not on your machine.
    It calls the exact same train_bc.py script as the local workflow --
    the only difference is the config file, which points paths to /data/.

    Returns the Volume path where bc_best.pt was saved.
    """
    import os
    import subprocess

    # Add the nav_policy src/ to PYTHONPATH so `import nav_policy` works.
    # This is instant -- no pip install needed since the code was injected
    # by add_local_dir above.
    env = {
        **os.environ,
        "PYTHONPATH": "/workspace/nav_policy/src",
    }

    # Confirm the GPU is visible (shows up in the dashboard logs).
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"[modal] GPU: {result.stdout.strip()}", flush=True)

    # Run training.
    cmd = [
        sys.executable, "scripts/train_bc.py",
        "--config", "configs/flightroom_modal.yaml",
        "--run-tag", run_tag,
    ]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    if checkpoint_dir:
        cmd += ["--checkpoint-dir", checkpoint_dir]
    print(f"[modal] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/workspace/nav_policy", env=env)

    # Volumes need an explicit commit() to guarantee writes are persisted
    # before the container exits.
    data_volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"Training exited with code {proc.returncode}")

    out_dir = checkpoint_dir if checkpoint_dir else f"{VOLUME_MOUNT_PATH}/checkpoints_flightroom"
    ckpt_path = f"{out_dir}/bc_best.pt"
    print(f"[modal] Done. Checkpoint at {ckpt_path}", flush=True)
    return ckpt_path


# ── 5. Local Entrypoint (BC) ────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    run_tag: str = "flightroom_bc_v1",
    resume_from: str = "",
    checkpoint_dir: str = "",
):
    """
    Trigger remote BC training and print the checkpoint location.

    Usage:
        modal run nav_policy/modal_train.py
        modal run nav_policy/modal_train.py --run-tag my_run_v2
    """
    print(f"[local] Submitting BC training job  run_tag='{run_tag}' ...")
    ckpt = train_bc.remote(run_tag=run_tag, resume_from=resume_from, checkpoint_dir=checkpoint_dir)
    print(f"[local] Training complete.  Checkpoint: {ckpt}")
    print()
    print("Download with:")
    print(
        f"  modal volume get {DATA_VOLUME_NAME} "
        f"checkpoints_flightroom/bc_best.pt "
        f"nav_policy/data/checkpoints_modal/bc_best.pt"
    )


# ── 6. Flow Matching Training ───────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    volumes={
        VOLUME_MOUNT_PATH: data_volume,
    },
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 6,    # 6-hour limit (FM trains ~2× slower than BC)
)
def train_fm(
    run_tag: str = "flightroom_fm_v1",
    resume_from: str = "",
    checkpoint_dir: str = "",
) -> str:
    """Run OT-CFM FlowMatchingPolicy training in the cloud.

    Returns the Volume path to fm_best.pt.
    """
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": "/workspace/nav_policy/src"}

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"[modal] GPU: {result.stdout.strip()}", flush=True)

    cmd = [
        sys.executable, "scripts/train_fm.py",
        "--config", "configs/flightroom_fm_modal.yaml",
        "--run-tag", run_tag,
    ]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    if checkpoint_dir:
        cmd += ["--checkpoint-dir", checkpoint_dir]
    print(f"[modal] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/workspace/nav_policy", env=env)

    data_volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"FM training exited with code {proc.returncode}")

    out_dir = checkpoint_dir if checkpoint_dir else f"{VOLUME_MOUNT_PATH}/checkpoints_flightroom_fm"
    ckpt_path = f"{out_dir}/fm_best.pt"
    print(f"[modal] Done. Checkpoint at {ckpt_path}", flush=True)
    return ckpt_path


@app.local_entrypoint()
def main_fm(
    run_tag: str = "flightroom_fm_v1",
    resume_from: str = "",
    checkpoint_dir: str = "",
):
    """Trigger remote FM training.

    Usage:
        modal run nav_policy/modal_train.py::main_fm
        modal run nav_policy/modal_train.py::main_fm --run-tag my_fm_v2
    """
    print(f"[local] Submitting FM training job  run_tag='{run_tag}' ...")
    ckpt = train_fm.remote(run_tag=run_tag, resume_from=resume_from, checkpoint_dir=checkpoint_dir)
    print(f"[local] FM training complete.  Checkpoint: {ckpt}")
    print()
    print("Download with:")
    print(
        f"  modal volume get {DATA_VOLUME_NAME} "
        f"checkpoints_flightroom_fm/fm_best.pt "
        f"nav_policy/data/checkpoints_modal/fm_best.pt"
    )


# ── 7. Ablation Training Functions ─────────────────────────────────────────────

def _train_fm_with_config(config_name: str, run_tag: str, ckpt_subdir: str) -> str:
    import os, subprocess
    env = {**os.environ, "PYTHONPATH": "/workspace/nav_policy/src"}
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"[modal] GPU: {result.stdout.strip()}", flush=True)
    cmd = [
        sys.executable, "scripts/train_fm.py",
        "--config", f"configs/{config_name}",
        "--run-tag", run_tag,
    ]
    print(f"[modal] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/workspace/nav_policy", env=env)
    data_volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"FM training exited with code {proc.returncode}")
    ckpt_path = f"{VOLUME_MOUNT_PATH}/{ckpt_subdir}/fm_best.pt"
    print(f"[modal] Done. Checkpoint at {ckpt_path}", flush=True)
    return ckpt_path


@app.function(
    image=image, gpu="A100",
    volumes={VOLUME_MOUNT_PATH: data_volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 6,
)
def train_fm_fulltrajs(run_tag: str = "fulltrajs_v1") -> str:
    """Ablation A: full spawn-to-goal trajectories only (no shuffled clips)."""
    return _train_fm_with_config(
        "flightroom_fm_modal_fulltrajs.yaml", run_tag,
        "checkpoints_flightroom_fm_fulltrajs",
    )


@app.function(
    image=image, gpu="A100",
    volumes={VOLUME_MOUNT_PATH: data_volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=60 * 60 * 6,
)
def train_fm_shuffled(run_tag: str = "shuffled_v1") -> str:
    """Ablation B: shuffled 2-second domain-randomized clips only."""
    return _train_fm_with_config(
        "flightroom_fm_modal_shuffled.yaml", run_tag,
        "checkpoints_flightroom_fm_shuffled",
    )


@app.local_entrypoint()
def main_fm_fulltrajs(run_tag: str = "fulltrajs_v1"):
    """Train ablation A (full trajectories).

    Usage:
        modal run modal_train.py::main_fm_fulltrajs
        modal run modal_train.py::main_fm_fulltrajs --run-tag fulltrajs_v2
    """
    print(f"[local] Submitting fulltrajs FM job  run_tag='{run_tag}' ...")
    ckpt = train_fm_fulltrajs.remote(run_tag=run_tag)
    print(f"[local] Done.  Checkpoint: {ckpt}")
    print(f"  modal volume get {DATA_VOLUME_NAME} checkpoints_flightroom_fm_fulltrajs/fm_best.pt data/checkpoints_modal/fm_fulltrajs_best.pt")


@app.local_entrypoint()
def main_fm_shuffled(run_tag: str = "shuffled_v1"):
    """Train ablation B (shuffled clips).

    Usage:
        modal run modal_train.py::main_fm_shuffled
        modal run modal_train.py::main_fm_shuffled --run-tag shuffled_v2
    """
    print(f"[local] Submitting shuffled FM job  run_tag='{run_tag}' ...")
    ckpt = train_fm_shuffled.remote(run_tag=run_tag)
    print(f"[local] Done.  Checkpoint: {ckpt}")
    print(f"  modal volume get {DATA_VOLUME_NAME} checkpoints_flightroom_fm_shuffled/fm_best.pt data/checkpoints_modal/fm_shuffled_best.pt")


# ── 9. Dataset Build (CPU, raw → processed) ─────────────────────────────────────
# Runs build_dataset_flightroom.py inside Modal using raw data from vlead-raw-data
# volume. Output lands in vlead-data at /processed_flightroom.
# Run with: modal run modal_train.py::build_dataset_remote_local

@app.function(
    image=image,
    cpu=8,                   # CPU-only job; no GPU needed for dataset building
    memory=32768,            # 32 GB RAM for video decoding
    volumes={
        VOLUME_MOUNT_PATH: data_volume,
        "/raw_data": raw_volume,
    },
    timeout=60 * 60 * 4,    # 4-hour limit for large datasets
)
def build_dataset_remote() -> str:
    """Convert raw SINGER data in vlead-raw-data volume → processed caches in vlead-data.

    Raw data layout expected in vlead-raw-data:/raw/:
        <run_name>/trajectories_val{NNNNN}.pt
        <run_name>/video_val_rollout_images_rgb{NNNNN}.mp4
        <run_name>/imgdata_val{NNNNN}.pt

    Output written to vlead-data:/processed_flightroom/.
    """
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": "/workspace/nav_policy/src"}

    # Symlink raw data into expected location relative to nav_policy root.
    raw_src = "/raw_data/raw"
    raw_dst = "/workspace/nav_policy/data/raw"
    os.makedirs("/workspace/nav_policy/data", exist_ok=True)
    if not os.path.exists(raw_dst):
        os.symlink(raw_src, raw_dst)

    proc_dst = "/workspace/nav_policy/data/processed_flightroom"
    if not os.path.exists(proc_dst):
        os.symlink(f"{VOLUME_MOUNT_PATH}/processed_flightroom", proc_dst)

    cmd = [
        sys.executable, "scripts/build_dataset_flightroom.py",
        "--config", "configs/flightroom_fm_modal.yaml",
    ]
    print(f"[modal] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/workspace/nav_policy", env=env)

    data_volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"build_dataset exited with code {proc.returncode}")

    out = f"{VOLUME_MOUNT_PATH}/processed_flightroom"
    print(f"[modal] Dataset built at {out}", flush=True)
    return out


@app.local_entrypoint()
def build_dataset_remote_local():
    """Trigger remote dataset build.

    Usage:
        modal run nav_policy/modal_train.py::build_dataset_remote_local
    """
    print("[local] Submitting dataset build job ...")
    out = build_dataset_remote.remote()
    print(f"[local] Dataset built at: {out}")


# ── 10. Offline Evaluation (CPU, no FiGS) ──────────────────────────────────────
# Runs eval_offline.py on the Modal volume using the processed cache + a checkpoint.
# Reports MSE metrics broken down by velocity component and horizon step.
# The 071733 held-out test run is used by default (via eval_offline_flightroom_fm.yaml).

@app.function(
    image=image,
    cpu=4,
    memory=16384,
    volumes={VOLUME_MOUNT_PATH: data_volume},
    timeout=60 * 60 * 1,    # offline eval typically takes < 10 minutes
)
def eval_fm_offline_remote(
    checkpoint_subdir: str = "checkpoints_flightroom_fm",
    output_subdir: str = "eval_fm_offline",
) -> str:
    """Run offline (no-FiGS) evaluation of the FM checkpoint on the test split.

    Evaluates on the 071733 held-out run (as configured in
    eval_offline_flightroom_fm.yaml). Writes summary.json + per_horizon.csv
    + predictions.npz to the vlead-data volume.
    """
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": "/workspace/nav_policy/src"}

    # Symlink processed data into the path the config expects.
    proc_dst = "/workspace/nav_policy/data/processed_flightroom"
    os.makedirs("/workspace/nav_policy/data", exist_ok=True)
    if not os.path.exists(proc_dst):
        os.symlink(f"{VOLUME_MOUNT_PATH}/processed_flightroom", proc_dst)

    ckpt_path = f"{VOLUME_MOUNT_PATH}/{checkpoint_subdir}/fm_best.pt"
    out_dir = f"{VOLUME_MOUNT_PATH}/{output_subdir}"

    cmd = [
        sys.executable, "scripts/eval_offline.py",
        "--config", "configs/eval_offline_flightroom_fm.yaml",
        "--checkpoint", ckpt_path,
        "--output-dir", out_dir,
    ]
    print(f"[modal] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/workspace/nav_policy", env=env)

    data_volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"eval_offline exited with code {proc.returncode}")

    print(f"[modal] Eval output at {out_dir}", flush=True)
    return out_dir


@app.local_entrypoint()
def main_eval_fm_offline(
    checkpoint_subdir: str = "checkpoints_flightroom_fm",
    output_subdir: str = "eval_fm_offline",
):
    """Run offline FM evaluation on Modal.

    Usage:
        modal run modal_train.py::main_eval_fm_offline
        modal run modal_train.py::main_eval_fm_offline --checkpoint-subdir checkpoints_flightroom_fm_fulltrajs --output-subdir eval_fulltrajs_offline
    """
    print(f"[local] Submitting offline eval  checkpoint={checkpoint_subdir} ...")
    out = eval_fm_offline_remote.remote(
        checkpoint_subdir=checkpoint_subdir,
        output_subdir=output_subdir,
    )
    print(f"[local] Eval complete. Results at: {out}")
    print("\nDownload summary:")
    print(f"  modal volume get {DATA_VOLUME_NAME} {output_subdir}/summary.json data/eval/fm_offline_summary.json")
    print(f"  modal volume get {DATA_VOLUME_NAME} {output_subdir}/per_horizon.csv data/eval/fm_per_horizon.csv")
