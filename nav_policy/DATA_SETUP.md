# nav_policy — Data Setup Guide

How to get the V-LEAD flightroom dataset from Google Drive into training.

---

## Google Drive folders and their roles

**Drive root:** https://drive.google.com/drive/u/1/folders/12e3-0i52MUwcUrF2ewhs0jBQXhfxqltW

| Folder name | Role | Description |
|---|---|---|
| `flightroom_ssv_exp_2026-05-22_064652_training_mode_shuffled_trajs` | **Train** | 2-second trajectory segments, domain-randomized (mass/force perturbations), pre-shuffled. **Most important training folder.** |
| `flightroom_ssv_exp_2026-05-22_071718` | **Train** | Full spawn-to-goal trajectories used as additional training data |
| `flightroom_ssv_exp_2026-05-22_071353` | **Train** | Same as above (second generation run) |
| `flightroom_ssv_exp_2026-05-22_071733_trajs-110` | **Val** | 110 full trajectories (no domain randomization). Held out — used only for validation MSE. |
| `backroom_exp_04-29-26` | **Test** | Different scene. **Not used for training or validation.** Reserved for closed-loop eval in FiGS. |
| `packardpark_exp_04-29-26` | **Test** | Different scene. Same as above. |

You only need to download the 4 **train** and **val** folders for training. The 2 test folders are optional (closed-loop eval only).

---

## Step 1 — Download from Chrome

Google Drive lets you download an entire folder as a `.zip`.

For each of the 4 folders (3 train + 1 val):
1. Open Google Drive in Chrome and navigate into the folder
2. Click the folder name in the breadcrumb to select it, or go up one level and right-click the folder
3. Choose **Download** — Chrome will download a `.zip` file named after the folder

> **Note:** Google Drive may split large folders into multiple zip parts (e.g. `folder_name.zip`, `folder_name (1).zip`). If this happens, extract all parts — they contain different files from the same folder.

You will end up with files like:
```
flightroom_ssv_exp_2026-05-22_064652_training_mode_shuffled_trajs.zip
flightroom_ssv_exp_2026-05-22_071718.zip
flightroom_ssv_exp_2026-05-22_071353.zip
flightroom_ssv_exp_2026-05-22_071733_trajs-110.zip
```

---

## Step 2 — Unzip to the correct location

Unzip each archive so the folder lands directly under `nav_policy/data/raw/`:

```
nav_policy/
└── data/
    └── raw/
        ├── flightroom_ssv_exp_2026-05-22_064652_training_mode_shuffled_trajs/
        │   ├── trajectories00000.pt
        │   ├── video_rgb00000.mp4
        │   └── ...
        ├── flightroom_ssv_exp_2026-05-22_071718/
        │   ├── trajectories_val00000.pt
        │   ├── video_val_rollout_images_rgb00000.mp4
        │   └── ...
        ├── flightroom_ssv_exp_2026-05-22_071353/
        │   └── ...
        └── flightroom_ssv_exp_2026-05-22_071733_trajs-110/
            └── ...
```

> `build_dataset_flightroom.py` auto-detects the file naming (`trajectories_val*.pt` vs `trajectories*.pt`) so no renaming is needed.

---

## Step 3 — Build the processed dataset

The training scripts do not read raw `.pt` + `.mp4` files directly — they use pre-built per-trajectory cache files. This step runs once and takes ~20–40 minutes depending on CPU speed.

**Run inside the nav_policy Docker container** (needed for ffmpeg + scipy on Windows):

```bash
cd nav_policy
docker compose run --rm nav_policy
# Inside the container:
python scripts/build_dataset_flightroom.py --config configs/flightroom_fm.yaml
```

Output lands at `nav_policy/data/processed_flightroom/`:
```
processed_flightroom/
├── manifest.json        ← window index (train/val split, T, H)
├── stats.json           ← CommandStats (mean/std for z-scoring)
├── flightroom_ssv_exp_2026-05-22_064652_training_mode_shuffled_trajs/cache/*.pt
├── flightroom_ssv_exp_2026-05-22_071718/cache/*.pt
├── flightroom_ssv_exp_2026-05-22_071353/cache/*.pt
└── flightroom_ssv_exp_2026-05-22_071733_trajs-110/cache/*.pt
```

**Verify:**
```bash
python scripts/inspect_manifest.py data/processed_flightroom/manifest.json
```
Expected output shows `splits = {'train': ~N, 'val': ~M}` with non-zero counts for both.

---

## Step 4 — Upload to Modal

### Path A: upload processed data (recommended)

The processed directory has no videos — it's much smaller than the raw data and uploads quickly.

```bash
# One-time setup (if not already done):
pip install modal
modal setup
modal volume create vlead-data

# Upload processed data (run from V-LEAD/nav_policy/ on your local machine):
modal volume put vlead-data data/processed_flightroom /processed_flightroom
```

The upload progress is shown in the terminal. For ~10k cache files this typically takes 5–15 minutes depending on your connection.

**Then train:**
```bash
# BC policy:
modal run modal_train.py

# Flow Matching policy (recommended — higher quality BC seed):
modal run modal_train.py::main_fm

# Override run name for W&B:
modal run modal_train.py::main_fm --run-tag my_fm_v1
```

**Download checkpoint after training:**
```bash
modal volume get vlead-data checkpoints_flightroom_fm/fm_best.pt data/checkpoints_modal/fm_best.pt
```

---

### Path B: upload raw data and build in the cloud (no local Docker needed)

Use this if you cannot run Docker locally. Builds take longer and require uploading videos.

```bash
# Create the raw data volume (one-time):
modal volume create vlead-raw-data

# Upload raw data:
modal volume put vlead-raw-data data/raw /raw

# Process in cloud (CPU job, no GPU cost):
modal run modal_train.py::build_dataset_remote_local

# Then train:
modal run modal_train.py::main_fm
```

---

## Sherlock (Stanford HPC) alternative

If you prefer Sherlock over Modal:

1. Copy the raw data to Sherlock scratch (use `scp` or `rsync` from your local machine after unzipping):
   ```bash
   rsync -avz data/raw/ <sunetid>@login.sherlock.stanford.edu:/scratch/users/<sunetid>/vlead_data/raw/
   ```

2. Update the paths in `configs/flightroom_fm_sherlock.yaml` (replace `YOUR_SUNETID`).

3. Submit the preprocessing job (CPU, no GPU):
   ```bash
   ssh <sunetid>@login.sherlock.stanford.edu
   cd ~/V-LEAD/nav_policy
   sbatch scripts/sherlock_build_dataset.sbatch
   ```

4. After it completes, submit training:
   ```bash
   sbatch scripts/sherlock_train_fm.sbatch
   ```

Logs appear in `logs/train_fm_<jobid>.out`. W&B logging is enabled by default (set `WANDB_API_KEY` in your shell before submitting).

---

## Quick reference

| Step | Command |
|---|---|
| Download | Chrome → right-click folder → Download |
| Unzip | Extract to `nav_policy/data/raw/<folder_name>/` |
| Build dataset | `python scripts/build_dataset_flightroom.py --config configs/flightroom_fm.yaml` |
| Verify | `python scripts/inspect_manifest.py data/processed_flightroom/manifest.json` |
| Upload to Modal | `modal volume put vlead-data data/processed_flightroom /processed_flightroom` |
| Train FM (Modal) | `modal run modal_train.py::main_fm` |
| Download checkpoint | `modal volume get vlead-data checkpoints_flightroom_fm/fm_best.pt data/checkpoints_modal/fm_best.pt` |
