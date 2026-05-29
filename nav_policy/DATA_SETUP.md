# nav_policy — Data Setup Guide

How to get the V-LEAD flightroom dataset from Google Drive into training.

---

## Google Drive folders and their roles

**Drive root:** https://drive.google.com/drive/u/1/folders/12e3-0i52MUwcUrF2ewhs0jBQXhfxqltW

There are three training configs, each using the folders differently:

| Folder name | `main_fm` (combined) | `main_fm_shuffled` | `main_fm_fulltrajs` | Description |
|---|:---:|:---:|:---:|---|
| `flightroom_ssv_exp_2026-05-22_064652_training_mode_shuffled_trajs` | Train | **Train** | — | 2-sec domain-randomized clips (~550 bundles) |
| `flightroom_ssv_exp_2026-05-22_071718` | Train | — | **Train** | Full spawn-to-goal trajectories (~220 bundles) |
| `flightroom_ssv_exp_2026-05-22_071353` | Train | — | **Train** | Same as above, second generation run |
| `flightroom_ssv_exp_2026-05-22_071733_trajs-110` | Val | Test | Test | 110 full trajectories, held out (no domain randomization) |

**Download all 4 folders** — they are shared across all three configs.

- **Combined (`main_fm`):** trains on all 3 folders, uses `071733` as val for early stopping.
- **Shuffled ablation (`main_fm_shuffled`):** trains on the 2-sec clips only, 15 fixed epochs, no val.
- **Full-trajs ablation (`main_fm_fulltrajs`):** trains on full trajectories only, 15 fixed epochs, no val.

---

## Step 1 — Download from Chrome

Google Drive lets you download an entire folder as a `.zip`.

For each of the 4 folders listed in the table above:
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

## Step 2 — Unzip and flatten

Google Drive zips an extra outer folder into the archive named after the download timestamp.  
You need to extract and then promote the inner folder one level up.

**Extract each zip** to any temp location (e.g. your Downloads folder).  
You will see a structure like:
```
Downloads/
└── 2026-05-22_064652_training_mode_shuffled_trajs/          ← outer wrapper (from zip)
    └── flightroom_ssv_exp_2026-05-22_064652_training_mode_shuffled_trajs/   ← actual data
        ├── trajectories00000.pt
        ├── video_rgb00000.mp4
        ├── video_depth00000.mp4
        ├── video_semantic00000.mp4
        ├── imgdata00000.pt
        └── ...
```

**Move the inner folder** (the one starting with `flightroom_ssv_exp_`) directly into `nav_policy/data/raw/`.  
Discard the outer wrapper folder.

**PowerShell shortcut** (run from `nav_policy/data/raw/`, repeat for each zip):
```powershell
# After extracting a zip into raw/, run this to flatten it:
# Replace <outer> with the wrapper folder name (e.g. 2026-05-22_064652_...)
Move-Item "<outer>\flightroom_ssv_exp_*" .
Remove-Item "<outer>"
```

**Target structure** (what `data/raw/` should look like when done):
```
nav_policy/
└── data/
    └── raw/
        ├── flightroom_ssv_exp_2026-05-22_064652_training_mode_shuffled_trajs/
        │   ├── trajectories00000.pt          (train-mode naming)
        │   ├── video_rgb00000.mp4
        │   └── ...  (~550 bundles × 5 file types = ~2750 files)
        ├── flightroom_ssv_exp_2026-05-22_071718/
        │   ├── trajectories_val00000.pt      (val-mode naming)
        │   ├── video_val_rollout_images_rgb00000.mp4
        │   └── ...  (~220 bundles × 5 file types = ~1100 files)
        ├── flightroom_ssv_exp_2026-05-22_071353/
        │   └── ...  (~1100 files)
        └── flightroom_ssv_exp_2026-05-22_071733_trajs-110/
            ├── trajectories_val00000.pt
            ├── video_val_rollout_images_rgb00000.mp4
            └── ...  (~110 bundles × 5 file types = ~550 files)
```

> `build_dataset_flightroom.py` auto-detects file naming (`trajectories_val*.pt` vs `trajectories*.pt`) — no renaming needed.

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

## Step 5 — Monitor training

**Modal dashboard:** https://modal.com/apps — live logs, GPU utilisation, cost per run.

**Weights & Biases:** https://wandb.ai — loss curves, val MSE per component. Project name is `vlead-fm`.

Training prints a summary line each epoch:
```
[epoch   1]  train_fm=0.8420  val_fm=0.9103  val_mse_lin=0.0412  val_mse_psi=0.0089  sec=187.3
  -> saved fm_best.pt (val_mse_overall=0.0326)
```
`val_mse_overall` is the headline metric (lower = better). A good FM seed for RL is typically below **0.03**.

For the ablation configs (`main_fm_fulltrajs`, `main_fm_shuffled`) there is no val set — the line reads `[no val]` and best is tracked by train loss instead.

---

## Step 6 — Download checkpoints

Run these from `V-LEAD/nav_policy/` after training completes:

```bash
mkdir -p data/checkpoints_modal

# Combined FM (all 3 train folders):
modal volume get vlead-data checkpoints_flightroom_fm/fm_best.pt \
    data/checkpoints_modal/fm_best.pt

# Ablation A — full trajectories:
modal volume get vlead-data checkpoints_flightroom_fm_fulltrajs/fm_best.pt \
    data/checkpoints_modal/fm_fulltrajs_best.pt

# Ablation B — shuffled clips:
modal volume get vlead-data checkpoints_flightroom_fm_shuffled/fm_best.pt \
    data/checkpoints_modal/fm_shuffled_best.pt
```

To download the training log CSV (epoch-by-epoch loss history):
```bash
modal volume get vlead-data checkpoints_flightroom_fm/log.csv \
    data/checkpoints_modal/fm_log.csv
```

---

## Step 7 — What comes next

### Inspect the checkpoint
```python
import torch
ckpt = torch.load("data/checkpoints_modal/fm_best.pt", weights_only=False)
print(ckpt["epoch"], ckpt["val_mse_overall"])   # epoch number + best val MSE
```

### SAC fine-tuning (RL)
Use the FM checkpoint as a warm start for SAC inside FiGS:
```bash
# Edit configs/sac_fm.yaml to set warm_start.checkpoint_path to your downloaded .pt
# Then (inside SINGER container with FiGS available):
python scripts/train_sac.py --config configs/sac_fm.yaml
```
The FM visual encoder is frozen and loaded into `BCEncoderFeatureExtractor`; the SAC actor/critic train from scratch on top of it.

### Closed-loop evaluation
```bash
# Inside SINGER container:
python scripts/eval_closed_loop.py --config configs/eval_closed_loop_test.yaml \
    --checkpoint data/checkpoints_modal/fm_best.pt
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
| Download data | Chrome → right-click Drive folder → Download |
| Unzip + flatten | Extract to `nav_policy/data/raw/<folder_name>/` |
| Upload raw to Modal | `modal volume put vlead-raw-data data/raw /raw` |
| Build dataset (cloud) | `modal run --detach modal_train.py::build_dataset_remote_local` |
| Train FM — combined | `modal run --detach modal_train.py::main_fm` |
| Train FM — full trajs | `modal run --detach modal_train.py::main_fm_fulltrajs` |
| Train FM — shuffled | `modal run --detach modal_train.py::main_fm_shuffled` |
| Monitor | https://modal.com/apps · https://wandb.ai |
| Download checkpoint | `modal volume get vlead-data checkpoints_flightroom_fm/fm_best.pt data/checkpoints_modal/fm_best.pt` |
| SAC fine-tuning | `python scripts/train_sac.py --config configs/sac_fm.yaml` |
