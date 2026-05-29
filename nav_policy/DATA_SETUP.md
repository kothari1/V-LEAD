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

## Step 3 — Upload to Modal and build dataset

All processing runs in the cloud — no Docker or local GPU needed.

```bash
# One-time setup:
pip install modal
modal setup
modal volume create vlead-data
modal volume create vlead-raw-data
modal secret create wandb WANDB_API_KEY=<your_key_from_wandb.ai/settings>

# Upload raw data (run from V-LEAD/nav_policy/):
modal volume put vlead-raw-data data/raw /raw

# Build processed dataset in the cloud (CPU job, ~30–60 min):
modal run --detach modal_train.py::build_dataset_remote_local
```

Wait for the `✓ Initialized. View run at https://modal.com/apps/...` line before closing the terminal. Monitor progress at modal.com/apps.

The build writes per-trajectory cache files + `manifest.json` + `stats.json` to the `vlead-data` volume. **If the job is interrupted**, just rerun the same command — it skips already-processed bundles instantly.

---

## Step 4 — Train

Once the dataset build completes, launch training. You can run all three in parallel:

```bash
# Combined (all 3 train folders + val, 10 epochs):
modal run --detach modal_train.py::main_fm

# Shuffled-clips ablation (2-sec clips only, 15 epochs, no val):
modal run --detach modal_train.py::main_fm_shuffled

# Full-trajs ablation (spawn-to-goal only, 15 epochs, no val):
modal run --detach modal_train.py::main_fm_fulltrajs
```

Wait for the `✓ Initialized` URL line before closing each terminal.

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
| One-time Modal setup | `modal setup && modal volume create vlead-data && modal volume create vlead-raw-data` |
| Create W&B secret | `modal secret create wandb WANDB_API_KEY=<key>` |
| Upload raw data | `modal volume put vlead-raw-data data/raw /raw` |
| Build dataset (cloud) | `modal run --detach modal_train.py::build_dataset_remote_local` |
| Train FM — combined | `modal run --detach modal_train.py::main_fm` |
| Train FM — shuffled | `modal run --detach modal_train.py::main_fm_shuffled` |
| Train FM — full trajs | `modal run --detach modal_train.py::main_fm_fulltrajs` |
| Monitor | https://modal.com/apps · https://wandb.ai |
| Download checkpoint (combined) | `modal volume get vlead-data checkpoints_flightroom_fm/fm_best.pt data/checkpoints_modal/fm_best.pt` |
| Download checkpoint (shuffled) | `modal volume get vlead-data checkpoints_flightroom_fm_shuffled/fm_best.pt data/checkpoints_modal/fm_shuffled_best.pt` |
| Download checkpoint (fulltrajs) | `modal volume get vlead-data checkpoints_flightroom_fm_fulltrajs/fm_best.pt data/checkpoints_modal/fm_fulltrajs_best.pt` |
| SAC fine-tuning | `python scripts/train_sac.py --config configs/sac_fm.yaml` |
