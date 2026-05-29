#!/bin/bash
#SBATCH --job-name=vlead_build_dataset
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_dataset_%j.out
#SBATCH --error=logs/build_dataset_%j.err
#
# USAGE:
#   cd $HOME/V-LEAD/nav_policy
#   mkdir -p logs
#   sbatch scripts/sherlock_build_dataset.sbatch
#
# BEFORE RUNNING:
#   1. Update SUNETID below.
#   2. Install rclone and configure a Google Drive remote named "gdrive":
#        module load rclone
#        rclone config  # follow prompts to create a remote named "gdrive"
#   3. Upload data from Google Drive:
#        rclone copy "gdrive:/path/to/vlead_folder" $SCRATCH/vlead_data/raw/<run_name>
#      Repeat for each Google Drive folder. The run names must match
#      the train_runs / val_runs lists in configs/flightroom_fm_sherlock.yaml.
#   4. Then submit this job.

set -euo pipefail

SUNETID="YOUR_SUNETID"                             # <-- update this
SCRATCH_ROOT="$SCRATCH"                             # Sherlock scratch: /scratch/users/$SUNETID
NAV_POLICY_DIR="$HOME/V-LEAD/nav_policy"
RAW_ROOT="$SCRATCH_ROOT/vlead_data/raw"
PROCESSED_ROOT="$SCRATCH_ROOT/vlead_data/processed_flightroom"

echo "=== Build dataset job ==="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "RAW_ROOT:       $RAW_ROOT"
echo "PROCESSED_ROOT: $PROCESSED_ROOT"

# ── Load modules ─────────────────────────────────────────────────────────────
# Check available modules with: module avail python
# Adjust version as needed for your Sherlock allocation.
module load python/3.10.0
module load ffmpeg/6.0

# ── Python environment ────────────────────────────────────────────────────────
# Install nav_policy if not already installed (editable, user-local).
cd "$NAV_POLICY_DIR"
pip install --quiet --user -e "." 2>/dev/null || true

# ── Symlink data paths so the script finds data/raw and writes to data/processed ──
mkdir -p "$RAW_ROOT" "$PROCESSED_ROOT"
mkdir -p "$NAV_POLICY_DIR/data"
[ -L "$NAV_POLICY_DIR/data/raw" ] || ln -s "$RAW_ROOT" "$NAV_POLICY_DIR/data/raw"
[ -L "$NAV_POLICY_DIR/data/processed_flightroom" ] || ln -s "$PROCESSED_ROOT" "$NAV_POLICY_DIR/data/processed_flightroom"

# ── Run build_dataset_flightroom.py ──────────────────────────────────────────
# This script converts raw SINGER .pt + .mp4 files → processed caches + manifest + stats.
# CPU-only; no GPU needed.
echo "=== Starting build_dataset ==="
PYTHONPATH="$NAV_POLICY_DIR/src" python scripts/build_dataset_flightroom.py \
    --config configs/flightroom_fm_sherlock.yaml

echo "=== Done. Processed data at: $PROCESSED_ROOT ==="
echo "Time: $(date)"
