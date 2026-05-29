#!/bin/bash
#SBATCH --job-name=vlead_train_fm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_fm_%j.out
#SBATCH --error=logs/train_fm_%j.err
#
# USAGE:
#   cd $HOME/V-LEAD/nav_policy
#   mkdir -p logs
#   sbatch scripts/sherlock_train_fm.sbatch
#
# Optionally override run tag:
#   sbatch --export=ALL,RUN_TAG=my_fm_v2 scripts/sherlock_train_fm.sbatch
#
# BEFORE RUNNING:
#   1. Update SUNETID below.
#   2. Ensure processed data exists: run sherlock_build_dataset.sbatch first.
#   3. Set WANDB_API_KEY environment variable for W&B logging:
#        echo 'export WANDB_API_KEY=<your_key>' >> ~/.bashrc && source ~/.bashrc
#   4. Update configs/flightroom_fm_sherlock.yaml to use your SUNETID in the paths.

set -euo pipefail

SUNETID="YOUR_SUNETID"                             # <-- update this
SCRATCH_ROOT="$SCRATCH"
NAV_POLICY_DIR="$HOME/V-LEAD/nav_policy"
PROCESSED_ROOT="$SCRATCH_ROOT/vlead_data/processed_flightroom"
CKPT_ROOT="$SCRATCH_ROOT/vlead_data/checkpoints_fm"
RUN_TAG="${RUN_TAG:-flightroom_fm_sherlock_v1}"

echo "=== FM Training job ==="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "RUN_TAG: $RUN_TAG"
echo "CKPT:    $CKPT_ROOT"

# ── Load modules ─────────────────────────────────────────────────────────────
module load python/3.10.0
module load cuda/12.4.0
module load cudnn/9.0.0

# ── Python environment ────────────────────────────────────────────────────────
cd "$NAV_POLICY_DIR"
pip install --quiet --user -e ".[wandb]" 2>/dev/null || true

# ── Symlink processed data ────────────────────────────────────────────────────
mkdir -p "$PROCESSED_ROOT" "$CKPT_ROOT" "$NAV_POLICY_DIR/data"
[ -L "$NAV_POLICY_DIR/data/processed_flightroom" ] || \
    ln -s "$PROCESSED_ROOT" "$NAV_POLICY_DIR/data/processed_flightroom"

# ── Run training ─────────────────────────────────────────────────────────────
echo "=== Starting FlowMatchingPolicy training ==="
PYTHONPATH="$NAV_POLICY_DIR/src" python scripts/train_fm.py \
    --config configs/flightroom_fm_sherlock.yaml \
    --run-tag "$RUN_TAG" \
    --checkpoint-dir "$CKPT_ROOT"

echo "=== Done. Best checkpoint: $CKPT_ROOT/fm_best.pt ==="
echo "Time: $(date)"

# ── Copy checkpoint back to home (optional, home has limited space) ──────────
# Uncomment if you want a local copy:
# mkdir -p "$HOME/vlead_checkpoints"
# cp "$CKPT_ROOT/fm_best.pt" "$HOME/vlead_checkpoints/fm_best_${RUN_TAG}.pt"
# echo "Copied to $HOME/vlead_checkpoints/fm_best_${RUN_TAG}.pt"
