#!/usr/bin/env bash
# Regenerate every poster figure under V-LEAD/results/ from the snapshot
# in V-LEAD/results/logs/.  Run from anywhere; paths resolve relative to
# this script.
#
# Usage:  bash regenerate.sh
#         pip install -r requirements.txt   # one-time
set -euo pipefail
cd "$(dirname "$0")"

for f in headline_bar.py algo_comparison.py long8h_snapshots.py \
         failure_modes.py phase_a_pie.py query_heatmap.py \
         training_stability.py; do
  echo "--- $f ---"
  python3 "$f"
done

echo
echo "All figures written to $(realpath ../../../results)/"
