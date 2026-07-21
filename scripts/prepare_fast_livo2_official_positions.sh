#!/usr/bin/env bash
set -euo pipefail

# Sample constrained feasible spoofer positions from a FAST-LIVO2 clean run.
#
# Usage:
#   CLEAN_CSV=.../clean_01_traj.csv bash prepare_fast_livo2_official_positions.sh

SLAMSPOOF_DIR="${SLAMSPOOF_DIR:-$HOME/catkin_ws/src/slamspoof}"
DATA_ROOT="${DATA_ROOT:-$HOME/catkin_ws/datasets/official/fast_livo2}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/experiments}"
CLEAN_CSV="${CLEAN_CSV:-$OUT_ROOT/runs/clean_01/clean_01_traj.csv}"
OUT_CSV="${OUT_CSV:-$OUT_ROOT/spoofer_positions_1580.csv}"

N_POSITIONS="${N_POSITIONS:-5}"
RANDOM_SEED="${RANDOM_SEED:-20260712}"
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}"
MIN_TRAJ_DIST="${MIN_TRAJ_DIST:-8}"
MAX_TRAJ_DIST="${MAX_TRAJ_DIST:-14}"
MIN_TRIGGER_FRAMES="${MIN_TRIGGER_FRAMES:-10}"
MIN_TRIGGER_RATIO="${MIN_TRIGGER_RATIO:-0.005}"
MAX_TRIGGER_RATIO="${MAX_TRIGGER_RATIO:-0.35}"

if [[ ! -f "$CLEAN_CSV" ]]; then
    echo "[ERROR] CLEAN_CSV not found: $CLEAN_CSV" >&2
    echo "        Run FAST-LIVO2 clean first, or set CLEAN_CSV explicitly." >&2
    exit 2
fi

mkdir -p "$(dirname "$OUT_CSV")"

python3 "$SLAMSPOOF_DIR/scripts/sample_random_spoofer_positions.py" \
    --traj "$CLEAN_CSV" \
    --out "$OUT_CSV" \
    --n "$N_POSITIONS" \
    --seed "$RANDOM_SEED" \
    --distance-threshold "$DISTANCE_THRESHOLD" \
    --min-traj-dist "$MIN_TRAJ_DIST" \
    --max-traj-dist "$MAX_TRAJ_DIST" \
    --min-trigger-frames "$MIN_TRIGGER_FRAMES" \
    --min-trigger-ratio "$MIN_TRIGGER_RATIO" \
    --max-trigger-ratio "$MAX_TRIGGER_RATIO"

echo "[OK] FAST-LIVO2 spoofer positions prepared: $OUT_CSV"
