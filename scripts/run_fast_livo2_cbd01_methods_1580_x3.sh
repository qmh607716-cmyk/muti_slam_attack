#!/usr/bin/env bash
set -euo pipefail

# Run FAST-LIVO2 CBD_Building_01 transfer experiments at D=15m, R=80deg.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/catkin_ws/datasets/official/fast_livo2}"
SEQUENCE="${SEQUENCE:-CBD_Building_01}"
BAG="${FAST_CBD_BAG:-$DATA_ROOT/raw_rosbags/FAST-LIVO2-Dataset/FAST-LIVO2官方数据集/${SEQUENCE}.bag}"
OUT_ROOT="${FAST_CBD_OUT_ROOT:-$DATA_ROOT/experiments_${SEQUENCE}}"
REF_CSV_CBD="${FAST_CBD_REF_CSV:-$OUT_ROOT/runs/clean_01/clean_01_traj.csv}"
POSITION_CSV_CBD="${FAST_CBD_POSITION_CSV:-$OUT_ROOT/method_spoofer_positions_1580.csv}"

BAG="$BAG" \
OUT_ROOT="$OUT_ROOT" \
REF_CSV="$REF_CSV_CBD" \
POSITION_CSV="$POSITION_CSV_CBD" \
N_RUNS="${N_RUNS:-3}" \
RUN_TAG="${RUN_TAG:-lio_proxy}" \
PLAY_RATE="${PLAY_RATE:-1.0}" \
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}" \
SPOOFING_RANGE="${SPOOFING_RANGE:-80}" \
bash "$SCRIPT_DIR/run_fast_livo2_methods_1580_x3.sh"
