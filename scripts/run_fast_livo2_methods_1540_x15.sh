#!/usr/bin/env bash
set -euo pipefail

# FAST-LIVO2 transfer suite at D=15m, R=40deg.
# Resumable: completed runs with metrics_complete.json are skipped.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PARAM_TAG="${PARAM_TAG:-1540}"
export DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}"
export SPOOFING_RANGE="${SPOOFING_RANGE:-40}"
export N_RUNS="${N_RUNS:-15}"
export START_RUN="${START_RUN:-1}"
export RUN_TAG="${RUN_TAG:-lio_proxy_1540}"
export PLAY_RATE="${PLAY_RATE:-1.0}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
export KEEP_ATTACK_BAG="${KEEP_ATTACK_BAG:-0}"

export OUT_ROOT="${OUT_ROOT:-$HOME/catkin_ws/datasets/official/fast_livo2/experiments_CBD_Building_01}"
export DATA_ROOT="${DATA_ROOT:-$HOME/catkin_ws/datasets/official/fast_livo2}"
export BAG="${BAG:-$DATA_ROOT/raw_rosbags/FAST-LIVO2-Dataset/FAST-LIVO2官方数据集/CBD_Building_01.bag}"
export REF_CSV="${REF_CSV:-$OUT_ROOT/runs/clean_01/clean_01_traj.csv}"
export POSITION_CSV="${POSITION_CSV:-$OUT_ROOT/method_spoofer_positions_${PARAM_TAG}.csv}"

exec bash "$SCRIPT_DIR/run_fast_livo2_methods_1580_x3.sh"
