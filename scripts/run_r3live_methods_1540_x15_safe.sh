#!/usr/bin/env bash
set -euo pipefail

# R3LIVE transfer suite at D=15m, R=40deg.
# Resumable: completed runs with metrics_complete.json are skipped.
# This wrapper uses conservative runtime defaults to reduce sustained load.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PARAM_TAG="${PARAM_TAG:-1540}"
export DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}"
export SPOOFING_RANGE="${SPOOFING_RANGE:-40}"
export N_RUNS="${N_RUNS:-15}"
export START_RUN="${START_RUN:-1}"
export RUN_TAG="${RUN_TAG:-proxy_1540}"
export PLAY_RATE="${PLAY_RATE:-1.0}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export DATA_ROOT="${DATA_ROOT:-$HOME/catkin_ws/datasets/official/r3live}"
export OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/experiments}"
export SEQUENCE="${SEQUENCE:-hku_campus_seq_00}"
export BAG="${BAG:-$DATA_ROOT/raw_rosbags/R3LIVE-Dataset/${SEQUENCE}.bag}"
export REF_CSV="${REF_CSV:-$OUT_ROOT/runs/clean_01/clean_01_traj.csv}"
export POSITION_CSV="${POSITION_CSV:-$OUT_ROOT/method_spoofer_positions_${PARAM_TAG}.csv}"

export RVIZ="${RVIZ:-false}"
export RECORD_OFFLINE_MAP="${RECORD_OFFLINE_MAP:-0}"
export KEEP_ATTACK_BAG="${KEEP_ATTACK_BAG:-0}"
export POST_RUN_CLEANUP="${POST_RUN_CLEANUP:-1}"
export COOLDOWN_SEC="${COOLDOWN_SEC:-180}"
export TARGET_START_WAIT="${TARGET_START_WAIT:-25}"
export POST_PLAY_WAIT="${POST_PLAY_WAIT:-20}"
export STOP_WAIT="${STOP_WAIT:-8}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

exec bash "$SCRIPT_DIR/run_r3live_methods_1580_x3.sh"
