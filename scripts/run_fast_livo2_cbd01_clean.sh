#!/usr/bin/env bash
set -euo pipefail

# Clean FAST-LIVO2 run for the larger official CBD_Building_01 sequence.
# Outputs are kept separate from the earlier Bright_Screen_Wall experiments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/catkin_ws/datasets/official/fast_livo2}"
SEQUENCE="${SEQUENCE:-CBD_Building_01}"
BAG="${BAG:-$DATA_ROOT/raw_rosbags/FAST-LIVO2-Dataset/FAST-LIVO2官方数据集/${SEQUENCE}.bag}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/experiments_${SEQUENCE}}"

unset SPOOFER_X SPOOFER_Y

MODE=clean \
METHOD_LABEL=fast_livo2 \
RUN_NAME="${RUN_NAME:-clean_01}" \
BAG="$BAG" \
OUT_ROOT="$OUT_ROOT" \
PLAY_RATE="${PLAY_RATE:-1.0}" \
bash "$SCRIPT_DIR/run_fast_livo2_official_once.sh"
