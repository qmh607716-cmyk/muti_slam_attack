#!/usr/bin/env bash
set -euo pipefail

# LVI-SAM handheld parameter sensitivity sweep for SMVS static attack.
# This complements the Bi-SMVS static sweep and keeps the original SMVS
# selected spoofer fixed while varying only the angular spoofing range.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LVI_DATASET_DIR="$HOME/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld"

N_RUNS="${N_RUNS:-3}"
START_RUN="${START_RUN:-1}"
RESET_SUMMARY="${RESET_SUMMARY:-1}"
PLAY_RATE="${PLAY_RATE:-1.0}"
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}"
SPOOFER_X="${SPOOFER_X:--18.500756557144932}"
SPOOFER_Y="${SPOOFER_Y:-70.38033214626245}"

run_one_range() {
    local range="$1"
    local tag="15${range}"
    local out_root="$LVI_DATASET_DIR/param_sweep/static_smvs_${tag}_x${N_RUNS}"

    echo
    echo "============================================================"
    echo "[SWEEP] SMVS static D=${DISTANCE_THRESHOLD} range=${range} x${N_RUNS}"
    echo "        output: ${out_root}"
    echo "============================================================"

    env \
        PARAM_TAG="$tag" \
        N_RUNS="$N_RUNS" \
        START_RUN="$START_RUN" \
        RESET_SUMMARY="$RESET_SUMMARY" \
        PLAY_RATE="$PLAY_RATE" \
        DISTANCE_THRESHOLD="$DISTANCE_THRESHOLD" \
        SPOOFING_RANGE="$range" \
        SPOOFER_X="$SPOOFER_X" \
        SPOOFER_Y="$SPOOFER_Y" \
        OUT_ROOT="$out_root" \
        bash "$SCRIPT_DIR/run_handheld_smvs_static_1580_x15.sh"
}

run_one_range 40
run_one_range 60
run_one_range 80

echo
echo "[SWEEP] Done."
echo "[SWEEP] Results are under: $LVI_DATASET_DIR/param_sweep"
