#!/usr/bin/env bash
set -euo pipefail

# LVI-SAM handheld trigger-distance sensitivity for Bi-SMVS static attack.
# The angular range is fixed to 60deg by default. Since D=15/R=60 is already
# produced by run_handheld_bismvs_static_param_sweep_x3.sh, this script only
# fills D=10 and D=20 unless DISTANCES is overridden.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LVI_DATASET_DIR="$HOME/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld"

N_RUNS="${N_RUNS:-3}"
START_RUN="${START_RUN:-1}"
RESET_SUMMARY="${RESET_SUMMARY:-1}"
PLAY_RATE="${PLAY_RATE:-1.0}"
SPOOFING_RANGE="${SPOOFING_RANGE:-60}"
DISTANCES="${DISTANCES:-10 20}"
SPOOFER_X="${SPOOFER_X:-31.28075677647965}"
SPOOFER_Y="${SPOOFER_Y:--102.07423272183334}"

run_one_distance() {
    local dist="$1"
    local tag="D${dist}_R${SPOOFING_RANGE}"
    local out_root="$LVI_DATASET_DIR/param_sweep/static_bismvs_${tag}_x${N_RUNS}"

    echo
    echo "============================================================"
    echo "[SWEEP] Bi-SMVS static D=${dist} range=${SPOOFING_RANGE} x${N_RUNS}"
    echo "        output: ${out_root}"
    echo "============================================================"

    env \
        PARAM_TAG="$tag" \
        N_RUNS="$N_RUNS" \
        START_RUN="$START_RUN" \
        RESET_SUMMARY="$RESET_SUMMARY" \
        PLAY_RATE="$PLAY_RATE" \
        DISTANCE_THRESHOLD="$dist" \
        SPOOFING_RANGE="$SPOOFING_RANGE" \
        SPOOFER_X="$SPOOFER_X" \
        SPOOFER_Y="$SPOOFER_Y" \
        OUT_ROOT="$out_root" \
        bash "$SCRIPT_DIR/run_handheld_bismvs_static_1580_x15.sh"
}

for dist in $DISTANCES; do
    run_one_distance "$dist"
done

echo
echo "[SWEEP] Done."
echo "[SWEEP] Results are under: $LVI_DATASET_DIR/param_sweep"
