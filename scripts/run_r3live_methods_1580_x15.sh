#!/usr/bin/env bash
set -euo pipefail

# Formal R3LIVE transfer suite at D=15m, R=80deg.
# Uses method_spoofer_positions_1580.csv, which should be generated after the
# LIO-SAM proxy graph if GRAPH_SOURCE=lio_sam is desired.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export N_RUNS="${N_RUNS:-15}"
export RUN_TAG="${RUN_TAG:-lio_proxy}"
export PLAY_RATE="${PLAY_RATE:-1.0}"

exec bash "$SCRIPT_DIR/run_r3live_methods_1580_x3.sh"
