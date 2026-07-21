#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_RUNS="${N_RUNS:-15}" \
RUN_TAG="${RUN_TAG:-lio_proxy}" \
PLAY_RATE="${PLAY_RATE:-1.0}" \
bash "$SCRIPT_DIR/run_fast_livo2_cbd01_methods_1580_x3.sh"
