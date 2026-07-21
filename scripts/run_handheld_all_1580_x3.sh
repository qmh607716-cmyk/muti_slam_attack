#!/usr/bin/env bash
set -euo pipefail

# Run the conservative D=15m / R=80deg handheld batch suite.
# Each attack condition runs 3 trials. Random baseline runs 3 positions per mode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_CLEAN="${RUN_CLEAN:-1}"
RUN_RANDOM="${RUN_RANDOM:-1}"

run_step() {
    local label="$1"
    shift
    echo
    echo "============================================================"
    echo "[SUITE] $label"
    echo "============================================================"
    "$@"
}

if [[ "$RUN_CLEAN" == "1" ]]; then
    run_step "clean no-attack x3" \
        bash "$SCRIPT_DIR/run_handheld_clean_no_attack_x3.sh"
fi

run_step "SMVS static 15/80 x3" \
    bash "$SCRIPT_DIR/run_handheld_smvs_static_1580_x3.sh"

run_step "Bi-SMVS static 15/80 x3" \
    bash "$SCRIPT_DIR/run_handheld_bismvs_static_1580_x3.sh"

run_step "SMVS removal 15/80 x3" \
    bash "$SCRIPT_DIR/run_handheld_smvs_removal_1580_x3.sh"

run_step "Bi-SMVS removal 15/80 x3" \
    bash "$SCRIPT_DIR/run_handheld_bismvs_removal_1580_x3.sh"

if [[ "$RUN_RANDOM" == "1" ]]; then
    run_step "random static 15/80 x3" \
        env MODE=static bash "$SCRIPT_DIR/run_handheld_random_spoof_1580_x3.sh"

    run_step "random removal 15/80 x3" \
        env MODE=removal bash "$SCRIPT_DIR/run_handheld_random_spoof_1580_x3.sh"
fi

echo
echo "[SUITE] Done."
