#!/usr/bin/env bash
set -euo pipefail

# Run static LVI-SAM ablation groups using prepared 15/80 spoofer positions.
#
# Default runs all five groups:
#   smvs_paper,smvs_graph_cma,bismvs_paper,bismvs_cma_no_graph,bismvs_graph_cma
#
# To run only a subset:
#   RUN_GROUPS=bismvs_paper,bismvs_cma_no_graph bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LVI_DATASET_DIR="$HOME/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld"

N_RUNS="${N_RUNS:-3}"
START_RUN="${START_RUN:-1}"
RESET_SUMMARY="${RESET_SUMMARY:-1}"
PLAY_RATE="${PLAY_RATE:-1.0}"
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}"
SPOOFING_RANGE="${SPOOFING_RANGE:-80}"
RUN_GROUPS="${RUN_GROUPS:-smvs_paper,smvs_graph_cma,bismvs_paper,bismvs_cma_no_graph,bismvs_graph_cma}"

POSITION_CSV="${POSITION_CSV:-$LVI_DATASET_DIR/ablation/positions_1580/method_spoofer_positions_1580.csv}"
OUT_BASE="${OUT_BASE:-$LVI_DATASET_DIR/ablation}"

if [[ ! -f "$POSITION_CSV" ]]; then
    echo "[INFO] position CSV missing; preparing positions first"
    DISTANCE_THRESHOLD="$DISTANCE_THRESHOLD" \
    SPOOFING_RANGE="$SPOOFING_RANGE" \
        bash "$SCRIPT_DIR/prepare_handheld_ablation_positions_1580.sh"
fi

read_position() {
    local method="$1"
    python3 - "$POSITION_CSV" "$method" <<'PY'
import csv
import sys

path, target = sys.argv[1:]
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("method") == target:
            print(row["spoofer_x"], row["spoofer_y"], row.get("selection_json", ""))
            raise SystemExit(0)
raise SystemExit(f"method not found in {path}: {target}")
PY
}

IFS=',' read -r -a GROUP_LIST <<< "$RUN_GROUPS"

for method in "${GROUP_LIST[@]}"; do
    method="$(echo "$method" | xargs)"
    if [[ -z "$method" ]]; then
        continue
    fi

    read -r sx sy selection_json < <(read_position "$method")
    out_root="$OUT_BASE/static_${method}_1580_x${N_RUNS}"

    echo
    echo "============================================================"
    echo "[ABLATION] ${method} static D=${DISTANCE_THRESHOLD} range=${SPOOFING_RANGE} x${N_RUNS}"
    echo "           spoofer=(${sx}, ${sy})"
    echo "           selection=${selection_json}"
    echo "           output=${out_root}"
    echo "============================================================"

    env \
        METHOD="$method" \
        METHOD_SLUG="$method" \
        PARAM_TAG="1580" \
        N_RUNS="$N_RUNS" \
        START_RUN="$START_RUN" \
        RESET_SUMMARY="$RESET_SUMMARY" \
        PLAY_RATE="$PLAY_RATE" \
        DISTANCE_THRESHOLD="$DISTANCE_THRESHOLD" \
        SPOOFING_RANGE="$SPOOFING_RANGE" \
        SPOOFER_X="$sx" \
        SPOOFER_Y="$sy" \
        OUT_ROOT="$out_root" \
        bash "$SCRIPT_DIR/run_handheld_bismvs_static_1580_x15.sh"
done

echo
echo "[DONE] handheld ablation static 15/80 x${N_RUNS}"
echo "Summaries are under: $OUT_BASE/static_*_1580_x${N_RUNS}/summary.csv"
