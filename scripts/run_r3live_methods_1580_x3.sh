#!/usr/bin/env bash
set -euo pipefail

# Run R3LIVE transfer experiments at D=15m, R=80deg for both placement
# methods. Assumes clean_01 and method_spoofer_positions_1580.csv already exist.

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
WS_SETUP="${WS_SETUP:-$HOME/catkin_ws/devel_catkin_tools/setup.bash}"
SLAMSPOOF_DIR="${SLAMSPOOF_DIR:-$HOME/catkin_ws/src/slamspoof}"

DATA_ROOT="${DATA_ROOT:-$HOME/catkin_ws/datasets/official/r3live}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/experiments}"
REF_CSV="${REF_CSV:-$OUT_ROOT/runs/clean_01/clean_01_traj.csv}"

N_RUNS="${N_RUNS:-3}"
START_RUN="${START_RUN:-1}"
PLAY_RATE="${PLAY_RATE:-1.0}"
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}"
SPOOFING_RANGE="${SPOOFING_RANGE:-80}"
PARAM_TAG="${PARAM_TAG:-1580}"
POSITION_CSV="${POSITION_CSV:-$OUT_ROOT/method_spoofer_positions_${PARAM_TAG}.csv}"
RUN_TAG="${RUN_TAG:-proxy}"
RUN_STATIC="${RUN_STATIC:-1}"
RUN_REMOVAL="${RUN_REMOVAL:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
COOLDOWN_SEC="${COOLDOWN_SEC:-90}"
POST_RUN_CLEANUP="${POST_RUN_CLEANUP:-1}"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

if [[ ! -f "$REF_CSV" ]]; then
    echo "[ERROR] REF_CSV not found: $REF_CSV" >&2
    exit 2
fi
if [[ ! -f "$POSITION_CSV" ]]; then
    echo "[ERROR] POSITION_CSV not found: $POSITION_CSV" >&2
    echo "        Run prepare_r3live_method_positions.sh first." >&2
    exit 2
fi

set +u
source "$ROS_SETUP"
source "$WS_SETUP"
set -u

read_position() {
    local method="$1"
    python3 - "$POSITION_CSV" "$method" <<'PY'
import csv
import sys
path, method = sys.argv[1:]
with open(path) as f:
    for row in csv.DictReader(f):
        if row["method"] == method:
            print(row["spoofer_x"], row["spoofer_y"])
            raise SystemExit(0)
raise SystemExit(f"method not found in {path}: {method}")
PY
}

append_existing_summary_if_missing() {
    local run_name="$1"
    local method="$2"
    local mode="$3"
    python3 - "$SUMMARY_CSV" "$OUT_ROOT" "$run_name" "$method" "$mode" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary, out_root, run_name, method, mode = sys.argv[1:]
summary = Path(summary)
run_dir = Path(out_root) / "runs" / run_name
metrics_path = run_dir / "eval" / "metrics_complete.json"
config_path = run_dir / f"{run_name}_attack_config.json"
if not metrics_path.exists() or not config_path.exists():
    raise SystemExit(0)

header = [
    "run", "method", "mode", "distance_threshold", "spoofing_range",
    "spoofer_x", "spoofer_y", "input_bag", "traj_bag", "traj_csv",
    "eval_dir", "ape_rmse", "rpe_1m_rmse", "rpe_10m_rmse", "rpe_max",
    "status",
]
if summary.exists():
    with summary.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run") == run_name:
                raise SystemExit(0)

cfg = json.load(config_path.open())
metrics = json.load(metrics_path.open())
main = cfg.get("main", {})
row = {
    "run": run_name,
    "method": method,
    "mode": mode,
    "distance_threshold": main.get("distance_threshold", ""),
    "spoofing_range": main.get("spoofing_range", ""),
    "spoofer_x": main.get("spoofer_x", ""),
    "spoofer_y": main.get("spoofer_y", ""),
    "input_bag": main.get("output_file", ""),
    "traj_bag": str(run_dir / f"{run_name}_traj.bag"),
    "traj_csv": str(run_dir / f"{run_name}_traj.csv"),
    "eval_dir": str(run_dir / "eval"),
    "ape_rmse": metrics["evo"]["ape_translation"].get("rmse", ""),
    "rpe_1m_rmse": metrics["evo"]["rpe_1m_translation"].get("rmse", ""),
    "rpe_10m_rmse": metrics["evo"]["rpe_10m_translation"].get("rmse", ""),
    "rpe_max": metrics["paper_metrics"].get("rpe_translation_max_m", ""),
    "status": "ok_existing",
}
summary.parent.mkdir(parents=True, exist_ok=True)
write_header = not summary.exists()
with summary.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

cleanup_leftovers() {
    if [[ "$POST_RUN_CLEANUP" -ne 1 ]]; then
        return
    fi
    echo "[INFO] Cleaning possible leftover R3LIVE/rosbag processes ..."
    pkill -INT -f "rosbag record.*${OUT_ROOT}/runs" 2>/dev/null || true
    pkill -INT -f "rosbag play.*${OUT_ROOT}/runs" 2>/dev/null || true
    pkill -INT -f "r3live_official.launch" 2>/dev/null || true
    pkill -INT -f "r3live_mapping" 2>/dev/null || true
    sleep 5
    pkill -TERM -f "rosbag record.*${OUT_ROOT}/runs" 2>/dev/null || true
    pkill -TERM -f "rosbag play.*${OUT_ROOT}/runs" 2>/dev/null || true
    pkill -TERM -f "r3live_official.launch" 2>/dev/null || true
    pkill -TERM -f "r3live_mapping" 2>/dev/null || true
}

run_one() {
    local method="$1"
    local mode="$2"
    local run_id="$3"
    local xy
    xy="$(read_position "$method")"
    local sx sy
    read -r sx sy <<< "$xy"
    local tag_part=""
    if [[ -n "$RUN_TAG" ]]; then
        tag_part="_${RUN_TAG}"
    fi
    local run_name="${method}_${mode}_${PARAM_TAG}${tag_part}_run_$(printf '%02d' "$run_id")"
    local metrics_path="$OUT_ROOT/runs/$run_name/eval/metrics_complete.json"
    if [[ "$SKIP_COMPLETED" -eq 1 && -f "$metrics_path" ]]; then
        append_existing_summary_if_missing "$run_name" "$method" "$mode"
        echo "[SKIP] R3LIVE $run_name already has metrics: $metrics_path"
        return
    fi

    echo
    echo "============================================================"
    echo "[R3LIVE] method=$method mode=$mode run=$run_id/$N_RUNS"
    echo "  spoofer=($sx, $sy), D=$DISTANCE_THRESHOLD, R=$SPOOFING_RANGE, play_rate=$PLAY_RATE"
    echo "============================================================"

    REF_CSV="$REF_CSV" \
    MODE="$mode" \
    METHOD_LABEL="$method" \
    RUN_NAME="$run_name" \
    PLAY_RATE="$PLAY_RATE" \
    DISTANCE_THRESHOLD="$DISTANCE_THRESHOLD" \
    SPOOFING_RANGE="$SPOOFING_RANGE" \
    SPOOFER_X="$sx" \
    SPOOFER_Y="$sy" \
        bash "$SLAMSPOOF_DIR/scripts/run_r3live_official_once.sh"

    cleanup_leftovers
    if [[ "$COOLDOWN_SEC" -gt 0 ]]; then
        echo "[INFO] Cooling down for ${COOLDOWN_SEC}s before next R3LIVE run ..."
        sleep "$COOLDOWN_SEC"
    fi
}

for method in smvs bismvs; do
    for run_id in $(seq "$START_RUN" "$N_RUNS"); do
        if [[ "$RUN_STATIC" -eq 1 ]]; then
            run_one "$method" static "$run_id"
        fi
        if [[ "$RUN_REMOVAL" -eq 1 ]]; then
            run_one "$method" removal "$run_id"
        fi
    done
done

echo
echo "[OK] R3LIVE method suite complete."
echo "     summary: $OUT_ROOT/summary.csv"
