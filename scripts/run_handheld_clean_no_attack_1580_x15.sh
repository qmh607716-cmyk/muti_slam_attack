#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch clean-repeatability experiment:
# handheld / no attack / clean replay / 15 runs
#
# Each run:
#   1) start/check roscore
#   2) launch LVI-SAM
#   3) record odometry
#   4) play CLEAN_BAG with --clock -r 1.0
#   5) wait for LVI-SAM post-processing
#   6) stop LVI-SAM first, then stop recorder
#   7) extract trajectory CSV
#   8) evaluate against clean/original trajectory
#   9) clean large intermediate bags
# ============================================================

N_RUNS=15
START_RUN=1
RESET_SUMMARY=1

ROS_SETUP="/opt/ros/noetic/setup.bash"
WS_SETUP="$HOME/catkin_ws/devel_catkin_tools/setup.bash"

SLAMSPOOF_DIR="$HOME/catkin_ws/src/slamspoof"
LVI_DATASET_DIR="$HOME/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld"
CONFIG_FILE="$SLAMSPOOF_DIR/config_lvisam.json"

# Reference clean trajectory CSV.
ORIG_CSV="$LVI_DATASET_DIR/original/handheld_original_traj.csv"

# Output root for 15 clean replay runs.
OUT_ROOT="$LVI_DATASET_DIR/repeat_clean/no_attack_1580_x15"

ODOM_TOPIC="/lvi_sam/lidar/mapping/odometry"

LVI_LAUNCH_PKG="lvi_sam"
# run.launch already has RViz disabled in this workspace.
LVI_LAUNCH_FILE="run.launch"

METHOD="clean"
PLATFORM="handheld"
MODE="no_attack"
DISTANCE_THRESHOLD=0
SPOOFING_RANGE=0
SPOOFER_X="NA"
SPOOFER_Y="NA"

LVI_START_WAIT=25
RECORD_START_WAIT=2
PLAY_RATE=1.0
POST_PLAY_WAIT=25
STOP_WAIT=5
MIN_CSV_ROWS=6000
KEEP_TRAJ_BAG=0

RUNS_DIR="$OUT_ROOT/runs"
LOG_DIR="$OUT_ROOT/logs"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
ROSCORE_LOG="$LOG_DIR/roscore.log"

mkdir -p "$RUNS_DIR" "$LOG_DIR"

source "$ROS_SETUP"
source "$WS_SETUP"

ROSCORE_PID=""
STARTED_ROSCORE=0
REC_PID=""
LVI_PID=""

safe_kill() {
    local pid="${1:-}"
    local name="${2:-process}"

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[INFO] Stopping $name pid=$pid ..."
        kill -INT "$pid" 2>/dev/null || true
        sleep "$STOP_WAIT"

        if kill -0 "$pid" 2>/dev/null; then
            echo "[WARN] $name did not stop after SIGINT, sending SIGTERM ..."
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
        fi

        if kill -0 "$pid" 2>/dev/null; then
            echo "[WARN] $name still alive, sending SIGKILL ..."
            kill -KILL "$pid" 2>/dev/null || true
        fi

        wait "$pid" 2>/dev/null || true
    fi
}

wait_for_ros_master() {
    local max_wait="${1:-30}"
    local i
    for i in $(seq 1 "$max_wait"); do
        if rosparam list >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

start_roscore_if_needed() {
    if rosparam list >/dev/null 2>&1; then
        echo "[INFO] Existing ROS master detected. Using existing roscore."
        STARTED_ROSCORE=0
        return 0
    fi

    echo "[INFO] No ROS master detected. Starting roscore inside this script ..."
    roscore > "$ROSCORE_LOG" 2>&1 &
    ROSCORE_PID=$!
    STARTED_ROSCORE=1

    if ! wait_for_ros_master 30; then
        echo "[ERROR] roscore did not become available within 30 seconds."
        echo "[ERROR] Check log: $ROSCORE_LOG"
        exit 1
    fi

    echo "[INFO] roscore started successfully. pid=$ROSCORE_PID"
}

cleanup_lvi_processes() {
    echo "[INFO] Cleaning possible leftover LVI-SAM / rosbag processes ..."
    pkill -f "roslaunch.*lvi_sam.*run" >/dev/null 2>&1 || true
    pkill -f "lvi_sam" >/dev/null 2>&1 || true
    pkill -f "rosbag record.*${ODOM_TOPIC}" >/dev/null 2>&1 || true
    sleep 3
}

cleanup_large_outputs_after_run() {
    if [[ "${KEEP_TRAJ_BAG}" -eq 0 ]]; then
        rm -f "$TRAJ_BAG" "$TRAJ_BAG.active" 2>/dev/null || true
    fi
}

cleanup_on_exit() {
    echo "[INFO] Global cleanup on script exit ..."
    safe_kill "${REC_PID:-}" "rosbag record"
    safe_kill "${LVI_PID:-}" "LVI-SAM roslaunch"
    cleanup_lvi_processes

    if [[ "$STARTED_ROSCORE" -eq 1 ]]; then
        safe_kill "${ROSCORE_PID:-}" "roscore"
        pkill -f "rosmaster" >/dev/null 2>&1 || true
        pkill -f "roscore" >/dev/null 2>&1 || true
        pkill -f "rosout" >/dev/null 2>&1 || true
    fi
}

trap cleanup_on_exit EXIT

extract_metric_from_evo_zip() {
    local zip_path="$1"
    local metric="${2:-rmse}"
    python3 - "$zip_path" "$metric" <<'PY'
import json, zipfile, sys, os
p = sys.argv[1]
metric = sys.argv[2]
if not os.path.exists(p):
    print("")
    sys.exit(0)
try:
    with zipfile.ZipFile(p) as z:
        stats = json.loads(z.read("stats.json"))
    print(stats.get(metric, ""))
except Exception:
    print("")
PY
}

max_numeric_metric() {
    python3 - "$@" <<'PY'
import math
import sys

vals = []
for raw in sys.argv[1:]:
    try:
        value = float(raw)
    except Exception:
        continue
    if math.isfinite(value):
        vals.append(value)
print(max(vals) if vals else "")
PY
}

append_summary_row() {
    local run_id="$1"
    local traj_bag="$2"
    local traj_csv="$3"
    local eval_dir="$4"
    local status="$5"

    local ape_zip="$eval_dir/evo_ape.txt"
    local rpe1_zip="$eval_dir/evo_rpe.txt"
    local rpe10_zip="$eval_dir/evo_rpe_10m.txt"

    local ape_rmse
    local rpe1_rmse
    local rpe10_rmse
    local rpe1_max
    local rpe10_max
    local rpe_max

    ape_rmse="$(extract_metric_from_evo_zip "$ape_zip")"
    rpe1_rmse="$(extract_metric_from_evo_zip "$rpe1_zip")"
    rpe10_rmse="$(extract_metric_from_evo_zip "$rpe10_zip")"
    rpe1_max="$(extract_metric_from_evo_zip "$rpe1_zip" max)"
    rpe10_max="$(extract_metric_from_evo_zip "$rpe10_zip" max)"
    rpe_max="$(max_numeric_metric "$rpe1_max" "$rpe10_max")"

    echo "${run_id},${PLATFORM},${METHOD},${MODE},${DISTANCE_THRESHOLD},${SPOOFING_RANGE},${SPOOFER_X},${SPOOFER_Y},${CLEAN_BAG},${traj_bag},${traj_csv},${eval_dir},${ape_rmse},${rpe1_rmse},${rpe10_rmse},${rpe1_max},${rpe10_max},${rpe_max},${status}" >> "$SUMMARY_CSV"
}

# Read clean input bag from config main.input_file. This script never generates an attacked bag.
CLEAN_BAG="$(
python3 - <<PY
import json
cfg = json.load(open("$CONFIG_FILE"))
print(cfg["main"]["input_file"])
PY
)"

if [[ -z "$CLEAN_BAG" ]]; then
    echo "[ERROR] Could not read main.input_file from $CONFIG_FILE"
    exit 1
fi

echo "[INFO] Config file: $CONFIG_FILE"
echo "[INFO] Clean bag from config main.input_file: $CLEAN_BAG"
echo "[INFO] Original/reference CSV: $ORIG_CSV"
echo "[INFO] Output root: $OUT_ROOT"
echo "[INFO] Method: $METHOD"
echo "[INFO] Mode: $MODE"
echo "[INFO] Replay rate: $PLAY_RATE"
echo "[INFO] Post-play wait: $POST_PLAY_WAIT seconds"
echo "[INFO] Start run: $START_RUN"
echo "[INFO] Total runs: $N_RUNS"

if [[ ! -f "$CLEAN_BAG" ]]; then
    echo "[ERROR] CLEAN_BAG not found: $CLEAN_BAG"
    echo "[ERROR] Edit CONFIG_FILE main.input_file or set CLEAN_BAG manually in this script."
    exit 1
fi

if [[ ! -f "$ORIG_CSV" ]]; then
    echo "[ERROR] ORIG_CSV not found: $ORIG_CSV"
    echo "Please edit ORIG_CSV in this script."
    exit 1
fi

start_roscore_if_needed
rosparam set /use_sim_time true

if [[ "$RESET_SUMMARY" -eq 1 ]]; then
    echo "[INFO] RESET_SUMMARY=1, overwriting summary: $SUMMARY_CSV"
    cat > "$SUMMARY_CSV" <<EOF2
run,platform,method,mode,distance_threshold,spoofing_range,spoofer_x,spoofer_y,input_bag,traj_bag,traj_csv,eval_dir,ape_rmse,rpe_1m_rmse,rpe_10m_rmse,rpe_1m_max,rpe_10m_max,rpe_max,status
EOF2
else
    echo "[INFO] RESET_SUMMARY=0, appending to existing summary if present."
    if [[ ! -f "$SUMMARY_CSV" ]]; then
        cat > "$SUMMARY_CSV" <<EOF2
run,platform,method,mode,distance_threshold,spoofing_range,spoofer_x,spoofer_y,input_bag,traj_bag,traj_csv,eval_dir,ape_rmse,rpe_1m_rmse,rpe_10m_rmse,rpe_1m_max,rpe_10m_max,rpe_max,status
EOF2
    fi
fi

for RUN_INDEX in $(seq "$START_RUN" "$N_RUNS"); do
    RUN_ID="$(printf "%02d" "$RUN_INDEX")"

    echo "============================================================"
    echo "[RUN $RUN_ID/$N_RUNS] handheld clean no-attack replay 15/80 suite"
    echo "============================================================"

    RUN_DIR="$RUNS_DIR/run_${RUN_ID}"
    mkdir -p "$RUN_DIR"

    TRAJ_BAG="$RUN_DIR/handheld_clean_no_attack_run_${RUN_ID}_traj.bag"
    TRAJ_CSV="$RUN_DIR/handheld_clean_no_attack_run_${RUN_ID}_traj.csv"
    EVAL_DIR="$RUN_DIR/eval"
    mkdir -p "$EVAL_DIR"

    LVI_LOG="$RUN_DIR/01_lvisam.log"
    RECORD_LOG="$RUN_DIR/02_record.log"
    PLAY_LOG="$RUN_DIR/03_play_clean_bag.log"
    EXTRACT_LOG="$RUN_DIR/04_extract_csv.log"
    EVAL_LOG="$RUN_DIR/05_eval.log"

    REC_PID=""
    LVI_PID=""

    rm -f "$TRAJ_BAG" "$TRAJ_BAG.active" "$TRAJ_CSV"
    rm -rf "$EVAL_DIR"
    mkdir -p "$EVAL_DIR"

    cleanup_lvi_processes
    rosparam set /use_sim_time true

    echo "[RUN $RUN_ID] Starting LVI-SAM ..."

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used \
            --format=csv,noheader > "$RUN_DIR/gpu_before.log" 2>/dev/null || true
        echo "[RUN $RUN_ID] GPU before: $(cat "$RUN_DIR/gpu_before.log" 2>/dev/null || echo '?')"
    fi

    roslaunch "$LVI_LAUNCH_PKG" "$LVI_LAUNCH_FILE" \
        > "$LVI_LOG" 2>&1 &
    LVI_PID=$!

    sleep "$LVI_START_WAIT"

    if ! kill -0 "$LVI_PID" 2>/dev/null; then
        echo "[ERROR] LVI-SAM launch exited before replay. Check: $LVI_LOG"
        tail -50 "$LVI_LOG" > "$RUN_DIR/lvi_launch_failed_tail.log" 2>/dev/null || true
        append_summary_row "$RUN_ID" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_lvi_launch"
        cleanup_lvi_processes
        cleanup_large_outputs_after_run
        continue
    fi

    echo "[RUN $RUN_ID] Starting rosbag record ..."
    rosbag record -O "$TRAJ_BAG" "$ODOM_TOPIC" \
        > "$RECORD_LOG" 2>&1 &
    REC_PID=$!

    sleep "$RECORD_START_WAIT"

    echo "[RUN $RUN_ID] Playing clean bag at rate ${PLAY_RATE} ..."
    set +e
    rosbag play "$CLEAN_BAG" --clock -r "$PLAY_RATE" \
        > "$PLAY_LOG" 2>&1
    PLAY_STATUS=$?
    set -e

    echo "[RUN $RUN_ID] Waiting ${POST_PLAY_WAIT}s for LVI-SAM post-processing ..."
    sleep "$POST_PLAY_WAIT"

    echo "[RUN $RUN_ID] Stopping LVI-SAM first ..."
    safe_kill "$LVI_PID" "LVI-SAM roslaunch"
    LVI_PID=""

    sleep 3

    echo "[RUN $RUN_ID] Stopping rosbag record ..."
    safe_kill "$REC_PID" "rosbag record"
    REC_PID=""

    cleanup_lvi_processes

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used \
            --format=csv,noheader > "$RUN_DIR/gpu_after.log" 2>/dev/null || true
        echo "[RUN $RUN_ID] GPU after: $(cat "$RUN_DIR/gpu_after.log" 2>/dev/null || echo '?')"
    fi

    if [[ "$PLAY_STATUS" -ne 0 ]]; then
        echo "[ERROR] rosbag play failed with status $PLAY_STATUS"
        append_summary_row "$RUN_ID" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_rosbag_play"
        cleanup_large_outputs_after_run
        continue
    fi

    if [[ ! -f "$TRAJ_BAG" ]]; then
        echo "[ERROR] Trajectory bag was not recorded: $TRAJ_BAG"
        append_summary_row "$RUN_ID" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_record_traj_bag"
        cleanup_large_outputs_after_run
        continue
    fi

    echo "[RUN $RUN_ID] Extracting trajectory CSV ..."
    set +e
    python3 "$SLAMSPOOF_DIR/scripts/extract_lvisam_odom_csv.py" \
        --bag "$TRAJ_BAG" \
        --out "$TRAJ_CSV" \
        > "$EXTRACT_LOG" 2>&1
    EXTRACT_STATUS=$?
    set -e

    if [[ "$EXTRACT_STATUS" -ne 0 || ! -f "$TRAJ_CSV" ]]; then
        echo "[ERROR] CSV extraction failed for run $RUN_ID"
        append_summary_row "$RUN_ID" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_extract_csv"
        cleanup_large_outputs_after_run
        continue
    fi

    CSV_ROWS=$(wc -l < "$TRAJ_CSV" 2>/dev/null || echo 0)
    CSV_ROWS=$((CSV_ROWS - 1))
    if [[ "$CSV_ROWS" -lt 0 ]]; then
        CSV_ROWS=0
    fi

    if [[ "$CSV_ROWS" -lt "$MIN_CSV_ROWS" ]]; then
        echo "[ERROR] run $RUN_ID: CSV has only $CSV_ROWS rows (< $MIN_CSV_ROWS); likely truncated trajectory."
        tail -50 "$LVI_LOG" > "$RUN_DIR/lvi_crash_tail.log" 2>/dev/null || true
        append_summary_row "$RUN_ID" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "truncated_csv_${CSV_ROWS}rows"
        cleanup_large_outputs_after_run
        continue
    fi

    echo "[OK] run $RUN_ID: CSV has $CSV_ROWS rows"

    echo "[RUN $RUN_ID] Evaluating clean repeatability ..."
    set +e
    python3 "$SLAMSPOOF_DIR/scripts/evaluate_attack.py" \
        --orig "$ORIG_CSV" \
        --att "$TRAJ_CSV" \
        --out-dir "$EVAL_DIR" \
        --title "handheld clean no-attack run ${RUN_ID}" \
        > "$EVAL_LOG" 2>&1
    EVAL_STATUS=$?
    set -e

    if [[ "$EVAL_STATUS" -ne 0 ]]; then
        echo "[WARN] Evaluation script failed for run $RUN_ID. Check: $EVAL_LOG"
        append_summary_row "$RUN_ID" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_eval"
        cleanup_large_outputs_after_run
        continue
    fi

    append_summary_row "$RUN_ID" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "ok"
    cleanup_large_outputs_after_run

    echo "[RUN $RUN_ID] Done."
    echo "[RUN $RUN_ID] Current summary: $SUMMARY_CSV"
done

echo "============================================================"
echo "[ALL DONE]"
echo "Summary CSV:"
echo "$SUMMARY_CSV"
echo "Logs and run folders:"
echo "$RUNS_DIR"
echo "============================================================"

# Final aggregate stats.
python3 - <<PY
import csv, statistics, os

summary = "$SUMMARY_CSV"
rows = []

def fnum(row, key):
    try:
        raw = row.get(key, "")
        return float(raw) if raw not in ("", None) else None
    except Exception:
        return None

if os.path.exists(summary):
    with open(summary) as f:
        for row in csv.DictReader(f):
            rows.append((
                row.get("run", ""),
                fnum(row, "ape_rmse"),
                fnum(row, "rpe_10m_rmse"),
                fnum(row, "rpe_max"),
                row.get("status", ""),
            ))

ok = [r for r in rows if r[4] == "ok" and r[1] is not None]
fail = [r for r in rows if not (r[4] == "ok" and r[1] is not None)]
apes = [r[1] for r in ok]
rpe10s = [r[2] for r in ok if r[2] is not None]
rpemaxs = [r[3] for r in ok if r[3] is not None]

print()
print(f"  Total rows : {len(rows)}")
print(f"  OK         : {len(ok)}")
print(f"  Failed     : {len(fail)}  ({', '.join(f'{r[0]}({r[4]})' for r in fail) or 'none'})")

def show_stats(label, vals, unit="m", digits=4):
    if not vals:
        return
    print(f"  {label} min    : {min(vals):.{digits}f} {unit}")
    print(f"  {label} median : {statistics.median(vals):.{digits}f} {unit}")
    print(f"  {label} mean   : {statistics.mean(vals):.{digits}f} {unit}")
    print(f"  {label} max    : {max(vals):.{digits}f} {unit}")
    if len(vals) > 1:
        print(f"  {label} std    : {statistics.stdev(vals):.{digits}f} {unit}")

if apes:
    show_stats("APE", apes)
    show_stats("RPE-10m", rpe10s)
    show_stats("RPE-max", rpemaxs)
    print()
    print("  Per-row APE / RPE-max:")
    for run_id, ape, _rpe10, rpe_max, status in rows:
        ape_s = "NA" if ape is None else f"{ape:.4f} m"
        rpe_s = "NA" if rpe_max is None else f"{rpe_max:.4f} m"
        print(f"    {run_id}: APE={ape_s:>12}  RPE-max={rpe_s:>12}  [{status}]")
PY
