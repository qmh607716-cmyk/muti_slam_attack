#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch experiment:
# handheld / SMVS / removal / D=30m / spoofing_range=80deg
#
# Each run:
#   1) start/check roscore
#   2) generate attacked bag
#   3) launch LVI-SAM
#   4) record odometry
#   5) play attacked bag with --clock -r 0.8
#   6) wait for LVI-SAM post-processing
#   7) stop LVI-SAM first, then stop recorder
#   8) extract trajectory CSV
#   9) evaluate against clean trajectory
#  10) clean large intermediate bags
# ============================================================

# -----------------------------
# User configuration
# -----------------------------

N_RUNS=15
START_RUN=1

# New experiment: overwrite summary.csv at the beginning.
# If the script is interrupted and you want to resume later, set RESET_SUMMARY=0
# and set START_RUN to the next unfinished run.
RESET_SUMMARY=1

ROS_SETUP="/opt/ros/noetic/setup.bash"
WS_SETUP="$HOME/catkin_ws/devel_catkin_tools/setup.bash"

SLAMSPOOF_DIR="$HOME/catkin_ws/src/slamspoof"
LVI_DATASET_DIR="$HOME/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld"

CONFIG_FILE="$SLAMSPOOF_DIR/config_lvisam.json"

# Clean/original trajectory CSV.
ORIG_CSV="$LVI_DATASET_DIR/original/handheld_original_traj.csv"

# Output root for this repeated experiment.
OUT_ROOT="$LVI_DATASET_DIR/repeat_removal/smvs_3080"

# Topic recorded from LVI-SAM.
ODOM_TOPIC="/lvi_sam/lidar/mapping/odometry"

# LVI-SAM launch command.
LVI_LAUNCH_PKG="lvi_sam"
LVI_LAUNCH_FILE="run.launch"

# Attack metadata.
METHOD="smvs"
PLATFORM="handheld"
MODE="removal"
DISTANCE_THRESHOLD=30
SPOOFING_RANGE=80
SPOOFER_X=-18.500756557144932
SPOOFER_Y=70.38033214626245

# Timing parameters.
LVI_START_WAIT=25
RECORD_START_WAIT=2
PLAY_RATE=1.0
POST_PLAY_WAIT=25
STOP_WAIT=5

# Minimum acceptable CSV row count.
MIN_CSV_ROWS=6000

# Storage control.
KEEP_TRAJ_BAG=0
REGENERATE_ATTACK_BAG_EACH_RUN=1
DELETE_ATTACK_BAG_AFTER_RUN=1

# -----------------------------
# Internal paths
# -----------------------------

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

# -----------------------------
# Helper functions
# -----------------------------

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

    pkill -f "roslaunch.*lvi_sam.*run.launch" >/dev/null 2>&1 || true
    pkill -f "lvi_sam" >/dev/null 2>&1 || true
    pkill -f "rosbag record.*${ODOM_TOPIC}" >/dev/null 2>&1 || true

    sleep 3
}

cleanup_large_outputs_after_run() {
    if [[ "${KEEP_TRAJ_BAG}" -eq 0 ]]; then
        rm -f "$TRAJ_BAG" "$TRAJ_BAG.active" 2>/dev/null || true
    fi

    if [[ "${DELETE_ATTACK_BAG_AFTER_RUN}" -eq 1 && "${REGENERATE_ATTACK_BAG_EACH_RUN}" -eq 1 && -n "${ATTACK_BAG:-}" ]]; then
        rm -f "${ATTACK_BAG}" "${ATTACK_BAG}.active" 2>/dev/null || true
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
    python3 - <<PY
import json, zipfile, sys, os
p = "$zip_path"
if not os.path.exists(p):
    print("")
    sys.exit(0)
try:
    with zipfile.ZipFile(p) as z:
        stats = json.loads(z.read("stats.json"))
    print(stats.get("rmse", ""))
except Exception:
    print("")
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

    ape_rmse="$(extract_metric_from_evo_zip "$ape_zip")"
    rpe1_rmse="$(extract_metric_from_evo_zip "$rpe1_zip")"
    rpe10_rmse="$(extract_metric_from_evo_zip "$rpe10_zip")"

    echo "${run_id},${PLATFORM},${METHOD},${MODE},${DISTANCE_THRESHOLD},${SPOOFING_RANGE},${SPOOFER_X},${SPOOFER_Y},${ATTACK_BAG},${traj_bag},${traj_csv},${eval_dir},${ape_rmse},${rpe1_rmse},${rpe10_rmse},${status}" >> "$SUMMARY_CSV"
}

run_already_ok() {
    local run_id="$1"
    if [[ -f "$SUMMARY_CSV" ]]; then
        if grep -q "^${run_id},${PLATFORM},${METHOD},${MODE}," "$SUMMARY_CSV" && \
           grep "^${run_id},${PLATFORM},${METHOD},${MODE}," "$SUMMARY_CSV" | tail -n 1 | grep -q ",ok$"; then
            return 0
        fi
    fi
    return 1
}

write_run_config() {
    local run_config="$1"
    local attack_bag="$2"

    python3 - "$CONFIG_FILE" "$run_config" "$attack_bag" "$MODE" \
        "$SPOOFER_X" "$SPOOFER_Y" "$DISTANCE_THRESHOLD" "$SPOOFING_RANGE" <<'PY'
import json
import os
import sys

base_path, out_path, attack_bag, mode, sx, sy, dist_th, spoof_range = sys.argv[1:]
with open(base_path) as f:
    cfg = json.load(f)

cfg["main"]["output_file"] = attack_bag
cfg["main"]["spoofing_mode"] = mode
cfg["main"]["spoofer_x"] = float(sx)
cfg["main"]["spoofer_y"] = float(sy)
cfg["main"]["distance_threshold"] = float(dist_th)
cfg["main"]["spoofing_range"] = float(spoof_range)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(cfg, f, indent=2)
PY
}

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Base config not found: $CONFIG_FILE"
    exit 1
fi

echo "[INFO] Config file: $CONFIG_FILE"
echo "[INFO] Original CSV: $ORIG_CSV"
echo "[INFO] Output root: $OUT_ROOT"
echo "[INFO] Method: $METHOD"
echo "[INFO] Mode: $MODE"
echo "[INFO] Spoofer: ($SPOOFER_X, $SPOOFER_Y)"
echo "[INFO] Replay rate: $PLAY_RATE"
echo "[INFO] Post-play wait: $POST_PLAY_WAIT seconds"
echo "[INFO] Start run: $START_RUN"
echo "[INFO] Total runs: $N_RUNS"

if [[ ! -f "$ORIG_CSV" ]]; then
    echo "[ERROR] ORIG_CSV not found: $ORIG_CSV"
    echo "Please edit ORIG_CSV in this script."
    exit 1
fi

# -----------------------------
# Start ROS master
# -----------------------------

start_roscore_if_needed
rosparam set /use_sim_time true

# -----------------------------
# Initialize summary
# -----------------------------

if [[ "$RESET_SUMMARY" -eq 1 ]]; then
    echo "[INFO] RESET_SUMMARY=1, overwriting summary: $SUMMARY_CSV"
    cat > "$SUMMARY_CSV" <<EOF
run,platform,method,mode,distance_threshold,spoofing_range,spoofer_x,spoofer_y,attack_bag,traj_bag,traj_csv,eval_dir,ape_rmse,rpe_1m_rmse,rpe_10m_rmse,status
EOF
else
    echo "[INFO] RESET_SUMMARY=0, appending to existing summary if present."
    if [[ ! -f "$SUMMARY_CSV" ]]; then
        cat > "$SUMMARY_CSV" <<EOF
run,platform,method,mode,distance_threshold,spoofing_range,spoofer_x,spoofer_y,attack_bag,traj_bag,traj_csv,eval_dir,ape_rmse,rpe_1m_rmse,rpe_10m_rmse,status
EOF
    fi
fi

# -----------------------------
# Main loop
# -----------------------------

for RUN_INDEX in $(seq "$START_RUN" "$N_RUNS"); do
    RUN_ID="$(printf "%02d" "$RUN_INDEX")"

    echo "============================================================"
    echo "[RUN $RUN_ID/$N_RUNS] handheld smvs removal D=30 range=80"
    echo "============================================================"

    if run_already_ok "$RUN_ID"; then
        echo "[RUN $RUN_ID] Already marked ok in summary.csv, skipping."
        continue
    fi

    RUN_DIR="$RUNS_DIR/run_${RUN_ID}"
    mkdir -p "$RUN_DIR"

    ATTACK_BAG="$RUN_DIR/handheld_attack_removal_smvs_3080_run_${RUN_ID}.bag"
    RUN_CONFIG="$RUN_DIR/config_smvs_removal_${RUN_ID}.json"
    TRAJ_BAG="$RUN_DIR/handheld_attack_removal_smvs_3080_run_${RUN_ID}_traj.bag"
    TRAJ_CSV="$RUN_DIR/handheld_attack_removal_smvs_3080_run_${RUN_ID}_traj.csv"
    EVAL_DIR="$RUN_DIR/eval"
    mkdir -p "$EVAL_DIR"

    EDIT_LOG="$RUN_DIR/01_generate_attacked_bag.log"
    LVI_LOG="$RUN_DIR/02_lvisam.log"
    RECORD_LOG="$RUN_DIR/03_record.log"
    PLAY_LOG="$RUN_DIR/04_play.log"
    EXTRACT_LOG="$RUN_DIR/05_extract_csv.log"
    EVAL_LOG="$RUN_DIR/06_eval.log"

    REC_PID=""
    LVI_PID=""

    rm -f "$TRAJ_BAG" "$TRAJ_BAG.active" "$TRAJ_CSV"
    rm -rf "$EVAL_DIR"
    mkdir -p "$EVAL_DIR"

    cleanup_lvi_processes
    rosparam set /use_sim_time true

    # --------------------------------------------------------
    # Stage 4: generate attacked bag
    # --------------------------------------------------------
    if [[ "$REGENERATE_ATTACK_BAG_EACH_RUN" -eq 1 || "$RUN_INDEX" -eq 1 ]]; then
        echo "[RUN $RUN_ID] Stage 4: generating attacked bag ..."
        rm -f "$ATTACK_BAG" "$ATTACK_BAG.active"
        write_run_config "$RUN_CONFIG" "$ATTACK_BAG"

        set +e
        roslaunch slamspoof_icra rosbag_editer_lvisam.launch \
            config_file_path:="$RUN_CONFIG" \
            > "$EDIT_LOG" 2>&1
        EDIT_STATUS=$?
        set -e

        if [[ "$EDIT_STATUS" -ne 0 ]]; then
            echo "[ERROR] Attack bag generation command failed with status $EDIT_STATUS"
            append_summary_row "$RUN_ID" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_generate_attack_bag_cmd"
            cleanup_large_outputs_after_run
            continue
        fi

        if [[ ! -f "$ATTACK_BAG" ]]; then
            echo "[ERROR] Attacked bag was not generated: $ATTACK_BAG"
            append_summary_row "$RUN_ID" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_generate_attack_bag"
            cleanup_large_outputs_after_run
            continue
        fi
    else
        echo "[RUN $RUN_ID] Reusing attacked bag: $ATTACK_BAG"
        if [[ ! -f "$ATTACK_BAG" ]]; then
            echo "[ERROR] REGENERATE_ATTACK_BAG_EACH_RUN=0 but attacked bag does not exist: $ATTACK_BAG"
            append_summary_row "$RUN_ID" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "missing_reused_attack_bag"
            cleanup_large_outputs_after_run
            continue
        fi
    fi

    # --------------------------------------------------------
    # Stage 5: get attack trajectory
    # --------------------------------------------------------

    echo "[RUN $RUN_ID] Starting LVI-SAM ..."
    rosparam set /use_sim_time true

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used \
            --format=csv,noheader > "$RUN_DIR/gpu_before.log" 2>/dev/null || true
        echo "[RUN $RUN_ID] GPU before: $(cat "$RUN_DIR/gpu_before.log" 2>/dev/null || echo '?')"
    fi

    roslaunch "$LVI_LAUNCH_PKG" "$LVI_LAUNCH_FILE" \
        > "$LVI_LOG" 2>&1 &
    LVI_PID=$!

    sleep "$LVI_START_WAIT"

    echo "[RUN $RUN_ID] Starting rosbag record ..."
    rosbag record -O "$TRAJ_BAG" "$ODOM_TOPIC" \
        > "$RECORD_LOG" 2>&1 &
    REC_PID=$!

    sleep "$RECORD_START_WAIT"

    echo "[RUN $RUN_ID] Playing attacked bag at rate ${PLAY_RATE} ..."
    set +e
    rosbag play "$ATTACK_BAG" --clock -r "$PLAY_RATE" \
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

    # --------------------------------------------------------
    # Extract CSV
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Sanity check: did LVI actually run the full bag?
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    echo "[RUN $RUN_ID] Evaluating ..."
    set +e
    python3 "$SLAMSPOOF_DIR/scripts/evaluate_attack.py" \
        --orig "$ORIG_CSV" \
        --att "$TRAJ_CSV" \
        --out-dir "$EVAL_DIR" \
        --title "handheld smvs removal D30 R80 run ${RUN_ID}" \
        --spoofer-x "$SPOOFER_X" \
        --spoofer-y "$SPOOFER_Y" \
        --distance-threshold "$DISTANCE_THRESHOLD" \
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

if os.path.exists(summary):
    with open(summary) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 16:
                continue
            run_id = row[0]
            status = row[15]
            try:
                ape = float(row[12])
            except Exception:
                ape = None
            rows.append((run_id, ape, status))

ok = [r for r in rows if r[2] == "ok" and r[1] is not None]
fail = [r for r in rows if not (r[2] == "ok" and r[1] is not None)]
apes = [r[1] for r in ok]

print()
print(f"  Total rows : {len(rows)}")
print(f"  OK         : {len(ok)}")
print(f"  Failed     : {len(fail)}  ({', '.join(f'{r[0]}({r[2]})' for r in fail) or 'none'})")

if apes:
    print(f"  APE min    : {min(apes):.2f} m")
    print(f"  APE median : {statistics.median(apes):.2f} m")
    print(f"  APE mean   : {statistics.mean(apes):.2f} m")
    print(f"  APE max    : {max(apes):.2f} m")
    if len(apes) > 1:
        print(f"  APE std    : {statistics.stdev(apes):.2f} m")
    print()
    print("  Per-row APE:")
    for run_id, ape, status in rows:
        if ape is None:
            print(f"    {run_id}:        NA  [{status}]")
        else:
            print(f"    {run_id}: {ape:8.2f} m  [{status}]")
PY
