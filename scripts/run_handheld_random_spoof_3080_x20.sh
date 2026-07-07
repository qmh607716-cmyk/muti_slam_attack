#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch experiment:
# handheld / Random feasible spoofer / static or removal
# D=30m / spoofing_range=80deg
#
# Usage:
#   MODE=static  bash run_handheld_random_spoof_3080_x20.sh
#   MODE=removal bash run_handheld_random_spoof_3080_x20.sh
#
# Random positions are sampled from the clean trajectory using geometry-only
# feasibility constraints. No SMVS/Bi-SMVS score is used.
# ============================================================

N_POSITIONS="${N_POSITIONS:-20}"
START_RUN="${START_RUN:-1}"
MODE="${MODE:-static}"
RANDOM_SEED="${RANDOM_SEED:-20260706}"
ATTACK_RNG_SEED="${ATTACK_RNG_SEED:-42}"

if [[ "$MODE" != "static" && "$MODE" != "removal" ]]; then
    echo "[ERROR] MODE must be static or removal. Got: $MODE"
    exit 1
fi

ROS_SETUP="/opt/ros/noetic/setup.bash"
WS_SETUP="$HOME/catkin_ws/devel_catkin_tools/setup.bash"

SLAMSPOOF_DIR="$HOME/catkin_ws/src/slamspoof"
LVI_DATASET_DIR="$HOME/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld"

BASE_CONFIG="$SLAMSPOOF_DIR/config_lvisam.json"
ORIG_CSV="$LVI_DATASET_DIR/original/handheld_original_traj.csv"

OUT_ROOT="$LVI_DATASET_DIR/repeat_${MODE}/random_3080"
POSITIONS_CSV="$OUT_ROOT/random_positions_seed_${RANDOM_SEED}.csv"

ODOM_TOPIC="/lvi_sam/lidar/mapping/odometry"
LVI_LAUNCH_PKG="lvi_sam"
LVI_LAUNCH_FILE="run.launch"

METHOD="random"
PLATFORM="handheld"
DISTANCE_THRESHOLD=30
SPOOFING_RANGE=80

LVI_START_WAIT=25
RECORD_START_WAIT=2
PLAY_RATE=0.8
POST_PLAY_WAIT=25
STOP_WAIT=5
MIN_CSV_ROWS=6000

KEEP_TRAJ_BAG="${KEEP_TRAJ_BAG:-0}"
DELETE_ATTACK_BAG_AFTER_RUN="${DELETE_ATTACK_BAG_AFTER_RUN:-1}"

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
    pkill -f "roslaunch.*lvi_sam.*run.launch" >/dev/null 2>&1 || true
    pkill -f "lvi_sam" >/dev/null 2>&1 || true
    pkill -f "rosbag record.*${ODOM_TOPIC}" >/dev/null 2>&1 || true
    sleep 3
}

cleanup_large_outputs_after_run() {
    if [[ "${KEEP_TRAJ_BAG}" -eq 0 ]]; then
        rm -f "$TRAJ_BAG" "$TRAJ_BAG.active" 2>/dev/null || true
    fi

    if [[ "${DELETE_ATTACK_BAG_AFTER_RUN}" -eq 1 ]]; then
        rm -f "$ATTACK_BAG" "$ATTACK_BAG.active" 2>/dev/null || true
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
    python3 - "$zip_path" <<'PY'
import json
import os
import sys
import zipfile

p = sys.argv[1]
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

read_random_position() {
    local run_index="$1"
    python3 - "$POSITIONS_CSV" "$run_index" <<'PY'
import csv
import sys

path = sys.argv[1]
idx = int(sys.argv[2])
with open(path) as f:
    rows = list(csv.DictReader(f))
if idx < 1 or idx > len(rows):
    raise SystemExit(f"run index {idx} is outside random position table size {len(rows)}")
r = rows[idx - 1]
fields = [
    "random_id", "spoofer_x", "spoofer_y", "anchor_idx",
    "min_traj_dist_m", "trigger_frames", "trigger_ratio",
    "trigger_start_s", "trigger_end_s",
]
print("\t".join(str(r[k]) for k in fields))
PY
}

write_run_config() {
    local run_config="$1"
    local attack_bag="$2"
    local spoofer_x="$3"
    local spoofer_y="$4"

    python3 - "$BASE_CONFIG" "$run_config" "$attack_bag" "$MODE" \
        "$spoofer_x" "$spoofer_y" "$DISTANCE_THRESHOLD" "$SPOOFING_RANGE" \
        "$ATTACK_RNG_SEED" <<'PY'
import json
import os
import sys

base_path, out_path, attack_bag, mode, sx, sy, dist_th, spoof_range, rng_seed = sys.argv[1:]
with open(base_path) as f:
    cfg = json.load(f)

cfg["main"]["output_file"] = attack_bag
cfg["main"]["spoofing_mode"] = mode
cfg["main"]["spoofer_x"] = float(sx)
cfg["main"]["spoofer_y"] = float(sy)
cfg["main"]["distance_threshold"] = float(dist_th)
cfg["main"]["spoofing_range"] = float(spoof_range)
cfg["main"]["rng_seed"] = int(rng_seed)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(cfg, f, indent=2)
PY
}

append_summary_row() {
    local run_id="$1"
    local random_id="$2"
    local anchor_idx="$3"
    local min_traj_dist="$4"
    local trigger_frames="$5"
    local trigger_ratio="$6"
    local trigger_start_s="$7"
    local trigger_end_s="$8"
    local traj_bag="$9"
    local traj_csv="${10}"
    local eval_dir="${11}"
    local status="${12}"

    local ape_zip="$eval_dir/evo_ape.txt"
    local rpe1_zip="$eval_dir/evo_rpe.txt"
    local rpe10_zip="$eval_dir/evo_rpe_10m.txt"

    local ape_rmse
    local rpe1_rmse
    local rpe10_rmse
    ape_rmse="$(extract_metric_from_evo_zip "$ape_zip")"
    rpe1_rmse="$(extract_metric_from_evo_zip "$rpe1_zip")"
    rpe10_rmse="$(extract_metric_from_evo_zip "$rpe10_zip")"

    echo "${run_id},${PLATFORM},${METHOD},${MODE},${DISTANCE_THRESHOLD},${SPOOFING_RANGE},${SPOOFER_X},${SPOOFER_Y},${random_id},${anchor_idx},${min_traj_dist},${trigger_frames},${trigger_ratio},${trigger_start_s},${trigger_end_s},${ATTACK_BAG},${traj_bag},${traj_csv},${eval_dir},${ape_rmse},${rpe1_rmse},${rpe10_rmse},${status}" >> "$SUMMARY_CSV"
}

echo "[INFO] Mode: $MODE"
echo "[INFO] Base config: $BASE_CONFIG"
echo "[INFO] Original CSV: $ORIG_CSV"
echo "[INFO] Output root: $OUT_ROOT"
echo "[INFO] Random seed: $RANDOM_SEED"
echo "[INFO] Attack RNG seed: $ATTACK_RNG_SEED"

if [[ ! -f "$ORIG_CSV" ]]; then
    echo "[ERROR] ORIG_CSV not found: $ORIG_CSV"
    exit 1
fi

if [[ ! -f "$POSITIONS_CSV" ]]; then
    echo "[INFO] Sampling random feasible spoofer positions ..."
    python3 "$SLAMSPOOF_DIR/scripts/sample_random_spoofer_positions.py" \
        --traj "$ORIG_CSV" \
        --out "$POSITIONS_CSV" \
        --n "$N_POSITIONS" \
        --seed "$RANDOM_SEED" \
        --distance-threshold "$DISTANCE_THRESHOLD" \
        --min-traj-dist 10 \
        --max-traj-dist 30 \
        --min-trigger-frames 50 \
        --min-trigger-ratio 0.005 \
        --max-trigger-ratio 0.25
else
    echo "[INFO] Reusing random positions: $POSITIONS_CSV"
fi

start_roscore_if_needed
rosparam set /use_sim_time true

if [[ ! -f "$SUMMARY_CSV" ]]; then
cat > "$SUMMARY_CSV" <<EOF
run,platform,method,mode,distance_threshold,spoofing_range,spoofer_x,spoofer_y,random_id,anchor_idx,min_traj_dist,trigger_frames,trigger_ratio,trigger_start_s,trigger_end_s,attack_bag,traj_bag,traj_csv,eval_dir,ape_rmse,rpe_1m_rmse,rpe_10m_rmse,status
EOF
fi

for RUN_INDEX in $(seq "$START_RUN" "$N_POSITIONS"); do
    RUN_ID="$(printf "%02d" "$RUN_INDEX")"

    IFS=$'\t' read -r RANDOM_ID SPOOFER_X SPOOFER_Y ANCHOR_IDX MIN_TRAJ_DIST TRIGGER_FRAMES TRIGGER_RATIO TRIGGER_START_S TRIGGER_END_S < <(read_random_position "$RUN_INDEX")

    echo "============================================================"
    echo "[RUN $RUN_ID/$N_POSITIONS] handheld random $MODE D=30 range=80"
    echo "[RUN $RUN_ID] Spoofer=($SPOOFER_X, $SPOOFER_Y), trigger=${TRIGGER_FRAMES} frames (${TRIGGER_RATIO})"
    echo "============================================================"

    RUN_DIR="$RUNS_DIR/run_${RUN_ID}"
    mkdir -p "$RUN_DIR"

    ATTACK_BAG="$RUN_DIR/handheld_attack_${MODE}_random_3080_run_${RUN_ID}.bag"
    RUN_CONFIG="$RUN_DIR/config_random_${MODE}_${RUN_ID}.json"
    TRAJ_BAG="$RUN_DIR/handheld_attack_${MODE}_random_3080_run_${RUN_ID}_traj.bag"
    TRAJ_CSV="$RUN_DIR/handheld_attack_${MODE}_random_3080_run_${RUN_ID}_traj.csv"
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

    rm -f "$ATTACK_BAG" "$ATTACK_BAG.active" "$TRAJ_BAG" "$TRAJ_BAG.active" "$TRAJ_CSV"
    rm -rf "$EVAL_DIR"
    mkdir -p "$EVAL_DIR"

    cleanup_lvi_processes
    rosparam set /use_sim_time true

    write_run_config "$RUN_CONFIG" "$ATTACK_BAG" "$SPOOFER_X" "$SPOOFER_Y"

    echo "[RUN $RUN_ID] Stage 4: generating attacked bag ..."
    set +e
    roslaunch slamspoof_icra rosbag_editer_lvisam.launch \
        config_file_path:="$RUN_CONFIG" \
        > "$EDIT_LOG" 2>&1
    EDIT_STATUS=$?
    set -e

    if [[ "$EDIT_STATUS" -ne 0 || ! -f "$ATTACK_BAG" ]]; then
        echo "[ERROR] Attack bag generation failed for run $RUN_ID"
        append_summary_row "$RUN_ID" "$RANDOM_ID" "$ANCHOR_IDX" "$MIN_TRAJ_DIST" "$TRIGGER_FRAMES" "$TRIGGER_RATIO" "$TRIGGER_START_S" "$TRIGGER_END_S" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_generate_attack_bag"
        cleanup_large_outputs_after_run
        continue
    fi

    echo "[RUN $RUN_ID] Starting LVI-SAM ..."
    rosparam set /use_sim_time true
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

    if [[ "$PLAY_STATUS" -ne 0 ]]; then
        echo "[ERROR] rosbag play failed with status $PLAY_STATUS"
        append_summary_row "$RUN_ID" "$RANDOM_ID" "$ANCHOR_IDX" "$MIN_TRAJ_DIST" "$TRIGGER_FRAMES" "$TRIGGER_RATIO" "$TRIGGER_START_S" "$TRIGGER_END_S" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_rosbag_play"
        cleanup_large_outputs_after_run
        continue
    fi

    if [[ ! -f "$TRAJ_BAG" ]]; then
        echo "[ERROR] Trajectory bag was not recorded: $TRAJ_BAG"
        append_summary_row "$RUN_ID" "$RANDOM_ID" "$ANCHOR_IDX" "$MIN_TRAJ_DIST" "$TRIGGER_FRAMES" "$TRIGGER_RATIO" "$TRIGGER_START_S" "$TRIGGER_END_S" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_record_traj_bag"
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
        append_summary_row "$RUN_ID" "$RANDOM_ID" "$ANCHOR_IDX" "$MIN_TRAJ_DIST" "$TRIGGER_FRAMES" "$TRIGGER_RATIO" "$TRIGGER_START_S" "$TRIGGER_END_S" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_extract_csv"
        cleanup_large_outputs_after_run
        continue
    fi

    CSV_ROWS=$(wc -l < "$TRAJ_CSV" 2>/dev/null || echo 0)
    CSV_ROWS=$((CSV_ROWS - 1))
    if [[ "$CSV_ROWS" -lt 0 ]]; then
        CSV_ROWS=0
    fi

    if [[ "$CSV_ROWS" -lt "$MIN_CSV_ROWS" ]]; then
        echo "[ERROR] run $RUN_ID: CSV has only $CSV_ROWS rows (< $MIN_CSV_ROWS)."
        tail -50 "$LVI_LOG" > "$RUN_DIR/lvi_crash_tail.log" 2>/dev/null || true
        append_summary_row "$RUN_ID" "$RANDOM_ID" "$ANCHOR_IDX" "$MIN_TRAJ_DIST" "$TRIGGER_FRAMES" "$TRIGGER_RATIO" "$TRIGGER_START_S" "$TRIGGER_END_S" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "truncated_csv_${CSV_ROWS}rows"
        cleanup_large_outputs_after_run
        continue
    fi

    echo "[RUN $RUN_ID] Evaluating ..."
    set +e
    python3 "$SLAMSPOOF_DIR/scripts/evaluate_attack.py" \
        --orig "$ORIG_CSV" \
        --att "$TRAJ_CSV" \
        --out-dir "$EVAL_DIR" \
        --title "handheld random ${MODE} D30 R80 run ${RUN_ID}" \
        --spoofer-x "$SPOOFER_X" \
        --spoofer-y "$SPOOFER_Y" \
        --distance-threshold "$DISTANCE_THRESHOLD" \
        > "$EVAL_LOG" 2>&1
    EVAL_STATUS=$?
    set -e

    if [[ "$EVAL_STATUS" -ne 0 ]]; then
        echo "[WARN] Evaluation script failed for run $RUN_ID. Check: $EVAL_LOG"
        append_summary_row "$RUN_ID" "$RANDOM_ID" "$ANCHOR_IDX" "$MIN_TRAJ_DIST" "$TRIGGER_FRAMES" "$TRIGGER_RATIO" "$TRIGGER_START_S" "$TRIGGER_END_S" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "failed_eval"
        cleanup_large_outputs_after_run
        continue
    fi

    append_summary_row "$RUN_ID" "$RANDOM_ID" "$ANCHOR_IDX" "$MIN_TRAJ_DIST" "$TRIGGER_FRAMES" "$TRIGGER_RATIO" "$TRIGGER_START_S" "$TRIGGER_END_S" "$TRAJ_BAG" "$TRAJ_CSV" "$EVAL_DIR" "ok"
    cleanup_large_outputs_after_run

    echo "[RUN $RUN_ID] Done."
    echo "[RUN $RUN_ID] Current summary: $SUMMARY_CSV"
done

echo "============================================================"
echo "[ALL DONE]"
echo "Summary CSV:"
echo "$SUMMARY_CSV"
echo "Random positions:"
echo "$POSITIONS_CSV"
echo "============================================================"
