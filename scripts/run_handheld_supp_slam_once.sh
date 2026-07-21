#!/usr/bin/env bash
set -euo pipefail

# Single-run helper for supplementary SLAM targets on handheld bags.
#
# This script is intentionally not used by the main LVI-SAM experiment.
# Run it only after the main experiment is idle, because it starts a ROS master,
# a target SLAM, rosbag record, and rosbag play.
#
# Environment variables:
#   TARGET     fast_livo2 | r3live
#   BAG        clean or attacked input bag
#   RUN_NAME   output subdirectory name
#   REF_CSV    optional reference CSV for evaluation
#   OUT_ROOT   output root

TARGET="${TARGET:-fast_livo2}"
BAG="${BAG:-$HOME/catkin_ws/src/LVI-SAM/datasets/handheld.bag}"
RUN_NAME="${RUN_NAME:-${TARGET}_clean_smoke}"
REF_CSV="${REF_CSV:-}"
OUT_ROOT="${OUT_ROOT:-$HOME/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/supplementary/${TARGET}}"

ROS_SETUP="/opt/ros/noetic/setup.bash"
MAIN_WS_SETUP="$HOME/catkin_ws/devel_catkin_tools/setup.bash"
SLAMSPOOF_DIR="$HOME/catkin_ws/src/slamspoof"

ODOM_TOPIC="/aft_mapped_to_init"
PLAY_RATE="${PLAY_RATE:-0.8}"
TARGET_START_WAIT="${TARGET_START_WAIT:-25}"
RECORD_START_WAIT="${RECORD_START_WAIT:-2}"
POST_PLAY_WAIT="${POST_PLAY_WAIT:-20}"
STOP_WAIT="${STOP_WAIT:-5}"
MIN_CSV_ROWS="${MIN_CSV_ROWS:-100}"

RUN_DIR="$OUT_ROOT/runs/$RUN_NAME"
LOG_DIR="$OUT_ROOT/logs"
TRAJ_BAG="$RUN_DIR/${RUN_NAME}_traj.bag"
TRAJ_CSV="$RUN_DIR/${RUN_NAME}_traj.csv"
EVAL_DIR="$RUN_DIR/eval"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$EVAL_DIR"

export ROS_HOME="${ROS_HOME:-/tmp/ros_home_slamspoof_supp}"
mkdir -p "$ROS_HOME"

source "$ROS_SETUP"
source "$MAIN_WS_SETUP"

case "$TARGET" in
    fast_livo2)
        TARGET_LAUNCH_PKG="slamspoof_icra"
        TARGET_LAUNCH_FILE="run_fast_livo2_handheld_smoke.launch"
        ;;
    r3live)
        TARGET_LAUNCH_PKG="slamspoof_icra"
        TARGET_LAUNCH_FILE="run_r3live_handheld_smoke.launch"
        ;;
    *)
        echo "[ERROR] Unsupported TARGET=$TARGET. Use fast_livo2 or r3live." >&2
        exit 2
        ;;
esac

if [[ ! -f "$BAG" ]]; then
    echo "[ERROR] BAG not found: $BAG" >&2
    exit 2
fi

safe_kill() {
    local pid="${1:-}"
    local name="${2:-process}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[INFO] Stopping $name pid=$pid ..."
        kill -INT "$pid" 2>/dev/null || true
        sleep "$STOP_WAIT"
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
        fi
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    fi
}

ROSCORE_PID=""
TARGET_PID=""
REC_PID=""

cleanup() {
    safe_kill "$TARGET_PID" "$TARGET roslaunch"
    safe_kill "$REC_PID" "rosbag record"
    safe_kill "$ROSCORE_PID" "roscore"
}
trap cleanup EXIT

if ! rosparam list >/dev/null 2>&1; then
    roscore > "$LOG_DIR/${RUN_NAME}_roscore.log" 2>&1 &
    ROSCORE_PID=$!
    sleep 5
fi

rosparam set use_sim_time true

echo "[INFO] Launching $TARGET ..."
roslaunch "$TARGET_LAUNCH_PKG" "$TARGET_LAUNCH_FILE" \
    > "$RUN_DIR/target_launch.log" 2>&1 &
TARGET_PID=$!
sleep "$TARGET_START_WAIT"

if ! kill -0 "$TARGET_PID" 2>/dev/null; then
    echo "[ERROR] $TARGET launch exited early. Check $RUN_DIR/target_launch.log" >&2
    exit 1
fi

echo "[INFO] Recording $ODOM_TOPIC ..."
rosbag record -O "$TRAJ_BAG" "$ODOM_TOPIC" \
    > "$RUN_DIR/record.log" 2>&1 &
REC_PID=$!
sleep "$RECORD_START_WAIT"

echo "[INFO] Playing $BAG at rate $PLAY_RATE ..."
set +e
rosbag play "$BAG" --clock -r "$PLAY_RATE" \
    > "$RUN_DIR/play.log" 2>&1
PLAY_STATUS=$?
set -e

sleep "$POST_PLAY_WAIT"
safe_kill "$TARGET_PID" "$TARGET roslaunch"
TARGET_PID=""
sleep 2
safe_kill "$REC_PID" "rosbag record"
REC_PID=""

if [[ "$PLAY_STATUS" -ne 0 ]]; then
    echo "[ERROR] rosbag play failed with status $PLAY_STATUS" >&2
    exit 1
fi

python3 "$SLAMSPOOF_DIR/scripts/extract_lvisam_odom_csv.py" \
    --bag "$TRAJ_BAG" \
    --topic "$ODOM_TOPIC" \
    --out "$TRAJ_CSV" \
    > "$RUN_DIR/extract.log" 2>&1

CSV_ROWS=$(wc -l < "$TRAJ_CSV")
CSV_ROWS=$((CSV_ROWS - 1))
if [[ "$CSV_ROWS" -lt "$MIN_CSV_ROWS" ]]; then
    echo "[ERROR] CSV has only $CSV_ROWS rows (< $MIN_CSV_ROWS)." >&2
    exit 1
fi

if [[ -n "$REF_CSV" ]]; then
    python3 "$SLAMSPOOF_DIR/scripts/evaluate_attack.py" \
        --orig "$REF_CSV" \
        --att "$TRAJ_CSV" \
        --out-dir "$EVAL_DIR" \
        --title "$TARGET: $RUN_NAME" \
        > "$RUN_DIR/eval.log" 2>&1
fi

echo "[OK] $TARGET run complete: $TRAJ_CSV"
