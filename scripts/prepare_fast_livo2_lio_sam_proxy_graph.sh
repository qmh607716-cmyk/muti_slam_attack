#!/usr/bin/env bash
set -euo pipefail

# Build an attacker-side LIO-SAM proxy graph for the FAST-LIVO2 official bag.
# The generated dump directory can be consumed by
# prepare_fast_livo2_method_positions.sh via GRAPH_SOURCE=lio_sam.

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
WS_SETUP="${WS_SETUP:-$HOME/catkin_ws/devel_catkin_tools/setup.bash}"
SLAMSPOOF_DIR="${SLAMSPOOF_DIR:-$HOME/catkin_ws/src/slamspoof}"

DATA_ROOT="${DATA_ROOT:-$HOME/catkin_ws/datasets/official/fast_livo2}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/experiments}"
BAG="${BAG:-$DATA_ROOT/raw_rosbags/FAST-LIVO2-Dataset/Bright_Screen_Wall.bag}"
PROXY_ROOT="${PROXY_ROOT:-$OUT_ROOT/lio_sam_proxy}"
LIO_BAG="${LIO_BAG:-$PROXY_ROOT/fast_livo2_lio_proxy.bag}"
GRAPH_DUMP_DIR="${GRAPH_DUMP_DIR:-$OUT_ROOT/lio_proxy_graph_dumps}"
LOG_DIR="${LOG_DIR:-$PROXY_ROOT/logs}"

PLAY_RATE="${PLAY_RATE:-1.0}"
CONVERT_BAG="${CONVERT_BAG:-1}"
TARGET_START_WAIT="${TARGET_START_WAIT:-8}"
POST_PLAY_WAIT="${POST_PLAY_WAIT:-8}"
STOP_WAIT="${STOP_WAIT:-5}"

if [[ ! -f "$BAG" ]]; then
    echo "[ERROR] FAST-LIVO2 bag not found: $BAG" >&2
    exit 2
fi

mkdir -p "$PROXY_ROOT" "$GRAPH_DUMP_DIR" "$LOG_DIR"

set +u
source "$ROS_SETUP"
source "$WS_SETUP"
set -u

if [[ "$CONVERT_BAG" -eq 1 || ! -f "$LIO_BAG" ]]; then
    echo "[STEP] Convert FAST-LIVO2 Livox bag to LIO-SAM proxy bag"
    python3 "$SLAMSPOOF_DIR/scripts/convert_fast_livo2_livox_to_liosam_bag.py" \
        --input "$BAG" \
        --output "$LIO_BAG" \
        > "$LOG_DIR/01_convert_bag.log" 2>&1
else
    echo "[SKIP] Reusing converted LIO-SAM proxy bag: $LIO_BAG"
fi

rm -rf "$GRAPH_DUMP_DIR"
mkdir -p "$GRAPH_DUMP_DIR"

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

wait_for_ros_master() {
    local max_wait="${1:-30}"
    for _ in $(seq 1 "$max_wait"); do
        if rosparam list >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

ROSCORE_PID=""
TARGET_PID=""
STARTED_ROSCORE=0
cleanup() {
    safe_kill "$TARGET_PID" "LIO-SAM proxy roslaunch"
    if [[ "$STARTED_ROSCORE" -eq 1 ]]; then
        safe_kill "$ROSCORE_PID" "roscore"
    fi
}
trap cleanup EXIT

if rosparam list >/dev/null 2>&1; then
    echo "[INFO] Existing ROS master detected."
else
    roscore > "$LOG_DIR/00_roscore.log" 2>&1 &
    ROSCORE_PID=$!
    STARTED_ROSCORE=1
    wait_for_ros_master 30 || {
        echo "[ERROR] roscore did not start. Check $LOG_DIR/00_roscore.log" >&2
        exit 1
    }
fi

rosparam set use_sim_time true

echo "[STEP] Launch LIO-SAM proxy and export graph dumps"
export LIO_GRAPH_DUMP_DIR="$GRAPH_DUMP_DIR"
roslaunch slamspoof_icra fast_livo2_lio_sam_proxy.launch \
    > "$LOG_DIR/02_lio_sam_proxy.log" 2>&1 &
TARGET_PID=$!
sleep "$TARGET_START_WAIT"
if ! kill -0 "$TARGET_PID" 2>/dev/null; then
    echo "[ERROR] LIO-SAM proxy exited early. Check $LOG_DIR/02_lio_sam_proxy.log" >&2
    exit 1
fi

echo "[STEP] Play converted LIO-SAM proxy bag"
rosbag play "$LIO_BAG" --clock -r "$PLAY_RATE" \
    > "$LOG_DIR/03_play_lio_proxy_bag.log" 2>&1
sleep "$POST_PLAY_WAIT"

safe_kill "$TARGET_PID" "LIO-SAM proxy roslaunch"
TARGET_PID=""

n_dumps=$(find "$GRAPH_DUMP_DIR" -maxdepth 1 -name 'dump_*.json' | wc -l)
if [[ "$n_dumps" -lt 2 ]]; then
    echo "[ERROR] Too few LIO-SAM graph dumps generated: $n_dumps" >&2
    echo "        Check $LOG_DIR/02_lio_sam_proxy.log" >&2
    exit 1
fi

echo "[OK] LIO-SAM proxy graph ready:"
echo "     graph_dumps: $GRAPH_DUMP_DIR"
echo "     dumps      : $n_dumps"
