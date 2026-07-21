#!/usr/bin/env bash
set -euo pipefail

# Run one FAST-LIVO2 official-dataset experiment.
#
# Clean:
#   MODE=clean RUN_NAME=clean_01 bash run_fast_livo2_official_once.sh
#
# Attack, after clean trajectory exists:
#   MODE=static RUN_NAME=static_01 SPOOFER_X=... SPOOFER_Y=... \
#     REF_CSV=.../clean_01_traj.csv bash run_fast_livo2_official_once.sh

MODE="${MODE:-clean}"              # clean | static | removal
METHOD_LABEL="${METHOD_LABEL:-fast_livo2}"
RUN_NAME="${RUN_NAME:-fast_livo2_${MODE}_01}"
PLAY_RATE="${PLAY_RATE:-1.0}"
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}"
SPOOFING_RANGE="${SPOOFING_RANGE:-80}"
ATTACK_RNG_SEED="${ATTACK_RNG_SEED:-42}"

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
WS_SETUP="${WS_SETUP:-$HOME/catkin_ws/devel_catkin_tools/setup.bash}"
SLAMSPOOF_DIR="${SLAMSPOOF_DIR:-$HOME/catkin_ws/src/slamspoof}"

DATA_ROOT="${DATA_ROOT:-$HOME/catkin_ws/datasets/official/fast_livo2}"
BAG="${BAG:-$DATA_ROOT/raw_rosbags/FAST-LIVO2-Dataset/Bright_Screen_Wall.bag}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/experiments}"
REF_CSV="${REF_CSV:-}"
POSITION_CSV="${POSITION_CSV:-$OUT_ROOT/spoofer_positions_1580.csv}"

ODOM_TOPIC="${ODOM_TOPIC:-/aft_mapped_to_init}"
RUN_DIR="$OUT_ROOT/runs/$RUN_NAME"
LOG_DIR="$OUT_ROOT/logs"
EVAL_DIR="$RUN_DIR/eval"
TRAJ_BAG="$RUN_DIR/${RUN_NAME}_traj.bag"
TRAJ_CSV="$RUN_DIR/${RUN_NAME}_traj.csv"
ATTACK_BAG="$RUN_DIR/${RUN_NAME}_attacked.bag"
ATTACK_CONFIG="$RUN_DIR/${RUN_NAME}_attack_config.json"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

TARGET_START_WAIT="${TARGET_START_WAIT:-15}"
RECORD_START_WAIT="${RECORD_START_WAIT:-2}"
POST_PLAY_WAIT="${POST_PLAY_WAIT:-12}"
STOP_WAIT="${STOP_WAIT:-5}"
MIN_CSV_ROWS="${MIN_CSV_ROWS:-50}"
KEEP_ATTACK_BAG="${KEEP_ATTACK_BAG:-1}"
RVIZ="${RVIZ:-false}"

if [[ "$MODE" != "clean" && "$MODE" != "static" && "$MODE" != "removal" ]]; then
    echo "[ERROR] MODE must be clean, static, or removal. Got: $MODE" >&2
    exit 2
fi

mkdir -p "$RUN_DIR" "$LOG_DIR" "$EVAL_DIR"

set +u
source "$ROS_SETUP"
source "$WS_SETUP"
set -u

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
REC_PID=""
STARTED_ROSCORE=0

cleanup() {
    safe_kill "$REC_PID" "rosbag record"
    safe_kill "$TARGET_PID" "FAST-LIVO2 roslaunch"
    if [[ "$STARTED_ROSCORE" -eq 1 ]]; then
        safe_kill "$ROSCORE_PID" "roscore"
    fi
}
trap cleanup EXIT

start_roscore_if_needed() {
    if rosparam list >/dev/null 2>&1; then
        echo "[INFO] Existing ROS master detected."
        return
    fi
    roscore > "$LOG_DIR/${RUN_NAME}_roscore.log" 2>&1 &
    ROSCORE_PID=$!
    STARTED_ROSCORE=1
    wait_for_ros_master 30 || {
        echo "[ERROR] roscore did not start. Check $LOG_DIR/${RUN_NAME}_roscore.log" >&2
        exit 1
    }
}

read_position_if_needed() {
    if [[ -n "${SPOOFER_X:-}" && -n "${SPOOFER_Y:-}" ]]; then
        return
    fi
    if [[ ! -f "$POSITION_CSV" ]]; then
        echo "[ERROR] SPOOFER_X/Y not set and POSITION_CSV not found: $POSITION_CSV" >&2
        exit 2
    fi
    read -r SPOOFER_X SPOOFER_Y < <(python3 - "$POSITION_CSV" <<'PY'
import csv, sys
with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("empty position csv")
print(rows[0]["spoofer_x"], rows[0]["spoofer_y"])
PY
)
}

write_attack_config() {
    python3 - "$ATTACK_CONFIG" "$BAG" "$ATTACK_BAG" "$REF_CSV" "$MODE" \
        "$SPOOFER_X" "$SPOOFER_Y" "$DISTANCE_THRESHOLD" "$SPOOFING_RANGE" \
        "$ATTACK_RNG_SEED" <<'PY'
import json, os, sys
out, bag, attack_bag, ref_csv, mode, sx, sy, dist, spoof_range, seed = sys.argv[1:]
cfg = {
    "main": {
        "input_file": bag,
        "output_file": attack_bag,
        "reference_file": ref_csv,
        "spoofing_mode": mode,
        "spoofer_x": float(sx),
        "spoofer_y": float(sy),
        "distance_threshold": float(dist),
        "lidar_topic": "/livox/lidar",
        "spoofing_range": float(spoof_range),
        "wall_dist": 15.0,
        "rng_seed": int(seed),
        "point_count_model": "original",
        "static_geometry_model": "beam_project",
    },
    "simulator": {
        "horizontal_resolution": 0.1,
        "vertical_lines": 6.0,
        "spoofing_rate": 0.3,
    },
    "filtering": {
        "minimum_measuring_distance": 0.0,
        "maximum_measuring_distance": 30.0,
    },
}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
PY
}

RUN_BAG="$BAG"
if [[ "$MODE" != "clean" ]]; then
    if [[ -z "$REF_CSV" ]]; then
        echo "[ERROR] REF_CSV is required for attack modes." >&2
        exit 2
    fi
    if [[ ! -f "$REF_CSV" ]]; then
        echo "[ERROR] REF_CSV not found: $REF_CSV" >&2
        exit 2
    fi
    read_position_if_needed
    write_attack_config
    echo "[INFO] Generating attacked bag: $ATTACK_BAG"
    python3 "$SLAMSPOOF_DIR/scripts/spoofing_editer_livox_fastlivo2.py" \
        --config "$ATTACK_CONFIG" > "$RUN_DIR/01_generate_attacked_bag.log" 2>&1
    RUN_BAG="$ATTACK_BAG"
fi

start_roscore_if_needed
rosparam set use_sim_time true

echo "[INFO] Launching FAST-LIVO2 ..."
roslaunch fast_livo mapping_avia.launch rviz:="$RVIZ" \
    > "$RUN_DIR/02_fast_livo2.log" 2>&1 &
TARGET_PID=$!
sleep "$TARGET_START_WAIT"
if ! kill -0 "$TARGET_PID" 2>/dev/null; then
    echo "[ERROR] FAST-LIVO2 exited early. Check $RUN_DIR/02_fast_livo2.log" >&2
    exit 1
fi

echo "[INFO] Recording $ODOM_TOPIC ..."
rosbag record -O "$TRAJ_BAG" "$ODOM_TOPIC" \
    > "$RUN_DIR/03_record.log" 2>&1 &
REC_PID=$!
sleep "$RECORD_START_WAIT"

echo "[INFO] Playing $RUN_BAG at rate $PLAY_RATE ..."
set +e
rosbag play "$RUN_BAG" --clock -r "$PLAY_RATE" \
    > "$RUN_DIR/04_play.log" 2>&1
PLAY_STATUS=$?
set -e
sleep "$POST_PLAY_WAIT"

safe_kill "$TARGET_PID" "FAST-LIVO2 roslaunch"
TARGET_PID=""
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
    > "$RUN_DIR/05_extract_csv.log" 2>&1

CSV_ROWS=$(($(wc -l < "$TRAJ_CSV") - 1))
if [[ "$CSV_ROWS" -lt "$MIN_CSV_ROWS" ]]; then
    echo "[ERROR] CSV has only $CSV_ROWS rows (< $MIN_CSV_ROWS)." >&2
    exit 1
fi

APE=""
RPE1=""
RPE10=""
RPEMAX=""
STATUS="ok"
if [[ "$MODE" != "clean" && -n "$REF_CSV" ]]; then
    python3 "$SLAMSPOOF_DIR/scripts/evaluate_attack.py" \
        --orig "$REF_CSV" \
        --att "$TRAJ_CSV" \
        --out-dir "$EVAL_DIR" \
        --title "FAST-LIVO2 official: $MODE $RUN_NAME" \
        --spoofer-x "$SPOOFER_X" \
        --spoofer-y "$SPOOFER_Y" \
        --distance-threshold "$DISTANCE_THRESHOLD" \
        > "$RUN_DIR/06_eval.log" 2>&1 || STATUS="eval_failed"

    if [[ -f "$EVAL_DIR/metrics_complete.json" ]]; then
        mapfile -t METRIC_VALUES < <(python3 - "$EVAL_DIR/metrics_complete.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
vals = [
    j["evo"]["ape_translation"].get("rmse", ""),
    j["evo"]["rpe_1m_translation"].get("rmse", ""),
    j["evo"]["rpe_10m_translation"].get("rmse", ""),
    j["paper_metrics"].get("rpe_translation_max_m", ""),
]
for v in vals:
    print("" if v is None else str(v))
PY
)
        APE="${METRIC_VALUES[0]:-}"
        RPE1="${METRIC_VALUES[1]:-}"
        RPE10="${METRIC_VALUES[2]:-}"
        RPEMAX="${METRIC_VALUES[3]:-}"
    fi
fi

if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "run,method,mode,distance_threshold,spoofing_range,spoofer_x,spoofer_y,input_bag,traj_bag,traj_csv,eval_dir,ape_rmse,rpe_1m_rmse,rpe_10m_rmse,rpe_max,status" > "$SUMMARY_CSV"
fi
SUMMARY_SPOOFER_X="${SPOOFER_X:-NA}"
SUMMARY_SPOOFER_Y="${SPOOFER_Y:-NA}"
if [[ "$MODE" == "clean" ]]; then
    SUMMARY_SPOOFER_X="NA"
    SUMMARY_SPOOFER_Y="NA"
fi
echo "${RUN_NAME},${METHOD_LABEL},${MODE},${DISTANCE_THRESHOLD},${SPOOFING_RANGE},${SUMMARY_SPOOFER_X},${SUMMARY_SPOOFER_Y},${RUN_BAG},${TRAJ_BAG},${TRAJ_CSV},${EVAL_DIR},${APE},${RPE1},${RPE10},${RPEMAX},${STATUS}" >> "$SUMMARY_CSV"

if [[ "$MODE" != "clean" && "$KEEP_ATTACK_BAG" -eq 0 ]]; then
    rm -f "$ATTACK_BAG" "$ATTACK_BAG.active"
fi

echo "[OK] FAST-LIVO2 $MODE run complete:"
echo "     traj_csv: $TRAJ_CSV"
echo "     summary : $SUMMARY_CSV"
