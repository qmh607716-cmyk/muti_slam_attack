#!/usr/bin/env bash
set -euo pipefail

# Build FAST-LIVO2 official-dataset SMVS/Bi-SMVS placement inputs and select
# one spoofer position for each method. This is the transfer-experiment
# counterpart of the LVI-SAM main pipeline's selection stage.

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
WS_SETUP="${WS_SETUP:-$HOME/catkin_ws/devel_catkin_tools/setup.bash}"
SLAMSPOOF_DIR="${SLAMSPOOF_DIR:-$HOME/catkin_ws/src/slamspoof}"

DATA_ROOT="${DATA_ROOT:-$HOME/catkin_ws/datasets/official/fast_livo2}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/experiments}"
BAG="${BAG:-$DATA_ROOT/raw_rosbags/FAST-LIVO2-Dataset/Bright_Screen_Wall.bag}"
CLEAN_CSV="${CLEAN_CSV:-$OUT_ROOT/runs/clean_01/clean_01_traj.csv}"

SEQUENCE="${SEQUENCE:-Bright_Screen_Wall}"
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}"
SPOOFING_RANGE="${SPOOFING_RANGE:-80}"
PARAM_TAG="${PARAM_TAG:-1580}"
MIN_TRAJ_DIST="${MIN_TRAJ_DIST:-8}"
MAX_TRAJ_DIST="${MAX_TRAJ_DIST:-14}"
SMVS_TOP_K="${SMVS_TOP_K:-10}"
BISMVS_TOP_K="${BISMVS_TOP_K:-20}"
CMA_CALLS="${CMA_CALLS:-200}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/left_camera/image/compressed}"
RECOMPUTE_VULNERABILITY="${RECOMPUTE_VULNERABILITY:-1}"
MIN_TRIGGER_RATIO="${MIN_TRIGGER_RATIO:-0.30}"
MAX_TRIGGER_RATIO="${MAX_TRIGGER_RATIO:-0.70}"
TARGET_TRIGGER_RATIO="${TARGET_TRIGGER_RATIO:-0.50}"
PROXY_NODE_STRIDE="${PROXY_NODE_STRIDE:-5}"
PROXY_LOOP_RADIUS="${PROXY_LOOP_RADIUS:-1.0}"
PROXY_LOOP_MIN_GAP="${PROXY_LOOP_MIN_GAP:-40}"
GRAPH_SOURCE="${GRAPH_SOURCE:-auto}"
GRAPH_DUMP_DIR="${GRAPH_DUMP_DIR:-$OUT_ROOT/lio_proxy_graph_dumps}"

VUL_ROOT="$OUT_ROOT/vulnerability"
SEL_ROOT="$OUT_ROOT/selections"
METHOD_POS_CSV="$OUT_ROOT/method_spoofer_positions_${PARAM_TAG}.csv"

SMVS_CSV="$VUL_ROOT/smvs/${SEQUENCE}_SMVS.csv"
SMVS_VUL="$VUL_ROOT/vul/vul_${SEQUENCE}_SMVS.csv"
BISMVS_CSV="$VUL_ROOT/smvs/${SEQUENCE}_BiSMVS.csv"
BISMVS_VUL="$VUL_ROOT/vul/vul_${SEQUENCE}_BiSMVS.csv"

SMVS_JSON="$SEL_ROOT/smvs_${PARAM_TAG}.json"
BISMVS_JSON="$SEL_ROOT/bismvs_${PARAM_TAG}.json"
BISMVS_VIZ="$SEL_ROOT/bismvs_${PARAM_TAG}.png"

if [[ ! -f "$BAG" ]]; then
    echo "[ERROR] BAG not found: $BAG" >&2
    exit 2
fi
if [[ ! -f "$CLEAN_CSV" ]]; then
    echo "[ERROR] clean trajectory not found: $CLEAN_CSV" >&2
    echo "        Run MODE=clean RUN_NAME=clean_01 run_fast_livo2_official_once.sh first." >&2
    exit 2
fi

set +u
source "$ROS_SETUP"
source "$WS_SETUP"
set -u
mkdir -p "$VUL_ROOT" "$SEL_ROOT"

if [[ "$RECOMPUTE_VULNERABILITY" -eq 1 || ! -f "$SMVS_CSV" || ! -f "$BISMVS_CSV" ]]; then
    echo "[STEP] Compute FAST-LIVO2 SMVS/Bi-SMVS CSVs"
    compute_args=(
        --bag "$BAG"
        --traj "$CLEAN_CSV"
        --out-root "$VUL_ROOT"
        --sequence "$SEQUENCE"
        --frame-stride "$FRAME_STRIDE"
        --image-topic "$IMAGE_TOPIC"
    )
    if [[ -n "$MAX_FRAMES" ]]; then
        compute_args+=(--max-frames "$MAX_FRAMES")
    fi
    python3 "$SLAMSPOOF_DIR/scripts/compute_fast_livo2_vulnerability.py" "${compute_args[@]}"
else
    echo "[SKIP] Reusing existing vulnerability CSVs under $VUL_ROOT"
fi

echo "[STEP] Select FAST-LIVO2 SMVS position with paper III-C selector"
python3 "$SLAMSPOOF_DIR/scripts/select_spoofer_from_smvs_paper.py" \
    --smvs "$SMVS_CSV" \
    --vul "$SMVS_VUL" \
    --ref-traj "$CLEAN_CSV" \
    --output "$SMVS_JSON" \
    --top-k "$SMVS_TOP_K" \
    --score-threshold -1000 \
    --score-direction larger \
    --spoof-distance "$DISTANCE_THRESHOLD" \
    --distance-threshold "$DISTANCE_THRESHOLD" \
    --spoofing-range "$SPOOFING_RANGE" \
    --line-formula paper \
    --candidate-side same_as_center \
    --match-mode timestamp

echo "[STEP] Select FAST-LIVO2 Bi-SMVS position with proxy-assisted selector"
python3 "$SLAMSPOOF_DIR/scripts/select_fast_livo2_bismvs_position.py" \
    --smvs "$BISMVS_CSV" \
    --traj "$CLEAN_CSV" \
    --spoofing-range "$SPOOFING_RANGE" \
    --distance-threshold "$DISTANCE_THRESHOLD" \
    --min-traj-dist "$MIN_TRAJ_DIST" \
    --max-traj-dist "$MAX_TRAJ_DIST" \
    --min-trigger-ratio "$MIN_TRIGGER_RATIO" \
    --max-trigger-ratio "$MAX_TRIGGER_RATIO" \
    --target-trigger-ratio "$TARGET_TRIGGER_RATIO" \
    --proxy-node-stride "$PROXY_NODE_STRIDE" \
    --proxy-loop-radius "$PROXY_LOOP_RADIUS" \
    --proxy-loop-min-gap "$PROXY_LOOP_MIN_GAP" \
    --graph-source "$GRAPH_SOURCE" \
    --graph-dump-dir "$GRAPH_DUMP_DIR" \
    --top-k "$BISMVS_TOP_K" \
    --visualize \
    --viz-path "$BISMVS_VIZ" \
    --output "$BISMVS_JSON"

python3 - "$SMVS_JSON" "$BISMVS_JSON" "$METHOD_POS_CSV" <<'PY'
import csv
import json
import os
import sys

smvs_json, bismvs_json, out_csv = sys.argv[1:]

def smvs_pos(path):
    j = json.load(open(path))
    x = j.get("main", {}).get("spoofer_x")
    y = j.get("main", {}).get("spoofer_y")
    if x is None or y is None:
        raise SystemExit(f"SMVS selector did not return a unique spoofer position: {path}")
    return float(x), float(y)

def bismvs_pos(path):
    j = json.load(open(path))
    opt = j.get("optim", {})
    x = opt.get("spoofer_x")
    y = opt.get("spoofer_y")
    if x is None or y is None:
        raise SystemExit(f"Bi-SMVS selector did not return a spoofer position: {path}")
    return float(x), float(y)

rows = []
sx, sy = smvs_pos(smvs_json)
rows.append({"method": "smvs", "spoofer_x": sx, "spoofer_y": sy, "selection_json": smvs_json})
bx, by = bismvs_pos(bismvs_json)
rows.append({"method": "bismvs", "spoofer_x": bx, "spoofer_y": by, "selection_json": bismvs_json})

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "spoofer_x", "spoofer_y", "selection_json"])
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] wrote {out_csv}")
for r in rows:
    print(f"  {r['method']}: ({r['spoofer_x']:.6f}, {r['spoofer_y']:.6f})")
PY

echo "[OK] FAST-LIVO2 method positions are ready:"
echo "     $METHOD_POS_CSV"
