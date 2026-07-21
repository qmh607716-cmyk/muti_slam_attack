#!/usr/bin/env bash
set -euo pipefail

# Prepare static-attack ablation positions on the handheld LVI-SAM sequence.
#
# Groups:
#   smvs_paper          : LiDAR-only SMVS + SLAMSpoof III-C placement
#   smvs_graph_cma      : LiDAR-only SMVS + graph-aware CMA placement
#   bismvs_paper        : Bi-SMVS score + SLAMSpoof III-C placement
#   bismvs_cma_no_graph : Bi-SMVS score + CMA placement without graph cues
#   bismvs_graph_cma    : Bi-SMVS score + graph-aware CMA placement,
#                         frozen to the main evaluated position by default

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLAMSPOOF_DIR="$HOME/catkin_ws/src/slamspoof"
LVI_DATASET_DIR="$HOME/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld"

DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}"
SPOOFING_RANGE="${SPOOFING_RANGE:-80}"
SPOOF_DISTANCE="${SPOOF_DISTANCE:-15}"
TOP_K="${TOP_K:-20}"
CMA_CALLS="${CMA_CALLS:-300}"
SEED="${SEED:-42}"
MIN_TRAJ_DIST="${MIN_TRAJ_DIST:-5}"
MAX_TRAJ_DIST="${MAX_TRAJ_DIST:-30}"
FULL_COORD_SOURCE="${FULL_COORD_SOURCE:-main_evaluated}"
FULL_SPOOFER_X="${FULL_SPOOFER_X:-31.28075677647965}"
FULL_SPOOFER_Y="${FULL_SPOOFER_Y:--102.07423272183334}"

SMVS_CSV="${SMVS_CSV:-$LVI_DATASET_DIR/smvs/05_31_14_52_49_SMVS.csv}"
SMVS_VUL_CSV="${SMVS_VUL_CSV:-$LVI_DATASET_DIR/vul/vul_05_31_14_52_49_SMVS.csv}"
BISMVS_CSV="${BISMVS_CSV:-$LVI_DATASET_DIR/smvs/07_08_23_25_42_BiSMVS.csv}"
BISMVS_VUL_CSV="${BISMVS_VUL_CSV:-$LVI_DATASET_DIR/vul/vul_07_08_23_25_42_BiSMVS.csv}"
TRAJ_CSV="${TRAJ_CSV:-$LVI_DATASET_DIR/original/handheld_original_traj.csv}"
GRAPH_DUMP_DIR="${GRAPH_DUMP_DIR:-$LVI_DATASET_DIR/graph_dumps}"

OUT_ROOT="${OUT_ROOT:-$LVI_DATASET_DIR/ablation/positions_1580}"
SEL_DIR="$OUT_ROOT/selections"
TMP_DIR="$OUT_ROOT/tmp"
POSITION_CSV="$OUT_ROOT/method_spoofer_positions_1580.csv"

mkdir -p "$SEL_DIR" "$TMP_DIR"

for required in "$SMVS_CSV" "$SMVS_VUL_CSV" "$BISMVS_CSV" "$BISMVS_VUL_CSV" "$TRAJ_CSV"; do
    if [[ ! -f "$required" ]]; then
        echo "[ERROR] missing input: $required" >&2
        exit 1
    fi
done

if [[ ! -d "$GRAPH_DUMP_DIR" ]]; then
    echo "[ERROR] graph dump dir not found: $GRAPH_DUMP_DIR" >&2
    exit 1
fi

SMVS_PAPER_JSON="$SEL_DIR/smvs_paper_1580.json"
SMVS_GRAPH_CMA_JSON="$SEL_DIR/smvs_graph_cma_1580.json"
BISMVS_PAPER_JSON="$SEL_DIR/bismvs_paper_1580.json"
BISMVS_CMA_NO_GRAPH_JSON="$SEL_DIR/bismvs_cma_no_graph_1580.json"
BISMVS_GRAPH_CMA_JSON="$SEL_DIR/bismvs_graph_cma_recomputed_1580.json"
BISMVS_GRAPH_CMA_EVALUATED_JSON="$SEL_DIR/bismvs_graph_cma_main_evaluated_1580.json"

BISMVS_AS_SMVS="$TMP_DIR/bismvs_as_smvs_for_paper_selector.csv"
BISMVS_AS_VUL="$TMP_DIR/bismvs_as_vul_for_paper_selector.csv"
SMVS_GRAPH_CMA_SMVS="$TMP_DIR/smvs_graph_cma_smvs.csv"
SMVS_GRAPH_CMA_VUL="$TMP_DIR/smvs_graph_cma_vul.csv"

echo "[STEP] Build Bi-SMVS adapter CSVs for the paper selector"
python3 - "$BISMVS_CSV" "$BISMVS_VUL_CSV" "$BISMVS_AS_SMVS" "$BISMVS_AS_VUL" <<'PY'
import os
import sys
import pandas as pd

bismvs_path, bivul_path, out_smvs, out_vul = sys.argv[1:]
bismvs = pd.read_csv(bismvs_path)
bivul = pd.read_csv(bivul_path)

required_smvs = {"timestamp", "x", "y", "z", "frame_bi_smvs"}
required_vul = {"timestamp", "x", "y", "z", "vec_x", "vec_y", "frame_bi_smvs"}
missing_smvs = required_smvs - set(bismvs.columns)
missing_vul = required_vul - set(bivul.columns)
if missing_smvs:
    raise SystemExit(f"Bi-SMVS CSV missing columns: {sorted(missing_smvs)}")
if missing_vul:
    raise SystemExit(f"Bi-SMVS vulnerability CSV missing columns: {sorted(missing_vul)}")

os.makedirs(os.path.dirname(out_smvs), exist_ok=True)
bismvs[["timestamp", "x", "y", "z", "frame_bi_smvs"]].rename(
    columns={"frame_bi_smvs": "smvs"}
).to_csv(out_smvs, index=False)

bivul[["timestamp", "x", "y", "z", "vec_x", "vec_y", "frame_bi_smvs"]].rename(
    columns={"frame_bi_smvs": "smvs"}
).to_csv(out_vul, index=False)
print(f"[OK] wrote {out_smvs}")
print(f"[OK] wrote {out_vul}")
PY

echo "[STEP] Build LiDAR-only SMVS adapter CSVs for graph-aware CMA"
python3 - "$BISMVS_CSV" "$BISMVS_VUL_CSV" "$SMVS_GRAPH_CMA_SMVS" "$SMVS_GRAPH_CMA_VUL" <<'PY'
import os
import sys
import pandas as pd

bismvs_path, bivul_path, out_smvs, out_vul = sys.argv[1:]
bismvs = pd.read_csv(bismvs_path)
bivul = pd.read_csv(bivul_path)

required_smvs = {"timestamp", "x", "y", "z", "vul_angle_deg", "vec_x", "vec_y", "frame_l_smvs"}
missing_smvs = required_smvs - set(bismvs.columns)
if missing_smvs:
    raise SystemExit(f"Bi-SMVS CSV missing columns for LiDAR-only adapter: {sorted(missing_smvs)}")

required_l_vul = {"timestamp"} | {f"l_vul_{i:02d}" for i in range(72)}
missing_vul = required_l_vul - set(bivul.columns)
if missing_vul:
    raise SystemExit(f"Bi-SMVS vulnerability CSV missing LiDAR vulnerability columns: {sorted(missing_vul)[:8]}")

os.makedirs(os.path.dirname(out_smvs), exist_ok=True)
cols = ["timestamp", "x", "y", "z", "vul_angle_deg", "vec_x", "vec_y", "frame_l_smvs"]
if "yaw" in bismvs.columns:
    cols.insert(4, "yaw")
bismvs[cols].to_csv(out_smvs, index=False)

out = bivul.copy()
for i in range(72):
    out[f"bi_vul_{i:02d}"] = out[f"l_vul_{i:02d}"]
if "frame_l_smvs" in out.columns:
    out["frame_bi_smvs"] = out["frame_l_smvs"]
elif "frame_bi_smvs" in out.columns and "frame_l_smvs" in bismvs.columns:
    score_map = bismvs[["timestamp", "frame_l_smvs"]].drop_duplicates("timestamp")
    out = out.drop(columns=["frame_bi_smvs"], errors="ignore").merge(score_map, on="timestamp", how="left")
    out = out.rename(columns={"frame_l_smvs": "frame_bi_smvs"})
out.to_csv(out_vul, index=False)
print(f"[OK] wrote {out_smvs}")
print(f"[OK] wrote {out_vul}")
PY

echo "[STEP] Select SMVS + paper III-C position"
python3 "$SLAMSPOOF_DIR/scripts/select_spoofer_from_smvs_paper.py" \
    --smvs "$SMVS_CSV" \
    --vul "$SMVS_VUL_CSV" \
    --ref-traj "$TRAJ_CSV" \
    --top-k "$TOP_K" \
    --score-threshold -1000 \
    --score-direction larger \
    --spoof-distance "$SPOOF_DISTANCE" \
    --distance-threshold "$DISTANCE_THRESHOLD" \
    --spoofing-range "$SPOOFING_RANGE" \
    --output "$SMVS_PAPER_JSON"

echo "[STEP] Select SMVS + graph-aware CMA position"
python3 "$SLAMSPOOF_DIR/scripts/select_spoofer_bi_bo.py" \
    --smvs "$SMVS_GRAPH_CMA_SMVS" \
    --vul "$SMVS_GRAPH_CMA_VUL" \
    --traj "$TRAJ_CSV" \
    --spoofing-range "$SPOOFING_RANGE" \
    --distance-threshold "$DISTANCE_THRESHOLD" \
    --min-traj-dist "$MIN_TRAJ_DIST" \
    --max-traj-dist "$MAX_TRAJ_DIST" \
    --top-k "$TOP_K" \
    --cma-calls "$CMA_CALLS" \
    --seed "$SEED" \
    --graph-dump-dir "$GRAPH_DUMP_DIR" \
    --visualize \
    --viz-path "$SEL_DIR/smvs_graph_cma_1580.png" \
    --output "$SMVS_GRAPH_CMA_JSON"

echo "[STEP] Select Bi-SMVS + paper III-C position"
python3 "$SLAMSPOOF_DIR/scripts/select_spoofer_from_smvs_paper.py" \
    --smvs "$BISMVS_AS_SMVS" \
    --vul "$BISMVS_AS_VUL" \
    --ref-traj "$TRAJ_CSV" \
    --top-k "$TOP_K" \
    --score-threshold -1000 \
    --score-direction larger \
    --spoof-distance "$SPOOF_DISTANCE" \
    --distance-threshold "$DISTANCE_THRESHOLD" \
    --spoofing-range "$SPOOFING_RANGE" \
    --output "$BISMVS_PAPER_JSON"

echo "[STEP] Select Bi-SMVS + CMA position without graph cues"
python3 "$SLAMSPOOF_DIR/scripts/select_spoofer_bi_bo.py" \
    --smvs "$BISMVS_CSV" \
    --vul "$BISMVS_VUL_CSV" \
    --traj "$TRAJ_CSV" \
    --spoofing-range "$SPOOFING_RANGE" \
    --distance-threshold "$DISTANCE_THRESHOLD" \
    --min-traj-dist "$MIN_TRAJ_DIST" \
    --max-traj-dist "$MAX_TRAJ_DIST" \
    --top-k "$TOP_K" \
    --cma-calls "$CMA_CALLS" \
    --seed "$SEED" \
    --visualize \
    --viz-path "$SEL_DIR/bismvs_cma_no_graph_1580.png" \
    --output "$BISMVS_CMA_NO_GRAPH_JSON"

echo "[STEP] Select Bi-SMVS + graph-aware CMA position"
python3 "$SLAMSPOOF_DIR/scripts/select_spoofer_bi_bo.py" \
    --smvs "$BISMVS_CSV" \
    --vul "$BISMVS_VUL_CSV" \
    --traj "$TRAJ_CSV" \
    --spoofing-range "$SPOOFING_RANGE" \
    --distance-threshold "$DISTANCE_THRESHOLD" \
    --min-traj-dist "$MIN_TRAJ_DIST" \
    --max-traj-dist "$MAX_TRAJ_DIST" \
    --top-k "$TOP_K" \
    --cma-calls "$CMA_CALLS" \
    --seed "$SEED" \
    --graph-dump-dir "$GRAPH_DUMP_DIR" \
    --visualize \
    --viz-path "$SEL_DIR/bismvs_graph_cma_1580.png" \
    --output "$BISMVS_GRAPH_CMA_JSON"

GRAPH_CMA_TABLE_JSON="$BISMVS_GRAPH_CMA_JSON"
if [[ "$FULL_COORD_SOURCE" == "main_evaluated" ]]; then
    echo "[STEP] Use main-evaluated Bi-SMVS graph+CMA position for the ablation table"
    python3 - "$BISMVS_GRAPH_CMA_EVALUATED_JSON" "$FULL_SPOOFER_X" "$FULL_SPOOFER_Y" \
        "$DISTANCE_THRESHOLD" "$SPOOFING_RANGE" "$BISMVS_GRAPH_CMA_JSON" <<'PY'
import json
import os
import sys

out_json, sx, sy, dist, spoof_range, recomputed_json = sys.argv[1:]
data = {
    "method": "bi_smvs_graph_cma_main_evaluated_position",
    "reason": (
        "This row freezes the full-method spoofer coordinate used by the main "
        "LVI-SAM repeated experiments, so the ablation table is directly "
        "comparable with the reported main result. The current recomputed "
        "graph-aware selection is kept separately for diagnostics."
    ),
    "optim": {
        "spoofer_x": float(sx),
        "spoofer_y": float(sy),
    },
    "params": {
        "distance_threshold": float(dist),
        "spoofing_range": float(spoof_range),
    },
    "diagnostic_recomputed_selection_json": recomputed_json,
}
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"[OK] wrote {out_json}")
PY
    GRAPH_CMA_TABLE_JSON="$BISMVS_GRAPH_CMA_EVALUATED_JSON"
elif [[ "$FULL_COORD_SOURCE" != "recomputed" ]]; then
    echo "[ERROR] FULL_COORD_SOURCE must be 'main_evaluated' or 'recomputed', got: $FULL_COORD_SOURCE" >&2
    exit 1
fi

echo "[STEP] Write ablation position table"
python3 - "$POSITION_CSV" \
    "$SMVS_PAPER_JSON" "$SMVS_GRAPH_CMA_JSON" "$BISMVS_PAPER_JSON" "$BISMVS_CMA_NO_GRAPH_JSON" "$GRAPH_CMA_TABLE_JSON" \
    "$DISTANCE_THRESHOLD" "$SPOOFING_RANGE" <<'PY'
import csv
import json
import os
import sys

out_csv = sys.argv[1]
json_paths = sys.argv[2:7]
dist = sys.argv[7]
spoof_range = sys.argv[8]

specs = [
    ("smvs_paper", "smvs", "paper_iii_c", "none", False, json_paths[0]),
    ("smvs_graph_cma", "smvs", "cma", "lio_sam_proxy", True, json_paths[1]),
    ("bismvs_paper", "bismvs", "paper_iii_c", "none", False, json_paths[2]),
    ("bismvs_cma_no_graph", "bismvs", "cma", "none", True, json_paths[3]),
    ("bismvs_graph_cma", "bismvs", "cma", "lio_sam_proxy", True, json_paths[4]),
]

def read_position(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "main" in data and data["main"].get("spoofer_x") is not None:
        return float(data["main"]["spoofer_x"]), float(data["main"]["spoofer_y"])
    if "optim" in data and data["optim"].get("spoofer_x") is not None:
        return float(data["optim"]["spoofer_x"]), float(data["optim"]["spoofer_y"])
    if data.get("spoofer_x") is not None:
        return float(data["spoofer_x"]), float(data["spoofer_y"])
    raise SystemExit(f"cannot find spoofer_x/spoofer_y in {path}")

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "method", "score_input", "selector", "graph", "cma",
        "spoofer_x", "spoofer_y", "distance_threshold",
        "spoofing_range", "selection_json",
    ])
    writer.writeheader()
    for method, score_input, selector, graph, cma, path in specs:
        sx, sy = read_position(path)
        writer.writerow({
            "method": method,
            "score_input": score_input,
            "selector": selector,
            "graph": graph,
            "cma": str(cma).lower(),
            "spoofer_x": f"{sx:.12f}",
            "spoofer_y": f"{sy:.12f}",
            "distance_threshold": dist,
            "spoofing_range": spoof_range,
            "selection_json": path,
        })

print(f"[OK] wrote {out_csv}")
with open(out_csv, encoding="utf-8") as f:
    print(f.read().strip())
PY

echo "[DONE] ablation positions ready:"
echo "$POSITION_CSV"
