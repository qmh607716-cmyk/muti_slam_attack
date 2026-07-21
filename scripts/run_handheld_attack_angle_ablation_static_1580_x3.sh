#!/usr/bin/env bash
set -euo pipefail

# Sanity ablation for attack-sector coordinate handling on LVI-SAM.
# It keeps the same dataset, spoofer location, D/R, spoofing mode, and attack
# geometry, and only switches the angle used to cut the LiDAR sector:
#   lidar_local: yaw-aware direction from robot to fixed roadside spoofer
#   world      : world-frame bearing without subtracting robot yaw

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LVI_DATASET_DIR="${LVI_DATASET_DIR:-$HOME/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld}"

N_RUNS="${N_RUNS:-3}"
START_RUN="${START_RUN:-1}"
BASE_OUT_ROOT="${BASE_OUT_ROOT:-$LVI_DATASET_DIR/angle_ablation/static_bismvs_1580_x3}"

DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-15}"
SPOOFING_RANGE="${SPOOFING_RANGE:-80}"
SPOOFER_X="${SPOOFER_X:-31.28075677647965}"
SPOOFER_Y="${SPOOFER_Y:--102.07423272183334}"

echo "============================================================"
echo "[ANGLE ABLATION] handheld / Bi-SMVS / static"
echo "  D=$DISTANCE_THRESHOLD, R=$SPOOFING_RANGE, runs=$N_RUNS"
echo "  spoofer=($SPOOFER_X, $SPOOFER_Y)"
echo "  output=$BASE_OUT_ROOT"
echo "============================================================"

for angle_mode in lidar_local world; do
    echo
    echo "============================================================"
    echo "[ANGLE ABLATION] attack_angle_mode=$angle_mode"
    echo "============================================================"

    OUT_ROOT="$BASE_OUT_ROOT/$angle_mode" \
    ATTACK_ANGLE_MODE="$angle_mode" \
    N_RUNS="$N_RUNS" \
    START_RUN="$START_RUN" \
    RESET_SUMMARY=1 \
    DISTANCE_THRESHOLD="$DISTANCE_THRESHOLD" \
    SPOOFING_RANGE="$SPOOFING_RANGE" \
    SPOOFER_X="$SPOOFER_X" \
    SPOOFER_Y="$SPOOFER_Y" \
        bash "$SCRIPT_DIR/run_handheld_bismvs_static_1580_x3.sh"
done

python3 - "$BASE_OUT_ROOT" <<'PY'
import csv
import os
import statistics
import sys

base = sys.argv[1]
print()
print("============================================================")
print("[ANGLE ABLATION SUMMARY]")
print("============================================================")
for mode in ("lidar_local", "world"):
    summary = os.path.join(base, mode, "summary.csv")
    rows = []
    if os.path.exists(summary):
        with open(summary, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") != "ok":
                    continue
                try:
                    rows.append((
                        float(row.get("ape_rmse", "")),
                        float(row.get("rpe_10m_rmse", "")),
                    ))
                except Exception:
                    pass

    if not rows:
        print(f"{mode:11s}: no valid runs")
        continue

    ape = [r[0] for r in rows]
    rpe10 = [r[1] for r in rows]
    ape_std = statistics.stdev(ape) if len(ape) > 1 else 0.0
    rpe_std = statistics.stdev(rpe10) if len(rpe10) > 1 else 0.0
    print(
        f"{mode:11s}: n={len(rows)}  "
        f"APE={statistics.mean(ape):.3f} +/- {ape_std:.3f} m  "
        f"RPE-10m={statistics.mean(rpe10):.3f} +/- {rpe_std:.3f} m"
    )
PY

echo
echo "[OK] Angle ablation complete."
echo "     $BASE_OUT_ROOT"
