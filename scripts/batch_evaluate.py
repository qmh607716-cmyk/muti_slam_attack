#!/usr/bin/env python3
"""
batch_evaluate.py
==================
Converts all attack bags to CSV, then runs evaluate_attack.py for each.
"""
import os, subprocess, sys

# ── Experiment definitions ──────────────────────────────────────────────────────
# format: (dataset, mode, selector, dt_suffix, spoofer_x, spoofer_y, dt)
EXPERIMENTS = [
    # HANDHELD
    ("handheld", "removal", "bismvs", "1580",  29.80,  -117.91, 15.0),
    ("handheld", "removal", "bismvs", "3080",  29.80,  -117.91, 30.0),
    ("handheld", "removal", "smvs",   "1580", 193.91,   -12.71, 15.0),
    ("handheld", "removal", "smvs",   "3080", 193.91,   -12.71, 30.0),
    ("handheld", "static",  "bismvs", "1580",  29.80,  -117.91, 15.0),
    ("handheld", "static",  "bismvs", "3080",  29.80,  -117.91, 30.0),
    ("handheld", "static",  "smvs",   "1580", 193.91,   -12.71, 15.0),
    ("handheld", "static",  "smvs",   "3080", 193.91,   -12.71, 30.0),
    # JACKAL
    ("jackal",   "removal", "bismvs", "1580", 135.40,   218.77, 15.0),
    ("jackal",   "removal", "bismvs", "3080", 135.40,   218.77, 30.0),
    ("jackal",   "removal", "smvs",   "1580",  32.28,    10.81, 15.0),
    ("jackal",   "removal", "smvs",   "3080",  32.28,    10.81, 30.0),
    ("jackal",   "static",  "bismvs", "1580", 135.40,   218.77, 15.0),
    ("jackal",   "static",  "bismvs", "3080", 135.40,   218.77, 30.0),
    ("jackal",   "static",  "smvs",   "1580",  32.28,    10.81, 15.0),
    ("jackal",   "static",  "smvs",   "3080",  32.28,    10.81, 30.0),
    # KITTI  (success_threshold = 1.0m, KITTI odometry paper)
    ("kitti",    "removal", "bismvs", "1580",  35.90,   -20.47, 15.0),
    ("kitti",    "removal", "bismvs", "3080",  35.90,   -20.47, 30.0),
    ("kitti",    "removal", "smvs",   "1580", 201.25,  -198.46, 15.0),
    ("kitti",    "removal", "smvs",   "3080", 201.25,  -198.46, 30.0),
    ("kitti",    "static",  "bismvs", "1580",  35.90,   -20.47, 15.0),
    ("kitti",    "static",  "bismvs", "3080",  35.90,   -20.47, 30.0),
    ("kitti",    "static",  "smvs",   "1580", 201.25,  -198.46, 15.0),
    ("kitti",    "static",  "smvs",   "3080", 201.25,  -198.46, 30.0),
]

# ── Per-dataset success_threshold (APE-RMSE, m) ────────────────────────────────
SUCCESS_THRESHOLD = {
    "handheld": 4.2,   # SLAMSpoof ICRA'25 standard
    "jackal":   4.2,
    "kitti":    1.0,   # KITTI odometry paper
}

BASE = "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets"
SCRIPTS = "/home/qu_menghao/catkin_ws/src/slamspoof/scripts"
EVAL_OUT = "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/batch_eval_3080"
CSV_DIR = "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/batch_csv_3080"

os.makedirs(EVAL_OUT, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# ── Step 1: Convert all bags to CSV ───────────────────────────────────────────
# Already-converted originals
originals_csv = {
    "handheld": f"{BASE}/slamspoof_handheld/original/handheld_original_traj.csv",
    "jackal":   f"{BASE}/slamspoof_jackal/original/jackal_original_traj.csv",
    "kitti":    f"{BASE}/slamspoof_kitti/original/kitti_original_traj.csv",
}

bag_to_csv = os.path.join(SCRIPTS, "bag2csv.py")

# KITTI bag filenames differ from handheld/jackal convention
BAG_NAME_BY_DATASET = {
    "kitti": {
        "removal": "kitti_removal_traj.bag",
        "static":  "kitti_static_traj.bag",
    },
    "handheld": {
        "removal": "handheld_attack_removal_traj.bag",
        "static":  "handheld_attack_static_traj.bag",
    },
    "jackal": {
        "removal": "jackal_attack_removal_traj.bag",
        "static":  "jackal_attack_static_traj.bag",
    },
}

for dataset, mode, selector, dt_sfx, sx, sy, dt in EXPERIMENTS:
    attack_name = f"{dataset}_attack_{mode}_{selector}_{dt_sfx}"
    bag_dir = f"{BASE}/slamspoof_{dataset}/attack_{mode}/{selector}_{dt_sfx}"
    bag_name = BAG_NAME_BY_DATASET[dataset][mode]
    bag_path = os.path.join(bag_dir, bag_name)
    csv_path = os.path.join(CSV_DIR, f"{attack_name}.csv")

    if not os.path.exists(bag_path):
        print(f"[SKIP] {bag_path} not found")
        continue

    if not os.path.exists(csv_path):
        print(f"[CONVERT] {attack_name}...")
        subprocess.run(
            ["python3", bag_to_csv, bag_path, "-o", csv_path],
            check=True,
        )
    else:
        print(f"[EXISTS] {csv_path}")

print("\n[READY] All CSVs converted.\n")

# ── Step 2: Run evaluate_attack.py for each experiment ─────────────────────────
print("="*70)
print("  Running evaluate_attack.py for all experiments...")
print("="*70)

for dataset, mode, selector, dt_sfx, sx, sy, dt in EXPERIMENTS:
    attack_name = f"{dataset}_attack_{mode}_{selector}_{dt_sfx}"
    csv_path = os.path.join(CSV_DIR, f"{attack_name}.csv")
    orig_csv = originals_csv[dataset]
    out_dir = os.path.join(EVAL_OUT, attack_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[SKIP] CSV not found: {csv_path}")
        continue

    success_thr = SUCCESS_THRESHOLD[dataset]
    title = f"{dataset.upper()}: {mode} | {selector.upper()}-dt{dt_sfx} (spoofer=({sx},{sy}))"

    cmd = [
        "python3", os.path.join(SCRIPTS, "evaluate_attack.py"),
        "--orig", orig_csv,
        "--att", csv_path,
        "--out-dir", out_dir,
        "--title", title,
        "--spoofer-x", str(sx),
        "--spoofer-y", str(sy),
        "--distance-threshold", str(dt),
        "--spoofing-range", "80",
        "--success-threshold", str(success_thr),
    ]

    print(f"\n{'─'*70}")
    print(f"  [{attack_name}]")
    print(f"  --> {out_dir}")
    print(f"  --> spoofer=({sx}, {sy}), dt={dt}m, range=80°, success_thr={success_thr}m")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {attack_name} failed with code {result.returncode}")
