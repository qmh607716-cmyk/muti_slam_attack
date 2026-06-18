#!/usr/bin/env python3
"""
evaluate_gps.py
评估所有实验组（original + attack）相对于 GPS GT 的 APE / RPE。
排除 30180 组（distance_threshold=30, spoofing_range=180）。

输出格式：
  Dataset | Mode | Method | SMVSParam | vs | APE-RMSE(m) | APE-max(m) | RPE-max-1m(m) | RPE-max-10m(m)
"""

import os
import sys
import json
import math
import subprocess
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d


# ============================================================================
# TUM 格式写文件
# ============================================================================

def _to_relative_time(t: np.ndarray) -> np.ndarray:
    """把 Unix 时间戳转成相对时间（秒）。如果已经是相对时间（<1e9）则不变。"""
    if len(t) > 0 and t[0] > 1e9:
        return t - t[0]
    return t


def write_tum(csv_path: str, out_path: str, has_quat: bool = True):
    """把 LVI-SAM CSV 或 GPS CSV 写成 TUM 格式（时间统一成相对秒）。"""
    df = pd.read_csv(csv_path)
    t = _to_relative_time(df["time"].values)

    if has_quat and "qx" in df.columns:
        x = df["x"].values
        y = df["y"].values
        z = df["z"].values
        qx = df["qx"].values
        qy = df["qy"].values
        qz = df["qz"].values
        qw = df["qw"].values
        rows = [f"{t[i]:.9f} {x[i]:.9f} {y[i]:.9f} {z[i]:.9f} "
                f"{qx[i]:.9f} {qy[i]:.9f} {qz[i]:.9f} {qw[i]:.9f}"
                for i in range(len(df))]
    else:
        x = df["x"].values
        y = df["y"].values
        z = df["z"].values
        rows = [f"{t[i]:.9f} {x[i]:.9f} {y[i]:.9f} {z[i]:.9f} 0 0 0 1"
                for i in range(len(df))]

    with open(out_path, "w") as f:
        f.write("\n".join(rows))


# ============================================================================
# evo 单次运行
# ============================================================================

def run_evo_one(ref_tum: str, est_tum: str, out_dir: str, delta_m: float = None) -> dict:
    """运行 evo_ape，返回 stats dict。"""
    os.makedirs(out_dir, exist_ok=True)

    ape_zip  = os.path.join(out_dir, "ape.zip")
    rpe_zip  = os.path.join(out_dir, "rpe.zip")

    for p in [ape_zip, rpe_zip]:
        if os.path.exists(p):
            os.remove(p)

    stats = {}

    # APE
    cmd_ape = ["evo_ape", "tum", ref_tum, est_tum,
               "-a", "-r", "trans_part",
               "--save_results", ape_zip]
    r = subprocess.run(cmd_ape, capture_output=True, text=True, timeout=60)
    if r.returncode == 0:
        stats.update(_parse_evo_zip(ape_zip, prefix="ape"))

    # RPE - 1m
    cmd_rpe1 = ["evo_rpe", "tum", ref_tum, est_tum,
                "-a", "-r", "trans_part",
                "--delta", "1", "--delta_unit", "m",
                "--save_results", rpe_zip]
    r = subprocess.run(cmd_rpe1, capture_output=True, text=True, timeout=60)
    if r.returncode == 0:
        stats.update(_parse_evo_zip(rpe_zip, prefix="rpe1"))

    # RPE - 10m
    rpe10_zip = os.path.join(out_dir, "rpe10.zip")
    if os.path.exists(rpe10_zip):
        os.remove(rpe10_zip)
    cmd_rpe10 = ["evo_rpe", "tum", ref_tum, est_tum,
                 "-a", "-r", "trans_part",
                 "--delta", "10", "--delta_unit", "m",
                 "--save_results", rpe10_zip]
    r = subprocess.run(cmd_rpe10, capture_output=True, text=True, timeout=60)
    if r.returncode == 0:
        stats.update(_parse_evo_zip(rpe10_zip, prefix="rpe10"))

    return stats


def _parse_evo_zip(zip_path: str, prefix: str) -> dict:
    """从 evo zip 结果解析 stats.json。"""
    try:
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            data = json.loads(z.read("stats.json"))
        result = {}
        for key in ["rmse", "mean", "max", "std", "median", "min"]:
            result[f"{prefix}_{key}"] = data.get(key, None)
        return result
    except Exception:
        return {f"{prefix}_{k}": None for k in ["rmse", "mean", "max", "std", "median", "min"]}


# ============================================================================
# 完整评估：GPS GT vs SLAM (original/attack)
# ============================================================================

def evaluate_against_gps(gps_path: str,
                          slam_path: str,
                          out_dir: str) -> dict:
    """
    用 GPS 作参考，SLAM 轨迹作估计，计算 APE/RPE。
    GPS 是 5Hz，SLAM 也是 ~5Hz，直接用时间插值对齐。
    """
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        gps_tum  = os.path.join(tmp, "gps.tum")
        slam_tum = os.path.join(tmp, "slam.tum")

        # GPS: 无四元数
        write_tum(gps_path, gps_tum, has_quat=False)
        # SLAM: 有四元数（但 evo_ape/rpe 只用平移部分，方向不影响）
        write_tum(slam_path, slam_tum, has_quat=True)

        stats = run_evo_one(gps_tum, slam_tum, out_dir)

    return stats


# ============================================================================
# Raw error（补充参考）
# ============================================================================

def raw_error(slam_path: str, gps_path: str) -> dict:
    """
    直接计算 SLAM 相对于 GPS 的原始误差（不做 Umeyama 对齐）。
    对齐方法由用户指定，这里只做原始误差。
    """
    slam = pd.read_csv(slam_path)
    gps  = pd.read_csv(gps_path)

    t_s = _to_relative_time(slam["time"].values)
    t_g = _to_relative_time(gps["time"].values)

    # 插值 GPS 到 SLAM 时间戳
    ix = interp1d(t_g, gps["x"].values, kind="linear", bounds_error=False, fill_value=np.nan)
    iy = interp1d(t_g, gps["y"].values, kind="linear", bounds_error=False, fill_value=np.nan)
    iz = interp1d(t_g, gps["z"].values, kind="linear", bounds_error=False, fill_value=np.nan)

    xg = ix(t_s)
    yg = iy(t_s)
    zg = iz(t_s)

    valid = np.isfinite(xg)
    if not valid.all():
        n_bad = (~valid).sum()
        print(f"  [WARN] {n_bad} GPS frames have no data near SLAM timestamps", flush=True)

    xs = slam["x"].values[valid]
    ys = slam["y"].values[valid]
    zs = slam["z"].values[valid]

    dx = xs - xg[valid]
    dy = ys - yg[valid]
    dz = zs - zg[valid]

    err2d = np.sqrt(dx**2 + dy**2)
    err3d = np.sqrt(dx**2 + dy**2 + dz**2)

    return {
        "raw_rmse_2d": float(np.sqrt(np.mean(err2d**2))),
        "raw_rmse_3d": float(np.sqrt(np.mean(err3d**2))),
        "raw_mean_2d": float(err2d.mean()),
        "raw_max_2d":  float(err2d.max()),
        "raw_max_3d":  float(err3d.max()),
        "n_valid":     int(valid.sum()),
    }


# ============================================================================
# 主流程
# ============================================================================

RESULTS = []

def main():
    base = "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets"

    datasets = [
        dict(
            name="jackal",
            gps  = f"{base}/slamspoof_jackal/original/jackal_gps.csv",
            orig = f"{base}/slamspoof_jackal/original/jackal_original_traj.csv",
            attack_static  = f"{base}/slamspoof_jackal/attack_static",
            attack_removal = f"{base}/slamspoof_jackal/attack_removal",
            out  = f"{base}/slamspoof_jackal/gps_eval",
        ),
        dict(
            name="handheld",
            gps  = f"{base}/slamspoof_handheld/original/handheld_gps.csv",
            orig = f"{base}/slamspoof_handheld/original/handheld_original_traj.csv",
            attack_static  = f"{base}/slamspoof_handheld/attack_static",
            attack_removal = f"{base}/slamspoof_handheld/attack_removal",
            out  = f"{base}/slamspoof_handheld/gps_eval",
        ),
    ]

    for ds in datasets:
        # GPS vs Original
        print(f"\n{'='*60}")
        print(f"  {ds['name']} | Original vs GPS")
        print(f"{'='*60}")
        out_orig = os.path.join(ds["out"], "original")
        evo_stats_orig = evaluate_against_gps(ds["gps"], ds["orig"], out_orig)
        raw_stats_orig = raw_error(ds["orig"], ds["gps"])

        print(f"  APE RMSE:  {evo_stats_orig.get('ape_rmse', 'N/A'):.3f} m")
        print(f"  APE max:   {evo_stats_orig.get('ape_max', 'N/A'):.3f} m")
        print(f"  RPE1m max: {evo_stats_orig.get('rpe1_max', 'N/A'):.3f} m")
        print(f"  RPE10m max:{evo_stats_orig.get('rpe10_max', 'N/A'):.3f} m")
        print(f"  Raw RMSE 2D: {raw_stats_orig['raw_rmse_2d']:.3f} m")

        RESULTS.append({
            "dataset": ds["name"],
            "mode":    "original",
            "method":  "-",
            "param":   "-",
            "vs":      "GPS",
            "ape_rmse":   evo_stats_orig.get("ape_rmse"),
            "ape_max":     evo_stats_orig.get("ape_max"),
            "rpe1_max":    evo_stats_orig.get("rpe1_max"),
            "rpe10_max":   evo_stats_orig.get("rpe10_max"),
            "raw_rmse_2d": raw_stats_orig["raw_rmse_2d"],
            "raw_max_2d":  raw_stats_orig["raw_max_2d"],
        })

        # GPS vs Attack
        for attack_dir, attack_mode in [
            (ds["attack_static"],  "static"),
            (ds["attack_removal"], "removal"),
        ]:
            if not os.path.exists(attack_dir):
                continue
            for subdir in sorted(os.listdir(attack_dir)):
                sub_path = os.path.join(attack_dir, subdir)
                if not os.path.isdir(sub_path):
                    continue
                traj_files = [f for f in os.listdir(sub_path) if f.endswith("_traj.csv")]
                if not traj_files:
                    continue
                if "30180" in subdir:
                    print(f"\n  [SKIP] {ds['name']}/{attack_mode}/{subdir} (30180 excluded)")
                    continue

                if subdir.startswith("bismvs"):
                    method = "bismvs"
                elif subdir.startswith("smvs"):
                    method = "smvs"
                else:
                    continue

                param = subdir.replace("bismvs_", "").replace("smvs_", "").replace("_", "")
                if not param:
                    param = "full"

                for traj_file in traj_files:
                    traj_path = os.path.join(sub_path, traj_file)
                    print(f"\n{'─'*60}")
                    print(f"  {ds['name']} | {attack_mode:8s} | {method:6s} | {param:10s} vs GPS")
                    print(f"{'─'*60}")

                    out_att = os.path.join(ds["out"], f"{attack_mode}_{method}_{param}")
                    evo_stats = evaluate_against_gps(ds["gps"], traj_path, out_att)
                    raw_stats = raw_error(traj_path, ds["gps"])

                    ape_rmse = evo_stats.get("ape_rmse")
                    ape_max  = evo_stats.get("ape_max")
                    rpe1_max = evo_stats.get("rpe1_max")
                    rpe10_max= evo_stats.get("rpe10_max")

                    print(f"  APE RMSE:   {ape_rmse if ape_rmse else 'N/A':>8}")
                    print(f"  APE max:    {ape_max  if ape_max  else 'N/A':>8}")
                    print(f"  RPE1m max:  {rpe1_max if rpe1_max else 'N/A':>8}")
                    print(f"  RPE10m max: {rpe10_max if rpe10_max else 'N/A':>8}")
                    print(f"  Raw RMSE 2D:{raw_stats['raw_rmse_2d']:>8.3f} m")

                    RESULTS.append({
                        "dataset": ds["name"],
                        "mode":    attack_mode,
                        "method":  method,
                        "param":   param,
                        "vs":      "GPS",
                        "ape_rmse":   ape_rmse,
                        "ape_max":     ape_max,
                        "rpe1_max":    rpe1_max,
                        "rpe10_max":   rpe10_max,
                        "raw_rmse_2d": raw_stats["raw_rmse_2d"],
                        "raw_max_2d":  raw_stats["raw_max_2d"],
                    })

    # ---- 打印汇总表格 ----
    print("\n\n")
    print("=" * 100)
    print("  汇总：GPS GT vs SLAM (Original + Attacks)")
    print("=" * 100)
    print(f"{'Dataset':<10} {'Mode':<10} {'Method':<8} {'Param':<10} "
          f"{'APE-RMSE(m)':>12} {'APE-max(m)':>11} {'RPE1m-max(m)':>12} {'RPE10m-max(m)':>13} "
          f"{'RawRMSE-2D':>11}")
    print("-" * 100)

    for r in RESULTS:
        ape = f"{r['ape_rmse']:.3f}" if r['ape_rmse'] is not None else "   N/A   "
        ape_m = f"{r['ape_max']:.3f}" if r['ape_max'] is not None else "   N/A   "
        rpe1 = f"{r['rpe1_max']:.3f}" if r['rpe1_max'] is not None else "   N/A   "
        rpe10= f"{r['rpe10_max']:.3f}" if r['rpe10_max'] is not None else "    N/A   "
        raw2d = f"{r['raw_rmse_2d']:.3f}"
        flag = " ★" if "original" not in r['mode'] and r['ape_rmse'] and r['ape_rmse'] > 5.0 else ""
        print(f"{r['dataset']:<10} {r['mode']:<10} {r['method']:<8} {r['param']:<10} "
              f"{ape:>12} {ape_m:>11} {rpe1:>12} {rpe10:>13} {raw2d:>11}{flag}")

    # ---- 保存结果 ----
    out_json = "/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/gps_eval_summary.json"
    with open(out_json, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\n[OK] Saved: {out_json}")


if __name__ == "__main__":
    main()
