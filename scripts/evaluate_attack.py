#!/usr/bin/env python3
"""
evaluate_attack.py
==================
Industry-standard SLAM attack evaluation using evo + Umeyama alignment.

Key design decisions:
  1. Raw error (no alignment): measures total deviation = SLAM drift + attack effect
     - Original and attack trajectories share the same coordinate frame
     - Both start at (0,0,0) at the same timestamp
     - Any global offset is SLAM drift, not attack

  2. evo ATE (Umeyama aligned): measures attack effect only
     - Aligns attack traj to original via SE(3)
     - Subtracts global drift, leaving only attack-induced error

  3. evo RPE (1m/10m): measures local drift characteristics
     - 1m: fine-grained local drift
     - 10m: longer-range drift patterns

Input CSV format (LVI-SAM native):
    time,x,y,z,qx,qy,qz,qw

Usage:
    python3 evaluate_attack.py \
        --orig .../jackal_original_traj.csv \
        --att  .../jackal_attack_static_traj.csv \
        --out-dir .../eval \
        --title "jackal: static attack" \
        --spoofer-x 141 --spoofer-y 223 --distance-threshold 30
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Format conversion: CSV → TUM .txt (evo format)
# ============================================================================

def csv_to_tum(csv_path: str, out_path: str):
    """Write LVI-SAM CSV to TUM format for evo."""
    df = pd.read_csv(csv_path)
    required = {"time", "x", "y", "z", "qx", "qy", "qz", "qw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            f.write(
                f"{row['time']:.9f} "
                f"{row['x']:.9f} {row['y']:.9f} {row['z']:.9f} "
                f"{row['qx']:.9f} {row['qy']:.9f} {row['qz']:.9f} {row['qw']:.9f}\n"
            )


# ============================================================================
# SE(3) alignment: Umeyama algorithm (for reference / evo uses its own)
# ============================================================================

def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    """
    Compute optimal rigid transform (scale, R, t) from src → dst.
    Umeyama 1994, IEEE PAMI.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    C = (src_centered.T @ dst_centered) / n
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = np.trace(np.diag(S) @ np.diag([1, 1, np.linalg.det(R)])) / np.sum(src_centered ** 2)
    t = dst_mean - scale * (R @ src_mean)
    return float(scale), R, t


def apply_alignment(pts: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (R @ pts.T).T + t[np.newaxis, :]).astype(np.float64)


# ============================================================================
# Evo wrapper
# ============================================================================

def run_evo(orig_tum: str, att_tum: str, out_dir: str, title: str) -> dict:
    """
    Run evo_ape (ATE) and evo_rpe with pose relations aligned to SLAMSpoof paper.

    Paper metrics (Section IV-A):
      - RMSE of Absolute Pose Error (APE)
      - Maximum value of Relative Pose Error (RPE)

    evo provides per-relation breakdowns:
      - translation_part (meters): used for APE in m, RPE in m
      - angle_rad / angle_deg (radians/degrees): used for APE rotation, RPE rotation
    """
    import zipfile
    results = {
        "ape_translation": {},     # paper-style: APE in meters
        "ape_rotation_rad": {},    # paper-style: rotation APE in rad
        "ape_rotation_deg": {},    # paper-style: rotation APE in degrees
        "rpe_1m_translation": {},
        "rpe_1m_rotation_deg": {},
        "rpe_10m_translation": {},
        "rpe_10m_rotation_deg": {},
    }

    def _parse_evo_output(filepath: str) -> dict:
        """Parse evo zip results file (stats.json inside)."""
        try:
            with zipfile.ZipFile(filepath) as z:
                return json.loads(z.read("stats.json"))
        except Exception:
            return {}

    def _run_one(cmd, label, out_path, plot_path):
        # Remove all existing output files to avoid evo interactive overwrite prompt
        # evo with -a creates: <name>.txt, <name>_map.png, <name>_raw.png
        base = plot_path.rsplit(".", 1)[0] if plot_path else (out_path.rsplit(".", 1)[0] if out_path else None)
        for p in [out_path, plot_path]:
            if p and os.path.exists(p):
                os.remove(p)
        if base:
            for suffix in ["_map.png", "_raw.png"]:
                q = base + suffix
                if os.path.exists(q):
                    os.remove(q)
        print(f"[INFO] Running {label}...", flush=True)
        r = subprocess.run(cmd, check=False)
        if r.returncode == 0:
            stats = _parse_evo_output(out_path)
            rmse = stats.get("rmse", "N/A")
            print(f"[OK] {label} done: RMSE={rmse}", flush=True)
            return stats
        else:
            print(f"[WARN] {label} failed", file=sys.stderr)
            return {}

    # ---- APE — translation (paper: "APE" in meters) ----
    ape_txt = os.path.join(out_dir, "evo_ape.txt")
    ape_plot = os.path.join(out_dir, "evo_ape.png")
    results["ape_translation"] = _run_one(
        ["evo_ape", "tum", orig_tum, att_tum, "-a",
         "-r", "trans_part",
         "--save_plot", ape_plot,
         "--save_results", ape_txt],
        "evo_ape (translation)", ape_txt, ape_plot,
    )

    # ---- APE — rotation in radians ----
    ape_rot_txt = os.path.join(out_dir, "evo_ape_rot_rad.txt")
    ape_rot_plot = os.path.join(out_dir, "evo_ape_rot_rad.png")
    results["ape_rotation_rad"] = _run_one(
        ["evo_ape", "tum", orig_tum, att_tum, "-a",
         "-r", "angle_rad",
         "--save_plot", ape_rot_plot,
         "--save_results", ape_rot_txt],
        "evo_ape (rotation, rad)", ape_rot_txt, ape_rot_plot,
    )

    # ---- APE — rotation in degrees ----
    ape_degtxt = os.path.join(out_dir, "evo_ape_rot_deg.txt")
    ape_degplot = os.path.join(out_dir, "evo_ape_rot_deg.png")
    results["ape_rotation_deg"] = _run_one(
        ["evo_ape", "tum", orig_tum, att_tum, "-a",
         "-r", "angle_deg",
         "--save_plot", ape_degplot,
         "--save_results", ape_degtxt],
        "evo_ape (rotation, deg)", ape_degtxt, ape_degplot,
    )

    # ---- RPE — translation at 1m delta ----
    rpe_txt = os.path.join(out_dir, "evo_rpe.txt")
    rpe_plot = os.path.join(out_dir, "evo_rpe.png")
    results["rpe_1m_translation"] = _run_one(
        ["evo_rpe", "tum", orig_tum, att_tum, "-a",
         "-r", "trans_part",
         "--delta", "1", "--delta_unit", "m",
         "--save_plot", rpe_plot,
         "--save_results", rpe_txt],
        "evo_rpe (translation, 1m)", rpe_txt, rpe_plot,
    )

    # ---- RPE — rotation (deg) at 1m delta ----
    rpe_rot_txt = os.path.join(out_dir, "evo_rpe_rot.txt")
    rpe_rot_plot = os.path.join(out_dir, "evo_rpe_rot.png")
    results["rpe_1m_rotation_deg"] = _run_one(
        ["evo_rpe", "tum", orig_tum, att_tum, "-a",
         "-r", "angle_deg",
         "--delta", "1", "--delta_unit", "m",
         "--save_plot", rpe_rot_plot,
         "--save_results", rpe_rot_txt],
        "evo_rpe (rotation, 1m)", rpe_rot_txt, rpe_rot_plot,
    )

    # ---- RPE — translation at 10m delta ----
    rpe10_txt = os.path.join(out_dir, "evo_rpe_10m.txt")
    rpe10_plot = os.path.join(out_dir, "evo_rpe_10m.png")
    results["rpe_10m_translation"] = _run_one(
        ["evo_rpe", "tum", orig_tum, att_tum, "-a",
         "-r", "trans_part",
         "--delta", "10", "--delta_unit", "m",
         "--save_plot", rpe10_plot,
         "--save_results", rpe10_txt],
        "evo_rpe (translation, 10m)", rpe10_txt, rpe10_plot,
    )

    # ---- RPE — rotation (deg) at 10m delta ----
    rpe10_rot_txt = os.path.join(out_dir, "evo_rpe_10m_rot.txt")
    rpe10_rot_plot = os.path.join(out_dir, "evo_rpe_10m_rot.png")
    results["rpe_10m_rotation_deg"] = _run_one(
        ["evo_rpe", "tum", orig_tum, att_tum, "-a",
         "-r", "angle_deg",
         "--delta", "10", "--delta_unit", "m",
         "--save_plot", rpe10_rot_plot,
         "--save_results", rpe10_rot_txt],
        "evo_rpe (rotation, 10m)", rpe10_rot_txt, rpe10_rot_plot,
    )

    return results


# ============================================================================
# Custom attack metrics (raw error, since trajectories share frame)
# ============================================================================

def compute_attack_metrics(orig_df: pd.DataFrame, att_df: pd.DataFrame,
                            spoofer_xy=None, distance_threshold=None):
    """
    Compute attack metrics.

    IMPORTANT: For attack evaluation, we use RAW error (no alignment) because:
      - Both trajectories start at (0,0,0) at the same time
      - They share the same coordinate frame (LVI-SAM global frame)
      - Any global drift is SLAM noise, not attack effect
      - The attack effect is the LOCAL deviation between the two trajectories

    However, we also compute Umeyama-aligned error for reference, which gives
    the "attack effect after removing global drift".
    """
    # Interpolate attack to original timestamps
    t_orig = orig_df["time"].values
    t_att = att_df["time"].values

    from scipy.interpolate import interp1d
    interp_x = interp1d(t_att, att_df["x"].values, kind="linear", bounds_error=False, fill_value=np.nan)
    interp_y = interp1d(t_att, att_df["y"].values, kind="linear", bounds_error=False, fill_value=np.nan)
    interp_z = interp1d(t_att, att_df["z"].values, kind="linear", bounds_error=False, fill_value=np.nan)

    xa = interp_x(t_orig)
    ya = interp_y(t_orig)
    za = interp_z(t_orig)

    # Remove frames where attack data is unavailable (NaN)
    valid_mask = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(za)
    if not valid_mask.all():
        n_removed = (~valid_mask).sum()
        print(f"[WARN] {n_removed} frames have no attack data (time out of range) — skipped in metrics", flush=True)
        t_orig = t_orig[valid_mask]
        xa = xa[valid_mask]
        ya = ya[valid_mask]
        za = za[valid_mask]
        orig_pts = orig_df[["x", "y", "z"]].values[valid_mask]
    else:
        orig_pts = orig_df[["x", "y", "z"]].values

    att_pts = np.stack([xa, ya, za], axis=1)

    t_grid = t_orig - t_orig[0]  # relative time
    t_end = float(t_grid[-1])

    # --- Raw error (primary metric for attack evaluation) ---
    raw_err_3d = np.linalg.norm(att_pts - orig_pts, axis=1)
    raw_err_2d = np.linalg.norm((att_pts - orig_pts)[:, :2], axis=1)

    raw_metrics = {
        "rmse_3d_m": float(np.sqrt(np.mean(raw_err_3d ** 2))),
        "mean_3d_m": float(raw_err_3d.mean()),
        "max_3d_m": float(raw_err_3d.max()),
        "max_3d_idx": int(np.argmax(raw_err_3d)),
        "rmse_2d_m": float(np.sqrt(np.mean(raw_err_2d ** 2))),
        "mean_2d_m": float(raw_err_2d.mean()),
        "max_2d_m": float(raw_err_2d.max()),
    }

    # Deviation duration analysis
    deviation_duration = {}
    for thresh in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        mask = raw_err_2d > thresh
        deviation_duration[f"threshold_{thresh:.1f}m"] = {
            "duration_s": float(mask.sum() / len(mask) * t_end),
            "ratio": float(mask.mean()),
        }
    raw_metrics["deviation_duration"] = deviation_duration

    # Per-axis raw
    raw_per_axis = {}
    for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
        e = att_pts[:, idx] - orig_pts[:, idx]
        raw_per_axis[axis] = {
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mean_abs": float(np.mean(np.abs(e))),
            "max": float(np.max(np.abs(e))),
        }
    raw_metrics["per_axis"] = raw_per_axis

    # --- Umeyama-aligned error (reference only) ---
    scale, R, t = umeyama_alignment(att_pts, orig_pts)
    aligned = apply_alignment(att_pts, scale, R, t)

    aligned_err_3d = np.linalg.norm(aligned - orig_pts, axis=1)
    aligned_err_2d = np.linalg.norm((aligned - orig_pts)[:, :2], axis=1)

    aligned_metrics = {
        "rmse_3d_m": float(np.sqrt(np.mean(aligned_err_3d ** 2))),
        "mean_3d_m": float(aligned_err_3d.mean()),
        "max_3d_m": float(aligned_err_3d.max()),
        "rmse_2d_m": float(np.sqrt(np.mean(aligned_err_2d ** 2))),
        "mean_2d_m": float(aligned_err_2d.mean()),
        "max_2d_m": float(aligned_err_2d.max()),
        "alignment_scale": scale,
        "note": "Aligned for reference; raw error is the primary attack metric",
    }

    # --- Attack-specific metrics ---
    attack_metrics = {}
    if spoofer_xy is not None and distance_threshold is not None:
        sx, sy = spoofer_xy
        dist_to_spoofer = np.sqrt((orig_pts[:, 0] - sx) ** 2 + (orig_pts[:, 1] - sy) ** 2)
        in_zone = dist_to_spoofer <= distance_threshold

        if in_zone.any():
            zone_err = raw_err_2d[in_zone]
            attack_metrics["zone_duration_s"] = float(in_zone.sum() / len(in_zone) * t_end)
            attack_metrics["zone_ratio"] = float(in_zone.mean())
            attack_metrics["zone_rmse_2d_m"] = float(np.sqrt(np.mean(zone_err ** 2)))
            attack_metrics["zone_mean_2d_m"] = float(zone_err.mean())

            # Onset times
            for thresh in [0.5, 1.0, 2.0]:
                mask = raw_err_2d > thresh
                attack_metrics[f"onset_{thresh:.1f}m_s"] = float(t_grid[np.argmax(mask)]) if mask.any() else None

            # Drift velocity inside zone
            zone_t = t_grid[in_zone]
            if len(zone_t) > 2:
                A = np.vstack([zone_t, np.ones(len(zone_t))]).T
                slope, _ = np.linalg.lstsq(A, zone_err, rcond=None)[0]
                attack_metrics["drift_velocity_zone_mps"] = float(slope)

    return {
        "raw": raw_metrics,
        "aligned_ref": aligned_metrics,
        "attack_metrics": attack_metrics,
        "duration_s": t_end,
        "n_samples": len(t_grid),
    }


# ============================================================================
# Plotting
# ============================================================================

def _umeyama_alignment(src, dst):
    """Umeyama (1991) similarity alignment: dst ~ s * R @ src + t.

    Same formulation evo uses internally for APE / trajectory alignment
    (with_scale=True). Returns:
        aligned_src: (N, 3) - transformed source points
        R: (3, 3) rotation
        t: (3,) translation
        s: scalar scale
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = (src_c ** 2).sum() / src.shape[0]
    H = src_c.T @ dst_c / src.shape[0]
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = (Vt.T @ S @ U.T).T
    s = (D * np.diag(S)).sum() / var_src if var_src > 0 else 1.0
    t = mu_dst - s * R @ mu_src
    aligned = (s * (R @ src.T)).T + t
    return aligned, R, t, s


def plot_results(orig_df, att_df, metrics, out_dir, title,
                 spoofer_xy=None, distance_threshold=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_orig = orig_df["time"].values
    t_att = att_df["time"].values
    from scipy.interpolate import interp1d
    interp_x = interp1d(t_att, att_df["x"].values, kind="linear", fill_value="extrapolate")
    interp_y = interp1d(t_att, att_df["y"].values, kind="linear", fill_value="extrapolate")
    interp_z = interp1d(t_att, att_df["z"].values, kind="linear", fill_value="extrapolate")
    xa = interp_x(t_orig)
    ya = interp_y(t_orig)
    za = interp_z(t_orig)

    orig_pts = orig_df[["x", "y", "z"]].values
    att_pts = np.stack([xa, ya, za], axis=1)

    err2d = np.linalg.norm((att_pts - orig_pts)[:, :2], axis=1)
    err3d = np.linalg.norm(att_pts - orig_pts, axis=1)
    t_grid = t_orig - t_orig[0]

    # ---- 1) XY compare (raw, shared frame) ----
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(orig_pts[:, 0], orig_pts[:, 1], label="Original", linewidth=1.5, alpha=0.9)
    ax.plot(att_pts[:, 0], att_pts[:, 1], label="Attack", linewidth=1.5, alpha=0.9, linestyle="--")

    ax.scatter(orig_pts[0, 0], orig_pts[0, 1], marker="o", s=80,
               c="green", edgecolors="black", zorder=7, label="Start (orig)")
    ax.scatter(att_pts[0, 0], att_pts[0, 1], marker="o", s=80,
               c="blue", edgecolors="black", zorder=7, label="Start (attack)")
    ax.scatter(orig_pts[-1, 0], orig_pts[-1, 1], marker="s", s=80,
               c="green", edgecolors="black", zorder=7, label="End (orig)")
    ax.scatter(att_pts[-1, 0], att_pts[-1, 1], marker="s", s=80,
               c="blue", edgecolors="black", zorder=7, label="End (attack)")

    idx = int(np.argmax(err2d))
    ax.scatter(att_pts[idx, 0], att_pts[idx, 1], marker="*", s=300,
               c="red", edgecolors="black", zorder=8,
               label=f"Max 2D dev ({err2d[idx]:.1f}m)")

    if spoofer_xy is not None and distance_threshold is not None:
        circle = plt.Circle(spoofer_xy, distance_threshold,
                            fill=False, color="red", linestyle="--", linewidth=1.5,
                            zorder=5, label=f"Trigger zone (r={distance_threshold}m)")
        ax.add_patch(circle)
        ax.scatter(spoofer_xy[0], spoofer_xy[1], marker="X", s=200,
                   c="red", edgecolors="black", linewidths=1, zorder=6, label="Spoofer")
        ax.annotate(
            f"Spoofer ({spoofer_xy[0]:.0f}, {spoofer_xy[1]:.0f})",
            spoofer_xy, xytext=(10, 10), textcoords="offset points",
            fontsize=9, color="red",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="red", alpha=0.8), zorder=7,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{title}\n(Raw error, shared coordinate frame)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "xy_compare_raw.png"), dpi=200)
    plt.close(fig)

    # ---- 1b) Umeyama (SE(3) with scale) alignment, then XY compare ----
    # This is the alignment evo uses internally; applying it here lets us
    # visualise the "local" attack-induced deviation (after removing the
    # global rotation/translation/scale between the two trajectories).
    att_aligned, R_uma, t_uma, s_uma = _umeyama_alignment(orig_pts, att_pts)
    err2d_a = np.linalg.norm((att_aligned - orig_pts)[:, :2], axis=1)
    err3d_a = np.linalg.norm(att_aligned - orig_pts, axis=1)
    idx_a = int(np.argmax(err2d_a))

    fig_a, ax_a = plt.subplots(figsize=(12, 10))
    ax_a.plot(orig_pts[:, 0], orig_pts[:, 1], label="Original", linewidth=1.5, alpha=0.9)
    ax_a.plot(att_aligned[:, 0], att_aligned[:, 1], label="Attack (Umeyama aligned)",
              linewidth=1.5, alpha=0.9, linestyle="--")

    ax_a.scatter(orig_pts[0, 0], orig_pts[0, 1], marker="o", s=80,
                 c="green", edgecolors="black", zorder=7, label="Start (orig)")
    ax_a.scatter(att_aligned[0, 0], att_aligned[0, 1], marker="o", s=80,
                 c="blue", edgecolors="black", zorder=7, label="Start (aligned)")
    ax_a.scatter(orig_pts[-1, 0], orig_pts[-1, 1], marker="s", s=80,
                 c="green", edgecolors="black", zorder=7, label="End (orig)")
    ax_a.scatter(att_aligned[-1, 0], att_aligned[-1, 1], marker="s", s=80,
                 c="blue", edgecolors="black", zorder=7, label="End (aligned)")

    ax_a.scatter(att_aligned[idx_a, 0], att_aligned[idx_a, 1], marker="*", s=300,
                 c="red", edgecolors="black", zorder=8,
                 label=f"Max 2D dev ({err2d_a[idx_a]:.1f}m)")

    if spoofer_xy is not None and distance_threshold is not None:
        # Use the same world-frame spoofer coords; they're not in the
        # "aligned" frame, so draw the trigger zone on the original traj
        # as a reference, in original coordinates.
        circle = plt.Circle(spoofer_xy, distance_threshold,
                            fill=False, color="red", linestyle="--", linewidth=1.5,
                            zorder=5, label=f"Trigger zone (r={distance_threshold}m)")
        ax_a.add_patch(circle)
        ax_a.scatter(spoofer_xy[0], spoofer_xy[1], marker="X", s=200,
                     c="red", edgecolors="black", linewidths=1, zorder=6, label="Spoofer")

    ax_a.set_xlabel("x [m]")
    ax_a.set_ylabel("y [m]")
    ax_a.axis("equal")
    ax_a.legend(loc="best", fontsize=8)
    ax_a.set_title(
        f"{title}\n"
        f"(Umeyama SE(3)+scale aligned, s={s_uma:.3f}, |t|={np.linalg.norm(t_uma):.2f}m)"
    )
    ax_a.grid(True, alpha=0.3)
    fig_a.tight_layout()
    fig_a.savefig(os.path.join(out_dir, "xy_compare_aligned.png"), dpi=200)
    plt.close(fig_a)

    # ---- 2) Deviation over time (raw 2D) ----
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(t_grid, err2d, color="purple", linewidth=1.0, alpha=0.8, label="2D deviation (raw)")
    ax2.plot(t_grid, err3d, color="gray", linewidth=0.8, alpha=0.5, label="3D deviation")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    if spoofer_xy is not None and distance_threshold is not None:
        dist_to_spoofer = np.sqrt((orig_pts[:, 0] - spoofer_xy[0]) ** 2 + (orig_pts[:, 1] - spoofer_xy[1]) ** 2)
        trigger_mask = dist_to_spoofer <= distance_threshold
        if trigger_mask.any():
            t_start = t_grid[trigger_mask].min()
            t_end_ts = t_grid[trigger_mask].max()
            ax2.axvspan(t_start, t_end_ts, alpha=0.2, color="red", label="Attack zone")
            ax2.axvline(t_start, color="red", linestyle=":", linewidth=1)
            ax2.axvline(t_end_ts, color="red", linestyle=":", linewidth=1)

    ax2.scatter(t_grid[idx], err2d[idx], marker="*", s=200,
                c="red", zorder=8, label=f"Max ({err2d[idx]:.1f}m @ {t_grid[idx]:.1f}s)")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("deviation [m]")
    ax2.set_title(f"{title} — Deviation over time (raw)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "deviation_raw.png"), dpi=200)
    plt.close(fig2)

    # ---- 2b) Deviation over time (Umeyama aligned) ----
    fig2a, ax2a = plt.subplots(figsize=(12, 5))
    ax2a.plot(t_grid, err2d_a, color="purple", linewidth=1.0, alpha=0.8,
              label="2D deviation (aligned)")
    ax2a.plot(t_grid, err3d_a, color="gray", linewidth=0.8, alpha=0.5,
              label="3D deviation (aligned)")
    ax2a.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    if spoofer_xy is not None and distance_threshold is not None:
        dist_to_spoofer = np.sqrt((orig_pts[:, 0] - spoofer_xy[0]) ** 2 + (orig_pts[:, 1] - spoofer_xy[1]) ** 2)
        trigger_mask = dist_to_spoofer <= distance_threshold
        if trigger_mask.any():
            t_start = t_grid[trigger_mask].min()
            t_end_ts = t_grid[trigger_mask].max()
            ax2a.axvspan(t_start, t_end_ts, alpha=0.2, color="red", label="Attack zone")
            ax2a.axvline(t_start, color="red", linestyle=":", linewidth=1)
            ax2a.axvline(t_end_ts, color="red", linestyle=":", linewidth=1)

    ax2a.scatter(t_grid[idx_a], err2d_a[idx_a], marker="*", s=200,
                 c="red", zorder=8, label=f"Max ({err2d_a[idx_a]:.1f}m @ {t_grid[idx_a]:.1f}s)")
    ax2a.set_xlabel("time [s]")
    ax2a.set_ylabel("deviation [m]")
    ax2a.set_title(f"{title} — Deviation over time (Umeyama aligned)")
    ax2a.legend(loc="best", fontsize=8)
    ax2a.grid(True, alpha=0.3)
    fig2a.tight_layout()
    fig2a.savefig(os.path.join(out_dir, "deviation_aligned.png"), dpi=200)
    plt.close(fig2a)

    # ---- 3) Deviation duration vs threshold ----
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    thresholds = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    durations = []
    ratios = []
    dd = metrics["raw"]["deviation_duration"]
    for th in thresholds:
        info = dd[f"threshold_{th:.1f}m"]
        durations.append(info["duration_s"])
        ratios.append(info["ratio"])
    bars = ax3.bar([str(t) for t in thresholds], durations, color="steelblue", edgecolor="black")
    ax3.set_xlabel("Deviation threshold [m]")
    ax3.set_ylabel("Duration [s]")
    ax3.set_title("Deviation duration vs threshold (2D, raw)")
    for bar, dur, ratio in zip(bars, durations, ratios):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(durations) * 0.01,
                 f"{dur:.1f}s\n({ratio:.1%})", ha="center", va="bottom", fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "deviation_duration.png"), dpi=200)
    plt.close(fig3)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Industry-standard SLAM attack evaluation (evo + Umeyama alignment)"
    )
    parser.add_argument("--orig", required=True, help="Original trajectory CSV")
    parser.add_argument("--att", required=True, help="Attack trajectory CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--title", default="LVI-SAM Attack Evaluation")
    parser.add_argument("--spoofer-x", type=float, default=None)
    parser.add_argument("--spoofer-y", type=float, default=None)
    parser.add_argument("--distance-threshold", type=float, default=None,
                        help="Attack trigger radius (m)")
    parser.add_argument("--wall-dist", type=float, default=None)
    parser.add_argument("--spoofing-range", type=float, default=None)
    parser.add_argument("--spoofer-heading", type=float, default=None)
    parser.add_argument("--skip-evo", action="store_true", help="Skip evo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[START] {args.title}", flush=True)
    print(f"[INFO] orig: {args.orig}", flush=True)
    print(f"[INFO] att:  {args.att}", flush=True)
    print(f"[INFO] out:  {args.out_dir}", flush=True)

    # Load CSVs
    print("[INFO] Loading CSVs...", flush=True)
    orig_df = pd.read_csv(args.orig)
    att_df = pd.read_csv(args.att)
    print(f"[OK] Loaded {len(orig_df)} original poses, {len(att_df)} attack poses", flush=True)

    # Create TUM files (matched timestamps) for evo
    with tempfile.TemporaryDirectory() as tmp:
        orig_tum = os.path.join(tmp, "orig.txt")
        att_tum = os.path.join(tmp, "att.txt")

        # Interpolate attack pose components (position + orientation quaternion) onto orig timestamps
        from scipy.interpolate import interp1d
        t_orig = orig_df["time"].values
        t_att = att_df["time"].values
        interp_x = interp1d(t_att, att_df["x"].values, kind="linear", bounds_error=False, fill_value=np.nan)
        interp_y = interp1d(t_att, att_df["y"].values, kind="linear", bounds_error=False, fill_value=np.nan)
        interp_z = interp1d(t_att, att_df["z"].values, kind="linear", bounds_error=False, fill_value=np.nan)
        interp_qx = interp1d(t_att, att_df["qx"].values, kind="linear", bounds_error=False, fill_value=np.nan)
        interp_qy = interp1d(t_att, att_df["qy"].values, kind="linear", bounds_error=False, fill_value=np.nan)
        interp_qz = interp1d(t_att, att_df["qz"].values, kind="linear", bounds_error=False, fill_value=np.nan)
        interp_qw = interp1d(t_att, att_df["qw"].values, kind="linear", bounds_error=False, fill_value=np.nan)
        xa = interp_x(t_orig)
        ya = interp_y(t_orig)
        za = interp_z(t_orig)
        qxa = interp_qx(t_orig)
        qya = interp_qy(t_orig)
        qza = interp_qz(t_orig)
        qwa = interp_qw(t_orig)
        # Remove frames where attack data is missing (NaN after interpolation)
        valid_mask = (np.isfinite(xa) & np.isfinite(ya) & np.isfinite(za)
                      & np.isfinite(qxa) & np.isfinite(qya) & np.isfinite(qza) & np.isfinite(qwa))
        if not valid_mask.all():
            n_removed = (~valid_mask).sum()
            print(f"[WARN] Removing {n_removed} frames with missing attack data (out of range)", flush=True)
            xa = xa[valid_mask]
            ya = ya[valid_mask]
            za = za[valid_mask]
            qxa = qxa[valid_mask]
            qya = qya[valid_mask]
            qza = qza[valid_mask]
            qwa = qwa[valid_mask]
            orig_pts_trimmed = orig_df[["x", "y", "z"]].values[valid_mask]
        else:
            orig_pts_trimmed = orig_df[["x", "y", "z"]].values

        with open(orig_tum, "w") as f:
            valid_idx = 0
            for i in range(len(orig_df)):
                if not valid_mask[i]:
                    continue
                r = orig_df.iloc[i]
                f.write(f"{r['time']:.9f} {r['x']:.9f} {r['y']:.9f} {r['z']:.9f} "
                        f"{r['qx']:.9f} {r['qy']:.9f} {r['qz']:.9f} {r['qw']:.9f}\n")
                valid_idx += 1

        with open(att_tum, "w") as f:
            valid_idx = 0
            for i in range(len(orig_df)):
                if not valid_mask[i]:
                    continue
                r = orig_df.iloc[i]
                f.write(f"{r['time']:.9f} {xa[valid_idx]:.9f} {ya[valid_idx]:.9f} {za[valid_idx]:.9f} "
                        f"{qxa[valid_idx]:.9f} {qya[valid_idx]:.9f} {qza[valid_idx]:.9f} {qwa[valid_idx]:.9f}\n")
                valid_idx += 1

        evo_results = {}
        if not args.skip_evo:
            evo_results = run_evo(orig_tum, att_tum, str(out_dir), args.title)

    # Custom metrics
    metrics = compute_attack_metrics(
        orig_df, att_df,
        spoofer_xy=(args.spoofer_x, args.spoofer_y)
        if (args.spoofer_x is not None and args.spoofer_y is not None) else None,
        distance_threshold=args.distance_threshold,
    )

    # Plots
    plot_results(orig_df, att_df, metrics, str(out_dir), args.title,
                 spoofer_xy=(args.spoofer_x, args.spoofer_y),
                 distance_threshold=args.distance_threshold)

    # ---- Load evo stats from zip files (paper-aligned metrics) ----
    def load_evo_zip(path):
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                return json.loads(z.read("stats.json"))
        except Exception:
            return {}

    # Paper-style metrics (Section IV-A of SLAMSpoof, ICRA 2025)
    ape_translation_stats = load_evo_zip(os.path.join(out_dir, "evo_ape.txt"))
    ape_rotation_rad_stats = load_evo_zip(os.path.join(out_dir, "evo_ape_rot_rad.txt"))
    ape_rotation_deg_stats = load_evo_zip(os.path.join(out_dir, "evo_ape_rot_deg.txt"))
    rpe1_translation_stats = load_evo_zip(os.path.join(out_dir, "evo_rpe.txt"))
    rpe1_rotation_stats = load_evo_zip(os.path.join(out_dir, "evo_rpe_rot.txt"))
    rpe10_translation_stats = load_evo_zip(os.path.join(out_dir, "evo_rpe_10m.txt"))
    rpe10_rotation_stats = load_evo_zip(os.path.join(out_dir, "evo_rpe_10m_rot.txt"))

    # Paper uses "RMSE of APE" and "max value of RPE" as the two headline numbers.
    paper_ape_translation_rmse = ape_translation_stats.get("rmse")
    paper_rpe_max_m = max(
        rpe1_translation_stats.get("max", 0.0) or 0.0,
        rpe10_translation_stats.get("max", 0.0) or 0.0,
    )
    paper_rpe_max_deg = max(
        rpe1_rotation_stats.get("max", 0.0) or 0.0,
        rpe10_rotation_stats.get("max", 0.0) or 0.0,
    )

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print(f"  {args.title}")
    print("=" * 70)

    print("\n[Paper-aligned: SLAMSpoof ICRA'25]  APE-RMSE + max(RPE)")
    if paper_ape_translation_rmse is not None:
        print(f"  APE  translation RMSE:  {paper_ape_translation_rmse:.4f} m   <-- headline")
    if ape_rotation_deg_stats:
        print(f"  APE  rotation    RMSE:  {ape_rotation_deg_stats.get('rmse', 0):.4f} deg")
    print(f"  RPE  translation  max:  {paper_rpe_max_m:.4f} m   <-- headline")
    print(f"  RPE  rotation     max:  {paper_rpe_max_deg:.4f} deg")
    if paper_ape_translation_rmse is not None:
        verdict = "ATTACK SUCCESS (>=4.2m)" if paper_ape_translation_rmse >= 4.2 else "no effect (<4.2m)"
        print(f"  --> APE-RMSE verdict: {verdict}  (paper threshold = 4.2m)")

    print("\n-- evo APE translation (Umeyama SE(3) aligned) --")
    if ape_translation_stats:
        for k in ["rmse", "mean", "median", "std", "min", "max"]:
            if k in ape_translation_stats:
                print(f"  {k}: {ape_translation_stats[k]:.4f} m")
    else:
        print("  (not available)")

    print("\n-- evo APE rotation (deg, Umeyama SE(3) aligned) --")
    if ape_rotation_deg_stats:
        for k in ["rmse", "mean", "median", "std", "min", "max"]:
            if k in ape_rotation_deg_stats:
                print(f"  {k}: {ape_rotation_deg_stats[k]:.4f} deg")
    else:
        print("  (not available)")

    print("\n-- evo RPE translation, delta=1m --")
    if rpe1_translation_stats:
        for k in ["rmse", "mean", "median", "std", "min", "max"]:
            if k in rpe1_translation_stats:
                print(f"  {k}: {rpe1_translation_stats[k]:.4f} m")
    else:
        print("  (not available)")

    print("\n-- evo RPE rotation (deg), delta=1m --")
    if rpe1_rotation_stats:
        for k in ["rmse", "mean", "median", "std", "min", "max"]:
            if k in rpe1_rotation_stats:
                print(f"  {k}: {rpe1_rotation_stats[k]:.4f} deg")
    else:
        print("  (not available)")

    print("\n-- evo RPE translation, delta=10m --")
    if rpe10_translation_stats:
        for k in ["rmse", "mean", "median", "std", "min", "max"]:
            if k in rpe10_translation_stats:
                print(f"  {k}: {rpe10_translation_stats[k]:.4f} m")
    else:
        print("  (not available)")

    print("\n-- evo RPE rotation (deg), delta=10m --")
    if rpe10_rotation_stats:
        for k in ["rmse", "mean", "median", "std", "min", "max"]:
            if k in rpe10_rotation_stats:
                print(f"  {k}: {rpe10_rotation_stats[k]:.4f} deg")
    else:
        print("  (not available)")

    print("\n-- Raw Error (primary attack metric, no alignment) --")
    raw = metrics["raw"]
    print(f"  Duration:          {metrics['duration_s']:.1f} s  ({metrics['n_samples']} samples)")
    print(f"  3D RMSE:           {raw['rmse_3d_m']:.4f} m  |  mean: {raw['mean_3d_m']:.4f} m  |  max: {raw['max_3d_m']:.4f} m")
    print(f"  2D RMSE:           {raw['rmse_2d_m']:.4f} m  |  mean: {raw['mean_2d_m']:.4f} m  |  max: {raw['max_2d_m']:.4f} m")
    print(f"  Per-axis RMSE:")
    for axis, d in raw["per_axis"].items():
        print(f"    {axis}-axis: RMSE={d['rmse']:.4f} m, mean_abs={d['mean_abs']:.4f} m, max={d['max']:.4f} m")

    print("\n  Deviation duration (raw 2D):")
    for thresh, info in raw["deviation_duration"].items():
        print(f"    >{thresh}: {info['duration_s']:.1f}s ({info['ratio']:.1%})")

    am = metrics.get("attack_metrics", {})
    if am:
        print("\n-- Attack-Specific Metrics --")
        for k, v in am.items():
            if v is None:
                print(f"  {k}: N/A")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # ---- Save complete JSON ----
    paper_metrics = {
        "ape_translation_rmse_m": paper_ape_translation_rmse,
        "ape_rotation_rmse_deg": ape_rotation_deg_stats.get("rmse") if ape_rotation_deg_stats else None,
        "rpe_translation_max_m": paper_rpe_max_m,
        "rpe_rotation_max_deg": paper_rpe_max_deg,
        "success_threshold_m": 4.2,
        "attack_success": bool(paper_ape_translation_rmse is not None and paper_ape_translation_rmse >= 4.2),
        "note": "APE-RMSE translation + max(RPE) follow SLAMSpoof ICRA'25 Section IV-A",
    }

    complete = {
        "title": args.title,
        "paper_metrics": paper_metrics,    # <-- new: SLAMSpoof-aligned headline
        "evo": {
            "ape_translation": ape_translation_stats,
            "ape_rotation_rad": ape_rotation_rad_stats,
            "ape_rotation_deg": ape_rotation_deg_stats,
            "rpe_1m_translation": rpe1_translation_stats,
            "rpe_1m_rotation_deg": rpe1_rotation_stats,
            "rpe_10m_translation": rpe10_translation_stats,
            "rpe_10m_rotation_deg": rpe10_rotation_stats,
        },
        "raw_error": raw,
        "aligned_ref": metrics["aligned_ref"],
        "attack_metrics": am,
        "spoofer": {"x": args.spoofer_x, "y": args.spoofer_y, "distance_threshold": args.distance_threshold},
        "duration_s": metrics["duration_s"],
        "n_samples": metrics["n_samples"],
    }
    json_path = out_dir / "metrics_complete.json"
    with open(json_path, "w") as jf:
        json.dump(complete, jf, indent=2)
    print(f"\n[OK] Complete metrics: {json_path}")

    # List output files
    print(f"\n[OK] Output directory: {out_dir}")
    for fname in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname}  ({size // 1024} KB)")


if __name__ == "__main__":
    main()
