#!/usr/bin/env python3
"""
compare_lvisam_traj_csv.py
============================

SLAM attack evaluation following KITTI / ETH benchmark standards.

Key features
------------
1. **Umeyama SE(3) alignment** before computing error
   - Removes global drift / coordinate-frame offset between original and attack
   - Uses full rigid transform (rotation + translation + scale)

2. **ATE (Absolute Trajectory Error)**
   - RMSE, mean, median, max, std
   - Per-axis decomposition (x, y, z)

3. **RPE (Relative Pose Error)** on fixed-distance windows
   - Default windows: 1 m, 10 m
   - Reports translational RPE (rmse, mean, std)
   - Captures local drift / odometry quality

4. **Attack-specific metrics**
   - Onset time: first time deviation exceeds threshold
   - Drift velocity: mean / max slope of deviation during attack zone
   - Deviation duration: fraction of trajectory above threshold
   - Recovery ratio: deviation change in first 10 s after leaving zone

5. **Visualisation**
   - XY overlay (original vs attack)
   - Deviation time series with attack-zone shading
   - RPE cumulative plot (optional)

Usage
-----
    python3 compare_lvisam_traj_csv.py \\
        --orig orig.csv --att attack.csv \\
        --out-prefix results/compare \\
        --title "Ours vs Original" \\
        --spoofer-x 141 --spoofer-y 223 \\
        --distance-threshold 15 \\
        --wall-dist 15 --spoofing-range 80

Paper-ready output is printed to stdout and figures are saved to disk.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ===========================================================================
# Umeyama SE(3) alignment (no scipy dependency beyond numpy)
# ===========================================================================

def umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    estimate_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute optimal rigid transform (R, t, s) from src to dst via Umeyama.

    Parameters
    ----------
    src, dst : (N, 3) arrays of corresponding points.
    estimate_scale : if True, estimate a uniform scale factor.

    Returns
    -------
    R : (3, 3) rotation matrix
    t : (3,) translation vector
    s : scale factor (1.0 if estimate_scale=False)
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 point correspondences for alignment.")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Covariance
    C = (src_centered.T @ dst_centered) / n

    # SVD
    U, D, Vt = np.linalg.svd(C)
    V = Vt.T

    # Enforce right-handed coordinate system
    S = np.eye(3)
    if np.linalg.det(V @ U.T) < 0:
        V[:, -1] *= -1
        S[-1, -1] = -1

    R = V @ S @ U.T

    if estimate_scale:
        scale = np.trace(np.diag(D) @ S) / np.mean(src_centered ** 2)
    else:
        scale = 1.0

    t = dst_mean - scale * R @ src_mean
    return R, t, float(scale)


def apply_alignment(
    pts: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float,
) -> np.ndarray:
    """Apply rigid transform: aligned = s * R @ pts.T + t, returned as (N,3)."""
    return (s * (R @ pts.T).T + t[np.newaxis, :]).astype(np.float64)


# ===========================================================================
# ATE
# ===========================================================================

def compute_ate(
    aligned_att: np.ndarray,
    orig_pts: np.ndarray,
) -> Dict[str, float]:
    """Compute ATE between aligned attack and original trajectories."""
    err = aligned_att - orig_pts
    dist = np.linalg.norm(err, axis=1)

    per_axis = {
        "x_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "y_rmse": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "z_rmse": float(np.sqrt(np.mean(err[:, 2] ** 2))),
    }

    return {
        "ate_rmse": float(np.sqrt(np.mean(dist ** 2))),
        "ate_mean": float(np.mean(dist)),
        "ate_median": float(np.median(dist)),
        "ate_std": float(np.std(dist)),
        "ate_max": float(np.max(dist)),
        "ate_min": float(np.min(dist)),
        **per_axis,
    }


# ===========================================================================
# RPE  (fixed-distance window, e.g. 1 m, 10 m)
# ===========================================================================

def compute_rpe(
    orig_pts: np.ndarray,
    att_pts: np.ndarray,
    delta: float = 1.0,
) -> Dict[str, float]:
    """Compute relative pose error over fixed-distance windows.

    For each start index i, find the point j where cumulative distance
    from i is closest to delta, then compute relative displacement error.
    """
    if len(orig_pts) < 10:
        return {
            "rpe_rmse": float("nan"),
            "rpe_mean": float("nan"),
            "rpe_std": float("nan"),
            "rpe_max": float("nan"),
            "rpe_num_pairs": 0,
        }

    # Cumulative arc-length along original trajectory
    d_orig = np.linalg.norm(np.diff(orig_pts, axis=0), axis=1)
    cum_orig = np.concatenate([[0.0], np.cumsum(d_orig)])

    errs = []
    step = max(1, len(orig_pts) // 2000)  # sample at most ~2000 pairs
    for i in range(0, len(orig_pts) - 1, step):
        target = cum_orig[i] + delta
        if target > cum_orig[-1]:
            break
        # Binary search for j
        j = int(np.searchsorted(cum_orig, target))
        if j >= len(orig_pts):
            break

        dr_orig = orig_pts[j] - orig_pts[i]
        dr_att = att_pts[j] - att_pts[i]
        rel_err = np.linalg.norm(dr_orig - dr_att)
        errs.append(rel_err)

    if not errs:
        return {
            "rpe_rmse": float("nan"),
            "rpe_mean": float("nan"),
            "rpe_std": float("nan"),
            "rpe_max": float("nan"),
            "rpe_num_pairs": 0,
        }

    errs = np.array(errs)
    return {
        "rpe_rmse": float(np.sqrt(np.mean(errs ** 2))),
        "rpe_mean": float(np.mean(errs)),
        "rpe_std": float(np.std(errs)),
        "rpe_max": float(np.max(errs)),
        "rpe_num_pairs": int(len(errs)),
    }


# ===========================================================================
# Attack-specific metrics
# ===========================================================================

def compute_attack_metrics(
    orig_pts: np.ndarray,
    aligned_att: np.ndarray,
    spoofer_xy: Optional[Tuple[float, float]] = None,
    distance_threshold: Optional[float] = None,
    t_grid: Optional[np.ndarray] = None,
    attack_start_t: Optional[float] = None,
    attack_end_t: Optional[float] = None,
) -> Dict[str, float]:
    """Compute attack-effect metrics."""
    err = np.linalg.norm(aligned_att - orig_pts, axis=1)

    result: Dict[str, float] = {}

    # --- Onset times ---
    for thresh in [0.5, 1.0, 2.0, 5.0]:
        mask = err >= thresh
        if mask.any() and t_grid is not None:
            onset = float(t_grid[mask][0])
            result[f"onset_{thresh:.1f}m_s"] = onset
        else:
            result[f"onset_{thresh:.1f}m_s"] = float("nan")

    # --- Deviation duration ratio ---
    if t_grid is not None:
        total_t = float(t_grid[-1] - t_grid[0])
    else:
        total_t = float(len(err))
    for thresh in [0.5, 1.0, 2.0, 5.0]:
        ratio = float(np.mean(err >= thresh))
        result[f"dev_ratio_gt_{thresh:.1f}m"] = ratio

    # --- Attack zone drift velocity ---
    if (
        spoofer_xy is not None
        and distance_threshold is not None
        and t_grid is not None
    ):
        sx, sy = spoofer_xy
        dists = np.linalg.norm(orig_pts[:, :2] - np.array([sx, sy]), axis=1)
        in_zone = dists <= distance_threshold

        if in_zone.any() and t_grid is not None:
            t_zone = t_grid[in_zone]
            err_zone = err[in_zone]

            # Linear fit to deviation during attack
            if len(err_zone) > 5:
                coef = np.polyfit(t_zone - t_zone[0], err_zone, 1)
                result["drift_velocity_attack_zone_m_s"] = float(coef[0])
                result["drift_intercept_m"] = float(coef[1])
            else:
                result["drift_velocity_attack_zone_m_s"] = float("nan")
                result["drift_intercept_m"] = float("nan")

            result["attack_zone_duration_s"] = float(t_zone[-1] - t_zone[0])
        else:
            result["drift_velocity_attack_zone_m_s"] = float("nan")
            result["drift_intercept_m"] = float("nan")
            result["attack_zone_duration_s"] = 0.0
    else:
        result["drift_velocity_attack_zone_m_s"] = float("nan")
        result["drift_intercept_m"] = float("nan")
        result["attack_zone_duration_s"] = 0.0

    # --- Recovery ratio ---
    if (
        attack_end_t is not None
        and t_grid is not None
        and attack_end_t < t_grid[-1] - 10.0
    ):
        # Find 10-second window after leaving attack zone
        rec_end = min(attack_end_t + 10.0, t_grid[-1])
        rec_mask = (t_grid >= attack_end_t) & (t_grid <= rec_end)
        err_at_end = float(err[in_zone][-1]) if in_zone.any() else float("nan")
        err_rec_mean = float(np.mean(err[rec_mask])) if rec_mask.any() else float("nan")
        if not math.isnan(err_at_end) and not math.isnan(err_rec_mean) and err_at_end > 0:
            result["recovery_ratio_10s"] = float(err_rec_mean / err_at_end)
        else:
            result["recovery_ratio_10s"] = float("nan")
    else:
        result["recovery_ratio_10s"] = float("nan")

    return result


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SLAM attack trajectory evaluation (ATE/RPE + attack metrics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--orig", required=True, help="Original trajectory CSV")
    p.add_argument("--att", required=True, help="Attack trajectory CSV")
    p.add_argument("--out-prefix", required=True, help="Output path prefix")
    p.add_argument("--title", default="LVI-SAM Original vs Attack")
    p.add_argument("--spoofer-x", type=float, default=None)
    p.add_argument("--spoofer-y", type=float, default=None)
    p.add_argument("--distance-threshold", type=float, default=None,
                   help="Attack trigger radius (m)")
    p.add_argument("--wall-dist", type=float, default=None)
    p.add_argument("--spoofing-range", type=float, default=None)
    p.add_argument("--spoofer-heading", type=float, default=None)
    p.add_argument("--rpe-delta", type=float, default=1.0,
                   help="RPE fixed-distance window (m)")
    p.add_argument("--rpe-delta2", type=float, default=10.0,
                   help="Second RPE window (m), set 0 to skip")
    p.add_argument("--no-umeyama", action="store_true",
                   help="Skip Umeyama alignment (for debugging)")
    return p.parse_args()


def load_and_interp(
    path: str,
    t_ref: np.ndarray,
) -> np.ndarray:
    """Load CSV and interpolate x,y,z onto t_ref grid."""
    df = pd.read_csv(path)
    t = df["time"].values - df["time"].values[0]
    pts = np.column_stack([
        np.interp(t_ref, t, df["x"].values),
        np.interp(t_ref, t, df["y"].values),
        np.interp(t_ref, t, df["z"].values),
    ])
    return pts


def main() -> None:
    args = parse_args()

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load & align timestamps
    # ------------------------------------------------------------------
    df_orig = pd.read_csv(args.orig)
    df_att = pd.read_csv(args.att)

    to = df_orig["time"].values - df_orig["time"].values[0]
    ta = df_att["time"].values - df_att["time"].values[0]
    t_end = float(min(to[-1], ta[-1]))
    t_grid = np.linspace(0, t_end, 4000)

    orig_pts = load_and_interp(args.orig, t_grid)
    att_pts = load_and_interp(args.att, t_grid)

    # ------------------------------------------------------------------
    # Umeyama alignment
    # ------------------------------------------------------------------
    if not args.no_umeyama:
        R, t_vec, scale = umeyama_alignment(att_pts, orig_pts)
        aligned_att = apply_alignment(att_pts, R, t_vec, scale)
    else:
        aligned_att = att_pts.copy()
        R, t_vec, scale = np.eye(3), np.zeros(3), 1.0

    # ------------------------------------------------------------------
    # ATE
    # ------------------------------------------------------------------
    ate = compute_ate(aligned_att, orig_pts)

    # ------------------------------------------------------------------
    # RPE
    # ------------------------------------------------------------------
    rpe1 = compute_rpe(orig_pts, aligned_att, delta=args.rpe_delta)
    metrics: Dict[str, float] = {**ate, **rpe1}
    if args.rpe_delta2 and args.rpe_delta2 > 0:
        rpe2 = compute_rpe(orig_pts, aligned_att, delta=args.rpe_delta2)
        for k, v in rpe2.items():
            metrics[f"{k}_{args.rpe_delta2:.0f}m"] = v

    # ------------------------------------------------------------------
    # Attack zone from CSV timestamps
    # ------------------------------------------------------------------
    if args.spoofer_x is not None and args.distance_threshold is not None:
        dists = np.linalg.norm(
            orig_pts[:, :2] - np.array([args.spoofer_x, args.spoofer_y]), axis=1
        )
        in_zone = dists <= args.distance_threshold
        if in_zone.any():
            t0 = float(df_orig["time"].values[0])
            rel_times = df_orig["time"].values - t0
            # Interpolate which t_grid samples are in zone
            in_zone_grid = np.interp(t_grid, rel_times, in_zone.astype(float)) > 0.5
            attack_start_t = float(t_grid[in_zone_grid][0])
            attack_end_t = float(t_grid[in_zone_grid][-1])
        else:
            attack_start_t = attack_end_t = None
    else:
        attack_start_t = attack_end_t = None

    # ------------------------------------------------------------------
    # Attack-specific metrics
    # ------------------------------------------------------------------
    attack_m = compute_attack_metrics(
        orig_pts=orig_pts,
        aligned_att=aligned_att,
        spoofer_xy=(args.spoofer_x, args.spoofer_y)
        if args.spoofer_x is not None
        else None,
        distance_threshold=args.distance_threshold,
        t_grid=t_grid,
        attack_start_t=attack_start_t,
        attack_end_t=attack_end_t,
    )
    metrics.update(attack_m)

    # ------------------------------------------------------------------
    # Alignment meta
    # ------------------------------------------------------------------
    metrics["alignment_scale"] = scale
    metrics["alignment_translation_m"] = float(np.linalg.norm(t_vec))
    metrics["alignment_rotation_deg"] = float(
        np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"  {args.title}")
    print("=" * 60)

    print("\n[Alignment]")
    print(f"  Scale:           {scale:.6f}")
    print(f"  Translation:     {metrics['alignment_translation_m']:.4f} m")
    print(f"  Rotation:        {metrics['alignment_rotation_deg']:.4f} deg")

    print("\n[ATE]")
    print(f"  RMSE:            {ate['ate_rmse']:.4f} m")
    print(f"  Mean:            {ate['ate_mean']:.4f} m")
    print(f"  Median:          {ate['ate_median']:.4f} m")
    print(f"  Std:             {ate['ate_std']:.4f} m")
    print(f"  Max:             {ate['ate_max']:.4f} m")
    print(f"  Per-axis RMSE:   x={ate['x_rmse']:.4f}  "
          f"y={ate['y_rmse']:.4f}  z={ate['z_rmse']:.4f}")

    print("\n[RPE]")
    print(f"  delta={args.rpe_delta:.1f}m  RMSE={rpe1['rpe_rmse']:.4f} m  "
          f"mean={rpe1['rpe_mean']:.4f}  std={rpe1['rpe_std']:.4f}  "
          f"pairs={rpe1['rpe_num_pairs']}")
    if args.rpe_delta2 and args.rpe_delta2 > 0:
        key = f"rpe_rmse_{args.rpe_delta2:.0f}m"
        print(f"  delta={args.rpe_delta2:.1f}m  RMSE={metrics[key]:.4f} m  "
              f"pairs={metrics[f'rpe_num_pairs_{args.rpe_delta2:.0f}m']}")

    print("\n[Attack Zone]")
    if attack_start_t is not None:
        print(f"  Attack start:    {attack_start_t:.2f} s")
        print(f"  Attack end:      {attack_end_t:.2f} s")
        print(f"  Duration:        {attack_end_t - attack_start_t:.2f} s")
    else:
        print("  (no attack zone — spoofer position not provided)")

    print("\n[Onset Times]")
    for k in ["onset_0.5m_s", "onset_1.0m_s", "onset_2.0m_s", "onset_5.0m_s"]:
        v = attack_m.get(k, float("nan"))
        label = k.replace("onset_", "").replace("_s", "")
        if math.isnan(v):
            print(f"  {label:>8s}: never")
        else:
            print(f"  {label:>8s}: {v:.2f} s")

    print("\n[Attack Zone Drift]")
    v = attack_m.get("drift_velocity_attack_zone_m_s", float("nan"))
    print(f"  Drift velocity:  {v:.4f} m/s" if not math.isnan(v) else "  Drift velocity:  n/a")
    print(f"  Zone duration:   {attack_m.get('attack_zone_duration_s', 0.0):.2f} s")

    print("\n[Deviation Duration]")
    for thresh in [0.5, 1.0, 2.0, 5.0]:
        key = f"dev_ratio_gt_{thresh:.1f}m"
        ratio = attack_m.get(key, float("nan"))
        if not math.isnan(ratio):
            print(f"  >{thresh:.1f}m: {ratio*100:.2f}% of trajectory")

    rec = attack_m.get("recovery_ratio_10s", float("nan"))
    if not math.isnan(rec):
        print(f"\n  Recovery (10s post-zone): {rec:.4f}  (<1 = recovering)")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    json_path = str(out) + "_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OK] metrics saved: {json_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    err = np.linalg.norm(aligned_att - orig_pts, axis=1)
    idx = int(np.argmax(err))

    # ---- XY compare ----
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(orig_pts[:, 0], orig_pts[:, 1], label="Original", lw=1.5, alpha=0.9)
    ax.plot(aligned_att[:, 0], aligned_att[:, 1], label="Attack (aligned)",
            lw=1.5, alpha=0.9, ls="--")

    ax.scatter(orig_pts[0, 0], orig_pts[0, 1], marker="o", s=80,
               c="green", edgecolors="black", zorder=7, label="Start (orig)")
    ax.scatter(aligned_att[0, 0], aligned_att[0, 1], marker="o", s=80,
               c="blue", edgecolors="black", zorder=7, label="Start (att)")
    ax.scatter(orig_pts[-1, 0], orig_pts[-1, 1], marker="s", s=80,
               c="green", edgecolors="black", zorder=7, label="End (orig)")
    ax.scatter(aligned_att[-1, 0], aligned_att[-1, 1], marker="s", s=80,
               c="blue", edgecolors="black", zorder=7, label="End (att)")
    ax.scatter(orig_pts[idx, 0], orig_pts[idx, 1], marker="*", s=300,
               c="red", edgecolors="black", zorder=8,
               label=f"Max dev ({ate['ate_max']:.1f}m)")

    _draw_spoofer(ax, args, df_orig)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out) + "_xy_compare.png", dpi=200)
    plt.close(fig)

    # ---- Deviation time series ----
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(t_grid, err, color="purple", lw=1.0, alpha=0.8, label="Deviation")
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    _shade_attack_zone(ax2, t_grid, orig_pts, args, df_orig)
    ax2.scatter(t_grid[idx], err[idx], marker="*", s=200,
                c="red", zorder=8,
                label=f"Max ({ate['ate_max']:.1f} m at {t_grid[idx]:.1f}s)")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("deviation [m]")
    ax2.set_title(args.title + " — Deviation over time (aligned)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(str(out) + "_deviation.png", dpi=200)
    plt.close(fig2)

    # ---- Per-axis deviation ----
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    labels = ["x", "y", "z"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for axis, lbl, c in zip(range(3), labels, colors):
        ax3.plot(t_grid, aligned_att[:, axis] - orig_pts[:, axis],
                 color=c, lw=0.8, alpha=0.8, label=f"{lbl} deviation")
    ax3.axhline(0, color="gray", ls="--", lw=0.8)
    _shade_attack_zone(ax3, t_grid, orig_pts, args)
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("deviation [m]")
    ax3.set_title("Per-axis deviation (aligned)")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(str(out) + "_per_axis_deviation.png", dpi=200)
    plt.close(fig3)

    print(f"[OK] saved: {out}_xy_compare.png")
    print(f"[OK] saved: {out}_deviation.png")
    print(f"[OK] saved: {out}_per_axis_deviation.png")


# ===========================================================================
# Plot helpers
# ===========================================================================

def _draw_spoofer(ax, args, df_orig) -> None:
    """Draw spoofer marker and attack zone on XY plot."""
    if args.spoofer_x is None:
        return

    sx, sy = args.spoofer_x, args.spoofer_y

    # Trigger zone
    if args.distance_threshold is not None:
        circle = plt.Circle(
            (sx, sy), args.distance_threshold,
            fill=False, color="red", ls="--", lw=1.5, zorder=5,
            label=f"Trigger zone (r={args.distance_threshold}m)",
        )
        ax.add_patch(circle)

    # Marker
    ax.scatter(sx, sy, marker="X", s=200, c="red",
               edgecolors="black", linewidths=1, zorder=6, label="Spoofer")
    ax.annotate(
        f"Spoofer\n({sx:.1f}, {sy:.1f})",
        (sx, sy), xytext=(10, 10), textcoords="offset points",
        fontsize=8, color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="red", alpha=0.8),
        zorder=7,
    )

    # Beam + arc
    if args.spoofing_range is not None and args.wall_dist is not None:
        half = np.radians(args.spoofing_range / 2.0)
        heading = np.radians(args.spoofer_heading) if args.spoofer_heading else None

        if heading is not None:
            beam_x = [sx, sx + args.wall_dist * 1.5 * np.cos(heading)]
            beam_y = [sy, sy + args.wall_dist * 1.5 * np.sin(heading)]
            ax.plot(beam_x, beam_y, color="orange", lw=1.5, ls="-", alpha=0.7,
                    zorder=4, label=f"Spoof beam ({args.spoofing_range}deg)")

            theta = np.linspace(
                np.degrees(heading - half),
                np.degrees(heading + half),
                200,
            )
            arc_x = sx + args.wall_dist * np.cos(np.radians(theta))
            arc_y = sy + args.wall_dist * np.sin(np.radians(theta))
            ax.plot(arc_x, arc_y, color="orange", lw=1.2, ls="-", alpha=0.6, zorder=4)

        else:
            theta = np.linspace(0, 360, 400)
            arc_x = sx + args.wall_dist * np.cos(np.radians(theta))
            arc_y = sy + args.wall_dist * np.sin(np.radians(theta))
            ax.plot(arc_x, arc_y, color="orange", lw=1.0, ls="-", alpha=0.4,
                    zorder=4, label=f"Wall dist={args.wall_dist}m")

    # Triggered points
    if args.distance_threshold is not None:
        ox = df_orig["x"].values
        oy = df_orig["y"].values
        dists = np.sqrt((ox - sx) ** 2 + (oy - sy) ** 2)
        mask = dists <= args.distance_threshold
        if mask.any():
            ax.scatter(ox[mask], oy[mask], c="orange", s=20, alpha=0.7,
                       zorder=5, label=f"Triggered (n={mask.sum()})")


def _shade_attack_zone(ax, t_grid, orig_pts, args, df_orig) -> None:
    """Shade attack zone on a time-axis plot."""
    if args.spoofer_x is None or args.distance_threshold is None:
        return
    sx, sy = args.spoofer_x, args.spoofer_y
    dists = np.linalg.norm(orig_pts[:, :2] - np.array([sx, sy]), axis=1)
    mask = dists <= args.distance_threshold
    if not mask.any():
        return
    t0 = float(df_orig["time"].values[0])
    rel = df_orig["time"].values - t0
    idx_last = len(mask) - 1 - mask[::-1].argmax()
    t_start = float(np.interp(mask.argmax(), np.arange(len(mask)), rel))
    t_end_ts = float(np.interp(idx_last, np.arange(len(mask)), rel))
    ax.axvspan(t_start, t_end_ts, alpha=0.15, color="red",
               label="Attack zone")
    ax.axvline(t_start, color="red", ls=":", lw=1)
    ax.axvline(t_end_ts, color="red", ls=":", lw=1)


if __name__ == "__main__":
    main()
