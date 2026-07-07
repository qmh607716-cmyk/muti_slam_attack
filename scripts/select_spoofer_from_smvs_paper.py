#!/usr/bin/env python3
"""
select_spoofer_from_smvs_paper.py
=================================

A paper-faithful implementation of SLAMSpoof Section III-C
"Attack Location Selection" from:

  SLAMSpoof: Practical LiDAR Spoofing Attacks on Localization Systems
  Guided by Scan Matching Vulnerability Analysis, ICRA 2025.

Inputs:
  smvs.csv columns:
      timestamp, x, y, z, smvs

  vulnerability csv columns:
      x, y, z, vec_x, vec_y, smvs
    Optional but recommended:
      timestamp, frame_id

Output:
  A JSON file containing spoofer_x / spoofer_y and diagnostic information.

Important note about Eq. (7):
  The paper prints:
      g(x) = -1/m * x + (-m*Cx + Cy)

  However, the text says the perpendicular line passes through (Cx, Cy).
  Geometrically, the intercept should be:
      Cy + Cx/m

  To support both interpretations, this script provides:
      --line-formula paper      : use the printed Eq. (7) exactly
      --line-formula geometric  : use the geometrically correct line through C
      --line-formula both       : output both candidates

Default is "paper" for strict reproduction of the printed paper equation.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-9


@dataclass
class Ray:
    frame_index: int
    timestamp: Optional[float]
    x: float
    y: float
    dx: float
    dy: float
    smvs: float
    matched_vul_index: int
    matched_distance: float


@dataclass
class Line:
    # General-form line: a*x + b*y + c = 0
    a: float
    b: float
    c: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-faithful SLAMSpoof III-C spoofer placement selector."
    )

    parser.add_argument("--smvs", required=True,
                        help="SMVS CSV path. Required columns: timestamp, x, y, z, smvs (x/y/z are auto-filled from --ref-traj if missing)")
    parser.add_argument("--vul", required=True,
                        help="Vulnerability-direction CSV path. Required columns: x, y, z, vec_x, vec_y, smvs (x/y/z are auto-filled from --ref-traj if missing)")
    parser.add_argument("--ref-traj", default=None,
                        help="Reference trajectory CSV with time,x,y,z columns. Used to fill missing x/y/z in SMVS and vul CSVs via timestamp matching.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. If omitted, only prints the result.")

    parser.add_argument("--top-k", type=int, default=10,
                        help="Use top-k high frame-wise SMVS frames. Default: 10")
    parser.add_argument("--score-threshold", type=float, default=-1000.0,
                        help="Only consider frames with SMVS above this threshold. Default: -1000.0")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Outlier removal threshold in standard deviations. Paper uses ±2σ. Default: 2.0")
    parser.add_argument("--time-window", type=float, default=None,
                        help="Time window in seconds for filtering frames. If specified, frames with timestamp more than this from the median timestamp will be excluded. Default: None (no filtering)")

    parser.add_argument("--spoof-distance", type=float, default=15.0,
                        help="Distance from the trajectory intersection point to the spoofer. Paper suggests 10-15m. Default: 15.0")
    parser.add_argument("--distance-threshold", type=float, default=15.0,
                        help="distance_threshold written to config JSON. Default: 15.0")
    parser.add_argument("--spoofing-range", type=float, default=80.0,
                        help="spoofing_range written to config JSON. Default: 80.0")

    parser.add_argument("--line-formula", choices=["paper", "geometric", "both"], default="paper",
                        help=(
                            "paper: use printed Eq.(7), b=-m*Cx+Cy; "
                            "geometric: use line through C, b=Cy+Cx/m; "
                            "both: output both. Default: paper"
                        ))
    parser.add_argument("--candidate-side", choices=["same_as_center", "positive_normal", "negative_normal", "both"],
                        default="same_as_center",
                        help=(
                            "How to choose the side on the perpendicular placement line. "
                            "same_as_center places the spoofer on the side of the critical-object center relative to the trajectory. "
                            "both outputs both candidates. Default: same_as_center"
                        ))

    parser.add_argument("--match-mode", choices=["auto", "index", "timestamp", "nearest_xy"], default="auto",
                        help=(
                            "How to match each high-SMVS frame to a vulnerable direction row. "
                            "auto uses timestamp if both CSVs have timestamp, else index if lengths match, else nearest_xy."
                        ))
    parser.add_argument("--max-match-distance", type=float, default=None,
                        help="Optional warning threshold for nearest_xy matching distance in meters.")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print output JSON but do not write the file.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print diagnostic information.")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise SystemExit(f"{name} missing required columns: {sorted(missing)}")


def load_csvs(
    smvs_path: str, vul_path: str, ref_traj_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    smvs_df = pd.read_csv(smvs_path)
    vul_df = pd.read_csv(vul_path)

    require_columns(smvs_df, {"timestamp", "smvs"}, "SMVS CSV")
    require_columns(vul_df, {"vec_x", "vec_y", "smvs"}, "Vulnerability CSV")

    smvs_df = smvs_df.copy()
    vul_df = vul_df.copy()

    smvs_df["_row_index"] = np.arange(len(smvs_df), dtype=int)
    vul_df["_row_index"] = np.arange(len(vul_df), dtype=int)

    for col in ["timestamp", "smvs"]:
        smvs_df[col] = pd.to_numeric(smvs_df[col], errors="coerce")
    for col in ["vec_x", "vec_y", "smvs"]:
        vul_df[col] = pd.to_numeric(vul_df[col], errors="coerce")
    if "timestamp" in vul_df.columns:
        vul_df["timestamp"] = pd.to_numeric(vul_df["timestamp"], errors="coerce")

    for col in ["x", "y", "z"]:
        if col not in smvs_df.columns:
            smvs_df[col] = np.nan
        if col not in vul_df.columns:
            vul_df[col] = np.nan
        smvs_df[col] = pd.to_numeric(smvs_df[col], errors="coerce")
        vul_df[col] = pd.to_numeric(vul_df[col], errors="coerce")

    # Fill only missing coordinates. Existing SMVS/vulnerability coordinates are
    # the data to reproduce; overwriting all rows changes the placement result.
    need_traj_fill = (
        smvs_df[["x", "y", "z"]].isna().any(axis=1).any() or
        vul_df[["x", "y", "z"]].isna().any(axis=1).any()
    )
    if need_traj_fill and ref_traj_path is not None:
        traj_df = pd.read_csv(ref_traj_path)
        require_columns(traj_df, {"time", "x", "y", "z"}, "Reference trajectory CSV")
        traj_time = pd.to_numeric(traj_df["time"], errors="coerce").to_numpy()
        traj_x = pd.to_numeric(traj_df["x"], errors="coerce").to_numpy()
        traj_y = pd.to_numeric(traj_df["y"], errors="coerce").to_numpy()
        traj_z = pd.to_numeric(traj_df["z"], errors="coerce").to_numpy()
        n_traj = len(traj_df)

        def fill_missing_xyz(df: pd.DataFrame, name: str) -> None:
            missing = df[["x", "y", "z"]].isna().any(axis=1)
            if not missing.any():
                return

            idx = df.index[missing].to_numpy()
            row_idx = df.loc[idx, "_row_index"].to_numpy(dtype=int)

            fill_x = np.full(len(idx), np.nan, dtype=float)
            fill_y = np.full(len(idx), np.nan, dtype=float)
            fill_z = np.full(len(idx), np.nan, dtype=float)

            if name == "vul":
                aligned = row_idx < len(smvs_df)
                if aligned.any():
                    sx = smvs_df.iloc[row_idx[aligned]]["x"].to_numpy(dtype=float)
                    sy = smvs_df.iloc[row_idx[aligned]]["y"].to_numpy(dtype=float)
                    sz = smvs_df.iloc[row_idx[aligned]]["z"].to_numpy(dtype=float)
                    finite = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(sz)
                    aligned_pos = np.flatnonzero(aligned)
                    fill_x[aligned_pos[finite]] = sx[finite]
                    fill_y[aligned_pos[finite]] = sy[finite]
                    fill_z[aligned_pos[finite]] = sz[finite]

            remaining = ~(np.isfinite(fill_x) & np.isfinite(fill_y) & np.isfinite(fill_z))
            if remaining.any():
                traj_idx = np.clip(row_idx[remaining], 0, max(n_traj - 1, 0))
                if "timestamp" in df.columns:
                    t_vals = df.loc[idx[remaining], "timestamp"].to_numpy(dtype=float)
                    finite_t = np.isfinite(t_vals) & np.isfinite(traj_time).any()
                    if finite_t.any():
                        nearest = np.argmin(
                            np.abs(traj_time[np.newaxis, :] - t_vals[finite_t, np.newaxis]),
                            axis=1,
                        )
                        traj_idx[finite_t] = nearest

                rem_pos = np.flatnonzero(remaining)
                fill_x[rem_pos] = traj_x[traj_idx]
                fill_y[rem_pos] = traj_y[traj_idx]
                fill_z[rem_pos] = traj_z[traj_idx]

            df.loc[idx, "x"] = fill_x
            df.loc[idx, "y"] = fill_y
            df.loc[idx, "z"] = fill_z
            print(
                f"[INFO] Filled {len(idx)} missing {name} x/y/z rows from {ref_traj_path}",
                file=sys.stderr,
            )

        fill_missing_xyz(smvs_df, "SMVS")
        fill_missing_xyz(vul_df, "vul")
    elif need_traj_fill:
        print(
            "[WARN] SMVS or vul CSV has missing x/y/z values and --ref-traj "
            "was not provided; missing rows will be dropped.",
            file=sys.stderr,
        )

    smvs_df = smvs_df.dropna(subset=["timestamp", "x", "y", "z", "smvs"])
    vul_df = vul_df.dropna(subset=["x", "y", "z", "vec_x", "vec_y", "smvs"])

    if len(smvs_df) == 0:
        raise SystemExit("SMVS CSV has no valid numeric rows.")
    if len(vul_df) == 0:
        raise SystemExit("Vulnerability CSV has no valid numeric rows.")

    return smvs_df, vul_df


def select_high_smvs_frames(
    smvs_df: pd.DataFrame,
    threshold: float,
    top_k: int,
    time_window: Optional[float] = None,
) -> pd.DataFrame:
    if top_k < 2:
        raise SystemExit("--top-k must be >= 2 because intersections require at least two rays.")

    # Sort by SMVS (direction depends on SMVS convention):
    #   - Positive SMVS (higher=better): use nlargest
    #   - Negative SMVS (lower=better, as in jackal dataset): use nsmallest
    if smvs_df["smvs"].max() > 0:
        candidates = smvs_df[smvs_df["smvs"] > threshold].copy()
        if len(candidates) == 0:
            print(f"[WARN] No frames with smvs > {threshold}; falling back to all frames.", file=sys.stderr)
            candidates = smvs_df.copy()
        candidates = candidates.nlargest(min(top_k, len(candidates)), "smvs").reset_index(drop=True)
    else:
        candidates = smvs_df[smvs_df["smvs"] < threshold].copy()
        if len(candidates) == 0:
            print(f"[WARN] No frames with smvs < {threshold}; falling back to all frames.", file=sys.stderr)
            candidates = smvs_df.copy()
        candidates = candidates.nsmallest(min(top_k, len(candidates)), "smvs").reset_index(drop=True)

    # Apply time window filter if specified
    if time_window is not None and time_window > 0 and len(candidates) > 0:
        median_t = float(candidates["timestamp"].median())
        before_filter = len(candidates)
        candidates = candidates[
            (candidates["timestamp"] >= median_t - time_window) &
            (candidates["timestamp"] <= median_t + time_window)
        ].reset_index(drop=True)

        if len(candidates) < 2:
            print(f"[WARN] Time window filter left only {len(candidates)} frames; reverting to all selected.", file=sys.stderr)
            candidates = smvs_df.nlargest(min(top_k, len(smvs_df)), "smvs").reset_index(drop=True)
        else:
            print(f"[TIME FILTER] Removed {before_filter - len(candidates)} frames outside time window ±{time_window}s from median t={median_t:.2f}s", file=sys.stderr)

    return candidates


def decide_match_mode(smvs_df: pd.DataFrame, vul_df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested

    if "timestamp" in vul_df.columns and vul_df["timestamp"].notna().any():
        return "timestamp"

    if len(smvs_df) == len(vul_df):
        return "index"

    return "nearest_xy"


def match_vulnerability_row(
    frame_row: pd.Series,
    smvs_df: pd.DataFrame,
    vul_df: pd.DataFrame,
    mode: str,
) -> Tuple[pd.Series, float]:
    if mode == "timestamp":
        if "timestamp" not in vul_df.columns:
            raise SystemExit("match-mode=timestamp requires a timestamp column in vulnerability CSV.")
        idx = (vul_df["timestamp"] - frame_row["timestamp"]).abs().idxmin()
        matched = vul_df.loc[idx]
        # Report spatial distance as diagnostic.
        dist = float(math.hypot(matched["x"] - frame_row["x"], matched["y"] - frame_row["y"]))
        return matched, dist

    if mode == "index":
        original_idx = int(frame_row["_row_index"])
        if original_idx >= len(vul_df):
            raise SystemExit("match-mode=index failed because CSV lengths/indices do not align.")
        matched = vul_df.iloc[original_idx]
        dist = float(math.hypot(matched["x"] - frame_row["x"], matched["y"] - frame_row["y"]))
        return matched, dist

    if mode == "nearest_xy":
        dx = vul_df["x"].to_numpy() - float(frame_row["x"])
        dy = vul_df["y"].to_numpy() - float(frame_row["y"])
        distances = np.hypot(dx, dy)
        pos = int(np.argmin(distances))
        return vul_df.iloc[pos], float(distances[pos])

    raise ValueError(f"unknown match mode: {mode}")


def build_rays(
    high_df: pd.DataFrame,
    smvs_df: pd.DataFrame,
    vul_df: pd.DataFrame,
    match_mode: str,
    max_match_distance: Optional[float],
    verbose: bool,
) -> List[Ray]:
    rays: List[Ray] = []

    for _, row in high_df.iterrows():
        # Skip frames with zero position (outside ref_traj range, fill fell back to origin)
        if abs(float(row["x"])) < EPS and abs(float(row["y"])) < EPS:
            if verbose:
                print(f"[WARN] skip frame row={int(row['_row_index'])}: zero position (x=y=0).", file=sys.stderr)
            continue

        matched, matched_distance = match_vulnerability_row(row, smvs_df, vul_df, match_mode)

        # Vulnerable direction vector: from robot (x,y) pointing toward critical object.
        # Negate because the stored vec points AWAY from the critical object (cos/sin
        # reconstructed from theta=atan2(y,x)+180 flips the direction).
        vx = -float(matched["vec_x"])
        vy = -float(matched["vec_y"])
        norm = math.hypot(vx, vy)
        if norm < EPS:
            if verbose:
                print(f"[WARN] skip frame row={int(row['_row_index'])}: zero vulnerable direction.", file=sys.stderr)
            continue

        vx /= norm
        vy /= norm

        if max_match_distance is not None and matched_distance > max_match_distance:
            print(
                f"[WARN] large frame-vulnerability match distance: "
                f"frame_row={int(row['_row_index'])}, vul_row={int(matched['_row_index'])}, "
                f"dist={matched_distance:.3f} m",
                file=sys.stderr,
            )

        ray = Ray(
            frame_index=int(row["_row_index"]),
            timestamp=float(row["timestamp"]) if not pd.isna(row["timestamp"]) else None,
            x=float(row["x"]),
            y=float(row["y"]),
            dx=float(vx),
            dy=float(vy),
            smvs=float(row["smvs"]),
            matched_vul_index=int(matched["_row_index"]),
            matched_distance=matched_distance,
        )
        rays.append(ray)

        if verbose:
            print(
                f"[RAY] frame={ray.frame_index}, t={ray.timestamp}, "
                f"P=({ray.x:.3f},{ray.y:.3f}), d=({ray.dx:.4f},{ray.dy:.4f}), "
                f"smvs={ray.smvs:.3f}, vul={ray.matched_vul_index}, match_dist={ray.matched_distance:.3f}",
                file=sys.stderr,
            )

    if len(rays) < 2:
        raise SystemExit("Fewer than two valid rays. Cannot compute intersections.")

    return rays


def intersect_rays(r1: Ray, r2: Ray) -> Optional[Tuple[float, float, float, float]]:
    """
    Intersect two half-lines:
      P1 + t1*d1, t1 >= 0
      P2 + t2*d2, t2 >= 0

    Returns:
      (x, y, t1, t2) or None.
    """
    det = r1.dx * r2.dy - r1.dy * r2.dx
    if abs(det) < EPS:
        return None

    px = r2.x - r1.x
    py = r2.y - r1.y

    t1 = (px * r2.dy - py * r2.dx) / det
    t2 = (px * r1.dy - py * r1.dx) / det

    ix = r1.x + t1 * r1.dx
    iy = r1.y + t1 * r1.dy
    return float(ix), float(iy), float(t1), float(t2)


def compute_intersections(rays: List[Ray]) -> pd.DataFrame:
    rows = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            out = intersect_rays(rays[i], rays[j])
            if out is None:
                continue
            ix, iy, t1, t2 = out
            rows.append({
                "x": ix,
                "y": iy,
                "ray_i": i,
                "ray_j": j,
                "frame_i": rays[i].frame_index,
                "frame_j": rays[j].frame_index,
                "t_i": t1,
                "t_j": t2,
            })

    if len(rows) == 0:
        raise SystemExit("No valid half-line intersections were found.")

    return pd.DataFrame(rows)


def remove_outliers_sigma(points_df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    x = points_df["x"].to_numpy(dtype=float)
    y = points_df["y"].to_numpy(dtype=float)

    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    std_x = float(np.std(x))
    std_y = float(np.std(y))

    # Degenerate dimensions should not remove all points.
    x_ok = np.ones_like(x, dtype=bool) if std_x < EPS else np.abs(x - mean_x) <= sigma * std_x
    y_ok = np.ones_like(y, dtype=bool) if std_y < EPS else np.abs(y - mean_y) <= sigma * std_y

    mask = x_ok & y_ok
    filtered = points_df.loc[mask].copy()

    if len(filtered) == 0:
        print("[WARN] Sigma filtering removed all intersections; using unfiltered intersections.", file=sys.stderr)
        filtered = points_df.copy()

    filtered.attrs["mean_x"] = mean_x
    filtered.attrs["mean_y"] = mean_y
    filtered.attrs["std_x"] = std_x
    filtered.attrs["std_y"] = std_y

    return filtered


def bbox_center(points_df: pd.DataFrame) -> Tuple[float, float, Dict[str, float]]:
    min_x = float(points_df["x"].min())
    max_x = float(points_df["x"].max())
    min_y = float(points_df["y"].min())
    max_y = float(points_df["y"].max())

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    info = {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "center_x": cx,
        "center_y": cy,
    }
    return cx, cy, info


def fit_trajectory_y_mx_n(high_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Paper says the linear function is derived by least squares on top SMVS points.
    Here we use the high-SMVS frames selected by threshold + top-k.
    """
    x = high_df["x"].to_numpy(dtype=float)
    y = high_df["y"].to_numpy(dtype=float)

    if len(x) < 2:
        raise SystemExit("Need at least two trajectory points for linear fitting.")

    if np.std(x) < EPS:
        raise SystemExit(
            "The selected trajectory points are nearly vertical in x-y. "
            "The paper's y=m*x+n model is ill-conditioned for this case."
        )

    A = np.column_stack([x, np.ones_like(x)])
    m, n = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(n)


def trajectory_line(m: float, n: float) -> Line:
    # y = m*x+n  ->  m*x - y + n = 0
    return Line(a=m, b=-1.0, c=n)


def perpendicular_line_paper(m: float, cx: float, cy: float) -> Line:
    """
    Printed Eq. (7) in the paper:
        g(x) = -1/m * x + (-m*Cx + Cy)

    This is kept as a literal reproduction option. It does not generally pass
    through C=(Cx,Cy), despite the surrounding text saying that it should.
    """
    if abs(m) < EPS:
        # Eq. (7) is undefined when m=0; use the text's perpendicular-line
        # interpretation for this degenerate case.
        return Line(a=1.0, b=0.0, c=-cx)

    q = -1.0 / m
    b = -m * cx + cy
    return Line(a=q, b=-1.0, c=b)


def perpendicular_line_geometric(m: float, cx: float, cy: float) -> Line:
    """
    Geometrically correct perpendicular line through C=(cx,cy).
    """
    if abs(m) < EPS:
        return Line(a=1.0, b=0.0, c=-cx)

    q = -1.0 / m
    b = cy + cx / m
    return Line(a=q, b=-1.0, c=b)


def intersect_lines(l1: Line, l2: Line) -> Optional[Tuple[float, float]]:
    det = l1.a * l2.b - l2.a * l1.b
    if abs(det) < EPS:
        return None

    # Solve:
    # a1*x + b1*y = -c1
    # a2*x + b2*y = -c2
    x = ((-l1.c) * l2.b - (-l2.c) * l1.b) / det
    y = (l1.a * (-l2.c) - l2.a * (-l1.c)) / det
    return float(x), float(y)


def line_direction(line: Line) -> np.ndarray:
    """
    A direction vector parallel to a*x+b*y+c=0 is (b, -a).
    """
    d = np.array([line.b, -line.a], dtype=float)
    norm = np.linalg.norm(d)
    if norm < EPS:
        raise SystemExit("Degenerate line direction.")
    return d / norm


def signed_line_value(line: Line, x: float, y: float) -> float:
    return line.a * x + line.b * y + line.c


def place_from_trajectory_intersection(
    traj: Line,
    place_line: Line,
    cx: float,
    cy: float,
    distance: float,
    candidate_side: str,
) -> Dict[str, object]:
    foot = intersect_lines(traj, place_line)
    if foot is None:
        raise SystemExit("Trajectory line and placement line are parallel; cannot place spoofer.")

    hx, hy = foot
    d = line_direction(place_line)

    p_pos = np.array([hx, hy], dtype=float) + distance * d
    p_neg = np.array([hx, hy], dtype=float) - distance * d

    candidates = {
        "positive_normal": {"spoofer_x": float(p_pos[0]), "spoofer_y": float(p_pos[1])},
        "negative_normal": {"spoofer_x": float(p_neg[0]), "spoofer_y": float(p_neg[1])},
    }

    if candidate_side == "both":
        selected = None
    elif candidate_side in ("positive_normal", "negative_normal"):
        selected = candidates[candidate_side]
    elif candidate_side == "same_as_center":
        # Select the candidate on the same side of the trajectory as the bounding-box center.
        center_sign = signed_line_value(traj, cx, cy)
        pos_sign = signed_line_value(traj, float(p_pos[0]), float(p_pos[1]))
        neg_sign = signed_line_value(traj, float(p_neg[0]), float(p_neg[1]))

        if abs(center_sign) < EPS:
            # If C is exactly on the trajectory, there is no well-defined "same side".
            # Return both candidates and mark no unique selection.
            selected = None
        elif center_sign * pos_sign >= 0:
            selected = candidates["positive_normal"]
        elif center_sign * neg_sign >= 0:
            selected = candidates["negative_normal"]
        else:
            # Should not happen, but keep it safe.
            selected = None
    else:
        raise ValueError(f"Unknown candidate_side: {candidate_side}")

    return {
        "trajectory_intersection": {"x": float(hx), "y": float(hy)},
        "candidates": candidates,
        "selected": selected,
    }


def choose_primary_variant(variant_outputs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """
    Use paper result as primary if available; otherwise use geometric.
    """
    if "paper" in variant_outputs:
        return variant_outputs["paper"]
    return variant_outputs["geometric"]


def select_spoofer(
    smvs_df: pd.DataFrame,
    vul_df: pd.DataFrame,
    top_k: int,
    score_threshold: float,
    sigma: float,
    spoof_distance: float,
    line_formula: str,
    candidate_side: str,
    match_mode_requested: str,
    max_match_distance: Optional[float],
    verbose: bool,
    time_window: Optional[float] = None,
) -> Dict[str, object]:
    high_df = select_high_smvs_frames(smvs_df, score_threshold, top_k, time_window)
    match_mode = decide_match_mode(smvs_df, vul_df, match_mode_requested)

    rays = build_rays(
        high_df=high_df,
        smvs_df=smvs_df,
        vul_df=vul_df,
        match_mode=match_mode,
        max_match_distance=max_match_distance,
        verbose=verbose,
    )

    intersections = compute_intersections(rays)
    filtered = remove_outliers_sigma(intersections, sigma=sigma)

    # Critical object center (Cx, Cy): bbox of filtered intersection points, per paper III-C.
    cx, cy, bbox = bbox_center(filtered)

    m, n = fit_trajectory_y_mx_n(high_df)
    traj = trajectory_line(m, n)

    variants_to_compute = ["paper", "geometric"] if line_formula == "both" else [line_formula]
    variant_outputs: Dict[str, Dict[str, object]] = {}

    for variant in variants_to_compute:
        if variant == "paper":
            place_line = perpendicular_line_paper(m, cx, cy)
            formula_note = "printed Eq.(7): g(x)=-1/m*x+(-m*Cx+Cy)"
        elif variant == "geometric":
            place_line = perpendicular_line_geometric(m, cx, cy)
            formula_note = "geometrically through C: g(x)=-1/m*x+(Cy+Cx/m)"
        else:
            raise ValueError(variant)

        placement = place_from_trajectory_intersection(
            traj=traj,
            place_line=place_line,
            cx=cx,
            cy=cy,
            distance=spoof_distance,
            candidate_side=candidate_side,
        )

        selected = placement["selected"]
        variant_outputs[variant] = {
            "formula_note": formula_note,
            "placement_line_general": asdict(place_line),
            "trajectory_intersection": placement["trajectory_intersection"],
            "candidates": placement["candidates"],
            "selected": selected,
        }

    primary = choose_primary_variant(variant_outputs)
    primary_selected = primary["selected"]

    result: Dict[str, object] = {
        "method": "paper_III_C_attack_location_selection",
        "line_formula_mode": line_formula,
        "candidate_side": candidate_side,
        "match_mode": match_mode,
        "spoofer_x": None if primary_selected is None else primary_selected["spoofer_x"],
        "spoofer_y": None if primary_selected is None else primary_selected["spoofer_y"],
        "critical_object_center": {"x": float(cx), "y": float(cy)},
        "trajectory_fit": {"m": float(m), "n": float(n), "model": "y=m*x+n"},
        "bbox": bbox,
        "counts": {
            "num_smvs_rows": int(len(smvs_df)),
            "num_vul_rows": int(len(vul_df)),
            "num_high_smvs_frames": int(len(high_df)),
            "num_rays": int(len(rays)),
            "num_intersections": int(len(intersections)),
            "num_filtered_intersections": int(len(filtered)),
        },
        "time_window_filter": {
            "enabled": time_window is not None,
            "value": time_window,
        },
        "sigma_filter": {
            "sigma": float(sigma),
            "mean_x": float(filtered.attrs.get("mean_x", np.nan)),
            "mean_y": float(filtered.attrs.get("mean_y", np.nan)),
            "std_x": float(filtered.attrs.get("std_x", np.nan)),
            "std_y": float(filtered.attrs.get("std_y", np.nan)),
        },
        "variants": variant_outputs,
        "top_smvs_frames": [
            {
                "frame_index": int(r["_row_index"]),
                "timestamp": float(r["timestamp"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "z": float(r["z"]),
                "smvs": float(r["smvs"]),
            }
            for _, r in high_df.iterrows()
        ],
        "rays": [asdict(r) for r in rays],
        "notes": [
            "The high-SMVS frames are selected by threshold then top-k sorting.",
            "The trajectory line is fitted using the selected top-SMVS frames, as described in Section III-C.",
            "Half-line intersections require t>=0 on both rays.",
            "If candidate_side=both or the critical center lies exactly on the fitted trajectory, spoofer_x/y may be null and both candidates are provided under variants.",
        ],
    }

    return result


def build_config_output(result: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    return {
        "main": {
            "spoofer_x": result["spoofer_x"],
            "spoofer_y": result["spoofer_y"],
            "distance_threshold": args.distance_threshold,
            "spoofing_range": args.spoofing_range,
        },
        "selection_info": result,
    }


def main() -> None:
    args = parse_args()

    smvs_df, vul_df = load_csvs(args.smvs, args.vul, args.ref_traj)

    result = select_spoofer(
        smvs_df=smvs_df,
        vul_df=vul_df,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        sigma=args.sigma,
        spoof_distance=args.spoof_distance,
        line_formula=args.line_formula,
        candidate_side=args.candidate_side,
        match_mode_requested=args.match_mode,
        max_match_distance=args.max_match_distance,
        verbose=args.verbose,
        time_window=args.time_window,
    )

    output_data = build_config_output(result, args)

    print("=" * 72)
    print("SLAMSpoof III-C spoofer placement selection")
    print("=" * 72)
    print(f"method:            {result['method']}")
    print(f"line_formula_mode: {result['line_formula_mode']}")
    print(f"candidate_side:    {result['candidate_side']}")
    print(f"match_mode:        {result['match_mode']}")
    print(f"critical center:   ({result['critical_object_center']['x']:.6f}, {result['critical_object_center']['y']:.6f})")
    print(f"trajectory:        y = {result['trajectory_fit']['m']:.6f} * x + {result['trajectory_fit']['n']:.6f}")

    if result["spoofer_x"] is not None and result["spoofer_y"] is not None:
        print(f"selected spoofer:  ({result['spoofer_x']:.6f}, {result['spoofer_y']:.6f})")
    else:
        print("selected spoofer:  not unique; see variants.*.candidates in JSON")

    print(f"intersections:     {result['counts']['num_intersections']} raw, {result['counts']['num_filtered_intersections']} after sigma filter")

    if args.output:
        if args.dry_run:
            print("\n[DRY RUN] Would write:")
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
        else:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] wrote {args.output}")
    else:
        print("\nJSON output:")
        print(json.dumps(output_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
