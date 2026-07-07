#!/usr/bin/env python3
"""
Sample feasible random spoofer locations from a reference trajectory.

The sampler is intentionally score-agnostic: it only uses route geometry and
attack feasibility constraints. This makes the generated positions suitable as
a random baseline against SMVS/Bi-SMVS placement methods.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _load_traj(path: str):
    df = pd.read_csv(path)
    required = {"x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Trajectory CSV missing columns: {sorted(missing)}")

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time", "x", "y"]).reset_index(drop=True)
        t = df["time"].to_numpy(dtype=np.float64)
        t = t - t[0]
    else:
        df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
        t = np.arange(len(df), dtype=np.float64)

    pts = df[["x", "y"]].to_numpy(dtype=np.float64)
    if len(pts) < 3:
        raise SystemExit("Trajectory must contain at least 3 valid poses.")
    return pts, t


def _cumdist(pts: np.ndarray) -> np.ndarray:
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    return np.concatenate([[0.0], np.cumsum(d)])


def _unit_tangent(pts: np.ndarray, idx: int) -> np.ndarray:
    i0 = max(0, idx - 2)
    i1 = min(len(pts) - 1, idx + 2)
    v = pts[i1] - pts[i0]
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def sample_positions(args):
    pts, t = _load_traj(args.traj)
    s = _cumdist(pts)
    total_len = float(s[-1])
    rng = np.random.default_rng(args.seed)

    margin_s = total_len * args.route_margin_ratio
    valid_idx = np.where((s >= margin_s) & (s <= total_len - margin_s))[0]
    if len(valid_idx) == 0:
        valid_idx = np.arange(1, len(pts) - 1)

    accepted = []
    attempts = 0
    max_attempts = max(args.max_attempts, args.n * 100)

    while len(accepted) < args.n and attempts < max_attempts:
        attempts += 1
        idx = int(rng.choice(valid_idx))
        tangent = _unit_tangent(pts, idx)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        side = int(rng.choice([-1, 1]))
        offset = float(rng.uniform(args.min_traj_dist, args.max_traj_dist))
        candidate = pts[idx] + side * offset * normal

        dists = np.hypot(pts[:, 0] - candidate[0], pts[:, 1] - candidate[1])
        min_dist = float(dists.min())
        if min_dist < args.min_traj_dist or min_dist > args.max_traj_dist:
            continue

        if accepted:
            arr = np.asarray([[p["spoofer_x"], p["spoofer_y"]] for p in accepted], dtype=np.float64)
            sep = np.hypot(arr[:, 0] - candidate[0], arr[:, 1] - candidate[1]).min()
            if sep < args.min_spoofer_separation:
                continue

        trigger = dists <= args.distance_threshold
        trigger_frames = int(trigger.sum())
        trigger_ratio = float(trigger.mean())
        if trigger_frames < args.min_trigger_frames:
            continue
        if trigger_ratio < args.min_trigger_ratio or trigger_ratio > args.max_trigger_ratio:
            continue

        trigger_idx = np.where(trigger)[0]
        if len(trigger_idx) == 0:
            continue
        if trigger_idx[0] < len(pts) * args.route_margin_ratio:
            continue
        if trigger_idx[-1] > len(pts) * (1.0 - args.route_margin_ratio):
            continue

        accepted.append({
            "random_id": len(accepted) + 1,
            "spoofer_x": float(candidate[0]),
            "spoofer_y": float(candidate[1]),
            "anchor_idx": idx,
            "anchor_x": float(pts[idx, 0]),
            "anchor_y": float(pts[idx, 1]),
            "side": side,
            "offset_m": offset,
            "min_traj_dist_m": min_dist,
            "trigger_frames": trigger_frames,
            "trigger_ratio": trigger_ratio,
            "trigger_start_s": float(t[trigger_idx[0]]),
            "trigger_end_s": float(t[trigger_idx[-1]]),
            "seed": int(args.seed),
        })

    if len(accepted) < args.n:
        raise SystemExit(
            f"Only sampled {len(accepted)} feasible positions after {attempts} attempts. "
            "Relax constraints or increase --max-attempts."
        )

    return accepted


def main():
    parser = argparse.ArgumentParser(
        description="Sample constrained-random spoofer positions from a trajectory."
    )
    parser.add_argument("--traj", required=True, help="Reference trajectory CSV with x,y and optional time columns.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--n", type=int, default=20, help="Number of random positions.")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--distance-threshold", type=float, default=30.0)
    parser.add_argument("--min-traj-dist", type=float, default=10.0)
    parser.add_argument("--max-traj-dist", type=float, default=30.0)
    parser.add_argument("--min-trigger-frames", type=int, default=50)
    parser.add_argument("--min-trigger-ratio", type=float, default=0.005)
    parser.add_argument("--max-trigger-ratio", type=float, default=0.25)
    parser.add_argument("--route-margin-ratio", type=float, default=0.05)
    parser.add_argument("--min-spoofer-separation", type=float, default=10.0)
    parser.add_argument("--max-attempts", type=int, default=10000)
    args = parser.parse_args()

    rows = sample_positions(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "random_id", "spoofer_x", "spoofer_y",
        "anchor_idx", "anchor_x", "anchor_y", "side", "offset_m",
        "min_traj_dist_m", "trigger_frames", "trigger_ratio",
        "trigger_start_s", "trigger_end_s", "seed",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} random positions: {out}")
    for r in rows:
        print(
            f"  {r['random_id']:02d}: "
            f"({r['spoofer_x']:.3f}, {r['spoofer_y']:.3f}) "
            f"min_dist={r['min_traj_dist_m']:.2f}m "
            f"trigger={r['trigger_frames']} frames ({100*r['trigger_ratio']:.2f}%)"
        )


if __name__ == "__main__":
    main()
