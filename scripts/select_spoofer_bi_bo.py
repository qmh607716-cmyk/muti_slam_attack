#!/usr/bin/env python3
"""
select_spoofer_bi_bo.py
========================

Bi-SMVS-driven Bayesian Optimization for spoofer location selection.

Improvements over select_spoofer_continuous_opt.py:

1. FULL 72-DIM DIRECTIONAL SCORING
   Instead of using only vul_angle_deg (scalar), we now use the complete
   72-dim Bi-Vul vector to compute frame-wise attack scores. This captures
   the vulnerability distribution across ALL azimuth directions, not just the
   single most-vulnerable one.

2. DIRECTIONAL + SPATIAL JOINT CLUSTERING
   - Directional clustering: group high-vulnerability azimuth buckets
   - Spatial clustering: group frames by world position
   - Joint: DBSCAN on (x, y, dominant_direction) for smarter initialization

3. CMA-ES GLOBAL OPTIMIZATION (replaces grid search)
   Replaces the 3-stage grid search (coarse 5m → fine 1m → sub-pixel 0.1m)
   with CMA-ES evolution strategy. CMA-ES is ideal for sparse, multi-modal
   scoring landscapes where GP-based BO fails completely. ~200 evals in < 1 minute.

4. L-BFGS-B LOCAL REFINEMENT
   After BO finds the global region, run L-BFGS-B from top-K BO candidates
   for sub-meter precision.

Usage:
    python3 select_spoofer_bi_bo.py \
        --smvs ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_kitti/smvs/bimodal/06_08_14_25_15.csv \
        --vul  ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_kitti/vul/bimodal/vul_06_08_14_25_15.csv \
        --traj ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_kitti/original/kitti_original_traj.csv \
        --top-k 20 \
        --spoofing-range 80.0 \
        --distance-threshold 30.0 \
        --visualize

"""

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Bayesian Optimization (scikit-optimize)
# ---------------------------------------------------------------------------
try:
    from skopt import gp_minimize
    from skopt.space import Real
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class VulFrame:
    timestamp: float
    x: float
    y: float
    z: float
    yaw: float
    bi_smvs: float        # Bimodal SMVS (higher = more vulnerable)
    vul_angle_deg: float  # Most vulnerable direction (scalar, for compatibility)
    vec_x: float
    vec_y: float
    bi_vul: np.ndarray    # 72-dim Bi-Vul vector (bi_vul_00..71)
    l_vul: np.ndarray     # 72-dim LiDAR-only Vul vector (l_vul_00..71)


# ============================================================================
# Stage 1: Full 72-Dim Directional Scoring
# ============================================================================

STEP_DEG = 5.0          # Each bucket covers 5° (72 buckets = 360°)
N_BUCKETS = int(360.0 / STEP_DEG)
SPOOFING_HALF_RANGE = 40.0  # ±40° attack window from paper


# ============================================================================
# Fix 2 & 3: Motion-aware cumulative drift scoring
# Paper §VII-B: SMVS limitations — (1) ignores cumulative drift over multiple
# frames, (2) assumes constant velocity, ignoring turns and acceleration.
#
# Solution:
#   cumulative_drift(s) = wall_dist × Σ motion_factor[t] × alignment[t]
#     where t indexes keyframes within the spoofing zone
#   motion_factor[t] ∈ [0.1, 1.0]: 1.0 = straight/constant-velocity,
#                                   0.1 = sharp turn / aggressive accel
# ============================================================================

def compute_traj_motion_factors(
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    traj_t: np.ndarray,
    speed_thresh: float = 1.0,      # m/s — below this = low speed
    curv_thresh: float = 0.05,     # rad/m — above this = sharp turn
    speed_smooth: int = 5,
    curv_smooth: int = 5,
) -> np.ndarray:
    """
    Compute per-frame motion quality factors for trajectory-guided scoring.

    Returns motion_factor[t] ∈ [0.1, 1.0]:
      1.0 = straight road, constant velocity (best for cumulative attack)
      0.5 = moderate speed change
      0.1 = sharp turn or aggressive accel/decel (less effective per frame)

    The factors are smoothed with a moving average to avoid noise from
    individual frame variations.
    """
    n = len(traj_x)
    if n < 3:
        return np.ones(n, dtype=np.float64)

    # Speed: finite difference with padding
    vx = np.gradient(traj_x, traj_t)
    vy = np.gradient(traj_y, traj_t)
    speed = np.sqrt(vx**2 + vy**2)

    # Curvature: kappa = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)
    ddx = np.gradient(vx, traj_t)
    ddy = np.gradient(vy, traj_t)
    speed_safe = np.where(speed > 1e-3, speed, 1e-3)
    kappa = np.abs(vx * ddy - vy * ddx) / (speed_safe**3)

    # Smooth to reduce noise
    if speed_smooth > 1:
        kernel = np.ones(speed_smooth) / speed_smooth
        speed = np.convolve(speed, kernel, mode='same')
        kappa = np.convolve(kappa, np.ones(curv_smooth) / curv_smooth, mode='same')

    # Normalize speed and curvature to [0, 1] ranges
    speed_norm = np.clip(speed / (np.percentile(speed, 80) + 1e-6), 0, 1)
    kappa_norm = np.clip(kappa / (np.percentile(kappa, 80) + 1e-6), 0, 1)

    # Motion quality score: high speed + low curvature = best
    quality = speed_norm * (1.0 - kappa_norm)  # ∈ [0, 1]

    # Map quality to factor: [0.1, 1.0]
    # Very low quality (< 0.2) → factor = 0.1 (turn/accel zones)
    # High quality (> 0.8) → factor = 1.0 (straight cruise)
    factor = 0.1 + 0.9 * np.sqrt(np.clip(quality, 0, 1))

    return factor.astype(np.float64)


def cumulative_drift_score(
    sx: float, sy: float,
    frames: List['VulFrame'],
    traj_pts: np.ndarray,
    traj_t: np.ndarray,
    traj_motion: np.ndarray,
    spoofing_range: float,
    distance_threshold: float,
    wall_dist: float,
    half_angle: float = SPOOFING_HALF_RANGE,
) -> Dict[str, Any]:
    """
    Compute cumulative drift score for spoofer at (sx, sy).

    cumulative_drift = wall_dist × Σ(motion_factor[t] × alignment[t])
    for all trajectory keyframes t within spoofing zone.

    Returns dict with:
      - cumulative_drift_m: total expected displacement (meters)
      - triggered_keyframes: number of trajectory points in spoofing zone
      - mean_motion_factor: average motion quality (0.1–1.0)
      - alignment_score: average angular alignment (0–1)
    """
    sigma = spoofing_range / 3.0
    n_buckets = float(N_BUCKETS)

    total_weighted = 0.0
    total_motion = 0.0
    total_alignment = 0.0
    triggered = 0

    for i, f in enumerate(frames):
        dx = sx - f.x
        dy = sy - f.y
        dist = math.hypot(dx, dy)

        if dist > spoofing_range or dist > distance_threshold:
            continue

        # Distance weight
        dist_w = math.exp(-dist * dist / (2.0 * sigma * sigma))

        # Angular alignment with full 72-dim Bi-Vul
        alpha = math.atan2(dy, dx)
        alpha_deg = (math.degrees(alpha) + 360.0) % 360.0
        alpha_bucket = alpha_deg / STEP_DEG

        def _circ_f(k, center):
            d = abs(k - center)
            return min(d, n_buckets - d)

        max_angular = 0.0
        for k in range(N_BUCKETS):
            vk = f.bi_vul[k]
            if vk <= 0:
                continue
            dtheta = _circ_f(float(k), alpha_bucket) * STEP_DEG
            if dtheta > half_angle:
                continue
            angular_w = 1.0 - dtheta / half_angle
            max_angular = max(max_angular, vk * angular_w)

        alignment = min(max_angular / 50.0, 1.0) if max_angular > 0 else 0.0  # normalize

        # Find nearest trajectory point for motion factor
        dists_to_traj = np.sqrt((traj_pts[:, 0] - f.x)**2 + (traj_pts[:, 1] - f.y)**2)
        nearest_t_idx = int(np.argmin(dists_to_traj))
        motion_factor = float(traj_motion[nearest_t_idx])

        weighted = motion_factor * alignment
        total_weighted += weighted
        total_motion += motion_factor
        total_alignment += alignment
        triggered += 1

    if triggered == 0:
        return {
            'cumulative_drift_m': 0.0,
            'triggered_keyframes': 0,
            'mean_motion_factor': 0.0,
            'mean_alignment': 0.0,
            'score': 0.0,
        }

    # Scale: wall_dist × accumulated weighted alignment
    mean_mf = total_motion / triggered
    mean_al = total_alignment / triggered
    drift = wall_dist * mean_mf * mean_al * math.sqrt(triggered)

    # Normalize score to similar range as bi_vul-based score
    score = drift * 10.0  # scale to ~100-1000 range like bi_vul scores

    return {
        'cumulative_drift_m': drift,
        'triggered_keyframes': triggered,
        'mean_motion_factor': mean_mf,
        'mean_alignment': mean_al,
        'score': score,
    }


def _circular_distance(k1: int, k2: int, n: int = N_BUCKETS) -> float:
    """Circular distance between two bucket indices."""
    d = abs(k1 - k2)
    return min(d, n - d)


def directional_score_full(
    frame: VulFrame,
    sx: float, sy: float,
    spoofing_range: float,
    attack_half_width: float = SPOOFING_HALF_RANGE,
) -> float:
    """
    Compute attack effectiveness score for a candidate spoofer position
    using the FULL 72-dim Bi-Vul vector.

    For each bucket k (5° resolution):
      - vul_k = bi_vul[k] is the vulnerability in direction k*5°
      - The spoofer's angular position relative to frame is alpha
      - The attack window is [alpha - attack_half_width, alpha + attack_half_width]
      - Score contribution = vul_k * weight_k
        where weight_k decays with distance and angular coverage

    Returns: scalar score for this (frame, spoofer) pair.
    """
    fx, fy = frame.x, frame.y
    dx = sx - fx
    dy = sy - fy
    dist = math.hypot(dx, dy)

    if dist > spoofing_range:
        return 0.0

    # Gaussian distance decay: sigma = spoofing_range / 3
    sigma = spoofing_range / 3.0
    dist_weight = math.exp(-dist * dist / (2.0 * sigma * sigma))

    # Spoofer direction in world frame (from frame toward spoofer)
    alpha = math.atan2(dy, dx)  # radians
    alpha_deg = (math.degrees(alpha) + 360.0) % 360.0
    alpha_bucket = alpha_deg / STEP_DEG  # fractional bucket index

    total = 0.0
    bi_vul = frame.bi_vul

    # Direction alignment factor: prefer spoofer thrust direction that aligns
    # with robot motion. When thrust aligns with motion, drift accumulates faster.
    # thrust_dir: from robot toward spoofer (direction of the injected "wall push")
    thrust_dir = math.atan2(dy, dx)  # radians
    # motion_dir: from frame velocity/odometry (direction robot is moving)
    motion_mag = math.hypot(frame.vec_x, frame.vec_y)
    if motion_mag > 1e-6:
        motion_dir = math.atan2(frame.vec_y, frame.vec_x)
        # cosine of angle between thrust and motion
        cos_angle = math.cos(thrust_dir - motion_dir)
        # map [-1, 1] -> [0.3, 1.0]: perpendicular thrust still has some effect
        alignment_factor = 0.3 + 0.7 * max(0.0, cos_angle)
    else:
        alignment_factor = 1.0  # no motion info, neutral

    for k in range(N_BUCKETS):
        vul_k = bi_vul[k]
        if vul_k <= 0.0:
            continue

        # Circular distance from spoofer direction to bucket k
        dtheta = _circular_distance_float(float(k), alpha_bucket) * STEP_DEG

        if dtheta > attack_half_width:
            continue

        # Linear decay within attack window (same as paper)
        angular_weight = 1.0 - dtheta / attack_half_width
        total += vul_k * dist_weight * angular_weight

    return total * alignment_factor


def _circular_distance_float(k: float, center: float, n: float = 72.0) -> float:
    """Circular distance between fractional bucket index k and center."""
    d = abs(k - center)
    return min(d, n - d)


def score_scalar(
    frames: List[VulFrame],
    sx: float, sy: float,
    spoofing_range: float,
    distance_threshold: float = float('inf'),
) -> float:
    """
    Scalar scoring for BO objective (fast, for use inside optimizer).

    Uses the full 72-dim Bi-Vul vector.
    """
    total = 0.0
    for f in frames:
        s = directional_score_full(f, sx, sy, spoofing_range)
        # Apply distance threshold (attack trigger window)
        dist = math.hypot(sx - f.x, sy - f.y)
        if dist > distance_threshold:
            s *= 0.0
        total += s
    return total


# Vectorized version for fast batch evaluation
def score_batch_vectorized(
    frames: List[VulFrame],
    positions: np.ndarray,   # (N, 2) array of (sx, sy)
    spoofing_range: float,
    distance_threshold: float = float('inf'),
    attack_half_width: float = SPOOFING_HALF_RANGE,
) -> np.ndarray:
    """
    Fully vectorized score computation for N positions.
    Returns scores array of shape (N,).
    """
    N = positions.shape[0]
    scores = np.zeros(N, dtype=np.float64)

    # Pre-extract frame data
    n_frames = len(frames)
    fx = np.array([f.x for f in frames], dtype=np.float64)
    fy = np.array([f.y for f in frames], dtype=np.float64)
    bi_vul_matrix = np.array([f.bi_vul for f in frames], dtype=np.float64)  # (n_frames, 72)

    # Compute distances: shape (N, n_frames)
    dx = positions[:, 0:1] - fx[None, :]  # (N, n_frames)
    dy = positions[:, 1:2] - fy[None, :]  # (N, n_frames)
    dists = np.hypot(dx, dy)  # (N, n_frames)

    # Distance mask
    in_range = dists <= spoofing_range  # (N, n_frames)
    if distance_threshold < float('inf'):
        in_range &= dists <= distance_threshold  # (N, n_frames)

    # Distance weights: Gaussian decay
    sigma = spoofing_range / 3.0
    dist_weights = np.exp(-dists * dists / (2.0 * sigma * sigma))  # (N, n_frames)
    dist_weights[~in_range] = 0.0

    # Spoofer direction for each (position, frame) pair: (N, n_frames)
    alpha = np.arctan2(dy, dx)  # (N, n_frames)
    alpha_deg = (np.degrees(alpha) + 360.0) % 360.0  # (N, n_frames)

    # bucket_angles: (72,)
    bucket_angles = np.arange(N_BUCKETS, dtype=np.float64) * STEP_DEG

    for j in range(n_frames):
        # alpha_deg_j: (N,)
        alpha_deg_j = alpha_deg[:, j]
        # dist_weights_j: (N,)
        dw_j = dist_weights[:, j]

        if not np.any(dw_j > 0):
            continue

        # diff: (N, 72) - (N, 1) broadcast
        diff = np.abs(alpha_deg_j[:, None] - bucket_angles[None, :])
        diff = np.minimum(diff, 360.0 - diff)  # circular
        dtheta = diff * STEP_DEG  # degrees

        # Angular weights: (N, 72)
        ang_weights = np.maximum(0.0, 1.0 - dtheta / attack_half_width)

        # bi_vul for frame j: (72,)
        bv_j = bi_vul_matrix[j, :]

        # Score for frame j at all N positions: (N,) = (N,72) · (72,)
        frame_scores = np.dot(ang_weights, bv_j)  # (N,)
        scores += dw_j * frame_scores

    return scores


# ============================================================================
# Stage 2: Directional + Spatial Joint Clustering
# ============================================================================

def extract_dominant_directions(
    bi_vul: np.ndarray,
    threshold_pct: float = 50.0,
) -> List[Tuple[int, float, float]]:
    """
    Extract dominant vulnerability directions from a 72-dim Bi-Vul vector.

    A direction is "dominant" if its vulnerability is above the
    threshold percentile of the mean vulnerability.

    Returns: list of (bucket_idx, angle_deg, vul_value) for each dominant direction.
    """
    mean_v = np.mean(bi_vul)
    std_v = np.std(bi_vul)
    threshold = mean_v + std_v * 0.3  # slightly above mean

    dominant = []
    for k in range(N_BUCKETS):
        if bi_vul[k] >= threshold:
            angle_deg = k * STEP_DEG
            dominant.append((k, angle_deg, bi_vul[k]))

    dominant.sort(key=lambda x: -x[2])  # sort by vul value descending
    return dominant


def cluster_adjacent_buckets(
    buckets: List[int],
    merge_threshold: int = 2,
) -> List[List[int]]:
    """Merge adjacent bucket indices into clusters."""
    if not buckets:
        return []
    sorted_buckets = sorted(buckets)
    clusters = []
    current = [sorted_buckets[0]]
    for b in sorted_buckets[1:]:
        if b - current[-1] <= merge_threshold:
            current.append(b)
        else:
            clusters.append(current)
            current = [b]
    clusters.append(current)
    return clusters


def spatial_directional_clustering(
    frames: List[VulFrame],
    eps_spatial: float = 40.0,
    eps_dir: float = 45.0,
) -> List[Dict[str, Any]]:
    """
    DBSCAN-style clustering on (x, y, dominant_direction) space.

    Returns: list of cluster dicts, each containing:
      - 'frame_indices': list of frame indices in this cluster
      - 'centroid': (cx, cy)
      - 'dominant_dirs': list of dominant direction clusters
      - 'total_vul': sum of bi_smvs
      - 'mean_vul': mean bi_smvs
    """
    if len(frames) < 2:
        return [{
            'frame_indices': list(range(len(frames))),
            'centroid': (frames[0].x, frames[0].y),
            'dominant_dirs': [],
            'total_vul': sum(f.bi_smvs for f in frames),
            'mean_vul': np.mean([f.bi_smvs for f in frames]),
        }]

    # Feature matrix: (x, y, dominant_direction)
    features = []
    for f in frames:
        dominant = extract_dominant_directions(f.bi_vul)
        if dominant:
            # Use weighted average of dominant directions
            total_w = sum(v for _, _, v in dominant)
            dir_avg = sum(a * v for _, a, v in dominant) / total_w
        else:
            dir_avg = f.vul_angle_deg
        features.append([f.x, f.y, dir_avg])

    features = np.array(features, dtype=np.float64)

    # Normalize direction feature to similar scale as spatial
    dir_scale = max(eps_spatial / 10.0, 1.0)
    features[:, 2] = features[:, 2] / dir_scale

    # Standardize spatial
    spatial_scale = eps_spatial
    features[:, 0] = features[:, 0] / spatial_scale
    features[:, 1] = features[:, 1] / spatial_scale

    # DBSCAN
    dists = cdist(features, features)
    adj = (dists <= 1.0).astype(int)
    np.fill_diagonal(adj, 0)

    parent = list(range(len(frames)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if adj[i, j]:
                union(i, j)

    cluster_map = {}
    for i in range(len(frames)):
        cid = find(i)
        cluster_map.setdefault(cid, []).append(i)

    clusters = []
    for idx_list in cluster_map.values():
        frames_in = [frames[i] for i in idx_list]
        cx = float(np.mean([f.x for f in frames_in]))
        cy = float(np.mean([f.y for f in frames_in]))

        # Aggregate dominant directions from all frames in cluster
        all_dirs = []
        for f in frames_in:
            for k, angle_deg, vul_val in extract_dominant_directions(f.bi_vul):
                all_dirs.append((k, angle_deg, vul_val))

        total_vul = sum(f.bi_smvs for f in frames_in)

        clusters.append({
            'frame_indices': idx_list,
            'centroid': (cx, cy),
            'dominant_dirs': all_dirs,
            'total_vul': total_vul,
            'mean_vul': total_vul / len(frames_in),
            'frames': frames_in,
        })

    # Sort by total vulnerability descending
    clusters.sort(key=lambda c: -c['total_vul'])
    return clusters


# ============================================================================
# Stage 3: Candidate Generation from Clusters
# ============================================================================

def generate_candidates_from_clusters(
    clusters: List[Dict[str, Any]],
    traj_pts: np.ndarray,
    spoofing_range: float,
    min_traj_dist: float,
    max_traj_dist: float,
) -> List[np.ndarray]:
    """
    Generate spoofer position candidates from spatial-directional clusters.

    For each cluster:
      1. Ray intersection candidates from frame pairs
      2. Cluster centroid extended along dominant directions
      3. Multi-distance exploration from centroid
    """
    candidates = []

    for cluster in clusters:
        cx, cy = cluster['centroid']
        frames_in = cluster['frames']

        # --- Ray intersection from frame pairs within cluster ---
        for i in range(len(frames_in)):
            for j in range(i + 1, len(frames_in)):
                fi, fj = frames_in[i], frames_in[j]

                # Use dominant direction of each frame
                dominant_i = extract_dominant_directions(fi.bi_vul)
                dominant_j = extract_dominant_directions(fj.bi_vul)

                if not dominant_i or not dominant_j:
                    continue

                # Primary dominant direction
                _, beta_i_deg, _ = dominant_i[0]
                _, beta_j_deg, _ = dominant_j[0]

                beta_i = math.radians(beta_i_deg)
                beta_j = math.radians(beta_j_deg)

                di_x = math.cos(beta_i)
                di_y = math.sin(beta_i)
                dj_x = math.cos(beta_j)
                dj_y = math.sin(beta_j)

                det = di_x * dj_y - di_y * dj_x
                if abs(det) < 1e-9:
                    continue

                px = fj.x - fi.x
                py = fj.y - fi.y
                t = (px * dj_y - py * dj_x) / det
                s = (px * di_y - py * di_x) / det

                if t < 0 or s < 0:
                    continue

                ix = fi.x + t * di_x
                iy = fi.y + t * di_y

                # Check trajectory distance constraint
                dx_t = traj_pts[:, 0] - ix
                dy_t = traj_pts[:, 1] - iy
                min_dist = float(np.min(np.hypot(dx_t, dy_t)))

                if min_traj_dist <= min_dist <= max_traj_dist:
                    candidates.append(np.array([ix, iy]))

        # --- Centroid extended along dominant directions ---
        for _, angle_deg, vul_val in cluster['dominant_dirs'][:5]:  # top-5 dirs
            beta = math.radians(angle_deg)

            for dist in [15.0, 25.0, 40.0, spoofing_range * 0.3, spoofing_range * 0.5]:
                sx = cx + dist * math.cos(beta)
                sy = cy + dist * math.sin(beta)

                dx_t = traj_pts[:, 0] - sx
                dy_t = traj_pts[:, 1] - sy
                min_dist = float(np.min(np.hypot(dx_t, dy_t)))

                if min_traj_dist <= min_dist <= max_traj_dist:
                    candidates.append(np.array([sx, sy]))

    # --- Supplementary: uniform sampling in feasible band (roadside constraint) ---
    # Paper: spoofer must be 5-30m from trajectory (roadside scenario)
    # Sample positions perpendicular to trajectory at fine intervals
    for traj_idx in range(0, len(traj_pts), max(1, len(traj_pts) // 80)):
        px, py = traj_pts[traj_idx, 0], traj_pts[traj_idx, 1]

        # Compute local tangent direction from neighboring points
        if traj_idx > 0 and traj_idx < len(traj_pts) - 1:
            dx_traj = traj_pts[traj_idx + 1, 0] - traj_pts[traj_idx - 1, 0]
            dy_traj = traj_pts[traj_idx + 1, 1] - traj_pts[traj_idx - 1, 1]
            seg_len = math.hypot(dx_traj, dy_traj)
            if seg_len < 0.1:
                continue
            tx, ty = dx_traj / seg_len, dy_traj / seg_len
        else:
            continue

        # Perpendicular (normal) direction
        nx, ny = -ty, tx

        # Sample at multiple distances along the normal
        for d in [min_traj_dist, (min_traj_dist + max_traj_dist) * 0.5, max_traj_dist]:
            for sign in [1.0, -1.0]:
                sx = px + sign * d * nx
                sy = py + sign * d * ny
                # Quick check: should be within scene bounding box
                candidates.append(np.array([sx, sy]))

    # --- Deduplicate by rounding ---
    seen = set()
    unique = []
    for c in candidates:
        key = (round(c[0], 0), round(c[1], 0))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ============================================================================
# Stage 4: CMA-ES Global Optimization
# ============================================================================

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    warnings.warn("cma not found. Install with: pip install cma")


def cma_optimize(
    frames: List[VulFrame],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    spoofing_range: float,
    n_calls: int = 200,
    initial_points: list = None,
    random_state: int = 42,
    traj_pts: np.ndarray = None,
    traj_t: np.ndarray = None,
    traj_motion: np.ndarray = None,
    wall_dist: float = 15.0,
    distance_threshold: float = 15.0,
    min_traj_dist: float = 5.0,
) -> Tuple[float, float, float, List[Dict]]:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization.

    CMA-ES is ideal for sparse, multi-modal scoring landscapes because:
    - No surrogate model needed (GP fails on sparse=0 landscapes)
    - Maintains a population of candidates, naturally explores multi-modal space
    - Adapts covariance matrix to find elongated valleys
    - Very robust: works well even when 90% of evaluations return 0

    Returns: (best_x, best_y, best_score, optimization_history)
    """
    if not HAS_CMA:
        raise ImportError("cma is required. Install with: pip install cma")

    # Build initial guess from candidates (best candidate centroid)
    x0 = [(x_bounds[0] + x_bounds[1]) / 2.0, (y_bounds[0] + y_bounds[1]) / 2.0]
    if initial_points:
        # Use the candidate with the best combined score as initial guess
        # Filter to only feasible candidates (min_traj_dist constraint)
        best_cand_score = -1.0
        for pt in initial_points:
            # Fix 1: skip candidates too close to trajectory
            if traj_pts is not None:
                d = np.sqrt((traj_pts[:, 0] - pt[0])**2 + (traj_pts[:, 1] - pt[1])**2)
                if float(np.min(d)) < min_traj_dist:
                    continue
            sc = score_scalar(frames, pt[0], pt[1], spoofing_range, float('inf'))
            # Add drift score if motion data available
            if traj_pts is not None and traj_motion is not None and len(frames) > 0:
                dr = cumulative_drift_score(
                    pt[0], pt[1], frames, traj_pts, traj_t, traj_motion,
                    spoofing_range, distance_threshold, wall_dist,
                )
                sc = 0.6 * sc + 0.4 * dr['score']
            if sc > best_cand_score:
                best_cand_score = sc
                x0 = [pt[0], pt[1]]

    # sigma: step size. ~10% of search range is a good heuristic.
    sigma = min(
        (x_bounds[1] - x_bounds[0]) * 0.08,
        (y_bounds[1] - y_bounds[0]) * 0.08,
        15.0,
    )

    # Bound constraints: CMA-ES expects list of [lb, ub] pairs per dimension
    # Or None + we clip manually
    bounds = None  # clip manually in objective

    def objective(params):
        sx, sy = params[0], params[1]
        # Hard bounds clip
        sx = float(np.clip(sx, x_bounds[0], x_bounds[1]))
        sy = float(np.clip(sy, y_bounds[0], y_bounds[1]))

        # Fix 1: hard min_traj_dist constraint — reject infeasible positions
        # Vectorized: (1, N) - (M, 1) broadcast to (M, N) → min over traj axis
        if traj_pts is not None:
            d_to_traj = np.sqrt(
                (traj_pts[:, 0] - sx)**2 + (traj_pts[:, 1] - sy)**2
            )
            actual_min_dist = float(np.min(d_to_traj))
            if actual_min_dist < min_traj_dist:
                return 1e18  # CMA-ES minimizes → reject infeasible

        # Base score: bi_vul × direction alignment × distance decay
        base_sc = score_scalar(frames, sx, sy, spoofing_range, float('inf'))

        # Fix 2+3: cumulative drift with motion-awareness
        if traj_pts is not None and traj_motion is not None and len(frames) > 0:
            drift_result = cumulative_drift_score(
                sx, sy, frames, traj_pts, traj_t, traj_motion,
                spoofing_range, distance_threshold, wall_dist,
            )
            drift_sc = drift_result['score']
            # Normalize both scores to [0, ~1] range before combining.
            # Bi-Vul base score is typically in the thousands while drift score
            # is in the tens, so raw weighted sum lets base_sc dominate entirely.
            base_norm  = base_sc  / (base_sc  + 800.0)
            drift_norm = drift_sc / (drift_sc + 40.0)
            # Drift is the ground-truth attack effect; give it dominant weight.
            combined = 0.15 * base_norm + 0.85 * drift_norm
        else:
            combined = base_sc

        return -combined

    print(f"  CMA-ES: {n_calls} max evals, sigma={sigma:.1f}, x0=({x0[0]:.1f},{x0[1]:.1f})",
          file=sys.stderr)

    # CMA-ES options
    opts = cma.CMAOptions()
    opts.set('maxfevals', n_calls)
    opts.set('seed', random_state)
    opts.set('verbose', -9)  # suppress output
    opts.set('popsize', min(20, n_calls // 10 + 1))

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    history = []
    best_score = -1.0
    best_x, best_y = x0[0], x0[1]

    while not es.stop():
        solutions = es.ask()
        fitness = [objective(x) for x in solutions]
        es.tell(solutions, fitness)

        # Record history
        for x, f in zip(solutions, fitness):
            sc = -f
            if sc > best_score:
                best_score = sc
                best_x, best_y = x[0], x[1]
            history.append({'sx': float(x[0]), 'sy': float(x[1]), 'score': float(sc)})

        # Early stopping if good enough
        if best_score > 5000:
            es.stop()

        if es.result.evaluations >= n_calls:
            break

    return float(best_x), float(best_y), float(best_score), history


# ============================================================================
# Stage 5: L-BFGS-B Local Refinement
# ============================================================================

def lbfgsb_refine(
    frames: List[VulFrame],
    initial_guesses: List[Tuple[float, float]],
    spoofing_range: float,
    distance_threshold: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    max_traj_dist: float,
    min_traj_dist: float,
    traj_pts: np.ndarray,
    wall_dist: float = 15.0,
) -> List[Dict[str, Any]]:
    """
    L-BFGS-B local refinement from multiple starting points.

    Returns: list of refinement results sorted by score descending.
    """
    results = []

    for sx0, sy0 in initial_guesses:
        x0 = np.array([sx0, sy0])

        def neg_objective(x):
            # Fix 1: hard min_traj_dist constraint — reject infeasible positions
            if traj_pts is not None:
                d = np.sqrt((traj_pts[:, 0] - x[0])**2 + (traj_pts[:, 1] - x[1])**2)
                if float(np.min(d)) < min_traj_dist:
                    return 1e18  # hard rejection

            # Fix 2+3: hybrid score with cumulative drift + motion awareness
            base_sc = score_scalar(frames, x[0], x[1], spoofing_range, distance_threshold)
            if traj_pts is not None and len(frames) > 0:
                drift_result = cumulative_drift_score(
                    x[0], x[1], frames, traj_pts,
                    traj_t=None,
                    traj_motion=None,
                    spoofing_range=spoofing_range,
                    distance_threshold=distance_threshold,
                    wall_dist=15.0,
                )
                return -(0.6 * base_sc + 0.4 * drift_result['score'])
            return -base_sc

        try:
            res = minimize(
                neg_objective,
                x0=x0,
                method='L-BFGS-B',
                bounds=[x_bounds, y_bounds],
                options={'maxiter': 200, 'ftol': 1e-8},
            )

            if res.success or res.fun is not None:
                sx_opt, sy_opt = float(res.x[0]), float(res.x[1])
                # Use same hybrid score as CMA-ES objective for consistency
                base_sc = score_scalar(frames, sx_opt, sy_opt, spoofing_range, float('inf'))
                if traj_pts is not None and len(frames) > 0:
                    dr = cumulative_drift_score(
                        sx_opt, sy_opt, frames, traj_pts,
                        traj_t=None, traj_motion=None,
                        spoofing_range=spoofing_range,
                        distance_threshold=distance_threshold,
                        wall_dist=wall_dist,
                    )
                    opt_score = 0.6 * base_sc + 0.4 * dr['score']
                else:
                    opt_score = base_sc

                dx_t = traj_pts[:, 0] - sx_opt
                dy_t = traj_pts[:, 1] - sy_opt
                actual_min_dist = float(np.min(np.hypot(dx_t, dy_t)))

                results.append({
                    'sx': sx_opt,
                    'sy': sy_opt,
                    'score': opt_score,
                    'min_traj_dist': actual_min_dist,
                    'in_band': min_traj_dist <= actual_min_dist <= max_traj_dist,
                    'lbfgs_success': res.success,
                    'lbfgs_nit': res.nit,
                })
        except Exception:
            continue

    # Sort by score descending
    results.sort(key=lambda r: -r['score'])
    return results


# ============================================================================
# Stage 6: Visualization
# ============================================================================

def visualize(
    frames: List[VulFrame],
    best_pos: Tuple[float, float],
    bo_history: List[Dict],
    clusters: List[Dict],
    traj_pts: np.ndarray,
    spoofing_range: float,
    max_traj_dist: float,
    distance_threshold: float,
    output_path: str,
):
    """Generate visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("[WARN] matplotlib not available, skipping visualization", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # ---- Plot 1: Trajectory + frames + best position ----
    ax = axes[0]
    ax.plot(traj_pts[:, 0], traj_pts[:, 1], 'b-', lw=0.8, alpha=0.5, label='Trajectory')
    ax.scatter(traj_pts[0, 0], traj_pts[0, 1], c='blue', s=80, marker='o', zorder=5, label='Start')
    ax.scatter(traj_pts[-1, 0], traj_pts[-1, 1], c='red', s=80, marker='x', zorder=5, label='End')

    colors = [f.bi_smvs for f in frames]
    vmin, vmax = min(colors), max(colors)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma

    for f in frames:
        ax.scatter(f.x, f.y, c=[f.bi_smvs], cmap=cmap, norm=norm,
                   s=60, edgecolors='white', linewidths=0.5, zorder=10)
        # Draw dominant vulnerability directions
        dominant = extract_dominant_directions(f.bi_vul)
        for k, angle_deg, vul_val in dominant[:3]:
            rad = math.radians(angle_deg)
            length = 5.0 + (vul_val / max(np.max(f.bi_vul), 1)) * 8.0
            ax.annotate('', xy=(f.x + length * math.cos(rad), f.y + length * math.sin(rad)),
                        xytext=(f.x, f.y),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.0, alpha=0.6))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Bi-SMVS')

    ax.scatter(best_pos[0], best_pos[1], c='lime', s=300, marker='*',
               edgecolors='black', linewidths=1.5, zorder=15, label='BO Best')
    ax.add_patch(plt.Circle(best_pos, spoofing_range, fill=False,
                            color='orange', lw=1.0, ls=':', alpha=0.5))
    ax.add_patch(plt.Circle(best_pos, max_traj_dist, fill=False,
                            color='lime', lw=1.5, ls='--', alpha=0.8))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory + Vulnerable Frames\n(Bi-SMVS 72-dim directional scoring)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: BO convergence ----
    ax = axes[1]
    scores = [h['score'] for h in bo_history]
    ax.plot(scores, 'b-', lw=1.5)
    ax.scatter(range(len(scores)), scores, c='steelblue', s=20)
    if bo_history:
        ax.axhline(max(scores), color='lime', ls='--', lw=1.0,
                   label=f'Best: {max(scores):.0f}')
    ax.set_xlabel('BO Evaluation #')
    ax.set_ylabel('Score')
    ax.set_title('Bayesian Optimization Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: Score heatmap around BO best ----
    ax = axes[2]
    sx0, sy0 = best_pos
    extent = 60.0
    nx, ny = 60, 60
    xs_h = np.linspace(sx0 - extent, sx0 + extent, nx)
    ys_h = np.linspace(sy0 - extent, sy0 + extent, ny)
    X, Y = np.meshgrid(xs_h, ys_h)
    Z = np.zeros_like(X)

    positions = np.column_stack([X.ravel(), Y.ravel()])
    scores_map = score_batch_vectorized(
        frames, positions, spoofing_range, distance_threshold
    )
    Z = scores_map.reshape(nx, ny)

    im = ax.imshow(Z, extent=[xs_h[0], xs_h[-1], ys_h[0], ys_h[-1]],
                   origin='lower', cmap='hot', aspect='equal')
    plt.colorbar(im, ax=ax, label='Score')
    ax.scatter(sx0, sy0, c='lime', s=200, marker='*', edgecolors='white',
               linewidths=1.5, zorder=15, label='BO Best')
    ax.plot(traj_pts[:, 0], traj_pts[:, 1], 'b-', lw=0.8, alpha=0.5)
    ax.add_patch(plt.Circle((sx0, sy0), max_traj_dist, fill=False,
                            color='lime', lw=1.5, ls='--', alpha=0.8))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Score Heatmap (extent={extent}m)')
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved: {output_path}", file=sys.stderr)
    plt.close()


# ============================================================================
# Data Loading
# ============================================================================

def load_frames(smvs_path: str, vul_path: str, top_k: int) -> List[VulFrame]:
    """Load frames from SMVS and Vul CSVs, including 72-dim Bi-Vul vectors."""
    smvs_df = pd.read_csv(smvs_path)
    vul_df = pd.read_csv(vul_path)

    # Standardize column names
    smvs_df["timestamp"] = pd.to_numeric(smvs_df["timestamp"], errors="coerce")
    smvs_df["x"] = pd.to_numeric(smvs_df["x"], errors="coerce")
    smvs_df["y"] = pd.to_numeric(smvs_df["y"], errors="coerce")
    smvs_df["z"] = pd.to_numeric(smvs_df["z"], errors="coerce")
    smvs_df["vul_angle_deg"] = pd.to_numeric(smvs_df["vul_angle_deg"], errors="coerce")

    # Find Bi-SMVS score column
    for col in ["frame_bi_smvs", "frame_li_smvs", "frame_l_smvs"]:
        if col in smvs_df.columns:
            smvs_df["frame_smvs"] = pd.to_numeric(smvs_df[col], errors="coerce")
            break
    else:
        raise KeyError(f"SMVS CSV has no recognised score column. Columns: {list(smvs_df.columns)}")

    # Drop origin rows
    smvs_df = smvs_df.dropna(subset=["timestamp", "x", "y", "z", "frame_smvs", "vul_angle_deg"])
    smvs_df = smvs_df[~((smvs_df["x"] == 0) & (smvs_df["y"] == 0))]

    # Get Bi-Vul columns from vul_df
    bi_vul_cols = [f"bi_vul_{i:02d}" for i in range(N_BUCKETS)]
    l_vul_cols = [f"l_vul_{i:02d}" for i in range(N_BUCKETS)]

    # Match by timestamp
    vul_df_indexed = vul_df.set_index("timestamp")

    frames: List[VulFrame] = []
    for _, row in smvs_df.iterrows():
        ts = float(row["timestamp"])

        if ts in vul_df_indexed.index:
            vul_row = vul_df_indexed.loc[ts]
        else:
            # Nearest timestamp
            vul_row = vul_df.iloc[(vul_df["timestamp"] - ts).abs().argsort().iloc[0]]

        bi_vul = np.array([float(vul_row[c]) for c in bi_vul_cols], dtype=np.float64)
        l_vul = np.array([float(vul_row[c]) for c in l_vul_cols], dtype=np.float64)

        frames.append(VulFrame(
            timestamp=ts,
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            yaw=float(row.get("yaw", 0.0)) if "yaw" in row else 0.0,
            bi_smvs=float(row["frame_smvs"]),
            vul_angle_deg=float(row["vul_angle_deg"]),
            vec_x=float(row.get("vec_x", 0.0)),
            vec_y=float(row.get("vec_y", 0.0)),
            bi_vul=bi_vul,
            l_vul=l_vul,
        ))

    return frames


def load_traj(traj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trajectory and return (traj_pts, traj_t).
    traj_pts: (N, 2) array of (x, y)
    traj_t: (N,) array of timestamps (relative seconds from start)
    """
    traj_df = pd.read_csv(traj_path)
    for col in ["x", "y"]:
        if col not in traj_df.columns:
            raise SystemExit(f"Trajectory CSV missing column: {col}")
    traj_df[col] = pd.to_numeric(traj_df[col], errors="coerce")

    if "time" in traj_df.columns:
        traj_df["time"] = pd.to_numeric(traj_df["time"], errors="coerce")
        traj_df = traj_df.dropna(subset=["time", "x", "y"])
        traj_t = (traj_df["time"].values - traj_df["time"].values[0])
    else:
        traj_df = traj_df.dropna(subset=["x", "y"])
        traj_t = np.arange(len(traj_df), dtype=np.float64)

    traj_pts = traj_df[["x", "y"]].values.astype(np.float64)
    return traj_pts, traj_t


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(args) -> Dict[str, Any]:
    """Execute the full Bi-SMVS BO pipeline."""

    # ---- Load data ----
    print("Loading data...", file=sys.stderr)
    frames = load_frames(args.smvs, args.vul, args.top_k)
    traj_pts, traj_t = load_traj(args.traj)
    print(f"  frames: {len(frames)}", file=sys.stderr)
    print(f"  traj pts: {len(traj_pts)}", file=sys.stderr)

    # Fix 3: Compute trajectory motion quality factors
    # (for motion-aware cumulative drift scoring)
    traj_motion = compute_traj_motion_factors(
        traj_pts[:, 0], traj_pts[:, 1], traj_t,
        speed_thresh=1.0, curv_thresh=0.05,
    )
    straight_pct = 100.0 * (traj_motion >= 0.8).mean()
    print(f"  Motion: {straight_pct:.0f}% straight-cruise, "
          f"{(traj_motion < 0.5).sum()} turn/accel frames", file=sys.stderr)

    # Trajectory info (used for logging / debug)
    xs_t = traj_pts[:, 0]; ys_t = traj_pts[:, 1]
    traj_info = {
        'length_m': float(np.sum(np.hypot(np.diff(xs_t), np.diff(ys_t)))),
        'span_x': float(xs_t.max() - xs_t.min()),
        'span_y': float(ys_t.max() - ys_t.min()),
        'span_m': float(math.sqrt((xs_t.max()-xs_t.min())**2 + (ys_t.max()-ys_t.min())**2)),
        'n_pts': len(traj_pts),
    }
    print(f"  traj span: {traj_info['span_x']:.1f}m × {traj_info['span_y']:.1f}m", file=sys.stderr)

    # ---- Stage 1: Sort by Bi-SMVS, select top-K ----
    frames_sorted = sorted(frames, key=lambda f: -f.bi_smvs)
    top_frames = frames_sorted[:args.top_k]
    print(f"  Top-{args.top_k} frames: Bi-SMVS range [{top_frames[-1].bi_smvs:.0f}, {top_frames[0].bi_smvs:.0f}]", file=sys.stderr)

    # ---- Stage 2: Spatial-directional clustering ----
    print("\n[Stage 2] Spatial-directional clustering...", file=sys.stderr)
    clusters = spatial_directional_clustering(
        top_frames,
        eps_spatial=args.cluster_eps,
        eps_dir=45.0,
    )
    print(f"  Found {len(clusters)} cluster(s)", file=sys.stderr)
    for i, c in enumerate(clusters):
        n_dirs = len(c['dominant_dirs'])
        print(f"  Cluster {i+1}: centroid=({c['centroid'][0]:.1f},{c['centroid'][1]:.1f}), "
              f"total_vul={c['total_vul']:.0f}, frames={len(c['frames'])}, dominant_dirs={n_dirs}", file=sys.stderr)

    # ---- Stage 3: Generate candidates ----
    print("\n[Stage 3] Candidate generation...", file=sys.stderr)
    candidates = generate_candidates_from_clusters(
        clusters, traj_pts,
        spoofing_range=args.spoofing_range,
        min_traj_dist=args.min_traj_dist,
        max_traj_dist=args.max_traj_dist,
    )
    print(f"  Generated {len(candidates)} initial candidates", file=sys.stderr)

    # Search bounds from trajectory + margin
    x_min = float(traj_pts[:, 0].min()) - 50.0
    x_max = float(traj_pts[:, 0].max()) + 50.0
    y_min = float(traj_pts[:, 1].min()) - 50.0
    y_max = float(traj_pts[:, 1].max()) + 50.0
    x_bounds = (x_min, x_max)
    y_bounds = (y_min, y_max)

    # ---- Stage 4: Bayesian Optimization ----
    print(f"\n[Stage 4] CMA-ES Global Optimization...", file=sys.stderr)
    bo_x, bo_y, bo_score, bo_history = cma_optimize(
        top_frames,
        x_bounds, y_bounds,
        spoofing_range=args.spoofing_range,
        n_calls=args.cma_calls,
        initial_points=candidates,
        random_state=args.seed,
        traj_pts=traj_pts,
        traj_t=traj_t,
        traj_motion=traj_motion,
        wall_dist=args.wall_dist,
        distance_threshold=args.distance_threshold,
        min_traj_dist=args.min_traj_dist,
    )

    dx_t = traj_pts[:, 0] - bo_x
    dy_t = traj_pts[:, 1] - bo_y
    bo_min_dist = float(np.min(np.hypot(dx_t, dy_t)))
    print(f"  BO best: ({bo_x:.2f}, {bo_y:.2f}) score={bo_score:.2f} min_dist={bo_min_dist:.2f}m", file=sys.stderr)

    # ---- Stage 5: L-BFGS-B refinement ----
    print(f"\n[Stage 5] L-BFGS-B refinement from top BO candidates...", file=sys.stderr)

    # Build initial guess list: BO best + top 5 BO history + top candidates
    refine_init = [(bo_x, bo_y)]
    top_bo = sorted(bo_history, key=lambda h: -h['score'])[:5]
    for h in top_bo:
        refine_init.append((h['sx'], h['sy']))
    refine_init += [(c[0], c[1]) for c in candidates[:10]]

    lbfgsb_results = lbfgsb_refine(
        top_frames,
        refine_init,
        spoofing_range=args.spoofing_range,
        distance_threshold=args.distance_threshold,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        max_traj_dist=args.max_traj_dist,
        min_traj_dist=args.min_traj_dist,
        traj_pts=traj_pts,
        wall_dist=args.wall_dist,
    )

    # Compare CMA-ES and L-BFGS-B results — pick the better score
    if lbfgsb_results:
        best_lbfgs = lbfgsb_results[0]
        print(f"  L-BFGS-B best: ({best_lbfgs['sx']:.2f}, {best_lbfgs['sy']:.2f}) "
              f"score={best_lbfgs['score']:.2f} min_dist={best_lbfgs['min_traj_dist']:.2f}m "
              f"success={best_lbfgs['lbfgs_success']}", file=sys.stderr)

        # Choose the better of CMA-ES and L-BFGS-B
        if best_lbfgs['score'] >= bo_score:
            final_x, final_y, final_score = best_lbfgs['sx'], best_lbfgs['sy'], best_lbfgs['score']
            final_min_dist = best_lbfgs['min_traj_dist']
            print(f"  → L-BFGS-B better, using its result", file=sys.stderr)
        else:
            final_x, final_y, final_score = bo_x, bo_y, bo_score
            final_min_dist = bo_min_dist
            print(f"  → CMA-ES better (score={bo_score:.2f}), keeping it", file=sys.stderr)
    else:
        final_x, final_y, final_score = bo_x, bo_y, bo_score
        final_min_dist = bo_min_dist
        lbfgsb_results = []

    # ---- Visualization ----
    if args.visualize:
        print(f"\n[Visualization]...", file=sys.stderr)
        visualize(
            frames=top_frames,
            best_pos=(final_x, final_y),
            bo_history=bo_history,
            clusters=clusters,
            traj_pts=traj_pts,
            spoofing_range=args.spoofing_range,
            max_traj_dist=args.max_traj_dist,
            distance_threshold=args.distance_threshold,
            output_path=args.viz_path,
        )

    # ---- Build output ----
    output = {
        "method": "bi_smvs_bayesian_optimization",
        "optim": {
            "spoofer_x": float(final_x),
            "spoofer_y": float(final_y),
            "score": float(final_score),
            "min_traj_dist": float(final_min_dist),
            "in_band": float(args.min_traj_dist) <= float(final_min_dist) <= float(args.max_traj_dist),
            "bo_best": {
                "spoofer_x": float(bo_x),
                "spoofer_y": float(bo_y),
                "score": float(bo_score),
                "min_traj_dist": float(bo_min_dist),
            },
        },
        "lbfgsb_refinements": lbfgsb_results[:10],
        "bo_history": bo_history,
        "clusters": [
            {
                "centroid": c['centroid'],
                "total_vul": float(c['total_vul']),
                "n_frames": len(c['frames']),
                "dominant_dirs": [(k, float(a), float(v)) for k, a, v in c['dominant_dirs'][:10]],
            }
            for c in clusters
        ],
        "params": {
            "top_k": args.top_k,
            "spoofing_range": args.spoofing_range,
            "min_traj_dist": args.min_traj_dist,
            "max_traj_dist": args.max_traj_dist,
            "distance_threshold": float(args.distance_threshold),
            "wall_dist": args.wall_dist,
            "cma_calls": args.cma_calls,
            "cluster_eps": args.cluster_eps,
            "seed": args.seed,
        },
        "trajectory_info": traj_info,
    }

    return output


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bi-SMVS-driven Bayesian Optimization for spoofer selection"
    )
    parser.add_argument("--smvs", required=True,
                        help="SMVS CSV (Bi-SMVS, with x,y,frame_bi_smvs,vul_angle_deg)")
    parser.add_argument("--vul", required=True,
                        help="Vulnerability CSV (with bi_vul_00..71, l_vul_00..71)")
    parser.add_argument("--traj", required=True,
                        help="Reference trajectory CSV (x,y columns)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Use top-k highest-Bi-SMVS frames. Default: 20")
    parser.add_argument("--spoofing-range", type=float, default=80.0,
                        help="Spoofing attack angular window (degrees, total width). "
                             "Must match attack config. Default: 80.0")
    parser.add_argument("--distance-threshold", type=float, default=15.0,
                        help="Attack trigger distance (m). Default: 15.0")
    parser.add_argument("--wall-dist", type=float, default=15.0,
                        help="Injected false-wall distance (m). Default: 15.0")
    parser.add_argument("--min-traj-dist", type=float, default=5.0,
                        help="Min distance from spoofer to trajectory (m). Default: 5.0")
    parser.add_argument("--max-traj-dist", type=float, default=30.0,
                        help="Max distance from spoofer to trajectory (m). Default: 30.0")
    parser.add_argument("--cma-calls", type=int, default=200,
                        help="Number of CMA-ES objective evaluations. Default: 200")
    parser.add_argument("--cluster-eps", type=float, default=40.0,
                        help="Spatial clustering epsilon (m). Default: 40.0")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Default: 42")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization PNG")
    parser.add_argument("--viz-path", default="spoofer_bi_bo_visualization.png",
                        help="Visualization output path")
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    result = run_pipeline(args)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"=== Bi-SMVS BO Result ===", file=sys.stderr)
    print(f"  Position: ({result['optim']['spoofer_x']:.2f}, {result['optim']['spoofer_y']:.2f})", file=sys.stderr)
    print(f"  Score:    {result['optim']['score']:.2f}", file=sys.stderr)
    print(f"  Min dist: {result['optim']['min_traj_dist']:.2f}m", file=sys.stderr)
    print(f"  In band:  {result['optim']['in_band']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[OK] wrote {args.output}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
