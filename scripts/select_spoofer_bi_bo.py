#!/usr/bin/env python3
"""
select_spoofer_bi_bo.py
========================

Bi-SMVS-driven CMA-ES Optimization for spoofer location selection.

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

4. STRUCTURAL ISOLATION + DOMINANCE (attack persistence)
   After reachability, two structural factors determine whether injected bias persists:
     - Isolation: how few competing constraints (loop, visual, IMU) surround affected nodes
     - Dominance: whether LiDAR constraints dominate over competing constraints
   Combined score = 0.35·reach + 0.25·isolation + 0.25·dominance + 0.15·bivul_gate
   (bivul_gate is binary: if 0, score drops to 0, acting as a feasibility gate)

Usage:
    python3 select_spoofer_bi_bo.py \
        --smvs ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_kitti/smvs/bismvs/06_08_14_25_15.csv \
        --vul  ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_kitti/vul/bismvs/vul_06_08_14_25_15.csv \
        --traj ~/catkin_ws/src/LVI-SAM/datasets/slamspoof_kitti/original/kitti_original_traj.csv \
        --top-k 20 \
        --spoofing-range 80.0 \
        --distance-threshold 30.0 \
        --visualize

"""

import argparse
import glob
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ============================================================================
# Graph Data Cache  (populated once, reused across all CMA evaluations)
# ============================================================================
_graph_cache: Dict[str, Any] = {}


def _precompute_graph_data(dump_dir: str) -> Dict[str, Any]:
    """Load all graph dumps ONCE and return compact lookup structures.

    Returns a dict with:
      - node_arr : (N, 3)  [node_id, x, y]
      - node_nid : (N,)    node IDs
      - factor_nids : list of sets, one per factor, containing node IDs
      - factor_source : list of "odometry"|"loop_closure"|"prior"|"other"
      - factor_info  : list of float (info weight = sum(1/noise_i))
    """
    import time

    cache_key = dump_dir
    if cache_key in _graph_cache:
        return _graph_cache[cache_key]

    t0 = time.time()

    dump_files = sorted(
        glob.glob(os.path.join(dump_dir, "dump_*.json")),
        key=lambda p: int(re.search(r'dump_(\d+)\.json', p).group(1)),
    )

    if not dump_files:
        _graph_cache[cache_key] = None
        return None

    # FIX: Use the LAST dump which contains the FULL cumulative graph.
    # After the C++ hook fix, each dump uses isam->getFactorsUnsafe() which returns
    # the complete accumulated graph (all odometry + loop closure + GPS factors).
    # No need to merge across dumps — the last dump has everything.
    last_path = dump_files[-1]
    with open(last_path) as fh:
        last_d = json.load(fh)

    nodes = last_d.get("nodes", [])
    node_nids = np.array([int(n["id"]) for n in nodes], dtype=np.int32)
    node_arr = np.stack([node_nids.astype(float),
                          np.array([n.get("x", 0.0) for n in nodes]),
                          np.array([n.get("y", 0.0) for n in nodes])], axis=1)

    # factor_nids[i] = set of node IDs involved in factor i
    # factor_source[i] = source string
    # factor_info[i]  = information weight
    factor_nids: List[set] = []
    factor_source: List[str] = []
    factor_info: List[float] = []

    for fac in last_d.get("factors", []):
        keys = []
        for k in fac.get("keys", []):
            m = re.search(r'X(\d+)', k)
            if m:
                keys.append(int(m.group(1)))
        if not keys:
            continue
        src = fac.get("source", "other")
        noise = fac.get("noise", [])
        info = sum(1.0 / max(float(s), 1e-12) for s in noise) if noise else 0.0
        factor_nids.append(set(keys))
        factor_source.append(src)
        factor_info.append(info)

    result = dict(
        node_arr=node_arr,
        node_nids=node_nids,
        factor_nids=factor_nids,
        factor_source=factor_source,
        factor_info=factor_info,
    )
    _graph_cache[cache_key] = result

    # Print factor type breakdown for debugging
    from collections import Counter
    src_counts = Counter(factor_source)
    print(f"[GraphCache] loaded last dump ({len(dump_files)} total available), "
          f"{len(node_nids)} nodes, {len(factor_nids)} factors "
          f"({dict(src_counts)}) in {time.time()-t0:.1f}s", file=sys.stderr)
    return result


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


def _gaussian_weight(dist: float, sigma: float) -> float:
    """Gaussian decay weight for a given distance."""
    return math.exp(-dist * dist / (2.0 * sigma * sigma))


def compute_reach(
    sx: float, sy: float,
    frames: List['VulFrame'],
    distance_threshold: float,
) -> float:
    """
    Geometric reachability: how many trajectory points (represented by vulnerable
    frames) are within the spoofer's attack range, weighted by proximity.

    reach(S) = Σ_{f in frames, ||S - f|| < distance_threshold} Gaussian(dist)

    This replaces the binary "in range / out of range" check with a smooth
    decay so that spoofer positions near many frames score higher even if no
    single frame is extremely close.

    Returns value in [0, 1] (normalized by n_frames).
    """
    if distance_threshold <= 0:
        return 0.0
    sigma = float(distance_threshold) / 3.0
    reach = 0.0
    for f in frames:
        dist = math.hypot(sx - f.x, sy - f.y)
        if dist < distance_threshold:
            reach += _gaussian_weight(dist, sigma)
    n = len(frames)
    if n == 0:
        return 0.0
    return min(reach / n, 1.0)


def compute_bivul_gate(
    sx: float, sy: float,
    frames: List['VulFrame'],
    distance_threshold: float,
    gate_threshold: float = 0.15,
    half_angle: float = SPOOFING_HALF_RANGE,
) -> float:
    """
    Bi-Vul direction score: for the spoofer position S = (sx, sy), sample the
    72-dim directional vulnerability vector of each reachable frame along the
    S -> frame direction, and return the best normalized score in [0, 1].

    The score is used as a continuous quality factor in the weighted scoring
    formula. Higher values mean the spoofer faces a more vulnerable direction
    toward at least one triggered frame.
    """
    sigma = float(distance_threshold) / 3.0 if distance_threshold > 0 else 1.0
    n_buckets = float(N_BUCKETS)
    best_gate = 0.0

    for f in frames:
        dx = sx - f.x
        dy = sy - f.y
        dist = math.hypot(dx, dy)
        if dist > distance_threshold:
            continue

        alpha = math.atan2(dy, dx)
        alpha_deg = (math.degrees(alpha) + 180.0) % 360.0
        alpha_bucket = alpha_deg / STEP_DEG

        vals = []
        for k in range(N_BUCKETS):
            vk = f.bi_vul[k]
            if vk <= 0:
                continue
            dtheta = abs(k - alpha_bucket)
            dtheta = min(dtheta, n_buckets - dtheta) * STEP_DEG
            if dtheta <= half_angle:
                vals.append(vk * (1.0 - dtheta / half_angle))

        if vals:
            mean_vul = sum(vals) / len(vals)
            normalized = min(mean_vul / 50.0, 1.0)
            best_gate = max(best_gate, normalized)

    return best_gate


def compute_isolation(
    sx: float, sy: float,
    frames: List['VulFrame'],
    traj_pts: np.ndarray,
    graph_dump_dir: Optional[str] = None,
    distance_threshold: float = 15.0,
    loop_closure_radius: float = 50.0,
    loop_time_gap: float = 5.0,
) -> float:
    """
    Structural isolation: how isolated are the affected nodes in the factor graph?

    High isolation -> the attacked region has few competing constraints
    (few loop closures, visual factors, IMU factors) -> the injected bias
    persists in the optimization.

    Strategy:
      1. Find all frames within distance_threshold of S (affected frames).
      2. Map frames to nearest factor-graph nodes (using pre-cached data).
      3. Count competing constraint edges (non-odometry) touching each affected node.
      4. isolation = 1 / (1 + avg_competing_edges_per_node)

    Falls back to a distance-only heuristic when no dump files are available:
      isolation = 1 - 0.5 * (n_affected / n_total)
    """
    affected = []
    for f in frames:
        dist = math.hypot(sx - f.x, sy - f.y)
        if dist < distance_threshold:
            affected.append((f, dist))

    if not affected:
        return 0.0

    n_affected = len(affected)

    gdata = _precompute_graph_data(graph_dump_dir) if graph_dump_dir else None
    if gdata is None:
        total_frames = len(frames)
        if total_frames == 0:
            return 0.0
        frac = n_affected / total_frames
        return float(np.clip(1.0 - 0.5 * frac, 0.0, 1.0))

    node_arr = gdata["node_arr"]
    node_nids = gdata["node_nids"]
    factor_nids = gdata["factor_nids"]
    factor_source = gdata["factor_source"]

    affected_nid_set = set()
    for f, _ in affected:
        dists = np.hypot(node_arr[:, 1] - f.x, node_arr[:, 2] - f.y)
        nearest_idx = int(np.argmin(dists))
        affected_nid_set.add(int(node_nids[nearest_idx]))

    competing_counts: Dict[int, int] = {nid: 0 for nid in affected_nid_set}
    for f_nids, f_src in zip(factor_nids, factor_source):
        if f_src == "odometry":
            continue
        for nid in f_nids:
            if nid in competing_counts:
                competing_counts[nid] += 1

    avg_competing = float(np.mean(list(competing_counts.values()))) if competing_counts else 0.0
    isolation = 1.0 / (1.0 + avg_competing)
    return float(np.clip(isolation, 0.0, 1.0))


def compute_dominance(
    sx: float, sy: float,
    frames: List['VulFrame'],
    traj_pts: np.ndarray,
    graph_dump_dir: Optional[str] = None,
    distance_threshold: float = 15.0,
) -> float:
    """
    Constraint dominance: do LiDAR constraints dominate over competing
    constraints (loop, visual, IMU) at the affected nodes?

    dominance(S) = mean over affected nodes of:
        LiDAR_info / (LiDAR_info + loop_info + visual_info + imu_info + eps)

    where info = sum of 1/noise_i for each dimension (trace of information matrix).

    Falls back to 0.5 (neutral) when no dump files are available.
    """
    affected = []
    for f in frames:
        dist = math.hypot(sx - f.x, sy - f.y)
        if dist < distance_threshold:
            affected.append(f)

    if not affected:
        return 0.0

    gdata = _precompute_graph_data(graph_dump_dir) if graph_dump_dir else None
    if gdata is None:
        return 0.5

    node_arr = gdata["node_arr"]
    node_nids = gdata["node_nids"]
    factor_nids = gdata["factor_nids"]
    factor_source = gdata["factor_source"]
    factor_info = gdata["factor_info"]

    affected_nid_set = set()
    for f in affected:
        dists = np.hypot(node_arr[:, 1] - f.x, node_arr[:, 2] - f.y)
        nearest_idx = int(np.argmin(dists))
        affected_nid_set.add(int(node_nids[nearest_idx]))

    lidar_info: Dict[int, float] = {nid: 0.0 for nid in affected_nid_set}
    competing_info: Dict[int, float] = {nid: 0.0 for nid in affected_nid_set}

    for f_nids, f_src, f_info in zip(factor_nids, factor_source, factor_info):
        for nid in f_nids:
            if nid in lidar_info:
                lidar_info[nid] += f_info
                if f_src != "odometry":
                    competing_info[nid] += f_info

    doms = []
    for nid in affected_nid_set:
        l_i = lidar_info.get(nid, 0.0)
        c_i = competing_info.get(nid, 0.0)
        total = l_i + c_i
        if total > 1e-12:
            doms.append(l_i / total)
        else:
            doms.append(0.5)

    if doms:
        return float(np.clip(np.mean(doms), 0.0, 1.0))
    return 0.5


def new_score_formula(
    sx: float, sy: float,
    frames: List['VulFrame'],
    traj_pts: np.ndarray,
    distance_threshold: float,
    graph_dump_dir: Optional[str] = None,
    bivul_gate_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    4-factor scoring formula for spoofer position S = (sx, sy):

        score(S) = 0.35·reach(S) + 0.25·isolation(S) +
                   0.25·dominance(S) + 0.15·bivul_gate(S)

    bivul_gate is the continuous normalized vulnerability direction score
    in [0, 1], NOT a binary gate. It contributes as a quality factor:
    higher values mean the spoofer sits in a more vulnerable direction.
    The gate_threshold parameter is retained for API compatibility but is
    no longer used as a hard cutoff.

    where:
      - reach(S)       : geometric reachability (how many frames are in range)
      - isolation(S)   : structural isolation of affected nodes in factor graph
      - dominance(S)   : LiDAR constraint dominance over competing constraints
      - bivul_gate(S)  : continuous vulnerability direction score in [0, 1]
    """
    reach = compute_reach(sx, sy, frames, distance_threshold)
    bivul = compute_bivul_gate(sx, sy, frames, distance_threshold, bivul_gate_threshold)
    isolation = compute_isolation(sx, sy, frames, traj_pts, graph_dump_dir, distance_threshold)
    dominance = compute_dominance(sx, sy, frames, traj_pts, graph_dump_dir, distance_threshold)

    combined = (
        0.35 * reach +
        0.25 * isolation +
        0.25 * dominance +
        0.15 * bivul
    )

    return {
        'score': float(combined),
        'reach': float(reach),
        'isolation': float(isolation),
        'dominance': float(dominance),
        'bivul_gate': float(bivul),
    }


# ============================================================================


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

            for dist in [15.0, 25.0, 40.0, max_traj_dist * 0.5, max_traj_dist * 0.7]:
                sx = cx + dist * math.cos(beta)
                sy = cy + dist * math.sin(beta)

                dx_t = traj_pts[:, 0] - sx
                dy_t = traj_pts[:, 1] - sy
                min_dist = float(np.min(np.hypot(dx_t, dy_t)))

                if min_traj_dist <= min_dist <= max_traj_dist:
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
    n_calls: int = 200,
    initial_points: list = None,
    random_state: int = 42,
    traj_pts: np.ndarray = None,
    distance_threshold: float = 15.0,
    min_traj_dist: float = 5.0,
    graph_dump_dir: Optional[str] = None,
) -> Tuple[float, float, float, List[Dict]]:
    """
    CMA-ES optimization using the 4-factor formula:

        score(S) = 0.35·reach(S) + 0.25·isolation(S) +
                   0.25·dominance(S) + 0.15·bivul_gate(S)

    bivul_gate is the continuous normalized vulnerability direction score in
    [0, 1], used as a quality factor in the weighted sum (not a binary gate).
    """
    if not HAS_CMA:
        raise ImportError("cma is required. Install with: pip install cma")

    # Build initial guess from candidates
    x0 = [(x_bounds[0] + x_bounds[1]) / 2.0, (y_bounds[0] + y_bounds[1]) / 2.0]
    best_cand_x, best_cand_y, best_cand_score = x0[0], x0[1], -1.0
    if initial_points:
        for pt in initial_points:
            if traj_pts is not None:
                d = np.sqrt((traj_pts[:, 0] - pt[0])**2 + (traj_pts[:, 1] - pt[1])**2)
                if float(np.min(d)) < min_traj_dist:
                    continue
            score = new_score_formula(
                pt[0], pt[1], frames, traj_pts, distance_threshold,
                graph_dump_dir=graph_dump_dir,
            )
            if sc := score['score']:
                if sc > best_cand_score:
                    best_cand_score = sc
                    best_cand_x = pt[0]
                    best_cand_y = pt[1]
                    x0 = [pt[0], pt[1]]

    sigma = min(
        (x_bounds[1] - x_bounds[0]) * 0.08,
        (y_bounds[1] - y_bounds[0]) * 0.08,
        15.0,
    )

    def objective(params):
        sx, sy = params[0], params[1]
        sx = float(np.clip(sx, x_bounds[0], x_bounds[1]))
        sy = float(np.clip(sy, y_bounds[0], y_bounds[1]))

        if traj_pts is not None:
            d_to_traj = np.sqrt(
                (traj_pts[:, 0] - sx)**2 + (traj_pts[:, 1] - sy)**2
            )
            if float(np.min(d_to_traj)) < min_traj_dist:
                return 1e18

        new_res = new_score_formula(
            sx, sy, frames, traj_pts, distance_threshold,
            graph_dump_dir=graph_dump_dir,
        )
        return -new_res['score']

    print(f"  CMA-ES: {n_calls} max evals, sigma={sigma:.1f}, x0=({x0[0]:.1f},{x0[1]:.1f})",
          file=sys.stderr)

    opts = cma.CMAOptions()
    opts.set('maxfevals', n_calls)
    opts.set('seed', random_state)
    opts.set('verbose', -9)
    opts.set('popsize', min(20, n_calls // 10 + 1))

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    history = []
    best_score = -1.0
    best_x, best_y = x0[0], x0[1]

    while not es.stop():
        solutions = es.ask()
        fitness = [objective(x) for x in solutions]
        es.tell(solutions, fitness)

        for x, f in zip(solutions, fitness):
            sc = -f
            if sc > best_score:
                best_score = sc
                best_x, best_y = x[0], x[1]
            history.append({'sx': float(x[0]), 'sy': float(x[1]), 'score': float(sc)})

        if best_score > 0.995:
            es.stop()
        if es.result.evaluations >= n_calls:
            break

    return float(best_x), float(best_y), float(best_score), history


def visualize(
    frames: List[VulFrame],
    best_pos: Tuple[float, float],
    bo_history: List[Dict],
    clusters: List[Dict],
    traj_pts: np.ndarray,
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
    ax.add_patch(plt.Circle(best_pos, distance_threshold, fill=False,
                            color='orange', lw=1.0, ls=':', alpha=0.5,
                            label=f'distance_threshold={distance_threshold}m'))
    ax.add_patch(plt.Circle(best_pos, max_traj_dist, fill=False,
                            color='lime', lw=1.5, ls='--', alpha=0.8,
                            label=f'max_traj_dist={max_traj_dist}m'))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory + Vulnerable Frames\n(Bi-SMVS 72-dim directional scoring)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: CMA-ES convergence ----
    ax = axes[1]
    raw_scores = [h['score'] for h in bo_history]
    real_scores = [s for s in raw_scores if s > -1e15]
    real_indices = [i for i, s in enumerate(raw_scores) if s > -1e15]

    if real_scores:
        ax.scatter(real_indices, real_scores, c='steelblue', s=20, zorder=5)
        ax.plot(real_indices, real_scores, 'b-', lw=1.0, alpha=0.5)
        running_best = np.maximum.accumulate(real_scores)
        ax.plot(real_indices, running_best, 'lime', lw=2.0, zorder=6,
                label=f'Best so far: {running_best[-1]:.4f}')
        ax.legend(loc='lower right', fontsize=9)
    else:
        ax.set_ylim(0, 1)

    n_invalid = len(raw_scores) - len(real_scores)
    ax.set_xlabel('CMA-ES Evaluation #')
    ax.set_ylabel('Score')
    ax.set_title(f'CMA-ES Convergence  ({n_invalid} invalid/penalized)')
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: Score heatmap around BO best (new formula) ----
    ax = axes[2]
    sx0, sy0 = best_pos
    extent = float(max_traj_dist)
    nx, ny = 40, 40
    xs_h = np.linspace(sx0 - extent, sx0 + extent, nx)
    ys_h = np.linspace(sy0 - extent, sy0 + extent, ny)
    Z = np.zeros((ny, nx))

    # Evaluate new formula on grid
    for iy in range(ny):
        for ix in range(nx):
            # min_traj_dist constraint
            d_min = float(np.min(np.hypot(traj_pts[:, 0] - xs_h[ix], traj_pts[:, 1] - ys_h[iy])))
            if d_min < 5.0:  # min_traj_dist heuristic
                Z[iy, ix] = 0.0
            else:
                try:
                    res = new_score_formula(xs_h[ix], ys_h[iy], frames,
                                            traj_pts, distance_threshold,
                                            graph_dump_dir=None)
                    Z[iy, ix] = res['score']
                except Exception:
                    Z[iy, ix] = 0.0

    im = ax.imshow(Z, extent=[xs_h[0], xs_h[-1], ys_h[0], ys_h[-1]],
                   origin='lower', cmap='hot', aspect='equal')
    plt.colorbar(im, ax=ax, label='Score (weighted sum)')
    ax.scatter(sx0, sy0, c='lime', s=200, marker='*', edgecolors='white',
               linewidths=1.5, zorder=15, label='CMA-ES Best')
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
    """Execute the spoofer selection pipeline."""
    print("Loading data...", file=sys.stderr)
    frames = load_frames(args.smvs, args.vul, args.top_k)
    traj_pts, traj_t = load_traj(args.traj)
    print(f"  frames: {len(frames)}", file=sys.stderr)
    print(f"  traj pts: {len(traj_pts)}", file=sys.stderr)

    xs_t = traj_pts[:, 0]; ys_t = traj_pts[:, 1]
    traj_info = {
        'length_m': float(np.sum(np.hypot(np.diff(xs_t), np.diff(ys_t)))),
        'span_x': float(xs_t.max() - xs_t.min()),
        'span_y': float(ys_t.max() - ys_t.min()),
        'span_m': float(math.sqrt((xs_t.max()-xs_t.min())**2 + (ys_t.max()-ys_t.min())**2)),
        'n_pts': len(traj_pts),
    }
    print(f"  traj span: {traj_info['span_x']:.1f}m x {traj_info['span_y']:.1f}m", file=sys.stderr)

    # ---- Select top-K frames by Bi-SMVS ----
    frames_sorted = sorted(frames, key=lambda f: -f.bi_smvs)
    top_frames = frames_sorted[:args.top_k]
    print(f"  Top-{args.top_k} frames: Bi-SMVS range [{top_frames[-1].bi_smvs:.0f}, {top_frames[0].bi_smvs:.0f}]", file=sys.stderr)

    # ---- Spatial-directional clustering ----
    print("\n[Clustering] Spatial-directional clustering...", file=sys.stderr)
    clusters = spatial_directional_clustering(
        top_frames,
        eps_spatial=args.cluster_eps,
        eps_dir=45.0,
    )
    print(f"  Found {len(clusters)} cluster(s)", file=sys.stderr)
    for i_c, c in enumerate(clusters):
        n_dirs = len(c['dominant_dirs'])
        print(f"  Cluster {i_c+1}: centroid=({c['centroid'][0]:.1f},{c['centroid'][1]:.1f}), "
              f"total_vul={c['total_vul']:.0f}, frames={len(c['frames'])}, dominant_dirs={n_dirs}", file=sys.stderr)

    # ---- Generate candidates ----
    print("\n[Candidates] Generating from clusters...", file=sys.stderr)
    candidates = generate_candidates_from_clusters(
        clusters, traj_pts,
        spoofing_range=args.spoofing_range,
        min_traj_dist=args.min_traj_dist,
        max_traj_dist=args.max_traj_dist,
    )
    print(f"  Generated {len(candidates)} initial candidates", file=sys.stderr)

    pad = float(args.max_traj_dist)
    x_min = float(traj_pts[:, 0].min()) - pad
    x_max = float(traj_pts[:, 0].max()) + pad
    y_min = float(traj_pts[:, 1].min()) - pad
    y_max = float(traj_pts[:, 1].max()) + pad
    x_bounds = (x_min, x_max)
    y_bounds = (y_min, y_max)

    # ---- CMA-ES Optimization ----
    print(f"\n[CMA-ES] Running optimization...", file=sys.stderr)
    bo_x, bo_y, bo_score, bo_history = cma_optimize(
        top_frames,
        x_bounds, y_bounds,
        n_calls=args.cma_calls,
        initial_points=candidates,
        random_state=args.seed,
        traj_pts=traj_pts,
        distance_threshold=args.distance_threshold,
        min_traj_dist=args.min_traj_dist,
        graph_dump_dir=args.graph_dump_dir,
    )

    score_components = {}
    try:
        score_components = new_score_formula(
            bo_x, bo_y, top_frames, traj_pts, args.distance_threshold,
            graph_dump_dir=args.graph_dump_dir,
            bivul_gate_threshold=args.bivul_gate_threshold,
        )
    except Exception as exc:
        print(f"[WARN] new_score_formula failed on final pos: {exc}", file=sys.stderr)

    dx_t = traj_pts[:, 0] - bo_x
    dy_t = traj_pts[:, 1] - bo_y
    bo_min_dist = float(np.min(np.hypot(dx_t, dy_t)))
    print(f"  Best: ({bo_x:.2f}, {bo_y:.2f}) score={bo_score:.4f} min_dist={bo_min_dist:.2f}m", file=sys.stderr)

    if args.visualize:
        print(f"\n[Visualization]...", file=sys.stderr)
        visualize(
            frames=top_frames,
            best_pos=(bo_x, bo_y),
            bo_history=bo_history,
            clusters=clusters,
            traj_pts=traj_pts,
            max_traj_dist=args.max_traj_dist,
            distance_threshold=args.distance_threshold,
            output_path=args.viz_path,
        )

    output = {
        "method": "bi_smvs_cma_optimization",
        "optim": {
            "spoofer_x": float(bo_x),
            "spoofer_y": float(bo_y),
            "score": float(bo_score),
            "min_traj_dist": float(bo_min_dist),
            "in_band": float(args.min_traj_dist) <= float(bo_min_dist) <= float(args.max_traj_dist),
        },
        "cma_history": bo_history,
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
            "cma_calls": args.cma_calls,
            "cluster_eps": args.cluster_eps,
            "seed": args.seed,
        },
        "trajectory_info": traj_info,
        "score_components": score_components,
    }
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bi-SMVS-driven CMA-ES Optimization for spoofer selection"
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
                        help="Spoofing attack angular window (degrees, total width). Default: 80.0")
    parser.add_argument("--distance-threshold", type=float, default=15.0,
                        help="Attack trigger distance (m). Default: 15.0")
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
    parser.add_argument("--graph-dump-dir",
                        help="Path to pre-dumped LVI-SAM factor graph JSON files "
                             "(from $LVI_GRAPH_DUMP_DIR). Used for isolation and dominance computation.")
    parser.add_argument("--bivul-gate-threshold", type=float, default=0.15,
                        help="Minimum normalized bi_vul in attack window for bi-vul gate. Default: 0.15")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization PNG")
    parser.add_argument("--viz-path", default="spoofer_bo_visualization.png",
                        help="Visualization output path")
    return parser.parse_args()


def main():
    args = parse_args()

    result = run_pipeline(args)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"=== Bi-SMVS CMA-ES Result ===", file=sys.stderr)
    print(f"  Position: ({result['optim']['spoofer_x']:.2f}, {result['optim']['spoofer_y']:.2f})", file=sys.stderr)
    print(f"  Score:    {result['optim']['score']:.2f}", file=sys.stderr)
    print(f"  Min dist: {result['optim']['min_traj_dist']:.2f}m", file=sys.stderr)
    print(f"  In band:  {result['optim']['in_band']}", file=sys.stderr)

    sc = result.get('score_components', {})
    print(f"  Reach:    {sc.get('reach', 0):.4f}", file=sys.stderr)
    print(f"  Isolation:{sc.get('isolation', 0):.4f}", file=sys.stderr)
    print(f"  Dominance:{sc.get('dominance', 0):.4f}", file=sys.stderr)
    print(f"  Bi-Vul:   {sc.get('bivul_gate', 0):.4f}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[OK] wrote {args.output}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
