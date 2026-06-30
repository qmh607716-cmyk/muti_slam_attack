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

3. GRAPH-AWARE SCORING (LIO-SAM proxy model):
   LIO-SAM factor graph is a pure CHAIN (prior → X0 ─ X1 ─ X2 ─ ...).
   Each node has exactly 2 edges. The LiDAR-IMU coupling strength at each
   edge is characterised by edge length and yaw change:
     - Large edge (sparse env) → LiDAR constraint is weak → attack easier
     - Large yaw change → IMU dominates → LiDAR constraint relatively weaker
   Combined into: lidar_dominance(S) = normalized_edge_length × normalized_yaw_change
   Combined with: graph_coverage(S) = affected_nodes / total_nodes
   Final: score = opportunity(S) × persistence(S)
   (multiplicative: both must be high for success)

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
    """
    Load ALL graph dumps and merge into a single cumulative factor graph.

    LIO-SAM and LVI-SAM dump ONE factor per dump file (the new odometry factor
    added at the current keyframe). The last dump only contains 1 factor (the most
    recent odometry edge), so we MUST merge all dumps to reconstruct the full graph.

    For LIO-SAM:
      - Each dump has N nodes (all cumulative) and 1 factor (current odometry).
      - Merged graph: N nodes + N factors (1 prior + N-1 odometry).

    Returns a dict with:
      - node_arr : (N, 3)  [node_id, x, y]
      - node_nid : (N,)    node IDs
      - node_quat : (N, 4) [qx, qy, qz, qw]
      - factor_nids : list of sets, one per factor, containing node IDs
      - factor_source : list of "prior"|"odometry"
      - factor_info  : list of float (info weight = sum(1/noise_i))
      - factor_noise : list of noise arrays for debugging
      - chain_edges : list of (nid_a, nid_b, edge_length, angle_change_rad)
                      Only odometry edges with consecutive node IDs (LIO-SAM chain).
      - chain_edge_lengths : (N-1,) array of edge lengths
      - chain_angles : (N-1,) array of yaw angle changes between consecutive nodes
      - graph_stats : dict of statistics about the merged graph
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

    # ── Merge ALL dumps to reconstruct the full factor graph ──────────────────
    # NOTE: All fidx values are 0, so we deduplicate by (keys + source) tuple.
    seen_factors: set = set()
    all_nodes: Dict[int, dict] = {}
    all_factors: List[dict] = []

    for path in dump_files:
        with open(path) as fh:
            d = json.load(fh)

        for node in d.get("nodes", []):
            nid = int(node["id"])
            all_nodes[nid] = node

        for fac in d.get("factors", []):
            keys = tuple(sorted(fac.get("keys", [])))
            src = fac.get("source", "unknown")
            key_sig = (keys, src)
            if key_sig not in seen_factors:
                seen_factors.add(key_sig)
                all_factors.append(fac)

    node_nids = np.array(sorted(all_nodes.keys()), dtype=np.int32)

    # ── GPS coordinate alignment ───────────────────────────────────────────────────
    # The ISAM2-optimized coordinates in graph dumps don't match the SMVS/GPS
    # coordinate system. We align via time-based matching:
    #   node ID → bag timestamp → GPS trajectory position
    #
    # Detection: look for *_gps.csv in the sibling "original/" directory
    gps_x_aligned: Optional[np.ndarray] = None
    gps_y_aligned: Optional[np.ndarray] = None

    _gd_parent = os.path.dirname(dump_dir.rstrip(os.sep))  # e.g. .../slamspoof_handheld
    _original_dir = os.path.join(_gd_parent, "original")

    _gps_files = []
    if os.path.isdir(_original_dir):
        _gps_files = sorted([
            os.path.join(_original_dir, f)
            for f in os.listdir(_original_dir)
            if "_gps" in f.lower() and f.endswith(".csv")
        ])

    for _gps_path in _gps_files:
        if os.path.isfile(_gps_path):
            try:
                _gps_df = pd.read_csv(_gps_path)
                _gps_df.columns = _gps_df.columns.str.strip()
                _gps_df = _gps_df.dropna()
                _gps_t = _gps_df["time"].values.astype(np.float64)
                _gps_x = _gps_df["x"].values.astype(np.float64)
                _gps_y = _gps_df["y"].values.astype(np.float64)
                # Map node IDs to GPS positions via linear time interpolation
                n_total = max(int(node_nids.max()) + 1, 1)
                n_gps = len(_gps_t)
                # Estimate bag duration
                bag_duration = float(_gps_t[-1]) if len(_gps_t) > 1 else float(n_total)
                node_times = np.arange(n_total, dtype=np.float64) * bag_duration / max(n_total - 1, 1)
                # Clamp to GPS range
                node_times = np.clip(node_times, _gps_t.min(), _gps_t.max())
                gps_x_aligned = np.interp(node_times, _gps_t, _gps_x)
                gps_y_aligned = np.interp(node_times, _gps_t, _gps_y)
                # Build map: node_id → gps_index
                gps_x_for_nids = np.array([gps_x_aligned[int(n)] if int(n) < len(gps_x_aligned) else 0.0 for n in node_nids], dtype=np.float64)
                gps_y_for_nids = np.array([gps_y_aligned[int(n)] if int(n) < len(gps_y_aligned) else 0.0 for n in node_nids], dtype=np.float64)
                node_arr = np.stack([
                    node_nids.astype(float),
                    gps_x_for_nids,
                    gps_y_for_nids,
                ], axis=1)
                node_quat = np.stack([
                    np.array([all_nodes[n].get("qx", 0.0) for n in node_nids], dtype=np.float64),
                    np.array([all_nodes[n].get("qy", 0.0) for n in node_nids], dtype=np.float64),
                    np.array([all_nodes[n].get("qz", 0.0) for n in node_nids], dtype=np.float64),
                    np.array([all_nodes[n].get("qw", 1.0) for n in node_nids], dtype=np.float64),
                ], axis=1)
                break
            except Exception:
                pass

    # Fallback: use raw ISAM2 coordinates if GPS not found
    if gps_x_aligned is None:
        node_arr = np.stack([
            node_nids.astype(float),
            np.array([all_nodes[n].get("x", 0.0) for n in node_nids], dtype=np.float64),
            np.array([all_nodes[n].get("y", 0.0) for n in node_nids], dtype=np.float64),
        ], axis=1)
        node_quat = np.stack([
            np.array([all_nodes[n].get("qx", 0.0) for n in node_nids], dtype=np.float64),
            np.array([all_nodes[n].get("qy", 0.0) for n in node_nids], dtype=np.float64),
            np.array([all_nodes[n].get("qz", 0.0) for n in node_nids], dtype=np.float64),
            np.array([all_nodes[n].get("qw", 1.0) for n in node_nids], dtype=np.float64),
        ], axis=1)

    factor_nids: List[set] = []
    factor_source: List[str] = []
    factor_info: List[float] = []
    factor_noise: List[List[float]] = []

    for fac in all_factors:
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
        factor_noise.append([float(s) for s in noise] if noise else [])

    # ── Precompute chain metrics for LIO-SAM ──────────────────────────────────
    # LIO-SAM has a pure chain: consecutive node IDs are connected by odometry edges.
    # We extract edge lengths and yaw angle changes to characterise the
    # LiDAR-IMU coupling strength at each segment.
    chain_edges: List[Tuple] = []
    chain_edge_lengths: List[float] = []
    chain_angles: List[float] = []

    def _quat_to_yaw(qx, qy, qz, qw):
        """Yaw from quaternion (ROS REP-103 convention)."""
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    # Sort nodes by ID to follow chain order
    node_quat_by_id = {nid: node_quat[i] for i, nid in enumerate(node_nids)}
    node_pos_by_id = {nid: node_arr[i] for i, nid in enumerate(node_nids)}

    for i in range(len(node_nids) - 1):
        nid_a = int(node_nids[i])
        nid_b = int(node_nids[i + 1])
        # Check if this is a consecutive chain edge
        if abs(nid_b - nid_a) == 1:
            pa = node_pos_by_id[nid_a]
            pb = node_pos_by_id[nid_b]
            dx = pb[1] - pa[1]
            dy = pb[2] - pa[2]
            edge_len = math.sqrt(dx * dx + dy * dy)

            ya = _quat_to_yaw(*node_quat_by_id[nid_a])
            yb = _quat_to_yaw(*node_quat_by_id[nid_b])
            d_yaw = ((yb - ya + math.pi) % (2.0 * math.pi)) - math.pi
            # Magnitude of angular change
            angle_change = abs(d_yaw)

            chain_edges.append((nid_a, nid_b, edge_len, angle_change))
            chain_edge_lengths.append(edge_len)
            chain_angles.append(angle_change)

    chain_edge_lengths_arr = np.array(chain_edge_lengths, dtype=np.float64)
    chain_angles_arr = np.array(chain_angles, dtype=np.float64)

    # ── Compute betweenness centrality (Brandes algorithm) ───────────────────────
    # Betweenness: fraction of shortest paths passing through each node.
    # High betweenness = node is an "information bottleneck" = attack amplification.
    # This is the CORE shared metric across LIO-SAM and LVI-SAM.
    from collections import defaultdict

    adj: Dict[int, List[int]] = defaultdict(list)
    for f_nids in factor_nids:
        nid_list = list(f_nids)
        if len(nid_list) == 2:
            adj[nid_list[0]].append(nid_list[1])
            adj[nid_list[1]].append(nid_list[0])

    all_graph_nodes = sorted(adj.keys())
    n_graph = len(all_graph_nodes)

    bc: Dict[int, float] = {n: 0.0 for n in all_graph_nodes}
    for s in all_graph_nodes:
        S: List[int] = []
        P: Dict[int, List[int]] = {v: [] for v in all_graph_nodes}
        sigma: Dict[int, float] = {v: 0.0 for v in all_graph_nodes}
        sigma[s] = 1.0
        d: Dict[int, int] = {v: -1 for v in all_graph_nodes}
        d[s] = 0
        Q: List[int] = [s]
        while Q:
            v = Q.pop(0)
            S.append(v)
            for w in adj[v]:
                if d[w] < 0:
                    d[w] = d[v] + 1
                    Q.append(w)
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta: Dict[int, float] = {v: 0.0 for v in all_graph_nodes}
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]
    if n_graph > 2:
        scale = 1.0 / ((n_graph - 1) * (n_graph - 2))
        for v in bc:
            bc[v] *= scale

    # ── Compute per-node stiffness and degree ───────────────────────────────────
    # Stiffness: sum of factor_info at each node (information weight)
    # Degree: number of factors touching each node
    node_stiffness: Dict[int, float] = defaultdict(float)
    node_degree: Dict[int, int] = defaultdict(int)
    for f_idx, f_nids in enumerate(factor_nids):
        fi = factor_info[f_idx]
        for nid in f_nids:
            node_stiffness[nid] += fi
            node_degree[nid] += 1

    # Align all per-node arrays with node_nids order
    # Use log1p(stiffness) to compress extreme values from info=1/noise
    bc_arr = np.array([float(bc.get(int(nid), 0.0)) for nid in node_nids], dtype=np.float64)
    stiffness_arr = np.array(
        [math.log1p(node_stiffness.get(int(nid), 0.0)) for nid in node_nids],
        dtype=np.float64
    )
    degree_arr = np.array([float(node_degree.get(int(nid), 0)) for nid in node_nids], dtype=np.float64)

    result = dict(
        node_arr=node_arr,
        node_nids=node_nids,
        node_quat=node_quat,
        factor_nids=factor_nids,
        factor_source=factor_source,
        factor_info=factor_info,
        factor_noise=factor_noise,
        chain_edges=chain_edges,
        chain_edge_lengths=chain_edge_lengths_arr,
        chain_angles=chain_angles_arr,
        node_betweenness=bc_arr,
        node_stiffness=stiffness_arr,
        node_degree=degree_arr,
    )
    _graph_cache[cache_key] = result

    # ── Print graph statistics ───────────────────────────────────────────────
    from collections import Counter
    src_counts = Counter(factor_source)
    # Node connectivity: how many factors touch each node?
    node_factor_count = {nid: 0 for nid in node_nids}
    for f_nids in factor_nids:
        for nid in f_nids:
            node_factor_count[nid] = node_factor_count.get(nid, 0) + 1

    avg_connectivity = sum(node_factor_count.values()) / max(len(node_factor_count), 1)
    max_connectivity = max(node_factor_count.values()) if node_factor_count else 0

    print(f"[GraphCache] merged {len(dump_files)} dumps, "
          f"{len(node_nids)} nodes, {len(factor_nids)} factors "
          f"({dict(src_counts)})", file=sys.stderr)
    print(f"[GraphCache] node connectivity: avg={avg_connectivity:.2f}, max={max_connectivity}",
          file=sys.stderr)

    if len(chain_edge_lengths_arr) > 0:
        print(f"[GraphCache] chain edges: {len(chain_edge_lengths_arr)}, "
              f"edge_len mean={chain_edge_lengths_arr.mean():.2f}m "
              f"std={chain_edge_lengths_arr.std():.2f}m "
              f"[{chain_edge_lengths_arr.min():.2f}, {chain_edge_lengths_arr.max():.2f}]m",
              file=sys.stderr)
        print(f"[GraphCache] yaw change: mean={chain_angles_arr.mean():.3f}rad "
              f"std={chain_angles_arr.std():.3f}rad",
              file=sys.stderr)

    print(f"[GraphCache] betweenness: max={bc_arr.max():.4f}, "
          f"mean={bc_arr.mean():.4f}, nodes_with_0={int((bc_arr == 0).sum())}",
          file=sys.stderr)
    print(f"[GraphCache] stiffness(log1p): max={stiffness_arr.max():.2f}, "
          f"mean={stiffness_arr.mean():.2f}",
          file=sys.stderr)
    print(f"[GraphCache] degree: max={int(degree_arr.max())}, "
          f"mean={degree_arr.mean():.2f}, "
          f"deg4_nodes={int((degree_arr >= 4).sum())}",
          file=sys.stderr)

    print(f"[GraphCache] loaded in {time.time()-t0:.1f}s", file=sys.stderr)
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


# ============================================================================
# Stage 1: New Scoring Functions — LIO-SAM Graph-Aware Design
# ============================================================================
#
# Design rationale (LIO-SAM as proxy for LVI-SAM):
#
# LIO-SAM factor graph is a CHAIN:
#   prior → X0 ──odometry── X1 ──odometry── X2 ── ... ── X5751
#   Each node has exactly 2 edges (prior + one odometry, except endpoints).
#   No loop closure, no GPS, no visual factors.
#
# KEY INSIGHT: In LIO-SAM, the LiDAR-IMU coupling determines attack success.
# Each odometry edge = [LiDAR scan-matching] + [IMU preintegration] → combined constraint.
# The SLAM optimiser balances both to estimate robot motion.
#
# When edge length is LARGE (sparse environment, P10=0.6m, P90=286m):
#   → LiDAR must match scans over a large distance → scan-matching is harder/looser
#   → Optimiser trusts IMU preintegration more → LiDAR constraint is "weaker"
#   → ATTACK EASIER: injecting fake LiDAR points pushes the pose in the attack direction
#
# When edge length is SMALL (dense environment):
#   → LiDAR finds tight correspondences → scan-matching dominates
#   → LiDAR constraint is "stronger" → harder to bias the estimate
#   → ATTACK HARDER
#
# Similarly for yaw change: large rotation → IMU preintegration dominates →
# LiDAR constraint relatively weaker → more vulnerable to attack.
#
# NEW SCORING FORMULA:
#   score(S) = opportunity(S) × persistence(S)
#
# where:
#   opportunity(S) = reach(S) × bivul(S)
#     → Geometric: can I reach vulnerable frames? (reach)
#     → Directional: am I in a vulnerable direction? (bivul)
#
#   persistence(S) = lidar_dominance(S) × graph_coverage(S)
#     → Proxy model: how weak is the LiDAR constraint at affected edges? (lidar_dominance)
#     → Structural: how much of the trajectory is affected? (graph_coverage)
#
# This is multiplicative (not additive): if either factor is 0, the score is 0.
# This is physically correct: you need BOTH a vulnerable direction AND a weak
# LiDAR constraint for the attack to succeed.


def compute_lidar_dominance(
    sx: float, sy: float,
    frames: List['VulFrame'],
    gdata: Optional[Dict[str, Any]] = None,
    distance_threshold: float = 15.0,
) -> float:
    """
    LiDAR Dominance: how "weak" are the LiDAR constraints at the attack edges?

    LIO-SAM (LiDAR-IMU tightly coupled):
      The odometry edge at segment i spans distance d_i and yaw change Δθ_i.
      The "LiDAR weakness" of this edge is:
        lidar_weakness_i = (d_i / d_ref) × (Δθ_i / Δθ_ref)
      where d_ref and Δθ_ref are the medians (robust to outliers).

      A larger edge (sparse environment) → LiDAR has fewer correspondences →
      optimizer trusts IMU more → LiDAR constraint is relatively weaker →
      this is GOOD for attack → high lidar_dominance.

      We use: lidar_dominance = 1 / (1 + median_edge_length_normalized)
      Range: (0, 1]. Higher = more vulnerable.

    LVI-SAM (with loop closure / GPS):
      Falls back to: lidar_dominance = 1 / (1 + avg_loop_count_per_node)
      More loop closures → stronger overall constraint → lower dominance → harder to attack.

    Fallback (no graph): returns 0.5 (neutral).

    NOTE: caller passes gdata directly to avoid repeated loading in CMA-ES hot loop.
    """
    affected = []
    for f in frames:
        dist = math.hypot(sx - f.x, sy - f.y)
        if dist < distance_threshold:
            affected.append(f)

    if not affected:
        return 0.0

    if gdata is None:
        return 0.5

    node_arr = gdata["node_arr"]
    node_nids = gdata["node_nids"]
    factor_source = gdata["factor_source"]
    chain_edge_lengths = gdata.get("chain_edge_lengths")
    chain_angles = gdata.get("chain_angles")

    has_loop_closure = any(s == "loop_closure" for s in factor_source)

    if has_loop_closure:
        # LVI-SAM: dominance from loop closure count
        loop_counts: Dict[int, int] = {nid: 0 for nid in node_nids}
        for f_nids, f_src in zip(gdata["factor_nids"], gdata["factor_source"]):
            if f_src == "loop_closure":
                for nid in f_nids:
                    if nid in loop_counts:
                        loop_counts[nid] += 1

        # Map frames to affected nodes
        affected_nid_set = set()
        for f in affected:
            dists = np.hypot(node_arr[:, 1] - f.x, node_arr[:, 2] - f.y)
            nearest_idx = int(np.argmin(dists))
            affected_nid_set.add(int(node_nids[nearest_idx]))

        if not affected_nid_set:
            return 0.0

        counts = [loop_counts.get(nid, 0) for nid in affected_nid_set]
        avg_loop = float(np.mean(counts)) if counts else 0.0
        return float(np.clip(1.0 / (1.0 + avg_loop), 0.0, 1.0))

    else:
        # LIO-SAM: dominance from edge length / yaw change
        if chain_edge_lengths is None or len(chain_edge_lengths) == 0:
            return 0.5

        # Find the nearest chain edge for each affected frame
        # Build map: node_id → edge index (for consecutive edges)
        # Edge i connects node_nids[i] → node_nids[i+1]
        nid_to_idx = {int(nid): i for i, nid in enumerate(node_nids)}

        affected_edge_lengths = []
        affected_yaw_changes = []
        for f in affected:
            dists = np.hypot(node_arr[:, 1] - f.x, node_arr[:, 2] - f.y)
            nearest_idx = int(np.argmin(dists))
            nid = int(node_nids[nearest_idx])

            # Find the edge that this frame's pose "belongs" to
            # The pose is estimated relative to adjacent nodes
            if nid in nid_to_idx:
                idx = nid_to_idx[nid]
                # Use both adjacent edges if possible
                if idx < len(chain_edge_lengths):
                    affected_edge_lengths.append(chain_edge_lengths[idx])
                    affected_yaw_changes.append(chain_angles[idx])
                if idx > 0:
                    affected_edge_lengths.append(chain_edge_lengths[idx - 1])
                    affected_yaw_changes.append(chain_angles[idx - 1])

        if not affected_edge_lengths:
            return 0.5

        el_arr = np.array(affected_edge_lengths, dtype=np.float64)
        ya_arr = np.array(affected_yaw_changes, dtype=np.float64)

        # Use median of chain as reference (robust to outliers)
        d_ref = float(np.median(chain_edge_lengths))
        ya_ref = float(np.median(chain_angles))
        ya_ref = max(ya_ref, 1e-4)  # avoid div by zero

        # Compute combined weakness score
        # d_normalized: how much larger is the affected edge vs median
        d_norm = np.clip(el_arr / d_ref, 0.0, 10.0)
        # yaw_norm: how much larger is the yaw change vs median
        ya_norm = np.clip(ya_arr / ya_ref, 0.0, 10.0)

        # Combined: geometric mean of both normalizations
        combined_norm = np.sqrt(d_norm * ya_norm)
        # Convert to dominance score: high normalization → high dominance
        # Using sigmoid-like: dominance = norm / (1 + norm)
        dominance = combined_norm / (1.0 + combined_norm)

        return float(np.clip(dominance.mean(), 0.0, 1.0))


def compute_attack_persistence(
    sx: float, sy: float,
    frames: List['VulFrame'],
    gdata: Optional[Dict[str, Any]] = None,
    distance_threshold: float = 15.0,
) -> float:
    """
    Attack Persistence: how "stuck" is the bias in the factor graph?

    The key insight from LIO-SAM chain topology:
      Each node is ONLY connected to its immediate neighbours.
      Therefore, injected bias at node i ONLY propagates to neighbours i-1 and i+1.
      The attack's "persistence" = how many keyframes in the affected zone?

    In LIO-SAM chain (pure sequential edges):
      coverage = |{affected keyframes}| / |{all keyframes}|
      This is always tiny for localized attacks (e.g. 2 frames / 19901 = 0.01%).
      We compress via sigmoid so that even small coverage is meaningful:
        persistence = sigmoid((n_affected - 1) / tau)
      where tau=5 is a half-life (5 frames → 50% coverage score).

    In LVI-SAM (with loop closure):
      persistence = average factor density inverse at affected nodes
      More factors per node → more constraint → bias is absorbed → lower persistence.
      persistence = 1 / (1 + avg_factor_count_in_zone)

    The "tau" parameter (half-life = 5 frames) means:
      1 frame → 17% persistence
      5 frames → 50% persistence
      10 frames → 73% persistence
      20 frames → 90% persistence
    This captures the intuition: even a few affected frames are meaningful.

    NOTE: caller passes gdata directly to avoid repeated loading in CMA-ES hot loop.
    """
    affected = []
    for f in frames:
        dist = math.hypot(sx - f.x, sy - f.y)
        if dist < distance_threshold:
            affected.append(f)

    if not affected:
        return 0.0

    if gdata is None:
        # Fallback: fraction of frames in range
        frac = len(affected) / max(len(frames), 1)
        # Sigmoid compression: tau=0.05 means 5% coverage → 50%
        tau = 0.05
        persistence = frac / (frac + tau)
        return float(np.clip(persistence, 0.0, 1.0))

    node_arr = gdata["node_arr"]
    node_nids = gdata["node_nids"]
    factor_source = gdata["factor_source"]

    has_loop_closure = any(s == "loop_closure" for s in factor_source)

    # Map frames to affected graph nodes
    affected_nid_set = set()
    for f in affected:
        dists = np.hypot(node_arr[:, 1] - f.x, node_arr[:, 2] - f.y)
        nearest_idx = int(np.argmin(dists))
        affected_nid_set.add(int(node_nids[nearest_idx]))

    if not affected_nid_set:
        return 0.0

    if has_loop_closure:
        # LVI-SAM: persistence from local factor density
        factor_counts: Dict[int, int] = {nid: 0 for nid in affected_nid_set}
        for f_nids in gdata["factor_nids"]:
            for nid in f_nids:
                if nid in factor_counts:
                    factor_counts[nid] += 1

        counts = list(factor_counts.values())
        avg_count = float(np.mean(counts)) if counts else 1.0
        persistence = 1.0 / (1.0 + avg_count)
    else:
        # LIO-SAM: sigmoid-scaled coverage
        # n_affected keyframes in range. In LIO-SAM chain, each → 1 node.
        n_affected = len(affected_nid_set)
        # Sigmoid with half-life at 5 keyframes
        tau = 5.0
        persistence = float(n_affected) / (float(n_affected) + tau)

    return float(np.clip(persistence, 0.0, 1.0))


def compute_structural_score(
    sx: float, sy: float,
    frames: List['VulFrame'],
    gdata: Optional[Dict[str, Any]] = None,
    distance_threshold: float = 15.0,
) -> float:
    """
    Structural score: betweenness-based attack amplification factor.

    DESIGN RATIONALE:
      Betweenness centrality measures how many shortest paths pass through a node.
      High betweenness nodes are "information bottlenecks" — injecting bias here
      propagates to many other nodes along many paths.

      The shared metric between LIO-SAM (white-box) and LVI-SAM (black-box):
        LIO-SAM: real betweenness computed from factor graph edges
        LVI-SAM: spatial-revisit proxy (nodes frequently revisited spatially
                 are estimated to have high betweenness in the factor graph)

      For a spoofer at (sx, sy), we find the nearest trajectory frames
      and compute the average betweenness at those nodes:
        structural_score = mean(betweenness of affected nodes)

      High betweenness → bias propagates to many nodes → high attack amplification
      Low betweenness  → bias stays local → limited attack effect
    """
    if gdata is None:
        return 0.0

    node_arr = gdata.get("node_arr")
    node_betweenness = gdata.get("node_betweenness")
    node_stiffness = gdata.get("node_stiffness")
    node_degree = gdata.get("node_degree")

    if node_arr is None or node_betweenness is None:
        return 0.0

    # Find frames within range
    affected_bc = []
    affected_stiff = []
    for f in frames:
        dist = math.hypot(sx - f.x, sy - f.y)
        if dist < distance_threshold:
            dists = np.hypot(node_arr[:, 1] - f.x, node_arr[:, 2] - f.y)
            nearest_idx = int(np.argmin(dists))
            affected_bc.append(node_betweenness[nearest_idx])
            if node_stiffness is not None:
                affected_stiff.append(node_stiffness[nearest_idx])

    if not affected_bc:
        return 0.0

    avg_bc = float(np.mean(affected_bc))

    # Normalize: map [0, max_bc] → [0, 1]
    # Use observed max across all nodes as reference
    max_bc = float(node_betweenness.max()) if len(node_betweenness) > 0 else 1.0
    if max_bc > 0:
        bc_norm = avg_bc / max_bc
    else:
        bc_norm = 0.0

    # Stiffness adjustment: high stiffness → lower score (node is well-constrained)
    # stiffness is information weight: high = many/strong factors → hard to move
    # Only apply if we have per-node stiffness data
    stiffness_score = 1.0
    if node_stiffness is not None and affected_stiff:
        avg_stiff = float(np.mean(affected_stiff))
        max_stiff = float(node_stiffness.max()) if len(node_stiffness) > 0 else 1.0
        if max_stiff > 0:
            # Convert to "weakness": 1 - (stiffness / max) → 1 means very weak
            weakness = 1.0 - min(avg_stiff / max_stiff, 1.0)
            # Geometric mean of betweenness and weakness
            stiffness_score = math.sqrt(bc_norm * weakness) if bc_norm > 0 else 0.0
        else:
            stiffness_score = bc_norm
    else:
        stiffness_score = bc_norm

    return float(np.clip(stiffness_score, 0.0, 1.0))


def new_score_formula(
    sx: float, sy: float,
    frames: List['VulFrame'],
    traj_pts: np.ndarray,
    distance_threshold: float,
    bivul_gate_threshold: float = 0.15,
    gdata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LIO-SAM-aware scoring formula with TWO COMPETING OBJECTIVES:

        score(S) = opportunity(S) + alpha * structural(S)

    where:

        opportunity(S) = reach(S) × bivul_gate(S)
          reach(S)       : geometric reachability (Gaussian-weighted, [0,1])
          bivul_gate(S)  : directional vulnerability score [0, 1]

        structural(S) = structural_score(S) × graph_coverage(S)
          structural_score(S) : betweenness-based attack amplification [0,1]
                               LIO-SAM: real betweenness from factor graph
                               LVI-SAM: spatial-revisit proxy (shared metric)
                               High betweenness → bias propagates to many nodes
          graph_coverage(S)  : sigmoid(affected_keyframes / tau)

        alpha = 0.3 (structural factors contribute at most 30% of max score)

    DESIGN RATIONALE:
      Geographic reach and LiDAR dominance are ANTI-CORRELATED:
        - Dense regions (small edges): many SMVS frames in range, but strong LiDAR
          constraints → reach is high, lidar_dominance is low
        - Sparse regions (large edges): few frames in range, but weak LiDAR constraints
          → reach is low, lidar_dominance is high

      Betweenness centrality captures a fundamentally different signal:
        - High betweenness nodes are information bottlenecks in the factor graph
        - Attacking a high-betweenness node amplifies bias across many trajectories
        - This is a SHARED metric: both LIO-SAM (white-box) and LVI-SAM (black-box)
          can use spatial-revisit frequency as a proxy for graph betweenness

      The ADDITIVE form (opportunity + 0.3 * structural) handles this:
        - Dense region: opportunity ≈ 0.2 * 0.3 = 0.06, structural ≈ 0.8 * 0.29 = 0.02
          → score ≈ 0.07 + 0.01 = 0.08
        - Sparse region: opportunity ≈ 0.0 (no frames in range), but if a few frames
          exist: opportunity > 0, and lidar_dominance boosts the score
        - Both regions get meaningful scores, ranked by the COMBINATION.

      The multiplicative coupling in opportunity (reach × bivul) still acts as an
      AND gate: both must be non-zero for opportunity > 0.

      This is fundamentally different from the original additive formula where
      isolated metrics like isolation and dominance were always ~1.0 and
      dominated the score by their fixed coefficients.
    """
    reach = compute_reach(sx, sy, frames, distance_threshold)
    bivul = compute_bivul_gate(sx, sy, frames, distance_threshold, bivul_gate_threshold)
    lidar_dom = compute_lidar_dominance(sx, sy, frames, gdata, distance_threshold)
    coverage = compute_attack_persistence(sx, sy, frames, gdata, distance_threshold)
    structural_bc = compute_structural_score(sx, sy, frames, gdata, distance_threshold)

    opportunity = reach * bivul
    # Blend betweenness-based structural score with lidar_dominance
    # geometric mean of both → both must be non-trivial for high structural score
    structural_blended = math.sqrt(structural_bc * lidar_dom) if (structural_bc > 0 and lidar_dom > 0) else 0.0
    structural = structural_blended * coverage
    alpha = 0.3
    score = opportunity + alpha * structural

    return {
        'score': float(score),
        'opportunity': float(opportunity),
        'structural': float(structural),
        'reach': float(reach),
        'bivul_gate': float(bivul),
        'lidar_dominance': float(lidar_dom),
        'graph_coverage': float(coverage),
        'structural_bc': float(structural_bc),
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
    gdata: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, float, List[Dict]]:
    """
    CMA-ES optimization using the betweenness-aware formula:

        score(S) = opportunity(S) + 0.3 * structural(S)
        opportunity(S) = reach(S) × bivul_gate(S)
        structural(S) = structural_bc(S) × graph_coverage(S)
        structural_bc(S) = betweenness-based attack amplification

    The betweenness component captures graph-theoretic vulnerability:
    high-betweenness nodes are information bottlenecks that amplify bias.
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
                gdata=gdata,
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
            gdata=gdata,
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
    min_traj_dist: float = 5.0,
    gdata=None,
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
            if d_min < min_traj_dist:
                Z[iy, ix] = 0.0
            else:
                try:
                    res = new_score_formula(xs_h[ix], ys_h[iy], frames,
                                            traj_pts, distance_threshold,
                                            gdata=gdata)
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

    # ---- Load graph data (for betweenness + stiffness) ----
    gdata = _precompute_graph_data(args.graph_dump_dir) if args.graph_dump_dir else None

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
        gdata=gdata,
    )

    score_components = {}
    try:
        score_components = new_score_formula(
            bo_x, bo_y, top_frames, traj_pts, args.distance_threshold,
            bivul_gate_threshold=args.bivul_gate_threshold,
            gdata=gdata,
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
            min_traj_dist=args.min_traj_dist,
            gdata=gdata,
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
    print(f"  Opportunity:{sc.get('opportunity', 0):.4f}", file=sys.stderr)
    print(f"  Structural:{sc.get('structural', 0):.4f}", file=sys.stderr)
    print(f"  Reach:    {sc.get('reach', 0):.4f}", file=sys.stderr)
    print(f"  Bi-Vul:   {sc.get('bivul_gate', 0):.4f}", file=sys.stderr)
    print(f"  LiDAR dom:{sc.get('lidar_dominance', 0):.4f}", file=sys.stderr)
    print(f"  Coverage:  {sc.get('graph_coverage', 0):.4f}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[OK] wrote {args.output}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
