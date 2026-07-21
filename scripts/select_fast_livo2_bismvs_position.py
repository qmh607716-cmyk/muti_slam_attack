#!/usr/bin/env python3
"""Proxy-assisted Bi-SMVS spoofer selector for transfer experiments.

Some transfer targets do not expose the same graph dump interface used by the
LVI-SAM main experiment. For transfer experiments we therefore build or load an
attacker-side surrogate route graph. This proxy is not treated as the victim
SLAM's internal optimizer graph; it supplies route-level structural cues for
placement, while Bi-SMVS supplies the sensing-level vulnerability signal.
"""

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def load_traj(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    for col in ("x", "y"):
        if col not in df:
            raise SystemExit(f"trajectory missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    if len(df) < 2:
        raise SystemExit("trajectory has too few valid rows")
    return df[["x", "y"]].to_numpy(np.float64)


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _brandes_betweenness(n_nodes: int, adj: List[List[int]]) -> np.ndarray:
    bc = np.zeros(n_nodes, dtype=np.float64)
    for s in range(n_nodes):
        stack: List[int] = []
        pred: List[List[int]] = [[] for _ in range(n_nodes)]
        sigma = np.zeros(n_nodes, dtype=np.float64)
        sigma[s] = 1.0
        dist = np.full(n_nodes, -1, dtype=np.int32)
        dist[s] = 0
        queue = [s]
        q_head = 0
        while q_head < len(queue):
            v = queue[q_head]
            q_head += 1
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n_nodes, dtype=np.float64)
        while stack:
            w = stack.pop()
            if sigma[w] <= 0:
                continue
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]
    if n_nodes > 2:
        bc /= float((n_nodes - 1) * (n_nodes - 2))
    return bc


def resample_polyline(points: np.ndarray, n_out: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if n_out <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(points) == 1 or n_out == 1:
        return np.repeat(points[:1], n_out, axis=0)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(points[:1], n_out, axis=0)
    q = np.linspace(0.0, s[-1], n_out)
    x = np.interp(q, s, points[:, 0])
    y = np.interp(q, s, points[:, 1])
    return np.stack([x, y], axis=1)


def align_graph_nodes_to_traj(raw_nodes: np.ndarray, traj: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Align proxy graph coordinates to the victim clean trajectory frame.

    LIO-SAM and the victim SLAM may publish trajectories in different odometry frames.
    For placement scoring we only need route-level structure in the victim
    trajectory frame, so we fit a 2D similarity transform by arc-length order.
    """
    raw_nodes = np.asarray(raw_nodes, dtype=np.float64)
    if len(raw_nodes) < 2:
        return raw_nodes.copy(), {"scale": 1.0, "rotation_rad": 0.0, "rmse": 0.0}

    target = resample_polyline(traj, len(raw_nodes))
    zx = raw_nodes[:, 0] + 1j * raw_nodes[:, 1]
    zy = target[:, 0] + 1j * target[:, 1]
    mx = zx.mean()
    my = zy.mean()
    denom = float(np.sum(np.abs(zx - mx) ** 2))
    if denom < 1e-12:
        return target, {"scale": 1.0, "rotation_rad": 0.0, "rmse": 0.0}
    a = np.sum(np.conj(zx - mx) * (zy - my)) / denom
    aligned_z = a * (zx - mx) + my
    aligned = np.stack([aligned_z.real, aligned_z.imag], axis=1)
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return aligned, {
        "scale": float(abs(a)),
        "rotation_rad": float(np.angle(a)),
        "rmse": rmse,
    }


def _graph_dict_from_nodes_edges(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    edge_sources: List[str],
    source: str,
    alignment: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    n_nodes = len(nodes)
    source_by_edge: Dict[Tuple[int, int], str] = {}
    for k, (i, j) in enumerate(edges):
        e = tuple(sorted((int(i), int(j))))
        if e[0] == e[1]:
            continue
        source_by_edge[e] = edge_sources[k] if k < len(edge_sources) else (
            "odometry" if abs(e[0] - e[1]) == 1 else "loop_closure"
        )
    edge_set = sorted(source_by_edge)
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    edge_len = []
    loop_counts = np.zeros(n_nodes, dtype=np.float64)
    for k, (i, j) in enumerate(edge_set):
        if i < 0 or j < 0 or i >= n_nodes or j >= n_nodes:
            continue
        adj[i].append(j)
        adj[j].append(i)
        edge_len.append(float(np.linalg.norm(nodes[i] - nodes[j])))
        src = source_by_edge.get((i, j), "odometry" if abs(i - j) == 1 else "loop_closure")
        if src != "odometry" and src != "chain":
            loop_counts[i] += 1.0
            loop_counts[j] += 1.0

    bc = _brandes_betweenness(n_nodes, adj) if n_nodes else np.zeros(0)
    degree = np.asarray([len(a) for a in adj], dtype=np.float64)

    local_motion = np.zeros(n_nodes, dtype=np.float64)
    if n_nodes > 1:
        chain_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        local_motion[0] = chain_lengths[0]
        local_motion[-1] = chain_lengths[-1]
        for i in range(1, n_nodes - 1):
            local_motion[i] = 0.5 * (chain_lengths[i - 1] + chain_lengths[i])

    num_chain = sum(1 for i, j in edge_set if abs(i - j) == 1)
    num_loop = max(len(edge_set) - num_chain, 0)
    result = {
        "nodes": np.asarray(nodes, dtype=np.float64),
        "edges": np.asarray(edge_set, dtype=np.int32),
        "betweenness": normalize01(bc),
        "degree": degree,
        "loop_counts": loop_counts,
        "local_motion": normalize01(local_motion),
        "edge_lengths": np.asarray(edge_len, dtype=np.float64),
        "num_chain_edges": np.asarray([num_chain], dtype=np.int32),
        "num_loop_edges": np.asarray([num_loop], dtype=np.int32),
        "source": source,
    }
    if alignment is not None:
        result["alignment"] = alignment
    return result


def load_lio_sam_graph(dump_dir: str, traj: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    if not dump_dir:
        return None
    dump_files = sorted(
        glob.glob(os.path.join(os.path.expanduser(dump_dir), "dump_*.json")),
        key=lambda p: int(re.search(r"dump_(\d+)\.json", os.path.basename(p)).group(1)),
    )
    if not dump_files:
        return None

    all_nodes: Dict[int, dict] = {}
    edge_sources_by_pair: Dict[Tuple[int, int], str] = {}

    for path in dump_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        for node in d.get("nodes", []):
            try:
                all_nodes[int(node["id"])] = node
            except Exception:
                continue
        for fac in d.get("factors", []):
            keys = []
            for k in fac.get("keys", []):
                m = re.search(r"X(\d+)", str(k))
                if m:
                    keys.append(int(m.group(1)))
            if len(keys) != 2:
                continue
            i, j = sorted(keys)
            src = str(fac.get("source", "odometry" if abs(i - j) == 1 else "loop_closure"))
            edge_sources_by_pair[(i, j)] = src

    if len(all_nodes) < 2:
        return None

    node_ids = sorted(all_nodes)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    raw_nodes = np.asarray(
        [[float(all_nodes[n].get("x", 0.0)), float(all_nodes[n].get("y", 0.0))] for n in node_ids],
        dtype=np.float64,
    )
    nodes, alignment = align_graph_nodes_to_traj(raw_nodes, traj)
    edges: List[Tuple[int, int]] = []
    edge_sources: List[str] = []
    for (a, b), src in sorted(edge_sources_by_pair.items()):
        if a in id_to_idx and b in id_to_idx:
            edges.append((id_to_idx[a], id_to_idx[b]))
            edge_sources.append(src)

    if not edges:
        edges = [(i, i + 1) for i in range(len(node_ids) - 1)]
        edge_sources = ["odometry"] * len(edges)

    graph = _graph_dict_from_nodes_edges(
        nodes,
        edges,
        edge_sources,
        source="lio_sam_graph_dump",
        alignment=alignment,
    )
    graph["node_ids"] = np.asarray(node_ids, dtype=np.int32)
    graph["dump_files"] = np.asarray([len(dump_files)], dtype=np.int32)
    return graph


def build_proxy_graph(
    traj: np.ndarray,
    stride: int,
    loop_radius: float,
    loop_min_gap: int,
    max_loop_edges_per_node: int,
) -> Dict[str, np.ndarray]:
    stride = max(1, int(stride))
    node_indices = list(range(0, len(traj), stride))
    if node_indices[-1] != len(traj) - 1:
        node_indices.append(len(traj) - 1)
    nodes = traj[node_indices].astype(np.float64)
    n_nodes = len(nodes)

    edge_set = set()
    edge_sources: List[str] = []
    for i in range(n_nodes - 1):
        edge_set.add((i, i + 1))
        edge_sources.append("chain")

    loop_counts = np.zeros(n_nodes, dtype=np.int32)
    min_gap_nodes = max(1, int(math.ceil(loop_min_gap / stride)))
    if loop_radius > 0 and n_nodes > 2:
        for i in range(n_nodes):
            candidates = []
            for j in range(i + min_gap_nodes, n_nodes):
                d = float(np.linalg.norm(nodes[i] - nodes[j]))
                if d <= loop_radius:
                    candidates.append((d, j))
            candidates.sort(key=lambda x: x[0])
            for _d, j in candidates[:max_loop_edges_per_node]:
                e = (i, j)
                if e not in edge_set:
                    edge_set.add(e)
                    edge_sources.append("proximity")
                    loop_counts[i] += 1
                    loop_counts[j] += 1

    edges = sorted(edge_set)
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    edge_len = []
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
        edge_len.append(float(np.linalg.norm(nodes[i] - nodes[j])))

    bc = _brandes_betweenness(n_nodes, adj)
    degree = np.asarray([len(a) for a in adj], dtype=np.float64)

    chain_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1) if n_nodes > 1 else np.zeros(0)
    local_motion = np.zeros(n_nodes, dtype=np.float64)
    if len(chain_lengths):
        local_motion[0] = chain_lengths[0]
        local_motion[-1] = chain_lengths[-1]
        for i in range(1, n_nodes - 1):
            local_motion[i] = 0.5 * (chain_lengths[i - 1] + chain_lengths[i])

    graph = {
        "nodes": nodes,
        "node_indices": np.asarray(node_indices, dtype=np.int32),
        "edges": np.asarray(edges, dtype=np.int32),
        "betweenness": normalize01(bc),
        "degree": degree,
        "loop_counts": loop_counts.astype(np.float64),
        "local_motion": normalize01(local_motion),
        "edge_lengths": np.asarray(edge_len, dtype=np.float64),
        "num_chain_edges": np.asarray([max(n_nodes - 1, 0)], dtype=np.int32),
        "num_loop_edges": np.asarray([int(loop_counts.sum() // 2)], dtype=np.int32),
        "source": "trajectory_surrogate",
    }
    return graph


def load_frames(smvs_path: str, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(smvs_path)
    required = {"timestamp", "x", "y", "z", "yaw", "frame_bi_smvs", "vul_angle_deg"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Bi-SMVS CSV missing columns: {sorted(missing)}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))
    df = df[~((df["x"].abs() < 1e-9) & (df["y"].abs() < 1e-9))]
    if len(df) == 0:
        raise SystemExit("Bi-SMVS CSV has no valid non-origin rows")
    return df.nlargest(min(top_k, len(df)), "frame_bi_smvs").reset_index(drop=True)


def min_dist_to_traj(sx: float, sy: float, traj: np.ndarray) -> float:
    return float(np.min(np.hypot(traj[:, 0] - sx, traj[:, 1] - sy)))


def trigger_ratio(sx: float, sy: float, traj: np.ndarray, distance_threshold: float) -> float:
    d = np.hypot(traj[:, 0] - sx, traj[:, 1] - sy)
    return float((d <= distance_threshold).mean())


def local_bucket_angle_to_world(vul_angle_deg: float, yaw: float) -> float:
    # Bucket convention stores LiDAR +X as 180 deg.
    local = math.radians(vul_angle_deg - 180.0)
    return yaw + local


def direction_alignment(frame, sx: float, sy: float, spoofing_range: float) -> float:
    dx = sx - float(frame.x)
    dy = sy - float(frame.y)
    if abs(dx) + abs(dy) < 1e-9:
        return 0.0
    world_angle = math.atan2(dy, dx)
    local_angle = world_angle - float(frame.yaw)
    bucket_angle = (math.degrees(local_angle) + 180.0) % 360.0
    diff = abs(wrap_deg(bucket_angle - float(frame.vul_angle_deg)))
    half = max(spoofing_range * 0.5, 1.0)
    if diff > half:
        return 0.0
    return float(1.0 - diff / half)


def score_candidate(
    sx: float,
    sy: float,
    frames: pd.DataFrame,
    traj: np.ndarray,
    min_traj_dist: float,
    max_traj_dist: float,
    distance_threshold: float,
    spoofing_range: float,
    min_trigger_ratio: float,
    max_trigger_ratio: float,
    target_trigger_ratio: float,
    proxy_graph: Optional[Dict[str, np.ndarray]],
) -> Tuple[float, Dict[str, float]]:
    dmin = min_dist_to_traj(sx, sy, traj)
    if dmin < min_traj_dist or dmin > max_traj_dist:
        return -1.0, {"min_traj_dist": dmin, "in_band": 0.0}

    scores = frames["frame_bi_smvs"].to_numpy(np.float64)
    smax = float(scores.max()) if len(scores) else 1.0
    accum = 0.0
    affected = 0
    align_sum = 0.0
    for row in frames.itertuples(index=False):
        dist = math.hypot(sx - float(row.x), sy - float(row.y))
        if dist > distance_threshold:
            continue
        align = direction_alignment(row, sx, sy, spoofing_range)
        if align <= 0:
            continue
        affected += 1
        align_sum += align
        dist_w = 1.0 - abs(dist - min(max(dist, min_traj_dist), max_traj_dist)) / max(max_traj_dist, 1.0)
        dist_w = float(np.clip(dist_w, 0.2, 1.0))
        vul_w = float(row.frame_bi_smvs) / max(smax, 1e-9)
        accum += vul_w * align * dist_w

    cover = trigger_ratio(sx, sy, traj, distance_threshold)
    if cover < min_trigger_ratio or cover > max_trigger_ratio:
        return -1.0, {
            "min_traj_dist": dmin,
            "in_band": 1.0,
            "affected_top_frames": float(affected),
            "trigger_ratio": cover,
            "trigger_in_band": 0.0,
        }
    if affected == 0 or cover <= 0:
        return -1.0, {
            "min_traj_dist": dmin,
            "in_band": 1.0,
            "affected_top_frames": 0,
            "trigger_ratio": cover,
            "trigger_in_band": 1.0,
        }

    mean_align = align_sum / affected
    trigger_sigma = max((max_trigger_ratio - min_trigger_ratio) / 4.0, 1e-3)
    trigger_quality = math.exp(-((cover - target_trigger_ratio) ** 2) / (2.0 * trigger_sigma ** 2))

    structural_importance = 0.5
    graph_coverage = cover
    lidar_dominance = 0.5
    proxy_score = 0.5
    affected_proxy_nodes = 0
    if proxy_graph is not None:
        nodes = proxy_graph["nodes"]
        dnode = np.hypot(nodes[:, 0] - sx, nodes[:, 1] - sy)
        affected_mask = dnode <= distance_threshold
        affected_proxy_nodes = int(affected_mask.sum())
        if affected_proxy_nodes > 0:
            structural_importance = float(np.mean(proxy_graph["betweenness"][affected_mask]))
            graph_coverage = float(affected_proxy_nodes / max(len(nodes), 1))
            motion = float(np.mean(proxy_graph["local_motion"][affected_mask]))
            loop_density = float(np.mean(proxy_graph["loop_counts"][affected_mask] > 0.0))
            # High local motion implies more route-level scan-matching exposure;
            # dense proximity edges imply more redundant correction, so they reduce
            # this proxy dominance.
            lidar_dominance = float(np.clip(0.65 * motion + 0.35 * (1.0 - loop_density), 0.0, 1.0))
            proxy_score = float(np.clip(
                0.45 * structural_importance +
                0.30 * graph_coverage +
                0.25 * lidar_dominance,
                0.0,
                1.0,
            ))

    score = accum + 0.25 * mean_align + 0.35 * trigger_quality + 0.50 * proxy_score
    return float(score), {
        "min_traj_dist": dmin,
        "in_band": 1.0,
        "affected_top_frames": float(affected),
        "trigger_ratio": cover,
        "trigger_in_band": 1.0,
        "trigger_quality": float(trigger_quality),
        "mean_direction_alignment": float(mean_align),
        "vulnerability_direction_score": float(accum),
        "proxy_score": float(proxy_score),
        "structural_importance": float(structural_importance),
        "lidar_dominance": float(lidar_dominance),
        "graph_coverage": float(graph_coverage),
        "affected_proxy_nodes": float(affected_proxy_nodes),
    }


def generate_candidates(
    frames: pd.DataFrame,
    traj: np.ndarray,
    min_traj_dist: float,
    max_traj_dist: float,
    spoofing_range: float,
    n_random: int,
    seed: int,
) -> List[Tuple[float, float]]:
    candidates: List[Tuple[float, float]] = []

    offsets = np.linspace(min_traj_dist, max_traj_dist, 8)
    jitters = np.linspace(-0.35 * spoofing_range, 0.35 * spoofing_range, 9)
    for row in frames.itertuples(index=False):
        base = local_bucket_angle_to_world(float(row.vul_angle_deg), float(row.yaw))
        for dist in offsets:
            for jitter in jitters:
                a = base + math.radians(float(jitter))
                candidates.append((float(row.x) + float(dist) * math.cos(a),
                                   float(row.y) + float(dist) * math.sin(a)))

    rng = np.random.default_rng(seed)
    pad = max_traj_dist
    xmin, xmax = float(traj[:, 0].min() - pad), float(traj[:, 0].max() + pad)
    ymin, ymax = float(traj[:, 1].min() - pad), float(traj[:, 1].max() + pad)
    for _ in range(n_random):
        candidates.append((float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax))))

    # Add normal offsets from every 20th trajectory sample, useful for short paths.
    for i in range(1, len(traj) - 1, max(1, len(traj) // 40)):
        prev_pt = traj[i - 1]
        next_pt = traj[i + 1]
        tangent = next_pt - prev_pt
        norm = np.linalg.norm(tangent)
        if norm < 1e-9:
            continue
        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]])
        for side in (-1.0, 1.0):
            for dist in offsets:
                p = traj[i] + side * dist * normal
                candidates.append((float(p[0]), float(p[1])))

    seen = set()
    unique = []
    for sx, sy in candidates:
        key = (round(sx, 3), round(sy, 3))
        if key in seen:
            continue
        seen.add(key)
        unique.append((sx, sy))
    return unique


def visualize(path: str, frames: pd.DataFrame, traj: np.ndarray,
              best: Tuple[float, float], distance_threshold: float,
              min_traj_dist: float, max_traj_dist: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(traj[:, 0], traj[:, 1], color="#2563eb", lw=2.0, label="clean trajectory")
    sc = ax.scatter(frames["x"], frames["y"], c=frames["frame_bi_smvs"], cmap="turbo",
                    s=34, edgecolors="black", linewidths=0.3, label="top Bi-SMVS frames")
    sx, sy = best
    ax.scatter([sx], [sy], marker="*", s=240, color="#dc2626", edgecolors="white", linewidths=1.0,
               label="selected spoofer")
    ax.add_patch(plt.Circle((sx, sy), distance_threshold, fill=False, ls="--", color="#dc2626", alpha=0.35))
    ax.add_patch(plt.Circle((sx, sy), min_traj_dist, fill=False, ls=":", color="#64748b", alpha=0.25))
    ax.add_patch(plt.Circle((sx, sy), max_traj_dist, fill=False, ls=":", color="#64748b", alpha=0.25))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=ax, label="Bi-SMVS")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smvs", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--spoofing-range", type=float, default=80.0)
    p.add_argument("--distance-threshold", type=float, default=15.0)
    p.add_argument("--min-traj-dist", type=float, default=8.0)
    p.add_argument("--max-traj-dist", type=float, default=14.0)
    p.add_argument("--min-trigger-ratio", type=float, default=0.30)
    p.add_argument("--max-trigger-ratio", type=float, default=0.70)
    p.add_argument("--target-trigger-ratio", type=float, default=0.50)
    p.add_argument("--no-proxy-graph", action="store_true")
    p.add_argument("--graph-source", choices=("auto", "lio_sam", "trajectory", "none"), default="auto")
    p.add_argument("--graph-dump-dir", default=None)
    p.add_argument("--proxy-node-stride", type=int, default=5)
    p.add_argument("--proxy-loop-radius", type=float, default=1.0)
    p.add_argument("--proxy-loop-min-gap", type=int, default=40)
    p.add_argument("--proxy-max-loop-edges-per-node", type=int, default=2)
    p.add_argument("--n-random", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--viz-path", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_frames(args.smvs, args.top_k)
    traj = load_traj(args.traj)
    proxy_graph = None
    graph_source_used = "none"
    if args.no_proxy_graph or args.graph_source == "none":
        proxy_graph = None
    elif args.graph_source in ("auto", "lio_sam"):
        proxy_graph = load_lio_sam_graph(args.graph_dump_dir, traj) if args.graph_dump_dir else None
        if proxy_graph is not None:
            graph_source_used = "lio_sam_graph_dump"
        elif args.graph_source == "lio_sam":
            raise SystemExit(f"No valid LIO-SAM graph dumps found in: {args.graph_dump_dir}")
    if proxy_graph is None and not args.no_proxy_graph and args.graph_source in ("auto", "trajectory"):
        proxy_graph = build_proxy_graph(
            traj,
            args.proxy_node_stride,
            args.proxy_loop_radius,
            args.proxy_loop_min_gap,
            args.proxy_max_loop_edges_per_node,
        )
        graph_source_used = "trajectory_surrogate"
    candidates = generate_candidates(
        frames, traj, args.min_traj_dist, args.max_traj_dist,
        args.spoofing_range, args.n_random, args.seed,
    )

    best_score = -1.0
    best_xy = None
    best_comp: Dict[str, float] = {}
    for sx, sy in candidates:
        score, comp = score_candidate(
            sx, sy, frames, traj,
            args.min_traj_dist, args.max_traj_dist,
            args.distance_threshold, args.spoofing_range,
            args.min_trigger_ratio, args.max_trigger_ratio,
            args.target_trigger_ratio, proxy_graph,
        )
        if score > best_score:
            best_score = score
            best_xy = (sx, sy)
            best_comp = comp

    if best_xy is None or best_score < 0:
        raise SystemExit("No feasible Bi-SMVS spoofer position found.")

    result = {
        "method": "fast_livo2_proxy_assisted_bismvs_selector",
        "proxy_note": (
            "The graph is an attacker-side proxy for route-level structural "
            "cues. If graph_source is lio_sam_graph_dump, it is a LIO-SAM "
            "surrogate graph aligned to the victim clean trajectory frame; "
            "it is not the victim SLAM's internal optimizer graph."
        ),
        "optim": {
            "spoofer_x": float(best_xy[0]),
            "spoofer_y": float(best_xy[1]),
            "score": float(best_score),
            "min_traj_dist": float(best_comp.get("min_traj_dist", -1.0)),
            "in_band": bool(best_comp.get("in_band", 0.0) > 0.5),
        },
        "score_components": best_comp,
        "params": vars(args),
        "counts": {
            "top_frames": int(len(frames)),
            "candidates": int(len(candidates)),
        },
        "proxy_graph": None if proxy_graph is None else {
            "source": graph_source_used,
            "nodes": int(len(proxy_graph["nodes"])),
            "edges": int(len(proxy_graph["edges"])),
            "chain_edges": int(proxy_graph["num_chain_edges"][0]),
            "proximity_edges": int(proxy_graph["num_loop_edges"][0]),
            "dump_files": int(proxy_graph.get("dump_files", np.asarray([0]))[0]),
            "alignment": proxy_graph.get("alignment", None),
            "node_stride": int(args.proxy_node_stride),
            "loop_radius": float(args.proxy_loop_radius),
            "loop_min_gap": int(args.proxy_loop_min_gap),
        },
        "top_bismvs_frames": [
            {
                "timestamp": float(r.timestamp),
                "x": float(r.x),
                "y": float(r.y),
                "yaw": float(r.yaw),
                "frame_bi_smvs": float(r.frame_bi_smvs),
                "vul_angle_deg": float(r.vul_angle_deg),
            }
            for r in frames.itertuples(index=False)
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if args.visualize:
        viz = args.viz_path or os.path.splitext(args.output)[0] + ".png"
        visualize(viz, frames, traj, best_xy, args.distance_threshold, args.min_traj_dist, args.max_traj_dist)

    print("[OK] proxy-assisted Bi-SMVS selector")
    print(f"  position: ({best_xy[0]:.6f}, {best_xy[1]:.6f})")
    print(f"  score   : {best_score:.6f}")
    print(f"  min_dist: {best_comp.get('min_traj_dist', -1.0):.3f} m")
    print(f"  trigger : {100.0 * best_comp.get('trigger_ratio', -1.0):.2f}%")
    print(f"  output  : {args.output}")


if __name__ == "__main__":
    main()
