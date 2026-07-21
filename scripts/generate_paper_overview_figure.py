#!/usr/bin/env python3
"""
Generate a simplified paper overview figure aligned with the three stages in
the manuscript:

  Stage 1: Proxy factor-graph reconstruction
  Stage 2: Bi-SMVS estimation
  Stage 3: Spoofer placement selection

The figure uses real handheld trajectory / Bi-SMVS data, but keeps the visual
language conceptual enough for an overview figure.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D


N_BUCKETS = 72
BUCKET_DEG = 5.0


def yaw_from_quat(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def add_yaw(df):
    out = df.copy()
    if all(c in out.columns for c in ["qx", "qy", "qz", "qw"]):
        out["yaw"] = [yaw_from_quat(r.qx, r.qy, r.qz, r.qw) for r in out.itertuples()]
    else:
        out["yaw"] = np.arctan2(np.gradient(out["y"]), np.gradient(out["x"]))
    return out


def load_graph_edges(dump_dir, max_edges=900):
    if not dump_dir or not os.path.isdir(dump_dir):
        return []
    files = sorted(
        Path(dump_dir).glob("dump_*.json"),
        key=lambda p: int(re.search(r"dump_(\d+)\.json", p.name).group(1)),
    )
    nodes = {}
    edge_ids = set()
    if not files:
        return []
    step = max(1, len(files) // 70)
    for fp in files[::step]:
        try:
            obj = json.load(open(fp))
        except Exception:
            continue
        for n in obj.get("nodes", []):
            if all(k in n for k in ["id", "x", "y"]):
                nodes[int(n["id"])] = (float(n["x"]), float(n["y"]))
        for fac in obj.get("factors", []):
            keys = []
            for key in fac.get("keys", []):
                m = re.search(r"X(\d+)", str(key))
                if m:
                    keys.append(int(m.group(1)))
            if len(keys) == 2:
                edge_ids.add(tuple(sorted(keys)))
        if len(edge_ids) >= max_edges:
            break
    edges = []
    for a, b in list(edge_ids)[:max_edges]:
        if a in nodes and b in nodes:
            edges.append((nodes[a], nodes[b]))
    return edges


def extract_camera_frame(bag_path, rel_s, topic):
    import rosbag
    with rosbag.Bag(bag_path, "r") as bag:
        start = bag.get_start_time()
        target = start + float(rel_s)
        best = (float("inf"), None)
        for tpc, msg, t in bag.read_messages(topics=[topic]):
            dt = abs(t.to_sec() - target)
            if dt < best[0]:
                best = (dt, msg)
            if t.to_sec() > target + 1.0 and best[1] is not None:
                break
    if best[1] is None:
        return None
    arr = np.frombuffer(best[1].data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_lidar_scan(bag_path, rel_s, topic, max_points=18000):
    import rosbag
    from sensor_msgs import point_cloud2
    with rosbag.Bag(bag_path, "r") as bag:
        start = bag.get_start_time()
        target = start + float(rel_s)
        best = (float("inf"), None)
        for tpc, msg, t in bag.read_messages(topics=[topic]):
            dt = abs(t.to_sec() - target)
            if dt < best[0]:
                best = (dt, msg)
            if t.to_sec() > target + 1.0 and best[1] is not None:
                break
    if best[1] is None:
        return None
    pts = []
    for p in point_cloud2.read_points(best[1], field_names=("x", "y", "z", "intensity"),
                                      skip_nans=True):
        pts.append(p)
    if not pts:
        return None
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    return pts


def build_accumulated_lidar_map(bag_path, traj, point_topic, cache_path,
                                stride=90, max_frames=155, points_per_frame=520):
    """Accumulate sparse LiDAR scans into a map-like top-down point cloud."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path)
        return data["xy"], data["z"]

    import rosbag
    from sensor_msgs import point_cloud2

    traj_yaw = add_yaw(traj).sort_values("time")
    tt = traj_yaw["time"].to_numpy(dtype=float)
    tx = traj_yaw["x"].to_numpy(dtype=float)
    ty = traj_yaw["y"].to_numpy(dtype=float)
    yaw = traj_yaw["yaw"].to_numpy(dtype=float)

    xy_chunks = []
    z_chunks = []
    used = 0
    seen = 0
    rng = np.random.default_rng(7)

    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[point_topic]):
            if seen % stride != 0:
                seen += 1
                continue
            seen += 1
            t_abs = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else t.to_sec()
            if t_abs < tt[0] or t_abs > tt[-1]:
                continue

            pts = []
            for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = p
                if 1.5 < (x * x + y * y) ** 0.5 < 45.0 and -3.0 < z < 8.0:
                    pts.append((x, y, z))
            if not pts:
                continue
            pts = np.asarray(pts, dtype=np.float32)
            if len(pts) > points_per_frame:
                idx = rng.choice(len(pts), size=points_per_frame, replace=False)
                pts = pts[idx]

            px = np.interp(t_abs, tt, tx)
            py = np.interp(t_abs, tt, ty)
            pyaw = np.interp(t_abs, tt, yaw)
            c, s = math.cos(pyaw), math.sin(pyaw)
            wx = px + c * pts[:, 0] - s * pts[:, 1]
            wy = py + s * pts[:, 0] + c * pts[:, 1]
            xy_chunks.append(np.column_stack([wx, wy]))
            z_chunks.append(pts[:, 2])
            used += 1
            if used >= max_frames:
                break

    if xy_chunks:
        xy = np.vstack(xy_chunks)
        z = np.concatenate(z_chunks)
    else:
        xy = traj[["x", "y"]].to_numpy(dtype=float)
        z = np.zeros(len(xy), dtype=float)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, xy=xy, z=z)
    return xy, z


def top_frame(vul):
    idx = int(vul["frame_bi_smvs"].idxmax())
    row = vul.loc[idx]
    l = np.array([row.get(f"l_vul_{i:02d}", 0.0) for i in range(N_BUCKETS)], dtype=float)
    v = np.array([row.get(f"v_vul_{i:02d}", 0.0) for i in range(N_BUCKETS)], dtype=float)
    b = np.array([row.get(f"bi_vul_{i:02d}", 0.0) for i in range(N_BUCKETS)], dtype=float)
    return row, l, v, b


def candidate_points(traj, vul, n_frames=18, distances=(15, 22, 30)):
    traj_yaw = add_yaw(traj)
    top = vul.nlargest(n_frames, "frame_bi_smvs")
    pts = []
    for r in top.itertuples():
        idx = int(np.argmin((traj_yaw["x"].to_numpy() - r.x) ** 2 +
                            (traj_yaw["y"].to_numpy() - r.y) ** 2))
        yaw = float(traj_yaw.iloc[idx]["yaw"])
        local = math.radians(float(getattr(r, "vul_angle_deg", 180.0))) - math.pi
        world = yaw + local
        for d in distances:
            pts.append((r.x + d * math.cos(world), r.y + d * math.sin(world)))
    return np.asarray(pts, dtype=float)


def panel_box(ax, color):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           fill=False, ec=color, lw=2.2, clip_on=False))


def oblique_project(x, y, z=0.0):
    """Project world/map points to an oblique map view, closer to the paper figure."""
    return x + 0.10 * y, 0.34 * y + 2.2 * z


def arrow_between(fig, ax0, ax1):
    p0 = ax0.get_position()
    p1 = ax1.get_position()
    y = (p0.y0 + p0.y1) / 2.0
    fig.patches.append(FancyArrowPatch(
        (p0.x1 + 0.01, y), (p1.x0 - 0.01, y),
        transform=fig.transFigure, arrowstyle="simple",
        mutation_scale=32, fc="#5A9C25", ec="#24520E",
        lw=1.2, alpha=0.95, zorder=50,
    ))


def make_clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def simplified_route_graph(traj, step=95):
    pts = traj[["x", "y"]].to_numpy(dtype=float)
    idx = np.arange(0, len(pts), step)
    if idx[-1] != len(pts) - 1:
        idx = np.r_[idx, len(pts) - 1]
    nodes = pts[idx]
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    # Add a few real route-revisit links to communicate loop/long-range support
    # without drawing a dense graph.
    for i in range(0, len(nodes), 7):
        d = np.linalg.norm(nodes - nodes[i], axis=1)
        candidates = np.where((d < 18.0) & (np.abs(np.arange(len(nodes)) - i) > 8))[0]
        if len(candidates):
            j = int(candidates[0])
            edges.append((nodes[i], nodes[j]))
    return nodes, edges


def draw_stage1(ax, traj, graph_edges, map_xy, map_z):
    ax.set_title("1. Obtain route map and proxy graph", loc="left",
                 fontsize=12, fontweight="bold")
    panel_box(ax, "#03A65A")
    make_clean_axes(ax)

    map_ax = ax.inset_axes([0.055, 0.14, 0.89, 0.72])
    if map_xy is not None and len(map_xy):
        lo, hi = np.percentile(map_z, [3, 97])
        c = np.clip(map_z, lo, hi)
        mx, my = oblique_project(map_xy[:, 0], map_xy[:, 1], map_z)
        map_ax.scatter(mx, my, c=c, cmap="turbo",
                       s=0.16, alpha=0.62, linewidths=0, zorder=0)
    nodes, simple_edges = simplified_route_graph(traj)
    projected_edges = []
    for a, b in simple_edges:
        au, av = oblique_project(a[0], a[1], 0.0)
        bu, bv = oblique_project(b[0], b[1], 0.0)
        projected_edges.append([(au, av), (bu, bv)])
    lc = LineCollection(projected_edges,
                        colors="#222222", linewidths=0.65,
                        alpha=0.25, zorder=1)
    map_ax.add_collection(lc)
    tx, ty = oblique_project(traj["x"].to_numpy(), traj["y"].to_numpy(), 0.0)
    map_ax.plot(tx, ty, color="#D7191C", lw=1.8,
                alpha=0.92, zorder=3)
    nx, ny = oblique_project(nodes[:, 0], nodes[:, 1], 0.0)
    map_ax.scatter(nx, ny, s=8, c="#6E6E6E",
                   alpha=0.6, zorder=4)
    sx0, sy0 = oblique_project(traj.iloc[0]["x"], traj.iloc[0]["y"], 0.0)
    sx1, sy1 = oblique_project(traj.iloc[-1]["x"], traj.iloc[-1]["y"], 0.0)
    map_ax.scatter(sx0, sy0, s=30, c="#03A65A",
                   edgecolor="white", lw=0.7, zorder=5)
    map_ax.scatter(sx1, sy1, s=28, c="#D7191C",
                   edgecolor="white", lw=0.7, zorder=5)
    make_clean_axes(map_ax)
    map_ax.set_facecolor("#083B4C")
    map_ax.patch.set_alpha(0.12)
    map_ax.text(0.05, 0.08, "trajectory", color="#D7191C",
                transform=map_ax.transAxes, fontsize=9.5, fontweight="bold")
    map_ax.annotate("", xy=(0.27, 0.19), xytext=(0.12, 0.10),
                    xycoords=map_ax.transAxes, textcoords=map_ax.transAxes,
                    arrowprops=dict(arrowstyle="-", color="#D7191C", lw=1.3))

    ax.text(0.07, 0.055,
            "LiDAR map + clean trajectory\nProxy graph provides route-level structure",
            transform=ax.transAxes, fontsize=8.2, color="#004D2A",
            fontweight="bold")


def draw_direction_wheel(ax, l_vul, v_vul, bi_vul):
    ax.set_aspect("equal")
    make_clean_axes(ax)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)

    dom = int(np.argmax(bi_vul))
    dom_center = (dom + 0.5) * BUCKET_DEG
    # Matplotlib wedge: 0 deg is +x, counter-clockwise.
    wedge = Wedge((0, 0), 1.0, dom_center - 40, dom_center + 40,
                  facecolor="#D7191C", alpha=0.18,
                  edgecolor="#D7191C", lw=1.8, ls="--")
    ax.add_patch(wedge)

    # Draw a compact directional score ring from real Bi-SMVS.
    b_norm = bi_vul / (np.max(bi_vul) + 1e-9)
    for i, val in enumerate(b_norm):
        if val < 0.12:
            continue
        ang = math.radians((i + 0.5) * BUCKET_DEG)
        r0, r1 = 0.62, 0.62 + 0.35 * val
        ax.plot([r0 * math.cos(ang), r1 * math.cos(ang)],
                [r0 * math.sin(ang), r1 * math.sin(ang)],
                color="#D7191C", alpha=0.62, lw=1.5)

    ax.add_patch(Circle((0, 0), 0.60, fill=False, ec="#444", lw=1.0))
    ax.arrow(0, 0, 0.48, 0, width=0.015, head_width=0.08,
             head_length=0.08, color="#0047AB", length_includes_head=True)
    ax.text(0.54, 0.03, "LiDAR", fontsize=7, color="#0047AB")

    # Camera FOV cue.
    ax.add_patch(Wedge((0, 0), 0.48, -35, 35,
                       facecolor="#2E8B57", alpha=0.18,
                       edgecolor="#2E8B57", lw=1.2))
    ax.text(-0.65, -0.82, "green: visual support\nred: retained vulnerability",
            fontsize=7.2, color="#333")


def draw_stage2(ax, traj, vul, l_vul, v_vul, bi_vul):
    ax.set_title("2. Color route by frame-wise Bi-SMVS", loc="left",
                 fontsize=12, fontweight="bold")
    panel_box(ax, "#E0A800")
    make_clean_axes(ax)

    traj_ax = ax.inset_axes([0.06, 0.15, 0.82, 0.72])
    pts = vul.dropna(subset=["x", "y", "frame_bi_smvs"]).copy()
    sample = pts.iloc[::max(1, len(pts) // 2600)]
    xy = sample[["x", "y"]].to_numpy(dtype=float)
    if len(xy) > 1:
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        vals = sample["frame_bi_smvs"].to_numpy(dtype=float)[:-1]
        lc = LineCollection(segments, cmap="turbo", linewidths=2.8, alpha=0.95)
        lc.set_array(vals)
        traj_ax.add_collection(lc)
        cb = plt.colorbar(lc, ax=traj_ax, fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=5.5)
        cb.set_label("Bi-SMVS", fontsize=6.5)
    else:
        traj_ax.plot(traj["x"], traj["y"], color="#D7191C", lw=1.5)

    high = pts.nlargest(1, "frame_bi_smvs").iloc[0]
    low_candidates = pts.nsmallest(max(8, len(pts) // 30), "frame_bi_smvs")
    low = low_candidates.iloc[len(low_candidates) // 2]
    traj_ax.scatter([high["x"]], [high["y"]], s=90, facecolor="none",
                    edgecolor="#D7191C", lw=2.0, zorder=5)
    traj_ax.scatter([low["x"]], [low["y"]], s=90, facecolor="none",
                    edgecolor="#0067B1", lw=2.0, zorder=5)
    traj_ax.annotate("High Bi-SMVS\nvulnerable", xy=(high["x"], high["y"]),
                     xytext=(-55, 30), textcoords="offset points",
                     color="#D7191C", fontsize=8.5, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#D7191C", lw=1.4))
    traj_ax.annotate("Low Bi-SMVS\nmore robust", xy=(low["x"], low["y"]),
                     xytext=(22, -42), textcoords="offset points",
                     color="#0067B1", fontsize=8.2, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#0067B1", lw=1.4))
    traj_ax.set_aspect("equal")
    make_clean_axes(traj_ax)

    ax.text(0.06, 0.055,
            "Bi-SMVS colors the route after visual-support filtering.",
            transform=ax.transAxes, fontsize=7.8, color="#5A3B00",
            fontweight="bold")


def draw_stage3(ax, traj, vul, candidates, sx, sy, radius):
    ax.set_title("3. Select spoofer from vulnerable route geometry", loc="left",
                 fontsize=12, fontweight="bold")
    panel_box(ax, "#0067B1")
    make_clean_axes(ax)

    show_ax = ax.inset_axes([0.05, 0.13, 0.64, 0.75])
    show_ax.plot(traj["x"], traj["y"], color="#0047AB", lw=1.15,
                 alpha=0.62, zorder=1)
    top = vul.nlargest(9, "frame_bi_smvs")
    show_ax.scatter(top["x"], top["y"], s=36, facecolor="white",
                    edgecolor="#D7191C", lw=1.5, zorder=4)
    if len(candidates):
        show_ax.scatter(candidates[:, 0], candidates[:, 1], s=19,
                        c="#FFC107", edgecolor="#7A4F00",
                        linewidth=0.25, alpha=0.75, zorder=3)
    show_ax.scatter([sx], [sy], marker="*", s=260, c="#00D26A",
                    edgecolor="black", lw=0.9, zorder=7)
    show_ax.add_patch(Circle((sx, sy), radius, fill=False,
                             ec="#D7191C", lw=1.3, ls="--", zorder=5))
    for r in top.head(4).itertuples():
        show_ax.plot([sx, r.x], [sy, r.y], color="#0072CE",
                     lw=0.85, ls="--", alpha=0.75, zorder=2)
    show_ax.set_aspect("equal")
    show_ax.grid(True, alpha=0.16)
    show_ax.set_xticks([])
    show_ax.set_yticks([])
    show_ax.set_xlabel("")
    show_ax.set_ylabel("")

    legend_items = [
        Line2D([0], [0], color="#0047AB", lw=1.2, label="trajectory"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#D7191C", lw=0, label="high Bi-SMVS frames"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#FFC107",
               markeredgecolor="#7A4F00", lw=0, label="feasible candidates"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#00D26A",
               markeredgecolor="black", markersize=12, lw=0, label="selected spoofer"),
    ]
    show_ax.legend(handles=legend_items, loc="upper right", fontsize=6.7, frameon=True)

    detail_ax = ax.inset_axes([0.72, 0.18, 0.24, 0.64])
    detail_ax.set_xlim(0, 1)
    detail_ax.set_ylim(0, 1)
    make_clean_axes(detail_ax)
    detail_ax.text(0.50, 0.96, "placement principle", ha="center",
                   va="top", fontsize=7.8, color="#003F88", fontweight="bold")
    cx, cy = 0.50, 0.55
    detail_ax.scatter([cx], [cy], s=190, marker="*", c="#00D26A",
                      edgecolor="black", zorder=5)
    score_terms = [
        ("reachability", "#2474B7", 90, 0.30),
        ("directional\nalignment", "#D7191C", 210, 0.31),
        ("graph\npersistence", "#7B3F98", 330, 0.31),
    ]
    for txt, col, deg, rr in score_terms:
        ang = math.radians(deg)
        px, py = cx + rr * math.cos(ang), cy + rr * math.sin(ang)
        detail_ax.plot([cx, px], [cy, py], color=col, lw=1.35, alpha=0.75)
        detail_ax.scatter([px], [py], s=92, c=col, edgecolor="white", lw=0.7, zorder=4)
        detail_ax.text(px, py - 0.10 if deg == 90 else py + 0.07,
                       txt, ha="center", va="center", fontsize=6.8, color=col)
    detail_ax.text(0.50, 0.10, r"$\mathcal{S}(S)=O(S)+\alpha B(S)$",
                   ha="center", fontsize=8.1, color="#003F88")

    ax.text(0.05, 0.030,
            "Stage 3 selects the highest-score feasible roadside location.",
            transform=ax.transAxes, fontsize=7.8, color="#003F88",
            fontweight="bold")


def make_figure(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    traj = pd.read_csv(args.traj).dropna(subset=["x", "y"])
    vul = pd.read_csv(args.vul).dropna(subset=["x", "y", "frame_bi_smvs"])
    row, l_vul, v_vul, bi_vul = top_frame(vul)
    cache_path = Path(args.output) / "accumulated_lidar_map_cache.npz"
    map_xy, map_z = build_accumulated_lidar_map(
        args.bag, traj, args.point_topic, cache_path,
        stride=args.map_stride, max_frames=args.map_frames,
        points_per_frame=args.map_points_per_frame,
    )
    graph_edges = load_graph_edges(args.graph)
    cand = candidate_points(traj, vul)

    fig = plt.figure(figsize=(14.8, 4.25), dpi=220)
    gs = fig.add_gridspec(1, 3, left=0.035, right=0.985, top=0.84, bottom=0.12,
                          wspace=0.13, width_ratios=[1.03, 1.03, 1.06])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    draw_stage1(ax1, traj, graph_edges, map_xy, map_z)
    draw_stage2(ax2, traj, vul, l_vul, v_vul, bi_vul)
    draw_stage3(ax3, traj, vul, cand, args.spoofer_x, args.spoofer_y,
                args.distance_threshold)

    arrow_between(fig, ax1, ax2)
    arrow_between(fig, ax2, ax3)

    fig.suptitle("Overview of the Three-Stage Bi-SMVS Attack Planning Pipeline",
                 fontsize=13, fontweight="bold", y=0.965)

    png = out / "bismvs_paper_overview.png"
    pdf = out / "bismvs_paper_overview.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {png}")
    print(f"[OK] saved {pdf}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/handheld.bag")
    p.add_argument("--traj", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_original_traj.csv")
    p.add_argument("--vul", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/vul/vul_07_08_23_25_42_BiSMVS.csv")
    p.add_argument("--graph", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/graph_dumps")
    p.add_argument("--image-topic", default="/camera/image_raw/compressed")
    p.add_argument("--point-topic", default="/points_raw")
    p.add_argument("--map-stride", type=int, default=90)
    p.add_argument("--map-frames", type=int, default=155)
    p.add_argument("--map-points-per-frame", type=int, default=520)
    p.add_argument("--spoofer-x", type=float, default=31.28075677647965)
    p.add_argument("--spoofer-y", type=float, default=-102.07423272183334)
    p.add_argument("--distance-threshold", type=float, default=30.0)
    p.add_argument("--output", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/overview_figures")
    return p.parse_args()


if __name__ == "__main__":
    make_figure(parse_args())
