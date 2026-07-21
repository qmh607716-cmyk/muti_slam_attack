#!/usr/bin/env python3
"""
Generate a paper-style three-stage overview figure using the handheld data.

The figure is intentionally close to the SLAMSpoof overview style, but the
content follows the Bi-SMVS pipeline:
  1. route data: real camera frame, LiDAR scan, and trajectory
  2. frame-wise Bi-SMVS estimation: L-Vul, visual suppression, Bi-SMVS
  3. graph-aware spoofer placement: high-Bi-SMVS frames, candidates, final site
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


def load_traj(path):
    df = pd.read_csv(path).dropna(subset=["x", "y"])
    return df


def load_vul(path):
    df = pd.read_csv(path).dropna(subset=["x", "y", "frame_bi_smvs"])
    return df


def yaw_from_quat(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def add_yaw_to_traj(df):
    out = df.copy()
    if all(c in out.columns for c in ["qx", "qy", "qz", "qw"]):
        out["yaw"] = [
            yaw_from_quat(r.qx, r.qy, r.qz, r.qw) for r in out.itertuples()
        ]
    else:
        dx = np.gradient(out["x"].to_numpy())
        dy = np.gradient(out["y"].to_numpy())
        out["yaw"] = np.arctan2(dy, dx)
    return out


def load_graph_edges(dump_dir, max_edges=1800):
    if not dump_dir or not os.path.isdir(dump_dir):
        return []
    files = sorted(
        Path(dump_dir).glob("dump_*.json"),
        key=lambda p: int(re.search(r"dump_(\d+)\.json", p.name).group(1)),
    )
    if not files:
        return []

    nodes = {}
    edge_ids = set()
    # Sample dumps across time to avoid heavy overplotting.
    step = max(1, len(files) // 90)
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
            for k in fac.get("keys", []):
                m = re.search(r"X(\d+)", str(k))
                if m:
                    keys.append(int(m.group(1)))
            if len(keys) == 2:
                edge_ids.add(tuple(sorted(keys)))
        if len(edge_ids) > max_edges:
            break

    edges = []
    for a, b in list(edge_ids)[:max_edges]:
        if a in nodes and b in nodes:
            edges.append((nodes[a], nodes[b]))
    return edges


def extract_real_sensor_frame(bag_path, target_rel_s, point_topic, image_topic,
                              max_points=26000):
    import rosbag
    from sensor_msgs import point_cloud2

    with rosbag.Bag(bag_path, "r") as bag:
        start = bag.get_start_time()
        target_abs = start + float(target_rel_s)
        best_img = (float("inf"), None)
        best_cloud = (float("inf"), None)

        for topic, msg, t in bag.read_messages(topics=[point_topic, image_topic]):
            dt = abs(t.to_sec() - target_abs)
            if topic == image_topic and dt < best_img[0]:
                best_img = (dt, msg)
            elif topic == point_topic and dt < best_cloud[0]:
                best_cloud = (dt, msg)
            if t.to_sec() > target_abs + 1.0 and best_img[1] is not None and best_cloud[1] is not None:
                break

    image = None
    if best_img[1] is not None:
        arr = np.frombuffer(best_img[1].data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    points = None
    if best_cloud[1] is not None:
        pts = []
        for p in point_cloud2.read_points(
            best_cloud[1], field_names=("x", "y", "z", "intensity"), skip_nans=True
        ):
            pts.append(p)
        if pts:
            pts = np.asarray(pts, dtype=np.float32)
            if len(pts) > max_points:
                idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
                pts = pts[idx]
            points = pts

    return image, points


def vulnerable_vectors(vul_df):
    idx = int(vul_df["frame_bi_smvs"].idxmax())
    row = vul_df.loc[idx]
    l = np.array([row.get(f"l_vul_{i:02d}", 0.0) for i in range(N_BUCKETS)], dtype=float)
    v = np.array([row.get(f"v_vul_{i:02d}", 0.0) for i in range(N_BUCKETS)], dtype=float)
    b = np.array([row.get(f"bi_vul_{i:02d}", 0.0) for i in range(N_BUCKETS)], dtype=float)
    return row, l, v, b


def generate_candidates(traj_df, vul_df, n_frames=28, distances=(15, 22, 30)):
    top = vul_df.nlargest(n_frames, "frame_bi_smvs")
    traj = add_yaw_to_traj(traj_df)
    cand = []
    for r in top.itertuples():
        idx = int(np.argmin((traj["x"].to_numpy() - r.x) ** 2 + (traj["y"].to_numpy() - r.y) ** 2))
        yaw = float(traj.iloc[idx]["yaw"])
        angle = math.radians(float(getattr(r, "vul_angle_deg", 180.0))) - math.pi
        world = yaw + angle
        for d in distances:
            cand.append((r.x + d * math.cos(world), r.y + d * math.sin(world)))
    return np.asarray(cand, dtype=float)


def draw_arrow_between(fig, ax0, ax1, color="#5A9C25"):
    p0 = ax0.get_position()
    p1 = ax1.get_position()
    x0 = p0.x1 + 0.008
    x1 = p1.x0 - 0.008
    y = (p0.y0 + p0.y1) / 2.0
    arr = FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure,
        arrowstyle="simple", mutation_scale=32,
        fc=color, ec="#24520E", lw=1.3, alpha=0.96, zorder=30,
    )
    fig.patches.append(arr)


def setup_panel_border(ax, color, lw=2.5):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           fill=False, ec=color, lw=lw, clip_on=False))


def make_figure(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = load_traj(args.traj)
    vul = load_vul(args.vul)
    smvs = pd.read_csv(args.smvs) if args.smvs and os.path.exists(args.smvs) else vul
    high_row, l_vul, v_vul, bi_vul = vulnerable_vectors(vul)
    target_rel_s = float(high_row["timestamp"])
    image, cloud = extract_real_sensor_frame(
        args.bag, target_rel_s, args.point_topic, args.image_topic
    )

    graph_edges = load_graph_edges(args.graph)
    candidates = generate_candidates(traj, vul)

    fig = plt.figure(figsize=(14.2, 4.1), dpi=220)
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.08, 1.05, 1.12],
        left=0.035, right=0.985, top=0.86, bottom=0.12, wspace=0.13,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # ── Stage 1: real data ───────────────────────────────────────────────
    ax1.set_title("1. Pre-collect route data", loc="left", fontsize=12, fontweight="bold")
    ax1.set_axis_off()
    setup_panel_border(ax1, "#03A65A")

    if image is not None:
        inset_img = ax1.inset_axes([0.04, 0.53, 0.48, 0.40])
        inset_img.imshow(image)
        inset_img.set_axis_off()
        inset_img.set_title("camera", fontsize=7, pad=1)

    inset_pc = ax1.inset_axes([0.53, 0.53, 0.43, 0.40])
    if cloud is not None:
        x, y, z, inten = cloud[:, 0], cloud[:, 1], cloud[:, 2], cloud[:, 3]
        mask = (np.abs(x) < 45) & (np.abs(y) < 45) & (z > -4) & (z < 8)
        c = np.clip(inten[mask], np.percentile(inten[mask], 5), np.percentile(inten[mask], 98))
        inset_pc.scatter(x[mask], y[mask], c=c, cmap="turbo", s=0.22, alpha=0.85, linewidths=0)
    inset_pc.set_aspect("equal")
    inset_pc.set_xlim(-35, 35)
    inset_pc.set_ylim(-35, 35)
    inset_pc.set_xticks([])
    inset_pc.set_yticks([])
    inset_pc.set_title("LiDAR scan", fontsize=7, pad=1)

    inset_route = ax1.inset_axes([0.08, 0.08, 0.84, 0.36])
    inset_route.plot(traj["x"], traj["y"], color="#0047AB", lw=1.0, alpha=0.8)
    inset_route.scatter(high_row["x"], high_row["y"], s=28, c="red", edgecolor="white", lw=0.5, zorder=4)
    inset_route.annotate("selected frame", xy=(high_row["x"], high_row["y"]), xytext=(20, 20),
                         textcoords="offset points", color="red", fontsize=7,
                         arrowprops=dict(arrowstyle="->", color="red", lw=1))
    inset_route.set_aspect("equal")
    inset_route.set_xticks([])
    inset_route.set_yticks([])
    inset_route.set_title("clean trajectory", fontsize=7, pad=1)

    ax1.text(0.05, 0.47, "raw bag: camera + LiDAR + IMU", transform=ax1.transAxes,
             fontsize=8.5, color="#004D2A", fontweight="bold")

    # ── Stage 2: frame-wise Bi-SMVS ─────────────────────────────────────
    ax2.set_title("2. Compute frame-wise Bi-SMVS", loc="left", fontsize=12, fontweight="bold")
    setup_panel_border(ax2, "#E0A800")
    angles = (np.arange(N_BUCKETS) + 0.5) * BUCKET_DEG
    l_norm = l_vul / (np.max(l_vul) + 1e-9)
    bi_norm = bi_vul / (np.max(l_vul) + 1e-9)
    suppression = np.clip(l_norm - bi_norm, 0, 1)
    ax2.bar(angles, l_norm, width=4.5, color="#2474B7", alpha=0.38, label="LiDAR SMVS")
    ax2.bar(angles, bi_norm, width=3.0, color="#D7191C", alpha=0.85, label="Bi-SMVS")
    ax2.fill_between(angles, 0, suppression, color="#2E8B57", alpha=0.22, step="mid",
                     label="visual suppression")
    dom = int(np.argmax(bi_vul))
    dom_angle = (dom + 0.5) * BUCKET_DEG
    ax2.axvspan(dom_angle - args.spoofing_range / 2.0, dom_angle + args.spoofing_range / 2.0,
                color="red", alpha=0.08, lw=0)
    ax2.annotate("high Bi-SMVS\nvulnerable sector", xy=(dom_angle, bi_norm[dom]),
                 xytext=(dom_angle + 40, 0.88), fontsize=9, color="red",
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax2.text(0.06, 0.12, "visual support suppresses\nLiDAR-only vulnerability",
             transform=ax2.transAxes, fontsize=8, color="#145A32")
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("LiDAR azimuth bucket [deg]", fontsize=8)
    ax2.set_ylabel("normalized score", fontsize=8)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.legend(loc="upper left", fontsize=7, frameon=True)

    # ── Stage 3: placement ──────────────────────────────────────────────
    ax3.set_title("3. Stage-3 graph-aware placement", loc="left", fontsize=12, fontweight="bold")
    setup_panel_border(ax3, "#0067B1")

    if graph_edges:
        segs = [[a, b] for a, b in graph_edges]
        lc = LineCollection(segs, colors="#9E9E9E", linewidths=0.35, alpha=0.18, zorder=0)
        ax3.add_collection(lc)
    ax3.plot(traj["x"], traj["y"], color="#003F88", lw=1.0, alpha=0.45, label="trajectory", zorder=1)

    show = smvs.dropna(subset=["x", "y", "frame_bi_smvs"]).iloc[::max(1, len(smvs) // 1600)]
    sc = ax3.scatter(show["x"], show["y"], c=show["frame_bi_smvs"], cmap="plasma",
                     s=7, alpha=0.68, linewidths=0, zorder=2)
    cb = fig.colorbar(sc, ax=ax3, fraction=0.035, pad=0.01)
    cb.set_label("Bi-SMVS", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    top = vul.nlargest(9, "frame_bi_smvs")
    ax3.scatter(top["x"], top["y"], s=34, facecolor="none", edgecolor="red",
                lw=1.2, zorder=5, label="high Bi-SMVS frames")
    if len(candidates):
        ax3.scatter(candidates[:, 0], candidates[:, 1], s=16, c="#FFC107",
                    edgecolor="#7A4F00", linewidth=0.25, alpha=0.65, zorder=3,
                    label="feasible candidates")

    sx, sy = args.spoofer_x, args.spoofer_y
    ax3.scatter([sx], [sy], marker="*", s=260, c="#00D26A", edgecolor="black",
                linewidth=0.9, zorder=8, label="selected spoofer")
    ax3.add_patch(Circle((sx, sy), args.distance_threshold, fill=False,
                         ls="--", lw=1.2, ec="red", alpha=0.88, zorder=4))

    for r in top.head(4).itertuples():
        ax3.plot([sx, r.x], [sy, r.y], color="#0080FF", ls="--", lw=0.8, alpha=0.7, zorder=4)
    ax3.annotate("selected roadside\nspoofing location", xy=(sx, sy), xytext=(-128, 30),
                 textcoords="offset points", fontsize=8.5, color="#004D2A", ha="right",
                 arrowprops=dict(arrowstyle="->", color="#004D2A", lw=1.1))
    ax3.text(0.04, 0.05,
             r"$\mathcal{S}(S)=O(S)+\alpha B(S)$" + "\nreachability + direction + graph persistence",
             transform=ax3.transAxes, fontsize=8, color="#003F88",
             bbox=dict(facecolor="white", edgecolor="#B0C4DE", alpha=0.85, pad=3))

    ax3.set_aspect("equal")
    pad = 25
    xmin, xmax = min(traj["x"].min(), sx) - pad, max(traj["x"].max(), sx) + pad + 18
    ymin, ymax = min(traj["y"].min(), sy) - pad, max(traj["y"].max(), sy) + pad
    ax3.set_xlim(xmin, xmax)
    ax3.set_ylim(ymin, ymax)
    ax3.set_xlabel("x [m]", fontsize=8)
    ax3.set_ylabel("y [m]", fontsize=8)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper right", fontsize=6.7, frameon=True)

    draw_arrow_between(fig, ax1, ax2)
    draw_arrow_between(fig, ax2, ax3)

    fig.suptitle("Overview of Bi-SMVS-Guided LiDAR Spoofing Against LVI-SAM",
                 fontsize=13, fontweight="bold", y=0.975)

    png = out_dir / "bismvs_stage3_overview.png"
    pdf = out_dir / "bismvs_stage3_overview.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {png}")
    print(f"[OK] saved {pdf}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/handheld.bag")
    p.add_argument("--traj", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/original/handheld_original_traj.csv")
    p.add_argument("--smvs", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/smvs/07_08_23_25_42_BiSMVS.csv")
    p.add_argument("--vul", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/vul/vul_07_08_23_25_42_BiSMVS.csv")
    p.add_argument("--graph", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/graph_dumps")
    p.add_argument("--point-topic", default="/points_raw")
    p.add_argument("--image-topic", default="/camera/image_raw/compressed")
    p.add_argument("--spoofer-x", type=float, default=31.28075677647965)
    p.add_argument("--spoofer-y", type=float, default=-102.07423272183334)
    p.add_argument("--distance-threshold", type=float, default=30.0)
    p.add_argument("--spoofing-range", type=float, default=80.0)
    p.add_argument("--output", default="/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_handheld/overview_figures")
    return p.parse_args()


if __name__ == "__main__":
    make_figure(parse_args())
