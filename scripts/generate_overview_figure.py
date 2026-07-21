#!/usr/bin/env python3
"""
generate_overview_figure.py
============================

Generate all material for the Bi-SMVS overview figure (IEEE paper-ready).

Pipeline
--------
Stage 1 → Proxy factor-graph reconstruction
Stage 2 → Bi-SMVS estimation (LiDAR + Visual → Bi-Vul vector)
Stage 3 → Graph-aware spoofer placement + attack generation

Outputs
--------
  fig1 _stage1_proxy_graph.pdf   — 2D trajectory + graph edges + node labels
  fig2_stage2_bivul_vectors.pdf   — Polar bar charts: L-Vul / V-Vul / Bi-Vul
  fig3_stage3_placement_map.pdf  — Candidate CMA-ES heatmap + final location
  fig4_stage3_attack_demo.pdf     — Spoofed point cloud (attack window) + clean
  fig5_overview_flowchart.pdf     — System pipeline flowchart

Usage
-----
  python3 generate_overview_figure.py \\
      --smvs   /path/to/smvs.csv    \\
      --vul    /path/to/vul.csv    \\
      --traj   /path/to/traj.csv   \\
      --graph  /path/to/graph_dir  \\
      --spoofer-x -18.5 --spoofer-y 70.4 \\
      --output  ./overview_figures/

Dependencies
-----------
  numpy, pandas, matplotlib, scipy
  (optional: networkx for Stage-1 graph drawing)

Author: Menghao Qu
"""

import os
import sys
import argparse
import math
import json
import glob
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge, Circle, Arc
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
N_BUCKETS = 72
STEP_DEG = 5.0
SPOOFING_RANGE = 80.0      # degrees
DISTANCE_THRESH = 30.0     # metres
WALL_DIST = 15.0          # metres


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Proxy factor-graph reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def load_graph_data(dump_dir):
    """Merge all graph dumps and compute node positions + edge list."""
    dump_files = sorted(
        glob.glob(os.path.join(dump_dir, "dump_*.json")),
        key=lambda p: int(re.search(r'dump_(\d+)\.json', p).group(1)),
    )
    if not dump_files:
        return None, None, None

    all_nodes = {}
    all_factors = []
    seen = set()

    for path in dump_files:
        with open(path) as fh:
            d = json.load(fh)
        for node in d.get("nodes", []):
            nid = int(node["id"])
            all_nodes[nid] = node
        for fac in d.get("factors", []):
            keys = tuple(sorted(fac.get("keys", [])))
            src = fac.get("source", "other")
            sig = (keys, src)
            if sig not in seen:
                seen.add(sig)
                all_factors.append(fac)

    # GPS alignment fallback
    parent = os.path.dirname(dump_dir.rstrip(os.sep))
    original_dir = os.path.join(parent, "original")
    gps_x_aligned = None
    node_nids = sorted(all_nodes.keys())
    n_total = max(int(node_nids[-1]) + 1, 1)

    gps_files = []
    if os.path.isdir(original_dir):
        gps_files = sorted([
            os.path.join(original_dir, f)
            for f in os.listdir(original_dir)
            if "_gps" in f.lower() and f.endswith(".csv")
        ])

    for gps_path in gps_files:
        try:
            gdf = pd.read_csv(gps_path)
            gdf.columns = gdf.columns.str.strip()
            gdf = gdf.dropna()
            gt = gdf["time"].values.astype(float)
            gx = gdf["x"].values.astype(float)
            gy = gdf["y"].values.astype(float)
            bag_dur = float(gt[-1]) if len(gt) > 1 else float(n_total)
            node_times = np.clip(
                np.arange(n_total) * bag_dur / max(n_total - 1, 1),
                gt.min(), gt.max()
            )
            gps_x_aligned = np.interp(node_times, gt, gx)
            gps_y_aligned = np.interp(node_times, gt, gy)
            break
        except Exception:
            pass

    if gps_x_aligned is None:
        gps_x_aligned = np.array([all_nodes.get(n, {}).get("x", 0.0) for n in node_nids])
        gps_y_aligned = np.array([all_nodes.get(n, {}).get("y", 0.0) for n in node_nids])

    # Build node dict
    node_x = np.array([gps_x_aligned[i] if i < len(gps_x_aligned) else 0.0
                        for i, n in enumerate(node_nids)])
    node_y = np.array([gps_y_aligned[i] if i < len(gps_y_aligned) else 0.0
                        for i, n in enumerate(node_nids)])

    # Build edges
    edges = []
    for fac in all_factors:
        keys = []
        for k in fac.get("keys", []):
            m = re.search(r'X(\d+)', k)
            if m:
                keys.append(int(m.group(1)))
        if len(keys) == 2:
            a, b = keys
            if a in all_nodes and b in all_nodes:
                edges.append((a, b))

    return node_nids, node_x, node_y, edges


def draw_stage1_proxy_graph(ax, node_nids, node_x, node_y, edges,
                             traj_x, traj_y, spoofer_x, spoofer_y):
    """Draw proxy pose graph with edges coloured by factor type."""
    if node_nids is None:
        # Fallback: just draw trajectory
        ax.plot(traj_x, traj_y, 'b-', lw=1.5, alpha=0.6, label='Trajectory')
        return

    # colour edges
    edge_colour = '#888888'
    for a, b in edges:
        try:
            ia = node_nids.index(a)
            ib = node_nids.index(b)
            ax.plot([node_x[ia], node_x[ib]],
                    [node_y[ia], node_y[ib]],
                    color=edge_colour, lw=0.8, alpha=0.4, zorder=2)
        except (ValueError, IndexError):
            pass

    # Draw nodes
    ax.scatter(node_x, node_y, s=12, c='#2196F3', zorder=5, alpha=0.7)

    # Draw trajectory overlay
    if traj_x is not None:
        ax.plot(traj_x, traj_y, 'b-', lw=1.5, alpha=0.5, zorder=1, label='Trajectory')

    # Spoofer
    if spoofer_x is not None:
        ax.scatter(spoofer_x, spoofer_y, marker='X', s=250,
                    c='red', edgecolors='black', lw=1.2, zorder=10,
                    label=f'Spoofer')
        circle = Circle((spoofer_x, spoofer_y),
                         DISTANCE_THRESH, fill=False,
                         color='red', ls='--', lw=1.5, zorder=4,
                         label=f'Trigger zone (r={DISTANCE_THRESH}m)')
        ax.add_patch(circle)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Stage 1 — Proxy Factor-Graph Reconstruction',
                 fontweight='bold', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Bi-SMVS estimation (polar bar charts)
# ─────────────────────────────────────────────────────────────────────────────

def load_frame_data(smvs_path, vul_path):
    """Load a representative frame's L-Vul, V-Vul, Bi-Vul vectors."""
    vul_df = pd.read_csv(vul_path)
    l_cols = [f"l_vul_{i:02d}" for i in range(N_BUCKETS)]
    v_cols = [f"v_vul_{i:02d}" for i in range(N_BUCKETS)]
    b_cols = [f"bi_vul_{i:02d}" for i in range(N_BUCKETS)]

    # Drop first row (origin placeholder) and pick a high-bi-smvs frame
    vul_df = vul_df.dropna(subset=l_cols)
    bi_cols = [f"bi_vul_{i:02d}" for i in range(N_BUCKETS)]
    vul_df["bi_sum"] = vul_df[bi_cols].sum(axis=1)
    vul_df = vul_df.sort_values("bi_sum", ascending=False)

    # Pick top-3 diverse frames (one near start, one mid, one late)
    if len(vul_df) < 3:
        row = vul_df.iloc[len(vul_df) // 2]
    else:
        rows = [vul_df.iloc[0], vul_df.iloc[len(vul_df)//2], vul_df.iloc[-1]]
        row = rows[0]  # use the most vulnerable one for the overview

    l_vul = np.array([float(row[c]) for c in l_cols], dtype=float)
    v_vul = np.array([float(row[c]) for c in v_cols], dtype=float)
    b_vul = np.array([float(row[c]) for c in b_cols], dtype=float)

    return l_vul, v_vul, b_vul


def draw_polar_bar(ax, values, colour, label, alpha=0.7):
    """Draw a polar bar chart (bar per 5° bucket, 72 bars = 360°)."""
    angles = np.linspace(0, 360, N_BUCKETS, endpoint=False)
    # convert to radians for polar
    rad = np.radians(90 - angles)  # rotate so 0° is up
    widths = np.radians(360 / N_BUCKETS) * 0.9

    # normalise to [0, 1] per vector for visual comparison
    vmax = values.max() if values.max() > 0 else 1.0
    heights = values / vmax * 0.8  # scale so tallest bar = 0.8 (out of radius 1)

    bars = ax.bar(rad, heights, width=widths, bottom=0.0,
                  color=colour, alpha=alpha, edgecolor='none')
    ax.set_ylim(0, 1.1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(label, pad=20, fontsize=10, fontweight='bold', y=1.05)


def draw_stage2_bivul_vectors(ax_L, ax_V, ax_B, l_vul, v_vul, b_vul):
    """Draw three polar bar charts: L-Vul, V-Vul, Bi-Vul."""
    draw_polar_bar(ax_L, l_vul, '#1976D2', 'L-Vul (LiDAR-only)', alpha=0.8)
    draw_polar_bar(ax_V, v_vul, '#388E3C', 'V-Vul (Visual support)', alpha=0.8)
    draw_polar_bar(ax_B, b_vul, '#D32F2F', 'Bi-SMVS (Fused)', alpha=0.8)

    # Annotate the dominant direction with an arrow (skip if ax is None)
    for ax, vul_arr in [(ax_L, l_vul), (ax_V, v_vul), (ax_B, b_vul)]:
        if ax is None:
            continue
        dominant_idx = int(np.argmax(vul_arr))
        dom_angle = (dominant_idx + 0.5) * STEP_DEG
        rad_dom = np.radians(90 - dom_angle)
        vmax = vul_arr.max()
        ax.annotate('',
                    xy=(rad_dom, min(vul_arr.max() / (vul_arr.max() + 1e-9) * 0.8 + 0.1, 1.0)),
                    xytext=(rad_dom, 0.0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Graph-aware spoofer placement (score heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def load_trajectory(traj_path):
    df = pd.read_csv(traj_path)
    for col in ["x", "y"]:
        if col not in df.columns:
            raise SystemExit(f"Trajectory CSV missing column: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    if "time" in df.columns:
        t = df["time"].values - df["time"].values[0]
    else:
        t = np.arange(len(df), dtype=float)
    return df["x"].values, df["y"].values, t


def draw_stage3_placement_map(ax, traj_x, traj_y, frames_df, spoofer_x, spoofer_y,
                               candidate_x=None, candidate_y=None):
    """Draw trajectory coloured by Bi-SMVS + score heatmap + candidate scatter."""
    if frames_df is not None:
        scatter = ax.scatter(
            frames_df["x"], frames_df["y"],
            c=frames_df["frame_bi_smvs"],
            cmap='plasma', s=30, alpha=0.7, zorder=5,
            label='Keyframes (Bi-SMVS colour)'
        )
        plt.colorbar(scatter, ax=ax, label='Bi-SMVS', shrink=0.8)

    ax.plot(traj_x, traj_y, 'b-', lw=1.5, alpha=0.4, zorder=1, label='Trajectory')
    ax.scatter(traj_x[0], traj_y[0], marker='o', s=80, c='lime',
                edgecolors='black', zorder=10, label='Start')
    ax.scatter(traj_x[-1], traj_y[-1], marker='s', s=80, c='red',
                edgecolors='black', zorder=10, label='End')

    # Candidate locations
    if candidate_x is not None and len(candidate_x) > 0:
        ax.scatter(candidate_x, candidate_y, s=20, c='orange', alpha=0.4,
                   zorder=4, label=f'CMA-ES candidates (n={len(candidate_x)})')

    # Spoofer
    ax.scatter(spoofer_x, spoofer_y, marker='*', s=400,
               c='lime', edgecolors='black', lw=1.5, zorder=12,
               label=f'Bi-SMVS spoofer')
    ax.add_patch(Circle((spoofer_x, spoofer_y), DISTANCE_THRESH,
                          fill=False, color='red', ls='--', lw=1.5, zorder=6))

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Stage 3 — Graph-Aware Spoofer Placement',
                 fontweight='bold', fontsize=11)
    ax.legend(loc='upper right', fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 inset: Bi-SMVS vector comparison (rectangular bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def draw_bivul_comparison(ax_top, ax_bot, l_vul, b_vul):
    """Compare LiDAR-only vs Bi-SMVS: show how visual suppresses the score."""
    k = np.arange(N_BUCKETS)
    angles = (k + 0.5) * STEP_DEG

    # Normalise
    l_norm = l_vul / (l_vul.max() + 1e-9)
    b_norm = b_vul / (l_vul.max() + 1e-9)

    ax_top.bar(angles, l_norm, width=4.0, color='#1976D2', alpha=0.7,
               label='L-Vul')
    ax_top.bar(angles, b_norm, width=2.0, color='#D32F2F', alpha=0.7,
               label='Bi-SMVS')
    ax_top.set_ylabel('Normalised L-Vul / Bi-SMVS')
    ax_top.set_xlim(0, 360)
    ax_top.legend(fontsize=8)
    ax_top.set_title('Directional Vulnerability Comparison',
                      fontweight='bold', fontsize=10)
    ax_top.grid(True, alpha=0.3, axis='y')

    # Suppression map: L-Vul - Bi-SMVS
    supp = l_norm - b_norm
    ax_bot.bar(angles, supp, width=4.0, color='#388E3C', alpha=0.7,
               label='Visual suppression')
    ax_bot.set_xlabel('Horizontal azimuth [°]')
    ax_bot.set_ylabel('Suppression = L − Bi')
    ax_bot.set_xlim(0, 360)
    ax_bot.legend(fontsize=8)
    ax_bot.set_title('Visual Corrective Support (where L-Vul is suppressed)',
                      fontweight='bold', fontsize=10)
    ax_bot.grid(True, alpha=0.3, axis='y')


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Attack point-cloud illustration (clean vs spoofed)
# ─────────────────────────────────────────────────────────────────────────────

def draw_attack_pointcloud_demo(ax, raw_cloud, spoofed_cloud,
                                 center_deg, half_range,
                                 wall_dist):
    """Illustrate clean vs spoofed point cloud in polar view."""
    def polar_points(xyz):
        r = np.linalg.norm(xyz[:, :2], axis=1)
        theta = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
        return r, theta

    if raw_cloud is not None and len(raw_cloud) > 0:
        r_raw, t_raw = polar_points(raw_cloud)
        ax.scatter(t_raw, r_raw, s=0.5, c='#90CAF9', alpha=0.3,
                   label='Clean points', zorder=1)

    if spoofed_cloud is not None and len(spoofed_cloud) > 0:
        r_spf, t_spf = polar_points(spoofed_cloud)
        ax.scatter(t_spf, r_spf, s=0.5, c='#EF9A9A', alpha=0.3,
                   label='Spoofed points', zorder=2)

    # Mark attack window
    w = Wedge((0, 0), wall_dist * 1.2,
              center_deg - half_range,
              center_deg + half_range,
              facecolor='red', alpha=0.1, edgecolor='red',
              ls='--', lw=1.5, zorder=3, label='Attack window')

    # Mark spoofed wall
    wall_theta = np.linspace(center_deg - half_range,
                              center_deg + half_range, 100) % 360.0
    wall_r = np.full_like(wall_theta, wall_dist, dtype=float)
    ax.plot(wall_theta, wall_r, 'r-', lw=2.0, zorder=5, label=f'False wall (r={wall_dist}m)')

    ax.set_xlabel('Azimuth θ [°]')
    ax.set_ylabel('Range r [m]')
    ax.set_xlim(-10, 370)
    ax.set_ylim(0, 50)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_title('Stage 3 — Attack Point-Cloud Modification',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline flowchart (Stage 1 → Stage 2 → Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

def draw_pipeline_flowchart(fig):
    """Draw the 3-stage pipeline as a clean flowchart."""
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')

    # Stage boxes
    box_style = dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='#1565C0', lw=2.0)

    stage1_text = ('Stage 1\n'
                    'Proxy Factor-Graph\n'
                    'Reconstruction\n'
                    '─────────────────\n'
                    'Keyframe nodes\n'
                    'Odometry edges\n'
                    'Betweenness centrality')

    stage2_text = ('Stage 2\n'
                   'Bi-SMVS Estimation\n'
                   '─────────────────\n'
                   'G-ICP Hessian → L-Vul\n'
                   'Image cues → V-Vul\n'
                   'Fusion → Bi-SMVS')

    stage3_text = ('Stage 3\n'
                   'Spoofer Placement\n'
                   '─────────────────\n'
                   'Clustering + CMA-ES\n'
                   'Graph-aware scoring\n'
                   'Attack rosbag generation')

    ax.text(1.5, 1.0, stage1_text, ha='center', va='center',
            fontsize=9, fontfamily='monospace', bbox=box_style,
            transform=ax.transData)

    ax.text(5.0, 1.0, stage2_text, ha='center', va='center',
            fontsize=9, fontfamily='monospace', bbox=box_style,
            transform=ax.transData)

    ax.text(8.5, 1.0, stage3_text, ha='center', va='center',
            fontsize=9, fontfamily='monospace', bbox=box_style,
            transform=ax.transData)

    # Arrows
    arrow_style = dict(arrowstyle='->', color='#1565C0', lw=2.0,
                       connectionstyle='arc3,rad=0.0')
    ax.annotate('', xy=(3.7, 1.0), xytext=(2.8, 1.0), arrowprops=arrow_style)
    ax.annotate('', xy=(6.2, 1.0), xytext=(7.1, 1.0), arrowprops=arrow_style)

    # Centre labels
    ax.text(3.25, 1.0, 'Frames\n+ Imgs', ha='center', va='center',
            fontsize=7.5, color='#555555', transform=ax.transData)
    ax.text(6.75, 1.0, 'Spoofer\nLocation', ha='center', va='center',
            fontsize=7.5, color='#555555', transform=ax.transData)

    # Top title
    ax.text(5.0, 1.88, 'Bi-SMVS Attack-Planning Pipeline',
             ha='center', va='center', fontsize=13,
             fontweight='bold', color='#0D47A1')

    # Bottom: input → output
    ax.annotate('', xy=(0.0, 1.0), xytext=(0.3, 1.0),
                arrowprops=dict(arrowstyle='-', color='gray', lw=1.0))
    ax.text(0.0, 1.12, 'Pre-collected\nRoute Data', ha='center', va='bottom',
            fontsize=8, color='#555555')
    ax.text(0.0, 0.85, '(bag + trajectory)', ha='center', va='top',
            fontsize=7, color='#888888')

    ax.annotate('', xy=(10.0, 1.0), xytext=(9.7, 1.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
    ax.text(10.0, 1.12, 'Attacked\nRosbag', ha='center', va='bottom',
            fontsize=8, color='#555555')
    ax.text(10.0, 0.85, '(LiDAR-only\nspoofing)', ha='center', va='top',
            fontsize=7, color='#888888')

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Main figure assembly
# ─────────────────────────────────────────────────────────────────────────────

def generate_overview_figure(args):
    Path(args.output).mkdir(parents=True, exist_ok=True)
    out = Path(args.output)

    # ── Load data ──────────────────────────────────────────────────────────
    traj_x, traj_y, traj_t = load_trajectory(args.traj)
    l_vul, v_vul, b_vul = load_frame_data(args.smvs, args.vul)

    node_nids, node_x, node_y, edges = None, None, None, None
    if args.graph and os.path.isdir(args.graph):
        result = load_graph_data(args.graph)
        if result is not None:
            node_nids, node_x, node_y, edges = result

    frames_df = None
    if args.smvs and os.path.exists(args.smvs):
        try:
            frames_df = pd.read_csv(args.smvs)
            frames_df = frames_df.dropna(subset=["x", "y", "frame_bi_smvs"])
        except Exception:
            pass

    # ── Figure 1: Stage 1 — Proxy factor graph ────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    draw_stage1_proxy_graph(ax1, node_nids, node_x, node_y, edges,
                             traj_x, traj_y,
                             args.spoofer_x, args.spoofer_y)
    fig1.tight_layout()
    fig1.savefig(out / 'fig1_stage1_proxy_graph.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig(out / 'fig1_stage1_proxy_graph.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"[OK] Saved fig1_stage1_proxy_graph")

    # ── Figure 2: Stage 2 — Polar Bi-SMVS vectors ───────────────────────
    fig2 = plt.figure(figsize=(16, 5))
    gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.35)

    ax_L = fig2.add_subplot(gs2[0, 0], projection='polar')
    ax_V = fig2.add_subplot(gs2[0, 1], projection='polar')
    ax_B = fig2.add_subplot(gs2[0, 2], projection='polar')

    draw_stage2_bivul_vectors(ax_L, ax_V, ax_B, l_vul, v_vul, b_vul)

    fig2.suptitle('Stage 2 — Bi-SMVS: LiDAR × Visual Fusion',
                  fontsize=13, fontweight='bold', y=1.02)
    fig2.tight_layout()
    fig2.savefig(out / 'fig2_stage2_bivul_vectors.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig(out / 'fig2_stage2_bivul_vectors.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"[OK] Saved fig2_stage2_bivul_vectors")

    # ── Figure 3: Stage 3 — Placement map + Bi-SMVS comparison ──────────
    fig3, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: placement map
    draw_stage3_placement_map(axes[0], traj_x, traj_y, frames_df,
                             args.spoofer_x, args.spoofer_y)

    # Centre: Bi-SMVS comparison (bar chart)
    ax_top = axes[1]
    ax_bot = axes[2]
    draw_bivul_comparison(ax_top, ax_bot, l_vul, b_vul)

    fig3.suptitle('Stage 3 — Spoofer Placement & Vulnerability Comparison',
                  fontsize=13, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(out / 'fig3_stage3_placement_map.pdf', dpi=300, bbox_inches='tight')
    fig3.savefig(out / 'fig3_stage3_placement_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"[OK] Saved fig3_stage3_placement_map")

    # ── Figure 4: Attack point-cloud demo ───────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})

    # Generate synthetic point cloud for illustration
    rng = np.random.default_rng(42)
    n_demo = 800
    demo_r = rng.uniform(2.0, 40.0, n_demo)
    demo_theta = rng.uniform(0, 360, n_demo)
    demo_x = demo_r * np.cos(np.radians(demo_theta))
    demo_y = demo_r * np.sin(np.radians(demo_theta))
    demo_xyz = np.column_stack([demo_x, demo_y, rng.uniform(-2, 2, n_demo)])

    # Spoofed: remove window + inject wall
    center_deg = 90.0
    half_range = SPOOFING_RANGE / 2.0
    theta_mask = (demo_theta - center_deg + 180) % 360 - 180
    in_window = np.abs(theta_mask) <= half_range
    kept_xyz = demo_xyz[~in_window]

    # Build false wall
    wall_n = 400
    wall_theta = rng.uniform(center_deg - half_range,
                              center_deg + half_range, wall_n)
    wall_r = np.full(wall_n, WALL_DIST)
    wall_x = wall_r * np.cos(np.radians(wall_theta))
    wall_y = wall_r * np.sin(np.radians(wall_theta))
    wall_z = rng.uniform(-2, 2, wall_n)
    wall_xyz = np.column_stack([wall_x, wall_y, wall_z])
    spoofed_xyz = np.vstack([kept_xyz, wall_xyz])

    draw_attack_pointcloud_demo(ax4, demo_xyz, spoofed_xyz,
                                center_deg, half_range, WALL_DIST)
    fig4.tight_layout()
    fig4.savefig(out / 'fig4_stage3_attack_demo.pdf', dpi=300, bbox_inches='tight')
    fig4.savefig(out / 'fig4_stage3_attack_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"[OK] Saved fig4_stage3_attack_demo")

    # ── Figure 5: Pipeline flowchart ─────────────────────────────────────
    fig5 = plt.figure(figsize=(14, 4))
    draw_pipeline_flowchart(fig5)
    fig5.savefig(out / 'fig5_overview_flowchart.pdf', dpi=300, bbox_inches='tight')
    fig5.savefig(out / 'fig5_overview_flowchart.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"[OK] Saved fig5_overview_flowchart")

    # ── Summary figure: 5 panels in one ─────────────────────────────────
    # Layout: 3 rows × 4 cols
    # Row 0: [s1_graph] [polar_L] [polar_V] [polar_B]
    # Row 1: [spacer_placeholder] [placement_map] [attack_demo] [bivul_bar]
    # Row 2: [flowchart_spacer] [bivul_bar_bottom] [comp_bottom___] [flowchart_right]
    fig_main = plt.figure(figsize=(22, 16))
    gs = matplotlib.gridspec.GridSpec(
        3, 4, figure=fig_main,
        height_ratios=[1.0, 1.0, 0.55],
        hspace=0.38, wspace=0.28,
        left=0.06, right=0.97, top=0.93, bottom=0.06
    )

    # ── Row 0 ──────────────────────────────────────────────────────────
    ax_s1 = fig_main.add_subplot(gs[0, 0])
    draw_stage1_proxy_graph(ax_s1, node_nids, node_x, node_y, edges,
                           traj_x, traj_y, args.spoofer_x, args.spoofer_y)

    ax_L = fig_main.add_subplot(gs[0, 1], projection='polar')
    ax_V = fig_main.add_subplot(gs[0, 2], projection='polar')
    ax_B_polar = fig_main.add_subplot(gs[0, 3], projection='polar')
    draw_stage2_bivul_vectors(ax_L, ax_V, ax_B_polar, l_vul, v_vul, b_vul)

    # ── Row 1 ──────────────────────────────────────────────────────────
    ax_s3 = fig_main.add_subplot(gs[1, 0])
    draw_stage3_placement_map(ax_s3, traj_x, traj_y, frames_df,
                               args.spoofer_x, args.spoofer_y)

    ax_att = fig_main.add_subplot(gs[1, 1], projection='polar')
    draw_attack_pointcloud_demo(ax_att, demo_xyz, spoofed_xyz,
                              center_deg, half_range, WALL_DIST)

    ax_comp_top = fig_main.add_subplot(gs[1, 2])
    k_all = np.arange(N_BUCKETS)
    angles_all = (k_all + 0.5) * STEP_DEG
    l_norm = l_vul / (l_vul.max() + 1e-9)
    b_norm = b_vul / (l_vul.max() + 1e-9)
    ax_comp_top.bar(angles_all, l_norm, width=4.0, color='#1976D2', alpha=0.7,
                    label='L-Vul')
    ax_comp_top.bar(angles_all, b_norm, width=2.0, color='#D32F2F', alpha=0.7,
                    label='Bi-SMVS')
    ax_comp_top.set_ylabel('Normalised score')
    ax_comp_top.set_xlim(0, 360)
    ax_comp_top.legend(fontsize=8)
    ax_comp_top.set_title('Vulnerability Direction\nComparison', fontweight='bold', fontsize=9)
    ax_comp_top.grid(True, alpha=0.3, axis='y')

    # Bi-SMVS suppression bar chart (zoomed high-suppression region)
    ax_bar = fig_main.add_subplot(gs[1, 3])
    supp = l_norm - b_norm
    colours_sup = ['#388E3C' if s > 0.1 else '#BDBDBD' for s in supp]
    ax_bar.bar(angles_all, supp, width=4.0, color=colours_sup, alpha=0.8)
    ax_bar.axhline(0, color='black', lw=0.8)
    ax_bar.set_xlabel('Horizontal azimuth [°]')
    ax_bar.set_ylabel('L − Bi')
    ax_bar.set_xlim(0, 360)
    ax_bar.set_title('Visual Corrective\nSuppression', fontweight='bold', fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis='y')

    # ── Row 2: CMA-ES convergence + suppression over full azimuth ────────
    ax_supp_full = fig_main.add_subplot(gs[2, :3])
    ax_supp_full.bar(angles_all, supp, width=4.0, color='#388E3C', alpha=0.7)
    ax_supp_full.set_xlabel('Horizontal azimuth [°]')
    ax_supp_full.set_ylabel('Suppression = L − Bi')
    ax_supp_full.set_xlim(0, 360)
    ax_supp_full.set_title('Visual Corrective Support over Full Azimuth Range',
                           fontweight='bold', fontsize=9)
    ax_supp_full.grid(True, alpha=0.3, axis='y')

    # ── Row 2 right: pipeline mini flowchart ───────────────────────────
    ax_flow = fig_main.add_subplot(gs[2, 3])
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 2)
    ax_flow.axis('off')
    for spine in ax_flow.spines.values():
        spine.set_visible(False)
    ax_flow.set_facecolor('#F5F5F5')

    for bx, by, txt in [(1.5, 1.0, 'Stage 1\nProxy Graph'),
                          (5.0, 1.0, 'Stage 2\nBi-SMVS'),
                          (8.5, 1.0, 'Stage 3\nPlacement')]:
        box = matplotlib.patches.FancyBboxPatch(
            (bx - 1.1, by - 0.38), 2.2, 0.76,
            boxstyle='round,pad=0.1',
            facecolor='white', edgecolor='#1565C0', lw=1.5,
            transform=ax_flow.transData)
        ax_flow.add_patch(box)
        ax_flow.text(bx, by, txt, ha='center', va='center',
                     fontsize=7, fontweight='bold', transform=ax_flow.transData)

    ax_flow.annotate('', xy=(3.5, 1.0), xytext=(2.6, 1.0),
                     arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))
    ax_flow.annotate('', xy=(7.0, 1.0), xytext=(6.1, 1.0),
                     arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))
    ax_flow.set_title('Pipeline', fontsize=8, fontweight='bold')

    fig_main.suptitle(
        'Overview — Bi-SMVS: Bi-Modal Scan-Matching Vulnerability Score\n'
        'for LiDAR-Visual-Inertial SLAM Attack Planning',
        fontsize=14, fontweight='bold', y=0.97
    )
    fig_main.savefig(out / 'fig_overview_combined.pdf', dpi=300, bbox_inches='tight')
    fig_main.savefig(out / 'fig_overview_combined.png', dpi=150, bbox_inches='tight')
    plt.close(fig_main)
    print(f"[OK] Saved fig_overview_combined")

    print(f"\nAll figures saved to: {out}/")
    print("Files generated:")
    for f in sorted(Path(out).iterdir()):
        print(f"  {f.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate Bi-SMVS overview figure materials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--smvs', required=True,
                  help='Path to Bi-SMVS SMVS CSV')
    p.add_argument('--vul', required=True,
                  help='Path to vulnerability CSV (72-dim bi_vul_XX)')
    p.add_argument('--traj', required=True,
                  help='Path to reference trajectory CSV (x, y, time)')
    p.add_argument('--graph',
                  help='Path to LVI-SAM graph dump directory (dump_*.json)')
    p.add_argument('--spoofer-x', type=float, default=-18.5)
    p.add_argument('--spoofer-y', type=float, default=70.4)
    p.add_argument('--output', default='./overview_figures',
                  help='Output directory')
    return p.parse_args()


if __name__ == '__main__':
    generate_overview_figure(parse_args())
