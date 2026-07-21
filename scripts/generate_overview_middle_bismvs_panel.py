#!/usr/bin/env python3
"""Generate the middle overview panel from real Bi-SMVS CSV data."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle, Wedge


N_BUCKETS = 72
BUCKET_DEG = 5.0


def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def line_collection_from_route(df, value_col, qmin, qmax, cmap="turbo"):
    pts = df[["x", "y"]].to_numpy(dtype=float)
    vals = df[value_col].to_numpy(dtype=float)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidths=2.2, alpha=0.96)
    lc.set_array(vals[:-1])
    lc.set_clim(qmin, qmax)
    return lc


def plot_route(ax, df, value_col, qmin, qmax, label, callout=None):
    lc = line_collection_from_route(df, value_col, qmin, qmax)
    ax.add_collection(lc)
    ax.set_xlim(df["x"].min() - 15, df["x"].max() + 15)
    ax.set_ylim(df["y"].min() - 15, df["y"].max() + 15)
    ax.set_aspect("equal")
    clean_axes(ax)
    ax.text(0.012, 0.91, label, transform=ax.transAxes, fontsize=7.2,
            fontweight="bold", color="#152238",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.82))

    if callout is not None:
        x, y, text, color = callout
        ax.scatter([x], [y], s=62, facecolor="none", edgecolor=color,
                   linewidth=1.6, zorder=5)
        ax.annotate(text, xy=(x, y), xytext=(18, -20), textcoords="offset points",
                    fontsize=6.5, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.1))
    return lc


def draw_directional_fusion(ax, vul_row):
    clean_axes(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)

    l = np.array([vul_row.get(f"l_vul_{i:02d}", 0.0) for i in range(N_BUCKETS)], dtype=float)
    v = np.array([vul_row.get(f"v_vul_{i:02d}", 0.0) for i in range(N_BUCKETS)], dtype=float)
    b = np.array([vul_row.get(f"bi_vul_{i:02d}", 0.0) for i in range(N_BUCKETS)], dtype=float)
    l_norm = l / (np.nanmax(l) + 1e-9)
    b_norm = b / (np.nanmax(b) + 1e-9)
    v_norm = v / (np.nanmax(v) + 1e-9)

    ax.add_patch(Circle((0, 0), 0.42, fill=False, ec="#2D3748", lw=1.0))
    ax.add_patch(Wedge((0, 0), 0.40, -35, 35, facecolor="#2E8B57",
                       edgecolor="#2E8B57", lw=1.0, alpha=0.18))
    ax.arrow(0, 0, 0.35, 0, color="#1F5AA6", width=0.012,
             head_width=0.065, head_length=0.070, length_includes_head=True)

    # Outer orange rays: LiDAR-only directional sensitivity.
    for i, val in enumerate(l_norm):
        if val < 0.10:
            continue
        ang = math.radians((i + 0.5) * BUCKET_DEG)
        r0, r1 = 0.47, 0.47 + 0.33 * val
        ax.plot([r0 * math.cos(ang), r1 * math.cos(ang)],
                [r0 * math.sin(ang), r1 * math.sin(ang)],
                color="#F59E0B", alpha=0.42, lw=1.05)

    # Inner red rays: retained fused vulnerability after visual correction.
    for i, val in enumerate(b_norm):
        if val < 0.13:
            continue
        ang = math.radians((i + 0.5) * BUCKET_DEG)
        r0, r1 = 0.50, 0.50 + 0.44 * val
        ax.plot([r0 * math.cos(ang), r1 * math.cos(ang)],
                [r0 * math.sin(ang), r1 * math.sin(ang)],
                color="#D7191C", alpha=0.78, lw=1.35)

    # Small green ticks indicate directions with visual support in the data.
    for i, val in enumerate(v_norm):
        if val < 0.58:
            continue
        ang = math.radians((i + 0.5) * BUCKET_DEG)
        r0, r1 = 0.96, 1.05
        ax.plot([r0 * math.cos(ang), r1 * math.cos(ang)],
                [r0 * math.sin(ang), r1 * math.sin(ang)],
                color="#2E8B57", alpha=0.85, lw=1.4)

    dom = int(np.argmax(b))
    dom_ang = (dom + 0.5) * BUCKET_DEG
    ax.add_patch(Wedge((0, 0), 1.08, dom_ang - 25, dom_ang + 25,
                       facecolor="none", edgecolor="#D7191C",
                       lw=1.3, linestyle="--", alpha=0.88))

    ax.text(0.0, 1.11, "Directional fusion", ha="center", va="bottom",
            fontsize=7.1, fontweight="bold", color="#152238")
    ax.text(-1.05, -1.06, "orange: LiDAR-only", fontsize=5.6, color="#B45309")
    ax.text(-1.05, -0.91, "red: fused vulnerability", fontsize=5.6, color="#B91C1C")
    ax.text(-1.05, -0.76, "green: visual support", fontsize=5.6, color="#166534")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bismvs", default="src/LVI-SAM/datasets/slamspoof_handheld/smvs/07_08_23_25_42_BiSMVS.csv")
    ap.add_argument("--vul", default="src/LVI-SAM/datasets/slamspoof_handheld/vul/vul_07_08_23_25_42_BiSMVS.csv")
    ap.add_argument("--out", default="/home/qu_menghao/Downloads/overview_middle_bismvs")
    args = ap.parse_args()

    bismvs = pd.read_csv(args.bismvs).dropna(subset=["x", "y", "frame_l_smvs", "frame_bi_smvs"])
    vul = pd.read_csv(args.vul).dropna(subset=["x", "y", "frame_bi_smvs"])

    # Use a shared robust color range so the two routes are visually comparable.
    both = np.r_[bismvs["frame_l_smvs"].to_numpy(float), bismvs["frame_bi_smvs"].to_numpy(float)]
    qmin, qmax = np.nanpercentile(both, [2, 98])

    # Pick one real frame where visual information suppresses LiDAR-only score strongly.
    damp = bismvs.assign(delta=bismvs["frame_l_smvs"] - bismvs["frame_bi_smvs"]).nlargest(1, "delta").iloc[0]
    vul_idx = int(np.argmin(np.abs(vul["timestamp"].to_numpy(float) - float(damp["timestamp"]))))
    vul_row = vul.iloc[vul_idx]

    fig = plt.figure(figsize=(7.25, 2.28), dpi=240)
    ax = fig.add_axes([0, 0, 1, 1])
    clean_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0.008, 0.012), 0.984, 0.976, fill=False,
                           ec="#E0A800", lw=2.2))
    ax.add_patch(Rectangle((0.008, 0.858), 0.984, 0.130,
                           fc="#F7B500", ec="#F7B500", lw=0))
    ax.text(0.030, 0.922, "2. Estimate frame-wise Bi-SMVS",
            fontsize=9.2, fontweight="bold", color="#171717", va="center")

    top = fig.add_axes([0.038, 0.505, 0.675, 0.330])
    bot = fig.add_axes([0.038, 0.175, 0.675, 0.330])
    wheel = fig.add_axes([0.735, 0.182, 0.225, 0.610])

    plot_route(top, bismvs, "frame_l_smvs", qmin, qmax,
               "LiDAR-only SMVS",
               callout=(float(damp["x"]), float(damp["y"]),
                        "LiDAR-sensitive", "#B45309"))
    lc = plot_route(bot, bismvs, "frame_bi_smvs", qmin, qmax,
                    "Bi-SMVS after visual support",
                    callout=(float(damp["x"]), float(damp["y"]),
                             "re-weighted", "#B91C1C"))
    draw_directional_fusion(wheel, vul_row)

    cb_ax = fig.add_axes([0.170, 0.092, 0.410, 0.018])
    cb = fig.colorbar(lc, cax=cb_ax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.2, length=2, pad=1.0)
    cb.set_label("low vulnerability  \u2192  high vulnerability", fontsize=5.9, labelpad=1.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.01, dpi=320)
    print(out.with_suffix(".svg"))
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
