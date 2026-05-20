#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--orig", required=True)
parser.add_argument("--att", required=True)
parser.add_argument("--out-prefix", required=True)
parser.add_argument("--title", default="LVI-SAM Original vs Attack")
parser.add_argument("--spoofer-x", type=float, default=None,
                    help="Spoofer world X coordinate")
parser.add_argument("--spoofer-y", type=float, default=None,
                    help="Spoofer world Y coordinate")
parser.add_argument("--distance-threshold", type=float, default=None,
                    help="Attack trigger radius in metres")
parser.add_argument("--wall-dist", type=float, default=None,
                    help="Static wall distance in metres (shown as arc from spoofer)")
parser.add_argument("--spoofing-range", type=float, default=None,
                    help="Angular spoofing window width in degrees")
parser.add_argument("--spoofer-heading", type=float, default=None,
                    help="Spoofer heading direction in degrees (toward robot), for drawing spoofing beam")
args = parser.parse_args()

orig = pd.read_csv(args.orig)
att = pd.read_csv(args.att)

to = orig["time"].values - orig["time"].values[0]
ta = att["time"].values - att["time"].values[0]

t_end = min(to[-1], ta[-1])
t_grid = np.linspace(0, t_end, 4000)

xo = np.interp(t_grid, to, orig["x"])
yo = np.interp(t_grid, to, orig["y"])
zo = np.interp(t_grid, to, orig["z"])

xa = np.interp(t_grid, ta, att["x"])
ya = np.interp(t_grid, ta, att["y"])
za = np.interp(t_grid, ta, att["z"])

err = np.sqrt((xo-xa)**2 + (yo-ya)**2 + (zo-za)**2)
rmse = np.sqrt(np.mean(err**2))

idx = int(np.argmax(err))

print("========== Attack Deviation ==========")
print(f"Compared duration: {t_end:.2f} s")
print(f"Mean deviation:    {err.mean():.4f} m")
print(f"RMSE deviation:    {rmse:.4f} m")
print(f"Max deviation:     {err.max():.4f} m")
print(f"Final deviation:   {err[-1]:.4f} m")

print("\nMax deviation details:")
print(f"Time from start:   {t_grid[idx]:.2f} s")
print(f"Original position: x={xo[idx]:.3f}, y={yo[idx]:.3f}, z={zo[idx]:.3f}")
print(f"Attack position:   x={xa[idx]:.3f}, y={ya[idx]:.3f}, z={za[idx]:.3f}")

t30 = np.linspace(0, min(30, t_end), 300)
xo30 = np.interp(t30, to, orig["x"])
yo30 = np.interp(t30, to, orig["y"])
zo30 = np.interp(t30, to, orig["z"])
xa30 = np.interp(t30, ta, att["x"])
ya30 = np.interp(t30, ta, att["y"])
za30 = np.interp(t30, ta, att["z"])
err30 = np.sqrt((xo30-xa30)**2 + (yo30-ya30)**2 + (zo30-za30)**2)

print("\nFirst 30s check:")
print("First 30s mean deviation:", float(err30.mean()))
print("First 30s max deviation: ", float(err30.max()))
print("First sample deviation:  ", float(err30[0]))
print("10s deviation:           ", float(err30[np.argmin(abs(t30-10))]))
print("30s deviation:           ", float(err30[-1]))

print("\nDeviation duration:")
for th in [0.5, 1, 2, 5, 10, 20, 50, 80, 100]:
    mask = err > th
    duration = mask.sum() / len(mask) * t_end
    print(f"deviation > {th:>5.1f} m: {duration:.2f} s, ratio={mask.mean():.4f}")

def draw_spoofer_annotation(ax, spoofer_x, spoofer_y, distance_threshold,
                            wall_dist, spoofing_range, spoofer_heading,
                            orig_df, att_df, t_grid):
    """Draw spoofer position and attack zone on the axes."""
    # ---- Trigger zone circle ----
    if distance_threshold is not None:
        circle = plt.Circle(
            (spoofer_x, spoofer_y),
            distance_threshold,
            fill=False,
            color="red",
            linestyle="--",
            linewidth=1.5,
            zorder=5,
            label=f"Trigger zone (r={distance_threshold}m)"
        )
        ax.add_patch(circle)

    # ---- Spoofer position marker ----
    ax.scatter(
        spoofer_x, spoofer_y,
        marker="X",
        s=200,
        c="red",
        edgecolors="black",
        linewidths=1,
        zorder=6,
        label="Spoofer"
    )
    ax.annotate(
        f"Spoofer\n({spoofer_x:.1f}, {spoofer_y:.1f})",
        (spoofer_x, spoofer_y),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="red", alpha=0.8),
        zorder=7,
    )

    # ---- Spoofing window: arc at wall_dist ----
    if spoofing_range is not None and wall_dist is not None:
        half_range = np.radians(spoofing_range / 2.0)
        heading_rad = np.radians(spoofer_heading) if spoofer_heading is not None else None

        if heading_rad is not None:
            # Draw beam direction line from spoofer outward
            beam_x = [spoofer_x, spoofer_x + wall_dist * 1.5 * np.cos(heading_rad)]
            beam_y = [spoofer_y, spoofer_y + wall_dist * 1.5 * np.sin(heading_rad)]
            ax.plot(
                beam_x, beam_y,
                color="orange",
                linewidth=1.5,
                linestyle="-",
                alpha=0.7,
                zorder=4,
                label=f"Spoof beam (range={spoofing_range}deg)"
            )

            # Draw spoofing window arc
            theta1 = np.degrees(heading_rad - half_range)
            theta2 = np.degrees(heading_rad + half_range)
            arc_theta = np.linspace(theta1, theta2, 200)
            arc_x = spoofer_x + wall_dist * np.cos(np.radians(arc_theta))
            arc_y = spoofer_y + wall_dist * np.sin(np.radians(arc_theta))
            ax.plot(
                arc_x, arc_y,
                color="orange",
                linewidth=1.2,
                linestyle="-",
                alpha=0.6,
                zorder=4,
            )
        else:
            # Fallback: circle at wall_dist showing the window size
            theta_center = np.linspace(0, 360, 400)
            arc_x = spoofer_x + wall_dist * np.cos(np.radians(theta_center))
            arc_y = spoofer_y + wall_dist * np.sin(np.radians(theta_center))
            ax.plot(
                arc_x, arc_y,
                color="orange",
                linewidth=1.0,
                linestyle="-",
                alpha=0.4,
                zorder=4,
                label=f"Wall dist={wall_dist}m"
            )

    # ---- Highlight frames where attack was triggered ----
    t0 = orig_df["time"].values[0]
    ox = orig_df["x"].values
    oy = orig_df["y"].values
    dx = ox - spoofer_x
    dy = oy - spoofer_y
    dist = np.sqrt(dx**2 + dy**2)

    if distance_threshold is not None:
        trigger_mask = dist <= distance_threshold
        if trigger_mask.sum() > 0:
            ax.scatter(
                ox[trigger_mask], oy[trigger_mask],
                c="orange",
                s=20,
                alpha=0.7,
                zorder=5,
                label=f"Attack triggered (n={trigger_mask.sum()})"
            )


# ---------------------------------------------------------------------------
# Main plotting logic
# ---------------------------------------------------------------------------

out_prefix = Path(args.out_prefix)
out_prefix.parent.mkdir(parents=True, exist_ok=True)

# ---- XY compare plot ----
fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(orig["x"], orig["y"], label="Original", linewidth=1.5, alpha=0.9)
ax.plot(att["x"], att["y"], label="Attack", linewidth=1.5, alpha=0.9)

# Mark start points
ax.scatter(orig["x"].iloc[0], orig["y"].iloc[0], marker="o", s=80,
           c="green", edgecolors="black", zorder=7, label="Start (orig)")
ax.scatter(att["x"].iloc[0], att["y"].iloc[0], marker="o", s=80,
           c="blue", edgecolors="black", zorder=7, label="Start (att)")

# Mark end points
ax.scatter(orig["x"].iloc[-1], orig["y"].iloc[-1], marker="s", s=80,
           c="green", edgecolors="black", zorder=7, label="End (orig)")
ax.scatter(att["x"].iloc[-1], att["y"].iloc[-1], marker="s", s=80,
           c="blue", edgecolors="black", zorder=7, label="End (att)")

# Mark max deviation point
ax.scatter(xo[idx], yo[idx], marker="*", s=300,
           c="red", edgecolors="black", zorder=8, label=f"Max dev ({err[idx]:.1f}m)")

# Draw spoofer annotation if provided
if args.spoofer_x is not None and args.spoofer_y is not None:
    draw_spoofer_annotation(
        ax,
        spoofer_x=args.spoofer_x,
        spoofer_y=args.spoofer_y,
        distance_threshold=args.distance_threshold,
        wall_dist=args.wall_dist,
        spoofing_range=args.spoofing_range,
        spoofer_heading=args.spoofer_heading,
        orig_df=orig,
        att_df=att,
        t_grid=t_grid,
    )

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.axis("equal")
ax.legend(loc="best", fontsize=8)
ax.set_title(args.title)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(str(out_prefix) + "_xy_compare.png", dpi=200)
plt.close(fig)

# ---- Deviation over time plot ----
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(t_grid, err, color="purple", linewidth=1.0, alpha=0.8)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)

# Shade the attack-trigger zone on time axis
if args.spoofer_x is not None and args.spoofer_y is not None:
    t0 = orig["time"].values[0]
    ox = orig["x"].values
    oy = orig["y"].values
    dx = ox - args.spoofer_x
    dy = oy - args.spoofer_y
    dist = np.sqrt(dx**2 + dy**2)
    if args.distance_threshold is not None:
        rel_times = orig["time"].values - t0
        trigger_mask = dist <= args.distance_threshold
        if trigger_mask.sum() > 0:
            t_start = rel_times[trigger_mask].min()
            t_end_ts = rel_times[trigger_mask].max()
            ax2.axvspan(t_start, t_end_ts, alpha=0.2, color="red",
                        label="Attack zone in time")
            ax2.axvline(t_start, color="red", linestyle=":", linewidth=1)
            ax2.axvline(t_end_ts, color="red", linestyle=":", linewidth=1)

ax2.scatter(t_grid[idx], err[idx], marker="*", s=200,
            c="red", zorder=8, label=f"Max ({err[idx]:.1f}m at {t_grid[idx]:.1f}s)")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("deviation [m]")
ax2.set_title(args.title + " — Deviation over time")
ax2.legend(loc="best", fontsize=8)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(str(out_prefix) + "_deviation.png", dpi=200)
plt.close(fig2)

print(f"\n[OK] saved: {out_prefix}_xy_compare.png")
print(f"[OK] saved: {out_prefix}_deviation.png")
