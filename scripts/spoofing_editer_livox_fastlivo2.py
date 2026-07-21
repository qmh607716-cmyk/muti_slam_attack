#!/usr/bin/env python3
"""Trajectory-guided Livox CustomMsg spoofing editor for FAST-LIVO2 bags.

This is the FAST-LIVO2 counterpart of ``spoofing_editer_lvisam.py``.  It keeps
all non-LiDAR topics unchanged and edits only a Livox ``CustomMsg`` topic such
as ``/livox/lidar``.
"""

import argparse
import copy
import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import rosbag
from livox_ros_driver.msg import CustomPoint


def quaternion_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def polar_mask_2d(x: np.ndarray, y: np.ndarray,
                  center_deg: float, half_range_deg: float) -> np.ndarray:
    theta_deg = (np.degrees(np.arctan2(y, x)) + 180.0) % 360.0
    center_mod = center_deg % 360.0
    delta = (theta_deg - center_mod + 180.0) % 360.0 - 180.0
    return np.abs(delta) <= half_range_deg


def attack_angle_lidar_local(robot_x: float, robot_y: float,
                             spoofer_x: float, spoofer_y: float,
                             robot_yaw: float) -> float:
    world_angle = math.atan2(spoofer_y - robot_y, spoofer_x - robot_x)
    local_angle = world_angle - robot_yaw
    return math.atan2(math.sin(local_angle), math.cos(local_angle))


class StageDState:
    def __init__(self, reference_file: str):
        df = pd.read_csv(reference_file)
        required = {"time", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"{reference_file} missing columns: {sorted(missing)}")

        self.ref_t = df["time"].to_numpy(dtype=np.float64)
        self.ref_x = df["x"].to_numpy(dtype=np.float64)
        self.ref_y = df["y"].to_numpy(dtype=np.float64)
        self.ref_z = df["z"].to_numpy(dtype=np.float64) if "z" in df else np.zeros(len(df))
        self.ref_qx = df["qx"].to_numpy(dtype=np.float64) if "qx" in df else np.zeros(len(df))
        self.ref_qy = df["qy"].to_numpy(dtype=np.float64) if "qy" in df else np.zeros(len(df))
        self.ref_qz = df["qz"].to_numpy(dtype=np.float64) if "qz" in df else np.zeros(len(df))
        self.ref_qw = df["qw"].to_numpy(dtype=np.float64) if "qw" in df else np.ones(len(df))
        self.ref_t0 = float(self.ref_t[0])
        self.bag_start_sec: Optional[float] = None

    def lookup(self, bag_time_sec: float):
        if self.bag_start_sec is None:
            self.bag_start_sec = float(bag_time_sec)
        rel_sec = float(bag_time_sec) - self.bag_start_sec
        query_t = self.ref_t0 + rel_sec
        idx = int(np.argmin(np.abs(self.ref_t - query_t)))
        yaw = quaternion_yaw(
            float(self.ref_qx[idx]), float(self.ref_qy[idx]),
            float(self.ref_qz[idx]), float(self.ref_qw[idx]),
        )
        return (
            float(self.ref_x[idx]),
            float(self.ref_y[idx]),
            float(self.ref_z[idx]),
            yaw,
            rel_sec,
        )


def _point_arrays(points):
    n = len(points)
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    for i, p in enumerate(points):
        x[i] = p.x
        y[i] = p.y
        z[i] = p.z
    return x, y, z


def _sample_attr(points, attr: str, n: int, rng: np.random.Generator, default: int):
    if points:
        vals = np.asarray([getattr(p, attr) for p in points], dtype=np.int64)
        return rng.choice(vals, size=n, replace=True)
    return np.full(n, default, dtype=np.int64)


def _sample_elevation(points, n: int, rng: np.random.Generator):
    if points:
        arr = np.asarray(
            [math.atan2(p.z, max(math.hypot(p.x, p.y), 1e-3)) for p in points],
            dtype=np.float64,
        )
        return rng.choice(arr, size=n, replace=True)
    return rng.uniform(math.radians(-15.0), math.radians(15.0), size=n)


def _make_custom_point(offset_time, x, y, z, reflectivity, tag, line):
    p = CustomPoint()
    p.offset_time = int(max(0, min(int(offset_time), 2 ** 32 - 1)))
    p.x = float(x)
    p.y = float(y)
    p.z = float(z)
    p.reflectivity = int(max(0, min(int(reflectivity), 255)))
    p.tag = int(max(0, min(int(tag), 255)))
    p.line = int(max(0, min(int(line), 255)))
    return p


def _random_livox_points(n: int, center_deg: float, half_range_deg: float,
                         source_points, rng: np.random.Generator,
                         r_min: float, r_max: float,
                         vertical_lines: int):
    if n <= 0:
        return []

    center_atan = (center_deg - 180.0) % 360.0
    theta_deg = rng.uniform(center_atan - half_range_deg,
                            center_atan + half_range_deg,
                            size=n) % 360.0
    theta = np.radians(theta_deg)
    radius = rng.uniform(r_min, r_max, size=n)
    elev = _sample_elevation(source_points, n, rng)

    refl = _sample_attr(source_points, "reflectivity", n, rng, 35)
    tag = _sample_attr(source_points, "tag", n, rng, 0)
    line = _sample_attr(source_points, "line", n, rng, 0)
    if not source_points:
        line = rng.integers(0, max(1, vertical_lines), size=n)
    offsets = _sample_attr(source_points, "offset_time", n, rng, 0)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = radius * np.tan(elev)

    return [
        _make_custom_point(offsets[i], x[i], y[i], z[i], refl[i], tag[i], line[i])
        for i in range(n)
    ]


def _beam_project_wall_points(n: int, wall_dist: float, affected_points,
                              rng: np.random.Generator, center_deg: float,
                              half_range_deg: float, vertical_lines: int):
    if n <= 0:
        return []
    if not affected_points:
        return _random_livox_points(
            n, center_deg, half_range_deg, [], rng,
            wall_dist, wall_dist, vertical_lines,
        )

    src_idx = rng.integers(0, len(affected_points), size=n)
    out = []
    for idx in src_idx:
        p0 = affected_points[int(idx)]
        norm = max(math.sqrt(p0.x * p0.x + p0.y * p0.y + p0.z * p0.z), 1e-3)
        scale = wall_dist / norm
        out.append(
            _make_custom_point(
                p0.offset_time,
                p0.x * scale,
                p0.y * scale,
                p0.z * scale,
                p0.reflectivity,
                p0.tag,
                p0.line,
            )
        )
    return out


def _count_static(spoofing_range: float, horizontal_resolution: float,
                  vertical_lines: float) -> int:
    return max(0, int((spoofing_range / horizontal_resolution) * vertical_lines))


def _count_noise(spoofing_range: float, horizontal_resolution: float,
                 vertical_lines: float, spoofing_rate: float) -> int:
    return max(0, int((spoofing_range / horizontal_resolution) * vertical_lines * spoofing_rate))


def edit_bag(config: dict):
    main = config["main"]
    sim = config.get("simulator", {})
    filt = config.get("filtering", {})

    input_file = main["input_file"]
    output_file = main["output_file"]
    lidar_topic = main.get("lidar_topic", "/livox/lidar")
    mode = main.get("spoofing_mode", "static")
    if mode not in ("static", "removal"):
        raise RuntimeError("FAST-LIVO2 Livox editor supports only static/removal")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    rng = np.random.default_rng(int(main.get("rng_seed", 42)))
    state = StageDState(main["reference_file"])

    spoofer_x = float(main["spoofer_x"])
    spoofer_y = float(main["spoofer_y"])
    distance_threshold = float(main.get("distance_threshold", 15.0))
    spoofing_range = float(main.get("spoofing_range", 80.0))
    half_range = spoofing_range / 2.0
    wall_dist = float(main.get("wall_dist", 15.0))
    geometry_model = main.get("static_geometry_model", "beam_project")
    point_count_model = main.get("point_count_model", "original")

    horizontal_resolution = float(sim.get("horizontal_resolution", 0.1))
    vertical_lines = float(sim.get("vertical_lines", 6.0))
    spoofing_rate = float(sim.get("spoofing_rate", 0.3))
    min_dist = float(filt.get("minimum_measuring_distance", 0.0))
    max_dist = float(filt.get("maximum_measuring_distance", 30.0))

    stats = {
        "total_frames": 0,
        "triggered_frames": 0,
        "removed_points_sum": 0,
        "removed_points_max": 0,
        "injected_points_sum": 0,
        "injected_points_max": 0,
    }

    print(
        f"[spoofing_editer_livox_fastlivo2] mode={mode} "
        f"topic={lidar_topic} range={spoofing_range}deg D={distance_threshold}m "
        f"wall={wall_dist}m model={geometry_model}"
    )

    with rosbag.Bag(input_file, "r") as src, rosbag.Bag(output_file, "w") as dst:
        for topic, msg, t in src.read_messages():
            if topic != lidar_topic:
                dst.write(topic, msg, t)
                continue

            stats["total_frames"] += 1
            robot_x, robot_y, _robot_z, robot_yaw, _rel_sec = state.lookup(t.to_sec())
            dist_to_spoofer = math.hypot(robot_x - spoofer_x, robot_y - spoofer_y)
            if dist_to_spoofer > distance_threshold:
                dst.write(topic, msg, t)
                continue

            stats["triggered_frames"] += 1
            center_rad = attack_angle_lidar_local(
                robot_x, robot_y, spoofer_x, spoofer_y, robot_yaw
            )
            center_deg = (math.degrees(center_rad) + 180.0) % 360.0

            points = list(msg.points)
            x, y, _z = _point_arrays(points)
            mask = polar_mask_2d(x, y, center_deg, half_range)
            kept = [p for p, m in zip(points, mask) if not m]
            affected = [p for p, m in zip(points, mask) if m]
            n_removed = len(affected)

            if point_count_model == "pure_removal":
                injected = []
            elif point_count_model == "equal_replace":
                n_inj = n_removed
                if mode == "static":
                    injected = _beam_project_wall_points(
                        n_inj, wall_dist, affected, rng,
                        center_deg, half_range, int(vertical_lines),
                    )
                else:
                    injected = _random_livox_points(
                        n_inj, center_deg, half_range, affected, rng,
                        max(0.1, min_dist), max(max_dist, min_dist + 0.1),
                        int(vertical_lines),
                    )
            elif mode == "static":
                n_inj = _count_static(spoofing_range, horizontal_resolution, vertical_lines)
                if geometry_model == "original_random":
                    injected = _random_livox_points(
                        n_inj, center_deg, half_range, affected, rng,
                        wall_dist, wall_dist, int(vertical_lines),
                    )
                else:
                    injected = _beam_project_wall_points(
                        n_inj, wall_dist, affected, rng,
                        center_deg, half_range, int(vertical_lines),
                    )
            else:
                n_inj = _count_noise(
                    spoofing_range, horizontal_resolution, vertical_lines, spoofing_rate
                )
                injected = _random_livox_points(
                    n_inj, center_deg, half_range, affected, rng,
                    max(0.1, min_dist), max(max_dist, min_dist + 0.1),
                    int(vertical_lines),
                )

            out_msg = copy.deepcopy(msg)
            out_msg.points = kept + injected
            out_msg.point_num = len(out_msg.points)
            stats["removed_points_sum"] += n_removed
            stats["removed_points_max"] = max(stats["removed_points_max"], n_removed)
            stats["injected_points_sum"] += len(injected)
            stats["injected_points_max"] = max(stats["injected_points_max"], len(injected))
            dst.write(topic, out_msg, t)

    t = stats["triggered_frames"]
    print("=" * 58)
    print("  spoofing_editer_livox_fastlivo2 summary")
    print("=" * 58)
    print(f"  Mode                 : {mode}")
    print(f"  Total frames         : {stats['total_frames']}")
    print(f"  Triggered frames     : {t} ({100.0 * t / max(stats['total_frames'], 1):.4f}%)")
    if t:
        print(f"  Mean removed/frame   : {stats['removed_points_sum'] / t:.1f}")
        print(f"  Max removed (frame)  : {stats['removed_points_max']}")
        print(f"  Mean injected/frame  : {stats['injected_points_sum'] / t:.1f}")
        print(f"  Max injected (frame) : {stats['injected_points_max']}")
    print(f"  Output bag           : {output_file}")
    print("=" * 58)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    edit_bag(config)


if __name__ == "__main__":
    main()
