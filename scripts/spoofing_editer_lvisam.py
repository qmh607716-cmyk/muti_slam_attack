#!/usr/bin/env python3
"""
spoofing_editer_lvisam.py
==========================
Stage D trajectory-guided LiDAR spoofing attack editor for LVI-SAM.

支持三种攻击模式（由 config["main"]["spoofing_mode"] 选择）：
  removal   — HFR 攻击：删除攻击窗口内点 → 补随机噪声点
  static    — 假墙注入：删除攻击窗口内点 → 注入固定距离的平面墙
  dynamic   — 动墙注入：删除攻击窗口内点 → 注入随时间周期变化的墙

Key design principles:
  1. 所有非 /points_raw topics 原样写回（camera, IMU, GPS 等）。
  2. /points_raw 保持 point_step=22 布局不变（x, y, z, intensity, ring, time）。
     注入的伪造点合成完整的 22-byte 记录，不破坏字段结构。
  3. 参考轨迹（原始 LVI-SAM odometry）只加载一次，用于判断是否触发攻击。
  4. 攻击方向在 LiDAR 局部坐标系下计算（减去机器人 yaw），
     无论机器人在世界哪个朝向，移除窗口始终正确。

Usage
-----
  roslaunch slamspoof_icra rosbag_editer_lvisam.launch

The config file is passed via ROS param ``config_file``.
"""

import os
import sys
import json
from typing import Optional

import rospy
import numpy as np
import pandas as pd
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

# Add the scripts/ directory to sys.path for functions/ subpackage.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from functions import spoofing_sim_lvisam
import registration_separated


# ---------------------------------------------------------------------------
# Helper: binary → xyz
# ---------------------------------------------------------------------------

def binary_to_xyz(binary: np.ndarray):
    """Extract (x, y, z) from raw uint8 binary records."""
    x = binary[:, 0:4].view(dtype=np.float32)
    y = binary[:, 4:8].view(dtype=np.float32)
    z = binary[:, 8:12].view(dtype=np.float32)
    return x.flatten(), y.flatten(), z.flatten()


# ---------------------------------------------------------------------------
# Helper: quaternion → yaw
# ---------------------------------------------------------------------------

def quaternion_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return the yaw (rotation around world Z) of a unit quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# State: loaded once per run
# ---------------------------------------------------------------------------

class StageDState:
    """
    Holds the reference trajectory and runtime state for Stage D
    trajectory-guided spoofing.

    For adaptive injection, optionally loads pre-computed Hessian eigenvector
    data (from calc_hessian_eigenvector) indexed by frame timestamp.
    """

    def __init__(self, config: dict, rng: np.random.Generator,
                 load_eigenvec: bool = False):
        self.rng = rng

        self.ref_file = config["main"].get("reference_file")
        if not self.ref_file:
            raise RuntimeError(
                "config['main']['reference_file'] is required for Stage D"
            )

        df = pd.read_csv(self.ref_file)
        self.ref_t  = df["time"].to_numpy()
        self.ref_x  = df["x"].to_numpy()
        self.ref_y  = df["y"].to_numpy()
        self.ref_z  = df["z"].to_numpy()
        self.ref_qx = df.get("qx", pd.Series([0.0] * len(df))).to_numpy()
        self.ref_qy = df.get("qy", pd.Series([0.0] * len(df))).to_numpy()
        self.ref_qz = df.get("qz", pd.Series([0.0] * len(df))).to_numpy()
        self.ref_qw = df.get("qw", pd.Series([1.0] * len(df))).to_numpy()

        self.ref_t0 = float(self.ref_t[0])

        self.bag_start_sec: Optional[float] = None

        self.stats = {
            "total_frames":       0,
            "triggered_frames":   0,
            "removed_points_sum": 0,
            "removed_points_max": 0,
            "injected_points_sum": 0,
            "injected_points_max": 0,
        }

        print(f"[Stage D] Loaded reference trajectory: {self.ref_file}")
        print(f"[Stage D] Reference poses: {len(df)}")

        # ── Adaptive injection: load Hessian eigenvector pre-computation ────────
        self.eigenvec_data: Optional[dict] = None
        if load_eigenvec:
            eig_file = config["main"].get("eigenvector_csv")
            if eig_file and os.path.exists(eig_file):
                eig_df = pd.read_csv(eig_file)
                self.eigenvec_data = {
                    "time":       eig_df["time"].to_numpy(),
                    "eigenvec_x": eig_df["eigenvec_x"].to_numpy()
                                 if "eigenvec_x" in eig_df.columns else None,
                    "eigenvec_y": eig_df["eigenvec_y"].to_numpy()
                                 if "eigenvec_y" in eig_df.columns else None,
                    "eigenvec_z": eig_df["eigenvec_z"].to_numpy()
                                 if "eigenvec_z" in eig_df.columns else None,
                    "dot_eig":    eig_df["dot_eig"].to_numpy()
                                 if "dot_eig" in eig_df.columns else None,
                }
                print(f"[Stage D] Loaded eigenvector pre-computation: {eig_file}")
            else:
                print(
                    f"[Stage D] WARNING: eigenvector_csv='{eig_file}' "
                    f"not found — adaptive injection falls back to rotate_rad=0"
                )

    def lookup(self, bag_timestamp_ns: int):
        """
        Return (x, y, z, yaw, bag_rel_sec) for the given bag timestamp.
        """
        now_sec = bag_timestamp_ns / 1e9 if bag_timestamp_ns > 1e12 else float(bag_timestamp_ns)

        if self.bag_start_sec is None:
            self.bag_start_sec = now_sec

        rel_sec = now_sec - self.bag_start_sec
        query_t = self.ref_t0 + rel_sec

        idx = int(np.argmin(np.abs(self.ref_t - query_t)))
        x   = float(self.ref_x[idx])
        y   = float(self.ref_y[idx])
        z   = float(self.ref_z[idx])
        yaw = quaternion_yaw(
            float(self.ref_qx[idx]),
            float(self.ref_qy[idx]),
            float(self.ref_qz[idx]),
            float(self.ref_qw[idx]),
        )

        return x, y, z, yaw, rel_sec

    def lookup_eigenvec(self, bag_timestamp_ns: int):
        """
        Return (eigenvec_xyz, dot_eig) for the given bag timestamp.

        eigenvec_xyz: (M, 3) numpy array of per-point translation eigenvectors
        dot_eig:     (M,) numpy array of dot products

        Returns (None, None) if eigenvector data is not loaded or lookup fails.
        """
        if self.eigenvec_data is None:
            return None, None

        now_sec = bag_timestamp_ns / 1e9 if bag_timestamp_ns > 1e12 else float(bag_timestamp_ns)
        if self.bag_start_sec is None:
            self.bag_start_sec = now_sec
        rel_sec = now_sec - self.bag_start_sec
        query_t = self.ref_t0 + rel_sec

        ev = self.eigenvec_data
        idx = int(np.argmin(np.abs(ev["time"] - query_t)))

        if ev["eigenvec_x"] is not None:
            ex = ev["eigenvec_x"][idx] if idx < len(ev["eigenvec_x"]) else 0.0
            ey = ev["eigenvec_y"][idx] if idx < len(ev["eigenvec_y"]) else 0.0
            ez = ev["eigenvec_z"][idx] if idx < len(ev["eigenvec_z"]) else 0.0
            eigenvec_xyz = np.array([[ex, ey, ez]], dtype=np.float64)
        else:
            eigenvec_xyz = np.zeros((1, 3), dtype=np.float64)

        dot_eig = ev["dot_eig"][idx] if ev["dot_eig"] is not None and idx < len(ev["dot_eig"]) else 0.0
        dot_eig = np.atleast_1d(np.array(dot_eig, dtype=np.float64))

        return eigenvec_xyz, dot_eig


# ---------------------------------------------------------------------------
# Attack direction helpers
# ---------------------------------------------------------------------------

def attack_angle_world(robot_x: float, robot_y: float,
                       spoofer_x: float, spoofer_y: float) -> float:
    """Angle from robot to spoofer in the world frame (radians)."""
    return np.arctan2(spoofer_y - robot_y, spoofer_x - robot_x)


def attack_angle_lidar_local(robot_x: float, robot_y: float,
                             spoofer_x: float, spoofer_y: float,
                             robot_yaw: float) -> float:
    """
    Angle from robot to spoofer in the LiDAR local frame (radians).

    Subtracts the robot's yaw so the removal/injection window is always
    expressed relative to the LiDAR's forward direction.
    """
    world_angle = attack_angle_world(robot_x, robot_y, spoofer_x, spoofer_y)
    # LVI-SAM convention (tf::Matrix3x3.getRPY, ROS REP-103):
    #   yaw=0 → robot faces +X (East). +90° → +Y (North).
    #   polar_mask_2d uses (atan2(y, x) + 180) % 360, which maps +X → 180°.
    #   We just need to subtract yaw to get local-frame direction; the
    #   caller will do the +180 wrap when comparing.
    local_angle = world_angle - robot_yaw
    return np.arctan2(np.sin(local_angle), np.cos(local_angle))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_file = rospy.get_param("config_file", "")
    if not config_file:
        raise rospy.ROSException(
            "rospy.get_param('config_file') is empty — pass via roslaunch <param>"
        )

    with open(config_file, "r") as f:
        config = json.load(f)

    output_file = config["main"]["output_file"]
    if os.path.exists(output_file):
        os.remove(output_file)

    # ── Load config ────────────────────────────────────────────────────────
    typestore       = get_typestore(Stores.ROS1_NOETIC)
    points_topic    = config["main"]["lidar_topic"]
    spoofing_mode   = config["main"].get("spoofing_mode", "removal")
    spoofer_x       = float(config["main"]["spoofer_x"])
    spoofer_y       = float(config["main"]["spoofer_y"])
    distance_thresh = float(config["main"]["distance_threshold"])
    spoofing_range  = float(config["main"]["spoofing_range"])
    half_range_deg  = spoofing_range / 2.0

    # LVI-SAM-compatible count model.
    # point_count_model:
    #   original      -> use SLAMSpoof original injection count formula
    #   equal_replace -> replace as many points as removed
    #   pure_removal  -> remove points without adding spoofed points
    point_count_model = config["main"].get("point_count_model", "original")
    static_geometry_model = config["main"].get("static_geometry_model", "original_random")
    wall_intensity = float(config["main"].get("wall_intensity", 120.0))

    # D-SLAMSpoof: square/corner/planar geometry parameters
    # square_scale_S: scaling constant for polar equation d_fake=S/(|sin|+|cos|)
    #   If None, defaults to wall_distance * sqrt(2) (makes average dist ≈ wall_distance)
    square_scale_S = config["main"].get("square_scale_S", None)
    # square_rotate_rad: rotation of the polar equation (radians)
    #   0.0       → corner/L-shape (adjacent edges facing LiDAR, default)
    #   π/4≈0.785 → planar wall (opposite edges, perpendicular to radial)
    square_rotate_raw = config["main"].get("square_rotate_rad", None)
    square_rotate_rad = None
    if square_rotate_raw is not None:
        import math
        square_rotate_rad = float(square_rotate_raw)
        if isinstance(square_rotate_raw, str) and "pi" in square_rotate_raw.lower():
            # Allow string like "pi/4" for convenience
            if square_rotate_raw.lower() == "pi/4":
                square_rotate_rad = math.pi / 4.0
            elif square_rotate_raw.lower() == "pi/2":
                square_rotate_rad = math.pi / 2.0
            elif square_rotate_raw.lower() == "pi":
                square_rotate_rad = math.pi
    elif static_geometry_model == "corner":
        import math
        square_rotate_rad = 0.0
    elif static_geometry_model == "planar":
        import math
        square_rotate_rad = math.pi / 4.0

    # D-SLAMSpoof: Oscillating Injection constraint parameters
    # M_corr: max correspondence distance of the target SLAM (conservative default: 1.0m)
    #   Increase if the SLAM uses a larger outlier rejection threshold.
    M_corr = float(config["main"].get("M_corr", 1.0))
    # lidar_scan_period: Δt between consecutive scans (VLP-16: 0.1s, HDL-64: 0.05s)
    lidar_scan_period = float(config["main"].get("lidar_scan_period", 0.1))
    # auto_cycle: if True, compute optimal t_cycle from M_corr constraint (D-SLAMSpoof Eq.4)
    auto_cycle = config["main"].get("auto_cycle", True)

    sim_cfg = config.get("simulator", {})
    horizontal_resolution = float(sim_cfg.get("horizontal_resolution", 0.1))
    vertical_lines = float(sim_cfg.get("vertical_lines", 16.0))
    spoofing_rate = float(sim_cfg.get("spoofing_rate", 0.1))

    wall_dist       = float(config["main"].get("wall_dist", 15.0))
    wall_dist_min   = float(config["main"].get("wall_distance_min", 5.0))
    wall_dist_max   = float(config["main"].get("wall_distance_max", 25.0))
    spoofing_cycle  = float(config["main"].get("spoofing_cycle", 2.0))

    if spoofing_mode not in ("removal", "static", "dynamic", "adaptive"):
        print(
            f"[spoofing_editer_lvisam] WARNING: Unknown spoofing_mode '{spoofing_mode}', using 'removal'."
        )
        spoofing_mode = "removal"

    print(
        f"[spoofing_editer_lvisam] mode={spoofing_mode}  "
        f"spoofing_range={spoofing_range}°  "
        f"wall_dist={wall_dist}m  "
        f"geometry_model={static_geometry_model}  "
        f"square_S={square_scale_S}  "
        f"square_rot={square_rotate_rad}  "
        f"M_corr={M_corr}m  "
        f"auto_cycle={auto_cycle}  "
        f"distance_thresh={distance_thresh}m"
    )

    # ── Initialise state (with a seeded RNG for reproducibility) ──────────
    rng_seed = int(config["main"].get("rng_seed", 42))
    rng = np.random.default_rng(rng_seed)
    # For adaptive mode, load pre-computed Hessian eigenvector data
    load_eigenvec = spoofing_mode == "adaptive"
    state = StageDState(config, rng, load_eigenvec=load_eigenvec)

    lidar_topic_length = int(config["main"].get("lidar_topic_length", 22))

    with Reader(config["main"]["input_file"]) as reader, \
         Writer(output_file) as writer:

        connections = {}

        for connection, timestamp, rawdata in reader.messages():

            # ── Create writer connection once per topic ────────────────────────
            if connection.topic not in connections:
                print(
                    f"[spoofing_editer_lvisam] add connection: {connection.topic}"
                )
                connections[connection.topic] = writer.add_connection(
                    connection.topic,
                    connection.msgtype,
                    typestore=typestore,
                )

            # ── Non-pointcloud topics: pass through unchanged ──────────────
            if connection.topic != points_topic:
                writer.write(connections[connection.topic], timestamp, rawdata)
                continue

            # ── PointCloud2 on /points_raw ──────────────────────────────────
            points_msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            state.stats["total_frames"] += 1

            point_step = points_msg.point_step
            n_points   = int(len(points_msg.data) / point_step)
            bin_points = np.frombuffer(
                points_msg.data, dtype=np.uint8
            ).reshape(n_points, point_step)

            robot_x, robot_y, robot_z, robot_yaw, bag_rel_sec = state.lookup(timestamp)

            dist_to_spoofer = np.sqrt(
                (robot_x - spoofer_x) ** 2 + (robot_y - spoofer_y) ** 2
            )
            is_triggered = dist_to_spoofer <= distance_thresh

            if is_triggered:
                state.stats["triggered_frames"] += 1

                # Attack direction in LiDAR local frame
                center_rad = attack_angle_lidar_local(
                    robot_x, robot_y, spoofer_x, spoofer_y, robot_yaw
                )
                # Convert to degrees for spoofing_sim_lvisam.
                # polar_mask_2d uses (atan2(y,x)+180)%360 convention, so we
                # wrap to the same [0, 360) range to match.
                center_deg  = (np.degrees(center_rad) + 180.0) % 360.0

                x, y, z = binary_to_xyz(bin_points)
                n_original = bin_points.shape[0]

                # Compute kept points (needed for stats)
                mask_keep = ~spoofing_sim_lvisam.polar_mask_2d(
                    spoofing_sim_lvisam._read_float32(bin_points, 0, 4),
                    spoofing_sim_lvisam._read_float32(bin_points, 4, 8),
                    center_deg, half_range_deg
                )
                kept = bin_points[mask_keep]

                if spoofing_mode == "removal":
                    modified = spoofing_sim_lvisam.removal_injection(
                        bin_points,
                        center_deg,
                        half_range_deg,
                        rng,
                        point_count_model=point_count_model,
                        horizontal_resolution=horizontal_resolution,
                        vertical_lines=vertical_lines,
                        spoofing_rate=spoofing_rate,
                        point_step=point_step,
                        lidar_scan_period=lidar_scan_period,
                    )

                elif spoofing_mode == "static":
                    modified = spoofing_sim_lvisam.static_injection(
                        bin_points,
                        center_deg,
                        half_range_deg,
                        wall_dist,
                        rng,
                        point_count_model=point_count_model,
                        horizontal_resolution=horizontal_resolution,
                        vertical_lines=vertical_lines,
                        static_geometry_model=static_geometry_model,
                        wall_intensity=wall_intensity,
                        square_scale_S=square_scale_S,
                        square_rotate_rad=square_rotate_rad,
                        point_step=point_step,
                        lidar_scan_period=lidar_scan_period,
                    )

                elif spoofing_mode == "dynamic":
                    modified = spoofing_sim_lvisam.dynamic_injection(
                        bin_points,
                        center_deg,
                        half_range_deg,
                        bag_rel_sec,
                        rng,
                        wall_dist_min=wall_dist_min,
                        wall_dist_max=wall_dist_max,
                        spoofing_cycle=spoofing_cycle,
                        point_count_model=point_count_model,
                        horizontal_resolution=horizontal_resolution,
                        vertical_lines=vertical_lines,
                        static_geometry_model=static_geometry_model,
                        wall_intensity=wall_intensity,
                        square_scale_S=square_scale_S,
                        square_rotate_rad=square_rotate_rad,
                        M_corr=M_corr,
                        lidar_scan_period=lidar_scan_period,
                        auto_cycle=auto_cycle,
                        point_step=point_step,
                    )

                elif spoofing_mode == "adaptive":
                    eigenvec_xyz, dot_eig = state.lookup_eigenvec(timestamp)
                    if eigenvec_xyz is None or dot_eig is None:
                        eigenvec_xyz = np.zeros((1, 3), dtype=np.float64)
                        dot_eig = np.ones(1, dtype=np.float64)
                    modified = spoofing_sim_lvisam.adaptive_injection(
                        bin_points,
                        center_deg,
                        half_range_deg,
                        bag_rel_sec,
                        eigenvec_xyz,
                        dot_eig,
                        rng,
                        base_wall_dist=wall_dist,
                        wall_dist_min=wall_dist_min,
                        wall_dist_max=wall_dist_max,
                        spoofing_cycle=spoofing_cycle,
                        M_corr=M_corr,
                        point_count_model=point_count_model,
                        horizontal_resolution=horizontal_resolution,
                        vertical_lines=vertical_lines,
                        wall_intensity=wall_intensity,
                        square_scale_S=square_scale_S,
                        lidar_scan_period=lidar_scan_period,
                        auto_cycle=auto_cycle,
                        point_step=point_step,
                    )

                # Statistics
                n_injected = max(0, modified.shape[0] - kept.shape[0])
                n_removed  = max(0, kept.shape[0] - modified.shape[0])
                state.stats["removed_points_sum"] += n_removed
                state.stats["removed_points_max"]   = max(
                    state.stats["removed_points_max"], n_removed
                )
                state.stats["injected_points_sum"] += n_injected
                state.stats["injected_points_max"]  = max(
                    state.stats["injected_points_max"], n_injected
                )

            else:
                modified = bin_points

            n_modified = modified.shape[0]

            points_msg.height     = 1
            points_msg.width     = modified.shape[0]
            points_msg.point_step = point_step
            points_msg.row_step  = modified.shape[0] * point_step
            points_msg.data      = modified.reshape(-1)

            rawdata_out = typestore.serialize_ros1(points_msg, connection.msgtype)
            writer.write(connections[connection.topic], timestamp, rawdata_out)

    # ── Print summary ────────────────────────────────────────────────────────
    s = state.stats
    n = s["total_frames"]
    t = s["triggered_frames"]
    print("=" * 54)
    print("  spoofing_editer_lvisam summary")
    print("=" * 54)
    print(f"  Mode                 : {spoofing_mode}")
    print(f"  Total frames         : {n}")
    print(f"  Triggered frames     : {t}  ({100.0 * t / max(n, 1):.4f}%)")
    if t > 0:
        print(
            f"  Mean removed/frame   : {s['removed_points_sum'] / t:.1f}"
        )
        print(
            f"  Max removed (frame)  : {s['removed_points_max']}"
        )
        if spoofing_mode != "removal":
            print(
                f"  Mean injected/frame  : {s['injected_points_sum'] / t:.1f}"
            )
            print(
                f"  Max injected (frame) : {s['injected_points_max']}"
            )
    print(f"  Output bag           : {output_file}")
    print("=" * 54)


if __name__ == "__main__":
    main()
