#!/usr/bin/env python3
"""Convert a Livox official bag to LIO-SAM proxy input topics.

Input:
  /livox/lidar  livox_ros_driver/CustomMsg
  /livox/imu    sensor_msgs/Imu

Output:
  /points_raw   sensor_msgs/PointCloud2 with x/y/z/intensity/ring/time
  /imu_raw      sensor_msgs/Imu

This converter is only for the LIO-SAM proxy graph generation path. It does not
modify the victim FAST-LIVO2 experiment bag.
"""

import argparse
import math
import os

import rosbag
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


FIELDS = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("intensity", 12, PointField.FLOAT32, 1),
    PointField("ring", 16, PointField.UINT16, 1),
    PointField("time", 18, PointField.FLOAT32, 1),
]


def livox_to_cloud(msg, frame_id: str) -> PointCloud2:
    header = msg.header
    header.frame_id = frame_id
    pts = []
    for p in msg.points:
        pts.append((
            float(p.x),
            float(p.y),
            float(p.z),
            float(getattr(p, "reflectivity", 0)),
            int(getattr(p, "line", 0)),
            float(getattr(p, "offset_time", 0)) * 1e-9,
        ))
    cloud = pc2.create_cloud(header, FIELDS, pts)
    cloud.is_dense = True
    return cloud


def ensure_valid_imu_orientation(msg, frame_id: str, fix_invalid: bool) -> bool:
    msg.header.frame_id = frame_id
    q = msg.orientation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm >= 0.1 and math.isfinite(norm):
        q.x /= norm
        q.y /= norm
        q.z /= norm
        q.w /= norm
        return False

    if not fix_invalid:
        return False

    q.x = 0.0
    q.y = 0.0
    q.z = 0.0
    q.w = 1.0
    return True


def convert(args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)

    n_lidar = 0
    n_imu = 0
    n_fixed_imu_orientation = 0
    with rosbag.Bag(args.input, "r") as src, rosbag.Bag(args.output, "w") as dst:
        for topic, msg, t in src.read_messages():
            if topic == args.lidar_topic:
                cloud = livox_to_cloud(msg, args.lidar_frame)
                dst.write(args.output_lidar_topic, cloud, t)
                n_lidar += 1
            elif topic == args.imu_topic:
                fixed = ensure_valid_imu_orientation(
                    msg, args.imu_frame, args.fix_invalid_imu_orientation
                )
                n_fixed_imu_orientation += int(fixed)
                dst.write(args.output_imu_topic, msg, t)
                n_imu += 1
            elif args.keep_clock and topic == "/clock":
                dst.write(topic, msg, t)

    print("[OK] converted Livox bag for LIO-SAM proxy")
    print(f"  input : {args.input}")
    print(f"  output: {args.output}")
    print(f"  lidar frames: {n_lidar}")
    print(f"  imu frames  : {n_imu}")
    print(f"  imu orientation fixed: {n_fixed_imu_orientation}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lidar-topic", default="/livox/lidar")
    p.add_argument("--imu-topic", default="/livox/imu")
    p.add_argument("--output-lidar-topic", default="/points_raw")
    p.add_argument("--output-imu-topic", default="/imu_raw")
    p.add_argument("--lidar-frame", default="base_link")
    p.add_argument("--imu-frame", default="imu_link")
    p.add_argument(
        "--fix-invalid-imu-orientation",
        dest="fix_invalid_imu_orientation",
        action="store_true",
        default=True,
        help=(
            "Replace zero/invalid IMU quaternions with identity orientation. "
            "This is intended for the attacker-side LIO-SAM proxy graph only."
        ),
    )
    p.add_argument(
        "--no-fix-invalid-imu-orientation",
        dest="fix_invalid_imu_orientation",
        action="store_false",
    )
    p.add_argument("--keep-clock", action="store_true")
    return p.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
