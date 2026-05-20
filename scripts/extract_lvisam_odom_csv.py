#!/usr/bin/env python3
import argparse
import csv
import rosbag
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--bag", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--topic", default="/lvi_sam/lidar/mapping/odometry")
args = parser.parse_args()

bag_path = Path(args.bag)
csv_path = Path(args.out)

rows = []

with rosbag.Bag(str(bag_path), "r") as bag:
    for tp, msg, t in bag.read_messages(topics=[args.topic]):
        stamp = msg.header.stamp.to_sec()
        if stamp == 0:
            stamp = t.to_sec()

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        rows.append([
            stamp,
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w,
        ])

if not rows:
    raise RuntimeError(f"No odometry messages found in {bag_path}")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "x", "y", "z", "qx", "qy", "qz", "qw"])
    writer.writerows(rows)

print(f"[OK] {csv_path}: {len(rows)} poses")
print("first time:", rows[0][0])
print("last time: ", rows[-1][0])
