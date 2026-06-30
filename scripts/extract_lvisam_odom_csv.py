#!/usr/bin/env python3
import argparse
import csv
import rosbag
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Extract odometry from bag as CSV. "
                "Auto-detects LVI-SAM vs LIO-SAM topic."
)
parser.add_argument("--bag", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--topic", default="auto",
                    help="Topic to extract. 'auto' detects LVI-SAM or LIO-SAM automatically.")
args = parser.parse_args()

bag_path = Path(args.bag)
csv_path = Path(args.out)

# ── Auto-detect topic if requested ─────────────────────────────────────────
def _detect_topic(bag_path):
    topics_found = set()
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, _, _ in bag.read_messages():
            topics_found.add(topic)

    candidates = [
        "/lvi_sam/lidar/mapping/odometry",
        "/lio_sam/mapping/odometry",
    ]
    for t in candidates:
        if t in topics_found:
            return t
    raise RuntimeError(
        f"No supported odometry topic found in {bag_path}.\n"
        f"Available topics: {sorted(topics_found)}"
    )

if args.topic == "auto":
    topic = _detect_topic(bag_path)
    print(f"[OK] Auto-detected topic: {topic}")
else:
    topic = args.topic

rows = []

with rosbag.Bag(str(bag_path), "r") as bag:
    for _, msg, t in bag.read_messages(topics=[topic]):
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
    raise RuntimeError(f"No odometry messages found in {bag_path} on topic {topic}")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "x", "y", "z", "qx", "qy", "qz", "qw"])
    writer.writerows(rows)

print(f"[OK] {csv_path}: {len(rows)} poses")
print("  topic :", topic)
print("  first :", rows[0][0])
print("  last  :", rows[-1][0])
