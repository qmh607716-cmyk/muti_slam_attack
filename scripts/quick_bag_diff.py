#!/usr/bin/env python3
"""Quick check: compare frame count and total bytes per timestamp."""
import time
from rosbags.rosbag1 import Reader as R1Reader

INPUT_BAG = '/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/jackal.bag'
ATT_BAG   = '/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_jackal/attack_removal/jackal_attack_removal.bag'

def index_bag(bag_path):
    """Index bag: ts -> (n_msgs, total_bytes)"""
    print(f'Indexing {bag_path.split("/")[-1]}...')
    t0 = time.time()
    info = {}
    topic_count = {}
    with R1Reader(bag_path) as r:
        for conn, timestamp, rawdata in r.messages():
            topic = conn.topic
            if topic not in topic_count:
                topic_count[topic] = 0
            topic_count[topic] += 1
            if '/points_raw' in topic:
                info[timestamp] = len(rawdata)
    elapsed = time.time() - t0
    total_bytes = sum(info.values())
    return info, topic_count, elapsed, total_bytes

info1, tc1, e1, b1 = index_bag(INPUT_BAG)
info2, tc2, e2, b2 = index_bag(ATT_BAG)

print(f'\n=== Topic counts ===')
all_topics = set(tc1.keys()) | set(tc2.keys())
for t in sorted(all_topics):
    c1 = tc1.get(t, 0)
    c2 = tc2.get(t, 0)
    if c1 != c2:
        print(f'  {t}: INPUT={c1}  ATTACK={c2}  DIFF={c2-c1}')
    else:
        print(f'  {t}: {c1} (same)')

print(f'\n=== PointCloud2 frames ===')
common_ts = sorted(info1.keys() & info2.keys())
print(f'Input frames:   {len(info1)}')
print(f'Attack frames:  {len(info2)}')
print(f'Common frames:  {len(common_ts)}')

print(f'Input bytes:    {b1/1e9:.2f} GB')
print(f'Attack bytes:   {b2/1e9:.2f} GB')
print(f'Byte diff:      {(b2-b1)/1e9:.2f} GB')

# Per-frame size comparison (just first 200 to gauge)
n_same_size = 0
n_diff_size = 0
same_size_ts = []
diff_size_ts = []

for ts in common_ts[:500]:
    if info1[ts] == info2[ts]:
        n_same_size += 1
        same_size_ts.append(ts)
    else:
        n_diff_size += 1
        diff_size_ts.append(ts)

print(f'\n=== First 500 common frames (size comparison only) ===')
print(f'Same size:  {n_same_size}')
print(f'Diff size:  {n_diff_size}')

if diff_size_ts:
    print(f'First different frame: ts={diff_size_ts[0]}')
    print(f'  Input size:  {info1[diff_size_ts[0]]}')
    print(f'  Attack size: {info2[diff_size_ts[0]]}')
    # Show surrounding frames
    idx = common_ts.index(diff_size_ts[0])
    for j in range(max(0,idx-2), min(len(common_ts), idx+3)):
        ts = common_ts[j]
        d1 = info1.get(ts, 'N/A')
        d2 = info2.get(ts, 'N/A')
        print(f'  ts={ts}: in={d1}, att={d2}, same={d1==d2}')
