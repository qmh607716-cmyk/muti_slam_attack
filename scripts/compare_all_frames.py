#!/usr/bin/env python3
"""Compare ALL point cloud frames between input and attack bags."""
import numpy as np, struct, time
from rosbags.rosbag1 import Reader as R1Reader
from rosbags.typesys.stores import get_typestore, Stores

INPUT_BAG = '/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/jackal.bag'
ATT_BAG   = '/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_jackal/attack_removal/jackal_attack_removal.bag'

ts_store = get_typestore(Stores.ROS1_NOETIC)

def extract_xyz(msg):
    """Fast xyz extraction from PointCloud2."""
    n = msg.width * msg.height
    ps = msg.point_step
    d = bytes(msg.data)
    xs = np.empty(n, dtype=np.float32)
    ys = np.empty(n, dtype=np.float32)
    zs = np.empty(n, dtype=np.float32)
    for i in range(n):
        xs[i] = struct.unpack_from('<f', d, i*ps)[0]
        ys[i] = struct.unpack_from('<f', d, i*ps+4)[0]
        zs[i] = struct.unpack_from('<f', d, i*ps+8)[0]
    return xs, ys, zs

def read_all_pts(bag_path):
    """Read all /points_raw frames from bag as {timestamp: (n, total_bytes)}."""
    print(f'  Reading {bag_path.split("/")[-1]}...')
    t0 = time.time()
    frames = {}
    with R1Reader(bag_path) as r:
        topic = None
        for conn, timestamp, rawdata in r.messages():
            if '/points_raw' in conn.topic:
                if topic is None:
                    topic = conn.topic
                    print(f'  Topic: {topic}, msgtype: {conn.msgtype}')
                msg = ts_store.deserialize_ros1(rawdata, conn.msgtype)
                n = msg.width * msg.height
                frames[timestamp] = (n, len(msg.data))
    print(f'  Read {len(frames)} frames in {time.time()-t0:.1f}s')
    return frames

print('=== Step 1: Read all frame metadata ===')
frames1 = read_all_pts(INPUT_BAG)
frames2 = read_all_pts(ATT_BAG)

common = sorted(frames1.keys() & frames2.keys())
only_in1 = sorted(frames1.keys() - frames2.keys())
only_in2 = sorted(frames2.keys() - frames1.keys())

print(f'\nCommon timestamps: {len(common)}')
print(f'Only in input:     {len(only_in1)}')
print(f'Only in attack:    {len(only_in2)}')

if len(only_in1) > 0:
    print(f'  First few only-in-input: {only_in1[:3]}')
if len(only_in2) > 0:
    print(f'  First few only-in-attack: {only_in2[:3]}')

# Check frame count match
if len(common) < min(len(frames1), len(frames2)):
    print('\nWARNING: Some frames have no counterpart!')

print(f'\n=== Step 2: Compare all {len(common)} frames ===')

# Sort by timestamp and compare
same_count = 0
n_pts_diff_count = 0
xyz_diff_count = 0
first_diff = None

d1_map = dict(frames1)
d2_map = dict(frames2)

# Progress reporting every 1000 frames
for i, ts in enumerate(common):
    n1, sz1 = d1_map[ts]
    n2, sz2 = d2_map[ts]

    if n1 != n2:
        n_pts_diff_count += 1
        if first_diff is None and n1 != n2:
            first_diff = ('n_pts', i, ts, n1, n2, sz1, sz2)

    if sz1 != sz2:
        if first_diff is None:
            first_diff = ('size', i, ts, n1, n2, sz1, sz2)

    if i > 0 and i % 1000 == 0:
        elapsed = i / len(common) * 100
        print(f'  Progress: {i}/{len(common)} ({elapsed:.1f}%)  same={same_count}, n_pts_diff={n_pts_diff_count}, xyz_diff={xyz_diff_count}')

print(f'\n=== Results ===')
print(f'Total compared: {len(common)}')
print(f'Frame count same, size same:  {same_count}')
print(f'Frame count different:       {n_pts_diff_count}')
print(f'Frame count same, size diff:  {same_count}')
print(f'xyz different:                {xyz_diff_count}')
print(f'First difference: {first_diff}')
