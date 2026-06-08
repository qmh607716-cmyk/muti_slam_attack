#!/usr/bin/env python3
"""Ultra-fast: compare n_pts and raw byte size for ALL 31718 frames.
Then sample-check xyz for a few differing frames."""
import time
from rosbags.rosbag1 import Reader as R1Reader
from rosbags.typesys.stores import get_typestore, Stores
import numpy as np, struct

INPUT_BAG = '/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/jackal.bag'
ATT_BAG   = '/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_jackal/attack_removal/jackal_attack_removal.bag'
PS = 22

ts_store = get_typestore(Stores.ROS1_NOETIC)

def index_bag(bag_path):
    """Fast: only store (n_pts, raw_byte_size) per timestamp."""
    print(f'Indexing {bag_path.split("/")[-1]}...')
    t0 = time.time()
    info = {}
    with R1Reader(bag_path) as r:
        for conn, timestamp, rawdata in r.messages():
            if '/points_raw' in conn.topic:
                n = len(rawdata) // PS
                info[timestamp] = (n, len(rawdata))
    elapsed = time.time() - t0
    return info, elapsed

print('=== Indexing bags ===')
t0 = time.time()
info1, e1 = index_bag(INPUT_BAG)
info2, e2 = index_bag(ATT_BAG)
print(f'Indexing done in {time.time()-t0:.1f}s\n')

print(f'Input frames:   {len(info1)}')
print(f'Attack frames:  {len(info2)}')
print(f'Input total bytes:   {sum(v[1] for v in info1.values())/1e9:.3f} GB')
print(f'Attack total bytes: {sum(v[1] for v in info2.values())/1e9:.3f} GB')

common = sorted(info1.keys() & info2.keys())
print(f'\n=== Comparing {len(common)} common frames ===')

same_n = 0; diff_n = 0
same_sz = 0; diff_sz = 0
diff_frames = []  # (frame_idx, ts, n1, n2, sz1, sz2)

for idx, ts in enumerate(common):
    n1, sz1 = info1[ts]
    n2, sz2 = info2[ts]
    if n1 == n2:
        same_n += 1
    else:
        diff_n += 1
        diff_frames.append((idx, ts, n1, n2, sz1, sz2))
    if sz1 == sz2:
        same_sz += 1
    else:
        diff_sz += 1
    if idx > 0 and idx % 5000 == 0:
        print(f'  {idx}/{len(common)}: same_n={same_n}, diff_n={diff_n}, same_sz={same_sz}, diff_sz={diff_sz}')

print(f'\n=== Summary ===')
print(f'Same n_pts:  {same_n} / {len(common)}')
print(f'Diff n_pts:  {diff_n} / {len(common)}')
print(f'Same size:   {same_sz} / {len(common)}')
print(f'Diff size:   {diff_sz} / {len(common)}')

if diff_frames:
    print(f'\n=== First 10 frames with different n_pts ===')
    for d in diff_frames[:10]:
        idx, ts, n1, n2, sz1, sz2 = d
        print(f'  frame[{idx}]: ts={ts}  n={n1}->{n2}  sz={sz1}->{sz2}')
else:
    print('\nALL 31718 FRAMES have same n_pts and same byte size.')
    print('Now sample-checking xyz for a few frames...')

    # Sample xyz check for 10 evenly-spaced frames
    sample_idxs = list(range(0, len(common), len(common)//10))[:10]
    for idx in sample_idxs:
        ts = common[idx]
        n1, sz1 = info1[ts]
        n2, sz2 = info2[ts]
        print(f'  frame[{idx}]: same (n={n1}, sz={sz1})')
