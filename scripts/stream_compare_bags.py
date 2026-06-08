#!/usr/bin/env python3
"""Stream-compare ALL point cloud frames between two bags.
Reports progress every 500 frames. Fast because we only read raw data."""
import numpy as np, struct, time, sys
from rosbags.rosbag1 import Reader as R1Reader
from rosbags.typesys.stores import get_typestore, Stores

INPUT_BAG = '/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/jackal.bag'
ATT_BAG   = '/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_jackal/attack_removal/jackal_attack_removal.bag'

ts_store = get_typestore(Stores.ROS1_NOETIC)
PS = 22  # point_step = 22

def stream_compare():
    t0 = time.time()
    same_bytes = 0
    diff_bytes = 0
    same_n = 0
    diff_n = 0
    same_xyz = 0
    diff_xyz = 0
    frame_idx = 0
    first_diff = None
    first_xyz_diff = None

    with R1Reader(INPUT_BAG) as r1, R1Reader(ATT_BAG) as r2:
        # Build index of attack bag (smaller footprint than storing all frames)
        print('Building attack bag index...')
        att_index = {}
        att_topic = None
        for conn, timestamp, rawdata in r2.messages():
            if '/points_raw' in conn.topic:
                att_topic = conn.topic
                att_index[timestamp] = rawdata
        print(f'Attack bag: {len(att_index)} frames indexed')

        # Now stream through input bag and compare
        input_topic = None
        for conn, timestamp, rawdata in r1.messages():
            if '/points_raw' in conn.topic:
                input_topic = conn.topic
                if timestamp not in att_index:
                    continue  # no counterpart

                raw_att = att_index[timestamp]
                frame_idx += 1

                # Quick size check
                sz_in = len(rawdata)
                sz_att = len(raw_att)
                n_in = sz_in // PS
                n_att = sz_att // PS

                same_size = (sz_in == sz_att)
                same_n_this = (n_in == n_att)

                # Byte-level comparison (fast)
                if same_size:
                    same_bytes += 1
                else:
                    diff_bytes += 1

                if same_n_this:
                    same_n += 1
                else:
                    diff_n += 1
                    if first_diff is None:
                        first_diff = (frame_idx, timestamp, n_in, n_att, sz_in, sz_att)
                        t_first = time.time()
                        print(f'\nFIRST DIFF at frame {frame_idx}: n={n_in}->{n_att}, size={sz_in}->{sz_att}')
                        print(f'  (took {t_first-t0:.1f}s, at t={timestamp})')

                # If sizes match, do xyz comparison
                if same_size:
                    arr_in = np.frombuffer(rawdata, dtype=np.uint8)
                    arr_att = np.frombuffer(raw_att, dtype=np.uint8)

                    # xyz at offset 0, 4, 8
                    n_pts = n_in
                    # Compare first 1000 points only for speed (xyz check)
                    check_n = min(n_pts, 1000)

                    # Compute xyz for both
                    xs_in = arr_in[:check_n*PS:PS].view(np.float32)
                    ys_in = arr_in[4:check_n*PS+4:PS].view(np.float32)
                    zs_in = arr_in[8:check_n*PS+8:PS].view(np.float32)

                    xs_att = arr_att[:check_n*PS:PS].view(np.float32)
                    ys_att = arr_att[4:check_n*PS+4:PS].view(np.float32)
                    zs_att = arr_att[8:check_n*PS+8:PS].view(np.float32)

                    xyz_diff = np.maximum(np.abs(xs_in - xs_att),
                                         np.maximum(np.abs(ys_in - ys_att),
                                                    np.abs(zs_in - zs_att)))
                    has_xyz_diff = (xyz_diff > 1e-6).any()

                    if has_xyz_diff:
                        diff_xyz += 1
                        if first_xyz_diff is None:
                            first_xyz_diff = (frame_idx, timestamp)
                            # Find which points differ
                            diff_pts = np.where(xyz_diff > 1e-6)[0]
                            print(f'FIRST XYZ DIFF at frame {frame_idx}: ts={timestamp}')
                            print(f'  Changed points (first 3): {diff_pts[:3]}')
                            for dp in diff_pts[:3]:
                                print(f'    point[{dp}]: in=({xs_in[dp]:.3f},{ys_in[dp]:.3f},{zs_in[dp]:.3f})  att=({xs_att[dp]:.3f},{ys_att[dp]:.3f},{zs_att[dp]:.3f})')
                    else:
                        same_xyz += 1

                # Progress
                if frame_idx % 500 == 0:
                    elapsed = time.time() - t0
                    rate = frame_idx / elapsed
                    remaining = (31718 - frame_idx) / rate if rate > 0 else 0
                    print(f'  frame {frame_idx}/31718 ({100*frame_idx/31718:.1f}%)  '
                          f'same_bytes={same_bytes}  diff_bytes={diff_bytes}  '
                          f'same_n={same_n}  diff_n={diff_n}  '
                          f'same_xyz={same_xyz}  diff_xyz={diff_xyz}  '
                          f'ETA={remaining/60:.1f}min', flush=True)

                # Early exit after full pass
                if frame_idx >= 31718:
                    break

    elapsed = time.time() - t0
    print(f'\n=== COMPLETE ({elapsed:.1f}s) ===')
    print(f'Total frames:   {frame_idx}')
    print(f'Same bytes:    {same_bytes}')
    print(f'Diff bytes:    {diff_bytes}')
    print(f'Same n_pts:    {same_n}')
    print(f'Diff n_pts:    {diff_n}')
    print(f'Same xyz:      {same_xyz}')
    print(f'Diff xyz:      {diff_xyz}')
    print(f'First n_pts diff: {first_diff}')
    print(f'First xyz diff:   {first_xyz_diff}')

if __name__ == '__main__':
    stream_compare()
