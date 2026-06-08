#!/usr/bin/env python3
"""Fast bag comparison using rosbags random access."""
import numpy as np

from rosbags.rosbag1 import Reader as R1Reader
from rosbags.rosbag2 import Reader as R2Reader
from rosbags.types import Typestore

INPUT_BAG = '/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/jackal.bag'
ATT_BAG   = '/home/qu_menghao/catkin_ws/src/LVI-SAM/datasets/slamspoof_jackal/attack_removal/jackal_attack_removal.bag'

typestore = Typestore()

def read_points(msg_bytes, msgtype, point_step):
    """Deserialize PointCloud2 and extract xyz as numpy array."""
    msg = typestore.deserialize_ros1(msg_bytes, msgtype)
    n = len(msg.data) // point_step
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, point_step)
    return arr

def compare_bags(bag1_path, bag2_path, max_frames=30):
    """Compare pointcloud frames between two bags."""
    ts1 = {}
    ts2 = {}

    with R1Reader(bag1_path) as r1, R1Reader(bag2_path) as r2:
        # Find /points_raw topic
        pt_topic1 = None
        pt_topic2 = None
        msgtype1 = None
        msgtype2 = None
        point_step1 = 22
        point_step2 = 22

        for c in r1.connections:
            if '/points_raw' in c.topic:
                pt_topic1 = c.topic
                msgtype1 = c.msgtype
                break
        for c in r2.connections:
            if '/points_raw' in c.topic:
                pt_topic2 = c.topic
                msgtype2 = c.msgtype
                break

        if pt_topic1 is None or pt_topic2 is None:
            print("ERROR: /points_raw topic not found")
            print(f"  Bag1 topics: {[c.topic for c in r1.connections]}")
            print(f"  Bag2 topics: {[c.topic for c in r2.connections]}")
            return

        print(f"Bag1 topic: {pt_topic1}, msgtype: {msgtype1}")
        print(f"Bag2 topic: {pt_topic2}, msgtype: {msgtype2}")

        # Read frames
        count = 0
        for conn, timestamp, rawdata in r1.messages():
            if conn.topic == pt_topic1:
                ts1[timestamp] = rawdata
                count += 1
                if count >= max_frames:
                    break

        count = 0
        for conn, timestamp, rawdata in r2.messages():
            if conn.topic == pt_topic2:
                ts2[timestamp] = rawdata
                count += 1
                if count >= max_frames:
                    break

    common_ts = sorted(set(ts1.keys()) & set(ts2.keys()))
    print(f"\n前 {max_frames} 帧对比 (timestamp 对齐):")
    print(f"  Bag1 帧数: {len(ts1)}, Bag2 帧数: {len(ts2)}, 共同: {len(common_ts)}")

    n_same = 0
    n_diff = 0
    for ts in common_ts[:20]:
        d1 = ts1[ts]
        d2 = ts2[ts]

        arr1 = np.frombuffer(d1, dtype=np.uint8).reshape(-1, point_step1)
        arr2 = np.frombuffer(d2, dtype=np.uint8).reshape(-1, point_step2)

        same = (arr1.shape == arr2.shape and np.array_equal(arr1, arr2))
        if same:
            n_same += 1
            status = "✅ 相同"
        else:
            n_diff += 1
            # 计算 xyz 差异
            xyz1 = arr1[:, :12].view(np.float32)[:, :3]
            n2 = min(len(arr1), len(arr2))
            xyz2 = arr2[:n2, :12].view(np.float32)[:, :3]
            max_diff = float(np.abs(xyz1[:n2] - xyz2).max())
            mean_diff = float(np.abs(xyz1[:n2] - xyz2).mean())
            n_pts_diff = abs(len(arr1) - len(arr2))
            status = f"❌ 不同 (n_pts Δ={n_pts_diff}, xyz max Δ={max_diff:.4f}m)"

        print(f"  ts={ts}: arr1={arr1.shape}, arr2={arr2.shape}  {status}")

    print(f"\n统计: 相同={n_same}, 不同={n_diff}")

    # 检查是否有任何帧被修改
    if n_diff == 0:
        print("\n✅ 所有对比帧完全相同 - 攻击从未修改点云")
    else:
        print(f"\n❌ 发现 {n_diff} 帧被修改")

if __name__ == '__main__':
    print("=== 对比输入 bag vs 攻击 bag (前30帧) ===\n")
    compare_bags(INPUT_BAG, ATT_BAG, max_frames=30)
