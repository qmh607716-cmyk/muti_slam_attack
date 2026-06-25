#!/usr/bin/env python3
"""bag2csv.py — Convert nav_msgs/Odometry bags to CSV for evaluate_attack.py"""
import os, sys, rosbag, argparse, pandas as pd

def bag_to_csv(bag_path: str, out_path: str):
    """Extract Odometry poses from bag → CSV (time,x,y,z,qx,qy,qz,qw)."""
    ts_list, x_list, y_list, z_list = [], [], [], []
    qx_list, qy_list, qz_list, qw_list = [], [], [], []

    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages():
            try:
                ts_list.append(float(msg.header.stamp.to_sec()))
                x_list.append(msg.pose.pose.position.x)
                y_list.append(msg.pose.pose.position.y)
                z_list.append(msg.pose.pose.position.z)
                qx_list.append(msg.pose.pose.orientation.x)
                qy_list.append(msg.pose.pose.orientation.y)
                qz_list.append(msg.pose.pose.orientation.z)
                qw_list.append(msg.pose.pose.orientation.w)
            except AttributeError:
                pass

    df = pd.DataFrame({
        "time": ts_list,
        "x": x_list, "y": y_list, "z": z_list,
        "qx": qx_list, "qy": qy_list, "qz": qz_list, "qw": qw_list,
    })
    df.to_csv(out_path, index=False)
    print(f"  [{len(df)} poses] {bag_path}\n  → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="Input .bag file")
    ap.add_argument("-o", "--out", required=True, help="Output .csv file")
    args = ap.parse_args()
    bag_to_csv(args.bag, args.out)
