#!/usr/bin/env python3
"""Adapt handheld PointCloud2 fields for R3LIVE.

The handheld bag uses x/y/z/intensity/ring/time with point_step=22. R3LIVE's
mapping node accepts pcl::PointXYZINormal, where curvature stores the per-point
time offset in milliseconds. This node keeps geometry/intensity unchanged and
maps time seconds -> curvature milliseconds.
"""

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


FIELDS_XYZINORMAL = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("intensity", 12, PointField.FLOAT32, 1),
    PointField("normal_x", 16, PointField.FLOAT32, 1),
    PointField("normal_y", 20, PointField.FLOAT32, 1),
    PointField("normal_z", 24, PointField.FLOAT32, 1),
    PointField("curvature", 28, PointField.FLOAT32, 1),
]


class R3LivePointsAdapter:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/points_raw")
        self.output_topic = rospy.get_param(
            "~output_topic", "/r3live/points_raw_xyzinormal"
        )
        self.time_scale = float(rospy.get_param("~time_to_curvature_scale", 1000.0))
        self.skip_nans = bool(rospy.get_param("~skip_nans", True))

        self.pub = rospy.Publisher(
            self.output_topic, PointCloud2, queue_size=3
        )
        self.sub = rospy.Subscriber(
            self.input_topic, PointCloud2, self.callback, queue_size=3
        )
        rospy.loginfo(
            "r3live_points_adapter: %s -> %s, curvature=time*%.3f",
            self.input_topic,
            self.output_topic,
            self.time_scale,
        )

    def callback(self, msg):
        try:
            src_points = pc2.read_points(
                msg,
                field_names=("x", "y", "z", "intensity", "time"),
                skip_nans=self.skip_nans,
            )
            out_points = [
                (x, y, z, intensity, 0.0, 0.0, 0.0, time_offset * self.time_scale)
                for x, y, z, intensity, time_offset in src_points
            ]
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0, "r3live_points_adapter failed to convert cloud: %s", exc
            )
            return

        out_msg = pc2.create_cloud(msg.header, FIELDS_XYZINORMAL, out_points)
        out_msg.is_dense = msg.is_dense and self.skip_nans
        self.pub.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("r3live_points_adapter")
    R3LivePointsAdapter()
    rospy.spin()
