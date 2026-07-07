#!/usr/bin/env python3
"""Adapt handheld PointCloud2 timing for FAST-LIVO2.

The handheld bags store Velodyne-style point fields as x/y/z/intensity/ring/time,
with ``time`` in seconds within the scan. FAST-LIVO2's Velodyne preprocessor
expects the ``time`` field in microseconds before it converts to milliseconds
internally. This node keeps geometry, intensity, and ring unchanged, and scales
only the per-point time field.
"""

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


FIELDS_VELODYNE_TIME_USEC = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("intensity", 12, PointField.FLOAT32, 1),
    PointField("time", 16, PointField.FLOAT32, 1),
    PointField("ring", 20, PointField.UINT16, 1),
]


class FastLivo2PointsAdapter:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/points_raw")
        self.output_topic = rospy.get_param("~output_topic", "/fast_livo2/points_raw")
        self.time_scale = float(rospy.get_param("~time_scale", 1.0e6))
        self.skip_nans = bool(rospy.get_param("~skip_nans", True))

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=3)
        self.sub = rospy.Subscriber(
            self.input_topic, PointCloud2, self.callback, queue_size=3
        )
        rospy.loginfo(
            "fast_livo2_points_adapter: %s -> %s, time*=%.3f",
            self.input_topic,
            self.output_topic,
            self.time_scale,
        )

    def callback(self, msg):
        try:
            src_points = pc2.read_points(
                msg,
                field_names=("x", "y", "z", "intensity", "ring", "time"),
                skip_nans=self.skip_nans,
            )
            out_points = [
                (x, y, z, intensity, time_offset * self.time_scale, int(ring))
                for x, y, z, intensity, ring, time_offset in src_points
            ]
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0, "fast_livo2_points_adapter failed to convert cloud: %s", exc
            )
            return

        out_msg = pc2.create_cloud(msg.header, FIELDS_VELODYNE_TIME_USEC, out_points)
        out_msg.is_dense = msg.is_dense and self.skip_nans
        self.pub.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("fast_livo2_points_adapter")
    FastLivo2PointsAdapter()
    rospy.spin()
