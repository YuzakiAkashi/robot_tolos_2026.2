#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np

from geometry_msgs.msg import PoseArray, Pose, PointStamped
import tf2_ros
import tf2_geometry_msgs


class LineProjector:

    def __init__(self):

        rospy.init_node("cone_line_projection")

        # 订阅 DBSCAN 输出
        rospy.Subscriber("/cones",
                         PoseArray,
                         self.callback,
                         queue_size=1)

        # 发布投影结果
        self.pub = rospy.Publisher("/cones_projected",
                                   PoseArray,
                                   queue_size=10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 线段端点（odom坐标系）
        self.A = np.array([
            5.210430145263672,
            3.891853094100952
        ])

        self.B = np.array([
            10.525156021118164,
            3.8432748317718506
        ])

        self.d = self.B - self.A
        self.dd = np.dot(self.d, self.d)

        rospy.loginfo("cone line projection node started")

    # 点从 base_link → odom
    def transform_point(self, x, y, frame_id):

        pt = PointStamped()

        pt.header.frame_id = frame_id
        pt.header.stamp = rospy.Time(0)

        pt.point.x = x
        pt.point.y = y
        pt.point.z = 0.0

        try:

            transform = self.tf_buffer.lookup_transform(
                "odom",
                frame_id,
                rospy.Time(0),
                rospy.Duration(0.2)
            )

            pt_out = tf2_geometry_msgs.do_transform_point(pt, transform)

            return np.array([
                pt_out.point.x,
                pt_out.point.y
            ])

        except Exception as e:

            rospy.logwarn("TF transform failed: %s", str(e))
            return None

    # 投影到线段
    def project_point(self, p):

        ap = p - self.A

        t = np.dot(ap, self.d) / self.dd

        # 限制在线段AB之间
        if t < 0.0 or t > 1.0:
            return None

        proj = self.A + t * self.d

        return proj

    def callback(self, msg):

        out = PoseArray()

        out.header.stamp = rospy.Time.now()
        out.header.frame_id = "odom"

        for pose in msg.poses:

            # 1. 坐标系转换
            p = self.transform_point(
                pose.position.x,
                pose.position.y,
                msg.header.frame_id
            )

            if p is None:
                continue

            # 2. 投影
            proj = self.project_point(p)

            new_pose = Pose()

            new_pose.position.x = proj[0]
            new_pose.position.y = proj[1]
            new_pose.position.z = 0.0

            new_pose.orientation.w = 1.0

            out.poses.append(new_pose)

        if len(out.poses) > 0:
            self.pub.publish(out)


if __name__ == "__main__":

    try:
        LineProjector()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass