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
            5.21,
            3.85
        ])

        self.B = np.array([
            10.52,
            3.85
        ])

        self.d = self.B - self.A
        self.dd = np.dot(self.d, self.d)

        # 直线长度
        self.line_length = np.linalg.norm(self.d)
        
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
    def transform_point(self, x, y, frame):

        try:
            trans = self.tf_buffer.lookup_transform(
                "odom",
                frame,
                rospy.Time(0),
                rospy.Duration(0.1)
            )

            pt = PointStamped()
            pt.header.frame_id = frame
            pt.point.x = x
            pt.point.y = y

            pt = tf2_geometry_msgs.do_transform_point(pt, trans)

            return np.array([pt.point.x, pt.point.y])

        except:
            return None

    def callback(self, msg):

        proj_points = []
        t_values = []

        for pose in msg.poses:

            p = self.transform_point(
                pose.position.x,
                pose.position.y,
                msg.header.frame_id
            )

            if p is None:
                continue

            ap = p - self.A
            t = np.dot(ap, self.d) / self.dd

            # 丢弃线段外点
            if t < 0.0 or t > 1.0:
                continue

            proj = self.A + t * self.d

            proj_points.append(proj)
            t_values.append(t)

        if len(proj_points) == 0:
            return

        # 按直线方向排序
        order = np.argsort(t_values)
        proj_points = [proj_points[i] for i in order]
        t_values = [t_values[i] for i in order]

        labels = ["A"] * len(proj_points)

        # 用t距离判断
        for i in range(len(t_values) - 1):

            dist = abs(t_values[i+1] - t_values[i]) * self.line_length

            if dist <= 0.75:
                labels[i] = "B"
                labels[i+1] = "B"

        out = PoseArray()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = "odom"

        for p, label in zip(proj_points, labels):

            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]

            # z用来标记类型
            if label == "A":
                pose.position.z = 0.0
            else:
                pose.position.z = 1.0

            pose.orientation.w = 1.0
            out.poses.append(pose)

        self.pub.publish(out)


if __name__ == "__main__":

    try:
        LineProjector()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass