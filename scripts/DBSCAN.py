#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
import tf2_ros
import tf2_geometry_msgs
from sklearn.cluster import DBSCAN
import math
import random


class ConeDetector:

    def __init__(self):

        rospy.init_node('cone_detector', anonymous=True)

        self.scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.output_frame = rospy.get_param('~output_frame', 'base_link')

        self.cluster_epsilon = rospy.get_param('~cluster_epsilon', 0.15)
        self.min_points = rospy.get_param('~min_points', 4)

        self.max_range = rospy.get_param('~max_range', 5.0)
        self.min_range = rospy.get_param('~min_range', 0.2)

        # 圆半径范围（锥桶过滤）
        self.min_radius = rospy.get_param('~min_radius', 0.05)
        self.max_radius = rospy.get_param('~max_radius', 0.25)

        # RANSAC参数
        self.ransac_iter = rospy.get_param('~ransac_iter', 30)
        self.ransac_thresh = rospy.get_param('~ransac_thresh', 0.03)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.scan_sub = rospy.Subscriber(self.scan_topic,
                                         LaserScan,
                                         self.scan_callback,
                                         queue_size=1)

        self.cone_pub = rospy.Publisher('/cones', PoseArray, queue_size=10)

        rospy.loginfo("cone detector started")

    # ------------------------------------------
    # LaserScan → Cartesian
    # ------------------------------------------

    def scan_to_cartesian(self, scan_msg):

        angles = np.linspace(scan_msg.angle_min,
                             scan_msg.angle_max,
                             len(scan_msg.ranges))

        ranges = np.array(scan_msg.ranges)

        valid = (ranges > self.min_range) & \
                (ranges < self.max_range) & \
                np.isfinite(ranges)

        ranges = ranges[valid]
        angles = angles[valid]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        return np.vstack((x, y)).T

    # ------------------------------------------
    # DBSCAN
    # ------------------------------------------

    def cluster_points(self, points):

        if len(points) < self.min_points:
            return np.array([])

        clustering = DBSCAN(eps=self.cluster_epsilon,
                            min_samples=self.min_points).fit(points)

        return clustering.labels_

    # ------------------------------------------
    # 三点确定圆
    # ------------------------------------------

    def fit_circle_3pts(self, p1, p2, p3):

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        temp = x2**2 + y2**2
        bc = (x1**2 + y1**2 - temp) / 2
        cd = (temp - x3**2 - y3**2) / 2

        det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)

        if abs(det) < 1e-6:
            return None

        cx = (bc*(y2-y3) - cd*(y1-y2)) / det
        cy = ((x1-x2)*cd - (x2-x3)*bc) / det

        r = math.sqrt((cx-x1)**2 + (cy-y1)**2)

        return np.array([cx, cy]), r

    # ------------------------------------------
    # RANSAC圆拟合
    # ------------------------------------------

    def ransac_circle(self, points):

        best_center = None
        best_inliers = 0
        best_radius = None

        if len(points) < 3:
            return None, None

        for _ in range(self.ransac_iter):

            ids = random.sample(range(len(points)), 3)

            res = self.fit_circle_3pts(points[ids[0]],
                                       points[ids[1]],
                                       points[ids[2]])

            if res is None:
                continue

            center, r = res

            if not (self.min_radius < r < self.max_radius):
                continue

            d = np.linalg.norm(points - center, axis=1)

            inliers = np.sum(np.abs(d - r) < self.ransac_thresh)

            if inliers > best_inliers:
                best_inliers = inliers
                best_center = center
                best_radius = r

        return best_center, best_radius

    # ------------------------------------------
    # TF变换
    # ------------------------------------------

    def transform_point(self, point, from_frame, to_frame, time):

        try:

            transform = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                time,
                rospy.Duration(0.1))

            pt = tf2_geometry_msgs.PointStamped()

            pt.header.frame_id = from_frame
            pt.header.stamp = time

            pt.point.x = point[0]
            pt.point.y = point[1]
            pt.point.z = 0.0

            trans = tf2_geometry_msgs.do_transform_point(pt, transform)

            return np.array([trans.point.x, trans.point.y])

        except Exception as e:

            rospy.logwarn_throttle(5, str(e))
            return None

    # ------------------------------------------
    # callback
    # ------------------------------------------

    def scan_callback(self, scan_msg):

        points = self.scan_to_cartesian(scan_msg)

        if len(points) == 0:
            return

        labels = self.cluster_points(points)

        if len(labels) == 0:
            return

        unique_labels = set(labels)

        cone_poses = []

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.output_frame

        for label in unique_labels:

            if label == -1:
                continue

            cluster = points[labels == label]

            if len(cluster) < 4:
                continue

            center, radius = self.ransac_circle(cluster)

            if center is None:
                continue

            if self.output_frame != scan_msg.header.frame_id:

                center = self.transform_point(center,
                                              scan_msg.header.frame_id,
                                              self.output_frame,
                                              scan_msg.header.stamp)

                if center is None:
                    continue

            pose = Pose()

            pose.position.x = center[0]
            pose.position.y = center[1]
            pose.position.z = 0.0

            pose.orientation.w = 1.0

            cone_poses.append(pose)

        if cone_poses:

            pose_array = PoseArray()
            pose_array.header = header
            pose_array.poses = cone_poses

            self.cone_pub.publish(pose_array)


if __name__ == '__main__':

    try:

        ConeDetector()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass