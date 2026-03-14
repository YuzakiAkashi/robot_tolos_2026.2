#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import random

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header

from sklearn.cluster import DBSCAN

import tf2_ros
import tf2_geometry_msgs


class ConeDetector:

    def __init__(self):

        rospy.init_node("cone_detector")

        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.output_frame = rospy.get_param("~output_frame", "base_link")

        self.cluster_eps = rospy.get_param("~cluster_eps", 0.12)
        self.min_points = rospy.get_param("~min_points", 5)

        self.min_range = rospy.get_param("~min_range", 0.5)
        self.max_range = rospy.get_param("~max_range", 5.0)

        self.min_radius = rospy.get_param("~min_radius", 0.05)
        self.max_radius = rospy.get_param("~max_radius", 0.4)

        self.ransac_iter = rospy.get_param("~ransac_iter", 100)
        self.ransac_thresh = rospy.get_param("~ransac_thresh", 0.02)

        self.match_dist = rospy.get_param("~match_dist", 0.35)

        self.ema_alpha = rospy.get_param("~ema_alpha", 0.35)

        self.last_centers = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.sub = rospy.Subscriber(self.scan_topic,
                                    LaserScan,
                                    self.scan_callback,
                                    queue_size=1)

        self.pub = rospy.Publisher("/cones", PoseArray, queue_size=10)

        rospy.loginfo("cone detector running")

    def scan_to_xy(self, scan):

        ranges = np.array(scan.ranges)

        angles = np.linspace(scan.angle_min,
                             scan.angle_max,
                             len(ranges))

        valid = (ranges > self.min_range) & \
                (ranges < self.max_range) & \
                np.isfinite(ranges)

        ranges = ranges[valid]
        angles = angles[valid]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        return np.vstack((x, y)).T

    def cluster(self, pts):

        if len(pts) < self.min_points:
            return np.array([])

        model = DBSCAN(eps=self.cluster_eps,
                       min_samples=self.min_points)

        labels = model.fit_predict(pts)

        return labels

    def fit_circle_3pts(self, p1, p2, p3):

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        temp = x2**2 + y2**2

        bc = (x1**2 + y1**2 - temp) / 2
        cd = (temp - x3**2 - y3**2) / 2

        det = (x1-x2)*(y2-y3) - (x2-x3)*(y1-y2)

        if abs(det) < 1e-6:
            return None

        cx = (bc*(y2-y3) - cd*(y1-y2)) / det
        cy = ((x1-x2)*cd - (x2-x3)*bc) / det

        r = math.sqrt((cx-x1)**2 + (cy-y1)**2)

        return np.array([cx, cy]), r

    def ransac_circle(self, pts):

        best_center = None
        best_score = 0
        best_r = None

        if len(pts) < 3:
            return None, None

        for _ in range(self.ransac_iter):

            ids = random.sample(range(len(pts)), 3)

            res = self.fit_circle_3pts(pts[ids[0]],
                                       pts[ids[1]],
                                       pts[ids[2]])

            if res is None:
                continue

            center, r = res

            if not (self.min_radius < r < self.max_radius):
                continue

            d = np.linalg.norm(pts - center, axis=1)

            inliers = np.sum(np.abs(d-r) < self.ransac_thresh)

            if inliers > best_score:
                best_score = inliers
                best_center = center
                best_r = r

        return best_center, best_r

    def transform_point(self, p, from_frame, to_frame, stamp):

        try:

            tf = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                stamp,
                rospy.Duration(0.1)
            )

            pt = tf2_geometry_msgs.PointStamped()

            pt.header.frame_id = from_frame
            pt.header.stamp = stamp

            pt.point.x = p[0]
            pt.point.y = p[1]
            pt.point.z = 0.0

            res = tf2_geometry_msgs.do_transform_point(pt, tf)

            return np.array([res.point.x, res.point.y])

        except:
            return None

    def nearest_previous(self, center):

        best = None
        best_d = 999

        for c in self.last_centers:

            d = np.linalg.norm(center - c)

            if d < best_d:
                best = c
                best_d = d

        if best_d < self.match_dist:
            return best

        return None

    def smooth(self, new, old):

        if old is None:
            return new

        return self.ema_alpha * new + (1-self.ema_alpha) * old

    def scan_callback(self, scan):

        pts = self.scan_to_xy(scan)

        if len(pts) == 0:
            return

        labels = self.cluster(pts)

        if len(labels) == 0:
            return

        centers = []

        for label in set(labels):

            if label == -1:
                continue

            cluster = pts[labels == label]

            if len(cluster) < 4:
                continue

            clusters_to_process = [cluster]

            # ---------- 新增：cluster宽度检测 ----------
            width = np.linalg.norm(cluster.max(axis=0) - cluster.min(axis=0))

            if width > 0.45 and len(cluster) >= 6:

                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=2, n_init=10).fit(cluster)

                clusters_to_process = [
                    cluster[kmeans.labels_ == 0],
                    cluster[kmeans.labels_ == 1]
                ]

            # ------------------------------------------

            for sub_cluster in clusters_to_process:

                if len(sub_cluster) < 4:
                    continue

                center, r = self.ransac_circle(sub_cluster)

                if center is None:
                    continue

            if center is None:
                continue

            if self.output_frame != scan.header.frame_id:

                center = self.transform_point(center,
                                              scan.header.frame_id,
                                              self.output_frame,
                                              scan.header.stamp)

                if center is None:
                    continue

            prev = self.nearest_previous(center)

            center = self.smooth(center, prev)

            centers.append(center)

        self.last_centers = centers

        if len(centers) == 0:
            return

        msg = PoseArray()

        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.output_frame

        for c in centers:

            pose = Pose()

            pose.position.x = c[0]
            pose.position.y = c[1]
            pose.position.z = 0

            pose.orientation.w = 1.0

            msg.poses.append(pose)

        self.pub.publish(msg)


if __name__ == "__main__":

    try:
        ConeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass