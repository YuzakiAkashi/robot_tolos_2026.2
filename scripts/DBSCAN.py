#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import Header
import tf2_ros
import tf2_geometry_msgs
from sklearn.cluster import DBSCAN
import math

class ConeDetector:
    def __init__(self):
        rospy.init_node('cone_detector', anonymous=True)

        # 参数
        self.scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.output_frame = rospy.get_param('~output_frame', 'base_link')  # 输出坐标系的参考系
        self.cluster_epsilon = rospy.get_param('~cluster_epsilon', 0.15)   # DBSCAN邻域半径 (米)
        self.min_points = rospy.get_param('~min_points', 3)                # 最小聚类点数
        self.max_range = rospy.get_param('~max_range', 5.0)                # 激光最大有效距离
        self.min_range = rospy.get_param('~min_range', 0.2)                # 激光最小有效距离

        # TF缓冲和监听器（用于坐标变换）
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 订阅激光数据
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)

        # 发布锥桶位置（PoseArray）
        self.cone_pub = rospy.Publisher('/cones', PoseArray, queue_size=10)

        rospy.loginfo("锥桶检测节点已启动")

    def scan_to_cartesian(self, scan_msg):
        """将激光扫描转换为笛卡尔坐标点集 (numpy数组, Nx2)"""
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
        ranges = np.array(scan_msg.ranges)

        # 过滤无效和超出范围的点
        valid = (ranges > self.min_range) & (ranges < self.max_range) & np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]

        # 极坐标转笛卡尔
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        points = np.vstack((x, y)).T
        return points

    def cluster_points(self, points):
        """使用DBSCAN对点进行聚类，返回每个点的标签"""
        if len(points) < self.min_points:
            return np.array([])
        clustering = DBSCAN(eps=self.cluster_epsilon, min_samples=self.min_points).fit(points)
        return clustering.labels_

    def transform_point_to_frame(self, point, from_frame, to_frame, time):
        """将点从from_frame变换到to_frame"""
        try:
            transform = self.tf_buffer.lookup_transform(to_frame,
                                                         from_frame,
                                                         time,
                                                         rospy.Duration(0.1))
            point_stamped = tf2_geometry_msgs.PointStamped()
            point_stamped.header.frame_id = from_frame
            point_stamped.header.stamp = time
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = 0.0

            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return np.array([transformed.point.x, transformed.point.y])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"坐标变换失败: {e}")
            return None

    def scan_callback(self, scan_msg):
        # 1. 转换为笛卡尔点
        points = self.scan_to_cartesian(scan_msg)
        if len(points) == 0:
            return

        # 2. 密度聚类
        labels = self.cluster_points(points)
        if len(labels) == 0:
            return

        # 3. 提取每个聚类的中心
        unique_labels = set(labels)
        cone_poses = []
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.output_frame

        for label in unique_labels:
            if label == -1:
                continue  # 忽略噪声点
            cluster_points = points[labels == label]

            # 计算质心
            centroid = np.mean(cluster_points, axis=0)

            # 如果需要变换到输出坐标系（假设激光雷达坐标系为scan_msg.header.frame_id）
            if self.output_frame != scan_msg.header.frame_id:
                transformed = self.transform_point_to_frame(centroid,
                                                            scan_msg.header.frame_id,
                                                            self.output_frame,
                                                            scan_msg.header.stamp)
                if transformed is None:
                    continue
                centroid = transformed

            # 创建Pose对象（仅设置位置，方向留空）
            pose = Pose()
            pose.position.x = centroid[0]
            pose.position.y = centroid[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0  # 无旋转
            cone_poses.append(pose)

        # 4. 发布锥桶位置数组
        if cone_poses:
            pose_array = PoseArray()
            pose_array.header = header
            pose_array.poses = cone_poses
            self.cone_pub.publish(pose_array)
            rospy.logdebug(f"检测到 {len(cone_poses)} 个锥桶")

if __name__ == '__main__':
    try:
        ConeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass