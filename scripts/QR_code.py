#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pyzbar.pyzbar import decode

class QRCodeReaderNode:
    def __init__(self):
        rospy.init_node('qr_code_reader', anonymous=True)

        # 参数
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
        self.publish_output = rospy.get_param('~publish_output', True)  # 是否发布带标注的图像
        self.output_topic = rospy.get_param('~output_topic', '/qr_image')
        self.show_image = rospy.get_param('~show_image', True)  # 是否显示OpenCV窗口
        self.skip_frames = rospy.get_param('~skip_frames', 0)   # 每处理一帧跳过的帧数（0表示每帧都处理）
        self.frame_count = 0

        # 初始化CvBridge
        self.bridge = CvBridge()

        # 订阅图像话题
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

        # 发布带标注的图像
        if self.publish_output:
            self.image_pub = rospy.Publisher(self.output_topic, Image, queue_size=10)

        rospy.loginfo("二维码读取节点已启动，订阅话题: %s", self.image_topic)

    def image_callback(self, msg):
        # 简单帧率控制：跳过部分帧
        if self.skip_frames > 0:
            self.frame_count += 1
            if self.frame_count % (self.skip_frames + 1) != 0:
                return

        try:
            # 将ROS图像转换为OpenCV图像（BGR格式）
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr("图像转换失败: %s", e)
            return

        # 转换为灰度图（pyzbar可以直接处理BGR或灰度）
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (51,51), 0)
        gray = cv2.divide(gray, blur, scale=255)

        clahe = cv2.createCLAHE(2.0, (8,8))
        gray = clahe.apply(gray)

        gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

        barcodes = decode(gray)

        # 解码二维码
        barcodes = decode(gray)

        # 处理检测到的每个二维码
        for barcode in barcodes:
            # 提取二维码数据
            data = barcode.data.decode("utf-8")
            barcode_type = barcode.type

            # 输出到控制台
            rospy.loginfo("检测到二维码: [%s] 内容: %s", barcode_type, data)

            # 在图像上绘制边界框和内容（如果准备发布或显示）
            if self.publish_output or self.show_image:
                points = barcode.polygon
                if len(points) == 4:
                    # 转换为整数坐标并绘制多边形
                    pts = np.array([(point.x, point.y) for point in points], dtype=np.int32)
                    cv2.polylines(cv_image, [pts], True, (0, 255, 0), 2)
                    # 在左上角显示文本
                    text = f"{data}"
                    x = min(pts[:, 0])
                    y = min(pts[:, 1]) - 10
                    cv2.putText(cv_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

        # 如果需要，显示图像
        if self.show_image:
            cv2.imshow("QR Code Reader", cv_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.signal_shutdown("用户退出")

        # 如果需要，发布带标注的图像
        if self.publish_output:
            try:
                out_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                out_msg.header = msg.header
                self.image_pub.publish(out_msg)
            except Exception as e:
                rospy.logerr("图像发布失败: %s", e)

    def on_shutdown(self):
        if self.show_image:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    node = QRCodeReaderNode()
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()