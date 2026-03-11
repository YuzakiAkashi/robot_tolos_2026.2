#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class HexBoardDetector:

    def __init__(self):

        rospy.init_node("hex_board_detector")

        self.bridge = CvBridge()
        self.frame = None

        rospy.Subscriber(
            "/camera/rgb/image_raw",
            Image,
            self.callback,
            queue_size=1
        )

        cv2.namedWindow("detector", cv2.WINDOW_NORMAL)

    def callback(self,msg):

        self.frame = self.bridge.imgmsg_to_cv2(msg,"bgr8")

    
    def detect_hexagons(self,frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0,150,150])
        upper_red1 = np.array([5,255,255])

        lower_red2 = np.array([175,150,150])
        upper_red2 = np.array([180,255,255])

        mask1 = cv2.inRange(hsv,lower_red1,upper_red1)
        mask2 = cv2.inRange(hsv,lower_red2,upper_red2)

        mask = mask1 + mask2
        mask = cv2.medianBlur(mask,5)

        # debug：查看红色分割效果
        #cv2.imshow("debug_red_mask",mask)

        contours,_ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        print("-------- new frame --------")
        print("contours:", len(contours))

        hexagons = []

        for i,c in enumerate(contours):

            area = cv2.contourArea(c)

            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,0.03*peri,True)

            vertices = len(approx)

            x,y,w,h = cv2.boundingRect(c)
            ratio = w/float(h)

            print("id:",i,
                "area:",area,
                "vertices:",vertices,
                "ratio:",ratio)

            # 面积过滤
            if area < 2000:
                print(" -> reject: area too small")
                continue

            # 六边形过滤
            if vertices < 5 or vertices > 7:
                print(" -> reject: not hexagon")
                continue

            # 柱子过滤
            if ratio < 0.3 or ratio > 3.0:
                print(" -> reject: bad ratio")
                continue

            M = cv2.moments(c)

            if M["m00"] == 0:
                print(" -> reject: zero moment")
                continue

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            print(" -> accepted hexagon")

            hexagons.append((approx,cx,cy,x,y,w,h))

        print("hexagons detected:",len(hexagons))

        return hexagons

    def classify_shape(self, roi):

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20,80,80])
        upper_yellow = np.array([40,255,255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask = cv2.medianBlur(mask,5)

        contours,_ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        count = 0

        for c in contours:

            area = cv2.contourArea(c)

            if area > 50:
                count += 1

        if count > 3:
            return "Female"
        else:
            return "Male"


    def decode_pattern(self,pattern):

        patterns = {

            ("Male","Male","Male"):"MMM",
            ("Male","Male","Female"):"MMF",
            ("Male","Female","Male"):"MFM",
            ("Female","Male","Male"):"FMM",
            ("Male","Female","Female"):"MFF",
            ("Female","Female","Male"):"FFM"
        }
        print("pattern:", pattern)
        key = tuple(pattern)

        if key in patterns:
            return patterns[key]

        return None


    def process(self,frame):

        draw = frame.copy()

        hexagons = self.detect_hexagons(frame)

        if len(hexagons) < 3:
            return draw

        # 按面积排序
        hexagons = sorted(
            hexagons,
            key=lambda x: cv2.contourArea(x[0]),
            reverse=True
        )

        # 取最大的三个
        hexagons = hexagons[:3]

        # 再按位置排序
        hexagons = sorted(hexagons,key=lambda x:(x[2],x[1]))      

        pattern = []

        for c,cx,cy,x,y,w,h in hexagons:  

            roi = frame[y:y+h, x:x+w]

            # 创建mask
            mask = np.zeros((h, w), dtype=np.uint8)

            # 将轮廓坐标转换到ROI坐标系
            contour_roi = c - [x, y]

            # 画六边形mask
            cv2.drawContours(mask, [contour_roi], -1, 255, -1)

            # 只保留六边形内部
            roi = cv2.bitwise_and(roi, roi, mask=mask)

            label = self.classify_shape(roi)

            pattern.append(label)

            color = (0,255,0) if label=="Male" else (255,0,0)

            cv2.drawContours(draw,[c],-1,color,3)

            cv2.putText(
                draw,
                label,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        board_type = self.decode_pattern(pattern)

        if board_type is not None:

            cv2.putText(
                draw,
                "Board {}".format(board_type),
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                3
            )

        return draw


    def run(self):

        rate = rospy.Rate(30)

        while not rospy.is_shutdown():

            if self.frame is not None:

                result = self.process(self.frame.copy())

                cv2.imshow("detector",result)

                cv2.waitKey(1)

            rate.sleep()


if __name__=="__main__":

    node = HexBoardDetector()

    node.run()