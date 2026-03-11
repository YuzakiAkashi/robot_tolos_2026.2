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

        lower_red1 = np.array([0,80,80])
        upper_red1 = np.array([10,255,255])

        lower_red2 = np.array([170,80,80])
        upper_red2 = np.array([180,255,255])

        mask1 = cv2.inRange(hsv,lower_red1,upper_red1)
        mask2 = cv2.inRange(hsv,lower_red2,upper_red2)

        mask = mask1 + mask2

        mask = cv2.medianBlur(mask,5)

        contours,_ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        hexagons = []

        for c in contours:

            area = cv2.contourArea(c)

            if area < 2000:
                continue

            peri = cv2.arcLength(c,True)

            approx = cv2.approxPolyDP(c,0.02*peri,True)

            if len(approx)==6:

                M = cv2.moments(c)

                if M["m00"]==0:
                    continue

                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])

                x,y,w,h = cv2.boundingRect(c)

                hexagons.append((c,cx,cy,x,y,w,h))

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
        #2print("pattern:", pattern)
        key = tuple(pattern)

        if key in patterns:
            return patterns[key]

        return None


    def process(self,frame):

        draw = frame.copy()

        hexagons = self.detect_hexagons(frame)

        if len(hexagons) < 3:
            return draw

        hexagons = sorted(hexagons,key=lambda x:(x[2],x[1]))

        pattern = []

        for c,cx,cy,x,y,w,h in hexagons[:3]:

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