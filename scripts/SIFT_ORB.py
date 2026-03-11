#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os, sys, threading, time
import concurrent.futures
from collections import defaultdict, deque

class FeatureMatchNode:
    def __init__(self):
        rospy.init_node('sift_orb_camera_detect', anonymous=True)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True

        # 参数
        self.template_dir = rospy.get_param('~template_dir', '/home/yuzaki/helloworld/src/my_robot_description/template')
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
        self.min_match_ratio = rospy.get_param('~min_match_ratio', 0.03)
        self.min_color_sim = rospy.get_param('~min_color_sim', 0.25)
        self.min_matches_for_homography = rospy.get_param('~min_matches_for_homography', 8)
        self.debug_window = rospy.get_param('~debug_window', True)
        
        # 多线程参数
        self.num_workers = rospy.get_param('~num_workers', 9)  # 工作线程数，通常设置为CPU核心数
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers)
        
        # 添加误识别过滤参数
        self.min_detection_area_ratio = rospy.get_param('~min_detection_area_ratio', 0.005)
        self.max_detection_area_ratio = rospy.get_param('~max_detection_area_ratio', 0.6)
        self.min_inlier_ratio = rospy.get_param('~min_inlier_ratio', 0.15)
        self.min_combined_score = rospy.get_param('~min_combined_score', 0.0001)
        
        # 增强的几何验证参数
        self.min_aspect_ratio = rospy.get_param('~min_aspect_ratio', 0.3)
        self.max_aspect_ratio = rospy.get_param('~max_aspect_ratio', 3.0)
        self.min_compactness = rospy.get_param('~min_compactness', 0.2)
        self.max_skew_angle = rospy.get_param('~max_skew_angle', 80)
        self.min_polygon_area_ratio = rospy.get_param('~min_polygon_area_ratio', 0.1)

        # 初始化特征提取器
        self.detector = cv2.SIFT_create(
            nfeatures=4000,
            nOctaveLayers=5,
            contrastThreshold=0.03,
            edgeThreshold=8,
            sigma=0.5
        )
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        rospy.loginfo(f"使用 {self.num_workers} 个工作线程进行并行匹配")
        rospy.loginfo("初始化成功...正在加载模板")

        # 加载模板
        self.templates = self.load_templates(self.template_dir)
        if not self.templates:
            rospy.logerr("未加载到任何模板，请检查路径")
            sys.exit(1)

        # 检测结果过滤
        self.last_detection_time = 0
        self.detection_cooldown = 2.0
        self.last_detected_object = None

        # 订阅图像
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo(f"🔹 订阅图像: {self.image_topic}")

        # 启动显示线程
        if self.debug_window:
            threading.Thread(target=self.display_loop, daemon=True).start()

        rospy.spin()

    def load_templates(self, template_dir):
        templates = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for f in sorted(os.listdir(template_dir)):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            path = os.path.join(template_dir, f)
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            kp, des = self.detector.detectAndCompute(gray, None)
            if des is None or len(kp) == 0:
                rospy.logwarn(f"[WARN] 模板 {f} 无特征点")
                continue
            templates.append({
                'name': f, 
                'image': img, 
                'kp': kp, 
                'des': des,
                'matcher': cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            })
            rospy.loginfo(f"加载模板: {f} ({len(kp)} 特征点)")
        rospy.loginfo(f"共加载 {len(templates)} 张模板")
        return templates

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed_gray = clahe.apply(gray)
            
            with self.lock:
                self.latest_frame = frame
                self.processed_frame = processed_gray
                
        except Exception as e:
            rospy.logwarn_throttle(5, f"图像转换失败: {e}")

    def match_single_template(self, template, kp_frame, des_frame, frame):
        """单个模板的匹配函数，用于并行处理"""
        try:
            matches = template['matcher'].knnMatch(template['des'], des_frame, k=2)
        except:
            return None
            
        good = [m for m, n in matches if len(matches[0]) >= 2 and m.distance < 0.75 * n.distance]
        match_ratio = len(good) / float(len(template['des']))
        
        if match_ratio > 0.01 and len(good) >= 4:  # 降低阈值，让更多候选进入后续处理
            src = np.float32([template['kp'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            
            if M is not None:
                inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
                
                # 几何验证
                tpl_h, tpl_w = template['image'].shape[:2]
                is_valid, bbox, compactness, skew_angle, aspect_ratio = self.geometric_validation(
                    M, (tpl_h, tpl_w), frame.shape[:2]
                )
                
                if not is_valid:
                    return None
                
                # 计算检测区域面积
                x, y, w, h = bbox
                detection_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                area_ratio = detection_area / frame_area
                
                # 面积过滤
                if area_ratio < self.min_detection_area_ratio or area_ratio > self.max_detection_area_ratio:
                    return None
                
                roi = self.safe_crop(frame, (x, y, w, h))
                color_sim = self.color_similarity_hsv(roi, template['image'])
                
                # 综合评分
                geometric_quality = compactness * (1 - skew_angle / 90)
                combined_score = match_ratio * inlier_ratio * geometric_quality * (1 + min(area_ratio, 0.1))
                
                return {
                    'name': template['name'][:-4], 
                    'ratio': combined_score,
                    'homography': M, 
                    'template': template, 
                    'color_sim': color_sim,
                    'area': detection_area,
                    'area_ratio': area_ratio,
                    'inlier_ratio': inlier_ratio,
                    'raw_match_ratio': match_ratio,
                    'bbox': bbox,
                    'compactness': compactness,
                    'skew_angle': skew_angle
                }
        return None

    def process_frame_parallel(self, frame):
        """并行处理帧，使用多线程同时匹配所有模板"""
        with self.lock:
            if hasattr(self, 'processed_frame'):
                gray_frame = self.processed_frame.copy()
            else:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 提取当前帧的特征点
        kp_frame, des_frame = self.detector.detectAndCompute(gray_frame, None)
        if des_frame is None:
            return frame, None
        
        # 并行匹配所有模板
        futures = []
        for template in self.templates:
            future = self.thread_pool.submit(
                self.match_single_template, 
                template, kp_frame, des_frame, frame
            )
            futures.append(future)
        
        # 收集结果
        best = None
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
                if best is None or result['ratio'] > best['ratio']:
                    best = result
        
        # 如果没有找到匹配，返回默认值
        if best is None:
            best = {'name': None, 'ratio': 0.0}
            
        return frame, best

    def output_detection_to_console(self, detection_result):
        """直接在控制台输出检测到的物体信息"""
        current_time = time.time()
        
        # 检查冷却时间（避免重复输出相似的检测）
        # if (current_time - self.last_detection_time < self.detection_cooldown and 
        #     self.last_detected_object == detection_result['name']):
        #     return False
            
        # 输出检测结果到控制台
        rospy.loginfo(f"检测到物体: {detection_result['name']} | "
                     f"{detection_result['ratio']:.3f} | "
                     f"{detection_result['area_ratio']:.3f}")
        
        self.last_detection_time = current_time
        self.last_detected_object = detection_result['name']
        return True

    def color_similarity_hsv(self, img_roi, template_img):
        if img_roi is None or template_img is None:
            return 0.0
        try:
            roi = cv2.resize(img_roi, (128,128))
            tpl = cv2.resize(template_img, (128,128))
            hsv1 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(tpl, cv2.COLOR_BGR2HSV)
            hist1 = cv2.calcHist([hsv1],[0,1],None,[50,60],[0,180,0,256])
            hist2 = cv2.calcHist([hsv2],[0,1],None,[50,60],[0,180,0,256])
            cv2.normalize(hist1,hist1,0,1,cv2.NORM_MINMAX)
            cv2.normalize(hist2,hist2,0,1,cv2.NORM_MINMAX)
            return float(cv2.compareHist(hist1,hist2,cv2.HISTCMP_CORREL))
        except:
            return 0.0

    def safe_crop(self, img, rect):
        h,w = img.shape[:2]
        x,y,rw,rh = rect
        x0=max(0,x); y0=max(0,y)
        x1=min(w,x+rw); y1=min(h,y+rh)
        if x1<=x0 or y1<=y0: return None
        return img[y0:y1,x0:x1]
    
    def geometric_validation(self, homography, template_size, frame_size):
        """几何验证函数（与之前相同）"""
        h, w = template_size
        frame_h, frame_w = frame_size
        
        pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1, 1, 2)
        
        try:
            dst_pts = cv2.perspectiveTransform(pts, homography)
            hull = cv2.convexHull(dst_pts)
            if not cv2.isContourConvex(hull):
                return False, None, 0, 0, 0
            
            x, y, bbox_w, bbox_h = cv2.boundingRect(dst_pts)
            bbox = (x, y, bbox_w, bbox_h)
            
            aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                return False, bbox, 0, 0, aspect_ratio
            
            poly_area = cv2.contourArea(dst_pts)
            bbox_area = bbox_w * bbox_h
            compactness = poly_area / bbox_area if bbox_area > 0 else 0
            
            if compactness < self.min_compactness:
                return False, bbox, compactness, 0, aspect_ratio
            
            if poly_area / bbox_area < self.min_polygon_area_ratio:
                return False, bbox, compactness, 0, aspect_ratio
            
            vectors = []
            for i in range(4):
                vec = dst_pts[(i+1)%4][0] - dst_pts[i][0]
                vectors.append(vec)
            
            max_skew = 0
            for i in range(4):
                v1 = vectors[i]
                v2 = vectors[(i+1)%4]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                skew = abs(angle - 90)
                if skew > max_skew:
                    max_skew = skew
            
            if max_skew > self.max_skew_angle:
                return False, bbox, compactness, max_skew, aspect_ratio
            
            edge_sum = 0
            for i in range(4):
                x1, y1 = dst_pts[i][0]
                x2, y2 = dst_pts[(i+1)%4][0]
                edge_sum += (x2 - x1) * (y2 + y1)
            
            if abs(edge_sum) < 1e-6:
                return False, bbox, compactness, max_skew, aspect_ratio
            
            return True, bbox, compactness, max_skew, aspect_ratio
            
        except Exception as e:
            return False, None, 0, 0, 0

    def process_frame(self, frame):
        """主处理函数，现在使用并行版本"""
        return self.process_frame_parallel(frame)

    def display_loop(self):
        cv2.namedWindow("SIFT/ORB Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SIFT/ORB Detection", 1920, 1080)
        
        last_warn = time.time()
        last_process_time = 0
        process_interval = 1.0 / 15  # 提高到15 FPS，因为并行处理更快
        
        while self.running and not rospy.is_shutdown():
            current_time = time.time()
            
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                    
            if frame is None:
                if time.time() - last_warn > 5:
                    rospy.logwarn("未接收到图像帧 (/cam)")
                    last_warn = time.time()
                time.sleep(0.1)
                continue

            if current_time - last_process_time >= process_interval:
                result, best = self.process_frame(frame)
                print(best)
                # 绘制检测结果
                if best and best['name'] and best['ratio'] > self.min_combined_score:
                    tpl_h, tpl_w = best['template']['image'].shape[:2]
                    pts = np.float32([[0,0], [0,tpl_h-1], [tpl_w-1,tpl_h-1], [tpl_w-1,0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, best['homography'])
                    
                    cv2.polylines(result, [np.int32(dst)], True, (0, 255, 0), 3)
                    x, y, w, h = best['bbox']
                    
                    info_text = f"{best['name']} (Score:{best['ratio']:.2f})"
                    cv2.putText(result, info_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    geom_text = f"Compact:{best['compactness']:.2f} Skew:{best['skew_angle']:.1f}°"
                    cv2.putText(result, geom_text, (x, y+h+25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    stats_text = f"Matches: {best['raw_match_ratio']:.2f} Inliers:{best['inlier_ratio']:.2f}"
                    cv2.putText(result, stats_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # 输出到控制台
                    self.output_detection_to_console(best)
                
                cv2.imshow("SIFT/ORB Detection", result)
                last_process_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.signal_shutdown("用户退出")
                break

    def on_shutdown(self):
        self.running = False
        self.thread_pool.shutdown(wait=False)
        try: 
            cv2.destroyAllWindows()
        except: 
            pass

if __name__ == "__main__":
    try:
        FeatureMatchNode()
    except rospy.ROSInterruptException:
        pass