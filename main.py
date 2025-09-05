#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车道线检测和车辆碰撞预警系统 - 主程序

功能描述:
- 实时处理行车记录仪视频
- 检测车道线并获取坐标
- 使用YOLO检测车辆并计算距离
- 提供碰撞预警功能
- PyQt界面显示实时视频和预警信息

作者: HXH Assistant
创建时间: 2025
"""
import math
import sys
import os
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QFileDialog, QSlider,
                             QGroupBox, QGridLayout, QFrame, QProgressBar, QTextEdit,
                             QSplitter, QScrollArea, QSpacerItem, QSizePolicy, QCheckBox, QColorDialog)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QLinearGradient, QBrush

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from core.distance_measure.distance_calculator import DistanceCalculator # 标定算法 + 单目距离测量
from core.lane_detection.lane_detector import LaneDetector # 车道线检测
from core.lane_detection.lane_region_draw import postprocess_coords_with_draw # 车道区域特效绘制
from core.detection.vehicle_detection import VehicleDetector # 车辆检测
from config.config_optimize import Config # QT配置文件
from core.vision.tech_visualizer import TechHUDVisualizer # 特效视觉

from collections import deque
from shapely.geometry import Polygon, Point
from PyQt5.QtCore import QMutex, QWaitCondition
# 代码文件开头添加如下代码来忽略 FutureWarning：`torch.cuda.amp.autocast(args...)` is deprecated.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class VideoProcessor(QThread):
    """
    视频处理线程类
    优化的视频处理线程类
    负责处理视频帧，进行车道线检测、车辆检测和距离计算
    采用生产者-消费者模式和帧缓冲优化性能
    """
    # 定义信号，用于向主线程传递处理结果
    # frame_processed = pyqtSignal(np.ndarray)  # 处理后的帧
    frame_ready = pyqtSignal()  # 帧准备好信号（不传递数据）
    warning_signal = pyqtSignal(str, str)  # 预警信号 (warning_level, message)
    stats_updated = pyqtSignal(dict)  # 统计信息更新信号
    first_frame_ready = pyqtSignal(np.ndarray)  # 第一帧准备好信号

    def __init__(self):
        super().__init__()
        self.cap = None
        self.is_running = False
        self.video_path = None

        # 车道线检测开关状态
        self.lane_detection_enabled = True

        # 初始化检测器
        # 1.车道线检测
        self.lane_detector = None  # 延迟初始化
        self._init_lane_detector()

        # 2.车辆检测
        # model_path = "models/engine/car_detector.engine" # TensorRT推理
        model_path = "models/yolov5/car_detector.pt" # pytorch推理（没有TensorRT框架的时候，默认推理模式）
        # model_path = "models/yolo11-seg/yolo11s-seg.pt" # 分割模型推理
        self.vehicle_detector = VehicleDetector(model_path)

        # 3.距离检测
        self.distance_calculator = DistanceCalculator()

        # 4.配置参数
        self.config = Config()

        # 5.统计信息
        self.frame_count = 0
        self.start_time = None
        self.vehicle_count = 0
        self.fps = 0
        self.run_time = 0

        # 6.性能优化参数
        self.detection_interval = 1  # 每1帧进行一次检测
        self.last_vehicles = []  # 缓存上次检测结果
        self.last_contours = []
        self.last_detect_lane_frame = np.array([]) # 缓存上次车道线检测后绘制的帧
        self.last_lane_polygon = None
        self.last_visual_datas = [None, None, None, None] # 缓存上次可视化数据结果
        self.target_fps = 30  # 目标帧率
        self.frame_time = 1.0 / self.target_fps  # 每帧时间间隔


        # 7.线程优化和帧缓冲
        self.setPriority(QThread.HighPriority)  # 设置高优先级
        self.frame_buffer = deque(maxlen=100)      # 帧缓冲队列，最多保存100帧
        self.buffer_mutex = QMutex()             # 缓冲区互斥锁
        self.current_frame = None                # 当前显示帧
        
        # 8.科技感可视化
        self.tech_visual_enabled = False         # 科技感可视化开关
        self.tech_visualizer = TechHUDVisualizer()  # 科技感可视化器

    def _init_lane_detector(self):
        """
        初始化车道线检测器
        """
        if self.lane_detection_enabled and self.lane_detector is None:
            try:
                engine_path = "core/lane_detection/weights/culane_res34.engine"
                config_path = "core/lane_detection/configs/culane_res34.py"
                ori_size = (1600, 320)  # 固定好size
                self.lane_detector = LaneDetector(engine_path, config_path, ori_size)
            except Exception as e:
                print(f"车道线检测器初始化失败: {e}")
                print(f"需要进行车道线检测器的环境配置和模型下载，可以选择torch推理，当运行的时候关闭车道线检测，即可正常运行！")
                self.lane_detector = None

    def set_lane_detection_enabled(self, enabled):
        """
        设置车道线检测开关状态
        """
        self.lane_detection_enabled = enabled
        if enabled:
            self._init_lane_detector()
        else:
            self.lane_detector = None

    def set_video_source(self, video_path):
        """
        设置视频源

        Args:
            video_path (str): 视频文件路径或摄像头索引
        """
        # 如果是摄像头，直接写入self
        if isinstance(video_path, int):
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.video_path = video_path
                        print(f"成功打开摄像头 {video_path}，尺寸: {frame.shape}")
                        self.first_frame_ready.emit(frame)
                    else:
                        print(f"摄像头 {video_path} 无法读取画面")
                else:
                    print(f"摄像头 {video_path} 不存在或无法打开")
                cap.release()
                return 0
            except Exception as e:
                print(f"设置视频源时出错: {e}")

        # 如果是视频文件，获取第一帧用于预览
        if isinstance(video_path, str) and os.path.exists(video_path):
            try:
                self.video_path = video_path
                temp_cap = cv2.VideoCapture(video_path)
                if temp_cap.isOpened():
                    ret, first_frame = temp_cap.read()
                    if ret and first_frame is not None:
                        print(f"成功读取第一帧，尺寸: {first_frame.shape}")
                        self.first_frame_ready.emit(first_frame)
                    else:
                        print("读取第一帧失败")
                else:
                    print(f"无法打开视频文件: {video_path}")
                temp_cap.release()
            except Exception as e:
                print(f"设置视频源时出错: {e}")
        else:
            print(f"视频路径无效或不存在: {video_path}")

    def start_processing(self):
        """
        开始视频处理
        """
        if self.video_path is None:
            return False

        # 打开视频源
        if isinstance(self.video_path, int):
            # 摄像头
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            # 视频文件
            self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            return False

        # 初始化统计信息
        self.frame_count = 0 # 处理帧数计数
        self.start_time = time.time()
        self.vehicle_count = 0 # 车辆数目
        self.fps = 0 # 帧率
        self.run_time = 0  # 检测运行时间

        self.is_running = True
        self.start()
        return True

    def get_latest_frame(self):
        """
        获取最新的处理帧（线程安全）

        Returns:
            np.ndarray: 最新的处理帧，如果没有则返回None
        """
        self.buffer_mutex.lock()
        try:
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()  # 返回最新帧的副本
            return self.current_frame.copy() if self.current_frame is not None else None
        finally:
            self.buffer_mutex.unlock()

    def stop_processing(self):
        """
        停止视频处理
        """
        self.is_running = False
        if self.cap:
            self.cap.release()

        # 清空缓冲区
        self.buffer_mutex.lock()
        try:
            self.frame_buffer.clear()
            self.current_frame = None
        finally:
            self.buffer_mutex.unlock()

    def run(self):
        """
        优化的线程主循环，采用帧缓冲
        """
        frame_start_time = time.time()

        # 主循环
        while self.is_running and self.cap and self.cap.isOpened():
            loop_start = time.time()

            # 读取帧
            ret, frame = self.cap.read()
            if not ret:
                break

            # 处理帧
            processed_frame = self.process_frame(frame)

            # 将处理后的帧放入缓冲区（线程安全）
            self.buffer_mutex.lock()
            try:
                self.frame_buffer.append(processed_frame.copy())
                self.current_frame = processed_frame
            finally:
                self.buffer_mutex.unlock()

            # 更新统计信息
            self.frame_count += 1 # 帧数加一

            current_time = time.time()
            if self.start_time and current_time - self.start_time > 0:
                # 比如说处理了30张图像，耗时2s，就是一秒钟处理了15张，也就是15fps
                self.fps = self.frame_count / (current_time - self.start_time)

            # 视频的运行时间
            run_time = (current_time - self.start_time) # 运行时间
            self.run_time = math.floor(run_time) # 向下取整
            # run_time = math.ceil(run_time) # 向上取整

            # 发送统计信息（可选降低频率）
            frequency_status = 1
            if self.frame_count % frequency_status == 0:
                # 发送统计信息
                stats = {
                    'fps': self.fps,
                    'vehicle_count': self.vehicle_count,
                    'frame_count': self.frame_count,
                    'run_time': self.run_time,

                }
                self.stats_updated.emit(stats)
            # 发送帧准备好信号（不传递帧数据）
            self.frame_ready.emit()

            # 帧率控制
            processing_time = time.time() - loop_start
            sleep_time = max(0.0, self.frame_time - processing_time)
            if sleep_time > 0.0:
                print("睡觉了啊！！！\n")
                self.msleep(int(sleep_time * 1000))



    def process_frame(self, frame):
        """
        a.处理单帧图像
        b.集成科技感可视化
        c.线程主要耗时部分

        Args:
            frame (np.ndarray): 输入帧

        Returns:
            np.ndarray: 处理后的帧
        """
        # 复制原始帧
        result_frame = frame.copy()
        # 性能优化：
        # 隔帧检测车辆、检测车道线，减少计算负担
        gap_detect = (self.frame_count % self.detection_interval == 0)

        # ✅1.车道线检测
        a = time.time()
        if gap_detect and self.lane_detection_enabled and self.lane_detector is not None:
            coords = self.lane_detector.get_lane_coordinates(result_frame)
            # 绘制车道线
            result_frame, left_lane, right_lane = postprocess_coords_with_draw(result_frame, coords)
            self.last_detect_lane_frame = result_frame

            # 构造封闭区域多边形（左车道 + 右车道反转）
            if left_lane and right_lane:
                polygon_points = left_lane + right_lane[::-1]
                # lane_polygon = Polygon(polygon_points)
                # 如果首尾不同，补上闭合点
                if polygon_points[0] != polygon_points[-1]:
                    polygon_points.append(polygon_points[0])

                # 只有数量足够才创建 Polygon
                if len(polygon_points) >= 4:
                    lane_polygon = Polygon(polygon_points)
                else:
                    lane_polygon = None  # 或者跳过

                self.last_lane_polygon = lane_polygon
            else:
                lane_polygon = None
        else:
            # 使用缓存的检测结果或原始帧（当车道线检测关闭时）
            if self.lane_detection_enabled and hasattr(self, 'last_detect_lane_frame') and self.last_detect_lane_frame is not None:
                result_frame = self.last_detect_lane_frame
                lane_polygon = self.last_lane_polygon
            else:
                result_frame = frame.copy()
                lane_polygon = None

        b = time.time()
        # print(f'检测车道耗时：{(b - a):.2f}s')

        # ✅2.目标检测
        c = time.time()
        if gap_detect:
            # 进行车辆检测
            vehicles, contours = self.vehicle_detector.detect_vehicles(frame)
            # 缓存检测结果
            self.last_vehicles = vehicles
            # 缓存掩码结果
            self.last_contours = contours
        else:
            # 使用缓存的检测结果
            vehicles = self.last_vehicles
            contours = self.last_contours
        # 更新车辆数量
        self.vehicle_count = len(vehicles)

        d = time.time()
        # print(f'检测车辆耗时：{(d - c):.2f}s')

        # ✅3.准备科技感可视化数据
        e = time.time()
        detections = []  # 科技感需要的检测数据列表
        warning_level = "safe"  # 预警等级：safe, warning, danger
        warning_message = "" # 提示信息

        all_distances = []  # 所有车辆距离
        danger_distances = []  # 危险距离列表
        warning_distances = []  # 预警距离列表
        distances_is_included_lane = {} # 车辆距离对应是否位于检测车道线上

        # ✅3.自动标定摄像头参数--暂未开放
        # self.distance_calculator.detect_lane_line(frame)
        # ✅3. 准备需要的数据参数
        # 判断检测结果是否为缓存的检测结果
        # if vehicles == self.last_vehicles:
        #     pass
        if self.frame_count % self.detection_interval == 0:
            for vehicle in vehicles:
                x1, y1, x2, y2, confidence, class_id = vehicle
                # 计算检测目标底部中心点
                x_center, y_bottom = self.vehicle_detector.get_vehicle_bottom_center(vehicle)
                point_center = (x_center, y_bottom)

                # a.先计算两点之间的距离
                _, y_distance_ab = self.distance_calculator.cal_distance_a2b((x1, y2), (x2, y2))

                # 一般来说，车宽大约在 1.6 米到 1.85 米 之间
                # 1.直接通过y_distance_ab，来排除误差较大的数据
                # if not 1.2 - 0.2 < y_distance_ab < 2 + 0.2:
                #     continue

                # 2.通过矫正标定参数，精准测量距离
                car_width = self.distance_calculator.calib_correction(y_distance_ab, (x1, y2), (x2, y2))
                if 1.6 <= car_width <= 1.85:
                    # print(f"矫正成功，车宽为{car_width:.2f}米")
                    pass
                else:
                    # print(f"矫正失败，车宽为{car_width:.2f}米")
                    pass
                # print(f"车宽为:{round(car_width, 2)}米") # 车宽为 car_width

                # b.再计算车辆到相机的距离
                x_distance, _ = self.distance_calculator.cal_distance_a2cam(point_center)
                # print(f"车辆到相机点的距离为：{round(x_distance, 2)}米") # 距离为x_distance

                # c.记录所有距离
                all_distances.append(x_distance)

                # d.获取车辆类别名称
                vehicle_class_name = self.vehicle_detector.vehicle_classes.get(class_id, "vehicle")

                # e.判断预警等级
                # 增加判定条件，只针对于在当前车道线的车辆才开启碰撞预警
                # 也就是判定(x_center, y_center)是否在 left_lane 和 right_lane 相交的区域里面

                if x_distance < self.config.SAFE_DISTANCE:
                    danger_distances.append(x_distance)
                    if lane_polygon: # 存在车道区域
                        # 判断是否在车道区域内
                        distances_is_included_lane[x_distance] = lane_polygon.contains(Point(point_center))

                elif self.config.SAFE_DISTANCE < x_distance < self.config.SAFE_DISTANCE * 1.5:
                    warning_distances.append(x_distance)
                    if lane_polygon: # 存在车道区域
                        # 判断是否在车道区域内
                        distances_is_included_lane[x_distance] = lane_polygon.contains(Point(point_center))

                # 构建检测数据用于科技感可视化
                detection_data = {
                    "box": (x1, y1, x2, y2),
                    "label": vehicle_class_name,
                    "score": confidence,
                    "distance": x_distance,
                    "class_id": class_id
                }
                detections.append(detection_data)

            # 确定整体预警等级和消息
            if danger_distances and distances_is_included_lane.get(min(danger_distances), True):
                warning_level = "danger"
                warning_message = f"危险！前方车辆距离：{min(danger_distances):.1f}米"

            elif warning_distances and distances_is_included_lane.get(min(warning_distances), True):
                warning_level = "warning"
                warning_message = f"注意！前方车辆距离：{min(warning_distances):.1f}米"

            else:
                warning_level = "safe"
                warning_message = "系统正常运行"

            self.last_visual_datas = [detections, warning_level, warning_message, distances_is_included_lane] # 缓存可视化数据结果

        else:

            detections, warning_level, warning_message, distances_is_included_lane = self.last_visual_datas

        f = time.time()
        # print(f'数据准备耗时：{(f - e):.2f}s')


        # 准备好的数据，以下内容需要：
        # 1.detections
        # 2.warning_level
        # 3.warning_message

        # ✅4. 根据开关选择渲染方式
        g = time.time()
        if not self.tech_visual_enabled:
            result_frame = self.opencv_visual(img=result_frame,
                                              detections=detections,
                                              distances_in_included_lane=distances_is_included_lane,
                                              contours=contours
            )
            h = time.time()
            # print(f"传统opencv绘制耗时：{(h - g):.2f}s")

        else:
            # 使用科技感可视化器渲染结果
            i = time.time()
            result_frame = self.tech_visualizer.visualize_detections(
                img=result_frame,
                detections=detections,
                frame_id=self.frame_count,
                safe_distance=self.config.SAFE_DISTANCE
            )
            j = time.time()
            print(f"使用科技感可视化渲染耗时：{(j - i):.2f}s")

        # 发送预警信号（可选：降低频率）
        frequency_warning = 1
        if self.frame_count % frequency_warning == 0:  # 每2帧发送一次预警信号
            self.warning_signal.emit(warning_level, warning_message)

        return result_frame



    def opencv_visual(self, img, detections, distances_in_included_lane, contours):

        # 创建原图掩码
        b_mask = np.zeros(img.shape[:2], np.uint8)

        # 使用传统OpenCV绘制
        result_img = img.copy()
        frame_height, frame_width = result_img.shape[:2]
        bottom_center = (frame_width // 2, frame_height)

        # 预设字体参数（避免重复计算）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        big_font_scale = 1.2

        # 计算所有车辆的距离并排序，只显示最近的前N个虚线
        vehicle_distances = []
        for detection in detections:
            x1, y1, x2, y2 = detection.get("box")
            confidence = detection.get("score")
            class_id = detection.get("class_id")
            x_center = (x1 + x2) // 2
            x_distance, _ = self.distance_calculator.cal_distance_a2cam((x_center, y2))
            vehicle = x1, y1, x2, y2, confidence, class_id

            vehicle_distances.append((x_distance, vehicle, x_center, y2))

        # 按距离排序，取最近的前5个
        vehicle_distances.sort(key=lambda x: x[0])
        max_dashed_lines = 5  # 只显示最近距离的前5个虚线

        for i, (x_distance, vehicle, x_center, y_bottom) in enumerate(vehicle_distances):
            x1, y1, x2, y2, confidence, class_id = vehicle

            # 获取车辆类别名称
            vehicle_class_name = self.vehicle_detector.vehicle_classes.get(class_id, "vehicle")


            if not distances_in_included_lane: # 如果不存在车道区域，则按照这样的数据来显示
                # 根据距离选择颜色和状态
                if x_distance < self.config.SAFE_DISTANCE:
                    color = (0, 0, 255)  # 红色 - 危险
                    status = "danger"
                elif x_distance < self.config.SAFE_DISTANCE * 1.5:
                    color = (0, 165, 255)  # 橙色 - 预警
                    status = "warning"
                else:
                    color = (0, 255, 0)  # 绿色 - 安全
                    status = "safe"

                # 绘制掩码（可选择）
                if contours:
                    _ = cv2.drawContours(b_mask, [contours[i]], -1, (255, 255, 255), cv2.FILLED)
                    # 生成彩色蒙版层(3通道)
                    mask_color = np.zeros_like(img)
                    mask_color[:] = color # 显示当前预警颜色
                    mask_bool = b_mask.astype(bool)

                    # 将彩色掩码叠加到图像上（你可以调节透明度 alpha）
                    alpha = 0.5
                    result_img[mask_bool] = cv2.addWeighted(result_img, 1.0, mask_color, alpha, 0)[mask_bool]
            else:
                # 根据距离选择颜色和状态
                if x_distance < self.config.SAFE_DISTANCE and distances_in_included_lane.get(x_distance):
                    color = (0, 0, 255)  # 红色 - 危险
                    status = "danger"
                elif x_distance < self.config.SAFE_DISTANCE * 1.5 and distances_in_included_lane.get(x_distance):
                    color = (0, 165, 255)  # 橙色 - 预警
                    status = "warning"
                else:
                    color = (0, 255, 0)  # 绿色 - 安全
                    status = "safe"

                # 绘制掩码（可选择）
                if contours:
                    _ = cv2.drawContours(b_mask, [contours[i]], -1, (255, 255, 255), cv2.FILLED)
                    # 生成彩色蒙版层(3通道)
                    mask_color = np.zeros_like(img)
                    mask_color[:] = color # 显示当前预警颜色
                    mask_bool = b_mask.astype(bool)

                    # 将彩色掩码叠加到图像上（你可以调节透明度 alpha）
                    alpha = 0.5
                    result_img[mask_bool] = cv2.addWeighted(result_img, 1.0, mask_color, alpha, 0)[mask_bool]


            # 绘制加粗的边界框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 6)

            # 准备显示文本（简化版本）
            class_text = f"cls: {vehicle_class_name}"
            confidence_text = f"conf: {confidence:.2f}"
            distance_text = f"distance: {x_distance:.1f}m"
            status_text = f"status: {status}"

            # 快速计算信息框尺寸（使用固定宽度优化性能）
            max_width = 280  # 固定宽度，避免重复计算
            info_height = 110

            # 绘制半透明信息背景框
            overlay = result_img.copy()
            cv2.rectangle(overlay, (x1, y1 - info_height), (x1 + max_width, y1), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, result_img, 0.3, 0, result_img)

            # 绘制信息框边框
            cv2.rectangle(result_img, (x1, y1 - info_height), (x1 + max_width, y1), color, 3)

            # 绘制文本信息（白色文字）
            text_color = (255, 255, 255)
            cv2.putText(result_img, class_text, (x1 + 10, y1 - 80), font, font_scale, text_color, thickness)
            cv2.putText(result_img, confidence_text, (x1 + 10, y1 - 55), font, font_scale, text_color, thickness)
            cv2.putText(result_img, distance_text, (x1 + 10, y1 - 30), font, font_scale, text_color, thickness)
            cv2.putText(result_img, status_text, (x1 + 10, y1 - 5), font, font_scale, color, thickness)

            # 在车辆底部中心绘制距离标记点
            cv2.circle(result_img, (x_center, y_bottom), 10, color, -1)
            cv2.circle(result_img, (x_center, y_bottom), 10, (255, 255, 255), 3)

            # 只为最近的前N个车辆绘制虚影连线
            if i < max_dashed_lines:
                self._draw_dashed_line(result_img, (x_center, y_bottom), bottom_center, color, 2, 15, 10)

                # 在连线中点显示距离数值
                mid_x = (x_center + bottom_center[0]) // 2
                mid_y = (y_bottom + bottom_center[1]) // 2
                if i == 0:
                    big_distance_text = f"{x_distance:.1f}m"
                    big_text_size = cv2.getTextSize(big_distance_text, font, big_font_scale, 3)[0]
                    text_x = mid_x - big_text_size[0] // 2
                    text_y = mid_y

                    # 距离文字背景
                    cv2.rectangle(result_img, (text_x - 5, text_y - 30), (text_x + big_text_size[0] + 5, text_y + 10),(0, 0, 0), -1)
                    cv2.rectangle(result_img, (text_x - 5, text_y - 30), (text_x + big_text_size[0] + 5, text_y + 10), color, 2)

                    cv2.putText(result_img, big_distance_text, (text_x, text_y), font, big_font_scale, (0, 255, 255), 3)
                    pass
                else:
                    big_distance_text = f"{x_distance:.1f}m"
                    big_text_size = cv2.getTextSize(big_distance_text, font, big_font_scale, 3)[0]
                    text_x = mid_x - big_text_size[0] // 2
                    text_y = mid_y

                    # 距离文字背景
                    cv2.rectangle(result_img, (text_x - 5, text_y - 30), (text_x + big_text_size[0] + 5, text_y + 10),(0, 0, 0), -1)
                    cv2.rectangle(result_img, (text_x - 5, text_y - 30), (text_x + big_text_size[0] + 5, text_y + 10),color, 2)

                    cv2.putText(result_img, big_distance_text, (text_x, text_y), font, big_font_scale, (0, 255, 0), 3)

        return result_img

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
        """
        绘制虚线的高效实现
        
        Args:
            img: 图像
            pt1: 起点 (x, y)
            pt2: 终点 (x, y)
            color: 颜色 (B, G, R)
            thickness: 线条粗细
            dash_length: 虚线段长度
            gap_length: 间隔长度
        """
        # 计算线段总长度和方向
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        total_length = int(np.sqrt(dx*dx + dy*dy))
        
        if total_length == 0:
            return
        
        # 单位方向向量
        unit_x = dx / total_length
        unit_y = dy / total_length
        
        # 绘制虚线段
        current_length = 0
        segment_length = dash_length + gap_length
        
        while current_length < total_length:
            # 计算当前段的起点
            start_x = int(pt1[0] + current_length * unit_x)
            start_y = int(pt1[1] + current_length * unit_y)
            
            # 计算当前段的终点
            end_length = min(current_length + dash_length, total_length)
            end_x = int(pt1[0] + end_length * unit_x)
            end_y = int(pt1[1] + end_length * unit_y)
            
            # 绘制实线段
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
            
            # 移动到下一段
            current_length += segment_length


class MainWindow(QMainWindow):
    """
    主窗口类
    提供用户界面，显示视频和控制功能
    """

    def __init__(self):
        super().__init__()
        self.lane_detection_enabled = True  # 车道线检测开关状态
        self.video_processor = VideoProcessor()
        # 同步车道线检测状态到VideoProcessor
        self.video_processor.set_lane_detection_enabled(self.lane_detection_enabled)
        self.current_video_path = None

        # 优化的UI更新机制
        self.display_timer = QTimer()  # UI显示定时器
        self.display_timer.timeout.connect(self.update_display)
        self.display_fps = 30  # UI显示帧率
        self.display_interval = 1000 // self.display_fps  # 显示间隔(ms)

        # 性能监控
        self.last_display_time = 0
        self.display_frame_count = 0
        self.ui_fps = 0

        # 帧显示优化
        self.pending_frame_update = False  # 是否有待更新的帧
        self.last_frame_data = None  # 上一帧数据缓存

        # 创建UI组件
        self.init_ui()

        # 连接信号和槽
        self.connect_signals()

        # 应用现代化样式
        self.apply_modern_style()

    def init_ui(self):
        """
        初始化用户界面
        """
        self.setWindowTitle("🚗  智能车道检测与碰撞预警系统 v3.0.0")
        self.setGeometry(50, 50, 1800, 1200)
        self.setMinimumSize(1600, 1000)

        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        central_layout = QHBoxLayout(central_widget)
        central_layout.addWidget(main_splitter)
        central_layout.setContentsMargins(10, 10, 10, 10)

        # 左侧视频区域
        video_widget = self.create_video_widget()
        main_splitter.addWidget(video_widget)

        # 右侧控制面板
        control_panel = self.create_control_panel()
        main_splitter.addWidget(control_panel)

        # 设置分割器比例
        main_splitter.setSizes([1300, 500])
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)

    def create_video_widget(self):
        """
        创建视频显示区域
        """
        video_widget = QFrame()
        video_widget.setObjectName("videoFrame")
        video_layout = QVBoxLayout(video_widget)
        video_layout.setSpacing(15)

        # 视频标题
        title_label = QLabel("📹 视频监控区域")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)  # 设置文字居中
        video_layout.addWidget(title_label)

        # 视频显示标签
        self.video_label = QLabel()
        # self.video_label.setFixedSize(640, 630)
        self.video_label.setMinimumSize(1200, 800)
        self.video_label.setObjectName("videoDisplay")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("🎬 请选择视频文件或摄像头开始检测")
        video_layout.addWidget(self.video_label)

        # 控制按钮区域
        control_frame = QFrame()
        control_frame.setObjectName("controlFrame")
        control_layout = QHBoxLayout(control_frame)
        control_layout.setSpacing(10)

        # 创建现代化按钮
        self.open_file_btn = QPushButton("📁 视频文件")
        self.open_camera_btn = QPushButton("📷 摄像头")
        self.color_picker_btn = QPushButton("🎨 主题")
        self.lane_detection_btn = QPushButton("🛣️ 车道线检测: 开启")
        self.start_btn = QPushButton("▶️ 开始检测")
        self.stop_btn = QPushButton("⏹️ 停止检测")

        # 设置按钮样式类
        self.open_file_btn.setObjectName("primaryButton")
        self.open_camera_btn.setObjectName("primaryButton")
        self.color_picker_btn.setObjectName("primaryButton")
        self.lane_detection_btn.setObjectName("successButton")
        self.start_btn.setObjectName("successButton")
        self.stop_btn.setObjectName("dangerButton")

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        control_layout.addWidget(self.open_file_btn)
        control_layout.addWidget(self.open_camera_btn)
        control_layout.addWidget(self.color_picker_btn)
        control_layout.addWidget(self.lane_detection_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)

        video_layout.addWidget(control_frame)

        return video_widget

    def create_control_panel(self):
        """
        创建右侧控制面板
        """
        panel = QFrame()
        panel.setObjectName("controlPanel")
        panel.setFixedWidth(480)
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)

        # 面板标题
        title = QLabel("⚙️ 系统控制中心")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        # 预警状态区域
        warning_section = self.create_warning_section()
        layout.addWidget(warning_section)

        # 系统状态区域
        status_section = self.create_status_section()
        layout.addWidget(status_section)

        # 参数设置区域
        params_section = self.create_params_section()
        layout.addWidget(params_section)

        # 性能设置区域
        performance_section = self.create_performance_section()
        layout.addWidget(performance_section)

        # 日志区域 - 填满剩余空间
        log_section = self.create_log_section()
        layout.addWidget(log_section, 1)  # 设置拉伸因子为1，让日志区域填满剩余空间

        return panel

    def create_warning_section(self):
        """
        创建预警状态区域
        """
        section = QGroupBox("🚨 预警状态")
        section.setObjectName("warningSection")
        layout = QVBoxLayout(section)

        self.warning_label = QLabel("系统正常运行")
        self.warning_label.setObjectName("warningDisplay")
        self.warning_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.warning_label)

        return section

    def create_status_section(self):
        """
        创建系统状态区域
        """
        section = QGroupBox("📊 实时状态")
        section.setObjectName("statusSection")
        layout = QGridLayout(section)
        layout.setSpacing(10)

        # 检测帧率显示
        layout.addWidget(QLabel("检测帧率:"), 0, 0)
        self.fps_label = QLabel("0 FPS")
        self.fps_label.setObjectName("statusValue")
        layout.addWidget(self.fps_label, 0, 1)

        # UI帧率显示
        layout.addWidget(QLabel("UI帧率:"), 1, 0)
        self.ui_fps_label = QLabel("0 FPS")
        self.ui_fps_label.setObjectName("statusValue")
        layout.addWidget(self.ui_fps_label, 1, 1)

        # 车辆数量显示
        layout.addWidget(QLabel("检测车辆:"), 2, 0)
        self.vehicle_count_label = QLabel("0 辆")
        self.vehicle_count_label.setObjectName("statusValue")
        layout.addWidget(self.vehicle_count_label, 2, 1)

        # 车道线状态
        layout.addWidget(QLabel("车道线:"), 3, 0)
        self.lane_status_label = QLabel("未检测")
        self.lane_status_label.setObjectName("statusValue")
        layout.addWidget(self.lane_status_label, 3, 1)

        # 运行时间
        layout.addWidget(QLabel("运行时间:"), 4, 0)
        self.run_time_label = QLabel("0")
        self.run_time_label.setObjectName("statusValue")
        layout.addWidget(self.run_time_label, 4, 1)

        # 处理帧数 - 已屏蔽
        # layout.addWidget(QLabel("处理帧数:"), 5, 0)
        # self.frame_count_label = QLabel("0")
        # self.frame_count_label.setObjectName("statusValue")
        # layout.addWidget(self.frame_count_label, 5, 1)


        return section

    def create_params_section(self):
        """
        创建参数设置区域
        """
        section = QGroupBox("🔧 参数调节")
        section.setObjectName("paramsSection")
        layout = QGridLayout(section)
        layout.setSpacing(15)

        # 安全距离设置
        layout.addWidget(QLabel("安全距离(米):"), 0, 0)
        self.safe_distance_slider = QSlider(Qt.Horizontal)
        self.safe_distance_slider.setRange(1, 30)
        self.safe_distance_slider.setValue(15)
        self.safe_distance_slider.setObjectName("modernSlider")
        self.safe_distance_label = QLabel("15")
        self.safe_distance_label.setObjectName("sliderValue")
        layout.addWidget(self.safe_distance_slider, 0, 1)
        layout.addWidget(self.safe_distance_label, 0, 2)

        # 检测置信度设置
        layout.addWidget(QLabel("检测置信度:"), 1, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setObjectName("modernSlider")
        self.confidence_label = QLabel("0.5")
        self.confidence_label.setObjectName("sliderValue")
        layout.addWidget(self.confidence_slider, 1, 1)
        layout.addWidget(self.confidence_label, 1, 2)

        # 科技感可视化开关
        layout.addWidget(QLabel("科技感效果:"), 2, 0)
        self.tech_visual_checkbox = QCheckBox("启用科技感HUD")
        self.tech_visual_checkbox.setChecked(False)  # 默认启用
        self.tech_visual_checkbox.setObjectName("modernCheckbox")
        layout.addWidget(self.tech_visual_checkbox, 2, 1, 1, 2)

        return section

    def create_performance_section(self):
        """
        创建性能设置区域
        """
        section = QGroupBox("⚡ 性能优化")
        section.setObjectName("performanceSection")
        layout = QGridLayout(section)
        layout.setSpacing(15)

        # 检测间隔设置
        layout.addWidget(QLabel("检测间隔(帧):"), 0, 0)
        self.detection_interval_slider = QSlider(Qt.Horizontal)
        self.detection_interval_slider.setRange(1, 10)
        self.detection_interval_slider.setValue(1)
        self.detection_interval_slider.setObjectName("modernSlider")
        self.detection_interval_label = QLabel("1")
        self.detection_interval_label.setObjectName("sliderValue")
        layout.addWidget(self.detection_interval_slider, 0, 1)
        layout.addWidget(self.detection_interval_label, 0, 2)

        # 目标帧率设置
        layout.addWidget(QLabel("目标帧率:"), 1, 0)
        self.target_fps_slider = QSlider(Qt.Horizontal)
        self.target_fps_slider.setRange(15, 60)
        self.target_fps_slider.setValue(30)
        self.target_fps_slider.setObjectName("modernSlider")
        self.target_fps_label = QLabel("30")
        self.target_fps_label.setObjectName("sliderValue")
        layout.addWidget(self.target_fps_slider, 1, 1)
        layout.addWidget(self.target_fps_label, 1, 2)

        return section

    def create_log_section(self):
        """
        创建日志显示区域
        """
        section = QGroupBox("📝 系统日志")
        section.setObjectName("logSection")
        layout = QVBoxLayout(section)

        self.log_text = QTextEdit()
        self.log_text.setObjectName("logDisplay")
        # 移除最大高度限制，让日志区域填满分配的空间
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        return section

    def apply_modern_style(self):
        """
        应用现代化样式 - 融合色调版本
        """
        style = """
        /* 主窗口样式 - 柔和蓝灰基调 */
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #f0f2f5, stop:0.5 #e8eaf0, stop:1 #dde1e8);
        }

        /* 左侧视频区域标题样式 - 统一蓝色系 */
        #titleLabel {
            font-size: 32px;
            font-weight: bold;
            color: white;
            padding: 22px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #4a90e2, stop:0.3 #5ba3f5, stop:0.7 #6bb6ff, stop:1 #7bc9ff);
            border-radius: 18px;
            text-align: center;
            border: 3px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 25px rgba(74, 144, 226, 0.4), inset 0 0 15px rgba(255, 255, 255, 0.2);
        }

        /* 右侧控制面板标题样式 - 协调蓝色 */
        #panelTitle {
            font-size: 24px;
            font-weight: bold;
            color: white;
            padding: 18px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #5ba3f5, stop:0.5 #6bb6ff, stop:1 #7bc9ff);
            border-radius: 15px;
            text-align: center;
            border: 2px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 6px 20px rgba(91, 163, 245, 0.4);
        }

        /* 视频显示区域 - 柔和背景 */
        #videoFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:0.5 #f8f9fb, stop:1 #f0f2f5);
            border: 2px solid rgba(74, 144, 226, 0.3);
            border-radius: 15px;
            padding: 18px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        }

        #videoDisplay {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #2c3e50, stop:0.3 #34495e, stop:0.7 #3d566e, stop:1 #455a7a);
            color: #ffffff;
            font-size: 28px;
            font-weight: bold;
            border: 3px solid rgba(74, 144, 226, 0.5);
            border-radius: 25px;
            padding: 40px;
            box-shadow: inset 0 0 30px rgba(74, 144, 226, 0.3), 0 0 20px rgba(74, 144, 226, 0.3);
        }

        /* 控制面板 - 温暖灰蓝色 */
        #controlPanel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f8f9fb, stop:0.3 #eef1f5, stop:0.7 #e8eaf0, stop:1 #dde1e8);
            border: 2px solid rgba(74, 144, 226, 0.4);
            border-radius: 15px;
            padding: 18px;
            box-shadow: 0 8px 25px rgba(74, 144, 226, 0.2);
        }

        /* 按钮样式 - 统一蓝色系 */
        #primaryButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #4a90e2, stop:0.3 #5ba3f5, stop:0.7 #6bb6ff, stop:1 #7bc9ff);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 22px 35px;
            border-radius: 15px;
            font-size: 20px;
            font-weight: bold;
            box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4), inset 0 0 10px rgba(255, 255, 255, 0.2);
        }

        #primaryButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #5ba3f5, stop:0.3 #6bb6ff, stop:0.7 #7bc9ff, stop:1 #8bd4ff);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(74, 144, 226, 0.5), inset 0 0 15px rgba(255, 255, 255, 0.3);
        }

        #primaryButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3a7bc8, stop:1 #2e6ba8);
            transform: translateY(1px);
        }

        #successButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #52c41a, stop:0.3 #73d13d, stop:0.7 #95de64, stop:1 #b7eb8f);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 18px 28px;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(82, 196, 26, 0.4), inset 0 0 8px rgba(255, 255, 255, 0.2);
        }

        #successButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #73d13d, stop:0.3 #95de64, stop:0.7 #b7eb8f, stop:1 #d9f7be);
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(82, 196, 26, 0.5);
        }

        #successButton:disabled {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #bfbfbf, stop:1 #d9d9d9);
            color: #8c8c8c;
        }

        #dangerButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ff4d4f, stop:0.3 #ff7875, stop:0.7 #ffa39e, stop:1 #ffccc7);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 18px 28px;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(255, 77, 79, 0.4), inset 0 0 8px rgba(255, 255, 255, 0.2);
        }

        #dangerButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ff7875, stop:0.3 #ffa39e, stop:0.7 #ffccc7, stop:1 #fff1f0);
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(255, 77, 79, 0.5);
        }

        #dangerButton:disabled {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #bfbfbf, stop:1 #d9d9d9);
            color: #8c8c8c;
        }

        /* 分组框样式 - 柔和背景 */
        QGroupBox {
            font-weight: bold;
            border: 2px solid rgba(74, 144, 226, 0.3);
            border-radius: 12px;
            margin-top: 12px;
            padding-top: 12px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:0.3 #f8f9fb, stop:0.7 #f0f2f5, stop:1 #e8eaf0);
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.15);
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 10px 0 10px;
            color: #2c3e50;
            font-size: 16px;
            font-weight: bold;
        }

        /* 预警显示 - 协调绿色 */
        #warningDisplay {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #52c41a, stop:1 #389e0d);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        }

        /* 状态值样式 - 柔和蓝灰 */
        #statusValue {
            color: #2c3e50;
            font-weight: bold;
            font-size: 16px;
            padding: 8px 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f0f2f5, stop:0.5 #e8eaf0, stop:1 #dde1e8);
            border-radius: 12px;
            border: 1px solid rgba(74, 144, 226, 0.2);
        }

        /* 滑块样式 - 统一蓝色 */
        #modernSlider::groove:horizontal {
            border: 2px solid rgba(74, 144, 226, 0.3);
            height: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #f0f2f5, stop:1 #dde1e8);
            border-radius: 6px;
        }

        #modernSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #4a90e2, stop:0.5 #5ba3f5, stop:1 #6bb6ff);
            border: 2px solid #3a7bc8;
            width: 22px;
            margin: -6px 0;
            border-radius: 11px;
            box-shadow: 0 3px 8px rgba(74, 144, 226, 0.4);
        }

        #modernSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #5ba3f5, stop:0.5 #6bb6ff, stop:1 #7bc9ff);
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.5);
        }

        #sliderValue {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #4a90e2, stop:1 #3a7bc8);
            color: white;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: bold;
            min-width: 35px;
            box-shadow: 0 3px 8px rgba(74, 144, 226, 0.3);
        }

        /* 日志显示 - 深色协调 */
        #logDisplay {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #2c3e50, stop:0.3 #34495e, stop:0.7 #3d566e, stop:1 #455a7a);
            color: #ecf0f1;
            border: 2px solid rgba(74, 144, 226, 0.4);
            border-radius: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            padding: 12px;
            box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.2), 0 4px 12px rgba(74, 144, 226, 0.2);
        }

        /* 控制框架 - 协调背景 */
        #controlFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f8f9fb, stop:0.3 #f0f2f5, stop:0.7 #e8eaf0, stop:1 #dde1e8);
            border: 2px solid rgba(74, 144, 226, 0.3);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.15);
        }

        /* 复选框样式 */
        #modernCheckbox {
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
            spacing: 8px;
        }

        #modernCheckbox::indicator {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid rgba(74, 144, 226, 0.5);
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:1 #f0f2f5);
        }

        #modernCheckbox::indicator:checked {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #4a90e2, stop:1 #3a7bc8);
            border: 2px solid #3a7bc8;
        }

        #modernCheckbox::indicator:checked:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #5ba3f5, stop:1 #4a90e2);
        }
        """
        self.setStyleSheet(style)

    def apply_modern_style_woman(self):
        """
        应用现代化样式 - 粉色渐变版本
        """
        style = """
        /* 主窗口样式 - 柔和蓝灰基调 */
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #f0f2f5, stop:0.5 #e8eaf0, stop:1 #fdeff4);
        }

        /* 左侧视频区域标题样式 - 统一粉色系 */
        #titleLabel {
            font-size: 32px;
            font-weight: bold;
            color: white;
            padding: 22px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f78fb3, stop:0.3 #f9a8c4, stop:0.7 #fbbbd4, stop:1 #fdd0e3);
            border-radius: 18px;
            text-align: center;
            border: 3px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 25px rgba(247, 143, 179, 0.4), inset 0 0 15px rgba(255, 255, 255, 0.2);
        }

        /* 右侧控制面板标题样式 - 协调粉色 */
        #panelTitle {
            font-size: 24px;
            font-weight: bold;
            color: white;
            padding: 18px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f9a8c4, stop:0.5 #fbbbd4, stop:1 #fdd0e3);
            border-radius: 15px;
            text-align: center;
            border: 2px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 6px 20px rgba(249, 168, 196, 0.4);
        }

        /* 视频显示区域 - 柔和背景 */
        #videoFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:0.5 #f8f9fb, stop:1 #f0f2f5);
            border: 2px solid rgba(247, 143, 179, 0.3);
            border-radius: 15px;
            padding: 18px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        }

        #videoDisplay {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #2c3e50, stop:0.3 #34495e, stop:0.7 #3d566e, stop:1 #455a7a);
            color: #ffffff;
            font-size: 28px;
            font-weight: bold;
            border: 3px solid rgba(247, 143, 179, 0.5);
            border-radius: 25px;
            padding: 40px;
            box-shadow: inset 0 0 30px rgba(247, 143, 179, 0.3), 0 0 20px rgba(247, 143, 179, 0.3);
        }

        /* 控制面板 - 温暖灰粉色 */
        #controlPanel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f8f9fb, stop:0.3 #fef1f5, stop:0.7 #fdeff4, stop:1 #fddce9);
            border: 2px solid rgba(247, 143, 179, 0.4);
            border-radius: 15px;
            padding: 18px;
            box-shadow: 0 8px 25px rgba(247, 143, 179, 0.2);
        }

        /* 按钮样式 - 统一粉色系 */
        #primaryButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f78fb3, stop:0.3 #f9a8c4, stop:0.7 #fbbbd4, stop:1 #fdd0e3);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 22px 35px;
            border-radius: 15px;
            font-size: 20px;
            font-weight: bold;
            box-shadow: 0 6px 20px rgba(247, 143, 179, 0.4), inset 0 0 10px rgba(255, 255, 255, 0.2);
        }

        #primaryButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f9a8c4, stop:0.3 #fbbbd4, stop:0.7 #fdd0e3, stop:1 #ffe3ef);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(247, 143, 179, 0.5), inset 0 0 15px rgba(255, 255, 255, 0.3);
        }

        #primaryButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #e07a9c, stop:1 #c96b8a);
            transform: translateY(1px);
        }

        /* 下面的绿色、红色按钮保持不变 */

        /* 分组框样式 - 柔和背景 */
        QGroupBox {
            font-weight: bold;
            border: 2px solid rgba(247, 143, 179, 0.3);
            border-radius: 12px;
            margin-top: 12px;
            padding-top: 12px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:0.3 #f8f9fb, stop:0.7 #f0f2f5, stop:1 #fdeff4);
            box-shadow: 0 4px 12px rgba(247, 143, 179, 0.15);
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 10px 0 10px;
            color: #2c3e50;
            font-size: 16px;
            font-weight: bold;
        }
        
         /* 预警显示 - 协调绿色 */
        #warningDisplay {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #52c41a, stop:1 #389e0d);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        }
        
        /* 状态值样式 - 柔和粉灰 */
        #statusValue {
            color: #2c3e50;
            font-weight: bold;
            font-size: 16px;
            padding: 8px 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #fdeff4, stop:0.5 #fddce9, stop:1 #fbcfe8);
            border-radius: 12px;
            border: 1px solid rgba(247, 143, 179, 0.2);
        }

        /* 滑块样式 - 粉色 */
        #modernSlider::groove:horizontal {
            border: 2px solid rgba(247, 143, 179, 0.3);
            height: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #fdeff4, stop:1 #fddce9);
            border-radius: 6px;
        }

        #modernSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f78fb3, stop:0.5 #f9a8c4, stop:1 #fbbbd4);
            border: 2px solid #e07a9c;
            width: 22px;
            margin: -6px 0;
            border-radius: 11px;
            box-shadow: 0 3px 8px rgba(247, 143, 179, 0.4);
        }

        #modernSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f9a8c4, stop:0.5 #fbbbd4, stop:1 #fdd0e3);
            box-shadow: 0 4px 12px rgba(247, 143, 179, 0.5);
        }

        #sliderValue {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f78fb3, stop:1 #e07a9c);
            color: white;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: bold;
            min-width: 35px;
            box-shadow: 0 3px 8px rgba(247, 143, 179, 0.3);
        }

        /* 日志显示 - 深色协调 */
        #logDisplay {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #2c3e50, stop:0.3 #34495e, stop:0.7 #3d566e, stop:1 #455a7a);
            color: #ecf0f1;
            border: 2px solid rgba(247, 143, 179, 0.4);
            border-radius: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            padding: 12px;
            box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.2), 0 4px 12px rgba(247, 143, 179, 0.2);
        }

        /* 控制框架 - 协调背景 */
        #controlFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f8f9fb, stop:0.3 #fdeff4, stop:0.7 #fddce9, stop:1 #fbcfe8);
            border: 2px solid rgba(247, 143, 179, 0.3);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(247, 143, 179, 0.15);
        }

        /* 复选框样式 */
        #modernCheckbox {
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
            spacing: 8px;
        }

        #modernCheckbox::indicator {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid rgba(247, 143, 179, 0.5);
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:1 #fdeff4);
        }

        #modernCheckbox::indicator:checked {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f78fb3, stop:1 #e07a9c);
            border: 2px solid #e07a9c;
        }

        #modernCheckbox::indicator:checked:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f9a8c4, stop:1 #f78fb3);
        }
        """
        self.setStyleSheet(style)

    def apply_modern_style_woman2(self):
        """
        应用现代化样式 - 融合色调版本（蓝色替换为浅粉色系）
        """
        style = """
        /* 主窗口样式 - 柔和蓝灰基调 */
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #f0f2f5, stop:0.5 #e8eaf0, stop:1 #dde1e8);
        }

        /* 左侧视频区域标题样式 - 统一粉色系 */
        #titleLabel {
            font-size: 32px;
            font-weight: bold;
            color: white;
            padding: 22px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f7b6c2, stop:0.3 #f9c6d1, stop:0.7 #fbd6e0, stop:1 #fde6ef);
            border-radius: 18px;
            text-align: center;
            border: 3px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 25px rgba(247, 182, 194, 0.4), inset 0 0 15px rgba(255, 255, 255, 0.2);
        }

        /* 右侧控制面板标题样式 - 协调粉色 */
        #panelTitle {
            font-size: 24px;
            font-weight: bold;
            color: white;
            padding: 18px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f9c6d1, stop:0.5 #fbd6e0, stop:1 #fde6ef);
            border-radius: 15px;
            text-align: center;
            border: 2px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 6px 20px rgba(249, 198, 209, 0.4);
        }

        /* 视频显示区域 - 柔和背景 */
        #videoFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:0.5 #f8f9fb, stop:1 #f0f2f5);
            border: 2px solid rgba(247, 143, 179, 0.3);
            border-radius: 15px;
            padding: 18px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        }

        #videoDisplay {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #2c3e50, stop:0.3 #34495e, stop:0.7 #3d566e, stop:1 #455a7a);
            color: #ffffff;
            font-size: 28px;
            font-weight: bold;
            border: 3px solid rgba(247, 143, 179, 0.5);
            border-radius: 25px;
            padding: 40px;
            box-shadow: inset 0 0 30px rgba(247, 143, 179, 0.3), 0 0 20px rgba(247, 143, 179, 0.3);
        }

        /* 控制面板 - 温暖灰粉色 */
        #controlPanel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f8f9fb, stop:0.3 #eef1f5, stop:0.7 #e8eaf0, stop:1 #dde1e8);
            border: 2px solid rgba(247, 182, 194, 0.4);
            border-radius: 15px;
            padding: 18px;
            box-shadow: 0 8px 25px rgba(247, 182, 194, 0.2);
        }

        /* 按钮样式 - 统一粉色系 */
        #primaryButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f7b6c2, stop:0.3 #f9c6d1, stop:0.7 #fbd6e0, stop:1 #fde6ef);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 22px 35px;
            border-radius: 15px;
            font-size: 20px;
            font-weight: bold;
            box-shadow: 0 6px 20px rgba(247, 182, 194, 0.4), inset 0 0 10px rgba(255, 255, 255, 0.2);
        }

        #primaryButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f9c6d1, stop:0.3 #fbd6e0, stop:0.7 #fde6ef, stop:1 #fff0f5);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(247, 182, 194, 0.5), inset 0 0 15px rgba(255, 255, 255, 0.3);
        }

        #primaryButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #e39aa7, stop:1 #d48090);
            transform: translateY(1px);
        }

        #successButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #52c41a, stop:0.3 #73d13d, stop:0.7 #95de64, stop:1 #b7eb8f);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 18px 28px;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(82, 196, 26, 0.4), inset 0 0 8px rgba(255, 255, 255, 0.2);
        }

        #successButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #73d13d, stop:0.3 #95de64, stop:0.7 #b7eb8f, stop:1 #d9f7be);
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(82, 196, 26, 0.5);
        }

        #successButton:disabled {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #bfbfbf, stop:1 #d9d9d9);
            color: #8c8c8c;
        }

        #dangerButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ff4d4f, stop:0.3 #ff7875, stop:0.7 #ffa39e, stop:1 #ffccc7);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 18px 28px;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(255, 77, 79, 0.4), inset 0 0 8px rgba(255, 255, 255, 0.2);
        }

        #dangerButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ff7875, stop:0.3 #ffa39e, stop:0.7 #ffccc7, stop:1 #fff1f0);
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(255, 77, 79, 0.5);
        }

        #dangerButton:disabled {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #bfbfbf, stop:1 #d9d9d9);
            color: #8c8c8c;
        }

        /* 分组框样式 - 柔和背景 */
        QGroupBox {
            font-weight: bold;
            border: 2px solid rgba(247, 182, 194, 0.3);
            border-radius: 12px;
            margin-top: 12px;
            padding-top: 12px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:0.3 #f8f9fb, stop:0.7 #f0f2f5, stop:1 #e8eaf0);
            box-shadow: 0 4px 12px rgba(247, 182, 194, 0.15);
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 10px 0 10px;
            color: #2c3e50;
            font-size: 16px;
            font-weight: bold;
        }

        /* 预警显示 - 协调绿色 */
        #warningDisplay {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #52c41a, stop:1 #389e0d);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        }

        /* 状态值样式 - 柔和灰粉 */
        #statusValue {
            color: #2c3e50;
            font-weight: bold;
            font-size: 16px;
            padding: 8px 15px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f0f2f5, stop:0.5 #e8eaf0, stop:1 #fde6ef);
            border-radius: 12px;
            border: 1px solid rgba(247, 182, 194, 0.2);
        }

        /* 滑块样式 - 统一粉色 */
        #modernSlider::groove:horizontal {
            border: 2px solid rgba(247, 182, 194, 0.3);
            height: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #f0f2f5, stop:1 #fde6ef);
            border-radius: 6px;
        }

        #modernSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f7b6c2, stop:0.5 #f9c6d1, stop:1 #fbd6e0);
            border: 2px solid #e39aa7;
            width: 22px;
            margin: -6px 0;
            border-radius: 11px;
            box-shadow: 0 3px 8px rgba(247, 182, 194, 0.4);
        }

        #modernSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f9c6d1, stop:0.5 #fbd6e0, stop:1 #fde6ef);
            box-shadow: 0 4px 12px rgba(247, 182, 194, 0.5);
        }

        #sliderValue {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f7b6c2, stop:1 #e39aa7);
            color: white;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: bold;
            min-width: 35px;
            box-shadow: 0 3px 8px rgba(247, 182, 194, 0.3);
        }

        /* 日志显示 - 深色协调 */
        #logDisplay {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #2c3e50, stop:0.3 #34495e, stop:0.7 #3d566e, stop:1 #455a7a);
            color: #ecf0f1;
            border: 2px solid rgba(247, 182, 194, 0.4);
            border-radius: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            padding: 12px;
            box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.2), 0 4px 12px rgba(247, 182, 194, 0.2);
        }

        /* 控制框架 - 协调背景 */
        #controlFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f8f9fb, stop:0.3 #f0f2f5, stop:0.7 #e8eaf0, stop:1 #dde1e8);
            border: 2px solid rgba(247, 182, 194, 0.3);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(247, 182, 194, 0.15);
        }

        /* 复选框样式 */
        #modernCheckbox {
            font-size: 14px;
            font-weight: bold;
            color: #2c3e50;
            spacing: 8px;
        }

        #modernCheckbox::indicator {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid rgba(247, 182, 194, 0.5);
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:1 #f0f2f5);
        }

        #modernCheckbox::indicator:checked {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f7b6c2, stop:1 #e39aa7);
            border: 2px solid #e39aa7;
        }

        #modernCheckbox::indicator:checked:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f9c6d1, stop:1 #f7b6c2);
        }
        """
        self.setStyleSheet(style)

    def connect_signals(self):
        """
        连接信号和槽
        """
        # 按钮信号
        self.open_file_btn.clicked.connect(self.open_video_file)
        self.open_camera_btn.clicked.connect(self.open_camera)
        # self.color_picker_btn.clicked.connect(self.choose_custom_color)
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)

        # 滑块信号
        self.safe_distance_slider.valueChanged.connect(self.update_safe_distance)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.detection_interval_slider.valueChanged.connect(self.update_detection_interval)
        self.target_fps_slider.valueChanged.connect(self.update_target_fps)
        
        # 复选框信号
        self.tech_visual_checkbox.stateChanged.connect(self.update_tech_visual)

        # 视频处理器信号
        # self.video_processor.frame_processed.connect(self.update_video_display)
        self.video_processor.frame_ready.connect(self.on_frame_ready)
        self.video_processor.warning_signal.connect(self.update_warning_status)
        self.video_processor.stats_updated.connect(self.update_stats_display)
        self.video_processor.first_frame_ready.connect(self.show_first_frame)

        # 若已存在颜色选择器按钮，则连接到颜色选择方法（不影响没有该按钮的情况）
        if hasattr(self, "color_picker_btn"):
            self.color_picker_btn.clicked.connect(self.choose_custom_color)

        # 车道线检测开关按钮信号
        if hasattr(self, "lane_detection_btn"):
            self.lane_detection_btn.clicked.connect(self.toggle_lane_detection)

    def choose_custom_color(self):
        """弹出颜色选择器并应用自定义主题主色（覆盖层方式，不破坏现有主题）"""
        color = QColorDialog.getColor(parent=self, title="选择主题主色")
        if color.isValid():
            self.apply_custom_theme_first_version(color.name())

    def toggle_lane_detection(self):
        """
        切换车道线检测开关
        """
        self.lane_detection_enabled = not self.lane_detection_enabled

        if self.lane_detection_enabled:
            self.lane_detection_btn.setText("🛣️ 车道线检测: 开启")
            self.lane_detection_btn.setObjectName("successButton")
            self.add_log("✅ 车道线检测已开启")
        else:
            self.lane_detection_btn.setText("🛣️ 车道线检测: 关闭")
            self.lane_detection_btn.setObjectName("dangerButton")
            self.add_log("❌ 车道线检测已关闭")

        # 重新应用样式
        self.lane_detection_btn.style().unpolish(self.lane_detection_btn)
        self.lane_detection_btn.style().polish(self.lane_detection_btn)

        # 更新VideoProcessor的车道线检测状态
        self.video_processor.set_lane_detection_enabled(self.lane_detection_enabled)

    def apply_custom_theme(self, primary_hex: str):
        """
        以覆盖层方式，使用用户选择的主色叠加到当前主题。
        覆盖 apply_modern_style 与 apply_modern_style_woman 中大部分适合主色强调的部件。
        多次点击颜色选择器会替换上一次的自定义覆盖，不会无限累积。
        """
        base = QColor(primary_hex)
        # 主色的亮/暗变化
        lighter1 = base.lighter(120).name()
        lighter2 = base.lighter(140).name()
        lighter3 = base.lighter(160).name()
        darker1 = base.darker(120).name()
        darker2 = base.darker(140).name()
        # 用于浅色背景的更淡层级
        vlight1 = base.lighter(180).name()
        vlight2 = base.lighter(200).name()
        vlight3 = base.lighter(220).name()

        r, g, b = base.red(), base.green(), base.blue()

        style = f"""
        /* ===== 自定义主题覆盖（动态主色）===== */

        /* 主窗口浅色背景（替换粉色/蓝灰背景为主色浅色系） */
        QMainWindow {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {vlight3}, stop:0.5 {vlight2}, stop:1 {vlight1});
        }}

        /* 标题：左侧视频标题与右侧控制面板标题 */
        #titleLabel {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {primary_hex}, stop:0.3 {lighter1}, stop:0.7 {lighter2}, stop:1 {lighter3});
            border: 3px solid rgba(255, 255, 255, 0.3);
            color: white;
        }}
        #panelTitle {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {lighter1}, stop:0.5 {lighter2}, stop:1 {lighter3});
            border: 2px solid rgba(255, 255, 255, 0.3);
            color: white;
        }}

        /* 容器边框强调与背景协调 */
        #videoFrame {{
            border: 2px solid rgba({r}, {g}, {b}, 0.3);
        }}
        #videoDisplay {{
            border: 3px solid rgba({r}, {g}, {b}, 0.5);
        }}
        #controlPanel {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {vlight3}, stop:0.3 {vlight2}, stop:0.7 {vlight1}, stop:1 {lighter3});
            border: 2px solid rgba({r}, {g}, {b}, 0.4);
        }}
        QGroupBox {{
            border: 2px solid rgba({r}, {g}, {b}, 0.3);
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:0.3 {vlight3}, stop:0.7 {vlight2}, stop:1 {vlight1});
        }}
        #statusValue {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {vlight3}, stop:0.5 {vlight2}, stop:1 {vlight1});
            border: 1px solid rgba({r}, {g}, {b}, 0.2);
            color: #2c3e50;
        }}

        /* 主按钮（primary） */
        #primaryButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {primary_hex}, stop:0.3 {lighter1}, stop:0.7 {lighter2}, stop:1 {lighter3});
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }}
        #primaryButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {lighter1}, stop:0.3 {lighter2}, stop:0.7 {lighter3}, stop:1 {primary_hex});
        }}
        #primaryButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {darker1}, stop:1 {darker2});
        }}

        /* 滑块槽与手柄 */
        #modernSlider::groove:horizontal {{
            border: 2px solid rgba({r}, {g}, {b}, 0.3);
            height: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {vlight3}, stop:1 {vlight2});
            border-radius: 6px;
        }}
        #modernSlider::handle:horizontal {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {primary_hex}, stop:0.5 {lighter1}, stop:1 {lighter2});
            border: 2px solid {darker1};
            width: 22px;
            margin: -6px 0;
            border-radius: 11px;
        }}
        #modernSlider::handle:horizontal:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {lighter1}, stop:0.5 {lighter2}, stop:1 {lighter3});
        }}

        /* 滑块数值徽标 */
        #sliderValue {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {primary_hex}, stop:1 {darker1});
            color: white;
        }}

        /* 日志与控制框架 */
        #logDisplay {{
            border: 2px solid rgba({r}, {g}, {b}, 0.4);
            border-radius: 12px;
        }}
        #controlFrame {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {vlight3}, stop:0.3 {vlight2}, stop:0.7 {vlight1}, stop:1 {lighter2});
            border: 2px solid rgba({r}, {g}, {b}, 0.3);
            border-radius: 12px;
        }}

        /* 复选框指示器 */
        #modernCheckbox::indicator {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid rgba({r}, {g}, {b}, 0.5);
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #ffffff, stop:1 {vlight3});
        }}
        #modernCheckbox::indicator:checked {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {primary_hex}, stop:1 {darker1});
            border: 2px solid {darker1};
        }}
        #modernCheckbox::indicator:checked:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {lighter1}, stop:1 {primary_hex});
        }}
        """

        # 移除上一次的自定义覆盖，避免叠加越来越长
        current = self.styleSheet() or ""
        prev_overlay = getattr(self, "_custom_theme_style", None)
        if prev_overlay:
            current = current.replace(prev_overlay, "")

        # 保存并应用新的覆盖
        self._custom_theme_style = style
        self.setStyleSheet(current + "\\n" + style)


    def apply_custom_theme_first_version(self, primary_hex: str):
        """
        以覆盖层方式，使用用户选择的主色叠加到当前主题。
        覆盖 apply_modern_style 中大多数适合用主色强调的蓝色系部件。
        多次点击颜色选择器时会替换上一次的自定义覆盖，不会无限累积。
        """
        base = QColor(primary_hex)
        lighter1 = base.lighter(120).name()
        lighter2 = base.lighter(140).name()
        lighter3 = base.lighter(160).name()
        darker1 = base.darker(120).name()
        darker2 = base.darker(140).name()
        r, g, b = base.red(), base.green(), base.blue()

        style = f"""
        /* ===== 自定义主题覆盖（动态主色）===== */

        /* 标题：左侧视频标题与右侧控制面板标题 */
        #titleLabel {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {primary_hex}, stop:0.3 {lighter1}, stop:0.7 {lighter2}, stop:1 {lighter3});
            border: 3px solid rgba(255, 255, 255, 0.3);
            color: white;
        }}
        #panelTitle {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {lighter1}, stop:0.5 {lighter2}, stop:1 {lighter3});
            border: 2px solid rgba(255, 255, 255, 0.3);
            color: white;
        }}

        /* 容器边框强调色（保持原有背景） */
        #videoFrame {{
            border: 2px solid rgba({r}, {g}, {b}, 0.3);
        }}
        #videoDisplay {{
            border: 3px solid rgba({r}, {g}, {b}, 0.5);
        }}
        #controlPanel {{
            border: 2px solid rgba({r}, {g}, {b}, 0.4);
        }}
        #logDisplay {{
            border: 2px solid rgba({r}, {g}, {b}, 0.4);
            border-radius: 12px;
        }}
        #controlFrame {{
            border: 2px solid rgba({r}, {g}, {b}, 0.3);
            border-radius: 12px;
        }}
        #statusValue {{
            border: 1px solid rgba({r}, {g}, {b}, 0.2);
        }}

        /* 主按钮（primary） */
        #primaryButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {primary_hex}, stop:0.3 {lighter1}, stop:0.7 {lighter2}, stop:1 {lighter3});
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }}
        #primaryButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {lighter1}, stop:0.3 {lighter2}, stop:0.7 {lighter3}, stop:1 {primary_hex});
        }}
        #primaryButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {darker1}, stop:1 {darker2});
        }}

        /* 滑块（groove 边框与 handle 主色） */
        #modernSlider::groove:horizontal {{
            border: 2px solid rgba({r}, {g}, {b}, 0.3);
        }}
        #modernSlider::handle:horizontal {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {primary_hex}, stop:0.5 {lighter1}, stop:1 {lighter2});
            border: 2px solid {darker1};
            width: 22px;
            margin: -6px 0;
            border-radius: 11px;
        }}
        #modernSlider::handle:horizontal:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {lighter1}, stop:0.5 {lighter2}, stop:1 {lighter3});
        }}

        /* 滑块数值徽标 */
        #sliderValue {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {primary_hex}, stop:1 {darker1});
            color: white;
        }}

        /* 复选框强调色 */
        #modernCheckbox::indicator {{
            border: 2px solid rgba({r}, {g}, {b}, 0.5);
        }}
        #modernCheckbox::indicator:checked {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {primary_hex}, stop:1 {darker1});
            border: 2px solid {darker1};
        }}
        #modernCheckbox::indicator:checked:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {lighter1}, stop:1 {primary_hex});
        }}
        """

        # 移除上一次的自定义覆盖，避免叠加越来越长
        current = self.styleSheet() or ""
        prev_overlay = getattr(self, "_custom_theme_style", None)
        if prev_overlay:
            current = current.replace(prev_overlay, "")

        # 保存并应用新的覆盖
        self._custom_theme_style = style
        self.setStyleSheet(current + "\n" + style)

    def open_video_file(self):
        """
        打开视频文件
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )

        if file_path:
            self.current_video_path = file_path
            self.video_processor.set_video_source(file_path)
            self.start_btn.setEnabled(True)
            filename = os.path.basename(file_path)
            self.video_label.setText(f"📁 已选择视频文件\n{filename}")
            self.add_log(f"✅ 成功加载视频文件: {filename}")
            self.add_log(f"📍 文件路径: {file_path}")

    def open_camera(self):
        """
        打开摄像头
        """
        self.current_video_path = None
        self.video_processor.set_video_source(0)  # 默认摄像头
        self.start_btn.setEnabled(True)
        self.video_label.setText("📷 已选择摄像头\n准备实时检测")
        self.add_log("📷 摄像头已准备就绪")
        self.add_log("⚡ 可以开始实时检测")

    def on_frame_ready(self):
        """
        当新帧准备好时的回调（异步处理）
        """
        self.pending_frame_update = True

    def update_display(self):
        """
        定时更新显示（独立于视频处理线程）
        """
        if self.pending_frame_update:
            # 从视频处理器获取最新帧
            latest_frame = self.video_processor.get_latest_frame()
            if latest_frame is not None:
                self.update_video_display(latest_frame)
                self.pending_frame_update = False

                # 更新UI帧率统计
                current_time = time.time()
                if self.last_display_time > 0:
                    self.display_frame_count += 1
                    time_diff = current_time - self.last_display_time
                    if time_diff >= 1.0:  # 每秒更新一次UI帧率
                        self.ui_fps = self.display_frame_count / time_diff
                        self.display_frame_count = 0
                        self.last_display_time = current_time
                else:
                    self.last_display_time = current_time

    def start_detection(self):
        """
        开始检测
        """
        if self.video_processor.start_processing():
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.open_file_btn.setEnabled(False)
            self.open_camera_btn.setEnabled(False)
            # 禁用车道线检测开关按钮
            if hasattr(self, 'lane_detection_btn'):
                self.lane_detection_btn.setEnabled(False)

            # 启动UI显示定时器
            self.display_timer.start(self.display_interval)

            self.add_log("🚀 检测系统已启动")
            self.add_log("🔍 开始进行车道线和车辆检测...")
            self.add_log(f"⚡ 性能优化: 检测间隔{self.video_processor.detection_interval}帧, 目标帧率{self.video_processor.target_fps}FPS")
            self.add_log(f"🖥️ UI显示帧率: {self.display_fps}FPS")
            self.add_log("💡 提示: 如遇卡顿可调整性能参数")
        else:
            self.add_log("❌ 检测启动失败，请检查视频源")


    def stop_detection(self):
        """
        停止检测
        """
        # 停止UI显示定时器
        self.display_timer.stop()

        self.video_processor.stop_processing()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.open_file_btn.setEnabled(True)
        self.open_camera_btn.setEnabled(True)
        # 重新启用车道线检测开关按钮
        if hasattr(self, 'lane_detection_btn'):
            self.lane_detection_btn.setEnabled(True)

        # 重置显示状态
        self.pending_frame_update = False
        self.last_frame_data = None

        self.add_log("⏹️ 检测系统已停止")
        self.add_log("💤 系统进入待机状态")


    def update_safe_distance(self, value):
        """
        更新安全距离设置

        Args:
            value (int): 滑块值
        """
        self.safe_distance_label.setText(str(value))
        # 更新配置
        self.video_processor.config.SAFE_DISTANCE = value


    def update_confidence(self, value):
        """
        更新检测置信度设置

        Args:
            value (int): 滑块值
        """
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.1f}")
        # 更新配置
        self.video_processor.vehicle_detector.set_confidence_threshold(confidence)

    def update_detection_interval(self, value):
        """
        更新检测间隔设置

        Args:
            value (int): 滑块值
        """
        self.detection_interval_label.setText(str(value))
        # 更新配置
        self.video_processor.detection_interval = value
        self.add_log(f"🔧 检测间隔已调整为每{value}帧检测一次")

    def update_target_fps(self, value):
        """
        更新目标帧率设置

        Args:
            value (int): 滑块值
        """
        self.target_fps_label.setText(str(value))
        # 更新配置
        self.video_processor.target_fps = value
        self.video_processor.frame_time = 1.0 / value
        self.add_log(f"🎯 目标帧率已调整为{value}FPS")

    def update_tech_visual(self, state):
        """
        更新科技感可视化开关

        Args:
            state (int): 复选框状态 (0=未选中, 2=选中)
        """
        enabled = state == 2  # Qt.Checked = 2
        self.video_processor.tech_visual_enabled = enabled
        
        if enabled:
            self.add_log("🎨 科技感HUD效果已启用")
        else:
            self.add_log("🎨 科技感HUD效果已禁用")


    def update_video_display(self, frame):
        """
        更新视频显示
        优化的视频显示更新
        Args:
            frame (np.ndarray): 视频帧
        """

        try:
            # 帧缓存优化：避免重复处理相同帧
            frame_hash = hash(frame.tobytes())
            if hasattr(self, '_last_frame_hash') and self._last_frame_hash == frame_hash:
                return
            self._last_frame_hash = frame_hash

            # 获取显示区域尺寸
            label_size = self.video_label.size()
            if label_size.width() <= 0 or label_size.height() <= 0:
                return

            target_width = min(label_size.width(), 1280)  # 限制最大宽度
            target_height = min(label_size.height(), 720)  # 限制最大高度

            # 智能缩放：只在必要时进行缩放
            height, width = frame.shape[:2]
            if width > target_width or height > target_height:
                # 计算缩放比例
                scale_w = target_width / width
                scale_h = target_height / height
                scale = min(scale_w, scale_h)

                new_width = int(width * scale)
                new_height = int(height * scale)

                # 使用更快的插值方法
                frame = cv2.resize(frame, (new_width, new_height),interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

            # 优化的Qt图像转换
            height, width, channel = frame.shape
            bytes_per_line = 3 * width

            # 确保数据连续性
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)

            # 创建QImage（避免数据拷贝）
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            q_image = q_image.rgbSwapped()

            # 创建Pixmap并设置
            pixmap = QPixmap.fromImage(q_image)

            # 只在尺寸不匹配时才进行最终缩放
            if pixmap.size() != label_size:
                pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.FastTransformation)

            self.video_label.setPixmap(pixmap)

        except Exception as e:
            # 错误处理：避免因单帧错误导致整个系统崩溃
            print(f"视频显示更新错误: {e}")
            pass

    def update_warning_status(self, warning_level, message):
        """
        更新预警状态

        Args:
            warning_level (str): 预警等级 ('safe', 'warning', 'danger')
            message (str): 预警消息
        """
        if warning_level == "danger":
            # 红色危险状态
            self.warning_label.setText(f"🚨 {message}")
            self.warning_label.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e74c3c, stop:1 #c0392b); "
                "color: white; padding: 15px; border-radius: 8px; font-size: 16px; font-weight: bold;"
            )
            self.add_log(f"🚨 危险预警: {message}")
        elif warning_level == "warning":
            # 黄色预警状态
            self.warning_label.setText(f"⚠️ {message}")
            self.warning_label.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f39c12, stop:1 #e67e22); "
                "color: white; padding: 15px; border-radius: 8px; font-size: 16px; font-weight: bold;"
            )
            self.add_log(f"⚠️ 黄色提醒: {message}")
        else:
            # 绿色安全状态
            self.warning_label.setText(f"✅ {message}")
            self.warning_label.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #27ae60, stop:1 #229954); "
                "color: white; padding: 15px; border-radius: 8px; font-size: 16px; font-weight: bold;"
            )


    def show_first_frame(self, frame):
        """
        显示视频第一帧

        Args:
            frame (np.ndarray): 第一帧图像
        """
        try:
            # 检查frame是否有效
            if frame is None or frame.size == 0:
                self.add_log("❌ 第一帧数据无效")
                return
            
            # 确保frame是3通道BGR图像
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.add_log(f"❌ 图像格式错误: {frame.shape}")
                return
            
            # 确保数据连续性
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # 转换为Qt图像格式
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            
            # 创建QImage并转换颜色格式（BGR -> RGB）
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            q_image = q_image.rgbSwapped()  # BGR转RGB

            # 缩放图像以适应标签大小
            pixmap = QPixmap.fromImage(q_image)
            
            # 获取标签尺寸并进行缩放
            label_size = self.video_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)
            else:
                self.video_label.setPixmap(pixmap)

            if self.current_video_path:
                filename = os.path.basename(self.current_video_path)
                self.add_log(f"🖼️ 已显示视频第一帧: {filename}")
            else:
                self.add_log("🖼️ 已显示视频第一帧")
                
        except Exception as e:
            self.add_log(f"❌ 显示第一帧失败: {str(e)}")
            print(f"显示第一帧错误: {e}")


    def update_stats_display(self, stats):
        """
        更新统计信息显示

        Args:
            stats (dict): 统计信息字典
        """
        fps = stats.get('fps', 0)
        vehicle_count = stats.get('vehicle_count', 0)
        frame_count = stats.get('frame_count', 0)
        run_time = stats.get('run_time', 0)

        self.fps_label.setText(f"{fps:.1f} FPS")
        self.vehicle_count_label.setText(f"{vehicle_count} 辆")
        # self.frame_count_label.setText(str(frame_count))  # 已屏蔽处理帧数显示
        self.run_time_label.setText(f"{run_time // 60}分{run_time % 60}秒")

        # 更新UI帧率显示
        if hasattr(self, 'ui_fps_label'):
            self.ui_fps_label.setText(f"{self.ui_fps:.1f} FPS")

        # 更新车道线状态（这里可以根据实际检测结果更新）
        if frame_count > 0:
            self.lane_status_label.setText("检测中")


    def add_log(self, message):
        """
        添加日志信息

        Args:
            message (str): 日志消息
        """
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)

        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


    def closeEvent(self, event):
        """
        窗口关闭事件处理

        Args:
            event: 关闭事件
        """
        # 停止所有定时器
        if hasattr(self, 'display_timer'):
            self.display_timer.stop()

        # 停止视频处理
        self.video_processor.stop_processing()

        # 等待线程结束
        if self.video_processor.isRunning():
            self.video_processor.wait(3000)  # 等待最多3秒

        event.accept()


def main():
    """
    主函数
    """
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 创建主窗口
    window = MainWindow()
    window.show()

    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()