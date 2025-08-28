#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆检测模块

功能描述:
- 使用YOLO模型进行车辆检测
- 提供车辆边界框坐标
- 支持实时视频处理
- 可配置检测置信度阈值

作者: HXH Assistant
创建时间: 2025
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
import torch
from pathlib import Path

from sympy.stats.sampling.sample_numpy import numpy
from ultralytics import YOLO
from core.utils import v5_inference
from core.detection.TensorRTPredictor import TensorRTPredictor # 导入包

device = "cuda" if torch.cuda.is_available() else 'cpu'

class VehicleDetector:
    """
    车辆检测器类
    使用YOLO模型检测图像中的车辆

    """

    def __init__(self, model_path: str = None):
        """
        初始化车辆检测器

        Args:
            model_path (str): YOLO模型配置文件路径
        """
        # 检测参数
        self.confidence_threshold = 0.5  # 置信度阈值

        # YOLO模型相关
        self.net = None
        self.classes = None
        self.model_type = None

        # 车辆类别ID（COCO数据集中的车辆相关类别）
        self.vehicle_classes = {
            2: 'motor',  # 摩托车
            3: 'car',  # 汽车
            4: 'tricycle', # 三轮车
            5: 'bus',  # 公交车
            6: 'minibus', # 迷你公交车
            7: 'truck',  # 卡车
        }

        # 使用分割模型，默认为False
        self.use_yolo_segment = False

        # 首先尝试加载YOLO模型
        self.load_yolo_model(model_path)

        # # 如果YOLO模型加载失败
        # if self.net is None:
        #     print("提示: YOLO检测模型未加载，将使用YOLO分割模型的方法")
        #     self.use_yolo_segment = True
        # else:
        #     self.use_yolo_segment = False



    def load_yolo_model(self, model_path: str = None):
        """
        加载YOLO模型

        Args:
            model_path (str): 模型文件路径
        """
        try:
            # 先判断模型的类型
            filepath = Path(model_path) # car_detector.engine
            ext = filepath.suffix[1:]  # 输出：engine
            self.model_type = ext  # 确定模型的类型

            if ext == 'pth' or ext == 'pt':
                # 判断是否为seg模型
                seg_type = filepath.stem.split('-')[-1] if '-' in filepath.stem else None
                if seg_type == 'seg':
                    # 加载YOLO分割模型的pt网络模型
                    # Load a pretrained YOLO11n-seg Segment model
                    model = YOLO(model_path)
                    self.net = model.eval().to(device) # 放在gpu上面
                    self.use_yolo_segment = True
                    print("YOLO分割模型加载成功！！\n")
                else:
                    # 加载YOLO的pt网络模型
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # 注意 source='local'
                    self.net = model.eval().to(device) # 放在gpu上面
                    print("YOLO检测模型加载成功！！\n")
            elif ext == 'engine' or ext == "trt" :
                # 加载YOLO的TensorRT模型
                trt_predictor = TensorRTPredictor(model_path)
                # 预热模型
                trt_predictor.warmup(10)
                self.net = trt_predictor
                print("YOLO引擎模型加载成功！！\n")
            else:
                raise ValueError

            # 加载类别名称
            cls_path = 'class.txt'
            self.load_class_names(cls_path)

        except Exception as e:
            print(f"YOLO模型加载失败: {e}\n")
            self.net = None

    def load_class_names(self, classes_path: str = None):
        """
        加载类别名称

        Args:
            classes_path (str): 类别文件路径
        """
        if classes_path and os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            # 使用COCO数据集的默认类别
            self.classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]


    def detect_vehicles_yolo(self, image: np.ndarray):
        """
        使用YOLO模型检测车辆

        Args:
            image (np.ndarray): 输入图像

        Returns:
            List[Tuple[int, int, int, int, float, int]]: 检测结果列表，每个元素为(x1, y1, x2, y2, confidence, class_id)
        """
        vehicles = []
        contours = []

        if self.net is None:
            return vehicles

        height, width = image.shape[:2]
        # torch推理
        if self.model_type == 'pt' or self.model_type == 'pth':
            detections = v5_inference.run_inference(image, self.net)
        # TensorRT推理
        elif self.model_type == 'engine' or self.model_type == 'trt':
            detections  = self.net.infer(image)
        else:
            detections = np.array([])

        for detection in detections:
            confidence = round(detection[4], 2) # 置信度
            class_id = round(detection[5]) # 车辆类别
            # 如果不属于车辆类别或者低于阈值，则跳过
            if class_id not in self.vehicle_classes or  confidence < self.confidence_threshold:
                continue
            # 只保留车辆类别且置信度高于阈值的检测
            x1 = round(detection[0])
            y1 = round(detection[1])
            x2 = round(detection[2])
            y2 = round(detection[3])
            vehicles.append([x1, y1, x2, y2, confidence, class_id])

        return vehicles, contours

    def detect_vehicles_yolo_seg(self, image: np.ndarray):
        """
        yolo分割模型

        Args:
            image (np.ndarray): 输入图像

        Returns:
           List[np.array([int, int, int, int, float, int])]: 检测结果列表
        """
        vehicles = []
        contours = []
        # 分割模型推理内容
        results = self.net.predict(image)
        for result in results:
            for c in result:
                vehicle = c.boxes.data.cpu().numpy()[0]
                confidence = round(vehicle[4], 2)  # 置信度
                class_id = round(vehicle[5])  # 车辆类别
                # 如果不属于车辆类别或者低于阈值，则跳过
                # if class_id not in self.vehicle_classes or confidence < self.confidence_threshold:
                #     continue
                # 只保留车辆类别且置信度高于阈值的检测
                x1 = round(vehicle[0])
                y1 = round(vehicle[1])
                x2 = round(vehicle[2])
                y2 = round(vehicle[3])

                vehicles.append([x1, y1, x2, y2, confidence, class_id])

                # 创建一个轮廓掩码
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                contours.append(contour)

        return vehicles, contours




    def detect_vehicles(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """
        检测车辆主函数

        Args:
            image (np.ndarray): 输入图像

        Returns:
            List[Tuple[int, int, int, int, float, int]]: 检测结果列表，每个元素为(x1, y1, x2, y2, confidence, class_id)
        """
        if self.use_yolo_segment:
            return self.detect_vehicles_yolo_seg(image)
        else:
            return self.detect_vehicles_yolo(image)


    def draw_detections(self, image: np.ndarray, vehicles: List[Tuple[int, int, int, int, float, int]], draw_labels: bool = True) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image (np.ndarray): 输入图像
            vehicles (List): 车辆检测结果
            draw_labels (bool): 是否绘制标签

        Returns:
            np.ndarray: 绘制检测框后的图像
        """
        result_image = image.copy()

        for vehicle in vehicles:
            x1, y1, x2, y2, confidence, class_id = vehicle

            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if draw_labels:
                # 获取车辆类别名称
                class_name = self.vehicle_classes.get(class_id)
                # 绘制类别和置信度标签
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                # 绘制标签背景
                cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),(x1 + label_size[0], y1), (0, 255, 0), -1)

                # 绘制标签文本
                cv2.putText(result_image, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return result_image


    def filter_vehicles_by_size(self, vehicles: List[Tuple[int, int, int, int, float, int]], min_area: int = 1000, max_area: int = 50000) -> List[Tuple[int, int, int, int, float, int]]:
        """
        根据大小过滤车辆检测结果

        Args:
            vehicles (List): 车辆检测结果
            min_area (int): 最小面积
            max_area (int): 最大面积

        Returns:
            List: 过滤后的车辆列表
        """
        filtered_vehicles = []

        for vehicle in vehicles:
            x1, y1, x2, y2, confidence, class_id = vehicle
            area = (x2 - x1) * (y2 - y1)

            if min_area <= area <= max_area:
                filtered_vehicles.append(vehicle)

        return filtered_vehicles


    def filter_vehicles_by_position(self, vehicles: List[Tuple[int, int, int, int, float, int]], image_height: int, roi_ratio: float = 0.6) -> List[Tuple[int, int, int, int, float, int]]:
        """
        根据位置过滤车辆检测结果（只保留图像下半部分的检测）

        Args:
            vehicles (List): 车辆检测结果
            image_height (int): 图像高度
            roi_ratio (float): 感兴趣区域比例

        Returns:
            List: 过滤后的车辆列表
        """
        filtered_vehicles = []
        roi_y_threshold = int(image_height * roi_ratio)

        for vehicle in vehicles:
            x1, y1, x2, y2, confidence, class_id = vehicle

            # 检查车辆中心是否在感兴趣区域内
            center_y = (y1 + y2) // 2
            if center_y > roi_y_threshold:
                filtered_vehicles.append(vehicle)

        return filtered_vehicles


    def get_vehicle_center(self, vehicle: Tuple[int, int, int, int, float, int]) -> Tuple[int, int]:
        """
        获取车辆边界框的中心点

        Args:
            vehicle (Tuple): 车辆检测结果

        Returns:
            Tuple[int, int]: 中心点坐标(x, y)
        """
        x1, y1, x2, y2, _, _ = vehicle
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return center_x, center_y


    def get_vehicle_bottom_center(self, vehicle):
        """
        获取车辆边界框底部中心点（用于距离计算）
        """
        x1, y1, x2, y2, _, _ = vehicle
        center_x = (x1 + x2) // 2
        bottom_y = y2
        return center_x, bottom_y


    def set_confidence_threshold(self, threshold: float):
        """
        设置置信度阈值

        Args:
            threshold (float): 置信度阈值 (0.0 - 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))



    def get_detection_statistics(self, vehicles: List[Tuple[int, int, int, int, float, int]]) -> dict:
        """
        获取检测统计信息

        Args:
            vehicles (List): 车辆检测结果

        Returns:
            dict: 统计信息字典
        """
        if not vehicles:
            return {
                'vehicle_count': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'vehicle_classes': {}
            }

        confidences = [vehicle[4] for vehicle in vehicles]
        
        # 统计各类别车辆数量
        vehicle_classes = {}
        for vehicle in vehicles:
            class_id = vehicle[5]
            class_name = self.vehicle_classes.get(class_id)
            vehicle_classes[class_name] = vehicle_classes.get(class_name, 0) + 1

        return {
            'vehicle_count': len(vehicles),
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'vehicle_classes': vehicle_classes
        }


# 测试代码
if __name__ == "__main__":
    # 创建车辆检测器实例
    model_file_path = "models/yolov5/car_detector.pt"
    detector = VehicleDetector(model_file_path)

    print("\n车辆检测模块已加载")
    print("使用方法:")
    print("1. 创建检测器实例: detector = VehicleDetector()")
    print("2. 检测车辆: vehicles = detector.detect_vehicles(image)")
    print("3. 绘制检测结果: result = detector.draw_detections(image, vehicles)")
