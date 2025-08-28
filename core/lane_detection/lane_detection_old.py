#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车道线检测模块

功能描述:
- 使用OpenCV进行车道线检测
- 实现霍夫变换检测直线
- 提供车道线坐标信息
- 支持实时视频处理

作者: HXH Assistant
创建时间: 2025
"""

import cv2
import numpy as np
import math


class LaneDetector:
    """
    车道线检测器类
    使用计算机视觉技术检测道路上的车道线
    """

    def __init__(self):
        """
        初始化车道线检测器
        """
        # 高斯模糊参数
        self.gaussian_blur_kernel = 5

        # Canny边缘检测参数
        self.canny_low_threshold = 1
        self.canny_high_threshold = 20

        # 霍夫变换参数
        self.hough_rho = 1  # 距离分辨率
        self.hough_theta = np.pi / 180  # 角度分辨率
        self.hough_threshold = 100  # 累加器阈值
        self.min_line_length = 5
        self.max_line_gap = 20  # 最大线段间隙

        # 感兴趣区域参数
        self.roi_vertices = None

    def preprocess_image(self, image):
        """
        图像预处理

        Args:
            image (np.ndarray): 输入图像

        Returns:
            np.ndarray: 预处理后的图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊，减少噪声
        blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 0)

        # Canny边缘检测
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)

        return edges

    def create_roi_mask(self, image):
        """
        创建感兴趣区域掩码

        Args:
            image (np.ndarray): 输入图像

        Returns:
            np.ndarray: ROI掩码
        """
        height, width = image.shape[:2]

        # 定义感兴趣区域的顶点（梯形区域）
        if self.roi_vertices is None:
            self.roi_vertices = np.array([
                [(int(width * 0.0), height),  # 左下角
                 (int(width * 0.0), int(height * 0.8)),  # 左上角
                 (int(width * 1.0), int(height * 0.8)),  # 右上角
                 (int(width * 1.0), height)]  # 右下角
            ], dtype=np.int32)

        # 创建掩码
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [self.roi_vertices], 255)

        return mask

    def apply_roi(self, image):
        """
        应用感兴趣区域

        Args:
            image (np.ndarray): 输入图像

        Returns:
            np.ndarray: 应用ROI后的图像
        """
        mask = self.create_roi_mask(image)
        masked_image = cv2.bitwise_and(image, mask)
        masked_image = cv2.resize(masked_image, (int(image.shape[1] / 3), int(image.shape[0] / 3)))
        cv2.imshow("mask_image", masked_image)
        cv2.waitKey(0)
        return masked_image

    def detect_lines_hough(self, image):
        """
        使用霍夫变换检测直线

        Args:
            image (np.ndarray): 边缘检测后的图像

        Returns:
            list: 检测到的线段列表
        """
        lines = cv2.HoughLinesP(
            image,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        # 绘制线段
        draw_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]  # line 是形如 [[x1, y1, x2, y2]]
                cv2.line(draw_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        draw_image = cv2.resize(draw_image, (640, 360))
        cv2.imshow('draw_image', draw_image)
        cv2.waitKey(0)

        return lines

    def classify_lines(self, lines, image_width):
        """
        将检测到的线段分类为左车道线和右车道线

        Args:
            lines (list): 检测到的线段
            image_width (int): 图像宽度

        Returns:
            tuple: (左车道线列表, 右车道线列表)
        """
        if lines is None:
            return [], []

        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算斜率
            if x2 - x1 == 0:  # 避免除零错误
                continue

            slope = (y2 - y1) / (x2 - x1)

            # 根据斜率和位置分类线段
            # 左车道线：负斜率，位于图像左半部分
            # 右车道线：正斜率，位于图像右半部分
            if slope < -0.5 and x1 < image_width / 2 and x2 < image_width / 2:
                left_lines.append(line[0])
            elif slope > 0.5 and x1 > image_width / 2 and x2 > image_width / 2:
                right_lines.append(line[0])

        return left_lines, right_lines

    def fit_lane_line(self, lines):
        """
        拟合车道线

        Args:
            lines (list): 线段列表

        Returns:
            tuple: (斜率, 截距) 或 None
        """
        if not lines:
            return None

        # 提取所有点
        points = []
        for line in lines:
            x1, y1, x2, y2 = line
            points.extend([(x1, y1), (x2, y2)])

        if len(points) < 2:
            return None

        # 使用最小二乘法拟合直线
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        # 计算拟合直线的参数
        coefficients = np.polyfit(x_coords, y_coords, 1)
        slope = coefficients[0]
        intercept = coefficients[1]

        return slope, intercept

    def calculate_lane_points(self, slope, intercept, image_height, y_start_ratio=0.6):
        """
        计算车道线在图像中的起始和结束点

        Args:
            slope (float): 直线斜率
            intercept (float): 直线截距
            image_height (int): 图像高度
            y_start_ratio (float): 起始y坐标比例

        Returns:
            tuple: ((x1, y1), (x2, y2))
        """
        if slope == 0:  # 避免除零错误
            return None

        # 计算y坐标
        y1 = int(image_height * y_start_ratio)  # 车道线起始点
        y2 = image_height  # 车道线结束点（图像底部）

        # 根据直线方程计算x坐标: y = slope * x + intercept => x = (y - intercept) / slope
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return (x1, y1), (x2, y2)

    def detect_lanes(self, image):
        """
        检测车道线主函数

        Args:
            image (np.ndarray): 输入图像

        Returns:
            dict: 车道线信息字典，包含左右车道线坐标
        """
        # 图像预处理
        processed_image = self.preprocess_image(image)

        # 应用感兴趣区域
        roi_image = self.apply_roi(processed_image)

        # roi_image = cv2.resize(roi_image, (320,180))
        # cv2.imshow("roi_image", roi_image)
        # cv2.waitKey(0)

        # 检测直线
        lines = self.detect_lines_hough(roi_image)

        if lines is None:
            return None

        # 分类线段
        left_lines, right_lines = self.classify_lines(lines, image.shape[1])

        if left_lines is not None:
            for line in left_lines:
                x1, y1, x2, y2 = line  # line 是形如 [[x1, y1, x2, y2]]
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            print(len(left_lines))
        if right_lines is not None:
            for line in right_lines:
                x1, y1, x2, y2 = line  # line 是形如 [[x1, y1, x2, y2]]
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print(len(right_lines))
        image = cv2.resize(image, (int(image.shape[1] / 3), int(image.shape[0] / 3)))
        cv2.imshow("image", image)
        cv2.waitKey(0)
        # 拟合车道线
        left_lane = self.fit_lane_line(left_lines)
        right_lane = self.fit_lane_line(right_lines)

        lane_info = {}

        # 计算左车道线坐标
        if left_lane:
            slope, intercept = left_lane
            points = self.calculate_lane_points(slope, intercept, image.shape[0])
            if points:
                lane_info['left_lane'] = {
                    'points': points,
                    'slope': slope,
                    'intercept': intercept
                }

        # 计算右车道线坐标
        if right_lane:
            slope, intercept = right_lane
            points = self.calculate_lane_points(slope, intercept, image.shape[0])
            if points:
                lane_info['right_lane'] = {
                    'points': points,
                    'slope': slope,
                    'intercept': intercept
                }

        return lane_info if lane_info else None

    def draw_lanes(self, image, lane_info, line_thickness=8):
        """
        在图像上绘制车道线

        Args:
            image (np.ndarray): 输入图像
            lane_info (dict): 车道线信息
            line_thickness (int): 线条粗细

        Returns:
            np.ndarray: 绘制车道线后的图像
        """
        if lane_info is None:
            return image

        result_image = image.copy()

        # 绘制左车道线
        if 'left_lane' in lane_info:
            points = lane_info['left_lane']['points']
            (x1, y1), (x2, y2) = points
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 255), line_thickness)  # 黄色

        # 绘制右车道线
        if 'right_lane' in lane_info:
            points = lane_info['right_lane']['points']
            (x1, y1), (x2, y2) = points
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 255), line_thickness)  # 黄色

        # 绘制车道区域
        if 'left_lane' in lane_info and 'right_lane' in lane_info:
            self.draw_lane_area(result_image, lane_info)

        return result_image

    def draw_lane_area(self, image, lane_info, alpha=0.3):
        """
        绘制车道区域

        Args:
            image (np.ndarray): 输入图像
            lane_info (dict): 车道线信息
            alpha (float): 透明度
        """
        if 'left_lane' not in lane_info or 'right_lane' not in lane_info:
            return

        # 获取车道线端点
        left_points = lane_info['left_lane']['points']
        right_points = lane_info['right_lane']['points']

        # 创建车道区域的四个顶点
        lane_vertices = np.array([
            left_points[0],  # 左上
            left_points[1],  # 左下
            right_points[1],  # 右下
            right_points[0]  # 右上
        ], dtype=np.int32)

        # 创建覆盖层
        overlay = image.copy()
        cv2.fillPoly(overlay, [lane_vertices], (0, 255, 0))  # 绿色填充

        # 混合原图像和覆盖层
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # def get_lane_center(self, lane_info, y_position):
    #     """
    #     获取指定y坐标处的车道中心点
    #
    #     Args:
    #         lane_info (dict): 车道线信息
    #         y_position (int): y坐标
    #
    #     Returns:
    #         int: 车道中心的x坐标，如果无法计算则返回None
    #     """
    #     if not lane_info or 'left_lane' not in lane_info or 'right_lane' not in lane_info:
    #         return None
    #
    #     # 计算左车道线在指定y位置的x坐标
    #     left_slope = lane_info['left_lane']['slope']
    #     left_intercept = lane_info['left_lane']['intercept']
    #     left_x = int((y_position - left_intercept) / left_slope)
    #
    #     # 计算右车道线在指定y位置的x坐标
    #     right_slope = lane_info['right_lane']['slope']
    #     right_intercept = lane_info['right_lane']['intercept']
    #     right_x = int((y_position - right_intercept) / right_slope)
    #
    #     # 计算中心点
    #     center_x = (left_x + right_x) // 2
    #
    #     return center_x
    #
    # def calculate_lane_width(self, lane_info, y_position):
    #     """
    #     计算指定y坐标处的车道宽度
    #
    #     Args:
    #         lane_info (dict): 车道线信息
    #         y_position (int): y坐标
    #
    #     Returns:
    #         int: 车道宽度（像素），如果无法计算则返回None
    #     """
    #     if not lane_info or 'left_lane' not in lane_info or 'right_lane' not in lane_info:
    #         return None
    #
    #     # 计算左车道线在指定y位置的x坐标
    #     left_slope = lane_info['left_lane']['slope']
    #     left_intercept = lane_info['left_lane']['intercept']
    #     left_x = int((y_position - left_intercept) / left_slope)
    #
    #     # 计算右车道线在指定y位置的x坐标
    #     right_slope = lane_info['right_lane']['slope']
    #     right_intercept = lane_info['right_lane']['intercept']
    #     right_x = int((y_position - right_intercept) / right_slope)
    #
    #     # 计算宽度
    #     width = abs(right_x - left_x)
    #
    #     return width
    #

    # def set_detection_parameters(self, **kwargs):
    #     """
    #     设置检测参数
    #
    #     Args:
    #         **kwargs: 参数字典
    #             - gaussian_blur_kernel: 高斯模糊核大小
    #             - canny_low_threshold: Canny低阈值
    #             - canny_high_threshold: Canny高阈值
    #             - hough_threshold: 霍夫变换阈值
    #             - min_line_length: 最小线段长度
    #             - max_line_gap: 最大线段间隙
    #     """
    #     if 'gaussian_blur_kernel' in kwargs:
    #         self.gaussian_blur_kernel = kwargs['gaussian_blur_kernel']
    #     if 'canny_low_threshold' in kwargs:
    #         self.canny_low_threshold = kwargs['canny_low_threshold']
    #     if 'canny_high_threshold' in kwargs:
    #         self.canny_high_threshold = kwargs['canny_high_threshold']
    #     if 'hough_threshold' in kwargs:
    #         self.hough_threshold = kwargs['hough_threshold']
    #     if 'min_line_length' in kwargs:
    #         self.min_line_length = kwargs['min_line_length']
    #     if 'max_line_gap' in kwargs:
    #         self.max_line_gap = kwargs['max_line_gap']


# 测试代码
if __name__ == "__main__":
    # 创建车道线检测器实例
    image = cv2.imread("./test_frames/frame_1790.jpg")
    detector = LaneDetector()
    lane_info = detector.detect_lanes(image)
    result = detector.draw_lanes(image, lane_info)
    # result = cv2.resize(result, (640,360))
    # result = cv2.resize(result, (320, 180))
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # 如果有测试图像，可以在这里进行测试
    # print("车道线检测模块已加载")
    # print("使用方法:")
    # print("1. 创建检测器实例: detector = LaneDetector()")
    # print("2. 检测车道线: lane_info = detector.detect_lanes(image)")
    # print("3. 绘制车道线: result = detector.draw_lanes(image, lane_info)")