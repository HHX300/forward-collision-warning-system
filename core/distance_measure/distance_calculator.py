#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
距离计算模块

功能描述:
- 基于车辆边界框计算距离
- 使用透视变换和几何关系估算距离
- 支持摄像头标定参数配置
- 提供多种距离计算方法

作者: HXH Assistant
创建时间: 2025
"""

import numpy as np
import math
from collections import namedtuple



# 参数类
class CalibParameter:
    def __init__(self, imageWidth, imageHeight, xa, ya, xb, yb, xc, yc, xd, yd, W, L):
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.xa = xa
        self.ya = ya
        self.xb = xb
        self.yb = yb
        self.xc = xc
        self.yc = yc
        self.xd = xd
        self.yd = yd
        self.W = W
        self.L = L


# 实现类
class SpeedMeasure:
    def __init__(self):
        self.calibrated = False
        self.s = 0  # Angle in radians
        self.t = 0  # Another angle parameter
        self.f = 0  # Focal length or scaling factor
        self.l = 0  # Length of the object in the scene
        self.h = 0  # Height of the object
        self.p = 0  # Perspective angle
        # 定义类似 CvPoint2D32f 的结构
        self.Point2D = namedtuple("Point2D", ["x", "y"])

    def calibration(self, calibP):
        # Center normalized coordinates
        xaf = calibP.xa - calibP.imageWidth / 2
        xbf = calibP.xb - calibP.imageWidth / 2
        xcf = calibP.xc - calibP.imageWidth / 2
        xdf = calibP.xd - calibP.imageWidth / 2
        yaf = calibP.imageHeight / 2 - calibP.ya
        ybf = calibP.imageHeight / 2 - calibP.yb
        ycf = calibP.imageHeight / 2 - calibP.yc
        ydf = calibP.imageHeight / 2 - calibP.yd

        calib_succeed = self.calibTwoVanish(xbf, ybf, xdf, ydf, xaf, yaf, xcf, ycf, calibP.L)
        self.p += 90

        if not calib_succeed:  # if p is not a number, must use one vanishing point method
            calib_succeed = self.calibOneVanish(xaf, yaf, xbf, ybf, xcf, ycf, xdf, ydf, calibP.W, calibP.L)
            if not calib_succeed:
                calib_succeed = self.calibOneVanish(xbf, ybf, xdf, ydf, xaf, yaf, xcf, ycf, calibP.L, calibP.W)
                self.p += 90
        else:
            if -360 < self.p < 360 and (int(self.p + 0.5) + 360 - 30) % 90 >= 30:  # (90 > p >= 60) or 180 > p >= 150)
                if (int(self.p + 0.5) + 360 + 45) % 180 >= 90:  # 135 > p > 45
                    self.calibOneVanish(xaf, yaf, xbf, ybf, xcf, ycf, xdf, ydf, calibP.W, calibP.L)
                else:
                    if self.calibOneVanish(xbf, ybf, xdf, ydf, xaf, yaf, xcf, ycf, calibP.L, calibP.W):
                        self.p += 90

        self.p = (int(self.p) + 179) % 360 - 179 + self.p - int(self.p)
        self.p = self.p * math.pi / 180
        self.calibrated = calib_succeed

    def calibOneVanish(self, xaf, yaf, xbf, ybf, xcf, ycf, xdf, ydf, width, length):
        # 专利算法，暂不开源
        return 1

    def calibTwoVanish(self, xaf, yaf, xbf, ybf, xcf, ycf, xdf, ydf, width):
        # 专利算法，暂不开源
        return 1

    def image2realXY(self, XImage, YImage):
        """
        将图像坐标 (XImage, YImage) 转换为真实世界坐标 (XReal, YReal)。
        :param XImage: 图像坐标的 x 值。
        :param YImage: 图像坐标的 y 值。
        :return: 转换后的真实世界坐标 (XReal, YReal)。
        """
        # 分母计算：与 XReal 和 YReal 的公式相同
        denominator = (
                XImage * math.cos(self.t) * math.sin(self.s) +
                YImage * math.cos(self.t) * math.cos(self.s) +
                self.f * math.sin(self.t)
        )

        if denominator == 0:
            raise ValueError("分母为零，无法计算真实坐标。请检查参数。")

        # 计算 XReal（真实世界 x 坐标）
        XReal = (
                math.sin(self.p) * self.l * (XImage * math.sin(self.s) + YImage * math.cos(self.s)) +
                math.cos(self.p) * self.l * math.sin(self.t) * (XImage * math.cos(self.s) - YImage * math.sin(self.s))
                ) / denominator

        # 计算 YReal（真实世界 y 坐标）
        YReal = (
                -math.cos(self.p) * self.l * (XImage * math.sin(self.s) + YImage * math.cos(self.s)) +
                math.sin(self.p) * self.l * math.sin(self.t) * (XImage * math.cos(self.s) - YImage * math.sin(self.s))
                ) / denominator

        return XReal, YReal


    def get_cam_xy(self):
        """
        计算相机在地平面 XY 上的投影位置

        参数:
        - h: 相机高度 (单位: 米)
        - p: 平移角（pan angle，单位: 弧度）
        - t: 倾斜角（tilt angle，单位: 弧度）

        返回:
        - Point2D(x, y): 相机在 XY 地面坐标系中的位置
        """
        x = -self.h * np.sin(self.p) * np.cos(self.t) / np.sin(self.t)
        y = self.h * np.cos(self.p) * np.cos(self.t) / np.sin(self.t)
        return self.Point2D(x, y)


# 应用类
class Use:
    def __init__(self):
        self.speed_measure = SpeedMeasure()
        self.calibP = CalibParameter(
            imageWidth=1920, # 输入图像的宽度
            imageHeight=1080, # 输入图像的高度
            xa=946, ya=697, # 左上角
            xb=1005, yb=838, # 左下角
            xc=1372, yc=679, # 右上角
            xd=1617, yd=772, # 右下角
            W=3.25, # 实际宽度
            L=2 # 实际长度
        )

    def run(self, point):
        # 1.标定得到标定参数
        self.speed_measure.calibration(self.calibP)
        # 2.传入坐标得到实际距离
        point_x = point[0]
        point_y = point[1]
        # 预处理，已省略

        XReal, YReal = self.speed_measure.image2realXY(point_x, point_y)
        return XReal, YReal



class DistanceCalculator:
    """
    距离计算类
    使用计算机视觉技术估算车辆的距离
    1. 首先检测到地面的矩形框
    2. 得到矩形框之后，传入数据进行标定
    """

    def __init__(self):
        # 1.初始化车道线检测模型
        self.detect_model = None
        # 2.预标定参数
        self.calibP = CalibParameter(
                    imageWidth=1920,
                    imageHeight=1080,
                    xa=856, ya=919,
                    xb=747, yb=997,
                    xc=1087, yc=917,
                    xd=1213, yd=992,
                    W=3.25,
                    L=4.5
                )
        # 3.初始化速度测量类
        self.speed_measure = SpeedMeasure()

        # 4.初始化未标定（是否标定成功）
        self.is_calibTrue = False

    def detect_lane_line(self, image):
        """
        通过传入的图像使用模型检测到车道线，返回需要的值
        self.calibP = CalibParameter(
                    imageWidth=1920,
                    imageHeight=1080,
                    xa=946, ya=697,
                    xb=1005, yb=838,
                    xc=1372, yc=679,
                    xd=1617, yd=772,
                    W=3.25,
                    L=3.5
                )
        说明 -->
        a(946,697):左上角,
        b(1005,838):左下角,
        c(1372, 679):右上角,
        d(1617, 772):右下角
        imageWidth:图像宽度
        imageHeight:图像高度
        W=3.25:矩形框宽度
        L=4.5:矩形框长度
        """
        # 获取图像的宽高 1920 * 1080
        img_w, img_h = image.shape
        # 车道线检测
        result = self.detect_model(image)
        a, b = result
        result = self.detect_model(image)
        c, d = result

        if a and b and c and d:
            # 长和宽根据国家标准已知
            w = 3.25
            l = 3.5
            self.calibP = CalibParameter(
                        imageWidth=img_w,
                        imageHeight=img_h,
                        xa=a[0], ya=a[1],
                        xb=b[0], yb=b[1],
                        xc=c[0], yc=c[1],
                        xd=d[0], yd=d[1],
                        W=w,
                        L=l
                    )
            self.is_calibTrue = True
            print("检测到车道线，标定成功！")
            pass
        else:
            print("检测到车道线，尚未标定成功！")
            pass


    def cal_distance_a2b(self, point_a, point_b):
        # 1.标定得到标定参数
        self.speed_measure.calibration(self.calibP)

        # 2.坐标预处理(相较于中心点的偏移)
        pointA_x, pointA_y = point_a[0], point_a[1]
        A_xReal, A_yReal = self.speed_measure.image2realXY(pointA_x, pointA_y)

        pointB_x, pointB_y = point_b[0], point_b[1]
        B_xReal, B_yReal = self.speed_measure.image2realXY(pointB_x, pointB_y)

        # 计算出x方向和y方向的距离
        x_distance = abs(A_xReal - B_xReal) # 6.292835816507722，单位m
        y_distance = abs(A_yReal - B_yReal) # 0.13401646258543254 单位m，y的距离有小范围的偏差是正常现象

        return x_distance, y_distance

    def cal_distance_a2b_no_calib(self, point_a, point_b):

        # 1.坐标预处理(相较于中心点的偏移)
        pointA_x, pointA_y = point_a[0], point_a[1]
        A_xReal, A_yReal = self.speed_measure.image2realXY(pointA_x, pointA_y)

        pointB_x, pointB_y = point_b[0], point_b[1]
        B_xReal, B_yReal = self.speed_measure.image2realXY(pointB_x, pointB_y)

        # 计算出x方向和y方向的距离
        x_distance = abs(A_xReal - B_xReal) # 6.292835816507722，单位m
        y_distance = abs(A_yReal - B_yReal) # 0.13401646258543254 单位m，y的距离有小范围的偏差是正常现象

        return x_distance, y_distance


    def cal_distance_a2cam(self, point_a):
        # 1.标定得到标定参数
        self.speed_measure.calibration(self.calibP)

        # 2.坐标预处理(相较于中心点的偏移)
        pointA_x, pointA_y = point_a[0], point_a[1]
        A_xReal, A_yReal = self.speed_measure.image2realXY(pointA_x, pointA_y)
        cam_xReal, cam_yReal = self.speed_measure.get_cam_xy()

        # 计算出x方向和y方向的距离
        x_distance = abs(A_xReal - cam_xReal) # 6.292835816507722，单位m
        y_distance = abs(A_yReal - cam_yReal) # 0.13401646258543254 单位m，y的距离有小范围的偏差是正常现象

        return x_distance, y_distance


    def calib_correction(self, rect_width, point_a, point_b):

        lower_car_width = 1.6
        upper_car_width = 1.85
        # 正常车辆的宽度范围：1.6 米到 1.85 米 之间
        if lower_car_width < rect_width < upper_car_width:
            return rect_width

        # 二分查找实现
        max_iter = 60 # 设置迭代上限，防止死循环
        left_deg = 0  # 调整角度
        right_deg = 45
        eps = 0.01 # 精度控制

        for i in range(max_iter):
            gap_deg = right_deg - left_deg
            if gap_deg <= eps:
                break

            mid_deg = (left_deg + right_deg) / 2.0
            t_rad = math.radians(mid_deg)  # 统一使用弧度
            s = math.sin(t_rad)

            # 更新系数（使用弧度）
            self.speed_measure.t = t_rad
            self.speed_measure.l = - self.speed_measure.h / s

            # 开始重新计算
            _, y_distance_ab = self.cal_distance_a2b_no_calib(point_a, point_b)
            rect_width = y_distance_ab  # 赋值给车宽

            # 正常车辆的宽度范围：1.6 米到 1.85 米 之间
            if lower_car_width < y_distance_ab < upper_car_width:
                return rect_width

            # 假设 y_distance_ab和mid_deg 是单调关系
            if y_distance_ab < 1.85:
                left_deg = mid_deg  # 太小，需要更大的角度
            else:
                right_deg = mid_deg  # 太大，需要更小的角度

        return rect_width

if __name__ == '__main__':
    distance_calculator = DistanceCalculator()
    a = (100, 200)
    b = (100, 200)
    distance_calculator.cal_distance_a2b(a, b)
    pass