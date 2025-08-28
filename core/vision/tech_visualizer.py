#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科技感可视化模块

功能描述:
- 提供科技感的HUD界面效果
- 动态雷达扫描效果
- 车辆锁定和预警可视化
- 能量条和距离显示
- 与现有检测系统无缝集成

作者: HXH Assistant
创建时间: 2025
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import colorsys
import math
import os
from typing import List, Dict, Tuple, Optional


class TechHUDVisualizer:
    """
    科技感HUD可视化器
    提供未来科技风格的车辆检测和预警界面
    """
    
    def __init__(self, icon_path=None):
        """
        初始化科技感可视化器
        
        Args:
            radar_radius (int): 雷达扫描半径
            icon_path (str): 图标路径（可选）
        """
        self.icon = None
        if icon_path and os.path.exists(icon_path):
            self.icon = Image.open(icon_path).convert("RGBA").resize((24, 24))
    
        # 预计算常用的数学值，避免重复计算
        self._sin_cache = {}
        self._cos_cache = {}
        
        # 预计算常用颜色值，避免重复计算
        self._color_cache = {}
        
        # 加载字体
        font_path = "font/STKAITI.TTF"
        try:
            self._font_cache = ImageFont.truetype(font_path, 20)  # 默认字体大小
        except (OSError, IOError):
            # 如果字体文件不存在，使用默认字体
            self._font_cache = ImageFont.load_default()
        
        # 可视化配置
        self.enable_radar = True
        self.enable_lock_target = True
        self.enable_energy_bar = True
        self.enable_tech_lines = True
        self.enable_floating_ui = True
        
        # 颜色配置
        self.safe_color = (0, 255, 0)      # 安全：绿色
        self.warning_color = (0, 255, 255)  # 警告：黄色
        self.danger_color = (0, 0, 255)     # 危险：红色
        self.tech_color = (0, 255, 255)     # 科技感：青色

    def _get_cached_sin(self, x):
        """缓存sin计算结果"""
        key = round(x, 2)
        if key not in self._sin_cache:
            self._sin_cache[key] = math.sin(x)
        return self._sin_cache[key]

    def _get_cached_cos(self, x):
        """缓存cos计算结果"""
        key = round(x, 2)
        if key not in self._cos_cache:
            self._cos_cache[key] = math.cos(x)
        return self._cos_cache[key]

    def _get_dynamic_color(self, frame_id, base_color=(0, 255, 255), min_v=0.85, max_v=1.0):
        """
        根据当前帧编号动态生成颜色，实现完美的呼吸效果
        
        Args:
            frame_id (int): 当前帧编号
            base_color (tuple): 基础颜色RGB
            min_v (float): 最小亮度值（提高以避免颜色太虚）
            max_v (float): 最大亮度值
            
        Returns:
            tuple: 动态变化的RGB颜色
        """
        # 使用缓存避免重复计算
        cache_key = (frame_id // 3, base_color, min_v, max_v)  # 调整缓存精度
        if cache_key in self._color_cache:
            return self._color_cache[cache_key]

        # 呼吸效果参数优化
        breath_speed = 6.0  # 呼吸速度，数值越小呼吸越慢

        # 改进的呼吸曲线：使用abs(sin)确保从frame_id=0开始就有良好的亮度
        sin_val = self._get_cached_sin(frame_id / breath_speed)
        # 使用绝对值确保始终为正，再进行平滑处理
        breath_factor = abs(sin_val) * 0.7 + 0.3  # 0.3-1.0范围，确保最低亮度

        # 计算动态亮度值
        v = min_v + (max_v - min_v) * breath_factor

        # 增强饱和度增强，避免颜色太虚
        saturation_boost = 1.25  # 提高饱和度增强系数

        # 针对不同颜色优化处理
        if base_color == (0, 255, 255):  # 青色
            # 保持青色特性，增强亮度变化
            result = (
                0,
                min(255, int(255 * v * saturation_boost)),
                min(255, int(255 * v * saturation_boost))
            )
        elif base_color == (255, 0, 255):  # 紫色
            result = (
                min(255, int(255 * v * saturation_boost)),
                0,
                min(255, int(255 * v * saturation_boost))
            )
        elif base_color == (255, 255, 0):  # 黄色
            result = (
                min(255, int(255 * v * saturation_boost)),
                min(255, int(255 * v * saturation_boost)),
                0
            )
        else:
            # 通用颜色处理，保持颜色比例
            r, g, b = base_color
            result = (
                min(255, int(r * v * saturation_boost)),
                min(255, int(g * v * saturation_boost)),
                min(255, int(b * v * saturation_boost))
            )

        # 缓存结果，限制缓存大小
        if len(self._color_cache) < 150:
            self._color_cache[cache_key] = result

        return result





    def _draw_corner_box(self, draw, x1, y1, x2, y2, color, lw=2, corner_len=20):
        """绘制科技感边角框"""
        if hasattr(draw, 'line'):  # PIL draw
            draw.line([(x1, y1), (x1 + corner_len, y1)], fill=color, width=lw)
            draw.line([(x1, y1), (x1, y1 + corner_len)], fill=color, width=lw)
            draw.line([(x2, y1), (x2 - corner_len, y1)], fill=color, width=lw)
            draw.line([(x2, y1), (x2, y1 + corner_len)], fill=color, width=lw)
            draw.line([(x1, y2), (x1 + corner_len, y2)], fill=color, width=lw)
            draw.line([(x1, y2), (x1, y2 - corner_len)], fill=color, width=lw)
            draw.line([(x2, y2), (x2 - corner_len, y2)], fill=color, width=lw)
            draw.line([(x2, y2), (x2, y2 - corner_len)], fill=color, width=lw)

    def _draw_tech_line(self, base_image, start_pt, end_pt, color1, color2, thickness=2, dash_length=20, gap=10):
        """
        在 base_image 图像上，从 start_pt 到 end_pt 画出具有科技感的渐变虚线，并带有发光效果。
        高性能优化版本：减少内存分配，优化计算逻辑

        参数:
            base_image: 原始图像（OpenCV BGR 格式）
            start_pt: 虚线起点坐标 (x, y)
            end_pt: 虚线终点坐标 (x, y)
            color1: 起点颜色 (R, G, B)
            color2: 终点颜色 (R, G, B)，用于渐变
            thickness: 虚线的线条粗细（默认2）
            dash_length: 每个虚线段的长度（默认20像素）
            gap: 虚线段之间的间隔（默认10像素）

        返回:
            处理后的图像（BGR 格式）
        """
        # 计算向量方向和长度
        vec = np.array(end_pt, dtype=np.float32) - np.array(start_pt, dtype=np.float32)
        dist = np.linalg.norm(vec)

        # 如果距离为 0 或太短，不进行绘制
        if dist == 0 or dist < 10:
            return base_image

        # 归一化方向向量
        direction = vec / dist

        # 计算可容纳多少个 dash + gap 的单元段
        segment_length = dash_length + gap
        steps = max(1, int(dist // segment_length))
        
        # 限制最大步数以控制性能
        # steps = min(steps, 15)

        # 预计算颜色差值，避免循环中重复计算
        color_diff = np.array(color2, dtype=np.float32) - np.array(color1, dtype=np.float32)
        
        # 预计算起点坐标，避免重复转换
        start_array = np.array(start_pt, dtype=np.float32)

        # 直接在原图上绘制，避免额外的图像复制
        for i in range(steps):
            # 优化：使用整数除法避免浮点运算
            if steps > 1:
                alpha = i / (steps - 1)
            else:
                alpha = 0

            # 当前 dash 起点 s，终点 e（优化：减少数组操作）
            offset = i * segment_length
            s = start_array + direction * offset
            e = s + direction * dash_length

            # 优化的颜色计算：直接计算BGR格式，避免中间变量
            current_color = color1 + alpha * color_diff
            bgr_color = (int(current_color[2]), int(current_color[1]), int(current_color[0]))

            # 转换为整数坐标
            s_int = (int(s[0]), int(s[1]))
            e_int = (int(e[0]), int(e[1]))

            # 绘制主线段
            cv2.line(base_image, s_int, e_int, bgr_color, thickness)

            # 简化的发光效果：直接绘制较粗的半透明线条
            if thickness > 1:
                 glow_color = (int(current_color[2] * 0.4), int(current_color[1] * 0.4), int(current_color[0] * 0.4))
                 cv2.line(base_image, s_int, e_int, glow_color, thickness + 3)
        # 画一个半径为 2 的红色点
        cv2.circle(base_image, end_pt, radius=2, color=(0, 0, 255), thickness=-1)

        # cv2.imshow("base",base_image)
        # cv2.waitKey(0)
        return base_image
    
    def _draw_particles(self, image, start_pt, end_pt, color, current_time, segment_idx):
        """
        绘制粒子效果
        """
        # 沿线段生成粒子
        num_particles = 8
        for p in range(num_particles):
            t = p / max(1, num_particles - 1)
            particle_pos = start_pt + t * (end_pt - start_pt)
            
            # 添加随机偏移和动画
            offset_x = 3 * math.sin(current_time / 100 + segment_idx + p)
            offset_y = 3 * math.cos(current_time / 150 + segment_idx + p)
            
            final_pos = (int(particle_pos[0] + offset_x), int(particle_pos[1] + offset_y))
            
            # 粒子大小动画
            particle_size = int(2 + math.sin(current_time / 80 + p) * 1)
            
            # 绘制发光粒子
            cv2.circle(image, final_pos, particle_size + 2, tuple(int(c * 0.3) for c in color), -1)
            cv2.circle(image, final_pos, particle_size, color, -1)
    
    def _draw_energy_ripples(self, image, start_pt, end_pt, color, current_time, segment_idx):
        """
        绘制能量波纹效果
        """
        # 在线段中点绘制能量波纹
        center = ((start_pt + end_pt) / 2).astype(int)
        
        # 多个波纹层
        for ripple in range(3):
            phase = current_time / 300 + segment_idx * 0.3 + ripple * 0.5
            radius = int(5 + 8 * math.sin(phase))
            alpha = 0.6 - ripple * 0.2
            
            if radius > 0:
                ripple_color = tuple(int(c * alpha) for c in color)
                cv2.circle(image, tuple(center), radius, ripple_color, 1)

    def _draw_ground_radar(self, draw, cx, cy, frame_id, detection_width):
        """
        绘制地面雷达扫描效果（根据检测框大小动态调整）
        
        Args:
            draw: PIL.ImageDraw 对象
            cx, cy: 雷达中心坐标
            frame_id: 当前帧编号
            detection_width: 检测框的宽度，用于动态调整雷达大小
        """
        # 根据检测框宽度动态计算雷达最大半径
        # 最大波纹刚好为检测框的宽度
        max_radar_radius = detection_width // 2
        
        # 设置最小和最大半径限制
        max_radar_radius = max(30, min(max_radar_radius, 200))
        
        # 进一步减少层数和波纹数量以提升性能
        num_layers = 1  # 减少到1层
        num_rings = 3   # 增加波纹数量以增强效果
        
        # 使用更大的步长减少计算频率
        step = max(1, frame_id // 3)  # 降低更新频率

        for j in range(num_layers):
            for i in range(num_rings):
                # 计算动态半径，基于检测框宽度
                base_radius = max_radar_radius * (i + 1) / num_rings
                radius = (step * 4 + i * 30 + j * 20) % max_radar_radius
                
                # 跳过太小的半径
                if radius < 10:
                    continue

                # 设置透明度，根据半径动态调整
                alpha = int(200 * (1 - radius / max_radar_radius))
                if alpha < 30:  # 跳过太透明的
                    continue

                # 青绿色雷达颜色
                color = (0, 255, 255, alpha)

                # 椭圆边框框选区域（横向扁的椭圆）
                # 椭圆的宽度与检测框宽度成比例
                ellipse_width = radius
                ellipse_height = radius // 3  # 保持扁平的椭圆形状
                
                bbox = [
                    cx - ellipse_width,
                    cy - ellipse_height,
                    cx + ellipse_width,
                    cy + ellipse_height
                ]

                # 绘制透明的椭圆形波纹
                draw.ellipse(bbox, outline=color, width=1)  # 减少线宽
    
    def _draw_floating_ui(self, draw, frame_id, x, y, w, h, label, score, distance=None, warning_level="safe"):
        """
        绘制科技感云朵状悬浮UI标签框
        
        Args:
            draw: PIL.ImageDraw 对象
            frame_id: 当前帧编号
            x, y: 检测框左上角坐标
            w, h: 检测框宽高
            label: 类别标签
            score: 置信度分数
            distance: 距离信息（可选）
            warning_level: 预警等级（"safe", "warning", "danger"）
        """
        # 根据检测框大小计算缩放因子
        base_size = 100
        avg_size = (w + h) / 2
        scale_factor = max(0.6, min(2.0, avg_size / base_size))
        
        # 动态悬浮效果
        angle_rad = math.radians((frame_id * 2) % 360)
        offset_x = 4 * self._get_cached_sin(angle_rad) * scale_factor
        offset_y = 2 * self._get_cached_cos(angle_rad) * scale_factor
        
        # 根据预警等级确定颜色
        if warning_level == "danger":
            base_color = (255, 60, 60)  # 红色
            glow_color = (255, 100, 100)
        elif warning_level == "warning":
            base_color = (255, 200, 60)  # 橙色
            glow_color = (255, 220, 100)
        else:
            base_color = (60, 255, 60)  # 绿色
            glow_color = (100, 255, 100)
        
        # 呼吸效果强度
        breath_intensity = 0.5 + 0.3 * self._get_cached_sin(frame_id / 6.0)
        
        # 根据缩放因子调整字体大小
        font_size = int(30 * scale_factor)
        try:
            font = ImageFont.truetype("/font/STKAITI.TTF", font_size)
        except (OSError, IOError):
            font = self._font_cache
        
        # 组合文本内容
        label_chinese = {
            'motor':'摩托车',
            'car':'汽车',
            'tricycle':'三轮车',
            'bus': '公交车',
            'minibus': 'mini公交车',
            'truck': '卡车',
        }
        main_text = f"{label_chinese.get(label, '汽车')}：{score:.1f}"
        if distance is not None:
            distance_text = f"距离：{distance:.1f}m"
            text_lines = [main_text, distance_text]
        else:
            text_lines = [main_text]
        
        # 使用字体计算文本尺寸
        text_bbox = font.getbbox(max(text_lines, key=len))
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        padding = int(8 * scale_factor)
        ui_width = text_width + padding * 2
        ui_height = len(text_lines) * (text_height + 5) + padding * 2
        
        # 计算位置（位于检测框上方）
        ui_x = x + w // 2 - ui_width // 2 + offset_x
        ui_y = y - ui_height - int(10 * scale_factor) + offset_y
        
        # 绘制科技感云朵状背景
        # self._draw_cloud_ui_background(draw, ui_x, ui_y, ui_width, ui_height, base_color, glow_color, scale_factor, breath_intensity, frame_id)
        
        # 绘制文字（充满悬浮框内部）
        text_x = ui_x + padding
        text_y = ui_y + padding
        
        # 文字颜色（彩虹色效果）
        # 根据帧数创建彩虹色循环
        hue = (frame_id * 3) % 360  # 色相循环
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)  # 饱和度和亮度最大
        text_color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 255)
        
        # 绘制每行文字
        for i, line in enumerate(text_lines):
            line_y = text_y + i * (text_height + 5)
            draw.text((text_x, line_y), line, fill=text_color, font=font)
    
    def _draw_cloud_ui_background(self, draw, x, y, width, height, base_color, glow_color, scale_factor, breath_intensity, frame_id):
        """
        绘制科技感云朵状UI背景
        
        Args:
            draw: PIL.ImageDraw 对象
            x, y: 背景左上角坐标
            width, height: 背景尺寸
            base_color: 基础颜色
            glow_color: 发光颜色
            scale_factor: 缩放因子
            breath_intensity: 呼吸强度
            frame_id: 当前帧编号
        """
        # 呼吸效果的透明度
        alpha = int(120 + 60 * breath_intensity)
        glow_alpha = int(80 + 40 * breath_intensity)
        
        # 云朵状外形的控制点
        cloud_points = self._generate_cloud_shape(x, y, width, height, scale_factor, frame_id)
        
        # 绘制发光外层（更大的云朵形状）
        glow_points = [(px + 2, py + 2) for px, py in cloud_points]
        draw.polygon(glow_points, fill=(*glow_color, glow_alpha), outline=None)
        
        # 绘制主体云朵形状
        draw.polygon(cloud_points, fill=(*base_color, alpha), outline=None)
        
        # 绘制边框
        border_color = (*base_color, min(255, alpha + 50))
        draw.polygon(cloud_points, outline=border_color, width=2)

        # 添加内部发光效果
        inner_glow_points = [(px + 1, py + 1) for px, py in cloud_points]
        inner_alpha = int(40 + 20 * breath_intensity)
        draw.polygon(inner_glow_points, fill=(255, 255, 255, inner_alpha), outline=None)
    
    def _generate_cloud_shape(self, x, y, width, height, scale_factor, frame_id):
        """
        生成云朵状的多边形点集
        
        Args:
            x, y: 起始坐标
            width, height: 基础尺寸
            scale_factor: 缩放因子
            frame_id: 当前帧编号（用于动态效果）
            
        Returns:
            list: 多边形点集
        """
        points = []
        num_points = 12  # 云朵边缘点数
        
        # 中心点
        cx = x + width // 2
        cy = y + height // 2
        
        # 基础半径
        base_radius_x = width // 2
        base_radius_y = height // 2
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            
            # 添加随机波动创造云朵效果
            wave_offset = 0.3 * self._get_cached_sin(frame_id / 10.0 + i)
            radius_variation = 0.7 + 0.3 * self._get_cached_sin(angle * 3 + frame_id / 15.0) + wave_offset
            
            # 计算点坐标
            px = cx + base_radius_x * radius_variation * math.cos(angle)
            py = cy + base_radius_y * radius_variation * math.sin(angle)
            
            points.append((int(px), int(py)))
        
        return points
    
    def _draw_fresh_ui_background(self, draw, x, y, width, height, scale_factor, breath_intensity):
        """
        绘制清新风格的UI背景
        
        Args:
            draw: PIL.ImageDraw 对象
            x, y: 背景左上角坐标
            width, height: 背景尺寸
            scale_factor: 缩放因子
            breath_intensity: 呼吸强度
        """
        # 圆角矩形的半径
        corner_radius = int(6 * scale_factor)
        
        # 主背景（白色半透明）
        bg_alpha = int(200 + 30 * breath_intensity)
        bg_color = (255, 255, 255, bg_alpha)
        
        # 绘制圆角矩形背景
        self._draw_rounded_rectangle(draw, x, y, width, height, corner_radius, bg_color)
        
        # 轻微的边框（淡灰色）
        border_color = (200, 200, 200, 150)
        border_width = max(1, int(1 * scale_factor))
        self._draw_rounded_rectangle(draw, x, y, width, height, corner_radius, None, border_color, border_width)
        
        # 顶部高光效果
        highlight_height = max(1, int(2 * scale_factor))
        highlight_color = (255, 255, 255, 100)
        self._draw_rounded_rectangle(draw, x + 1, y + 1, width - 2, highlight_height, corner_radius - 1, highlight_color)
    
    def _draw_rounded_rectangle(self, draw, x, y, width, height, radius, fill_color=None, outline_color=None, outline_width=1):
        """
        绘制圆角矩形
        
        Args:
            draw: PIL.ImageDraw 对象
            x, y: 矩形左上角坐标
            width, height: 矩形尺寸
            radius: 圆角半径
            fill_color: 填充颜色
            outline_color: 边框颜色
            outline_width: 边框宽度
        """
        # 使用PIL的rounded_rectangle方法（如果可用）
        try:
            bbox = [x, y, x + width, y + height]
            if fill_color:
                draw.rounded_rectangle(bbox, radius=radius, fill=fill_color)
            if outline_color:
                draw.rounded_rectangle(bbox, radius=radius, outline=outline_color, width=outline_width)
        except AttributeError:
            # 如果PIL版本不支持rounded_rectangle，使用传统方法
            if fill_color:
                draw.rectangle([x + radius, y, x + width - radius, y + height], fill=fill_color)
                draw.rectangle([x, y + radius, x + width, y + height - radius], fill=fill_color)
                draw.ellipse([x, y, x + 2*radius, y + 2*radius], fill=fill_color)
                draw.ellipse([x + width - 2*radius, y, x + width, y + 2*radius], fill=fill_color)
                draw.ellipse([x, y + height - 2*radius, x + 2*radius, y + height], fill=fill_color)
                draw.ellipse([x + width - 2*radius, y + height - 2*radius, x + width, y + height], fill=fill_color)
            
            if outline_color:
                # 简化的边框绘制
                draw.rectangle([x, y, x + width, y + height], outline=outline_color, width=outline_width)
    
    def _draw_tech_ui_frame(self, draw, x, y, width, height, scale_factor, glow_intensity, frame_id):
        """
        绘制科技风格的UI框架
        """
        # 六边形主框架
        corner_size = int(8 * scale_factor)
        
        # 主框架背景（半透明深色）
        bg_color = (10, 25, 45, 180)
        self._draw_hexagon_frame(draw, x, y, width, height, corner_size, bg_color)
        
        # 外层发光边框
        glow_color = (0, glow_intensity, 255, 120)
        border_width = max(1, int(2 * scale_factor))
        self._draw_hexagon_frame(draw, x, y, width, height, corner_size, None, glow_color, border_width)
        
        # 内层精细边框
        inner_color = (100, 200, 255, 200)
        inner_width = max(1, int(1 * scale_factor))
        self._draw_hexagon_frame(draw, x + 2, y + 2, width - 4, height - 4, corner_size - 1, None, inner_color, inner_width)
        
        # 动态扫描线效果
        scan_progress = (frame_id * 3) % (width + 20)
        if scan_progress < width:
            scan_x = x + scan_progress
            scan_color = (0, 255, 255, 150)
            # 垂直扫描线
            draw.line([(scan_x, y + 5), (scan_x, y + height - 5)], fill=scan_color, width=max(1, int(2 * scale_factor)))
            # 扫描线发光效果
            for i in range(1, 4):
                alpha = 100 - i * 25
                glow_scan_color = (0, 200, 255, alpha)
                draw.line([(scan_x - i, y + 5), (scan_x - i, y + height - 5)], fill=glow_scan_color, width=1)
                draw.line([(scan_x + i, y + 5), (scan_x + i, y + height - 5)], fill=glow_scan_color, width=1)
        
        # 角落装饰元素
        self._draw_corner_decorations(draw, x, y, width, height, scale_factor, glow_intensity)
    
    def _draw_hexagon_frame(self, draw, x, y, width, height, corner_size, fill_color=None, outline_color=None, outline_width=1):
        """
        绘制六边形风格的框架
        """
        # 计算六边形的关键点
        points = [
            (x + corner_size, y),
            (x + width - corner_size, y),
            (x + width, y + corner_size),
            (x + width, y + height - corner_size),
            (x + width - corner_size, y + height),
            (x + corner_size, y + height),
            (x, y + height - corner_size),
            (x, y + corner_size)
        ]
        
        if fill_color:
            draw.polygon(points, fill=fill_color)
        if outline_color:
            draw.polygon(points, outline=outline_color, width=outline_width)
    
    def _draw_corner_decorations(self, draw, x, y, width, height, scale_factor, glow_intensity):
        """
        绘制角落装饰元素
        """
        deco_size = int(6 * scale_factor)
        deco_color = (0, glow_intensity, 255, 180)
        line_width = max(1, int(1 * scale_factor))
        
        # 左上角
        draw.line([(x + 2, y + deco_size), (x + 2, y + 2), (x + deco_size, y + 2)], fill=deco_color, width=line_width)
        # 右上角
        draw.line([(x + width - deco_size, y + 2), (x + width - 2, y + 2), (x + width - 2, y + deco_size)], fill=deco_color, width=line_width)
        # 左下角
        draw.line([(x + 2, y + height - deco_size), (x + 2, y + height - 2), (x + deco_size, y + height - 2)], fill=deco_color, width=line_width)
        # 右下角
        draw.line([(x + width - deco_size, y + height - 2), (x + width - 2, y + height - 2), (x + width - 2, y + height - deco_size)], fill=deco_color, width=line_width)
    
    def _draw_cloud_shape(self, draw, x, y, width, height, fill_color=None, outline_color=None, outline_width=1):
        """
        绘制云朵形状
        
        Args:
            draw: PIL.ImageDraw 对象
            x, y: 云朵左上角坐标
            width, height: 云朵尺寸
            fill_color: 填充颜色
            outline_color: 边框颜色
            outline_width: 边框宽度
        """
        # 云朵由多个圆形组成
        circle_radius = height // 4
        
        # 主体矩形
        main_rect = [x + circle_radius, y + circle_radius//2, 
                    x + width - circle_radius, y + height - circle_radius//2]
        
        if fill_color:
            draw.rectangle(main_rect, fill=fill_color)
        if outline_color:
            draw.rectangle(main_rect, outline=outline_color, width=outline_width)
        
        # 云朵的圆形部分
        circles = [
            # 左侧圆
            [x, y + circle_radius//2, x + circle_radius*2, y + height - circle_radius//2],
            # 右侧圆
            [x + width - circle_radius*2, y + circle_radius//2, x + width, y + height - circle_radius//2],
            # 顶部左圆
            [x + circle_radius//2, y, x + circle_radius//2 + circle_radius*1.5, y + circle_radius*1.5],
            # 顶部右圆
            [x + width - circle_radius*1.5 - circle_radius//2, y, x + width - circle_radius//2, y + circle_radius*1.5],
            # 底部圆
            [x + width//2 - circle_radius//2, y + height - circle_radius, x + width//2 + circle_radius//2, y + height]
        ]
        
        for circle in circles:
            if fill_color:
                draw.ellipse(circle, fill=fill_color)
            if outline_color:
                draw.ellipse(circle, outline=outline_color, width=outline_width)

    def _draw_energy_bar(self, draw, x1, y1, x2, y2, distance, safe_distance=10):
        """
        绘制能量条（距离指示器）
        
        Args:
            draw: PIL.ImageDraw 对象
            x1, y1, x2, y2: 检测框坐标
            distance: 当前距离
            safe_distance: 安全距离阈值
        """
        # 计算能量比例
        energy_ratio = min(1.0, max(0.1, distance / safe_distance))
        
        # 能量条尺寸
        bar_height = 12
        bar_width = int(x2 - x1)
        
        # 能量条位置（在检测框下方）
        bar_x1 = x1
        bar_y1 = y2 + 5
        
        # 使用分段矩形绘制渐变
        segments = 20
        segment_width = bar_width // segments
        
        for i in range(segments):
            ratio = i / segments
            if ratio > energy_ratio:
                break
                
            # 渐变色从红到绿
            r = int(255 * (1 - ratio))
            g = int(255 * ratio)
            
            draw.rectangle(
                [
                    bar_x1 + i * segment_width,
                    bar_y1,
                    bar_x1 + (i + 1) * segment_width,
                    bar_y1 + bar_height
                ],
                fill=(r, g, 0, 255)
            )
        
        # 绘制剩余部分（灰色）
        draw.rectangle(
            [
                bar_x1 + int(bar_width * energy_ratio),
                bar_y1,
                bar_x1 + bar_width,
                bar_y1 + bar_height
            ],
            fill=(50, 50, 50, 180)
        )
        
        # 显示百分比
        font_size = 28
        try:
            font = ImageFont.truetype("/font/STKAITI.TTF", font_size)
        except (OSError, IOError):
            font = self._font_cache
        percentage_text = f"{int(energy_ratio * 100)}%"
        draw.text(
            (bar_x1 + bar_width + 5, bar_y1 - 12),
            percentage_text,
            fill=(255, 0, 0),
            font = font,

        )

    def _draw_lock_on_target(self, draw, x1, y1, x2, y2, frame_id, warning_level="safe"):
        """
        绘制目标锁定效果
        
        Args:
            draw: PIL.ImageDraw 对象
            x1, y1, x2, y2: 检测框坐标
            frame_id: 当前帧编号
            warning_level: 预警等级
        """
        # 根据预警等级选择颜色
        if warning_level == "danger":
            base_color = (255, 60, 60) # 红色
        elif warning_level == "warning":
            base_color = (255, 255, 60) # 橙色
        else:
            base_color = (60, 255, 60) # 绿色

        attribute_width = 5
        # 动态透明度
        alpha = int(150 + 100 * self._get_cached_sin(frame_id / 3.0))
        color = (*base_color, alpha)
        
        # 计算中心点
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # 动态半径
        r = 15 + int(5 * self._get_cached_sin(frame_id / 5.0))
        
        # 绘制锁定圆
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=color,
            width=attribute_width
        )
        
        # 绘制十字准星
        cross_size = r // 2
        draw.line(
            [(cx - cross_size, cy), (cx + cross_size, cy)],
            fill=color,
            width=attribute_width
        )
        draw.line(
            [(cx, cy - cross_size), (cx, cy + cross_size)],
            fill=color,
            width=attribute_width
        )

    def get_warning_level(self, distance, safe_distance=10):
        """
        根据距离确定预警等级
        
        Args:
            distance: 当前距离
            safe_distance: 安全距离阈值
            
        Returns:
            str: 预警等级 ("safe", "warning", "danger")
        """
        if distance < safe_distance:
            return "danger"
        elif distance < safe_distance * 1.5:
            return "warning"
        else:
            return "safe"

    def visualize_detections(self, img, detections, frame_id=0, safe_distance=10):
        """
        主要的可视化函数，将检测结果渲染为科技感界面
        
        Args:
            img: 输入图像（OpenCV BGR格式）
            detections: 检测结果列表，每个元素包含车辆信息和距离
            frame_id: 当前帧编号
            safe_distance: 安全距离阈值
            
        Returns:
            np.ndarray: 渲染后的图像
        """
        h, w = img.shape[:2]
        image_bottom_center = (w // 2, h - 1)
        
        # 创建OpenCV图像副本用于科技线条
        overlay_img = img.copy()
        
        # 转换为PIL格式用于高级绘制效果
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw = ImageDraw.Draw(pil_img, "RGBA")
        
        # 预计算动态颜色（减少重复计算）
        color1 = self._get_dynamic_color(frame_id, self.tech_color)
        color2 = self._get_dynamic_color(frame_id + 10, (255, 0, 255))
        
        # 批量处理检测结果以提升性能
        if not detections:
            return img
        
        for detection in detections:
            # 解析检测信息
            x1, y1, x2, y2 = detection["box"]
            label = detection.get("label", "vehicle")
            score = detection.get("score", 0.0)
            distance = detection.get("distance", 0.0)
            
            # 确定预警等级
            warning_level = self.get_warning_level(distance, safe_distance)
            
            # 根据预警等级获取检测框颜色
            if warning_level == "danger":
                base_color = (255, 60, 60, 255)  # 红色
            elif warning_level == "warning":
                base_color = (255, 200, 60, 255)  # 橙色
            else:
                base_color = (60, 255, 60, 255)  # 绿色

            # base_color = self._get_dynamic_color(frame_id)
            lw = 2
            corner_len = 35
            # 绘制科技感边角框
            if self.enable_floating_ui:
                self._draw_corner_box(draw, x1, y1, x2, y2, base_color[:3], lw=lw, corner_len=corner_len)
            
            # 绘制悬浮UI（传递预警等级）
            if self.enable_floating_ui:
                self._draw_floating_ui(draw, frame_id, x1, y1, x2-x1, y2-y1, label, score, distance, warning_level)
            
            # 绘制能量条
            if self.enable_energy_bar:
                self._draw_energy_bar(draw, x1, y1, x2, y2, distance, safe_distance)
            
            # 绘制目标锁定
            if self.enable_lock_target:
                self._draw_lock_on_target(draw, x1, y1, x2, y2, frame_id, warning_level)
            
            # 绘制科技连接线（从图像底部中心连接到目标锁定位置）
            if self.enable_tech_lines:
                # 连接线起点为图像底部中心，终点为目标锁定的中心位置
                lock_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                overlay_img = self._draw_tech_line(
                    overlay_img, image_bottom_center, lock_center,
                    color1=color1, color2=color2, thickness=2
                )
            
            # 绘制地面雷达
            if self.enable_radar:
                radar_center = ((x1 + x2) // 2, y2 + 20)
                detection_width = x2 - x1  # 计算检测框宽度
                self._draw_ground_radar(draw, radar_center[0], radar_center[1], frame_id, detection_width)
        
        # 转换回OpenCV格式
        img_final = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        # 最终合成（优化混合权重以减少计算）
        result = cv2.addWeighted(overlay_img, 0.3, img_final, 0.7, 0)
        
        return result

    def set_visualization_options(self, **options):
        """
        设置可视化选项
        
        Args:
            **options: 可视化选项字典
                - enable_radar: 是否启用雷达效果
                - enable_lock_target: 是否启用目标锁定
                - enable_energy_bar: 是否启用能量条
                - enable_tech_lines: 是否启用科技连接线
                - enable_floating_ui: 是否启用悬浮UI
        """
        for key, value in options.items():
            if hasattr(self, key):
                setattr(self, key, value)