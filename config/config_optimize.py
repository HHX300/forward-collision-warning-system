#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件

功能描述:
- 系统配置参数管理
- 检测参数设置
- 预警阈值配置
- 界面显示参数

作者: AI Assistant
创建时间: 2024
"""

import os
import json
from typing import Dict, Any


class Config:
    """
    配置管理类
    负责管理系统的各种配置参数
    """

    def __init__(self, config_file: str = None):
        """
        初始化配置

        Args:
            config_file (str): 配置文件路径
        """
        self.config_file = config_file or "config/config.json"

        # 默认配置参数
        self.default_config = {
            # 系统基本参数
            "SYSTEM": {
                "APP_NAME": "🚗 智能车道检测与碰撞预警系统",
                "VERSION": "3.0.0",
                "DEBUG_MODE": False,
                "LOG_LEVEL": "INFO",
                "ENABLE_EMOJI": True,  # 启用emoji图标
                "LANGUAGE": "zh_CN"  # 界面语言
            },

            # 视频处理参数
            "VIDEO": {
                "DEFAULT_FPS": 30,
                "FRAME_SKIP": 1,  # 跳帧处理，1表示不跳帧
                "RESIZE_WIDTH": 1280,
                "RESIZE_HEIGHT": 720,
                "ENABLE_RESIZE": True
            },

            # 车道线检测参数
            "LANE_DETECTION": {
                "GAUSSIAN_BLUR_KERNEL": 5,
                "CANNY_LOW_THRESHOLD": 50,
                "CANNY_HIGH_THRESHOLD": 150,
                "HOUGH_RHO": 1,
                "HOUGH_THETA_DEGREES": 1,
                "HOUGH_THRESHOLD": 50,
                "MIN_LINE_LENGTH": 100,
                "MAX_LINE_GAP": 50,
                "ROI_TOP_RATIO": 0.6,  # 感兴趣区域顶部比例
                "ROI_BOTTOM_RATIO": 1.0,  # 感兴趣区域底部比例
                "LANE_LINE_THICKNESS": 8,
                "LANE_AREA_ALPHA": 0.3  # 车道区域透明度
            },

            # 车辆检测参数
            "VEHICLE_DETECTION": {
                "CONFIDENCE_THRESHOLD": 0.5,
                "NMS_THRESHOLD": 0.4,
                "INPUT_SIZE": [416, 416],
                "MIN_VEHICLE_AREA": 1000,
                "MAX_VEHICLE_AREA": 50000,
                "ROI_RATIO": 0.6,  # 只检测图像下半部分
                "DETECTION_CLASSES": [2, 3, 5, 7],  # 车辆类别ID
                "MODEL_PATH": "./models/yolo/",
                "CONFIG_FILE": "yolov3.cfg",
                "WEIGHTS_FILE": "yolov3.weights",
                "CLASSES_FILE": "coco.names"
            },

            # 距离计算参数
            "DISTANCE_CALCULATION": {
                "METHOD": "perspective",  # 计算方法: perspective, size_based, ground_plane
                "CAMERA_HEIGHT": 1.5,  # 摄像头高度（米）
                "CAMERA_ANGLE": 10,  # 摄像头俯仰角（度）
                "FOCAL_LENGTH": 800,  # 焦距（像素）
                "PIXEL_SIZE": 0.0055,  # 像素尺寸（毫米）
                "VEHICLE_AVERAGE_WIDTH": 1.8,  # 平均车宽（米）
                "VEHICLE_AVERAGE_HEIGHT": 1.5,  # 平均车高（米）
                "VEHICLE_AVERAGE_LENGTH": 4.5,  # 平均车长（米）
                "MIN_DISTANCE": 1.0,  # 最小检测距离（米）
                "MAX_DISTANCE": 200.0  # 最大检测距离（米）
            },

            # 碰撞预警参数
            "COLLISION_WARNING": {
                "SAFE_DISTANCE": 30.0,  # 安全距离（米）
                "WARNING_DISTANCE": 20.0,  # 预警距离（米）
                "DANGER_DISTANCE": 10.0,  # 危险距离（米）
                "CRITICAL_DISTANCE": 5.0,  # 紧急距离（米）
                "REACTION_TIME": 1.5,  # 反应时间（秒）
                "DECELERATION": 7.0,  # 制动减速度（m/s²）
                "ENABLE_AUDIO_WARNING": True,
                "WARNING_DURATION": 2.0  # 预警持续时间（秒）
            },

            # 界面显示参数
            "UI": {
                "WINDOW_WIDTH": 1400,
                "WINDOW_HEIGHT": 900,
                "WINDOW_MIN_WIDTH": 1200,
                "WINDOW_MIN_HEIGHT": 800,
                "VIDEO_DISPLAY_WIDTH": 900,
                "VIDEO_DISPLAY_HEIGHT": 700,
                "CONTROL_PANEL_WIDTH": 400,
                "CONTROL_PANEL_MIN_WIDTH": 350,
                "SPLITTER_RATIO": [70, 30],  # 视频区域和控制面板比例
                "FONT_SIZE": 12,
                "TITLE_FONT_SIZE": 16,
                "BUTTON_FONT_SIZE": 11,
                "STATUS_FONT_SIZE": 14,
                "UPDATE_INTERVAL": 33,  # 界面更新间隔（毫秒）
                "SHOW_FPS": True,
                "SHOW_DETECTION_INFO": True,
                "SHOW_VEHICLE_COUNT": True,
                "SHOW_FRAME_COUNT": True,
                "ENABLE_LOG_DISPLAY": True,
                "LOG_MAX_LINES": 1000,  # 日志显示最大行数
                "THEME": "modern_dark",  # 界面主题: default, modern_dark, modern_light
                "ENABLE_ANIMATIONS": True,  # 启用动画效果
                "BUTTON_HOVER_EFFECT": True,  # 按钮悬停效果
                "GRADIENT_BACKGROUNDS": True  # 渐变背景
            },

            # 颜色配置
            "COLORS": {
                "LANE_LINE_COLOR": [0, 255, 255],  # 车道线颜色（BGR）
                "LANE_AREA_COLOR": [0, 255, 0],  # 车道区域颜色（BGR）
                "SAFE_VEHICLE_COLOR": [0, 255, 0],  # 安全车辆框颜色（BGR）
                "WARNING_VEHICLE_COLOR": [0, 165, 255],  # 预警车辆框颜色（BGR）
                "DANGER_VEHICLE_COLOR": [0, 0, 255],  # 危险车辆框颜色（BGR）
                "TEXT_COLOR": [255, 255, 255],  # 文本颜色（BGR）
                "BACKGROUND_COLOR": [0, 0, 0]  # 背景颜色（BGR）
            },

            # UI样式配置
            "UI_STYLES": {
                "PRIMARY_COLOR": "#3498db",  # 主色调
                "SECONDARY_COLOR": "#2c3e50",  # 辅助色
                "SUCCESS_COLOR": "#27ae60",  # 成功色
                "WARNING_COLOR": "#f39c12",  # 警告色
                "DANGER_COLOR": "#e74c3c",  # 危险色
                "BACKGROUND_COLOR": "#2c3e50",  # 背景色
                "SURFACE_COLOR": "#34495e",  # 表面色
                "TEXT_COLOR": "#ecf0f1",  # 文本色
                "BORDER_COLOR": "#7f8c8d",  # 边框色
                "HOVER_COLOR": "#3498db",  # 悬停色
                "ACTIVE_COLOR": "#2980b9",  # 激活色
                "DISABLED_COLOR": "#95a5a6",  # 禁用色
                "SHADOW_COLOR": "rgba(0,0,0,0.3)",  # 阴影色
                "BORDER_RADIUS": "8px",  # 圆角半径
                "BUTTON_HEIGHT": "40px",  # 按钮高度
                "INPUT_HEIGHT": "35px",  # 输入框高度
                "SPACING_SMALL": "8px",  # 小间距
                "SPACING_MEDIUM": "16px",  # 中间距
                "SPACING_LARGE": "24px"  # 大间距
            },

            # 性能优化参数
            "PERFORMANCE": {
                "ENABLE_GPU": False,  # 是否启用GPU加速
                "NUM_THREADS": 4,  # 处理线程数
                "MEMORY_LIMIT_MB": 1024,  # 内存限制（MB）
                "ENABLE_CACHING": True,  # 是否启用缓存
                "CACHE_SIZE": 100  # 缓存大小
            },

            # 日志配置
            "LOGGING": {
                "ENABLE_LOGGING": True,
                "LOG_FILE": "system.log",
                "LOG_MAX_SIZE_MB": 10,
                "LOG_BACKUP_COUNT": 5,
                "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

        # 加载配置
        self.config = self.load_config()

        # 设置快捷访问属性
        self.setup_quick_access()

    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件

        Returns:
            Dict[str, Any]: 配置字典
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)

                # 合并默认配置和加载的配置
                config = self.merge_config(self.default_config, loaded_config)
                print(f"配置文件已加载: {self.config_file}")
                return config

            except Exception as e:
                print(f"配置文件加载失败: {e}，使用默认配置")
                return self.default_config.copy()
        else:
            print("配置文件不存在，使用默认配置")
            return self.default_config.copy()

    def merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置字典

        Args:
            default (Dict): 默认配置
            loaded (Dict): 加载的配置

        Returns:
            Dict: 合并后的配置
        """
        result = default.copy()

        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_config(result[key], value)
            else:
                result[key] = value

        return result

    def save_config(self):
        """
        保存配置到文件
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print(f"配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"配置保存失败: {e}")

    def setup_quick_access(self):
        """
        设置快捷访问属性
        """
        # 常用参数的快捷访问
        self.SAFE_DISTANCE = self.config["COLLISION_WARNING"]["SAFE_DISTANCE"]
        self.WARNING_DISTANCE = self.config["COLLISION_WARNING"]["WARNING_DISTANCE"]
        self.DANGER_DISTANCE = self.config["COLLISION_WARNING"]["DANGER_DISTANCE"]
        self.CRITICAL_DISTANCE = self.config["COLLISION_WARNING"]["CRITICAL_DISTANCE"]

        self.CONFIDENCE_THRESHOLD = self.config["VEHICLE_DETECTION"]["CONFIDENCE_THRESHOLD"]
        self.NMS_THRESHOLD = self.config["VEHICLE_DETECTION"]["NMS_THRESHOLD"]

        self.CAMERA_HEIGHT = self.config["DISTANCE_CALCULATION"]["CAMERA_HEIGHT"]
        self.CAMERA_ANGLE = self.config["DISTANCE_CALCULATION"]["CAMERA_ANGLE"]

        self.LANE_LINE_THICKNESS = self.config["LANE_DETECTION"]["LANE_LINE_THICKNESS"]

        # UI相关快捷访问
        self.WINDOW_WIDTH = self.config["UI"]["WINDOW_WIDTH"]
        self.WINDOW_HEIGHT = self.config["UI"]["WINDOW_HEIGHT"]
        self.THEME = self.config["UI"]["THEME"]
        self.ENABLE_EMOJI = self.config["SYSTEM"]["ENABLE_EMOJI"]
        self.APP_NAME = self.config["SYSTEM"]["APP_NAME"]
        self.VERSION = self.config["SYSTEM"]["VERSION"]

    def get(self, section: str, key: str = None, default=None):
        """
        获取配置值

        Args:
            section (str): 配置节名
            key (str): 配置键名
            default: 默认值

        Returns:
            配置值
        """
        if key is None:
            return self.config.get(section, default)
        else:
            return self.config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value):
        """
        设置配置值

        Args:
            section (str): 配置节名
            key (str): 配置键名
            value: 配置值
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

        # 更新快捷访问属性
        self.update_quick_access(section, key, value)

    def update_quick_access(self, section: str, key: str, value):
        """
        更新快捷访问属性

        Args:
            section (str): 配置节名
            key (str): 配置键名
            value: 配置值
        """
        # 更新对应的快捷访问属性
        if section == "COLLISION_WARNING":
            if key == "SAFE_DISTANCE":
                self.SAFE_DISTANCE = value
            elif key == "WARNING_DISTANCE":
                self.WARNING_DISTANCE = value
            elif key == "DANGER_DISTANCE":
                self.DANGER_DISTANCE = value
            elif key == "CRITICAL_DISTANCE":
                self.CRITICAL_DISTANCE = value

        elif section == "VEHICLE_DETECTION":
            if key == "CONFIDENCE_THRESHOLD":
                self.CONFIDENCE_THRESHOLD = value
            elif key == "NMS_THRESHOLD":
                self.NMS_THRESHOLD = value

        elif section == "DISTANCE_CALCULATION":
            if key == "CAMERA_HEIGHT":
                self.CAMERA_HEIGHT = value
            elif key == "CAMERA_ANGLE":
                self.CAMERA_ANGLE = value

        elif section == "LANE_DETECTION":
            if key == "LANE_LINE_THICKNESS":
                self.LANE_LINE_THICKNESS = value

    def get_camera_params(self) -> Dict[str, Any]:
        """
        获取摄像头参数

        Returns:
            Dict[str, Any]: 摄像头参数字典
        """
        distance_config = self.config["DISTANCE_CALCULATION"]
        video_config = self.config["VIDEO"]

        return {
            'focal_length': distance_config["FOCAL_LENGTH"],
            'camera_height': distance_config["CAMERA_HEIGHT"],
            'camera_angle': distance_config["CAMERA_ANGLE"],
            'image_width': video_config["RESIZE_WIDTH"],
            'image_height': video_config["RESIZE_HEIGHT"],
            'pixel_size': distance_config["PIXEL_SIZE"]
        }

    def get_vehicle_params(self) -> Dict[str, Any]:
        """
        获取车辆参数

        Returns:
            Dict[str, Any]: 车辆参数字典
        """
        distance_config = self.config["DISTANCE_CALCULATION"]

        return {
            'average_width': distance_config["VEHICLE_AVERAGE_WIDTH"],
            'average_height': distance_config["VEHICLE_AVERAGE_HEIGHT"],
            'average_length': distance_config["VEHICLE_AVERAGE_LENGTH"]
        }

    def get_color(self, color_name: str) -> tuple:
        """
        获取颜色配置

        Args:
            color_name (str): 颜色名称

        Returns:
            tuple: BGR颜色值
        """
        color_list = self.config["COLORS"].get(color_name, [255, 255, 255])
        return tuple(color_list)

    def get_warning_distances(self) -> Dict[str, float]:
        """
        获取预警距离配置

        Returns:
            Dict[str, float]: 预警距离字典
        """
        warning_config = self.config["COLLISION_WARNING"]

        return {
            'safe': warning_config["SAFE_DISTANCE"],
            'warning': warning_config["WARNING_DISTANCE"],
            'danger': warning_config["DANGER_DISTANCE"],
            'critical': warning_config["CRITICAL_DISTANCE"]
        }

    def get_ui_config(self) -> Dict[str, Any]:
        """
        获取UI配置

        Returns:
            Dict[str, Any]: UI配置字典
        """
        return self.config["UI"]

    def get_ui_styles(self) -> Dict[str, str]:
        """
        获取UI样式配置

        Returns:
            Dict[str, str]: UI样式字典
        """
        return self.config["UI_STYLES"]

    def get_window_config(self) -> Dict[str, int]:
        """
        获取窗口配置

        Returns:
            Dict[str, int]: 窗口配置字典
        """
        ui_config = self.config["UI"]
        return {
            'width': ui_config["WINDOW_WIDTH"],
            'height': ui_config["WINDOW_HEIGHT"],
            'min_width': ui_config["WINDOW_MIN_WIDTH"],
            'min_height': ui_config["WINDOW_MIN_HEIGHT"],
            'video_width': ui_config["VIDEO_DISPLAY_WIDTH"],
            'video_height': ui_config["VIDEO_DISPLAY_HEIGHT"],
            'control_panel_width': ui_config["CONTROL_PANEL_WIDTH"]
        }

    def get_theme_colors(self) -> Dict[str, str]:
        """
        根据当前主题获取颜色配置

        Returns:
            Dict[str, str]: 主题颜色字典
        """
        styles = self.config["UI_STYLES"]
        theme = self.config["UI"]["THEME"]

        if theme == "modern_dark":
            return {
                'primary': styles["PRIMARY_COLOR"],
                'secondary': styles["SECONDARY_COLOR"],
                'success': styles["SUCCESS_COLOR"],
                'warning': styles["WARNING_COLOR"],
                'danger': styles["DANGER_COLOR"],
                'background': styles["BACKGROUND_COLOR"],
                'surface': styles["SURFACE_COLOR"],
                'text': styles["TEXT_COLOR"],
                'border': styles["BORDER_COLOR"]
            }
        elif theme == "modern_light":
            return {
                'primary': "#2980b9",
                'secondary': "#ecf0f1",
                'success': "#27ae60",
                'warning': "#f39c12",
                'danger': "#e74c3c",
                'background': "#ffffff",
                'surface': "#f8f9fa",
                'text': "#2c3e50",
                'border': "#bdc3c7"
            }
        else:  # default theme
            return {
                'primary': "#3498db",
                'secondary': "#95a5a6",
                'success': "#2ecc71",
                'warning': "#f1c40f",
                'danger': "#e74c3c",
                'background': "#ffffff",
                'surface': "#ecf0f1",
                'text': "#2c3e50",
                'border': "#bdc3c7"
            }

    def validate_config(self) -> bool:
        """
        验证配置的有效性

        Returns:
            bool: 配置是否有效
        """
        try:
            # 检查必要的配置节是否存在
            required_sections = [
                "SYSTEM", "VIDEO", "LANE_DETECTION", "VEHICLE_DETECTION",
                "DISTANCE_CALCULATION", "COLLISION_WARNING", "UI", "COLORS", "UI_STYLES"
            ]

            for section in required_sections:
                if section not in self.config:
                    print(f"缺少配置节: {section}")
                    return False

            # 检查数值范围
            if not (0.0 <= self.CONFIDENCE_THRESHOLD <= 1.0):
                print("置信度阈值超出范围 [0.0, 1.0]")
                return False

            if not (0.0 <= self.NMS_THRESHOLD <= 1.0):
                print("NMS阈值超出范围 [0.0, 1.0]")
                return False

            if self.SAFE_DISTANCE <= 0:
                print("安全距离必须大于0")
                return False

            print("配置验证通过")
            return True

        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

    def reset_to_default(self):
        """
        重置为默认配置
        """
        self.config = self.default_config.copy()
        self.setup_quick_access()
        print("配置已重置为默认值")

    def export_config(self, export_path: str):
        """
        导出配置到指定路径

        Args:
            export_path (str): 导出路径
        """
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print(f"配置已导出到: {export_path}")
        except Exception as e:
            print(f"配置导出失败: {e}")

    def import_config(self, import_path: str):
        """
        从指定路径导入配置

        Args:
            import_path (str): 导入路径
        """
        if not os.path.exists(import_path):
            print(f"配置文件不存在: {import_path}")
            return False

        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)

            # 合并配置
            self.config = self.merge_config(self.default_config, imported_config)
            self.setup_quick_access()

            print(f"配置已从 {import_path} 导入")
            return True

        except Exception as e:
            print(f"配置导入失败: {e}")
            return False

    def print_config(self):
        """
        打印当前配置
        """
        print("当前配置:")
        print(json.dumps(self.config, indent=2, ensure_ascii=False))


# 全局配置实例
_global_config = None


def get_config() -> Config:
    """
    获取全局配置实例

    Returns:
        Config: 配置实例
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_global_config(config: Config):
    """
    设置全局配置实例

    Args:
        config (Config): 配置实例
    """
    global _global_config
    _global_config = config


# 测试代码
if __name__ == "__main__":
    # 创建配置实例
    config = Config()

    print("配置模块已加载")
    print("使用方法:")
    print("1. 创建配置实例: config = Config()")
    print("2. 获取配置值: value = config.get('SECTION', 'KEY')")
    print("3. 设置配置值: config.set('SECTION', 'KEY', value)")
    print("4. 保存配置: config.save_config()")

    # 验证配置
    if config.validate_config():
        print("\n配置验证成功")
    else:
        print("\n配置验证失败")

    # 显示一些关键配置
    print(f"\n关键配置参数:")
    print(f"安全距离: {config.SAFE_DISTANCE} 米")
    print(f"置信度阈值: {config.CONFIDENCE_THRESHOLD}")
    print(f"摄像头高度: {config.CAMERA_HEIGHT} 米")
    print(f"车道线粗细: {config.LANE_LINE_THICKNESS} 像素")