import cv2
import numpy as np
import os
import sys
from typing import Tuple, List, Optional, Union
# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


class LaneDetector:
    """
    车道线检测器 - 简化接口封装

    使用示例:
        detector = LaneDetector(
            engine_path="path/to/model.engine",
            config_path="path/to/config.py",
            ori_size=(1600, 320)
        )

        # 获取车道线坐标
        coords = detector.get_lane_coordinates(image)
    """

    def __init__(self, engine_path: str, config_path: str = None, ori_size: Tuple[int, int] = (1600, 320), auto_find_config: bool = True):
        """
        初始化车道线检测器

        Args:
            engine_path: TensorRT引擎文件路径
            config_path: 配置文件路径，如果为None且auto_find_config=True，会自动查找
            ori_size: 原始图像尺寸 (width, height)
            auto_find_config: 是否自动查找配置文件
        """
        self.engine_path = engine_path
        self.ori_size = ori_size

        # 自动查找配置文件
        if config_path is None and auto_find_config:
            config_path = self._auto_find_config(engine_path)

        if config_path is None:
            raise ValueError("配置文件路径不能为空，请提供config_path或确保引擎文件同目录下有对应的配置文件")


        # 初始化检测器
        from trt_infer import UFLDv2
        self.detector = UFLDv2(engine_path, config_path, ori_size)

    def _auto_find_config(self, engine_path: str) -> Optional[str]:
        """
        根据引擎文件路径自动查找对应的配置文件
        """
        engine_dir = os.path.dirname(engine_path)
        engine_name = os.path.splitext(os.path.basename(engine_path))[0]

        # 在同目录下查找
        config_file = os.path.join(engine_dir, f"{engine_name}.py")
        if os.path.exists(config_file):
            return config_file

        # 在configs目录下查找
        configs_dir = os.path.join(os.path.dirname(engine_dir), "configs")
        if os.path.exists(configs_dir):
            config_file = os.path.join(configs_dir, f"{engine_name}.py")
            if os.path.exists(config_file):
                return config_file

        return None


    def get_lane_coordinates(self, image: np.ndarray) -> List[List[Tuple[int, int]]]:
        """
        检测单张图片中的车道线

        Args:
            image: 输入图像 (numpy array)


        Returns:
            处理后的图像 (如果draw_lanes=True) 或原图像
        """
        processed_img = image.copy()

        # 检测图像和还原坐标
        coords = detect_frame_and_restore_coords(image, self.detector)

        return coords


    # def __del__(self):
    #     """清理资源"""
    #     if hasattr(self, 'detector'):
    #         # 清理CUDA资源
    #         try:
    #             for allocation in self.detector.allocations:
    #                 allocation.free()
    #         except:
    #             pass








def detect_frame_and_restore_coords(frame, detector):
    # 1.原始的处理流程(固定参数)
    img = cv2.resize(frame, (1600, 903))
    img = img[583:903, :, :]

    # 2.获取坐标（修改forward方法以返回坐标），有且仅有获取坐标
    coords = detector.get_coordinates_only(img)  # 需要修改UFLDv2类

    # 3.还原坐标到原始frame
    is_numpy = True
    original_frame = frame.shape  # h w c
    if is_numpy:
        # 方法一，numpy
        coord_transformer = CoordinateTransformer(original_frame)
        restored_coords = coord_transformer.transform(coords)
    else:
        # 方法二
        restored_coords = restore_coordinates_fast(coords, original_frame)

    # 4.在原始frame上绘制
    # result_frame = frame.copy()
    # for lane in restored_coords:
    #     for coord in lane:
    #         cv2.circle(result_frame, coord, 2, (0, 255, 0), -1)

    return restored_coords









def restore_coordinates_fast(coords: List[List[Tuple[int, int]]], original_frame_shape: Tuple[int, int, int], ) -> List[List[Tuple[int, int]]]:
    """
    更快速的坐标还原函数 - 预计算缩放参数

    Args:
        coords: 模型输出的车道线坐标
        original_frame_shape: 原始frame的形状 (height, width, channels)

    Returns:
        还原到原始frame尺寸的坐标点
    """
    if not coords:
        return coords

    original_height, original_width = original_frame_shape[:2]

    # 预计算的缩放参数（基于固定的变换流程）
    scale_x = original_width / 1600.0
    scale_y = original_height / 903.0
    crop_offset_y = 583 * scale_y

    restored_coords = []

    for lane in coords:
        if not lane:
            restored_coords.append([])
            continue

        # 使用列表推导式进行快速变换
        restored_lane = [
            (
                max(0, min(original_width - 1, int(x * scale_x))),
                max(0, min(original_height - 1, int(y * scale_y + crop_offset_y)))
            )
            for x, y in lane
        ]
        restored_coords.append(restored_lane)

    return restored_coords












class CoordinateTransformer:
    """
    坐标变换器类 - 用于批量处理时的性能优化
    """

    def __init__(self, original_frame_shape: Tuple[int, int, int], ori_size: Tuple[int, int] = (1600, 320)):
        self.original_height, self.original_width = original_frame_shape[:2]
        # self.ori_width, self.ori_height = ori_size

        # 预计算变换参数
        self.scale_x = self.original_width / 1600.0
        self.scale_y = self.original_height / 903.0
        self.crop_offset_y = 583 * self.scale_y

    def transform(self, coords: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        """
        快速坐标变换
        """
        if not coords:
            return coords

        restored_coords = []

        for lane in coords:
            if not lane:
                restored_coords.append([])
                continue

            # 向量化处理
            lane_array = np.array(lane, dtype=np.float32)

            # 应用变换
            lane_array[:, 0] *= self.scale_x
            lane_array[:, 1] = lane_array[:, 1] * self.scale_y + self.crop_offset_y

            # 限制范围并转换为整数
            lane_array = np.clip(lane_array, 0, [self.original_width - 1, self.original_height - 1])
            restored_lane = [(int(x), int(y)) for x, y in lane_array]
            restored_coords.append(restored_lane)

        return restored_coords
