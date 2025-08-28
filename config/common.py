import cv2
import os
import json
from dataclasses import dataclass


# 定义一个数据类，用于存储视频信息
@dataclass
class VideoInfo:
    width: int          # 视频宽度（像素）
    height: int         # 视频高度（像素）
    fps: int          # 视频帧率（frames per second）
    frame_count: int    # 视频总帧数
    duration_sec: float # 视频时长（秒）
    codec: str          # 视频编码格式（fourcc码对应的字符串）
    name:str            # 视频名字
# 定义一个函数，用于获取视频的关键信息，返回一个 VideoInfo 对象
def video_args(video_path: str, is_video_info: bool = True) -> VideoInfo:
    """
    功能：获取传入视频的参数
    """
    cap = cv2.VideoCapture(video_path)  # 初始化视频捕获
    # 判断视频是否成功打开，没打开则抛出异常
    if not cap.isOpened():
        print(f"获取视频参数，无法打开视频文件：{video_path}")
        raise IOError(f"无法打开视频文件：{video_path}")

    # 获取视频名字
    filename = os.path.basename(video_path)  # 得到 "out.mp4"
    video_name = os.path.splitext(filename)[0]  # 去掉扩展名，得到 "out"
    # 获取视频的宽度（单位：像素），转成整数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 获取视频的高度（单位：像素），转成整数
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取视频帧率，返回浮点数（fps）
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    # 获取视频的总帧数，转成整数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 计算视频时长（秒） = 总帧数 / 帧率，fps 大于0时计算，否则为0防止除零错误
    duration_sec = round(frame_count / fps) if fps > 0 else 0
    # 获取视频的 fourcc 编码（整数），用于标识视频编码格式
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    # 将 fourcc 整数编码转成对应的4字符字符串，例如 'avc1'、'XVID' 等
    # 通过位运算依次取出4个字节，转换为字符后拼接成字符串
    codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    # 释放视频资源，关闭文件
    cap.release()

    if is_video_info:
        # 视频信息
        video_info_dic = {
            "视频名": f"{filename}",
            "视频帧率": f"{fps}帧",
            "视频宽高": f"{width} x {height}",
            "视频时长": f"{duration_sec}s",
            "视频编码格式": f"{codec}"
        }
        # 格式化输出为缩进良好的 JSON 字符串
        video_info_dic = json.dumps(video_info_dic, indent=4, ensure_ascii=False)
        print(f"\n{filename}视频参数信息：{video_info_dic}")
    # 返回封装好的 VideoInfo 对象，包含所有获取到的视频信息
    v_info = VideoInfo(width, height, fps, frame_count, duration_sec, codec, video_name)
    return v_info