import cv2
import tensorrt as trt
import pycuda.driver as cuda
cuda.init()
device = cuda.Device(0)
# import pycuda.autoinit # 创建了隐式全局 context

import os
import torch
import numpy as np
import time
from typing import Tuple
from core.utils.infer_script import scale_boxes, non_max_suppression, yolo_visualize
from core.utils.trt_infer_utils import preprocess, visualize
from core.utils.nms_numpy import non_max_suppression_numpy


class TensorRTPredictor:
    def __init__(self, engine_path: str):
        """初始化TensorRT预测器（显存分配在初始化阶段完成）"""

        # 每一个线程创建自己显式的cuda context
        self.cuda_context = device.make_context()

        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context(strategy=trt.ExecutionContextAllocationStrategy.STATIC)

        # 获取输入输出tensor名称
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # 获取输入输出形状
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # 预分配显存
        self._setup_buffers()

        # 创建固定流
        self.stream = cuda.Stream()
        self.is_warmed_up = False

    def _load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """加载TensorRT引擎"""
        load_start_time = time.time()
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        load_end_time = time.time()
        load_time = (load_end_time - load_start_time) * 1000
        print(f"加载引擎时间: {load_time:.2f} ms")
        return engine

    def _setup_buffers(self):
        """预分配输入输出显存"""
        # 计算缓冲区大小
        input_size = int(np.prod(self.input_shape)) * np.float32().itemsize
        output_size = int(np.prod(self.output_shape)) * np.float32().itemsize

        # 分配固定显存
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)

        # 预分配主机锁页内存
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)

        # 设置tensor地址
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def warmup(self, iterations: int = 10):
        """模型预热（使用预分配显存）"""
        if self.is_warmed_up:
            print("模型已经预热，跳过预热步骤")
            return

        warmup_start_time = time.time()
        dummy_input = np.random.rand(*self.input_shape).astype(np.float32)

        for _ in range(iterations):
            cuda.memcpy_htod_async(self.d_input, dummy_input, self.stream)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()

        warmup_end_time = time.time()
        warmup_time = (warmup_end_time - warmup_start_time) * 1000
        print(f"  预热时间: {warmup_time:.2f} ms")
        self.is_warmed_up = True

    def infer(self, image) -> Tuple[float, np.ndarray]:
        self.cuda_context.push()
        try:
            """执行推理（复用预分配显存）"""
            if not self.is_warmed_up:
                print("警告：模型尚未预热，推理性能可能受影响")

            # 预处理
            preprocess_time = time.time()
            input_data, r_p = preprocess(image)
            # print(f"预处理时间: {(time.time() - preprocess_time) * 1000:.2f} ms")

            # 每个线程使用自己的stream
            stream = cuda.Stream()

            # 异步拷贝数据
            cuda.memcpy_htod_async(self.d_input, input_data, stream)

            # 执行推理
            self.context.execute_async_v3(stream_handle=stream.handle)

            # 异步拷贝结果回主机
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, stream)

            stream.synchronize()

            # 后处理
            postprocess_time = time.time()
            # self.h_output = torch.from_numpy(self.h_output).cuda()
            # results = non_max_suppression(self.h_output)
            # results = results.cpu().numpy()
            # or 后处理
            results = non_max_suppression_numpy(self.h_output)

            # 还原坐标到原图
            results = scale_boxes(results, input_data, image, r_p)
            # print(f"后处理时间：{(time.time() - postprocess_time) * 1000:.2f} ms")
            return results[0]

        finally:
            self.cuda_context.pop()

    def __del__(self):
        """析构函数自动释放显存"""
        if hasattr(self, 'd_input'):
            self.d_input.free()
        if hasattr(self, 'd_output'):
            self.d_output.free()
        print("显存资源已释放")

        """✅ 清理资源，释放上下文"""
        # try:
        #     if hasattr(self, "context") and self.context:
        #         self.context.pop()
        #         del self.context
        # except Exception as e:
        #     print("Error cleaning up CUDA context:", e)


if __name__ == '__main__':
    try:
        # 初始化推理器
        trt_predictor = TensorRTPredictor(r"D:\PythonProject\Forward_Collision_Warning_System\models\engine\car_detector.engine")
        trt_predictor.warmup(10)

        # 1. 测试单张图像推理
        # image = cv2.imread("images/FILE250717-140233-000278F.jpg")
        # if image is not None:
        #     boxes = trt_predictor.infer(image)
        #     image = visualize(image, boxes)
        #     cv2.imshow('image', image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # 2. 测试视频推理
        video_path = r'D:\PythonProject\Forward_Collision_Warning_System\video\test-1080p.mp4'
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            exit(1)

        count = 0
        total_time = 0

        print("开始视频推理...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 推理全流程
            t1 = time.time()
            boxes = trt_predictor.infer(frame)
            image = visualize(frame, boxes)
            t2 = time.time()

            frame_time = t2 - t1
            total_time += frame_time
            fps = 1 / frame_time if frame_time > 0 else 0

            print(f"推理第{count + 1}帧耗时：{round(frame_time * 1000)}ms, FPS: {round(fps)}")

            # 可选：显示结果
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            count += 1

            # 可选：限制处理帧数
            # if count >= 100:
            #     break

        cap.release()
        cv2.destroyAllWindows()

        # 统计信息
        if count > 0:
            avg_time = total_time / count
            avg_fps = 1 / avg_time if avg_time > 0 else 0
            print(f"\\n📊 统计信息:")
            print(f"总帧数: {count}")
            print(f"平均每帧耗时: {avg_time * 1000:.2f}ms")
            print(f"平均FPS: {avg_fps:.2f}")






    except Exception as e:
        print(f"❌ 推理过程出错: {e}")
        import traceback

        traceback.print_exc()