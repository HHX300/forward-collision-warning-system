import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
cuda.init()
device = cuda.Device(0)

# import pycuda.autoinit # 创建了隐式全局 context
import torch
import argparse
import os
import sys
import time
import threading
import queue

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utilss.config import Config
from lane_region_draw import postprocess_coords_with_draw


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def softmax_np(x, axis=None):
    x = x - np.max(x, axis=axis, keepdims=True)  # 防止数值溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


class UFLDv2:
    def __init__(self, engine_path, config_path, ori_size):
        """
        初始化车道线检测模型
        Args:
            engine: TensorRT引擎
            config_path: 配置文件路径
            ori_size: 训练的图像尺寸 (width, height)
        """

        # 每一个线程创建自己显式的cuda context
        self.cuda_context = device.make_context()

        # 初始化 TensorRT 引擎
        self.engine = load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []

        # 分配输入输出显存空间 - 适配TensorRT 10.x
        for i in range(self.engine.num_io_tensors):  # 获取模型所有 I/O 张量数
            name = self.engine.get_tensor_name(i)  # 获取第 i 个张量名
            mode = self.engine.get_tensor_mode(name)  # 判断是输入还是输出
            dtype = self.engine.get_tensor_dtype(name)
            # 使用engine获取形状而不是context
            shape = self.engine.get_tensor_shape(name)

            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s

            allocation = cuda.mem_alloc(size)

            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)

            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
                self.batch_size = shape[0]
                # print(f"输入tensor: {name}, 形状: {shape}")
            else:
                self.outputs.append(binding)
                # print(f"输出tensor: {name}, 形状: {shape}")

        # 设置tensor地址 - TensorRT 10.x新API
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['allocation']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['allocation']))

        # 显式设置输入 shape（TensorRT 10 必须！）
        # for inp in self.inputs:
        #     self.context.set_input_shape(inp['name'], inp['shape'])

        # 加载配置参数
        cfg = Config.fromfile(config_path)

        self.ori_img_w, self.ori_img_h = ori_size
        # print(f"训练图像的尺寸: {self.ori_img_w} x {self.ori_img_h}")

        self.cut_height = int(cfg.train_height * (1 - cfg.crop_ratio)) # 裁剪高度
        self.input_width = cfg.train_width # 模型输入宽度: 1600
        self.input_height = cfg.train_height # 模型输入高度: 320
        self.num_row = cfg.num_row # 72
        self.num_col = cfg.num_col # 81
        # 锚点定义 - 用于将网格坐标转换为实际坐标
        # row_anchor: 定义检测的行位置 (从42%到100%的高度范围)
        self.row_anchor = np.linspace(0.42, 1, self.num_row)
        # col_anchor: 定义检测的列位置 (从0%到100%的宽度范围)
        self.col_anchor = np.linspace(0, 1, self.num_col)


    def preprocess(self, image):
        pre_time = time.time()
        image = image[self.cut_height:, :, :]
        image = cv2.resize(image, (self.input_width, self.input_height), cv2.INTER_CUBIC)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(np.float32(image[:, :, :, np.newaxis]), (3, 2, 0, 1))
        image = np.ascontiguousarray(image)
        # print(f"预处理耗时：{(time.time() - pre_time) * 1000:.2f}ms")
        return image


    def pred2coords_numpy(self, pred):
        post_time = time.time()

        # 获取预测结果的形状
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        # 获取最大置信度位置和存在性（沿第1维）
        max_indices_row = np.argmax(pred['loc_row'], axis=1)
        valid_row = np.argmax(pred['exist_row'], axis=1)
        max_indices_col = np.argmax(pred['loc_col'], axis=1)
        valid_col = np.argmax(pred['exist_col'], axis=1)

        coords = []

        row_lane_idx = [1, 2]
        col_lane_idx = [0, 3]

        # 行方向
        for i in row_lane_idx:
            tmp = []
            if np.sum(valid_row[0, :, i]) > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        center_idx = max_indices_row[0, k, i]
                        left = max(0, center_idx - self.input_width)
                        right = min(num_grid_row - 1, center_idx + self.input_width) + 1
                        all_ind = np.arange(left, right)
                        softmax_weights = softmax_np(pred['loc_row'][0, all_ind, k, i], axis=0)
                        out_tmp = np.sum(softmax_weights * all_ind.astype(np.float32)) + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.ori_img_w
                        tmp.append((int(out_tmp), int(self.row_anchor[k] * self.ori_img_h)))
                coords.append(tmp)

        # 列方向
        for i in col_lane_idx:
            tmp = []
            if np.sum(valid_col[0, :, i]) > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        center_idx = max_indices_col[0, k, i]
                        left = max(0, center_idx - self.input_width)
                        right = min(num_grid_col - 1, center_idx + self.input_width) + 1
                        all_ind = np.arange(left, right)
                        softmax_weights = softmax_np(pred['loc_col'][0, all_ind, k, i], axis=0)
                        out_tmp = np.sum(softmax_weights * all_ind.astype(np.float32)) + 0.5
                        out_tmp = out_tmp / (num_grid_col - 1) * self.ori_img_h
                        tmp.append((int(self.col_anchor[k] * self.ori_img_w), int(out_tmp)))
                coords.append(tmp)

        # print(f"后处理耗时：{(time.time() - post_time) * 1000:.2f}ms")
        return coords


    def pred2coords_tensor(self, pred):
        post_time = time.time()
        # 获取预测结果的形状
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        # 获取最大置信度位置和存在性
        max_indices_row = pred['loc_row'].argmax(1)  # 行方向最大置信度索引
        valid_row = pred['exist_row'].argmax(1)  # 行方向车道线存在性
        max_indices_col = pred['loc_col'].argmax(1)  # 列方向最大置信度索引
        valid_col = pred['exist_col'].argmax(1)  # 列方向车道线存在性

        coords = []

        # 车道线索引定义
        row_lane_idx = [1, 2]  # 处理第1和第2条车道线的行检测
        col_lane_idx = [0, 3]  # 处理第0和第3条车道线的列检测

        # 处理行方向检测的车道线
        for i in row_lane_idx:
            tmp = []
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - self.input_width), min(num_grid_row - 1, max_indices_row[0, k, i] + self.input_width) + 1)))
                        out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.ori_img_w
                        tmp.append((int(out_tmp), int(self.row_anchor[k] * self.ori_img_h)))
                coords.append(tmp)

        # 处理列方向检测的车道线
        for i in col_lane_idx:
            tmp = []
            if valid_col[0, :, i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        all_ind = torch.tensor(
                            list(range(max(0, max_indices_col[0, k, i] - self.input_width), min(num_grid_col - 1, max_indices_col[0, k, i] + self.input_width) + 1)))
                        out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col - 1) * self.ori_img_h
                        tmp.append((int(self.col_anchor[k] * self.ori_img_w), int(out_tmp)))
                coords.append(tmp)
        # print(f"后处理耗时：{(time.time() - post_time) * 1000:.2f}ms")
        return coords


    def get_coordinates_only(self, img):
        """只获取坐标，不绘制图像"""
        # 1.推理前 push cuda context
        # cuda_context = device.make_context()
        self.cuda_context.push()
        try:
            # 2.预处理输入图像，生产输入数组
            input_data = self.preprocess(img)

            # 3.创建 CUDA stream （每个线程使用自己的stream）
            infer_time = time.time()
            # stream = cuda.Stream()

            # 4.异步的拷贝输入数据 Host -> Device
            cuda.memcpy_htod_async(self.inputs[0]['allocation'], input_data)

            # 5.异步执行推理
            # for inp in self.inputs:
            #     self.context.set_input_shape(inp['name'], inp['shape'])

            a = self.context.execute_async_v3(stream_handle=0)

            # 6.异步拷贝输出数据 Device -> Host
            preds = {}
            for out in self.outputs:
                output = np.empty(out['shape'], dtype=out['dtype'])
                cuda.memcpy_dtoh_async(output, out['allocation'])
                preds[out['name']] = output
            # print(f"推理耗时：{(time.time() - infer_time) * 1000:.2f}ms")

            # 7.等待所有的cuda异步操作完成（重要）
            # stream.synchronize()

            # 8.后处理获取坐标结果
            coords = self.pred2coords_numpy(preds)
            return coords

        finally:
            # 2.无论是否报错，确保 pop cuda context
            self.cuda_context.pop()

    def forward(self, img):
        im0 = img.copy()
        input_data = self.preprocess(img)

        infer_time = time.time()
        preds = {}
        # 将预处理后的数据拷贝到GPU输入显存
        cuda.memcpy_htod(self.inputs[0]['allocation'], input_data)
        # 执行推理 - 使用TensorRT 10.x的新API
        a = self.context.execute_async_v3(stream_handle=0)
        # 从GPU输出显存拷贝回CPU
        for out in self.outputs:
            output = np.empty(out['shape'], dtype=out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            # preds[out['name']] = torch.tensor(output) # 转换成torch tensor(可选）
            preds[out['name']] = output
        # print(f"推理耗时：{(time.time() - infer_time) * 1000:.2f}ms")

        coords = self.pred2coords_numpy(preds)

        im0 = postprocess_coords_with_draw(im0, coords)

        return im0

    # def __del__(self):
    #     if hasattr(self, "context"):
    #         self.context.pop()  # 释放上下文资源

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/culane_res34.py', help='path to config file', type=str)
    parser.add_argument('--engine_path', default='weights/culane_res34.engine', help='path to engine file', type=str)
    parser.add_argument('--video_path', default='videos/example.mp4', help='path to video file', type=str)
    parser.add_argument('--ori_size', default=(1600, 320), help='size of original frame', type=tuple)
    return parser.parse_args()


class InferenceThread(threading.Thread):
    def __init__(self, net, input_queue, output_queue):
        super().__init__()
        self.net = net                      # UFLDv2 推理实例
        self.input_queue = input_queue      # 接收帧
        self.output_queue = output_queue    # 推理结果
        self.running = True

    def run(self):
        # 每个线程必须 push 自己的 CUDA context！
        # self.net.cuda_context.push()

        while self.running:
            if not self.input_queue.empty():
                img = self.input_queue.get()
                coords = self.net.get_coordinates_only(img)
                self.output_queue.put(coords)

        # self.net.cuda_context.pop()

    def stop(self):
        self.running = False

if __name__ == "__main__":
    # 手动设置参数
    engine_path = "./weights/culane_res34.engine"
    config_path = "./configs/culane_res34.py"
    video_path = r"D:\projects\forward-collision-warning-system\video\test-1080p.mp4"
    ori_size = (1600, 320)

    # 实例化模型
    # cap = cv2.VideoCapture(video_path)
    # is_net = UFLDv2(engine_path, config_path, ori_size)
    # count = 0
    # while True:
    #     s = time.time()
    #     success, frame = cap.read()
    #     img = cv2.resize(frame, (1600, 903))
    #     img = img[583:903, :, :]
    #     coords = is_net.get_coordinates_only(img)
    #     # print(coords)
    #     # result_img = is_net.forward(img)
    #     # result_img = cv2.resize(result_img, (480, 270))
    #     # cv2.imshow("result", result_img)
    #     # cv2.waitKey(1)  # 加个等待，防止窗口卡死
    #     # count += 1
    # 
    # 
    #     # print(f"推理一张图像耗时：{(time.time() - s) * 1000:.2f}ms")
    #     # if count > 10:
    #     #     break
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # 多线程
    from queue import Queue

    cap = cv2.VideoCapture(video_path)
    # engine_model = load_engine(engine_path)
    is_net = UFLDv2(engine_path, config_path, ori_size)

    # 队列用于线程间通信
    input_queue = Queue()
    output_queue = Queue()

    # 启动推理线程
    infer_thread = InferenceThread(is_net, input_queue, output_queue)
    infer_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (1600, 903))
        img = img[583:903, :, :]

        # 放入队列
        input_queue.put(img)

        # 读取结果（非阻塞检查）
        if not output_queue.empty():
            coords = output_queue.get()
            print(coords)

    cap.release()
    infer_thread.stop()
    infer_thread.join()