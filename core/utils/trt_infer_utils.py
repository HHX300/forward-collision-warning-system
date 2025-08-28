import numpy as np
import os
import time
import cv2


#region 预处理
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    填充为yolo需要的标准尺寸
    :param im:
    :param new_shape:
    :param color:
    :param auto:
    :param scaleFill:
    :param scaleup:
    :param stride:
    :return:
    """
    # 在保持步长倍数限制的前提下调整并填充图像
    shape = im.shape[:2]  # 当前形状 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 比例率 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小图像大小，不要夸大图像大小 (为了更好的mAP验证)
        r = min(r, 1.0)

    # 计算填充量
    ratio = r, r  # 宽，高比率
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高填充量
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 宽高填充量
    elif scaleFill:  # 拉长
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽高比例

    dw /= 2  # 将填充区域分为两侧
    dh /= 2

    if shape[::-1] != new_unpad:  # 更改尺寸
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 增加边框
    return im, ratio, (dw, dh)


def preprocess(im):
    # 将图片转换为rgb格式
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # 进行letterbox处理，保持原图长宽比
    im_resized, ratio, (dw, dh) = letterbox(im_rgb, new_shape=(640, 640), auto=False, scaleup=True)

    # 转换为CHW格式并确保数据类型为float32
    image = np.ascontiguousarray(im_resized.transpose(2, 0, 1)).astype(np.float32)

    # 简单归一化：将像素值从 0~255 映射到 0~1 (适合 YOLO 等轻量模型)
    image /= 255.0
    # 添加batch维度
    image = np.expand_dims(image, axis=0)

    ratio_pad = [ratio, (dw, dh)]

    return image, ratio_pad



def visualize(img, results):
    # 检查results是否为空或None
    if not results:
        return img

    for boxes in results:
        if boxes is None:
            continue

        for box in boxes:
            if isinstance(box, (np.ndarray, list)):
                # 处理numpy数组或列表
                if len(box) >= 6:
                    x1, y1, x2, y2 = map(int, box[:4])
                    conf = float(box[4])
                    cls = int(box[5])
                else:
                    print(f"警告：检测结果长度不足，跳过: {len(box)}")
                    continue
            else:
                print(f"警告：未知的结果类型，跳过: {type(box)}")
                continue

            # 验证坐标有效性
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 >= x2 or y1 >= y2:
                print(f"警告：无效坐标，跳过: ({x1}, {y1}, {x2}, {y2})")
                continue

            # 确保坐标在图像范围内
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            try:
                # 画框 - 确保坐标是整数元组
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # 画标签
                label_y = max(y1 - 10, 15)  # 确保标签不会超出图像顶部
                cv2.putText(img, f'cls:{cls}', (int(x1), int(label_y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 置信度标签位置
                conf_x = max(x1, x2 - 100)  # 避免标签超出图像右边界
                cv2.putText(img, f'conf:{conf:.2f}', (int(conf_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"绘制检测框时出错: {e}")
                print(f"坐标: ({x1}, {y1}, {x2}, {y2}), 置信度: {conf}, 类别: {cls}")
                continue

    return img