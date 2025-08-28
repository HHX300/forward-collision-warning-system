import cv2
import torch
import torchvision
import time
import numpy as np
# from script.logconfig import setup_logging

# region 配置准备
# logger = setup_logging()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# endregion

# region 预处理
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


def normalize_image(img_tensor, method='simple'):
    """
    图像归一化处理函数
    参数:
        img_tensor: torch.Tensor, shape: (C, H, W)
        method: str, 'simple' 或 'imagenet'
    返回:
        归一化后的 img_tensor
    """
    img_tensor = img_tensor.float()  # 转为 float 类型

    if method == 'simple':
        # 简单归一化：将像素值从 0~255 映射到 0~1 (适合 YOLO 等轻量模型)
        img_tensor /= 255.0

    elif method == 'imagenet':
        # 先缩放，再减均值除以标准差（适合 ImageNet、ResNet、ViT 等）
        img_tensor /= 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

    else:
        raise ValueError(f"不支持的归一化方式: {method}")

    return img_tensor


def preprocess(im):
    start = time.time()
    # 将图片转换为rgb格式
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # 进行letterbox处理，保持原图长宽比
    im_resized, ratio, (dw, dh) = letterbox(im_rgb, new_shape=(640, 640), auto=False, scaleup=True)
    # -- numpy -> tensor
    img_tensor = torch.from_numpy(im_resized).float()  # (H,W,C)
    img_tensor = img_tensor.to(device)  # 放在GPU or cpu
    img_tensor = normalize_image(img_tensor, method='simple')
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # 如果漏了 batch 维度
    ratio_pad = [ratio, (dw, dh)]
    return img_tensor, ratio_pad


# endregion


# region 后处理
def xywh2xyxy(x):
    # 将 nx4 的 box 从 [x, y, w, h] 转换为 [x1, y1, x2, y2]
    # 其中：xy 表示中心点坐标；xy1 表示左上角，xy2 表示右下角
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)  # 复制一份输入，避免原始数据被修改

    # 计算左上角 x1 = x - w/2
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    # 计算左上角 y1 = y - h/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    # 计算右下角 x2 = x + w/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    # 计算右下角 y2 = y + h/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y  # 返回 [x1, y1, x2, y2] 形式的结果


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
        计算两个 bbox 集合之间的 IoU（Intersection over Union）

        参数：
            box1: Tensor[N, 4]，N 个框，格式为 (x1, y1, x2, y2)
            box2: Tensor[M, 4]，M 个框，格式为 (x1, y1, x2, y2)
        返回：
            Tensor[N, M]，每个 box1[i] 与 box2[j] 之间的 IoU
        """

    # 将两个 box 集合分别扩展出维度，使其可以广播成 N×M 的组合
    # 例如 box1=[N,4] → [N,1,4]，box2=[M,4] → [1,M,4]
    # 然后 split/chunk 成：a1, a2 = box1 的左上角和右下角，b1, b2 同理
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)

    # 计算交集区域宽高，inter.shape = [N, M]
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # torch.min(a2, b2)：右下角取最小值；torch.max(a1, b1)：左上角取最大值
    # .clamp(0)：交集宽高为负时设为0；.prod(2)：计算 w*h 得到面积

    # 计算每个框的面积
    area1 = (a2 - a1).prod(2)  # [N,1]
    area2 = (b2 - b1).prod(2)  # [1,M]

    # 计算 IoU：交集 / 并集（加 eps 防止除以0）
    return inter / (area1 + area2 - inter + eps)


def non_max_suppression(
        prediction,  # 模型预测输出
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # IOU（重叠）阈值
        classes=None,  # 只保留特定类别（可选）
        agnostic=False,  # 是否类别无关（True时不同类别也会互相抑制）
        multi_label=False,  # 是否允许一个框属于多个类别
        labels=(),  # 用于自动标注的数据增强标签（仅用于训练或验证）
        max_det=300,  # 每张图最大保留的检测框数
        nm=0,  # 掩码个数（如使用segment分割时使用）
):
    """
       对模型推理结果进行非极大值抑制（NMS），过滤重叠框。
       返回：
           List，每张图像一个Tensor，shape为 (n, 6)，内容为 [x1, y1, x2, y2, confidence, class]
       """

    # 参数合法性检查
    assert 0 <= conf_thres <= 1, f'无效的置信度阈值 {conf_thres}, 应为0~1之间'
    assert 0 <= iou_thres <= 1, f'无效的IOU阈值 {iou_thres}, 应为0~1之间'
    # 如果 prediction 是 (推理输出, 损失输出) 的元组，只保留推理部分
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    device = prediction.device
    mps = 'mps' in device.type  # 是否是 MacOS MPS 后端（目前不完全支持）
    if mps:  # 如果是 MPS，需要将数据转移到 CPU 上执行 NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # 类别数（除去 box(4) + objectness(1) + mask(nm)）
    xc = prediction[..., 4] > conf_thres  # 置信度大于阈值的候选框布尔索引

    # 参数设置
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # 防止框之间在做类别偏移时冲突（通过添加 max_wh * cls）
    max_nms = 30000  # NMS 最大处理框数量
    time_limit = 0.5 + 0.05 * bs  # 超时时间 (seconds to quit after)
    redundant = True  # 是否运行冗余检测(用于Merge-NMS)
    multi_label &= nc > 1  # 如果类别数 > 1 且启用多标签标注
    merge = False  # 是否使用 merge-NMS

    t = time.time()
    mi = 5 + nc  # 掩码起始索引 (box+obj_conf + cls_conf)

    # 初始化输出列表，每张图初始为空
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):  # 遍历每张图像的预测结果
        # 只保留满足置信度要求的框
        x = x[xc[xi]]  # confidence

        # 如果提供了Label（比如自动标注阶段），将其合并到x中
        if labels and len(labels[xi]):
            lb = labels[xi]  # 形如 [cls,  x1, y1, x2, y2]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # 置信度
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # 如果当前图片没有合法框，跳过当前图片
        if not x.shape[0]:
            continue

        # 计算总置信度：obj_conf * cls_conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # 坐标格式转换：从 center_x, center_y, width, height to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]  # 只在 instance segmentation 模型中有用

        # 生成检测矩阵：包括坐标、置信度、类别、掩码（可选）
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            # 仅保留每个框中置信度最大的类别
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # 根据类别进行筛选（如只保留目标类别）
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # 框数量为0则跳过
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        # 根据置信度降序排序，最多保留 max_nms 个框
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # 为 NMS 生成 offset，使不同类别框之间不会互相影响（除非 agnostic=True）
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        # 进行 NMS
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # 限制最大检测框数

        # 如果启用 Merge-NMS，对重叠框进行加权平均合并
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        # 保存当前图像的最终结果
        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)

        # 如果耗时超过限制，提前终止
        if (time.time() - t) > time_limit:
            # logger.warning(f'⚠️ 警告：NMS耗时超过 {time_limit:.3f}s，已中止处理')
            print(f'⚠️ 警告：NMS耗时超过 {time_limit:.3f}s，已中止处理')
            break  # time limit exceeded

    return output


def postprocess(pred):  # 模型预测输出
    # 转换为张量
    if isinstance(pred, np.ndarray):
        # 创建新的张量，避免共享内存
        pred = torch.from_numpy(pred.copy()).to(device)
    # 后处理（NMS）
    pred = non_max_suppression(pred, 0.25, 0.45)
    return pred


# endregion


# region 后处理后还原坐标
def clip_boxes(boxes, shape):
    # 将边界框（xyxy 格式）裁剪到图像范围内，避免越界
    # 参数:
    #   boxes: 包含边界框的张量或数组，格式为 [x1, y1, x2, y2]
    #   shape: 图像的尺寸 (height, width)
    if isinstance(boxes, torch.Tensor):  # 如果是 PyTorch 张量（Tensor），逐个裁剪更快
        boxes[..., 0].clamp_(0, shape[1])  # x1：限制在 [0, 图像宽度]
        boxes[..., 1].clamp_(0, shape[0])  # y1：限制在 [0, 图像高度]
        boxes[..., 2].clamp_(0, shape[1])  # x2：限制在 [0, 图像宽度]
        boxes[..., 3].clamp_(0, shape[0])  # y2：限制在 [0, 图像高度]
    else:  # 如果是 NumPy 数组（clip 效率更高）
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # 裁剪 x1 和 x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # 裁剪 y1 和 y2


def scale_box(img1_shape, boxes, img0_shape, ratio_pad=None):
    """

    :param img1_shape: 模型输入图像的尺寸（如预处理后 640x640），格式为 (h, w)
    :param boxes: 边界框数组，通常为 N x 4 或更高维，格式为 [x1, y1, x2, y2]（左上角、右下角坐标）
    :param img0_shape: 原始图像的尺寸，格式同上
    :param ratio_pad: 可选，预处理时的缩放比例和填充尺寸（若已知，可直接传入）
    :return:
    """
    # 将边界框（格式为 xyxy）从 img1_shape 的坐标系缩放/变换到 img0_shape 的坐标系中。
    if ratio_pad is None:  # 若未提供 ratio_pad，则从 img1_shape 与 img0_shape 推算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]  # 缩放比例
        pad = ratio_pad[1]  # 填充 (pad_w, pad_h)

    boxes[..., [0, 2]] -= pad[0]  # 减去 x 向的 padding
    boxes[..., [1, 3]] -= pad[1]  # 减去 y 向的 padding
    boxes[..., :4] /= gain  # 缩放回原始尺寸
    clip_boxes(boxes, img0_shape)

    return boxes


def scale_boxes(results, resized_img, img, radio_pad):
    # 缩放坐标到原图
    scaled_results = []
    for result in results:
        if result is not None and len(result):
            # 创建结果副本，避免修改原始数据
            scaled_result = result.clone() if isinstance(result, torch.Tensor) else result.copy()
            scaled_result[:, :4] = scale_box(resized_img.shape[2:], scaled_result[:, :4], img.shape[:2],
                                             ratio_pad=radio_pad).round()
            scaled_results.append(scaled_result)
        else:
            scaled_results.append(result)
    return scaled_results  # 返回所有处理后的结果


# endregion


# region 可视化结果
def yolo_visualize(img, results):
    # 检查results是否为空或None
    if not results:
        return img

    for result in results:
        if result is None:
            continue

        if isinstance(result, torch.Tensor):
            # 确保tensor在CPU上并转换为numpy
            result_np = result.detach().cpu().numpy()

            # 检查结果维度
            if len(result_np.shape) == 1 and len(result_np) >= 6:
                x1, y1, x2, y2 = result_np[:4].astype(int)
                conf = float(result_np[4])
                cls = int(result_np[5])
            else:
                print(f"警告：检测结果格式不正确，跳过: {result_np.shape}")
                continue

        elif isinstance(result, (np.ndarray, list)):
            # 处理numpy数组或列表
            if len(result) >= 6:
                x1, y1, x2, y2 = map(int, result[:4])
                conf = float(result[4])
                cls = int(result[5])
            else:
                print(f"警告：检测结果长度不足，跳过: {len(result)}")
                continue
        else:
            print(f"警告：未知的结果类型，跳过: {type(result)}")
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
            cv2.putText(img, f'cls:{cls}', (int(x1), int(label_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 置信度标签位置
            conf_x = max(x1, x2 - 100)  # 避免标签超出图像右边界
            cv2.putText(img, f'conf:{conf:.2f}', (int(conf_x), int(label_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except Exception as e:
            print(f"绘制检测框时出错: {e}")
            print(f"坐标: ({x1}, {y1}, {x2}, {y2}), 置信度: {conf}, 类别: {cls}")
            continue

    return img


# endregion


if __name__ == '__main__':
    image = cv2.imread('./images/1.jpg')
    img_t, r_pad = preprocess(image)  # 前处理

    # ped = model(img_t) # 模型预测输出
    ped = 1
    ped = postprocess(ped)  # 后处理（NMS）
    ped = scale_boxes(ped, img_t, image, r_pad)  # 后处理（还原坐标）