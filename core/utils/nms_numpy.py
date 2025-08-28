import numpy as np
import time


def xywh2xyxy(x):
    """将坐标从 (center_x, center_y, width, height) 转换为 (x1, y1, x2, y2)"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def box_iou_numpy(box1, box2):
    """计算两组框之间的IoU"""

    # box1: (N, 4), box2: (M, 4)
    # 返回: (N, M) IoU矩阵

    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    # 计算交集
    inter_x1 = np.maximum(box1[:, None, 0], box2[:, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[:, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[:, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[:, 3])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    # 计算并集
    union_area = area1[:, None] + area2 - inter_area

    return inter_area / union_area


def nms_numpy(boxes, scores, iou_threshold):
    """numpy实现的NMS算法"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    # 按分数降序排序
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        # 选择分数最高的框
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # 计算当前框与其他框的IoU
        current_box = boxes[current:current + 1]
        other_boxes = boxes[indices[1:]]

        ious = box_iou_numpy(current_box, other_boxes)[0]

        # 保留IoU小于阈值的框
        indices = indices[1:][ious <= iou_threshold]

    return np.array(keep, dtype=np.int32)


def non_max_suppression_numpy(
        prediction,  # 模型预测输出 (numpy array)
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
           List，每张图像一个numpy array，shape为 (n, 6)，内容为 [x1, y1, x2, y2, confidence, class]
       """

    # 参数合法性检查
    assert 0 <= conf_thres <= 1, f'无效的置信度阈值 {conf_thres}, 应为0~1之间'
    assert 0 <= iou_thres <= 1, f'无效的IOU阈值 {iou_thres}, 应为0~1之间'

    # 如果 prediction 是 (推理输出, 损失输出) 的元组，只保留推理部分
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    # 确保是numpy数组
    if not isinstance(prediction, np.ndarray):
        prediction = np.array(prediction)

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # 类别数（除去 box(4) + objectness(1) + mask(nm)）
    xc = prediction[..., 4] > conf_thres  # 置信度大于阈值的候选框布尔索引

    # 参数设置
    max_wh = 7680  # 防止框之间在做类别偏移时冲突（通过添加 max_wh * cls）
    max_nms = 30000  # NMS 最大处理框数量
    time_limit = 0.5 + 0.05 * bs  # 超时时间 (seconds to quit after)
    redundant = True  # 是否运行冗余检测(用于Merge-NMS)
    multi_label &= nc > 1  # 如果类别数 > 1 且启用多标签标注
    merge = False  # 是否使用 merge-NMS

    t = time.time()
    mi = 5 + nc  # 掩码起始索引 (box+obj_conf + cls_conf)

    # 初始化输出列表，每张图初始为空
    output = [np.zeros((0, 6 + nm), dtype=np.float32)] * bs

    for xi, x in enumerate(prediction):  # 遍历每张图像的预测结果
        # 只保留满足置信度要求的框
        x = x[xc[xi]]  # confidence

        # 如果提供了Label（比如自动标注阶段），将其合并到x中
        if labels and len(labels[xi]):
            lb = labels[xi]  # 形如 [cls,  x1, y1, x2, y2]
            v = np.zeros((len(lb), nc + nm + 5), dtype=np.float32)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # 置信度
            for i, cls_idx in enumerate(lb[:, 0].astype(int)):
                v[i, cls_idx + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

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
            i, j = np.where(x[:, 5:mi] > conf_thres)
            x = np.concatenate((
                box[i],
                x[i, 5 + j][:, None],
                j[:, None].astype(np.float32),
                mask[i]
            ), 1)
        else:
            # 仅保留每个框中置信度最大的类别
            conf = np.max(x[:, 5:mi], axis=1, keepdims=True)
            j = np.argmax(x[:, 5:mi], axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(np.float32), mask), 1)
            x = x[conf.flatten() > conf_thres]

        # 根据类别进行筛选（如只保留目标类别）
        if classes is not None:
            classes_array = np.array(classes)
            mask = np.isin(x[:, 5], classes_array)
            x = x[mask]

        # 框数量为0则跳过
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        # 根据置信度降序排序，最多保留 max_nms 个框
        sort_indices = np.argsort(x[:, 4])[::-1][:max_nms]
        x = x[sort_indices]

        # 为 NMS 生成 offset，使不同类别框之间不会互相影响（除非 agnostic=True）
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        # 进行 NMS
        i = nms_numpy(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # 限制最大检测框数

        # 如果启用 Merge-NMS，对重叠框进行加权平均合并
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou_numpy(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None, :]  # box weights
            weights_sum = weights.sum(1, keepdims=True)
            weights_sum[weights_sum == 0] = 1  # 避免除零
            x[i, :4] = np.dot(weights, x[:, :4]) / weights_sum  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        # 保存当前图像的最终结果
        output[xi] = x[i]

        # 如果耗时超过限制，提前终止
        if (time.time() - t) > time_limit:
            print(f'⚠️ 警告：NMS耗时超过 {time_limit:.3f}s，已中止处理')
            break  # time limit exceeded

    return output


# 使用示例
if __name__ == "__main__":
    # 创建测试数据 (batch_size=1, num_boxes=100, 85个特征)
    # 85 = 4(box) + 1(objectness) + 80(classes)
    test_prediction = np.random.rand(1, 100, 85).astype(np.float32)

    # 设置一些合理的值
    test_prediction[..., :4] *= 640  # 坐标范围
    test_prediction[..., 4] = np.random.rand(1, 100) * 0.8 + 0.1  # objectness
    test_prediction[..., 5:] = np.random.rand(1, 100, 80) * 0.5  # class scores

    # 运行NMS
    results = non_max_suppression_numpy(test_prediction)
    print(f"检测到 {len(results[0])} 个目标")
    if len(results[0]) > 0:
        print(f"结果形状: {results[0].shape}")
        print(f"前5个结果:\n{results[0][:5]}")