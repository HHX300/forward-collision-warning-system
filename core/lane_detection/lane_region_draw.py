import cv2
import numpy as np
import time

def postprocess_coords_with_draw(image, coords):
    pp_coords_time = time.time()
    lane_nums = len(coords)
    # print(f"检测到的车道线数量为：{lane_nums}条")

    # 如果车道线的数量 小于 2，则直接返回原图
    if lane_nums < 2:
        return image, [], []

    left_lane = []
    right_lane = []
    idxToAverageX = {}
    lane_y_all = []
    for idx, lane in enumerate(coords):
        lane = np.array(lane)  # 转换为numpy
        average_x = int(np.mean(lane[:, 0])) # 计算所有x值的平均值
        average_y = int(np.mean(lane[:, 1])) # 计算所有y值的平均值
        idxToAverageX[idx] = average_x
        lane_y = lane[:, 1] # 车道线的y值
        lane_y_all.append(lane_y)

    # 如果车道线的数量 等于 2
    if lane_nums == 2:
        min_key = min(idxToAverageX, key=idxToAverageX.get)
        left_lane = coords[min_key] # 左车道线
        right_lane = coords[[k for k in idxToAverageX if k != min_key][0]] # 右车道线

    # 如果车道线的数量 等于 3
    elif lane_nums == 3:
        lane_jaccard = {}
        for i in range(lane_nums - 1):
            list1 = lane_y_all[i]
            for j in range(i + 1, lane_nums):
                list2 = lane_y_all[j]
                intersection = len(np.intersect1d(list1, list2))
                union = len(np.union1d(list1, list2))
                jaccard = round(intersection / union, 2)
                # print(f"Jaccard 相似度: {jaccard:.2f}")
                lane_jaccard[(i, j)] = jaccard
        max_key = max(lane_jaccard.keys(), key=lambda k: lane_jaccard[k])
        # 判断左右车道线
        idx0 = max_key[0]
        idx1 = max_key[1]
        # 赋值给我需要的两条线
        if idxToAverageX[idx0] < idxToAverageX[idx1]:
            left_lane = coords[idx0] # idx0为左车道线
            right_lane = coords[idx1]
        else:
            left_lane = coords[idx1] # idx1为右车道线
            right_lane = coords[idx0]

    # 如果车道线的数量 等于 4
    else:
        left_lane = coords[0]
        right_lane = coords[1]
        # cv2.waitKey(0)
        pass

    # 绘制特效
    image, processed_left_lane, processed_right_lane = draw_ui(image, left_lane, right_lane)
    # print(f"绘制车道的特效耗时：{(time.time() - pp_coords_time):.2f}s")
    return image, processed_left_lane, processed_right_lane



def draw_ui(img0, left_lane, right_lane):
    # 绘制图像
    # img0 = img.copy()
    # 1.绘制出坐标点
    for index, coord in enumerate(left_lane):
        if index == 0:
            cv2.putText(img0, "left", coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(img0, coord, 2, (0, 255, 0), -1)  # green

    for index, coord in enumerate(right_lane):
        if index == 0:
            cv2.putText(img0, "right", coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(img0, coord, 2, (0, 0, 255), -1)  # red

    # 方法二：绘制出坐标点
    # if len(left_lane) > 1:
    #     # 绘制左右车道线信息
    #     cv2.putText(img0, "left", left_lane[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
    #     # 批量模拟画点
    #     cv2.polylines(img0, [np.array(left_lane, dtype=np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA) # green
    #     pass
    # if len(right_lane) > 1:
    #     # 绘制左右车道线信息
    #     cv2.putText(img0, "right", right_lane[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
    #     # 批量模拟画点
    #     cv2.polylines(img0, [np.array(right_lane, dtype=np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)  # red
    #     pass

    # 2.绘制车道区域特效
    if len(left_lane) > 0 and len(right_lane) > 0:
        # 获取图像尺寸，用于边界限制
        img0_height, img0_width = img0.shape[:2]
        
        # 确保两条车道线有相同数量的点
        min_points = min(len(left_lane), len(right_lane))
        
        # 处理车道线点，确保不超出图像边界
        processed_left_lane = []
        processed_right_lane = []

        # 筛选出长度最小的列表
        if len(left_lane) < len(right_lane):
            right_lane_array = np.array(right_lane) # 转换为（n，2）数组
            # 以left_lane 为标准
            for left_point in left_lane:
                # 限制点在图像范围内
                left_x = max(0, min(left_point[0], img0_width))
                left_y = max(0, min(left_point[1], img0_height))
                right_y = max(0, min(left_point[1], img0_height))

                # 找到所有y等于left_y的位置
                matches = right_lane_array[right_lane_array[:, 1] == left_y]
                if len(matches) == 0:
                    continue
                right_x = max(0, min(int(matches[0][0]), img0_width))

                processed_left_lane.append((left_x, left_y))
                processed_right_lane.append((right_x, right_y))
        else:
            left_lane_array = np.array(left_lane)  # 转换为（n，2）数组
            # 以right_lane 为标准
            for right_lane in right_lane:
                # 限制点在图像范围内
                right_x = max(0, min(right_lane[0], img0_width))
                right_y = max(0, min(right_lane[1], img0_height))
                left_y = max(0, min(right_lane[1], img0_height))

                # 找到所有y等于right_y的位置
                matches = left_lane_array[left_lane_array[:, 1] == right_y]
                if len(matches) == 0:
                    continue
                left_x = max(0, min(int(matches[0][0]), img0_width))

                processed_left_lane.append((left_x, left_y))
                processed_right_lane.append((right_x, right_y))

        # 3.创建车道区域的多边形
        if len(processed_left_lane) >= 2 and len(processed_right_lane) >= 2:
            # 构建车道区域的闭合多边形
            lane_polygon = []

            # 添加左车道线的点（从上到下）
            for point in processed_left_lane:
                lane_polygon.append(point)

            # 添加右车道线的点（从下到上，形成闭合区域）
            for point in reversed(processed_right_lane):
                lane_polygon.append(point)

            # 转换为numpy数组
            lane_polygon = np.array(lane_polygon, dtype=np.int32)

            # 4.绘制多层虚影特效
            # 创建多个覆盖层实现渐变虚影效果
            overlay1 = img0.copy()
            overlay2 = img0.copy()
            overlay3 = img0.copy()

            # 第一层：最深的虚影（蓝绿色）
            cv2.fillPoly(overlay1, [lane_polygon], (255, 200, 0))  # 青色
            cv2.addWeighted(overlay1, 0.15, img0, 0.85, 0, img0)

            # 第二层：中等虚影（黄绿色）
            cv2.fillPoly(overlay2, [lane_polygon], (0, 255, 200))  # 黄绿色
            cv2.addWeighted(overlay2, 0.1, img0, 0.9, 0, img0)

            # 第三层：最浅的虚影（淡黄色）
            cv2.fillPoly(overlay3, [lane_polygon], (0, 255, 255))  # 黄色
            cv2.addWeighted(overlay3, 0.08, img0, 0.92, 0, img0)

            # 5.绘制车道边界线增强效果
            # 绘制左车道线的增强效果
            if len(processed_left_lane) >= 2:
                for i in range(len(processed_left_lane) - 1):
                    cv2.line(img0, processed_left_lane[i], processed_left_lane[i + 1], (0, 255, 255), 3)

            # 绘制右车道线的增强效果
            if len(processed_right_lane) >= 2:
                for i in range(len(processed_right_lane) - 1):
                    cv2.line(img0, processed_right_lane[i], processed_right_lane[i + 1], (0, 255, 255), 3)

    return img0, processed_left_lane, processed_right_lane


if __name__ == '__main__':
    pass
