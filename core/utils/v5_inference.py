import cv2
import torch
from core.utils.script import infer_script


def run_inference(image_bgr, model, manual=False, show=False):
    try:
        if manual:# 手动推理
            # 预处理
            img_t, r_pad = infer_script.preprocess(image_bgr)

            # 前向推理（返回 raw output）
            with torch.no_grad():
                # 传入为 (1 c h w)格式的数据
                pred = model(img_t)

            # 后处理
            results = infer_script.postprocess(pred)

            # 还原坐标到原图
            detections = infer_script.scale_boxes(results, img_t, image, r_pad)
            detections = detections.cpu().numpy()
        else:
            # 自动推理（YOLOv5 封装）
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                # 传入为 (h w c) 格式的rgb图像
                results = model(image_rgb)

            # 取出 xyxy 格式结果
            pred = results.xyxy[0]  # shape: [num_boxes, 6] -> [x1, y1, x2, y2, conf, cls]
            detections = pred.cpu().numpy()

        return detections

        # if show:
        #     img = infer_script.visualize(image, detections)
        #     cv2.imshow('show', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # else:
        #     mode = "手动推理" if manual else "自动推理"
        #     print(mode + "结果：\n" + str(detections))

    except Exception as e:
        print(e)




if __name__ == '__main__':
    image = cv2.imread(r'D:\projects\carport_recognizer\images\2.jpg')
    # 加载本地的yolov5模型
    weights = torch.hub.load('ultralytics/yolov5', 'custom', path=r'D:\projects\RM\fcw\models\yolov5\car_detector.pt')  # 注意 source='local'
    # 加载远程的yolov5s模型
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    weights.eval()
    # 推理全流程
    run_inference(image, weights, manual=True, show=True)