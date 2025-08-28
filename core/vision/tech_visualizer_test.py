#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科技感可视化器测试程序
用于测试TechHUDVisualizer的功能
"""

import cv2
import numpy as np
from tech_visualizer import TechHUDVisualizer

def create_test_image():
    """创建测试图像"""
    # # 创建一个640x480的测试图像
    # img = np.zeros((480, 640, 3), dtype=np.uint8)
    #
    # # 添加一些背景纹理
    # img[:] = (30, 30, 30)  # 深灰色背景
    #
    # # 添加一些道路线条
    # cv2.line(img, (0, 400), (640, 400), (100, 100, 100), 2)
    # cv2.line(img, (320, 480), (320, 300), (150, 150, 150), 1)
    img = cv2.imread('images/img.jpg')
    return img

def create_test_detections():
    """创建测试检测数据"""
    # detections = [
    #     {
    #         "box": (100, 200, 200, 300),
    #         "label": "car",
    #         "score": 0.85,
    #         "distance": 25.5,
    #         "class_id": 0
    #     },
    #     {
    #         "box": (300, 180, 420, 280),
    #         "label": "truck",
    #         "score": 0.92,
    #         "distance": 12.3,
    #         "class_id": 1
    #     },
    #     {
    #         "box": (450, 220, 550, 320),
    #         "label": "car",
    #         "score": 0.78,
    #         "distance": 35.8,
    #         "class_id": 0
    #     }
    # ]
    detections = [
         {'box': (0, 784, 480, 1078), 'class_id': 3.0, 'distance': 3.4157142540713203, 'label': 'car',
          'score': 0.962771475315094},
         {'box': (1219, 751, 1567, 1015), 'class_id': 3.0, 'distance': 4.3410333258213445, 'label': 'car',
          'score': 0.9521641135215759},
         {'box': (819, 780, 1283, 1079), 'class_id': 3.0, 'distance': 3.339747239513685, 'label': 'car',
          'score': 0.9451051950454712},
         {'box': (427, 816, 665, 965), 'class_id': 3.0, 'distance': 6.122748668595744, 'label': 'car',
          'score': 0.9346858859062195},
         {'box': (622, 813, 751, 920), 'class_id': 3.0, 'distance': 9.410980245278436, 'label': 'car',
          'score': 0.9299301505088806},
         {'box': (733, 829, 818, 898), 'class_id': 3.0, 'distance': 12.948916672906488, 'label': 'car',
          'score': 0.9082401394844055},
         {'box': (1177, 805, 1249, 872), 'class_id': 3.0, 'distance': 22.286873884430893, 'label': 'car',
          'score': 0.7130081653594971},
         {'box': (802, 835, 846, 882), 'class_id': 3.0, 'distance': 18.051135351964344, 'label': 'car',
          'score': 0.7085347175598145},
         {'box': (414, 827, 478, 877), 'class_id': 3.0, 'distance': 22.326540880662154, 'label': 'car',
          'score': 0.6608169674873352}
    ]
    return detections

def main():
    """主测试函数"""
    print("🎨 开始测试科技感可视化器...")
    
    try:
        # 创建可视化器
        visualizer = TechHUDVisualizer()
        print("✅ 科技感可视化器创建成功")
        
        # 创建测试数据
        test_img = create_test_image()
        test_detections = create_test_detections()
        
        print(f"📊 测试图像尺寸: {test_img.shape}")
        print(f"🚗 测试检测数量: {len(test_detections)}")
        
        # 进行可视化渲染
        result_img = visualizer.visualize_detections(
            img=test_img,
            detections=test_detections,
            frame_id=1,
            safe_distance=15.0
        )
        
        print("🎯 可视化渲染完成")
        
        # 保存结果图像
        output_path = "images/test_tech_visual_result.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"💾 结果已保存到: {output_path}")
        
        # 显示图像（如果有显示器）
        try:
            # result_img = cv2.resize(result_img, (960,540))
            cv2.imshow("科技感可视化测试", result_img)
            print("🖼️ 图像显示窗口已打开，按任意键关闭")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except ValueError as e:
            print("⚠️ 无法显示图像窗口（可能是无头环境）")
            print(f'error: {e}')
        print("✅ 科技感可视化器测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()