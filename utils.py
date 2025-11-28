# -*- coding: utf-8 -*-
"""
YOLOv8物品检测工具公共模块
包含公共常量、函数和工具类
"""

# COCO数据集80个类别名称映射
COCO_CATEGORY_NAMES = {
    0: 'person',  # 人
    1: 'bicycle',  # 自行车
    2: 'car',  # 汽车
    3: 'motorcycle',  # 摩托车
    4: 'airplane',  # 飞机
    5: 'bus',  # 公交车
    6: 'train',  # 火车
    7: 'truck',  # 卡车
    8: 'boat',  # 船
    9: 'traffic light',  # 红绿灯
    10: 'fire hydrant',  # 消防栓
    11: 'stop sign',  # 停车标志
    12: 'parking meter',  # 停车计时器
    13: 'bench',  # 长椅
    14: 'bird',  # 鸟
    15: 'cat',  # 猫
    16: 'dog',  # 狗
    17: 'horse',  # 马
    18: 'sheep',  # 羊
    19: 'cow',  # 牛
    20: 'elephant',  # 大象
    21: 'bear',  # 熊
    22: 'zebra',  # 斑马
    23: 'giraffe',  # 长颈鹿
    24: 'backpack',  # 背包
    25: 'umbrella',  # 雨伞
    26: 'handbag',  # 手提包
    27: 'tie',  # 领带
    28: 'suitcase',  # 行李箱
    29: 'frisbee',  # 飞盘
    30: 'skis',  # 滑雪板
    31: 'snowboard',  # 冲浪板
    32: 'sports ball',  # 球类
    33: 'kite',  # 风筝
    34: 'baseball bat',  # 棒球棒
    35: 'baseball glove',  # 棒球手套
    36: 'skateboard',  # 滑板
    37: 'surfboard',  # 冲浪板
    38: 'tennis racket',  # 网球拍
    39: 'bottle',  # 瓶子
    40: 'wine glass',  # 酒杯
    41: 'cup',  # 杯子
    42: 'fork',  # 叉子
    43: 'knife',  # 刀
    44: 'spoon',  # 勺子
    45: 'bowl',  # 碗
    46: 'banana',  # 香蕉
    47: 'apple',  # 苹果
    48: 'sandwich',  # 三明治
    49: 'orange',  # 橙子
    50: 'broccoli',  # 西兰花
    51: 'carrot',  # 胡萝卜
    52: 'hot dog',  # 热狗
    53: 'pizza',  # 披萨
    54: 'donut',  # 甜甜圈
    55: 'cake',  # 蛋糕
    56: 'chair',  # 椅子
    57: 'couch',  # 沙发
    58: 'potted plant',  # 盆栽
    59: 'bed',  # 床
    60: 'dining table',  # 餐桌
    61: 'toilet',  # 马桶
    62: 'tv',  # 电视
    63: 'laptop',  # 笔记本电脑
    64: 'mouse',  # 鼠标
    65: 'remote',  # 遥控器
    66: 'keyboard',  # 键盘
    67: 'cell phone',  # 手机
    68: 'microwave',  # 微波炉
    69: 'oven',  # 烤箱
    70: 'toaster',  # 烤面包机
    71: 'sink',  # 水槽
    72: 'refrigerator',  # 冰箱
    73: 'book',  # 书
    74: 'clock',  # 时钟
    75: 'vase',  # 花瓶
    76: 'scissors',  # 剪刀
    77: 'teddy bear',  # 泰迪熊
    78: 'hair drier',  # 吹风机
    79: 'toothbrush'  # 牙刷
}


def draw_detection_result(image, boxes, category_names, conf_threshold=0.5):
    """
    在图像上绘制检测结果
    
    Args:
        image: 原始图像
        boxes: 检测结果的边界框
        category_names: 类别名称映射
        conf_threshold: 置信度阈值
    
    Returns:
        绘制了检测结果的图像
    """
    import cv2
    
    annotated_image = image.copy()
    
    for box in boxes:
        # 获取边界框坐标
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 获取置信度和类别ID
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        
        # 只显示指定置信度以上的结果
        if conf < conf_threshold:
            continue
        
        # 只显示我们定义的类别
        if cls_id in category_names:
            # 获取类别名称
            cls_name = category_names[cls_id]
            
            # 绘制边界框
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制类别名称和置信度
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(annotated_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_image


def calculate_fps(start_time):
    """
    计算FPS
    
    Args:
        start_time: 开始时间
    
    Returns:
        fps: 每秒帧数
        inference_time: 推理时间
    """
    import time
    
    end_time = time.time()
    inference_time = end_time - start_time
    fps = 1 / inference_time if inference_time > 0 else 0
    
    return fps, inference_time
