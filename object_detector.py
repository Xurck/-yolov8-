import cv2
from ultralytics import YOLO
import argparse
import time
from utils import COCO_CATEGORY_NAMES, draw_detection_result, calculate_fps

def load_model(model_path):
    """加载YOLOv8模型"""
    try:
        print(f"正在加载模型: {model_path}")
        print("如果是首次运行，模型将自动从Ultralytics服务器下载...")
        model = YOLO(model_path)
        print(f"✓ 成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        print("请确保网络连接正常，或者手动下载模型文件后重试。")
        exit(1)

def detect_image(model, image_path, conf=0.5, imgsz=320):
    """图片检测"""
    print(f"\n=== 图片检测模式 ===")
    print(f"检测图片: {image_path}")
    print(f"置信度阈值: {conf}")
    print(f"推理尺寸: {imgsz}")
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ 无法读取图片: {image_path}")
        return
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行推理
    results = model(img, conf=conf, imgsz=imgsz)
    
    # 计算FPS
    fps, inference_time = calculate_fps(start_time)
    
    # 获取检测结果
    boxes = results[0].boxes
    
    # 绘制检测结果
    annotated_img = draw_detection_result(img, boxes, COCO_CATEGORY_NAMES, conf)
    
    # 添加FPS信息
    cv2.putText(annotated_img, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    print(f"✓ 检测完成")
    print(f"推理时间: {inference_time:.3f}秒")
    print(f"FPS: {fps:.1f}")
    print(f"检测到 {len(boxes)} 个目标")
    
    # 显示结果
    cv2.imshow("YOLOv8物品检测结果", annotated_img)
    print("\n按 'q' 退出，按任意其他键继续...")
    
    # 等待按键
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit(0)
    
    cv2.destroyAllWindows()

def detect_camera(model, camera_id=0, conf=0.5, imgsz=320):
    """摄像头实时检测"""
    print(f"\n=== 摄像头实时检测模式 ===")
    print(f"摄像头ID: {camera_id}")
    print(f"置信度阈值: {conf}")
    print(f"推理尺寸: {imgsz}")
    print("按 'q' 退出")
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"✗ 无法打开摄像头: {camera_id}")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    fps_history = []
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("✗ 无法读取摄像头帧")
            break
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行推理
        results = model(frame, conf=conf, imgsz=imgsz)
        
        # 计算推理时间
        inference_time = time.time() - start_time
        fps = 1 / inference_time
        fps_history.append(fps)
        
        # 保持FPS历史长度为10
        if len(fps_history) > 10:
            fps_history.pop(0)
        
        # 计算平均FPS
        avg_fps = sum(fps_history) / len(fps_history)
        
        # 复制原始帧用于自定义可视化
        annotated_frame = frame.copy()
        
        # 获取检测结果
        boxes = results[0].boxes
        
        # 自定义可视化
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 获取置信度和类别ID
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            
            # 只显示COCO数据集的80个物品类别
            if cls_id in CATEGORY_NAMES:
                # 获取类别名称
                cls_name = CATEGORY_NAMES[cls_id]
                
                # 绘制边界框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制类别名称和置信度
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 添加FPS信息
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow("YOLOv8物品实时检测", annotated_frame)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("✓ 摄像头检测已退出")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8物品检测智能识别工具')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help='模型路径，默认使用yolov8n预训练模型')
    parser.add_argument('--mode', type=str, default='camera', 
                        choices=['image', 'camera'], 
                        help='运行模式: image(图片检测) 或 camera(摄像头检测)')
    parser.add_argument('--source', type=str, default='0', 
                        help='检测源: 图片路径或摄像头ID')
    parser.add_argument('--conf', type=float, default=0.5, 
                        help='置信度阈值，默认0.5')
    parser.add_argument('--imgsz', type=int, default=320, 
                        help='推理尺寸，默认320')
    
    args = parser.parse_args()
    
    print("========================================")
    print("        YOLOv8物品检测智能识别工具        ")
    print("========================================")
    print(f"运行模式: {args.mode}")
    print(f"模型路径: {args.model}")
    print(f"检测源: {args.source}")
    print(f"置信度阈值: {args.conf}")
    print(f"推理尺寸: {args.imgsz}")
    print("========================================")
    
    # 加载模型
    model = load_model(args.model)
    
    # 根据模式执行检测
    if args.mode == 'image':
        detect_image(model, args.source, args.conf, args.imgsz)
    else:  # camera模式
        # 转换摄像头ID为整数
        camera_id = int(args.source)
        detect_camera(model, camera_id, args.conf, args.imgsz)

if __name__ == '__main__':
    main()