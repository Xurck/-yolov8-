from ultralytics import YOLO

print("正在下载YOLOv8n模型...")
# 使用YOLO类初始化会自动下载模型
model = YOLO("yolov8n.pt")
print("模型下载完成！")
print(f"模型路径: {model.model_path}")