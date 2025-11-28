from ultralytics import YOLO
import argparse
import os

# 解析命令行参数
parser = argparse.ArgumentParser(description='YOLOv8 Object Detection Training')
parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model path or name')
parser.add_argument('--data', type=str, default='object.yaml', help='Dataset configuration file')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch', type=int, default=16, help='Batch size')
parser.add_argument('--imgsz', type=int, default=640, help='Image size')
parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
parser.add_argument('--augment', action='store_true', default=True, help='Enable data augmentation')
parser.add_argument('--device', type=str, default='0', help='Device to use for training')
parser.add_argument('--workers', type=int, default=8, help='Number of workers')
parser.add_argument('--project', type=str, default='object_training', help='Project name')
parser.add_argument('--name', type=str, default='exp1', help='Experiment name')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'], help='Mode: train, val, or test')
parser.add_argument('--source', type=str, default=None, help='Source for prediction (required for test mode)')

args = parser.parse_args()

# 检查数据集配置文件是否存在
if not os.path.exists(args.data):
    print(f"Error: Dataset configuration file {args.data} not found!")
    print("Please create the dataset configuration file first.")
    exit(1)

# 加载模型
model = YOLO(args.model)

if args.mode == 'train':
    # 训练模型
    print(f"\n=== Training YOLOv8 on Dataset ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Initial LR: {args.lr0}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Augmentation: {args.augment}")
    print(f"Device: {args.device}")
    print(f"Workers: {args.workers}")
    print(f"Project: {args.project}")
    print(f"Experiment: {args.name}")
    print("========================================\n")
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        augment=args.augment,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name
    )
    
    print(f"\n=== Training Completed ===")
    print(f"Results saved to: {results.save_dir}")
    print(f"Best model: {os.path.join(results.save_dir, 'weights', 'best.pt')}")
    print(f"Last model: {os.path.join(results.save_dir, 'weights', 'last.pt')}")
    
elif args.mode == 'val':
    # 验证模型
    print(f"\n=== Validating YOLOv8 on Dataset ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Image Size: {args.imgsz}")
    print(f"Batch Size: {args.batch}")
    print(f"Device: {args.device}")
    print("========================================\n")
    
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
    
    print(f"\n=== Validation Results ===")
    print(f"mAP@0.5: {results.box.map:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map50_95:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
elif args.mode == 'test':
    # 测试模型
    if not args.source:
        print("Error: --source is required for test mode!")
        exit(1)
    
    print(f"\n=== Testing YOLOv8 on Dataset ===")
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device: {args.device}")
    print("========================================\n")
    
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        show=False,
        conf=0.5
    )
    
    print(f"\n=== Testing Completed ===")
    print(f"Results saved to: {results[0].save_dir}")

else:
    print(f"Error: Invalid mode {args.mode}!")
    print("Please choose from: train, val, test")
    exit(1)