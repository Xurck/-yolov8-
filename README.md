# YOLOv8物品检测智能识别工具

基于YOLOv8n轻量化模型开发的物品检测智能识别工具，支持图片识别与摄像头实时检测。

## 🌟 核心功能

- ✅ **双模式检测**：支持单图识别与摄像头实时检测
- ✅ **COCO 80类物品覆盖**：支持检测COCO数据集定义的80种常见物品
- ✅ **轻量化设计**：YOLOv8n模型，CPU可运行，无需GPU依赖
- ✅ **实时性能**：CPU环境下实现5-10FPS实时识别
- ✅ **自动标注**：自动标注物品类别与置信度
- ✅ **易于扩展**：预留模型微调、语音播报及桌面打包部署接口

## 📁 项目结构

```
yolov8-object-detector/
├── object_detector.py    # 命令行检测工具
├── object_gui.py         # GUI界面检测工具
├── train.py              # 模型训练脚本
├── download_model.py     # 模型下载脚本
├── utils.py              # 公共工具模块
├── logger.py             # 日志记录模块
├── object.yaml           # 数据集配置文件
├── requirements.txt      # 依赖声明文件
├── README.md             # 项目说明文档
├── .gitignore            # Git忽略配置
└── logs/                 # 日志文件目录
```

## 🛠️ 技术栈

- **模型**：YOLOv8n（轻量化目标检测模型）
- **框架**：Ultralytics
- **语言**：Python 3.8+
- **依赖**：OpenCV、ultralytics

## 📦 核心模块说明

### 1. 公共工具模块 (utils.py)
- COCO数据集80类别映射
- 检测结果绘制函数
- FPS计算函数

### 2. 日志记录模块 (logger.py)
- 自动创建日志目录
- 支持文件和控制台双输出
- 按日期命名日志文件
- 支持不同日志级别（INFO、ERROR等）

### 3. 命令行检测工具 (object_detector.py)
- 支持摄像头实时检测
- 支持图片检测
- 支持自定义参数配置
- 详细的日志记录

### 4. GUI界面检测工具 (object_gui.py)
- 直观的可视化操作界面
- 支持检测模式切换
- 实时显示检测结果
- 支持参数动态调整

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 基础使用

#### 摄像头实时检测（默认模式）
```bash
python object_detector.py
```

#### 图片检测
```bash
python object_detector.py --mode image --source path/to/image.jpg
```

### 2. 自定义参数

```bash
# 自定义模型、置信度和推理尺寸
python object_detector.py \
    --model yolov11n.pt \
    --mode camera \
    --source 0 \
    --conf 0.6 \
    --imgsz 480
```

### 3. 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model` | str | `yolov11n.pt` | 模型路径，支持预训练模型名称或本地路径 |
| `--mode` | str | `camera` | 运行模式：`image`（图片检测）或 `camera`（摄像头检测） |
| `--source` | str | `0` | 检测源：图片路径或摄像头ID |
| `--conf` | float | `0.5` | 置信度阈值，默认0.5 |
| `--imgsz` | int | `320` | 推理尺寸，默认320 |

## 📊 模型训练

### 1. 数据集准备

按照YOLO格式组织数据集：
```
object_dataset/
├── train/
│   ├── images/    # 训练集图像
│   └── labels/    # 训练集标注
├── val/
│   ├── images/    # 验证集图像
│   └── labels/    # 验证集标注
└── test/
    ├── images/    # 测试集图像
    └── labels/    # 测试集标注
```

### 2. 修改数据集配置

编辑 `object.yaml` 文件，配置数据集路径和类别：

```yaml
path: ./object_dataset  # 数据集路径
train: train/images      # 训练集图像路径
val: val/images          # 验证集图像路径
test: test/images        # 测试集图像路径

# 物品检测类别（示例，实际使用COCO数据集80类）
names:
  0: paper     # 纸
  1: plastic   # 塑料
  2: metal     # 金属
  3: glass     # 玻璃
  4: cardboard # 纸板
  5: kitchen   # 餐厨垃圾
```

### 3. 开始训练

```bash
# 基础训练
python train.py

# 自定义训练参数
python train.py \
    --model yolov11n.pt \
    --data object.yaml \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --lr0 0.01
```

### 4. 训练参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model` | str | `yolov11n.pt` | 模型路径或名称 |
| `--data` | str | `object.yaml` | 数据集配置文件 |
| `--epochs` | int | `100` | 训练轮数 |
| `--batch` | int | `16` | 批次大小 |
| `--imgsz` | int | `640` | 图像尺寸 |
| `--lr0` | float | `0.01` | 初始学习率 |
| `--weight_decay` | float | `0.0005` | 权重衰减 |
| `--augment` | bool | `True` | 启用数据增强 |
| `--device` | str | `0` | 训练设备 |
| `--workers` | int | `8` | 数据加载线程数 |
| `--project` | str | `object_training` | 项目名称 |
| `--name` | str | `exp1` | 实验名称 |
| `--mode` | str | `train` | 运行模式：`train`/`val`/`test` |

## 🧪 模型评估

### 1. 验证模型

```bash
python train.py --mode val --model runs/detect/exp1/weights/best.pt
```

### 2. 测试模型

```bash
python train.py --mode test --model runs/detect/exp1/weights/best.pt --source path/to/test/images/
```

## 📱 部署应用

### 1. 桌面应用打包

可以使用 `PyInstaller` 将应用打包为桌面程序：

```bash
pip install pyinstaller
pyinstaller --onefile --windowed object_detector.py
```

### 2. 边缘设备部署

支持部署到 Raspberry Pi、Jetson Nano 等边缘设备：

```bash
# 转换为ONNX格式（可选）
yolo export model=runs/detect/exp1/weights/best.pt format=onnx
```

## 📈 性能指标

| 模型 | 尺寸 | mAP@0.5 | 推理速度（CPU） |
|------|------|---------|----------------|
| YOLOv8n | 320x320 | ~0.75 | 5-10 FPS |
| YOLOv8n | 640x640 | ~0.82 | 3-5 FPS |

## 🎯 应用场景

- 🏠 **家庭垃圾分类**：帮助家庭成员快速识别垃圾类别
- 🏫 **社区垃圾分类站**：辅助居民进行垃圾分类
- 🛒 **超市/商场**：用于垃圾分类回收
- 🚛 **垃圾处理厂**：初步分类检测

## 🔧 扩展功能

- [ ] 语音播报功能
- [ ] 垃圾分类知识科普
- [ ] 历史记录统计
- [ ] 云端模型更新
- [ ] 多语言支持


**YOLOv8物品检测智能识别工具** - 让物品检测更简单！ 🔍✨