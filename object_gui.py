import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO
from utils import COCO_CATEGORY_NAMES, draw_detection_result

class ObjectDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8物品检测智能识别工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 初始化变量
        self.model = None
        self.cap = None
        self.is_running = False
        self.detection_thread = None
        
        # 默认参数
        self.conf_threshold = 0.5
        self.img_size = 320
        self.detection_mode = "camera"
        self.image_path = ""
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建顶部控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", padding="10")
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # 检测模式选择
        ttk.Label(self.control_frame, text="检测模式:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.mode_var = tk.StringVar(value="camera")
        mode_frame = ttk.Frame(self.control_frame)
        mode_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Radiobutton(mode_frame, text="摄像头实时检测", variable=self.mode_var, value="camera", command=self.on_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="图片检测", variable=self.mode_var, value="image", command=self.on_mode_change).pack(side=tk.LEFT, padx=10)
        
        # 图片选择按钮
        self.image_btn = ttk.Button(self.control_frame, text="选择图片", command=self.select_image, state=tk.DISABLED)
        self.image_btn.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # 置信度阈值
        ttk.Label(self.control_frame, text="置信度阈值:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_scale = ttk.Scale(self.control_frame, from_=0.1, to=1.0, variable=self.conf_var, orient=tk.HORIZONTAL, length=200)
        self.conf_scale.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        self.conf_label = ttk.Label(self.control_frame, text=f"{self.conf_var.get():.1f}")
        self.conf_label.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.conf_scale.bind("<Motion>", self.update_conf_label)
        
        # 推理尺寸
        ttk.Label(self.control_frame, text="推理尺寸:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.imgsz_var = tk.IntVar(value=320)
        imgsz_frame = ttk.Frame(self.control_frame)
        imgsz_frame.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Radiobutton(imgsz_frame, text="320x320", variable=self.imgsz_var, value=320).pack(side=tk.LEFT)
        ttk.Radiobutton(imgsz_frame, text="480x480", variable=self.imgsz_var, value=480).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(imgsz_frame, text="640x640", variable=self.imgsz_var, value=640).pack(side=tk.LEFT, padx=10)
        
        # 开始/停止按钮
        self.start_btn = ttk.Button(self.control_frame, text="开始检测", command=self.start_detection)
        self.start_btn.grid(row=3, column=0, padx=5, pady=10, sticky=tk.W)
        self.stop_btn = ttk.Button(self.control_frame, text="停止检测", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=3, column=1, padx=5, pady=10, sticky=tk.W)
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var, foreground="green")
        self.status_label.grid(row=3, column=2, padx=5, pady=10, sticky=tk.W)
        
        # 创建结果显示区域
        self.result_frame = ttk.LabelFrame(self.main_frame, text="检测结果", padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 图像显示区域
        self.canvas = tk.Canvas(self.result_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载YOLOv8模型"""
        try:
            self.status_var.set("正在加载模型...")
            self.root.update()
            self.model = YOLO("yolov8n.pt")
            self.status_var.set("模型加载成功，就绪")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {e}")
            self.status_var.set("模型加载失败")
    
    def on_mode_change(self):
        """检测模式改变时的处理"""
        self.detection_mode = self.mode_var.get()
        if self.detection_mode == "image":
            self.image_btn.config(state=tk.NORMAL)
        else:
            self.image_btn.config(state=tk.DISABLED)
    
    def select_image(self):
        """选择图片"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.status_var.set(f"已选择图片: {file_path}")
    
    def update_conf_label(self, event):
        """更新置信度标签"""
        self.conf_label.config(text=f"{self.conf_var.get():.1f}")
    
    def start_detection(self):
        """开始检测"""
        if not self.model:
            messagebox.showerror("错误", "模型未加载成功")
            return
        
        if self.detection_mode == "image" and not self.image_path:
            messagebox.showerror("错误", "请先选择图片")
            return
        
        # 更新状态
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("正在检测...")
        
        # 获取参数
        self.conf_threshold = self.conf_var.get()
        self.img_size = self.imgsz_var.get()
        
        # 启动检测线程
        self.detection_thread = threading.Thread(target=self.run_detection)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def stop_detection(self):
        """停止检测"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("检测已停止")
    
    def run_detection(self):
        """运行检测"""
        if self.detection_mode == "camera":
            self.run_camera_detection()
        else:
            self.run_image_detection()
    
    def run_camera_detection(self):
        """摄像头实时检测"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            self.stop_detection()
            return
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 执行推理
            results = self.model(frame, conf=self.conf_threshold, imgsz=self.img_size)
            
            # 获取检测结果
            boxes = results[0].boxes
            
            # 绘制检测结果
            annotated_frame = draw_detection_result(frame, boxes, COCO_CATEGORY_NAMES, self.conf_threshold)
            
            # 显示FPS信息
            cv2.putText(annotated_frame, "YOLOv8物品检测", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 转换为RGB格式并显示
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            self.display_image(rgb_frame)
        
        self.cap.release()
    
    def run_image_detection(self):
        """图片检测"""
        # 读取图片
        img = cv2.imread(self.image_path)
        if img is None:
            messagebox.showerror("错误", f"无法读取图片: {self.image_path}")
            self.stop_detection()
            return
        
        # 执行推理
        results = self.model(img, conf=self.conf_threshold, imgsz=self.img_size)
        
        # 获取检测结果
        boxes = results[0].boxes
        
        # 绘制检测结果
        annotated_img = draw_detection_result(img, boxes, COCO_CATEGORY_NAMES, self.conf_threshold)
        
        # 转换为RGB格式并显示
        rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        self.display_image(rgb_img)
        
        # 停止检测
        self.stop_detection()
    
    def display_image(self, image):
        """在Canvas上显示图像"""
        # 获取Canvas尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 调整图像大小以适应Canvas
        img = Image.fromarray(image)
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            # 图像更宽，以宽度为基准缩放
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            # 图像更高，以高度为基准缩放
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
        # 缩放图像
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 转换为PhotoImage
        self.photo = ImageTk.PhotoImage(resized_img)
        
        # 清除Canvas并显示新图像
        self.canvas.delete("all")
        # 居中显示图像
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
    
    def on_closing(self):
        """关闭窗口时的处理"""
        if self.is_running:
            self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()