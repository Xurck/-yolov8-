# -*- coding: utf-8 -*-
"""
日志记录模块
"""

import logging
import os
from datetime import datetime

# 确保日志目录存在
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 日志文件名
log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y-%m-%d")}.log')

# 配置日志记录器
logger = logging.getLogger('yolov8_detector')
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器到记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def get_logger():
    """
    获取日志记录器实例
    
    Returns:
        logger: 日志记录器实例
    """
    return logger
