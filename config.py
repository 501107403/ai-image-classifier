"""项目配置文件：集中管理所有超参数和路径设置。"""

import os

# ============== 数据配置 ==============
DATA_DIR = "./data"           # 数据集存放目录
NUM_CLASSES = 10              # 分类数量（0-9）
IMAGE_SIZE = 28               # 输入图像尺寸（MNIST 为 28x28）
IMAGE_CHANNELS = 1            # 图像通道数（灰度图为1）

# ============== 训练超参数 ==============
BATCH_SIZE = 64               # 批次大小
LEARNING_RATE = 0.001         # 学习率
NUM_EPOCHS = 10               # 训练轮数
WEIGHT_DECAY = 1e-4           # 权重衰减（L2正则化）
DROPOUT_RATE = 0.5            # Dropout 比率

# ============== 模型保存 ==============
CHECKPOINT_DIR = "./checkpoints"  # 模型保存目录
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# ============== 设备配置 ==============
# 自动选择可用设备：GPU > CPU
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== 数据增强 ==============
# 训练时的归一化均值和标准差（MNIST 统计值）
MEAN = 0.1307
STD = 0.3081
