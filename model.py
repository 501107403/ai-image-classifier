"""CNN 模型定义：两层卷积 + 两层全连接的手写数字分类网络。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class CNN(nn.Module):
    """卷积神经网络，用于 MNIST 手写数字分类。

    架构：
        Conv1 -> ReLU -> MaxPool -> Conv2 -> ReLU -> MaxPool
        -> Flatten -> FC1 -> ReLU -> Dropout -> FC2 -> Output
    """

    def __init__(self, num_classes=config.NUM_CLASSES, dropout_rate=config.DROPOUT_RATE):
        """初始化网络层。

        Args:
            num_classes: 分类数量，默认10（数字0-9）
            dropout_rate: Dropout 比率，默认0.5
        """
        super(CNN, self).__init__()

        # 第一层卷积：输入1通道(灰度)，输出32通道，3x3卷积核
        # 输入: [batch, 1, 28, 28] -> 输出: [batch, 32, 26, 26]
        self.conv1 = nn.Conv2d(
            in_channels=config.IMAGE_CHANNELS,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0
        )

        # 第二层卷积：输入32通道，输出64通道，3x3卷积核
        # 输入: [batch, 32, 13, 13] -> 输出: [batch, 64, 11, 11]
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )

        # 最大池化层：2x2，步长2（尺寸减半）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout 层：防止过拟合
        self.dropout = nn.Dropout(p=dropout_rate)

        # 全连接层1：卷积输出展平后维度为 64 * 5 * 5 = 1600
        # 经过两次池化：28 -> 26(conv1) -> 13(pool1) -> 11(conv2) -> 5(pool2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)

        # 全连接层2（输出层）：128 -> 10类
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入图像张量 [batch, 1, 28, 28]

        Returns:
            logits: 未归一化的分类分数 [batch, 10]
        """
        # 第一层卷积 + 激活 + 池化
        x = self.conv1(x)           # [batch, 32, 26, 26]
        x = F.relu(x)
        x = self.pool(x)            # [batch, 32, 13, 13]

        # 第二层卷积 + 激活 + 池化
        x = self.conv2(x)           # [batch, 64, 11, 11]
        x = F.relu(x)
        x = self.pool(x)            # [batch, 64, 5, 5]

        # 展平：[batch, 64, 5, 5] -> [batch, 1600]
        x = x.view(x.size(0), -1)

        # 全连接层1 + 激活 + Dropout
        x = self.fc1(x)             # [batch, 128]
        x = F.relu(x)
        x = self.dropout(x)

        # 全连接层2（输出层）
        x = self.fc2(x)             # [batch, 10]

        return x


def count_parameters(model):
    """统计模型可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    model = CNN()
    print(model)
    print(f"\n可训练参数数量: {count_parameters(model):,}")

    # 测试前向传播
    dummy_input = torch.randn(2, 1, 28, 28)  # 模拟2张28x28灰度图
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")  # 应为 [2, 10]
