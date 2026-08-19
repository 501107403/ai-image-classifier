"""数据加载模块：下载 MNIST 数据集并创建 DataLoader。"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


def get_transforms():
    """定义数据预处理和增强流程。

    Returns:
        train_transform: 训练集变换（含数据增强）
        test_transform: 测试集变换（仅归一化）
    """
    # 训练集：转为张量 + 归一化
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((config.MEAN,), (config.STD,))
    ])

    # 测试集：转为张量 + 归一化（不做增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((config.MEAN,), (config.STD,))
    ])

    return train_transform, test_transform


def get_dataloaders():
    """创建训练集和测试集的 DataLoader。

    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    train_transform, test_transform = get_transforms()

    # 下载并加载训练集
    train_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    # 下载并加载测试集
    test_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,       # 训练集打乱顺序
        num_workers=2,      # 数据加载线程数
        pin_memory=True     # 加速 GPU 数据传输
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,      # 测试集不需要打乱
        num_workers=2,
        pin_memory=True
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    return train_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载
    train_loader, test_loader = get_dataloaders()

    # 查看一个批次的数据形状
    images, labels = next(iter(train_loader))
    print(f"\n批次图像形状: {images.shape}")   # [batch_size, 1, 28, 28]
    print(f"批次标签形状: {labels.shape}")   # [batch_size]
    print(f"标签示例: {labels[:10].tolist()}")
