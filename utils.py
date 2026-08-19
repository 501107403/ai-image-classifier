"""工具函数：训练过程中的通用辅助函数。"""

import os
import torch
import numpy as np

import config


def set_seed(seed=42):
    """设置随机种子，保证实验可复现。

    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")


def save_checkpoint(model, optimizer, epoch, accuracy, path):
    """保存模型检查点。

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        accuracy: 当前准确率
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": accuracy
    }
    torch.save(checkpoint, path)
    print(f"模型已保存到: {path} (准确率: {accuracy:.2f}%)")


def load_checkpoint(model, optimizer, path):
    """加载模型检查点。

    Args:
        model: 模型
        optimizer: 优化器（可为None）
        path: 检查点路径

    Returns:
        epoch: 恢复的轮数
        accuracy: 恢复的准确率
    """
    if not os.path.exists(path):
        print(f"检查点不存在: {path}")
        return 0, 0.0

    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    accuracy = checkpoint.get("accuracy", 0.0)
    print(f"模型已加载: {path} (第{epoch}轮, 准确率: {accuracy:.2f}%)")
    return epoch, accuracy


def calculate_accuracy(outputs, labels):
    """计算批次准确率。

    Args:
        outputs: 模型输出 [batch, num_classes]
        labels: 真实标签 [batch]

    Returns:
        accuracy: 准确率（0-1之间）
    """
    _, predictions = torch.max(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total


def get_confusion_matrix(model, data_loader, num_classes=config.NUM_CLASSES):
    """计算混淆矩阵。

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        num_classes: 类别数

    Returns:
        confusion_matrix: 混淆矩阵 numpy 数组 [num_classes, num_classes]
    """
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)

            for true, pred in zip(labels, predictions):
                confusion_matrix[true.item()][pred.item()] += 1

    return confusion_matrix


def print_confusion_matrix(cm):
    """打印格式化的混淆矩阵。"""
    print("\n混淆矩阵 (行=真实, 列=预测):")
    print("      " + "  ".join(f"{i:4d}" for i in range(cm.shape[0])))
    print("     " + "-" * (6 * cm.shape[0]))
    for i in range(cm.shape[0]):
        print(f"{i:3d} | " + "  ".join(f"{cm[i][j]:4d}" for j in range(cm.shape[1])))
