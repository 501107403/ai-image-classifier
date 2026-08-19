"""评估脚本：在测试集上评估训练好的模型，输出准确率和混淆矩阵。"""

import torch
import torch.nn as nn

import config
from dataset import get_dataloaders
from model import CNN
from utils import load_checkpoint, calculate_accuracy, get_confusion_matrix, print_confusion_matrix


def evaluate(model, test_loader, criterion, device):
    """在测试集上完整评估模型。

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 计算设备

    Returns:
        test_loss: 测试集平均损失
        test_acc: 测试集准确率
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = len(test_loader)
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, labels)

            # 统计总体准确率
            _, predictions = torch.max(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    test_loss = running_loss / num_batches
    test_acc = total_correct / total_samples  # 总体准确率更准确

    return test_loss, test_acc


def main():
    """主评估函数。"""
    device = config.DEVICE
    print(f"使用设备: {device}")

    # 1. 加载测试数据
    print("\n=== 加载测试数据 ===")
    _, test_loader = get_dataloaders()

    # 2. 创建模型并加载权重
    print("\n=== 加载模型 ===")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    epoch, loaded_acc = load_checkpoint(model, None, config.MODEL_PATH)
    if epoch == 0:
        print("警告：未找到已训练的模型，请先运行 train.py")
        return

    # 3. 评估模型
    print("\n=== 开始评估 ===")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc * 100:.2f}%")
    print(f"错误率: {(1 - test_acc) * 100:.2f}%")

    # 4. 计算并打印混淆矩阵
    print("\n=== 混淆矩阵 ===")
    cm = get_confusion_matrix(model, test_loader)
    print_confusion_matrix(cm)

    # 5. 每类准确率
    print("\n=== 各类别准确率 ===")
    for i in range(config.NUM_CLASSES):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total * 100 if total > 0 else 0
        print(f"数字 {i}: {acc:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
