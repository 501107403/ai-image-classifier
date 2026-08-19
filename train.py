"""训练脚本：完整的模型训练流程，含验证和最佳模型保存。"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import CNN, count_parameters
from utils import set_seed, save_checkpoint, calculate_accuracy


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch。

    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备

    Returns:
        avg_loss: 平均训练损失
        avg_acc: 平均训练准确率
    """
    model.train()  # 设置为训练模式（启用 Dropout、BatchNorm 等）
    running_loss = 0.0
    running_acc = 0.0
    num_batches = len(train_loader)

    # tqdm 进度条
    pbar = tqdm(train_loader, desc="训练中", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播 + 优化
        optimizer.zero_grad()   # 清空梯度
        loss.backward()         # 反向传播计算梯度
        optimizer.step()        # 更新参数

        # 统计
        running_loss += loss.item()
        running_acc += calculate_accuracy(outputs, labels)

        # 更新进度条信息
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{calculate_accuracy(outputs, labels):.4f}"
        })

    avg_loss = running_loss / num_batches
    avg_acc = running_acc / num_batches
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device):
    """在验证集上评估模型。

    Args:
        model: 神经网络模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备

    Returns:
        avg_loss: 平均验证损失
        avg_acc: 平均验证准确率
    """
    model.eval()  # 设置为评估模式（关闭 Dropout）
    running_loss = 0.0
    running_acc = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():  # 不计算梯度，节省内存
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, labels)

    avg_loss = running_loss / num_batches
    avg_acc = running_acc / num_batches
    return avg_loss, avg_acc


def main():
    """主训练函数。"""
    # 1. 设置随机种子，保证可复现
    set_seed(42)

    # 2. 确定计算设备
    device = config.DEVICE
    print(f"使用设备: {device}")

    # 3. 加载数据
    print("\n=== 加载数据 ===")
    train_loader, test_loader = get_dataloaders()

    # 4. 创建模型
    print("\n=== 创建模型 ===")
    model = CNN().to(device)
    print(f"模型参数数量: {count_parameters(model):,}")

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失（多分类标准选择）
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 6. 学习率调度器（可选：随训练降低学习率）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 7. 训练循环
    print("\n=== 开始训练 ===")
    best_acc = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.NUM_EPOCHS} ---")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证（用测试集代替验证集）
        val_loss, val_acc = validate(
            model, test_loader, criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 打印本轮结果
        print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc * 100, config.MODEL_PATH
            )

    # 8. 训练完成
    print("\n=== 训练完成 ===")
    print(f"最佳验证准确率: {best_acc * 100:.2f}%")
    print(f"最佳模型保存在: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
