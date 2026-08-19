"""推理脚本：对单张图像进行数字识别预测。"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from model import CNN
from utils import load_checkpoint


def preprocess_image(image_path):
    """预处理输入图像，使其符合模型输入要求。

    Args:
        image_path: 图像文件路径

    Returns:
        tensor: 预处理后的图像张量 [1, 1, 28, 28]
    """
    # 定义预处理流程
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转为灰度图
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),  # 调整为28x28
        transforms.ToTensor(),
        transforms.Normalize((config.MEAN,), (config.STD,))
    ])

    # 打开图像并预处理
    image = Image.open(image_path).convert("L")  # 确保是灰度模式
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # 增加 batch 维度: [1, 1, 28, 28]

    return tensor


def predict(image_path, model, device):
    """对单张图像进行预测。

    Args:
        image_path: 图像路径
        model: 训练好的模型
        device: 计算设备

    Returns:
        predicted_class: 预测的数字类别
        probabilities: 各类别的概率分布
    """
    # 预处理图像
    image_tensor = preprocess_image(image_path).to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)  # 转为概率分布
        _, predicted_class = torch.max(outputs, dim=1)

    return predicted_class.item(), probabilities.squeeze(0).cpu().numpy()


def main():
    """主推理函数。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="手写数字识别推理")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    args = parser.parse_args()

    device = config.DEVICE
    print(f"使用设备: {device}")

    # 1. 创建模型并加载权重
    print("加载模型...")
    model = CNN().to(device)
    epoch, _ = load_checkpoint(model, None, config.MODEL_PATH)

    if epoch == 0:
        print("错误：未找到已训练的模型，请先运行 train.py")
        return

    # 2. 进行预测
    print(f"\n预测图像: {args.image}")
    predicted_digit, probs = predict(args.image, model, device)

    # 3. 输出结果
    print(f"\n=== 预测结果 ===")
    print(f"识别数字: {predicted_digit}")
    print(f"置信度: {probs[predicted_digit] * 100:.2f}%")

    print("\n各类别概率:")
    for i in range(config.NUM_CLASSES):
        bar = "█" * int(probs[i] * 20)
        print(f"  {i}: {probs[i] * 100:6.2f}% {bar}")


if __name__ == "__main__":
    main()
