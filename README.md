# AI Image Classifier - CNN 手写数字识别

基于 PyTorch 的卷积神经网络（CNN）图像分类项目，使用 MNIST 手写数字数据集，包含完整的训练、评估和推理流程。

## 项目结构

```
ai-image-classifier/
├── README.md          # 项目说明
├── requirements.txt   # 依赖包
├── .gitignore         # Git 忽略规则
├── config.py          # 超参数与配置
├── dataset.py         # 数据加载与预处理
├── model.py           # CNN 模型定义
├── train.py           # 训练脚本
├── evaluate.py        # 评估脚本
├── predict.py         # 单图推理脚本
└── utils.py           # 工具函数
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- torchvision

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练完成后，模型权重会保存到 `./checkpoints/best_model.pth`。

### 2. 评估模型

```bash
python evaluate.py
```

在测试集上计算准确率并输出混淆矩阵。

### 3. 单图推理

```bash
python predict.py --image path/to/image.png
```

## 模型架构

- 两层卷积层（Conv2d + ReLU + MaxPool）
- 两层全连接层
- Dropout 防止过拟合
- 输出 10 类（数字 0-9）

## 数据集

MNIST 手写数字数据集，包含 60000 张训练图像和 10000 张测试图像，自动下载。

## 预期效果

- 训练 10 个 epoch，测试集准确率可达 98%+
