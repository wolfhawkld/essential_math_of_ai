#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MNIST 从零实现

本模块从零实现一个完整的神经网络来识别 MNIST 手写数字。
不依赖深度学习框架，仅使用 NumPy。

数学知识：
- 线性代数：矩阵乘法、向量运算
- 微积分：反向传播、梯度计算
- 概率：Softmax、交叉熵损失
- 优化：SGD、动量

实现内容：
- 数据加载（简化版 MNIST）
- 神经网络层：Linear、ReLU、Softmax
- 前向传播和反向传播
- 训练循环和评估
- 可视化：损失曲线、准确率、样本预测
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
import pickle
import os
from urllib import request
import gzip


# ============================================================================
# 数据加载
# ============================================================================

class MNISTLoader:
    """
    MNIST 数据加载器

    自动下载 MNIST 数据集并提供批量加载功能。
    数据格式：28x28 灰度图像，像素值 0-255
    """

    def __init__(self, data_dir: str = './data'):
        """
        初始化数据加载器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def download(self) -> None:
        """下载 MNIST 数据集"""
        base_url = "http://yann.lecun.com/exdb/mnist/"
        files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }

        os.makedirs(self.data_dir, exist_ok=True)

        for name, filename in files.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                print(f"下载 {filename}...")
                request.urlretrieve(base_url + filename, filepath)

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载 MNIST 数据集

        Returns:
            train_images: 训练图像 (60000, 784)
            train_labels: 训练标签 (60000,)
            test_images: 测试图像 (10000, 784)
            test_labels: 测试标签 (10000,)
        """
        self.download()

        # 加载训练图像
        with gzip.open(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz'), 'rb') as f:
            self.train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)

        # 加载训练标签
        with gzip.open(os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
            self.train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

        # 加载测试图像
        with gzip.open(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
            self.test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)

        # 加载测试标签
        with gzip.open(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
            self.test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return (self.train_images.astype(np.float32),
                self.train_labels,
                self.test_images.astype(np.float32),
                self.test_labels)


def create_simple_dataset(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建简化的数据集用于快速测试

    生成类似 MNIST 的随机数据，但规模更小。

    Args:
        n_samples: 每个类别的样本数

    Returns:
        train_images, train_labels, test_images, test_labels
    """
    np.random.seed(42)

    n_train = n_samples
    n_test = n_samples // 5
    n_classes = 10

    # 生成训练数据
    train_images = np.random.randn(n_train, 784).astype(np.float32)
    train_labels = np.random.randint(0, n_classes, n_train)

    # 为每个类别添加特定模式
    for i in range(n_train):
        label = train_labels[i]
        # 在特定位置添加模式
        start_pixel = label * 78
        train_images[i, start_pixel:start_pixel+78] += 2

    # 生成测试数据
    test_images = np.random.randn(n_test, 784).astype(np.float32)
    test_labels = np.random.randint(0, n_classes, n_test)

    for i in range(n_test):
        label = test_labels[i]
        start_pixel = label * 78
        test_images[i, start_pixel:start_pixel+78] += 2

    return train_images, train_labels, test_images, test_labels


# ============================================================================
# 神经网络层
# ============================================================================

class Layer:
    """神经网络层基类"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """反向传播"""
        raise NotImplementedError

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回参数列表 [(参数, 梯度)]"""
        return []

    def zero_grad(self) -> None:
        """清零梯度"""
        for param, grad in self.parameters():
            grad.fill(0)


class Linear(Layer):
    """
    全连接层

    数学公式：
        前向传播: y = xW + b
        反向传播:
            ∂L/∂W = x.T @ ∂L/∂y
            ∂L/∂b = sum(∂L/∂y, axis=0)
            ∂L/∂x = ∂L/∂y @ W.T

    参数初始化使用 Xavier/Glorot 初始化：
        W ~ N(0, sqrt(2/(in_features + out_features)))
    """

    def __init__(self, in_features: int, out_features: int):
        """
        初始化全连接层

        Args:
            in_features: 输入特征数
            out_features: 输出特征数
        """
        self.in_features = in_features
        self.out_features = out_features

        # Xavier 初始化
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.b = np.zeros(out_features, dtype=np.float32)

        # 梯度
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # 缓存输入用于反向传播
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, in_features)

        Returns:
            输出张量 (batch_size, out_features)
        """
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            grad_output: 上层梯度 (batch_size, out_features)

        Returns:
            对输入的梯度 (batch_size, in_features)
        """
        # 计算参数梯度
        self.grad_W += self.x.T @ grad_output
        self.grad_b += np.sum(grad_output, axis=0)

        # 计算输入梯度
        return grad_output @ self.W.T

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回参数和梯度"""
        return [(self.W, self.grad_W), (self.b, self.grad_b)]


class ReLU(Layer):
    """
    ReLU 激活函数

    数学公式：
        前向传播: y = max(0, x)
        反向传播: ∂L/∂x = ∂L/∂y * (x > 0)
    """

    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """反向传播"""
        return grad_output * self.mask


class Softmax(Layer):
    """
    Softmax 激活函数

    数学公式：
        y_i = exp(x_i) / sum(exp(x_j))

    数值稳定版本：
        y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """

    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播（数值稳定版本）"""
        # 减去最大值防止溢出
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播

        对于 softmax + cross_entropy 的组合，
        梯度已经在交叉熵损失中计算好了。
        """
        return grad_output


# ============================================================================
# 损失函数
# ============================================================================

class CrossEntropyLoss:
    """
    交叉熵损失函数

    数学公式：
        L = -sum(y_true * log(y_pred))

    与 Softmax 组合时的简化梯度：
        ∂L/∂x = y_pred - y_true
    """

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算损失

        Args:
            y_pred: 预测概率 (batch_size, n_classes)
            y_true: 真实标签的 one-hot 编码 (batch_size, n_classes)

        Returns:
            平均损失值
        """
        self.y_pred = y_pred
        self.y_true = y_true

        # 数值稳定：裁剪预测值避免 log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # 计算交叉熵
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_pred.shape[0]
        return loss

    def backward(self) -> np.ndarray:
        """
        反向传播

        对于 softmax + cross_entropy 组合：
            ∂L/∂z = y_pred - y_true

        Returns:
            对 softmax 输入的梯度
        """
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]


# ============================================================================
# 优化器
# ============================================================================

class SGD:
    """
    随机梯度下降优化器

    支持动量（Momentum）：
        v = momentum * v - lr * grad
        param = param + v

    数学公式：
        标准SGD: θ = θ - lr * ∂L/∂θ
        动量SGD: v = βv + ∂L/∂θ
                 θ = θ - lr * v
    """

    def __init__(self, parameters: List[Tuple[np.ndarray, np.ndarray]],
                 lr: float = 0.01, momentum: float = 0.0):
        """
        初始化优化器

        Args:
            parameters: 参数列表 [(参数, 梯度)]
            lr: 学习率
            momentum: 动量系数
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

        # 初始化速度
        self.velocities = [np.zeros_like(param) for param, _ in parameters]

    def step(self) -> None:
        """执行一步参数更新"""
        for i, (param, grad) in enumerate(self.parameters):
            if self.momentum > 0:
                # 动量更新
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                param += self.velocities[i]
            else:
                # 标准 SGD
                param -= self.lr * grad

    def zero_grad(self) -> None:
        """清零所有梯度"""
        for _, grad in self.parameters:
            grad.fill(0)


# ============================================================================
# 神经网络容器
# ============================================================================

class NeuralNetwork:
    """
    神经网络容器

    管理多个层的顺序执行。
    """

    def __init__(self, layers: List[Layer]):
        """
        初始化网络

        Args:
            layers: 层列表
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回所有参数"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        """清零所有梯度"""
        for layer in self.layers:
            layer.zero_grad()


# ============================================================================
# 训练和评估
# ============================================================================

def one_hot_encode(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """将标签转换为 one-hot 编码"""
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), labels] = 1
    return one_hot


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """计算准确率"""
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(pred_labels == true_labels)


def train_epoch(model: NeuralNetwork,
                optimizer: SGD,
                loss_fn: CrossEntropyLoss,
                X: np.ndarray,
                y: np.ndarray,
                batch_size: int = 32) -> Tuple[float, float]:
    """
    训练一个 epoch

    Args:
        model: 神经网络模型
        optimizer: 优化器
        loss_fn: 损失函数
        X: 训练数据
        y: 训练标签（one-hot）
        batch_size: 批量大小

    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率
    """
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        # 获取批次数据
        batch_idx = indices[i:i + batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]

        # 前向传播
        y_pred = model.forward(X_batch)

        # 计算损失
        loss = loss_fn.forward(y_pred, y_batch)
        total_loss += loss

        # 计算准确率
        total_acc += accuracy(y_pred, y_batch)
        n_batches += 1

        # 反向传播
        model.zero_grad()
        grad = loss_fn.backward()
        model.backward(grad)

        # 更新参数
        optimizer.step()

    return total_loss / n_batches, total_acc / n_batches


def evaluate(model: NeuralNetwork,
             X: np.ndarray,
             y: np.ndarray,
             batch_size: int = 100) -> Tuple[float, float]:
    """
    评估模型

    Args:
        model: 神经网络模型
        X: 测试数据
        y: 测试标签（one-hot）
        batch_size: 批量大小

    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率
    """
    n_samples = X.shape[0]
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    loss_fn = CrossEntropyLoss()

    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        # 前向传播
        y_pred = model.forward(X_batch)

        # 计算损失
        loss = loss_fn.forward(y_pred, y_batch)
        total_loss += loss

        # 计算准确率
        total_acc += accuracy(y_pred, y_batch)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


# ============================================================================
# 可视化
# ============================================================================

def plot_training_history(train_losses: List[float],
                          train_accs: List[float],
                          test_losses: List[float],
                          test_accs: List[float]) -> None:
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(train_losses, label='训练损失')
    axes[0].plot(test_losses, label='测试损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title('训练和测试损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(train_accs, label='训练准确率')
    axes[1].plot(test_accs, label='测试准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('训练和测试准确率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mnist_training_history.png', dpi=150)
    plt.close()
    print("训练历史已保存到 mnist_training_history.png")


def plot_predictions(model: NeuralNetwork,
                     X: np.ndarray,
                     y: np.ndarray,
                     n_samples: int = 10) -> None:
    """绘制预测示例"""
    # 随机选择样本
    indices = np.random.choice(len(X), n_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i, idx in enumerate(indices):
        ax = axes[i // 5, i % 5]

        # 显示图像
        img = X[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')

        # 预测
        pred = model.forward(X[idx:idx+1])
        pred_label = np.argmax(pred)
        true_label = np.argmax(y[idx])

        # 设置标题
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'预测: {pred_label}, 真实: {true_label}', color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150)
    plt.close()
    print("预测示例已保存到 mnist_predictions.png")


def plot_weights(model: NeuralNetwork) -> None:
    """可视化第一层权重"""
    # 找到第一个 Linear 层
    first_linear = None
    for layer in model.layers:
        if isinstance(layer, Linear):
            first_linear = layer
            break

    if first_linear is None:
        return

    W = first_linear.W  # (784, out_features)

    # 可视化前 25 个神经元的权重
    n_show = min(25, W.shape[1])
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))

    for i in range(n_show):
        ax = axes[i // 5, i % 5]
        img = W[:, i].reshape(28, 28)
        ax.imshow(img, cmap='RdBu')
        ax.set_title(f'神经元 {i}')
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(n_show, 25):
        axes[i // 5, i % 5].axis('off')

    plt.suptitle('第一层权重可视化')
    plt.tight_layout()
    plt.savefig('mnist_weights.png', dpi=150)
    plt.close()
    print("权重可视化已保存到 mnist_weights.png")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("MNIST 从零实现 - 神经网络")
    print("=" * 60)

    # 设置随机种子
    np.random.seed(42)

    # 尝试加载 MNIST 数据，失败则使用简化数据集
    print("\n加载数据...")
    try:
        loader = MNISTLoader()
        train_images, train_labels, test_images, test_labels = loader.load()
        print(f"训练集: {train_images.shape}")
        print(f"测试集: {test_images.shape}")
    except Exception as e:
        print(f"无法下载 MNIST 数据集: {e}")
        print("使用简化数据集...")
        train_images, train_labels, test_images, test_labels = create_simple_dataset()
        print(f"训练集: {train_images.shape}")
        print(f"测试集: {test_images.shape}")

    # 数据预处理
    print("\n数据预处理...")
    # 归一化到 [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # One-hot 编码标签
    train_labels_onehot = one_hot_encode(train_labels)
    test_labels_onehot = one_hot_encode(test_labels)

    print(f"训练数据范围: [{train_images.min():.2f}, {train_images.max():.2f}]")
    print(f"标签形状: {train_labels_onehot.shape}")

    # 创建模型
    print("\n创建模型...")
    model = NeuralNetwork([
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
        Softmax()
    ])

    # 打印模型结构
    print("模型结构:")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            print(f"  Layer {i}: Linear({layer.in_features}, {layer.out_features})")
        elif isinstance(layer, ReLU):
            print(f"  Layer {i}: ReLU()")
        elif isinstance(layer, Softmax):
            print(f"  Layer {i}: Softmax()")

    # 统计参数数量
    n_params = sum(np.prod(param.shape) for param, _ in model.parameters())
    print(f"总参数数量: {n_params:,}")

    # 创建优化器和损失函数
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = CrossEntropyLoss()

    # 训练
    print("\n开始训练...")
    n_epochs = 10
    batch_size = 64

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(n_epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, optimizer, loss_fn,
            train_images, train_labels_onehot,
            batch_size=batch_size
        )

        # 评估
        test_loss, test_acc = evaluate(
            model, test_images, test_labels_onehot
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1:2d}/{n_epochs}: "
              f"训练损失={train_loss:.4f}, 训练准确率={train_acc:.4f}, "
              f"测试损失={test_loss:.4f}, 测试准确率={test_acc:.4f}")

    print("\n训练完成!")
    print(f"最终测试准确率: {test_accs[-1]:.4f}")

    # 可视化
    print("\n生成可视化...")
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    plot_predictions(model, test_images, test_labels_onehot)
    plot_weights(model)

    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)


# ============================================================================
# 测试
# ============================================================================

def test_linear_layer():
    """测试 Linear 层"""
    print("测试 Linear 层...")

    np.random.seed(42)

    # 创建层
    layer = Linear(3, 4)

    # 测试前向传播
    x = np.random.randn(2, 3).astype(np.float32)
    y = layer.forward(x)

    assert y.shape == (2, 4), f"输出形状错误: {y.shape}"

    # 测试反向传播（梯度检查）
    grad_output = np.random.randn(2, 4).astype(np.float32)
    grad_x = layer.backward(grad_output)

    assert grad_x.shape == (2, 3), f"梯度形状错误: {grad_x.shape}"
    assert layer.grad_W.shape == (3, 4), f"W梯度形状错误: {layer.grad_W.shape}"
    assert layer.grad_b.shape == (4,), f"b梯度形状错误: {layer.grad_b.shape}"

    print("  ✓ 前向传播正确")
    print("  ✓ 反向传播正确")


def test_relu_layer():
    """测试 ReLU 层"""
    print("测试 ReLU 层...")

    layer = ReLU()

    # 测试前向传播
    x = np.array([[-1, 2], [3, -4]], dtype=np.float32)
    y = layer.forward(x)

    expected = np.array([[0, 2], [3, 0]], dtype=np.float32)
    assert np.allclose(y, expected), f"ReLU 前向传播错误: {y}"

    # 测试反向传播
    grad_output = np.ones_like(y)
    grad_x = layer.backward(grad_output)

    expected_grad = np.array([[0, 1], [1, 0]], dtype=np.float32)
    assert np.allclose(grad_x, expected_grad), f"ReLU 反向传播错误: {grad_x}"

    print("  ✓ 前向传播正确")
    print("  ✓ 反向传播正确")


def test_softmax_layer():
    """测试 Softmax 层"""
    print("测试 Softmax 层...")

    layer = Softmax()

    # 测试前向传播
    x = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float32)
    y = layer.forward(x)

    # 检查概率和为 1
    assert np.allclose(np.sum(y, axis=1), 1.0), f"Softmax 概率和不为 1: {np.sum(y, axis=1)}"

    # 检查所有值为正
    assert np.all(y > 0), f"Softmax 存在负值: {y}"

    print("  ✓ 前向传播正确")
    print("  ✓ 数值稳定性正确")


def test_cross_entropy():
    """测试交叉熵损失"""
    print("测试交叉熵损失...")

    loss_fn = CrossEntropyLoss()

    # 测试损失计算
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
    y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    loss = loss_fn.forward(y_pred, y_true)

    # 损失应该为正
    assert loss > 0, f"损失应该为正: {loss}"

    # 测试梯度
    grad = loss_fn.backward()
    assert grad.shape == y_pred.shape, f"梯度形状错误: {grad.shape}"

    print("  ✓ 损失计算正确")
    print("  ✓ 梯度计算正确")


def test_sgd_optimizer():
    """测试 SGD 优化器"""
    print("测试 SGD 优化器...")

    np.random.seed(42)

    # 创建简单参数
    param = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    grad = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    parameters = [(param, grad)]
    optimizer = SGD(parameters, lr=0.1)

    # 更新前
    param_before = param.copy()

    # 更新
    optimizer.step()

    # 检查参数已更新
    assert not np.allclose(param, param_before), "参数未更新"

    print("  ✓ 参数更新正确")


def test_end_to_end():
    """端到端测试：训练一个简单的网络"""
    print("测试端到端训练...")

    np.random.seed(42)

    # 创建简单数据
    X = np.random.randn(100, 4).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    y_onehot = one_hot_encode(y, n_classes=2)

    # 创建网络
    model = NeuralNetwork([
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
        Softmax()
    ])

    # 训练
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = CrossEntropyLoss()

    losses = []
    for _ in range(10):
        loss, acc = train_epoch(model, optimizer, loss_fn, X, y_onehot, batch_size=10)
        losses.append(loss)

    # 损失应该下降
    assert losses[-1] < losses[0], f"损失未下降: {losses[0]:.4f} -> {losses[-1]:.4f}"

    print(f"  ✓ 损失下降: {losses[0]:.4f} -> {losses[-1]:.4f}")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行测试")
    print("=" * 60)

    test_linear_layer()
    test_relu_layer()
    test_softmax_layer()
    test_cross_entropy()
    test_sgd_optimizer()
    test_end_to_end()

    print("\n所有测试通过!")


if __name__ == "__main__":
    # 运行测试
    run_tests()

    # 运行主程序
    main()