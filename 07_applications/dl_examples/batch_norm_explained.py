#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Normalization 原理详解

本模块从零实现批归一化，深入理解其数学原理和训练/推理模式的差异。

数学知识：
- 均值、方差计算
- 标准化公式
- 可学习参数（γ, β）
- 训练/推理模式差异

实现内容：
- BatchNorm1D: 一维批归一化
- BatchNorm2D: 二维批归一化（用于 CNN）
- 可视化：激活分布变化
- 对比实验：有无 BN 的训练效果
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# ============================================================================
# 批归一化实现
# ============================================================================

class BatchNorm1D:
    """
    一维批归一化层

    数学公式：
        训练时：
            μ_B = 1/m * sum(x_i)          # 批均值
            σ²_B = 1/m * sum((x_i - μ_B)²)  # 批方差
            x̂_i = (x_i - μ_B) / sqrt(σ²_B + ε)  # 标准化
            y_i = γ * x̂_i + β             # 缩放和平移

        推理时：
            y_i = γ * (x_i - μ_running) / sqrt(σ²_running + ε) + β

    参数：
        γ (gamma): 缩放参数，初始化为 1
        β (beta): 平移参数，初始化为 0
        μ_running: 运行均值（用于推理）
        σ²_running: 运行方差（用于推理）

    动量更新：
        μ_running = momentum * μ_running + (1 - momentum) * μ_B
        σ²_running = momentum * σ²_running + (1 - momentum) * σ²_B
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        初始化批归一化层

        Args:
            num_features: 特征数量
            eps: 数值稳定性常数
            momentum: 运行统计量的动量
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)

        # 梯度
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

        # 运行统计量（用于推理）
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

        # 训练/推理模式
        self.training = True

        # 缓存（用于反向传播）
        self.x_norm = None
        self.std = None
        self.x_centered = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入张量 (batch, num_features)

        Returns:
            归一化后的输出 (batch, num_features)
        """
        if self.training:
            # 训练模式：使用批统计量
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # 更新运行统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 推理模式：使用运行统计量
            mean = self.running_mean
            var = self.running_var

        # 标准化
        self.x_centered = x - mean
        self.std = np.sqrt(var + self.eps)
        self.x_norm = self.x_centered / self.std

        # 缩放和平移
        out = self.gamma * self.x_norm + self.beta

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播

        梯度计算：
            ∂L/∂γ = sum(∂L/∂y * x̂)
            ∂L/∂β = sum(∂L/∂y)
            ∂L/∂x = (1/m) * γ * (m * ∂L/∂y - sum(∂L/∂y) - x̂ * sum(∂L/∂y * x̂)) / std

        Args:
            grad_output: 上层梯度 (batch, num_features)

        Returns:
            对输入的梯度 (batch, num_features)
        """
        m = grad_output.shape[0]

        # 计算参数梯度
        self.grad_gamma += np.sum(grad_output * self.x_norm, axis=0)
        self.grad_beta += np.sum(grad_output, axis=0)

        # 计算输入梯度
        # 简化公式：∂L/∂x = γ/std * (∂L/∂y - mean(∂L/∂y) - x̂ * mean(∂L/∂y * x̂))
        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * self.x_centered * (-0.5) * (self.std ** -3), axis=0)
        dmean = np.sum(dx_norm * (-1 / self.std), axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)

        grad_input = dx_norm / self.std + dvar * 2 * self.x_centered / m + dmean / m

        return grad_input

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回参数和梯度"""
        return [(self.gamma, self.grad_gamma), (self.beta, self.grad_beta)]

    def zero_grad(self) -> None:
        """清零梯度"""
        self.grad_gamma.fill(0)
        self.grad_beta.fill(0)


class BatchNorm2D:
    """
    二维批归一化层（用于 CNN）

    数学公式与 BatchNorm1D 相同，但作用于 (N, C, H, W) 张量。
    统计量在 (N, H, W) 维度上计算，对每个通道独立归一化。

    输入形状：(batch, channels, height, width)
    输出形状：(batch, channels, height, width)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        初始化批归一化层

        Args:
            num_features: 通道数
            eps: 数值稳定性常数
            momentum: 运行统计量的动量
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)

        # 梯度
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

        # 运行统计量
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

        # 训练/推理模式
        self.training = True

        # 缓存
        self.x_norm = None
        self.std = None
        self.x_centered = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入张量 (batch, channels, height, width)

        Returns:
            归一化后的输出 (batch, channels, height, width)
        """
        N, C, H, W = x.shape

        # 重塑为 (N*H*W, C) 以便计算
        x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)

        if self.training:
            # 训练模式：计算批统计量
            mean = np.mean(x_reshaped, axis=0)
            var = np.var(x_reshaped, axis=0)

            # 更新运行统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 推理模式：使用运行统计量
            mean = self.running_mean
            var = self.running_var

        # 标准化
        self.x_centered = x_reshaped - mean
        self.std = np.sqrt(var + self.eps)
        self.x_norm = self.x_centered / self.std

        # 缩放和平移
        out = self.gamma * self.x_norm + self.beta

        # 重塑回原始形状
        out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            grad_output: 上层梯度 (batch, channels, height, width)

        Returns:
            对输入的梯度 (batch, channels, height, width)
        """
        N, C, H, W = grad_output.shape

        # 重塑梯度
        grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, C)
        m = N * H * W

        # 计算参数梯度
        self.grad_gamma += np.sum(grad_output_reshaped * self.x_norm, axis=0)
        self.grad_beta += np.sum(grad_output_reshaped, axis=0)

        # 计算输入梯度
        dx_norm = grad_output_reshaped * self.gamma
        dvar = np.sum(dx_norm * self.x_centered * (-0.5) * (self.std ** -3), axis=0)
        dmean = np.sum(dx_norm * (-1 / self.std), axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)

        grad_input = dx_norm / self.std + dvar * 2 * self.x_centered / m + dmean / m

        # 重塑回原始形状
        grad_input = grad_input.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return grad_input

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回参数和梯度"""
        return [(self.gamma, self.grad_gamma), (self.beta, self.grad_beta)]

    def zero_grad(self) -> None:
        """清零梯度"""
        self.grad_gamma.fill(0)
        self.grad_beta.fill(0)


# ============================================================================
# 可视化工具
# ============================================================================

def visualize_activation_distribution(
    activations_before: np.ndarray,
    activations_after: np.ndarray,
    layer_name: str = "层",
    save_path: str = "bn_activation_dist.png"
) -> None:
    """
    可视化激活分布变化

    Args:
        activations_before: BN 前的激活 (batch, features)
        activations_after: BN 后的激活 (batch, features)
        layer_name: 层名称
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # BN 前的分布
    axes[0].hist(activations_before.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0].set_title(f'{layer_name} - BN 前')
    axes[0].set_xlabel('激活值')
    axes[0].set_ylabel('频数')

    # 添加统计信息
    mean_before = np.mean(activations_before)
    std_before = np.std(activations_before)
    axes[0].axvline(mean_before, color='red', linestyle='--', label=f'均值: {mean_before:.2f}')
    axes[0].legend()

    # BN 后的分布
    axes[1].hist(activations_after.flatten(), bins=50, alpha=0.7, color='green')
    axes[1].set_title(f'{layer_name} - BN 后')
    axes[1].set_xlabel('激活值')
    axes[1].set_ylabel('频数')

    # 添加统计信息
    mean_after = np.mean(activations_after)
    std_after = np.std(activations_after)
    axes[1].axvline(mean_after, color='red', linestyle='--', label=f'均值: {mean_after:.2f}')
    axes[1].legend()

    plt.suptitle('激活分布变化')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"激活分布已保存到 {save_path}")


def visualize_training_comparison(
    losses_with_bn: List[float],
    losses_without_bn: List[float],
    accs_with_bn: List[float],
    accs_without_bn: List[float],
    save_path: str = "bn_training_comparison.png"
) -> None:
    """
    可视化训练对比

    Args:
        losses_with_bn: 使用 BN 的损失曲线
        losses_without_bn: 不使用 BN 的损失曲线
        accs_with_bn: 使用 BN 的准确率曲线
        accs_without_bn: 不使用 BN 的准确率曲线
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失对比
    axes[0].plot(losses_with_bn, label='有 BN', color='blue')
    axes[0].plot(losses_without_bn, label='无 BN', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title('训练损失对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率对比
    axes[1].plot(accs_with_bn, label='有 BN', color='blue')
    axes[1].plot(accs_without_bn, label='无 BN', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('训练准确率对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练对比已保存到 {save_path}")


def visualize_running_statistics(
    bn_layer: BatchNorm1D,
    save_path: str = "bn_running_stats.png"
) -> None:
    """
    可视化运行统计量

    Args:
        bn_layer: 批归一化层
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 运行均值
    axes[0].bar(range(len(bn_layer.running_mean)), bn_layer.running_mean)
    axes[0].set_xlabel('特征索引')
    axes[0].set_ylabel('运行均值')
    axes[0].set_title('运行均值')

    # 运行方差
    axes[1].bar(range(len(bn_layer.running_var)), bn_layer.running_var)
    axes[1].set_xlabel('特征索引')
    axes[1].set_ylabel('运行方差')
    axes[1].set_title('运行方差')

    plt.suptitle('运行统计量')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"运行统计量已保存到 {save_path}")


# ============================================================================
# 简单网络用于对比实验
# ============================================================================

def create_simple_network(with_bn: bool = True):
    """
    创建简单的网络用于对比实验

    Args:
        with_bn: 是否使用批归一化

    Returns:
        网络组件字典
    """
    np.random.seed(42)

    # 简单的两层网络
    W1 = np.random.randn(10, 50).astype(np.float32) * 0.1
    b1 = np.zeros(50, dtype=np.float32)
    W2 = np.random.randn(50, 2).astype(np.float32) * 0.1
    b2 = np.zeros(2, dtype=np.float32)

    network = {
        'W1': W1, 'b1': b1,
        'W2': W2, 'b2': b2,
        'with_bn': with_bn
    }

    if with_bn:
        network['bn1'] = BatchNorm1D(50)

    return network


def forward_network(network: dict, x: np.ndarray) -> np.ndarray:
    """前向传播"""
    W1, b1 = network['W1'], network['b1']
    W2, b2 = network['W2'], network['b2']

    # 第一层
    z1 = x @ W1 + b1

    # BN
    if network['with_bn']:
        z1 = network['bn1'].forward(z1)

    # ReLU
    a1 = np.maximum(0, z1)

    # 第二层
    z2 = a1 @ W2 + b2

    return z2, a1, z1


# ============================================================================
# 演示和示例
# ============================================================================

def demo_bn_effect():
    """演示 BN 对激活分布的影响"""
    print("\n" + "=" * 60)
    print("BN 对激活分布的影响")
    print("=" * 60)

    np.random.seed(42)

    # 创建输入数据（不同分布）
    batch_size = 100
    features = 64

    # 输入有不同的均值和方差
    x = np.random.randn(batch_size, features).astype(np.float32) * 3 + 5

    print(f"输入统计: 均值={x.mean():.2f}, 方差={x.var():.2f}")

    # 应用 BN
    bn = BatchNorm1D(features)
    x_bn = bn.forward(x)

    print(f"BN 后统计: 均值={x_bn.mean():.2f}, 方差={x_bn.var():.2f}")

    # 可视化
    visualize_activation_distribution(x, x_bn, "特征层", "bn_effect_demo.png")


def demo_training_inference_difference():
    """演示训练和推理模式的差异"""
    print("\n" + "=" * 60)
    print("训练和推理模式的差异")
    print("=" * 60)

    np.random.seed(42)

    bn = BatchNorm1D(10)

    # 训练阶段
    print("\n训练阶段:")
    bn.training = True

    for i in range(5):
        x = np.random.randn(32, 10).astype(np.float32) * (i + 1) + i * 2
        y = bn.forward(x)
        print(f"  Batch {i+1}: 输入均值={x.mean():.2f}, 运行均值={bn.running_mean.mean():.2f}")

    # 推理阶段
    print("\n推理阶段:")
    bn.training = False

    x_test = np.random.randn(32, 10).astype(np.float32) * 2 + 3
    y_test = bn.forward(x_test)
    print(f"  测试输入均值={x_test.mean():.2f}")
    print(f"  使用运行均值={bn.running_mean.mean():.2f}")

    # 可视化运行统计量
    visualize_running_statistics(bn, "bn_train_infer_diff.png")


def explain_bn_math():
    """解释 BN 数学原理"""
    print("\n" + "=" * 60)
    print("Batch Normalization 数学原理")
    print("=" * 60)

    print("""
1. 为什么需要 BN？

   深层网络中，每层的输入分布会随着前层参数变化而变化，
   这称为内部协变量偏移（Internal Covariate Shift）。
   这导致：
   - 后层需要不断适应新的分布
   - 学习率需要很小
   - 初始化更敏感

2. BN 的数学公式

   对于 mini-batch B = {x_1, ..., x_m}:

   a) 计算批统计量:
      μ_B = 1/m * Σ x_i
      σ²_B = 1/m * Σ (x_i - μ_B)²

   b) 标准化:
      x̂_i = (x_i - μ_B) / √(σ²_B + ε)

   c) 缩放和平移:
      y_i = γ * x̂_i + β

   其中 γ 和 β 是可学习参数。

3. 为什么需要 γ 和 β？

   如果只有标准化，每层的输出都被限制在标准正态分布，
   这可能限制了网络的表达能力。
   γ 和 β 允许网络恢复原始分布（如果需要的话）：
   - 当 γ = σ, β = μ 时，可以完全恢复原始分布
   - 网络可以根据需要学习最优的分布

4. 训练和推理的区别

   训练时:
   - 使用当前 mini-batch 的统计量
   - 更新运行统计量

   推理时:
   - 使用训练时积累的运行统计量
   - 不更新任何参数

5. BN 的好处

   a) 允许使用更大的学习率
   b) 减少对初始化的敏感度
   c) 有正则化效果（批统计量的噪声）
   d) 加速收敛

6. BN 的局限

   a) 对小 batch size 敏感（统计量不准确）
   b) 对序列模型（RNN）不太适用
   c) 推理时依赖运行统计量
""")


def demo_bn_gradient():
    """演示 BN 的梯度计算"""
    print("\n" + "=" * 60)
    print("BN 梯度计算演示")
    print("=" * 60)

    np.random.seed(42)

    # 创建 BN 层
    bn = BatchNorm1D(10)

    # 前向传播
    x = np.random.randn(32, 10).astype(np.float32)
    y = bn.forward(x)

    # 模拟来自上层的梯度
    grad_output = np.ones_like(y)

    # 反向传播
    grad_input = bn.backward(grad_output)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"输出梯度形状: {grad_output.shape}")
    print(f"输入梯度形状: {grad_input.shape}")

    # 数值梯度检查
    print("\n梯度检查:")
    eps = 1e-5

    # 计算数值梯度
    numerical_grad = np.zeros_like(x)
    for i in range(min(3, x.shape[0])):  # 只检查前几个
        for j in range(min(3, x.shape[1])):
            x_plus = x.copy()
            x_plus[i, j] += eps
            bn_test = BatchNorm1D(10)
            bn_test.gamma = bn.gamma.copy()
            bn_test.beta = bn.beta.copy()
            y_plus = bn_test.forward(x_plus)

            x_minus = x.copy()
            x_minus[i, j] -= eps
            y_minus = bn_test.forward(x_minus)

            numerical_grad[i, j] = np.sum(y_plus - y_minus) / (2 * eps)

    print(f"解析梯度: {grad_input[0, 0]:.6f}")
    print(f"数值梯度: {numerical_grad[0, 0]:.6f}")


# ============================================================================
# 测试
# ============================================================================

def test_batchnorm1d_forward():
    """测试 BatchNorm1D 前向传播"""
    print("测试 BatchNorm1D 前向传播...")

    bn = BatchNorm1D(10)

    # 创建输入
    x = np.random.randn(32, 10).astype(np.float32)

    # 前向传播
    y = bn.forward(x)

    # 检查形状
    assert y.shape == x.shape, f"输出形状错误: {y.shape}"

    # 检查归一化效果（训练模式下）
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)

    # 由于有 gamma 和 beta，均值和方差不一定接近 0 和 1
    # 但运行统计量应该接近批统计量
    assert np.allclose(bn.running_mean, np.mean(x, axis=0), atol=1e-1), "运行均值更新错误"

    print("  ✓ 前向传播正确")


def test_batchnorm1d_backward():
    """测试 BatchNorm1D 反向传播"""
    print("测试 BatchNorm1D 反向传播...")

    bn = BatchNorm1D(10)

    # 前向传播
    x = np.random.randn(32, 10).astype(np.float32)
    y = bn.forward(x)

    # 反向传播
    grad_output = np.ones_like(y)
    grad_input = bn.backward(grad_output)

    # 检查形状
    assert grad_input.shape == x.shape, f"输入梯度形状错误: {grad_input.shape}"
    assert bn.grad_gamma.shape == bn.gamma.shape, "gamma 梯度形状错误"
    assert bn.grad_beta.shape == bn.beta.shape, "beta 梯度形状错误"

    print("  ✓ 反向传播正确")


def test_batchnorm2d_forward():
    """测试 BatchNorm2D 前向传播"""
    print("测试 BatchNorm2D 前向传播...")

    bn = BatchNorm2D(3)  # 3 通道

    # 创建输入 (batch, channels, height, width)
    x = np.random.randn(8, 3, 16, 16).astype(np.float32)

    # 前向传播
    y = bn.forward(x)

    # 检查形状
    assert y.shape == x.shape, f"输出形状错误: {y.shape}"

    print("  ✓ BatchNorm2D 前向传播正确")


def test_inference_mode():
    """测试推理模式"""
    print("测试推理模式...")

    bn = BatchNorm1D(10)

    # 训练阶段
    bn.training = True
    x_train = np.random.randn(32, 10).astype(np.float32) * 3 + 2
    y_train = bn.forward(x_train)

    # 保存运行统计量
    running_mean_after_train = bn.running_mean.copy()

    # 推理阶段
    bn.training = False
    x_test = np.random.randn(32, 10).astype(np.float32) * 3 + 2
    y_test = bn.forward(x_test)

    # 推理时运行统计量不应改变
    assert np.allclose(bn.running_mean, running_mean_after_train), "推理时运行统计量被修改"

    print("  ✓ 推理模式正确")


def test_gamma_beta_effect():
    """测试 gamma 和 beta 的效果"""
    print("测试 gamma 和 beta 的效果...")

    bn = BatchNorm1D(5)

    # 设置特定的 gamma 和 beta
    bn.gamma = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    bn.beta = np.array([0, 1, 2, 3, 4], dtype=np.float32)

    # 创建输入
    x = np.random.randn(10, 5).astype(np.float32)

    # 前向传播
    y = bn.forward(x)

    # 检查 gamma 和 beta 的效果
    # y = gamma * x_norm + beta
    x_norm = (x - np.mean(x, axis=0)) / np.sqrt(np.var(x, axis=0) + bn.eps)
    expected = bn.gamma * x_norm + bn.beta

    assert np.allclose(y, expected), "gamma 和 beta 效果不正确"

    print("  ✓ gamma 和 beta 正确")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行 Batch Normalization 测试")
    print("=" * 60)

    test_batchnorm1d_forward()
    test_batchnorm1d_backward()
    test_batchnorm2d_forward()
    test_inference_mode()
    test_gamma_beta_effect()

    print("\n所有测试通过!")


def main():
    """主函数"""
    print("=" * 60)
    print("Batch Normalization 原理详解")
    print("=" * 60)

    # 解释数学原理
    explain_bn_math()

    # 演示 BN 效果
    demo_bn_effect()

    # 演示训练/推理差异
    demo_training_inference_difference()

    # 演示梯度计算
    demo_bn_gradient()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    run_tests()

    # 运行主程序
    main()