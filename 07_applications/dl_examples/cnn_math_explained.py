#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN 数学原理详解

本模块从零实现卷积神经网络的核心组件，深入理解卷积运算的数学原理。

数学知识：
- 卷积运算数学原理
- 池化运算
- 特征图维度计算
- 反向传播中的梯度传播

实现内容：
- Conv2D: 2D 卷积层（前向和反向）
- MaxPool2D: 最大池化
- Flatten: 展平层
- 维度计算公式解释
- 可视化：卷积核、特征图、边缘检测效果
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy import signal


# ============================================================================
# 卷积层实现
# ============================================================================

class Conv2D:
    """
    2D 卷积层

    数学公式：
        前向传播: output[b, c_out, h_out, w_out] =
            sum over (c_in, kh, kw) of
            input[b, c_in, h+kh, w+kw] * kernel[c_out, c_in, kh, kw]

        输出维度计算：
            h_out = (h_in + 2*padding - kernel_h) / stride + 1
            w_out = (w_in + 2*padding - kernel_w) / stride + 1

    参数初始化：He 初始化
        W ~ N(0, sqrt(2/(in_channels * kernel_h * kernel_w)))
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0):
        """
        初始化卷积层

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小（正方形）
            stride: 步长
            padding: 填充
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He 初始化
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32) * scale
        self.bias = np.zeros(out_channels, dtype=np.float32)

        # 梯度
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        # 缓存
        self.input = None
        self.col = None  # im2col 结果

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入张量 (batch, in_channels, height, width)

        Returns:
            输出张量 (batch, out_channels, height_out, width_out)
        """
        self.input = x
        batch_size, _, h_in, w_in = x.shape

        # 计算输出维度
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 填充
        if self.padding > 0:
            x_padded = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            x_padded = x

        # im2col 变换
        self.col = self._im2col(x_padded, h_out, w_out)

        # 重塑权重为 (out_channels, in_channels * kernel_size * kernel_size)
        weight_col = self.weight.reshape(self.out_channels, -1)

        # 矩阵乘法
        out = self.col @ weight_col.T + self.bias

        # 重塑输出
        out = out.reshape(batch_size, h_out, w_out, self.out_channels)
        out = out.transpose(0, 3, 1, 2)  # (batch, out_channels, h_out, w_out)

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            grad_output: 上层梯度 (batch, out_channels, h_out, w_out)

        Returns:
            对输入的梯度 (batch, in_channels, h_in, w_in)
        """
        batch_size, _, h_out, w_out = grad_output.shape

        # 重塑梯度
        grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # 计算偏置梯度
        self.grad_bias += np.sum(grad_output_reshaped, axis=0)

        # 计算权重梯度
        self.grad_weight += (grad_output_reshaped.T @ self.col).reshape(self.weight.shape)

        # 计算输入梯度
        weight_col = self.weight.reshape(self.out_channels, -1)
        grad_col = grad_output_reshaped @ weight_col

        # col2im 变换
        grad_input = self._col2im(grad_col, batch_size)

        # 移除填充
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return grad_input

    def _im2col(self, x: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
        """
        将输入转换为列矩阵

        Args:
            x: 填充后的输入 (batch, in_channels, h_padded, w_padded)
            h_out, w_out: 输出维度

        Returns:
            col: 列矩阵 (batch * h_out * w_out, in_channels * kernel_size * kernel_size)
        """
        batch_size, _, h_padded, w_padded = x.shape
        col = np.zeros((batch_size, self.in_channels, self.kernel_size, self.kernel_size, h_out, w_out))

        for i in range(self.kernel_size):
            i_max = i + self.stride * h_out
            for j in range(self.kernel_size):
                j_max = j + self.stride * w_out
                col[:, :, i, j, :, :] = x[:, :, i:i_max:self.stride, j:j_max:self.stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(-1, self.in_channels * self.kernel_size * self.kernel_size)
        return col

    def _col2im(self, col: np.ndarray, batch_size: int) -> np.ndarray:
        """
        将列矩阵转换回图像格式

        Args:
            col: 列矩阵 (batch * h_out * w_out, in_channels * kernel_size * kernel_size)
            batch_size: 批量大小

        Returns:
            img: 图像张量 (batch, in_channels, h_padded, w_padded)
        """
        h_padded = self.input.shape[2] + 2 * self.padding
        w_padded = self.input.shape[3] + 2 * self.padding
        h_out = (h_padded - self.kernel_size) // self.stride + 1
        w_out = (w_padded - self.kernel_size) // self.stride + 1

        col = col.reshape(batch_size, h_out, w_out, self.in_channels, self.kernel_size, self.kernel_size)
        col = col.transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((batch_size, self.in_channels, h_padded, w_padded))

        for i in range(self.kernel_size):
            i_max = i + self.stride * h_out
            for j in range(self.kernel_size):
                j_max = j + self.stride * w_out
                img[:, :, i:i_max:self.stride, j:j_max:self.stride] += col[:, :, i, j, :, :]

        return img

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回参数和梯度"""
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]

    def zero_grad(self) -> None:
        """清零梯度"""
        self.grad_weight.fill(0)
        self.grad_bias.fill(0)


class MaxPool2D:
    """
    最大池化层

    数学公式：
        output[b, c, i, j] = max over (m, n) of
                            input[b, c, i*stride+m, j*stride+n]

    前向传播时记录最大值位置，反向传播时只将梯度传给最大值位置。
    """

    def __init__(self, pool_size: int = 2, stride: int = None):
        """
        初始化池化层

        Args:
            pool_size: 池化窗口大小
            stride: 步长（默认等于 pool_size）
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.input = None
        self.max_indices = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入张量 (batch, channels, height, width)

        Returns:
            输出张量 (batch, channels, height_out, width_out)
        """
        self.input = x
        batch_size, channels, h_in, w_in = x.shape

        h_out = (h_in - self.pool_size) // self.stride + 1
        w_out = (w_in - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, channels, h_out, w_out))
        self.max_indices = np.zeros((batch_size, channels, h_out, w_out, 2), dtype=np.int32)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                window = x[:, :, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]

                output[:, :, i, j] = np.max(window, axis=(2, 3))

                # 记录最大值位置
                for b in range(batch_size):
                    for c in range(channels):
                        idx = np.unravel_index(np.argmax(window[b, c]), (self.pool_size, self.pool_size))
                        self.max_indices[b, c, i, j] = [h_start + idx[0], w_start + idx[1]]

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            grad_output: 上层梯度 (batch, channels, h_out, w_out)

        Returns:
            对输入的梯度 (batch, channels, h_in, w_in)
        """
        batch_size, channels, h_in, w_in = self.input.shape
        grad_input = np.zeros_like(self.input)

        for i in range(grad_output.shape[2]):
            for j in range(grad_output.shape[3]):
                for b in range(batch_size):
                    for c in range(channels):
                        h_idx, w_idx = self.max_indices[b, c, i, j]
                        grad_input[b, c, h_idx, w_idx] += grad_output[b, c, i, j]

        return grad_input


class Flatten:
    """
    展平层

    将多维输入展平为一维向量。
    前向传播时记录原始形状用于反向传播。
    """

    def __init__(self):
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入张量 (batch, ...)

        Returns:
            展平后的张量 (batch, features)
        """
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播

        Args:
            grad_output: 上层梯度 (batch, features)

        Returns:
            重塑后的梯度 (batch, ...)
        """
        return grad_output.reshape(self.input_shape)


# ============================================================================
# 可视化工具
# ============================================================================

def visualize_convolution(image: np.ndarray, kernel: np.ndarray, title: str = "卷积运算") -> None:
    """
    可视化卷积过程

    Args:
        image: 输入图像 (h, w) 或 (h, w, c)
        kernel: 卷积核 (kh, kw) 或 (out_ch, in_ch, kh, kw)
        title: 图标题
    """
    if len(image.shape) == 2:
        image = image[np.newaxis, np.newaxis, :, :]
    elif len(image.shape) == 3:
        image = image.transpose(2, 0, 1)[np.newaxis, :, :, :]

    if len(kernel.shape) == 2:
        kernel = kernel[np.newaxis, np.newaxis, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 显示输入图像
    axes[0].imshow(image[0, 0], cmap='gray')
    axes[0].set_title('输入图像')
    axes[0].axis('off')

    # 显示卷积核
    if kernel.shape[0] == 1:
        axes[1].imshow(kernel[0, 0], cmap='RdBu')
    else:
        # 显示多个卷积核
        n_kernels = min(kernel.shape[0], 4)
        kernel_display = np.zeros((kernel.shape[2] * 2, kernel.shape[3] * 2))
        for i in range(min(n_kernels, 4)):
            r, c = i // 2, i % 2
            kernel_display[r * kernel.shape[2]:(r + 1) * kernel.shape[2],
                          c * kernel.shape[3]:(c + 1) * kernel.shape[3]] = kernel[i, 0]
        axes[1].imshow(kernel_display, cmap='RdBu')
    axes[1].set_title('卷积核')
    axes[1].axis('off')

    # 计算并显示卷积结果
    conv = Conv2D(image.shape[1], kernel.shape[0], kernel.shape[2])
    conv.weight = kernel
    output = conv.forward(image.astype(np.float32))

    axes[2].imshow(output[0, 0], cmap='gray')
    axes[2].set_title('输出特征图')
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('cnn_convolution_visual.png', dpi=150)
    plt.close()
    print(f"卷积可视化已保存到 cnn_convolution_visual.png")


def create_edge_detection_kernels() -> dict:
    """
    创建常见的边缘检测卷积核

    Returns:
        字典：名称 -> 卷积核
    """
    kernels = {}

    # Sobel 边缘检测
    kernels['Sobel X'] = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    kernels['Sobel Y'] = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    # Laplacian 边缘检测
    kernels['Laplacian'] = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    # 锐化
    kernels['Sharpen'] = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # 高斯模糊
    kernels['Gaussian'] = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16

    return kernels


def visualize_feature_maps(feature_maps: np.ndarray, title: str = "特征图") -> None:
    """
    可视化特征图

    Args:
        feature_maps: 特征图 (channels, height, width) 或 (batch, channels, height, width)
        title: 图标题
    """
    if len(feature_maps.shape) == 4:
        feature_maps = feature_maps[0]

    n_channels = min(feature_maps.shape[0], 16)
    n_cols = 4
    n_rows = (n_channels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))

    for i in range(n_channels):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(feature_maps[i], cmap='gray')
        ax.set_title(f'通道 {i}')
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(n_channels, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('cnn_feature_maps.png', dpi=150)
    plt.close()
    print(f"特征图已保存到 cnn_feature_maps.png")


def visualize_pooling(image: np.ndarray, pool_size: int = 2) -> None:
    """
    可视化池化效果

    Args:
        image: 输入图像 (height, width)
        pool_size: 池化窗口大小
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 原始图像
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'原始图像 {image.shape}')
    axes[0].axis('off')

    # 最大池化
    max_pool = MaxPool2D(pool_size)
    pooled = max_pool.forward(image[np.newaxis, np.newaxis, :, :].astype(np.float32))[0, 0]
    axes[1].imshow(pooled, cmap='gray')
    axes[1].set_title(f'最大池化 {pooled.shape}')
    axes[1].axis('off')

    # 平均池化
    from scipy.ndimage import uniform_filter
    avg_pooled = uniform_filter(image, size=pool_size)[::pool_size, ::pool_size]
    axes[2].imshow(avg_pooled, cmap='gray')
    axes[2].set_title(f'平均池化 {avg_pooled.shape}')
    axes[2].axis('off')

    plt.suptitle('池化效果对比')
    plt.tight_layout()
    plt.savefig('cnn_pooling_visual.png', dpi=150)
    plt.close()
    print(f"池化可视化已保存到 cnn_pooling_visual.png")


def explain_dimension_calculation():
    """解释维度计算公式"""
    print("\n" + "=" * 60)
    print("CNN 维度计算公式")
    print("=" * 60)

    print("""
卷积层输出维度：
    h_out = floor((h_in + 2*padding - kernel_h) / stride) + 1
    w_out = floor((w_in + 2*padding - kernel_w) / stride) + 1

示例：
    输入: 224x224x3
    卷积: kernel=3, stride=1, padding=1
    输出: (224 + 2*1 - 3) / 1 + 1 = 224x224xout_channels

    输入: 224x224x3
    卷积: kernel=3, stride=2, padding=1
    输出: (224 + 2*1 - 3) / 2 + 1 = 112x112xout_channels

池化层输出维度：
    h_out = floor((h_in - pool_size) / stride) + 1
    w_out = floor((w_in - pool_size) / stride) + 1

感受野计算：
    RF(layer) = RF(prev) + (kernel_size - 1) * stride_prev
""")


# ============================================================================
# 示例和演示
# ============================================================================

def create_sample_image() -> np.ndarray:
    """创建示例图像"""
    image = np.zeros((28, 28), dtype=np.float32)

    # 画一个简单的数字 "7"
    image[4:6, 4:24] = 1  # 顶部横线
    image[4:20, 22:24] = 1  # 右侧斜线

    return image


def demo_edge_detection():
    """演示边缘检测"""
    print("\n" + "=" * 60)
    print("边缘检测演示")
    print("=" * 60)

    # 创建示例图像
    image = create_sample_image()

    # 获取边缘检测卷积核
    kernels = create_edge_detection_kernels()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 原始图像
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    # 应用各种卷积核
    for i, (name, kernel) in enumerate(kernels.items()):
        conv = Conv2D(1, 1, 3)
        conv.weight = kernel[np.newaxis, np.newaxis, :, :]
        output = conv.forward(image[np.newaxis, np.newaxis, :, :])[0, 0]

        ax = axes[(i + 1) // 3, (i + 1) % 3]
        ax.imshow(output, cmap='gray')
        ax.set_title(f'{name}')
        ax.axis('off')

    plt.suptitle('边缘检测卷积核效果')
    plt.tight_layout()
    plt.savefig('cnn_edge_detection.png', dpi=150)
    plt.close()
    print("边缘检测结果已保存到 cnn_edge_detection.png")


def demo_feature_extraction():
    """演示特征提取"""
    print("\n" + "=" * 60)
    print("特征提取演示")
    print("=" * 60)

    # 创建示例图像
    image = create_sample_image()

    # 创建多个随机卷积核
    n_filters = 8
    conv = Conv2D(1, n_filters, 3, padding=1)
    output = conv.forward(image[np.newaxis, np.newaxis, :, :].astype(np.float32))

    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # 原始图像
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('输入')
    axes[0, 0].axis('off')

    # 卷积核
    for i in range(n_filters):
        if i < 4:
            ax = axes[0, i + 1]
        else:
            ax = axes[1, i - 4]

        ax.imshow(conv.weight[i, 0], cmap='RdBu')
        ax.set_title(f'卷积核 {i}')
        ax.axis('off')

    # 隐藏空子图
    axes[1, 4].axis('off')

    plt.suptitle('卷积核可视化')
    plt.tight_layout()
    plt.savefig('cnn_filters.png', dpi=150)
    plt.close()
    print("卷积核可视化已保存到 cnn_filters.png")

    # 特征图
    visualize_feature_maps(output[0], "卷积层特征图")


# ============================================================================
# 测试
# ============================================================================

def test_conv2d_forward():
    """测试 Conv2D 前向传播"""
    print("测试 Conv2D 前向传播...")

    np.random.seed(42)

    # 创建卷积层
    conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)

    # 创建输入
    x = np.random.randn(1, 1, 5, 5).astype(np.float32)

    # 前向传播
    output = conv.forward(x)

    # 检查输出形状
    expected_h = (5 - 3) // 1 + 1  # 3
    expected_w = (5 - 3) // 1 + 1  # 3
    assert output.shape == (1, 2, expected_h, expected_w), f"输出形状错误: {output.shape}"

    print("  ✓ 输出形状正确")


def test_conv2d_backward():
    """测试 Conv2D 反向传播"""
    print("测试 Conv2D 反向传播...")

    np.random.seed(42)

    # 创建卷积层
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3)

    # 创建输入
    x = np.random.randn(1, 1, 5, 5).astype(np.float32)

    # 前向传播
    output = conv.forward(x)

    # 反向传播
    grad_output = np.ones_like(output)
    grad_input = conv.backward(grad_output)

    # 检查梯度形状
    assert grad_input.shape == x.shape, f"输入梯度形状错误: {grad_input.shape}"
    assert conv.grad_weight.shape == conv.weight.shape, "权重梯度形状错误"

    print("  ✓ 反向传播形状正确")


def test_maxpool2d():
    """测试 MaxPool2D"""
    print("测试 MaxPool2D...")

    # 创建池化层
    pool = MaxPool2D(pool_size=2, stride=2)

    # 创建输入
    x = np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=np.float32)

    # 前向传播
    output = pool.forward(x)

    # 检查输出
    expected = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)
    assert np.allclose(output, expected), f"最大池化结果错误: {output}"

    print("  ✓ 最大池化正确")


def test_flatten():
    """测试 Flatten"""
    print("测试 Flatten...")

    flatten = Flatten()

    # 创建输入
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)

    # 前向传播
    output = flatten.forward(x)

    # 检查输出形状
    assert output.shape == (2, 48), f"展平后形状错误: {output.shape}"

    # 反向传播
    grad_output = np.ones_like(output)
    grad_input = flatten.backward(grad_output)

    # 检查梯度形状
    assert grad_input.shape == x.shape, f"梯度形状错误: {grad_input.shape}"

    print("  ✓ 展平层正确")


def test_dimension_calculation():
    """测试维度计算"""
    print("测试维度计算...")

    # 测试用例：(h_in, kernel, stride, padding) -> h_out
    test_cases = [
        (224, 3, 1, 1, 224),  # padding='same'
        (224, 3, 2, 1, 112),  # stride=2
        (224, 7, 2, 3, 112),  # 7x7 kernel
        (28, 3, 1, 0, 26),    # MNIST
    ]

    for h_in, kernel, stride, padding, expected in test_cases:
        h_out = (h_in + 2 * padding - kernel) // stride + 1
        assert h_out == expected, f"维度计算错误: {h_in} -> {h_out}, 期望 {expected}"

    print("  ✓ 维度计算正确")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行 CNN 测试")
    print("=" * 60)

    test_conv2d_forward()
    test_conv2d_backward()
    test_maxpool2d()
    test_flatten()
    test_dimension_calculation()

    print("\n所有测试通过!")


def main():
    """主函数"""
    print("=" * 60)
    print("CNN 数学原理详解")
    print("=" * 60)

    # 解释维度计算
    explain_dimension_calculation()

    # 演示边缘检测
    demo_edge_detection()

    # 演示特征提取
    demo_feature_extraction()

    # 演示池化
    image = create_sample_image()
    visualize_pooling(image)

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    run_tests()

    # 运行主程序
    main()