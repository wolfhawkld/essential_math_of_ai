# 深度学习应用案例

## 概述

本目录包含深度学习的综合应用案例，将线性代数、微积分、概率统计、优化等数学知识整合到实际项目中。每个案例都从零实现，帮助理解底层原理。

## 案例列表

### 1. [mnist_from_scratch.py](./mnist_from_scratch.py) - 从零实现神经网络

**数学知识**：
- 线性代数：矩阵乘法、向量运算
- 微积分：反向传播、梯度计算
- 概率：Softmax、交叉熵损失
- 优化：SGD、学习率调度

**实现内容**：
- 数据加载（简化版 MNIST）
- 神经网络层：Linear、ReLU、Softmax
- 前向传播和反向传播
- 训练循环和评估
- 可视化：损失曲线、准确率、样本预测

**学习目标**：
- 理解神经网络的基本构建块
- 掌握反向传播的数学原理
- 实现一个可运行的图像分类器

### 2. [cnn_math_explained.py](./cnn_math_explained.py) - CNN 数学原理详解

**数学知识**：
- 卷积运算数学原理
- 池化运算
- 特征图维度计算
- 反向传播中的梯度传播

**实现内容**：
- `Conv2D`: 2D 卷积层（前向和反向）
- `MaxPool2D`: 最大池化
- `Flatten`: 展平层
- 维度计算公式解释
- 可视化：卷积核、特征图、边缘检测效果

**学习目标**：
- 理解卷积运算的数学本质
- 掌握 CNN 中的维度变化
- 可视化理解特征提取过程

### 3. [attention_mechanism.py](./attention_mechanism.py) - 注意力机制详解

**数学知识**：
- 矩阵乘法（Q, K, V 计算）
- Softmax 归一化
- 缩放点积注意力
- 多头注意力

**实现内容**：
- `scaled_dot_product_attention`: 基础注意力
- `MultiHeadAttention`: 多头注意力
- 可视化：注意力权重热力图
- 示例：简单的序列到序列任务

**学习目标**：
- 理解注意力机制的核心思想
- 掌握 Query-Key-Value 的计算
- 可视化注意力权重的含义

### 4. [batch_norm_explained.py](./batch_norm_explained.py) - Batch Normalization 原理

**数学知识**：
- 均值、方差计算
- 标准化公式
- 可学习参数（γ, β）
- 训练/推理模式差异

**实现内容**：
- `BatchNorm1D`: 一维批归一化
- `BatchNorm2D`: 二维批归一化（用于 CNN）
- 可视化：激活分布变化
- 对比实验：有无 BN 的训练效果

**学习目标**：
- 理解批归一化的数学原理
- 掌握训练和推理时的差异
- 理解 BN 如何加速训练

## 学习路径

```
阶段一：基础
├── mnist_from_scratch.py     # 神经网络基础
└── cnn_math_explained.py      # 卷积网络原理

阶段二：进阶
├── attention_mechanism.py     # 注意力机制
└── batch_norm_explained.py    # 归一化技术
```

## 代码特点

### 1. 从零实现
所有核心算法都使用 NumPy 从零实现，不依赖 PyTorch 或 TensorFlow 等框架：
```python
# 手动实现前向传播
def forward(self, x):
    return x @ self.W + self.b

# 手动实现反向传播
def backward(self, grad_output):
    self.grad_W = self.x.T @ grad_output
    self.grad_b = np.sum(grad_output, axis=0)
    return grad_output @ self.W.T
```

### 2. 数学公式对照
代码注释中包含对应的数学公式：
```python
# 前向传播: y = xW + b
# 反向传播:
#   ∂L/∂W = x.T @ ∂L/∂y
#   ∂L/∂b = sum(∂L/∂y, axis=0)
#   ∂L/∂x = ∂L/∂y @ W.T
```

### 3. 完整的可视化
每个案例都包含可视化函数：
- 训练曲线（损失、准确率）
- 中间结果（特征图、注意力权重）
- 对比实验（不同配置的效果）

## 运行方式

```bash
# 运行单个案例
python mnist_from_scratch.py
python cnn_math_explained.py
python attention_mechanism.py
python batch_norm_explained.py

# 所有案例都会：
# 1. 运行内置测试验证正确性
# 2. 执行训练/演示
# 3. 生成可视化图表
```

## 依赖环境

```bash
pip install numpy matplotlib
```

注：所有代码只依赖 NumPy 和 Matplotlib，无需 GPU。

## 与数学模块的对应关系

| 案例文件 | 数学模块 | 核心知识点 |
|---------|---------|-----------|
| mnist_from_scratch.py | 01, 02, 03, 04 | 矩阵乘法、反向传播、交叉熵、SGD |
| cnn_math_explained.py | 01, 02 | 卷积运算、池化、梯度传播 |
| attention_mechanism.py | 01, 03 | 矩阵乘法、Softmax、缩放点积 |
| batch_norm_explained.py | 03, 04 | 均值方差、标准化、梯度计算 |

## 常见问题

### Q: 为什么不使用 PyTorch/TensorFlow？
A: 从零实现能帮助理解底层原理。理解原理后，使用框架会更得心应手。

### Q: 这些实现能用于生产环境吗？
A: 这些实现主要用于教学。生产环境建议使用优化过的框架。

### Q: 训练速度慢怎么办？
A: 教学实现优先考虑可读性。可以减少训练轮数或使用更小的网络结构进行快速实验。

## 相关资源

- [01_linear_algebra](../01_linear_algebra/) - 线性代数基础
- [02_calculus](../02_calculus/) - 微积分与自动微分
- [03_probability_statistics](../03_probability_statistics/) - 概率与统计
- [04_optimization](../04_optimization/) - 优化算法