# 07 综合应用 (Applications)

## 概述

这个目录包含深度学习和强化学习的综合应用案例，将前面学习的数学知识整合到实际项目中。每个案例都从零实现，帮助理解底层原理。

## 目录结构

```
07_applications/
├── README.md                   # 本文件
├── dl_examples/               # 深度学习案例
│   ├── README.md              # 深度学习案例概述
│   ├── mnist_from_scratch.py  # MNIST 从零实现
│   ├── cnn_math_explained.py  # CNN 数学原理详解
│   ├── attention_mechanism.py # 注意力机制详解
│   └── batch_norm_explained.py# Batch Normalization 原理
└── rl_examples/               # 强化学习案例
    ├── README.md              # 强化学习案例概述
    ├── q_learning.py          # Q-Learning 算法
    ├── policy_gradient.py     # REINFORCE 算法
    ├── actor_critic.py        # Actor-Critic 算法
    └── dqn.py                 # Deep Q-Network
```

## 深度学习案例

### [dl_examples/](./dl_examples/)

每个案例都会：
1. **明确数学背景** - 使用了哪些数学知识
2. **从零实现** - 不依赖高层API，理解底层原理
3. **可视化** - 展示中间过程和结果
4. **测试验证** - 内置测试函数验证正确性

#### 已实现的案例

| 案例 | 文件 | 数学知识 | 行数 |
|------|------|---------|------|
| MNIST 从零实现 | mnist_from_scratch.py | 矩阵乘法、反向传播、交叉熵、SGD | ~500 |
| CNN 数学原理 | cnn_math_explained.py | 卷积运算、池化、梯度传播 | ~450 |
| 注意力机制 | attention_mechanism.py | 矩阵乘法、Softmax、缩放点积 | ~400 |
| Batch Normalization | batch_norm_explained.py | 均值方差、标准化、梯度计算 | ~350 |

## 强化学习案例

### [rl_examples/](./rl_examples/)

每个案例都会：
1. **MDP建模** - 定义状态、动作、奖励
2. **算法实现** - 从零实现RL算法
3. **训练可视化** - 展示学习过程
4. **策略分析** - 理解学到的策略

#### 已实现的案例

| 案例 | 文件 | 数学知识 | 行数 |
|------|------|---------|------|
| Q-Learning | q_learning.py | 贝尔曼方程、值函数、TD学习 | ~400 |
| Policy Gradient | policy_gradient.py | 策略梯度定理、蒙特卡洛方法 | ~450 |
| Actor-Critic | actor_critic.py | 优势函数、Actor-Critic架构 | ~400 |
| DQN | dqn.py | 神经网络、经验回放、目标网络 | ~450 |

## 学习建议

### 深度学习路径
```
第一阶段：基础
1. mnist_from_scratch.py    → 理解神经网络基本原理
2. cnn_math_explained.py    → 理解卷积网络

第二阶段：进阶
3. attention_mechanism.py   → 理解注意力机制
4. batch_norm_explained.py  → 理解归一化技术
```

### 强化学习路径
```
第一阶段：表格法
1. q_learning.py            → 理解值函数学习
2. policy_gradient.py       → 理解策略优化

第二阶段：深度RL
3. dqn.py                   → 值函数神经网络近似
4. actor_critic.py          → 策略和值函数结合
```

## 实践原则

### 1. 先手动实现，再用框架
```python
# 第一遍：手动实现理解原理
def manual_forward(x, w, b):
    return x @ w + b

def manual_backward(x, w, grad_output):
    grad_w = x.T @ grad_output
    grad_x = grad_output @ w.T
    return grad_x, grad_w

# 第二遍：用PyTorch验证
import torch.nn as nn
linear = nn.Linear(in_features, out_features)
```

### 2. 可视化一切
- 损失曲线
- 参数分布
- 梯度流动
- 中间特征图
- 策略演化

### 3. 对比实验
- 不同超参数的影响
- 不同架构的对比
- 数学公式的实际效果

## 代码统计

| 模块 | 文件数 | 总行数 |
|------|--------|--------|
| dl_examples | 5 | ~2,100 |
| rl_examples | 5 | ~1,700 |
| **总计** | **10** | **~3,800** |

## 依赖环境

```bash
# 基础依赖（所有案例）
pip install numpy matplotlib

# 可选：用于对比验证
pip install torch torchvision
```

注：所有核心代码只依赖 NumPy 和 Matplotlib，无需 GPU。

## 如何使用这些案例

### 作为学习材料
1. 先阅读对应的数学模块（01-06）
2. 运行代码，观察输出
3. 修改代码，做实验验证理解
4. 尝试改进或扩展

### 作为参考实现
1. 查找类似问题的案例
2. 理解数学原理和实现细节
3. 迁移到自己的项目中

### 作为调试工具
1. 对比自己的实现
2. 检查中间结果是否一致
3. 理解为什么某些设计是必要的

## 与数学模块的对应关系

| 应用案例 | 数学模块 | 核心知识点 |
|---------|---------|-----------|
| mnist_from_scratch.py | 01, 02, 03, 04 | 矩阵乘法、反向传播、交叉熵、SGD |
| cnn_math_explained.py | 01, 02 | 卷积运算、池化、梯度传播 |
| attention_mechanism.py | 01, 03 | 矩阵乘法、Softmax、缩放点积 |
| batch_norm_explained.py | 03, 04 | 均值方差、标准化、梯度计算 |
| q_learning.py | 06_rl_math | 马尔可夫决策过程、贝尔曼方程、值迭代 |
| policy_gradient.py | 06_rl_math | 策略梯度定理、蒙特卡洛方法 |
| actor_critic.py | 06_rl_math | 优势函数、Actor-Critic 架构 |
| dqn.py | 01-04, 06_rl_math | 神经网络、优化、值函数近似 |

## 核心概念速查

### 深度学习

**前向传播**：
```
y = f(x; θ)
```

**反向传播**：
```
∂L/∂θ = ∂L/∂y · ∂y/∂θ
```

**梯度下降**：
```
θ ← θ - α · ∂L/∂θ
```

### 强化学习

**马尔可夫决策过程 (MDP)**：
- **状态 (S)**: 环境的表示
- **动作 (A)**: 智能体的行为选择
- **转移概率 (P)**: P(s'|s,a)
- **奖励 (R)**: 即时反馈
- **折扣因子 (γ)**: 未来奖励的权重

**值函数**：
- **状态值函数**: V(s) = E[G_t | S_t = s]
- **动作值函数**: Q(s,a) = E[G_t | S_t = s, A_t = a]
- **贝尔曼方程**: V(s) = R + γΣP(s'|s,a)V(s')

**策略梯度**：
- **目标**: 最大化期望回报 J(θ) = E[Σγ^t r_t]
- **梯度**: ∇J(θ) = E[∇log π(a|s) * Q(s,a)]
- **REINFORCE**: θ = θ + α * ∇log π(a|s) * G_t

## 相关资源

### 深度学习
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Dive into Deep Learning](https://d2l.ai/)

### 强化学习
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)

## 下一步

选择一个感兴趣的案例开始实践，或者返回 [主页](../README.md) 查看完整的学习路径。

---

*最后更新：2026-03-01*
*更新内容：完成 07_applications 模块，包含 4 个深度学习案例和 4 个强化学习案例*