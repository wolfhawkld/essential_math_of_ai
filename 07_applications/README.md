# 07 综合应用 (Applications)

## 概述

这个目录包含深度学习和强化学习的综合应用案例，将前面学习的数学知识整合到实际项目中。

## 目录结构

```
07_applications/
├── dl_examples/          # 深度学习案例
│   ├── mnist_from_scratch/
│   ├── cnn_math_explained/
│   ├── transformer_attention/
│   └── ...
│
└── rl_examples/          # 强化学习案例
    ├── q_learning/
    ├── policy_gradient/
    ├── actor_critic/
    └── ...
```

## 深度学习案例

### [dl_examples/](./dl_examples/)

每个案例都会：
1. **明确数学背景** - 使用了哪些数学知识
2. **从零实现** - 不依赖高层API，理解底层原理
3. **可视化** - 展示中间过程和结果
4. **对比实验** - 与标准实现对比验证

#### 计划中的案例
- **mnist_from_scratch.ipynb** - 从零实现神经网络
  - 用到：线性代数（矩阵乘法）、微积分（反向传播）
- **cnn_math_explained.ipynb** - CNN的数学原理
  - 用到：卷积运算、池化、特征图维度计算
- **attention_mechanism.ipynb** - 注意力机制详解
  - 用到：矩阵乘法、Softmax、线性变换
- **batch_norm_explained.ipynb** - Batch Normalization原理
  - 用到：均值、方差、标准化

## 强化学习案例

### [rl_examples/](./rl_examples/)

每个案例都会：
1. **MDP建模** - 定义状态、动作、奖励
2. **算法实现** - 从零实现RL算法
3. **训练可视化** - 展示学习过程
4. **策略分析** - 理解学到的策略

#### 计划中的案例
- **q_learning.ipynb** - Q-Learning算法
  - 环境：GridWorld, FrozenLake
  - 用到：贝尔曼方程、值函数
- **policy_gradient.ipynb** - REINFORCE算法
  - 环境：CartPole
  - 用到：策略梯度、蒙特卡洛采样
- **actor_critic.ipynb** - Actor-Critic算法
  - 环境：LunarLander
  - 用到：优势函数、策略和值函数的结合
- **dqn.ipynb** - Deep Q-Network
  - 环境：Atari Pong
  - 用到：神经网络、经验回放、目标网络

## 学习建议

### 深度学习路径
```
第一阶段：基础
1. mnist_from_scratch → 理解神经网络基本原理
2. cnn_math_explained → 理解卷积网络

第二阶段：进阶
3. attention_mechanism → 理解注意力机制
4. transformer_from_scratch → 理解Transformer架构

第三阶段：应用
5. 选择感兴趣的领域深入（CV/NLP/等）
```

### 强化学习路径
```
第一阶段：表格法
1. q_learning → 理解值函数学习
2. policy_gradient → 理解策略优化

第二阶段：深度RL
3. dqn → 值函数神经网络近似
4. actor_critic → 策略和值函数结合

第三阶段：高级算法
5. ppo, sac等现代算法
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

## 依赖环境

```bash
# 深度学习
pip install torch torchvision matplotlib numpy

# 强化学习
pip install gym gymnasium numpy matplotlib

# 可视化
pip install tensorboard seaborn plotly
```

## 如何使用这些案例

### 作为学习材料
1. 先阅读对应的数学模块（01-06）
2. 运行notebook，观察输出
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

## 贡献新案例

欢迎贡献新的应用案例！每个案例应包含：

### 必需内容
- [ ] README.md - 案例简介和数学背景
- [ ] 完整的notebook或Python脚本
- [ ] 数学公式推导（简要）
- [ ] 代码注释（关键部分）

### 推荐内容
- [ ] 可视化结果
- [ ] 超参数敏感性分析
- [ ] 与标准实现的对比
- [ ] 常见问题和调试技巧

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
