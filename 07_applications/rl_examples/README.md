# 强化学习应用案例

## 概述

本目录包含强化学习的综合应用案例，将马尔可夫决策过程、贝尔曼方程、策略梯度等数学知识整合到实际项目中。每个案例都从零实现，帮助理解底层原理。

## 案例列表

### 1. [q_learning.py](./q_learning.py) - Q-Learning 算法

**数学知识**：
- Q 值函数
- 贝尔曼方程
- TD 学习
- ε-greedy 探索

**实现内容**：
- `QLearningAgent`: Q-Learning 智能体
- `GridWorld`: 网格世界环境
- Q 表更新
- 可视化：Q 值热力图、策略演化

**学习目标**：
- 理解值函数的含义
- 掌握 TD 学习的更新规则
- 实现一个完整的 Q-Learning 算法

### 2. [policy_gradient.py](./policy_gradient.py) - REINFORCE 算法

**数学知识**：
- 策略梯度定理
- REINFORCE 算法
- 蒙特卡洛采样
- 基线减少方差

**实现内容**：
- `PolicyNetwork`: 策略网络
- `REINFORCEAgent`: REINFORCE 智能体
- `CartPole` 简化环境或自定义环境
- 可视化：训练曲线、策略分析

**学习目标**：
- 理解策略梯度的数学推导
- 掌握蒙特卡洛策略梯度
- 理解基线如何减少方差

### 3. [actor_critic.py](./actor_critic.py) - Actor-Critic 算法

**数学知识**：
- Actor-Critic 架构
- 优势函数
- 值函数估计
- 策略梯度

**实现内容**：
- `ActorNetwork`: 策略网络
- `CriticNetwork`: 值网络
- `A2CAgent`: Actor-Critic 智能体
- 可视化：Actor 和 Critic 学习过程

**学习目标**：
- 理解 Actor-Critic 架构
- 掌握优势函数的计算
- 实现 A2C 算法

### 4. [dqn.py](./dqn.py) - Deep Q-Network

**数学知识**：
- Q 值函数神经网络近似
- 经验回放
- 目标网络
- TD 误差

**实现内容**：
- `QNetwork`: Q 网络
- `ReplayBuffer`: 经验回放缓冲区
- `DQNAgent`: DQN 智能体
- 可视化：训练曲线、Q 值变化

**学习目标**：
- 理解 DQN 的核心创新
- 掌握经验回放和目标网络
- 实现一个完整的 DQN 算法

## 学习路径

```
阶段一：表格法
├── q_learning.py              # 值函数学习
└── policy_gradient.py         # 策略优化

阶段二：深度RL
├── dqn.py                     # 值函数近似
└── actor_critic.py            # 策略和值函数结合
```

## 算法对比

| 算法 | 类型 | 表达方式 | 优缺点 |
|------|------|---------|--------|
| Q-Learning | 值函数方法 | Q 表 | 简单高效，但只适用于离散小状态空间 |
| REINFORCE | 策略梯度 | 策略网络 | 可处理连续动作，但方差大 |
| Actor-Critic | 混合方法 | Actor+Critic | 结合两者优点，更稳定 |
| DQN | 值函数近似 | Q 网络 | 可处理大状态空间，需要经验回放 |

## 代码特点

### 1. 从零实现环境
所有环境都从零实现，不依赖 Gym/Gymnasium：
```python
class GridWorld:
    """网格世界环境"""
    def step(self, action):
        # 执行动作，返回 (下一状态, 奖励, 是否结束, 信息)
        ...
    def reset(self):
        # 重置环境
        ...
```

### 2. 数学公式对照
代码注释中包含对应的数学公式：
```python
# Q-Learning 更新公式：
# Q(s, a) = Q(s, a) + α * (r + γ * max_a' Q(s', a') - Q(s, a))
```

### 3. 完整的可视化
每个案例都包含可视化函数：
- 训练曲线（回报、损失）
- Q 值/策略可视化
- 学习过程分析

## 运行方式

```bash
# 运行单个案例
python q_learning.py
python policy_gradient.py
python actor_critic.py
python dqn.py

# 所有案例都会：
# 1. 运行内置测试验证正确性
# 2. 执行训练
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
| q_learning.py | 06_rl_math | 马尔可夫决策过程、贝尔曼方程、值迭代 |
| policy_gradient.py | 06_rl_math | 策略梯度定理、蒙特卡洛方法 |
| actor_critic.py | 06_rl_math | 优势函数、Actor-Critic 架构 |
| dqn.py | 01-04, 06_rl_math | 神经网络、优化、值函数近似 |

## 核心概念速查

### 马尔可夫决策过程 (MDP)
- **状态 (S)**: 环境的表示
- **动作 (A)**: 智能体的行为选择
- **转移概率 (P)**: P(s'|s,a)
- **奖励 (R)**: 即时反馈
- **折扣因子 (γ)**: 未来奖励的权重

### 值函数
- **状态值函数**: V(s) = E[G_t | S_t = s]
- **动作值函数**: Q(s,a) = E[G_t | S_t = s, A_t = a]
- **贝尔曼方程**: V(s) = R + γΣP(s'|s,a)V(s')

### 策略梯度
- **目标**: 最大化期望回报 J(θ) = E[Σγ^t r_t]
- **梯度**: ∇J(θ) = E[∇log π(a|s) * Q(s,a)]
- **REINFORCE**: θ = θ + α * ∇log π(a|s) * G_t

## 相关资源

- [06_rl_math](../06_rl_math/) - 强化学习数学基础
- [OpenAI Spinning Up](https://spinningup.openai.com/) - RL 算法教程
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL 算法库