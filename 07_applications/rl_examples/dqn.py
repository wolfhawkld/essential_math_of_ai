#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep Q-Network (DQN) 算法详解

本模块从零实现 DQN 算法，深入理解值函数近似和经验回放机制。

数学知识：
- Q 值函数神经网络近似
- 经验回放
- 目标网络
- TD 误差

实现内容：
- QNetwork: Q 网络
- ReplayBuffer: 经验回放缓冲区
- DQNAgent: DQN 智能体
- 可视化：训练曲线、Q 值变化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from collections import deque
import random


# ============================================================================
# 环境定义
# ============================================================================

class CartPole:
    """简化版 CartPole 环境"""

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.x_threshold = 2.4
        self.theta_threshold = 12 * 2 * np.pi / 360

        self.n_states = 4
        self.n_actions = 2
        self.state = None

    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        dt = 0.02
        x += x_dot * dt
        x_dot += xacc * dt
        theta += theta_dot * dt
        theta_dot += thetaacc * dt

        self.state = np.array([x, x_dot, theta, theta_dot])

        done = bool(
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold or theta > self.theta_threshold
        )

        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done, {}


# ============================================================================
# Q 网络
# ============================================================================

class QNetwork:
    """
    Q 网络用于近似 Q 值函数

    数学原理：
        Q(s,a; θ) ≈ Q*(s,a)

        输入：状态 s
        输出：每个动作的 Q 值 [Q(s,a_1), Q(s,a_2), ...]

    损失函数：
        L(θ) = E[(r + γ * max_a' Q(s',a'; θ-) - Q(s,a; θ))²]

    其中 θ- 是目标网络参数。
    """

    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 64):
        """
        初始化 Q 网络

        Args:
            n_states: 状态维度
            n_actions: 动作数量
            hidden_size: 隐藏层大小
        """
        self.n_states = n_states
        self.n_actions = n_actions

        # 初始化权重
        scale1 = np.sqrt(2.0 / n_states)
        scale2 = np.sqrt(2.0 / hidden_size)

        self.W1 = np.random.randn(n_states, hidden_size).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_size, dtype=np.float32)
        self.W3 = np.random.randn(hidden_size, n_actions).astype(np.float32) * scale2
        self.b3 = np.zeros(n_actions, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 状态 (n_states,) 或 (batch, n_states)

        Returns:
            Q 值 (n_actions,) 或 (batch, n_actions)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # 第一层 + ReLU
        h1 = np.maximum(0, x @ self.W1 + self.b1)

        # 第二层 + ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)

        # 输出层
        q_values = h2 @ self.W3 + self.b3

        if q_values.shape[0] == 1:
            return q_values.flatten()
        return q_values

    def copy_weights_from(self, other: 'QNetwork') -> None:
        """从另一个网络复制权重"""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


# ============================================================================
# 经验回放缓冲区
# ============================================================================

class ReplayBuffer:
    """
    经验回放缓冲区

    目的：
        1. 打破样本之间的相关性
        2. 提高样本利用效率
        3. 支持离线学习

    存储内容：
        (state, action, reward, next_state, done)

    采样方式：
        均匀随机采样
    """

    def __init__(self, capacity: int = 10000):
        """
        初始化缓冲区

        Args:
            capacity: 最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self,
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool) -> None:
        """
        存储一个转移

        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        随机采样一批转移

        Args:
            batch_size: 批量大小

        Returns:
            states, actions, rewards, next_states, dones
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return len(self.buffer)


# ============================================================================
# DQN 智能体
# ============================================================================

class DQNAgent:
    """
    DQN 智能体

    数学原理：

        Q-Learning 更新：
            Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

        DQN 的关键创新：
            1. 用神经网络近似 Q 函数
            2. 使用经验回放打破相关性
            3. 使用目标网络稳定训练

        目标网络：
            Q_target(s',a') 使用 θ- 参数
            θ- 定期从 θ 复制

        损失函数：
            L(θ) = E[(r + γ * max_a' Q(s',a'; θ-) - Q(s,a; θ))²]
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 hidden_size: int = 64,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 10):
        """
        初始化 DQN 智能体

        Args:
            n_states: 状态维度
            n_actions: 动作数量
            hidden_size: 隐藏层大小
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减
            buffer_size: 经验回放大小
            batch_size: 批量大小
            target_update_freq: 目标网络更新频率
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # 探索参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 网络
        self.q_network = QNetwork(n_states, n_actions, hidden_size)
        self.target_network = QNetwork(n_states, n_actions, hidden_size)
        self.target_network.copy_weights_from(self.q_network)

        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 训练计数
        self.update_count = 0

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（ε-greedy）

        Args:
            state: 当前状态
            training: 是否在训练模式

        Returns:
            选择的动作
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.q_network.forward(state)
            return np.argmax(q_values)

    def store_transition(self,
                         state: np.ndarray,
                         action: int,
                         reward: float,
                         next_state: np.ndarray,
                         done: bool) -> None:
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        """
        更新网络

        Returns:
            训练统计信息，如果缓冲区不够则返回 None
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 计算当前 Q 值
        current_q = self.q_network.forward(states)
        current_q = current_q[np.arange(self.batch_size), actions]

        # 计算目标 Q 值
        with np.errstate(all='ignore'):  # 忽略除零警告
            next_q = self.target_network.forward(next_states)
            max_next_q = np.max(next_q, axis=1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # 计算 TD 误差
        td_error = target_q - current_q

        # 反向传播更新
        self._backward(states, actions, target_q)

        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)

        # 衰减探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            'td_error': np.mean(np.abs(td_error)),
            'mean_q': np.mean(current_q)
        }

    def _backward(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> None:
        """
        反向传播更新网络参数

        Args:
            states: 状态批量
            actions: 动作批量
            targets: 目标 Q 值
        """
        batch_size = len(states)

        # 前向传播缓存
        h1 = np.maximum(0, states @ self.q_network.W1 + self.q_network.b1)
        h2 = np.maximum(0, h1 @ self.q_network.W2 + self.q_network.b2)
        q_values = h2 @ self.q_network.W3 + self.q_network.b3

        # 计算输出层梯度
        grad_output = np.zeros_like(q_values)
        grad_output[np.arange(batch_size), actions] = (q_values[np.arange(batch_size), actions] - targets) / batch_size

        # 反向传播
        grad_W3 = h2.T @ grad_output
        grad_b3 = np.sum(grad_output, axis=0)

        grad_h2 = grad_output @ self.q_network.W3.T
        grad_h2 = grad_h2 * (h2 > 0)

        grad_W2 = h1.T @ grad_h2
        grad_b2 = np.sum(grad_h2, axis=0)

        grad_h1 = grad_h2 @ self.q_network.W2.T
        grad_h1 = grad_h1 * (h1 > 0)

        grad_W1 = states.T @ grad_h1
        grad_b1 = np.sum(grad_h1, axis=0)

        # 更新参数
        self.q_network.W3 -= self.lr * grad_W3
        self.q_network.b3 -= self.lr * grad_b3
        self.q_network.W2 -= self.lr * grad_W2
        self.q_network.b2 -= self.lr * grad_b2
        self.q_network.W1 -= self.lr * grad_W1
        self.q_network.b1 -= self.lr * grad_b1


# ============================================================================
# 训练函数
# ============================================================================

def train_dqn(
    env: CartPole,
    agent: DQNAgent,
    n_episodes: int = 500,
    max_steps: int = 500,
    verbose: bool = True
) -> Dict[str, List]:
    """
    训练 DQN 智能体

    Args:
        env: 环境
        agent: 智能体
        n_episodes: 训练轮数
        max_steps: 每轮最大步数
        verbose: 是否打印进度

    Returns:
        训练统计信息
    """
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'td_errors': [],
        'mean_q_values': [],
        'epsilon': []
    }

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_td_errors = []
        episode_q_values = []

        for step in range(max_steps):
            # 选择动作
            action = agent.get_action(state, training=True)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储转移
            agent.store_transition(state, action, reward, next_state, done)

            # 更新网络
            update_stats = agent.update()
            if update_stats:
                episode_td_errors.append(update_stats['td_error'])
                episode_q_values.append(update_stats['mean_q'])

            episode_reward += reward
            state = next_state

            if done:
                break

        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(step + 1)
        stats['td_errors'].append(np.mean(episode_td_errors) if episode_td_errors else 0)
        stats['mean_q_values'].append(np.mean(episode_q_values) if episode_q_values else 0)
        stats['epsilon'].append(agent.epsilon)

        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(stats['episode_rewards'][-50:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"平均奖励: {avg_reward:.2f}, "
                  f"ε: {agent.epsilon:.3f}")

    return stats


# ============================================================================
# 可视化
# ============================================================================

def visualize_training(
    stats: Dict[str, List],
    save_path: str = "dqn_training.png"
) -> None:
    """可视化训练过程"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 奖励曲线
    axes[0, 0].plot(stats['episode_rewards'], alpha=0.6)
    window = min(50, len(stats['episode_rewards']))
    if window > 1:
        moving_avg = np.convolve(stats['episode_rewards'], np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(stats['episode_rewards'])), moving_avg, color='red', label='移动平均')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].set_title('每轮奖励')
    axes[0, 0].legend()

    # 探索率
    axes[0, 1].plot(stats['epsilon'])
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('ε')
    axes[0, 1].set_title('探索率衰减')

    # TD 误差
    axes[1, 0].plot(stats['td_errors'])
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('TD 误差')
    axes[1, 0].set_title('TD 误差变化')

    # Q 值
    axes[1, 1].plot(stats['mean_q_values'])
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('平均 Q 值')
    axes[1, 1].set_title('Q 值变化')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练可视化已保存到 {save_path}")


def visualize_q_values(
    agent: DQNAgent,
    env: CartPole,
    save_path: str = "dqn_q_values.png"
) -> None:
    """可视化 Q 值"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Q 值 vs 角度
    thetas = np.linspace(-0.3, 0.3, 50)
    q_values_left = []
    q_values_right = []

    for theta in thetas:
        state = np.array([0, 0, theta, 0])
        q_values = agent.q_network.forward(state)
        q_values_left.append(q_values[0])
        q_values_right.append(q_values[1])

    axes[0].plot(thetas, q_values_left, label='向左', color='blue')
    axes[0].plot(thetas, q_values_right, label='向右', color='red')
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('杆子角度')
    axes[0].set_ylabel('Q 值')
    axes[0].set_title('Q 值 vs 角度')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q 值 vs 位置
    xs = np.linspace(-2, 2, 50)
    q_values_left_x = []
    q_values_right_x = []

    for x in xs:
        state = np.array([x, 0, 0, 0])
        q_values = agent.q_network.forward(state)
        q_values_left_x.append(q_values[0])
        q_values_right_x.append(q_values[1])

    axes[1].plot(xs, q_values_left_x, label='向左', color='blue')
    axes[1].plot(xs, q_values_right_x, label='向右', color='red')
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('小车位置')
    axes[1].set_ylabel('Q 值')
    axes[1].set_title('Q 值 vs 位置')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Q 值可视化已保存到 {save_path}")


def compare_with_without_replay(
    env: CartPole,
    n_episodes: int = 300,
    save_path: str = "dqn_replay_comparison.png"
) -> None:
    """对比有无经验回放"""
    print("\n对比有无经验回放:")

    # 有经验回放
    agent_with_replay = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        buffer_size=10000,
        batch_size=32
    )
    stats_with = train_dqn(env, agent_with_replay, n_episodes, verbose=False)
    print(f"  有经验回放: 最终平均奖励 = {np.mean(stats_with['episode_rewards'][-50:]):.2f}")

    # 无经验回放（小缓冲区，立即学习）
    agent_without_replay = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        buffer_size=1,
        batch_size=1,
        target_update_freq=1
    )
    stats_without = train_dqn(env, agent_without_replay, n_episodes, verbose=False)
    print(f"  无经验回放: 最终平均奖励 = {np.mean(stats_without['episode_rewards'][-50:]):.2f}")

    # 可视化
    plt.figure(figsize=(10, 6))

    window = 20
    moving_avg_with = np.convolve(stats_with['episode_rewards'], np.ones(window)/window, mode='valid')
    moving_avg_without = np.convolve(stats_without['episode_rewards'], np.ones(window)/window, mode='valid')

    plt.plot(moving_avg_with, label='有经验回放', alpha=0.8)
    plt.plot(moving_avg_without, label='无经验回放', alpha=0.8)

    plt.xlabel('Episode')
    plt.ylabel('平均奖励')
    plt.title('DQN: 经验回放效果对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"经验回放对比已保存到 {save_path}")


# ============================================================================
# 数学原理说明
# ============================================================================

def explain_dqn_math():
    """解释 DQN 数学原理"""
    print("\n" + "=" * 60)
    print("DQN 数学原理")
    print("=" * 60)

    print("""
1. 值函数近似

   问题：Q 表无法处理大/连续状态空间

   解决：用神经网络近似 Q 函数
       Q(s,a; θ) ≈ Q*(s,a)

   输入：状态 s
   输出：所有动作的 Q 值

2. DQN 的三个关键创新

   a) 经验回放 (Experience Replay)
       - 存储转移 (s, a, r, s', done)
       - 随机采样训练
       - 打破样本相关性
       - 提高数据利用效率

   b) 目标网络 (Target Network)
       - 使用两个网络：Q 网络和目标网络
       - 目标网络参数 θ- 定期从 Q 网络复制
       - 目标：r + γ max_a' Q(s',a'; θ-)
       - 避免目标值随训练变化过快

   c) 损失函数
       L(θ) = E[(r + γ max_a' Q(s',a'; θ-) - Q(s,a; θ))²]

3. 训练过程

   初始化：
       - Q 网络参数 θ
       - 目标网络参数 θ- = θ
       - 空的经验回放缓冲区

   对于每个 episode:
       对于每一步：
           1. 用 ε-greedy 选择动作
           2. 执行动作，观察 r, s'
           3. 存储到经验回放
           4. 从缓冲区采样批量
           5. 计算目标 Q 值
           6. 更新 Q 网络
           7. 定期更新目标网络

4. 探索策略

   ε-greedy 衰减：
       ε_t = max(ε_min, ε_0 * decay^t)

   目的：
       - 早期：更多探索，学习环境
       - 后期：更多利用，优化策略

5. 目标网络更新频率

   太频繁：
       - 训练不稳定
       - 目标值变化太快

   太少：
       - 学习缓慢
       - 目标值过时

   典型值：每 10-1000 步更新一次

6. DQN 的变体

   Double DQN：
       - 解决过估计问题
       - 目标：r + γ Q(s', argmax_a' Q(s',a'); θ-)

   Dueling DQN：
       - 分离状态价值和优势函数
       - Q(s,a) = V(s) + A(s,a)

   Prioritized Experience Replay：
       - 重要转移优先采样
       - 基于 TD 误差计算优先级
""")


# ============================================================================
# 测试
# ============================================================================

def test_q_network():
    """测试 Q 网络"""
    print("测试 Q 网络...")

    network = QNetwork(n_states=4, n_actions=2)

    # 测试前向传播
    state = np.random.randn(4).astype(np.float32)
    q_values = network.forward(state)

    assert q_values.shape == (2,), f"Q 值形状错误: {q_values.shape}"

    # 测试批量前向传播
    states = np.random.randn(32, 4).astype(np.float32)
    q_values = network.forward(states)

    assert q_values.shape == (32, 2), f"批量 Q 值形状错误: {q_values.shape}"

    print("  ✓ Q 网络正确")


def test_replay_buffer():
    """测试经验回放"""
    print("测试经验回放...")

    buffer = ReplayBuffer(capacity=100)

    # 测试存储
    for i in range(50):
        buffer.push(
            np.random.randn(4),
            np.random.randint(2),
            np.random.randn(),
            np.random.randn(4),
            False
        )

    assert len(buffer) == 50, f"缓冲区大小错误: {len(buffer)}"

    # 测试采样
    states, actions, rewards, next_states, dones = buffer.sample(10)

    assert states.shape == (10, 4), f"状态形状错误: {states.shape}"
    assert actions.shape == (10,), f"动作形状错误: {actions.shape}"

    print("  ✓ 经验回放正确")


def test_dqn_agent():
    """测试 DQN 智能体"""
    print("测试 DQN 智能体...")

    env = CartPole()
    agent = DQNAgent(n_states=4, n_actions=2, buffer_size=100, batch_size=4)

    state = env.reset()

    # 测试动作选择
    action = agent.get_action(state, training=False)
    assert action in [0, 1], f"动作错误: {action}"

    # 测试存储
    next_state, reward, done, _ = env.step(action)
    agent.store_transition(state, action, reward, next_state, done)
    assert len(agent.replay_buffer) == 1, "存储失败"

    print("  ✓ DQN 智能体正确")


def test_learning():
    """测试学习"""
    print("测试学习过程...")

    env = CartPole()
    agent = DQNAgent(
        n_states=4,
        n_actions=2,
        learning_rate=0.001,
        buffer_size=1000,
        batch_size=32
    )

    stats = train_dqn(env, agent, n_episodes=200, verbose=False)

    final_avg = np.mean(stats['episode_rewards'][-20:])
    print(f"  最终平均奖励: {final_avg:.2f}")

    print("  ✓ 学习过程正常")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行 DQN 测试")
    print("=" * 60)

    test_q_network()
    test_replay_buffer()
    test_dqn_agent()
    test_learning()

    print("\n所有测试通过!")


def main():
    """主函数"""
    print("=" * 60)
    print("Deep Q-Network (DQN) 算法详解")
    print("=" * 60)

    # 解释数学原理
    explain_dqn_math()

    # 创建环境和智能体
    env = CartPole()
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.001,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=10
    )

    # 训练
    print("\n开始训练...")
    stats = train_dqn(env, agent, n_episodes=300, verbose=True)

    # 可视化
    visualize_training(stats)
    visualize_q_values(agent, env)

    # 对比实验
    compare_with_without_replay(env, n_episodes=200)

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    run_tests()

    # 运行主程序
    main()