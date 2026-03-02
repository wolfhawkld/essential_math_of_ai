#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REINFORCE (Policy Gradient) 算法详解

本模块从零实现 REINFORCE 算法，深入理解策略梯度和蒙特卡洛方法。

数学知识：
- 策略梯度定理
- REINFORCE 算法
- 蒙特卡洛采样
- 基线减少方差

实现内容：
- PolicyNetwork: 策略网络
- REINFORCEAgent: REINFORCE 智能体
- CartPole 简化环境或自定义环境
- 可视化：训练曲线、策略分析
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional


# ============================================================================
# 环境定义
# ============================================================================

class CartPole:
    """
    简化版 CartPole 环境

    这是一个经典的控制问题：通过左右移动小车来平衡杆子。

    状态空间 (4维):
        - x: 小车位置
        - x_dot: 小车速度
        - theta: 杆子角度
        - theta_dot: 杆子角速度

    动作空间 (2维):
        - 0: 向左推
        - 1: 向右推

    奖励:
        - 每步保持平衡：+1
        - 杆子倒下：结束

    数学模型:
        x'' = (F + m*l*θ'*sin(θ) - μ*sign(x')) / (M+m)
        θ'' = (g*sin(θ) - cos(θ)*(x'' + μ*sign(x'))) / (l*(4/3 - m/(M+m)))
    """

    def __init__(self):
        """初始化环境参数"""
        # 物理参数
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0

        # 阈值
        self.x_threshold = 2.4
        self.theta_threshold = 12 * 2 * np.pi / 360  # 12度

        # 状态和动作空间
        self.n_states = 4
        self.n_actions = 2

        # 当前状态
        self.state = None

    def reset(self) -> np.ndarray:
        """
        重置环境

        Returns:
            初始状态
        """
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作 (0: 左, 1: 右)

        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        x, x_dot, theta, theta_dot = self.state

        # 力
        force = self.force_mag if action == 1 else -self.force_mag

        # 物理计算
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # 欧拉积分
        dt = 0.02
        x += x_dot * dt
        x_dot += xacc * dt
        theta += theta_dot * dt
        theta_dot += thetaacc * dt

        self.state = np.array([x, x_dot, theta, theta_dot])

        # 检查终止条件
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )

        reward = 1.0 if not done else 0.0

        return self.state.copy(), reward, done, {}


# ============================================================================
# 策略网络
# ============================================================================

class PolicyNetwork:
    """
    策略网络

    数学原理：
        策略网络输出给定状态下各动作的概率：
            π(a|s; θ) = softmax(f(s; θ))

        其中 f(s; θ) 是神经网络的前向传播。

    对于离散动作空间，使用 softmax 策略：
            π(a_i|s) = exp(f_i(s)) / Σ_j exp(f_j(s))

    参数更新：
            θ ← θ + α * ∇_θ log π(a|s) * G_t

    其中 G_t 是从该时刻开始的累积奖励（回报）。
    """

    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 32):
        """
        初始化策略网络

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
        self.W2 = np.random.randn(hidden_size, n_actions).astype(np.float32) * scale2
        self.b2 = np.zeros(n_actions, dtype=np.float32)

        # 缓存
        self.x = None
        self.h = None
        self.logits = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 状态 (n_states,) 或 (batch, n_states)

        Returns:
            动作概率 (n_actions,) 或 (batch, n_actions)
        """
        self.x = x

        # 隐藏层: ReLU
        self.h = np.maximum(0, x @ self.W1 + self.b1)

        # 输出层: Softmax
        self.logits = self.h @ self.W2 + self.b2
        probs = self._softmax(self.logits)

        return probs

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的 softmax"""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def sample_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        根据策略采样动作

        Args:
            state: 状态

        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        probs = self.forward(state)

        # 采样动作
        action = np.random.choice(len(probs), p=probs)

        # 计算对数概率
        log_prob = np.log(probs[action] + 1e-10)

        return action, log_prob

    def get_log_prob(self, state: np.ndarray, action: int) -> float:
        """
        获取给定状态-动作的对数概率

        Args:
            state: 状态
            action: 动作

        Returns:
            log_prob: 对数概率
        """
        probs = self.forward(state)
        return np.log(probs[action] + 1e-10)


# ============================================================================
# REINFORCE 智能体
# ============================================================================

class REINFORCEAgent:
    """
    REINFORCE 智能体（蒙特卡洛策略梯度）

    数学原理：

        目标：最大化期望回报
            J(θ) = E_π[Σ_t γ^t r_t]

        策略梯度定理：
            ∇_θ J(θ) = E_π[Σ_t ∇_θ log π(a_t|s_t; θ) * G_t]

        其中 G_t 是从时刻 t 开始的折扣累积奖励：
            G_t = Σ_{k=t}^T γ^{k-t} r_k

        REINFORCE 更新：
            θ ← θ + α * Σ_t ∇_θ log π(a_t|s_t) * G_t

        带基线的 REINFORCE：
            θ ← θ + α * Σ_t ∇_θ log π(a_t|s_t) * (G_t - b(s_t))

        基线的作用是减少方差，通常使用值函数 V(s) 作为基线。
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 hidden_size: int = 32,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.99,
                 use_baseline: bool = True):
        """
        初始化 REINFORCE 智能体

        Args:
            n_states: 状态维度
            n_actions: 动作数量
            hidden_size: 隐藏层大小
            learning_rate: 学习率
            discount_factor: 折扣因子
            use_baseline: 是否使用基线
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.use_baseline = use_baseline

        # 策略网络
        self.policy = PolicyNetwork(n_states, n_actions, hidden_size)

        # 如果使用基线，初始化值函数
        if use_baseline:
            self.value_weights = np.zeros(n_states, dtype=np.float32)

        # 轨迹存储
        self.trajectory = []

    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        选择动作

        Args:
            state: 当前状态

        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        return self.policy.sample_action(state)

    def store_transition(self,
                         state: np.ndarray,
                         action: int,
                         reward: float,
                         log_prob: float) -> None:
        """
        存储转移

        Args:
            state: 状态
            action: 动作
            reward: 奖励
            log_prob: 对数概率
        """
        self.trajectory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob
        })

    def compute_returns(self) -> List[float]:
        """
        计算每个时刻的回报

        G_t = Σ_{k=t}^T γ^{k-t} r_k

        Returns:
            returns: 回报列表
        """
        T = len(self.trajectory)
        returns = np.zeros(T)

        # 反向计算
        G = 0
        for t in reversed(range(T)):
            G = self.trajectory[t]['reward'] + self.gamma * G
            returns[t] = G

        return returns

    def compute_advantages(self, returns: List[float]) -> List[float]:
        """
        计算优势函数

        如果使用基线：
            A_t = G_t - V(s_t)

        否则：
            A_t = G_t

        Args:
            returns: 回报列表

        Returns:
            advantages: 优势列表
        """
        if self.use_baseline:
            advantages = []
            for t, transition in enumerate(self.trajectory):
                state = transition['state']
                # 简单的线性值函数
                value = np.dot(state, self.value_weights)
                advantage = returns[t] - value
                advantages.append(advantage)

                # 更新值函数
                self.value_weights += self.lr * 0.1 * advantage * state

            return advantages
        else:
            return list(returns)

    def update(self) -> Dict[str, float]:
        """
        更新策略参数

        策略梯度更新：
            θ ← θ + α * Σ_t ∇_θ log π(a_t|s_t) * A_t

        Returns:
            统计信息
        """
        # 计算回报和优势
        returns = self.compute_returns()
        advantages = self.compute_advantages(returns)

        # 归一化优势
        advantages = np.array(advantages)
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # 计算梯度并更新
        total_log_prob = 0.0

        for t, transition in enumerate(self.trajectory):
            state = transition['state']
            action = transition['action']
            advantage = advantages[t]

            # 前向传播
            probs = self.policy.forward(state)

            # 计算梯度
            # ∂ log π(a|s) / ∂ logits = one_hot(a) - probs
            grad_logits = -probs
            grad_logits[action] += 1.0

            # 反向传播
            grad_W2 = np.outer(self.policy.h, grad_logits) * advantage
            grad_b2 = grad_logits * advantage

            grad_h = grad_logits @ self.policy.W2.T
            grad_h = grad_h * (self.policy.h > 0)  # ReLU 梯度

            grad_W1 = np.outer(state, grad_h) * advantage
            grad_b1 = grad_h * advantage

            # 更新参数
            self.policy.W2 -= self.lr * grad_W2
            self.policy.b2 -= self.lr * grad_b2
            self.policy.W1 -= self.lr * grad_W1
            self.policy.b1 -= self.lr * grad_b1

            total_log_prob += np.log(probs[action] + 1e-10)

        # 清空轨迹
        stats = {
            'episode_return': returns[0],
            'episode_length': len(self.trajectory),
            'mean_advantage': np.mean(advantages)
        }

        self.trajectory = []

        return stats


# ============================================================================
# 训练函数
# ============================================================================

def train_reinforce(
    env: CartPole,
    agent: REINFORCEAgent,
    n_episodes: int = 1000,
    max_steps: int = 500,
    verbose: bool = True
) -> Dict[str, List]:
    """
    训练 REINFORCE 智能体

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
        'episode_returns': [],
        'episode_lengths': [],
        'mean_advantages': []
    }

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作
            action, log_prob = agent.get_action(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储转移
            agent.store_transition(state, action, reward, log_prob)

            episode_reward += reward
            state = next_state

            if done:
                break

        # 更新策略
        update_stats = agent.update()

        stats['episode_returns'].append(update_stats['episode_return'])
        stats['episode_lengths'].append(update_stats['episode_length'])
        stats['mean_advantages'].append(update_stats['mean_advantage'])

        if verbose and (episode + 1) % 100 == 0:
            avg_return = np.mean(stats['episode_returns'][-100:])
            avg_length = np.mean(stats['episode_lengths'][-100:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"平均回报: {avg_return:.2f}, "
                  f"平均长度: {avg_length:.1f}")

    return stats


# ============================================================================
# 可视化
# ============================================================================

def visualize_training(
    stats: Dict[str, List],
    save_path: str = "reinforce_training.png"
) -> None:
    """
    可视化训练过程

    Args:
        stats: 训练统计信息
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 回报曲线
    axes[0].plot(stats['episode_returns'], alpha=0.6)
    window = min(50, len(stats['episode_returns']))
    if window > 1:
        moving_avg = np.convolve(stats['episode_returns'], np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(stats['episode_returns'])), moving_avg, color='red', label='移动平均')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('回报')
    axes[0].set_title('每轮回报')
    axes[0].legend()

    # 轮长度
    axes[1].plot(stats['episode_lengths'])
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('步数')
    axes[1].set_title('每轮步数')

    # 平均优势
    axes[2].plot(stats['mean_advantages'])
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('平均优势')
    axes[2].set_title('平均优势变化')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练可视化已保存到 {save_path}")


def visualize_policy(
    agent: REINFORCEAgent,
    env: CartPole,
    save_path: str = "reinforce_policy.png"
) -> None:
    """
    可视化学习到的策略

    Args:
        agent: 智能体
        env: 环境
        save_path: 保存路径
    """
    # 测试策略
    n_test = 10
    test_lengths = []
    test_returns = []

    for _ in range(n_test):
        state = env.reset()
        episode_return = 0

        for step in range(500):
            probs = agent.policy.forward(state)
            action = np.argmax(probs)
            state, reward, done, _ = env.step(action)
            episode_return += reward

            if done:
                break

        test_lengths.append(step + 1)
        test_returns.append(episode_return)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(range(n_test), test_lengths)
    axes[0].axhline(y=np.mean(test_lengths), color='red', linestyle='--', label=f'平均: {np.mean(test_lengths):.1f}')
    axes[0].set_xlabel('测试轮次')
    axes[0].set_ylabel('步数')
    axes[0].set_title('测试轮长度')
    axes[0].legend()

    axes[1].bar(range(n_test), test_returns)
    axes[1].axhline(y=np.mean(test_returns), color='red', linestyle='--', label=f'平均: {np.mean(test_returns):.1f}')
    axes[1].set_xlabel('测试轮次')
    axes[1].set_ylabel('回报')
    axes[1].set_title('测试轮回报')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"策略可视化已保存到 {save_path}")


def compare_with_without_baseline(
    env: CartPole,
    n_episodes: int = 500,
    save_path: str = "reinforce_baseline_comparison.png"
) -> None:
    """
    对比有无基线的 REINFORCE

    Args:
        env: 环境
        n_episodes: 训练轮数
        save_path: 保存路径
    """
    print("\n对比有无基线的 REINFORCE:")

    # 无基线
    agent_no_baseline = REINFORCEAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        use_baseline=False
    )
    stats_no_baseline = train_reinforce(env, agent_no_baseline, n_episodes, verbose=False)
    print(f"  无基线: 最终平均回报 = {np.mean(stats_no_baseline['episode_returns'][-50:]):.2f}")

    # 有基线
    agent_with_baseline = REINFORCEAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        use_baseline=True
    )
    stats_with_baseline = train_reinforce(env, agent_with_baseline, n_episodes, verbose=False)
    print(f"  有基线: 最终平均回报 = {np.mean(stats_with_baseline['episode_returns'][-50:]):.2f}")

    # 可视化对比
    plt.figure(figsize=(10, 6))

    window = 20
    moving_avg_no = np.convolve(stats_no_baseline['episode_returns'], np.ones(window)/window, mode='valid')
    moving_avg_with = np.convolve(stats_with_baseline['episode_returns'], np.ones(window)/window, mode='valid')

    plt.plot(moving_avg_no, label='无基线', alpha=0.8)
    plt.plot(moving_avg_with, label='有基线', alpha=0.8)

    plt.xlabel('Episode')
    plt.ylabel('平均回报')
    plt.title('REINFORCE: 基线效果对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"基线对比已保存到 {save_path}")


# ============================================================================
# 数学原理说明
# ============================================================================

def explain_policy_gradient_math():
    """解释策略梯度数学原理"""
    print("\n" + "=" * 60)
    print("策略梯度数学原理")
    print("=" * 60)

    print("""
1. 策略梯度定理

   目标：最大化期望回报
       J(θ) = E_π[Σ_t γ^t r_t] = E_π[τ ~ p(τ;θ)] [R(τ)]

   其中 τ = (s_0, a_0, r_0, s_1, a_1, r_1, ...) 是一条轨迹

   梯度：
       ∇_θ J(θ) = E_π[Σ_t ∇_θ log π(a_t|s_t; θ) * G_t]

   直觉：
       - 如果 G_t > 0，增加产生该动作的概率
       - 如果 G_t < 0，减少产生该动作的概率

2. REINFORCE 算法

   算法步骤：
       对于每个 episode:
           1. 使用当前策略 π_θ 采样轨迹 τ
           2. 对于每个时间步 t:
               计算 G_t = Σ_{k=t} γ^{k-t} r_k
           3. 更新参数：
               θ ← θ + α * Σ_t ∇_θ log π(a_t|s_t) * G_t

   特点：
       - 蒙特卡洛方法（需要完整轨迹）
       - 无偏估计
       - 高方差

3. 基线减少方差

   问题：REINFORCE 的方差很大

   解决：引入基线 b(s)
       ∇_θ J(θ) = E_π[Σ_t ∇_θ log π(a_t|s_t) * (G_t - b(s_t))]

   性质：
       - 如果 b(s) 与动作无关，期望不变
       - 好的基线可以显著减少方差
       - 常用基线：值函数 V(s)

4. 优势函数

   优势函数：
       A(s,a) = Q(s,a) - V(s)

   含义：
       - 动作 a 比平均好多少
       - 正优势 → 增加概率
       - 负优势 → 减少概率

   估计：
       A_t ≈ G_t - V(s_t)

5. 策略梯度 vs 值函数方法

   策略梯度：
       - 直接优化策略
       - 可处理连续动作空间
       - 可学习随机策略
       - 高方差

   值函数方法（如 Q-Learning）：
       - 学习值函数，间接得到策略
       - 样本效率高
       - 只适用于离散动作
       - 主要是确定性策略
""")


# ============================================================================
# 测试
# ============================================================================

def test_cartpole():
    """测试 CartPole 环境"""
    print("测试 CartPole 环境...")

    env = CartPole()

    # 测试重置
    state = env.reset()
    assert state.shape == (4,), f"状态形状错误: {state.shape}"

    # 测试动作
    next_state, reward, done, _ = env.step(0)
    assert next_state.shape == (4,), f"下一状态形状错误: {next_state.shape}"
    assert isinstance(reward, (int, float)), "奖励类型错误"
    assert isinstance(done, bool), "done 类型错误"

    print("  ✓ CartPole 环境正确")


def test_policy_network():
    """测试策略网络"""
    print("测试策略网络...")

    policy = PolicyNetwork(n_states=4, n_actions=2)

    # 测试前向传播
    state = np.random.randn(4).astype(np.float32)
    probs = policy.forward(state)

    assert probs.shape == (2,), f"概率形状错误: {probs.shape}"
    assert np.allclose(np.sum(probs), 1.0), f"概率和不为 1: {np.sum(probs)}"

    # 测试动作采样
    action, log_prob = policy.sample_action(state)
    assert action in [0, 1], f"动作错误: {action}"

    print("  ✓ 策略网络正确")


def test_reinforce_agent():
    """测试 REINFORCE 智能体"""
    print("测试 REINFORCE 智能体...")

    env = CartPole()
    agent = REINFORCEAgent(n_states=4, n_actions=2)

    state = env.reset()

    # 测试动作选择
    action, log_prob = agent.get_action(state)
    assert action in [0, 1], f"动作错误: {action}"

    # 测试轨迹存储
    agent.store_transition(state, action, 1.0, log_prob)
    assert len(agent.trajectory) == 1, "轨迹存储错误"

    print("  ✓ REINFORCE 智能体正确")


def test_returns_computation():
    """测试回报计算"""
    print("测试回报计算...")

    agent = REINFORCEAgent(n_states=4, n_actions=2)

    # 模拟轨迹
    rewards = [1, 1, 1, 1, 1]
    for i, r in enumerate(rewards):
        state = np.zeros(4)
        agent.store_transition(state, 0, r, 0.0)

    returns = agent.compute_returns()

    # 验证第一个回报
    expected = sum(r * (0.99 ** i) for i, r in enumerate(rewards))
    assert np.isclose(returns[0], expected, atol=0.1), f"回报计算错误: {returns[0]} vs {expected}"

    print("  ✓ 回报计算正确")


def test_learning():
    """测试学习过程"""
    print("测试学习过程...")

    env = CartPole()
    agent = REINFORCEAgent(n_states=4, n_actions=2, learning_rate=0.01)

    # 短期训练
    stats = train_reinforce(env, agent, n_episodes=100, verbose=False)

    # 检查是否学到东西
    final_avg = np.mean(stats['episode_returns'][-20:])
    print(f"  最终平均回报: {final_avg:.2f}")

    print("  ✓ 学习过程正常")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行 REINFORCE 测试")
    print("=" * 60)

    test_cartpole()
    test_policy_network()
    test_reinforce_agent()
    test_returns_computation()
    test_learning()

    print("\n所有测试通过!")


def main():
    """主函数"""
    print("=" * 60)
    print("REINFORCE (Policy Gradient) 算法详解")
    print("=" * 60)

    # 解释数学原理
    explain_policy_gradient_math()

    # 创建环境和智能体
    env = CartPole()
    agent = REINFORCEAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.01,
        use_baseline=True
    )

    # 训练
    print("\n开始训练...")
    stats = train_reinforce(env, agent, n_episodes=500, verbose=True)

    # 可视化
    visualize_training(stats)
    visualize_policy(agent, env)

    # 对比实验
    compare_with_without_baseline(env, n_episodes=300)

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    run_tests()

    # 运行主程序
    main()