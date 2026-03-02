#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Actor-Critic 算法详解

本模块从零实现 A2C (Advantage Actor-Critic) 算法，理解 Actor 和 Critic 的协作机制。

数学知识：
- Actor-Critic 架构
- 优势函数
- 值函数估计
- 策略梯度

实现内容：
- ActorNetwork: 策略网络
- CriticNetwork: 值网络
- A2CAgent: Actor-Critic 智能体
- 可视化：Actor 和 Critic 学习过程
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


# ============================================================================
# 复用 CartPole 环境
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
# Actor 网络（策略网络）
# ============================================================================

class ActorNetwork:
    """
    Actor 网络输出动作概率

    数学原理：
        π(a|s; θ) = softmax(f(s; θ))

    更新目标：
        最大化优势函数加权的策略概率
        L_actor = -E[log π(a|s) * A(s,a)]
    """

    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 32):
        self.n_states = n_states
        self.n_actions = n_actions

        scale1 = np.sqrt(2.0 / n_states)
        scale2 = np.sqrt(2.0 / hidden_size)

        self.W1 = np.random.randn(n_states, hidden_size).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, n_actions).astype(np.float32) * scale2
        self.b2 = np.zeros(n_actions, dtype=np.float32)

        # 缓存
        self.x = None
        self.h = None

        # 梯度
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.x = x
        self.h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        logits = self.h @ self.W2 + self.b2

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs

    def backward(self, action: int, advantage: float, lr: float) -> None:
        """
        反向传播

        策略梯度：
            ∂L/∂θ = -log π(a|s) * A(s,a)

        对于 softmax：
            ∂ log π(a|s) / ∂ logits = one_hot(a) - probs
        """
        probs = self.forward(self.x)

        # 梯度
        grad_logits = -probs
        grad_logits[action] += 1.0
        grad_logits *= advantage  # 加权

        # 反向传播
        self.grad_W2 = np.outer(self.h, grad_logits)
        self.grad_b2 = grad_logits

        grad_h = grad_logits @ self.W2.T * (self.h > 0)

        self.grad_W1 = np.outer(self.x, grad_h)
        self.grad_b1 = grad_h

        # 更新
        self.W2 -= lr * self.grad_W2
        self.b2 -= lr * self.grad_b2
        self.W1 -= lr * self.grad_W1
        self.b1 -= lr * self.grad_b1


# ============================================================================
# Critic 网络（值网络）
# ============================================================================

class CriticNetwork:
    """
    Critic 网络估计状态值函数

    数学原理：
        V(s; w) ≈ E[G_t | s_t = s]

    更新目标：
        最小化 TD 误差
        L_critic = E[(r + γV(s') - V(s))²]
    """

    def __init__(self, n_states: int, hidden_size: int = 32):
        self.n_states = n_states

        scale1 = np.sqrt(2.0 / n_states)
        scale2 = np.sqrt(2.0 / hidden_size)

        self.W1 = np.random.randn(n_states, hidden_size).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, 1).astype(np.float32) * scale2
        self.b2 = np.zeros(1, dtype=np.float32)

        # 缓存
        self.x = None
        self.h = None

    def forward(self, x: np.ndarray) -> float:
        """前向传播，返回 V(s)"""
        self.x = x
        self.h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        value = (self.h @ self.W2 + self.b2).item()
        return value

    def backward(self, td_error: float, lr: float) -> None:
        """
        反向传播

        值函数梯度：
            ∂L/∂w = (V(s) - (r + γV(s'))) * ∂V(s)/∂w
                  = td_error * ∂V(s)/∂w
        """
        # 输出层梯度
        grad_W2 = self.h.reshape(-1, 1) * td_error
        grad_b2 = np.array([td_error])

        # 隐藏层梯度
        grad_h = (self.W2.T * td_error).flatten() * (self.h > 0)

        # 输入层梯度
        grad_W1 = np.outer(self.x, grad_h) * td_error
        grad_b1 = grad_h

        # 更新
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1


# ============================================================================
# A2C 智能体
# ============================================================================

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) 智能体

    数学原理：

        Actor 更新：
            θ ← θ + α * ∇_θ log π(a|s) * A(s,a)

        Critic 更新：
            w ← w - β * (V(s) - (r + γV(s'))) * ∇_w V(s)

        优势函数：
            A(s,a) = r + γV(s') - V(s)  (TD 误差)

            或者 n-step 优势：
            A(s,a) = Σ_{i=0}^{n-1} γ^i r_i + γ^n V(s_n) - V(s)

    特点：
        - Actor 学习策略，Critic 学习值函数
        - Critic 为 Actor 提供基线，减少方差
        - 可以在线学习，不需要完整轨迹
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 hidden_size: int = 32,
                 actor_lr: float = 0.01,
                 critic_lr: float = 0.01,
                 discount_factor: float = 0.99):
        """
        初始化 A2C 智能体

        Args:
            n_states: 状态维度
            n_actions: 动作数量
            hidden_size: 隐藏层大小
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            discount_factor: 折扣因子
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = discount_factor

        # 创建网络
        self.actor = ActorNetwork(n_states, n_actions, hidden_size)
        self.critic = CriticNetwork(n_states, hidden_size)

    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        选择动作

        Args:
            state: 当前状态

        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        probs = self.actor.forward(state)
        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        return action, log_prob

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> Dict[str, float]:
        """
        更新网络参数

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束

        Returns:
            统计信息
        """
        # 计算 V(s) 和 V(s')
        value = self.critic.forward(state)
        next_value = 0.0 if done else self.critic.forward(next_state)

        # 计算 TD 目标和优势
        td_target = reward + self.gamma * next_value
        td_error = td_target - value  # 优势函数

        # 更新 Critic（最小化 TD 误差）
        self.critic.backward(td_error, self.critic_lr)

        # 更新 Actor（最大化优势加权的策略）
        self.actor.backward(action, td_error, self.actor_lr)

        return {
            'value': value,
            'td_error': td_error,
            'advantage': td_error
        }

    def update_batch(self,
                     states: List[np.ndarray],
                     actions: List[int],
                     rewards: List[float],
                     next_states: List[np.ndarray],
                     dones: List[bool]) -> Dict[str, float]:
        """
        批量更新（用于 n-step A2C）

        Args:
            states: 状态列表
            actions: 动作列表
            rewards: 奖励列表
            next_states: 下一状态列表
            dones: 结束标志列表

        Returns:
            统计信息
        """
        total_td_error = 0.0

        for i in range(len(states)):
            stats = self.update(
                states[i], actions[i], rewards[i],
                next_states[i], dones[i]
            )
            total_td_error += abs(stats['td_error'])

        return {
            'mean_td_error': total_td_error / len(states)
        }


# ============================================================================
# 训练函数
# ============================================================================

def train_a2c(
    env: CartPole,
    agent: A2CAgent,
    n_episodes: int = 1000,
    max_steps: int = 500,
    n_step: int = 5,
    verbose: bool = True
) -> Dict[str, List]:
    """
    训练 A2C 智能体

    Args:
        env: 环境
        agent: 智能体
        n_episodes: 训练轮数
        max_steps: 每轮最大步数
        n_step: n-step 更新步数
        verbose: 是否打印进度

    Returns:
        训练统计信息
    """
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'td_errors': [],
        'values': []
    }

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_td_errors = []
        episode_values = []

        # 存储轨迹
        trajectory = []

        for step in range(max_steps):
            # 选择动作
            action, _ = agent.get_action(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储
            trajectory.append((state, action, reward, next_state, done))

            episode_reward += reward
            state = next_state

            # n-step 更新
            if len(trajectory) >= n_step or done:
                # 计算 n-step 回报
                states, actions, rewards, next_states, dones = zip(*trajectory)

                # 更新
                update_stats = agent.update_batch(
                    list(states), list(actions), list(rewards),
                    list(next_states), list(dones)
                )

                episode_td_errors.append(update_stats['mean_td_error'])
                trajectory = []

            if done:
                break

        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(step + 1)
        stats['td_errors'].append(np.mean(episode_td_errors) if episode_td_errors else 0)

        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(stats['episode_rewards'][-100:])
            print(f"Episode {episode + 1}/{n_episodes}, 平均奖励: {avg_reward:.2f}")

    return stats


# ============================================================================
# 可视化
# ============================================================================

def visualize_training(
    stats: Dict[str, List],
    save_path: str = "a2c_training.png"
) -> None:
    """可视化训练过程"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 奖励曲线
    axes[0].plot(stats['episode_rewards'], alpha=0.6)
    window = min(50, len(stats['episode_rewards']))
    if window > 1:
        moving_avg = np.convolve(stats['episode_rewards'], np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(stats['episode_rewards'])), moving_avg, color='red', label='移动平均')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('奖励')
    axes[0].set_title('每轮奖励')
    axes[0].legend()

    # 轮长度
    axes[1].plot(stats['episode_lengths'])
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('步数')
    axes[1].set_title('每轮步数')

    # TD 误差
    axes[2].plot(stats['td_errors'])
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('TD 误差')
    axes[2].set_title('TD 误差变化')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练可视化已保存到 {save_path}")


def visualize_actor_critic(
    agent: A2CAgent,
    env: CartPole,
    save_path: str = "a2c_networks.png"
) -> None:
    """可视化 Actor 和 Critic 学习结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Actor 策略分析
    # 测试不同角度下的动作概率
    thetas = np.linspace(-0.3, 0.3, 50)
    probs_left = []
    probs_right = []

    for theta in thetas:
        state = np.array([0, 0, theta, 0])
        probs = agent.actor.forward(state)
        probs_left.append(probs[0])
        probs_right.append(probs[1])

    axes[0].plot(thetas, probs_left, label='向左', color='blue')
    axes[0].plot(thetas, probs_right, label='向右', color='red')
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('杆子角度')
    axes[0].set_ylabel('动作概率')
    axes[0].set_title('Actor: 动作概率 vs 角度')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Critic 值函数分析
    thetas = np.linspace(-0.3, 0.3, 50)
    values = []

    for theta in thetas:
        state = np.array([0, 0, theta, 0])
        value = agent.critic.forward(state)
        values.append(value)

    axes[1].plot(thetas, values, color='green')
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('杆子角度')
    axes[1].set_ylabel('状态值')
    axes[1].set_title('Critic: 状态值 vs 角度')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Actor-Critic 可视化已保存到 {save_path}")


# ============================================================================
# 数学原理说明
# ============================================================================

def explain_a2c_math():
    """解释 A2C 数学原理"""
    print("\n" + "=" * 60)
    print("Actor-Critic 数学原理")
    print("=" * 60)

    print("""
1. Actor-Critic 架构

   组成：
       - Actor（演员）: 策略网络 π(a|s; θ)，决定动作选择
       - Critic（评论家）: 值网络 V(s; w)，评估状态价值

   协作机制：
       1. Actor 选择动作
       2. 环境返回奖励和下一状态
       3. Critic 评估当前状态价值
       4. 计算优势函数 A(s,a)
       5. Actor 根据优势更新策略
       6. Critic 根据TD误差更新值函数

2. 优势函数

   定义：
       A(s,a) = Q(s,a) - V(s)

   TD 形式：
       A(s,a) = r + γV(s') - V(s)

   含义：
       - 动作 a 相比平均好多少
       - A > 0: 动作优于平均，增加概率
       - A < 0: 动作劣于平均，减少概率

3. 更新规则

   Actor 更新（策略梯度）：
       θ ← θ + α_actor * ∇_θ log π(a|s) * A(s,a)

   Critic 更新（TD学习）：
       δ = r + γV(s') - V(s)  (TD误差)
       w ← w - α_critic * δ * ∇_w V(s)

4. 与 REINFORCE 的区别

   REINFORCE:
       - 使用完整轨迹的回报 G_t
       - 蒙特卡洛方法
       - 高方差，无偏

   A2C:
       - 使用 Critic 估计的值函数
       - TD 方法
       - 低方差，有偏

5. A2C vs A3C

   A2C (Advantage Actor-Critic):
       - 同步更新
       - 单线程或多线程收集数据，统一更新

   A3C (Asynchronous A2C):
       - 异步更新
       - 多个智能体并行探索
       - 不需要经验回放

6. 超参数选择

   学习率：
       - Actor 通常比 Critic 小
       - 典型值: α_actor = 0.001, α_critic = 0.01

   n-step：
       - n=1: 单步 TD，低方差高偏差
       - n=∞: 蒙特卡洛，高方差低偏差
       - 通常 n=3~5 平衡两者
""")


# ============================================================================
# 测试
# ============================================================================

def test_networks():
    """测试网络"""
    print("测试 Actor 和 Critic 网络...")

    n_states, n_actions = 4, 2

    # Actor
    actor = ActorNetwork(n_states, n_actions)
    state = np.random.randn(n_states).astype(np.float32)
    probs = actor.forward(state)

    assert probs.shape == (n_actions,), f"Actor 输出形状错误: {probs.shape}"
    assert np.isclose(np.sum(probs), 1.0), "Actor 输出概率和不为 1"

    # Critic
    critic = CriticNetwork(n_states)
    value = critic.forward(state)

    assert isinstance(value, float), f"Critic 输出类型错误: {type(value)}"

    print("  ✓ 网络正确")


def test_a2c_agent():
    """测试 A2C 智能体"""
    print("测试 A2C 智能体...")

    env = CartPole()
    agent = A2CAgent(n_states=4, n_actions=2)

    state = env.reset()

    # 测试动作选择
    action, _ = agent.get_action(state)
    assert action in [0, 1], f"动作错误: {action}"

    # 测试更新
    next_state, reward, done, _ = env.step(action)
    stats = agent.update(state, action, reward, next_state, done)

    assert 'td_error' in stats, "更新返回缺少 td_error"
    assert 'advantage' in stats, "更新返回缺少 advantage"

    print("  ✓ A2C 智能体正确")


def test_learning():
    """测试学习"""
    print("测试学习过程...")

    env = CartPole()
    agent = A2CAgent(n_states=4, n_actions=2, actor_lr=0.01, critic_lr=0.01)

    stats = train_a2c(env, agent, n_episodes=200, verbose=False)

    final_avg = np.mean(stats['episode_rewards'][-20:])
    print(f"  最终平均奖励: {final_avg:.2f}")

    print("  ✓ 学习过程正常")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行 A2C 测试")
    print("=" * 60)

    test_networks()
    test_a2c_agent()
    test_learning()

    print("\n所有测试通过!")


def main():
    """主函数"""
    print("=" * 60)
    print("Actor-Critic (A2C) 算法详解")
    print("=" * 60)

    # 解释数学原理
    explain_a2c_math()

    # 创建环境和智能体
    env = CartPole()
    agent = A2CAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        actor_lr=0.01,
        critic_lr=0.01
    )

    # 训练
    print("\n开始训练...")
    stats = train_a2c(env, agent, n_episodes=500, verbose=True)

    # 可视化
    visualize_training(stats)
    visualize_actor_critic(agent, env)

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    run_tests()

    # 运行主程序
    main()