#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q-Learning 算法详解

本模块从零实现 Q-Learning 算法，深入理解值函数学习和 TD 方法。

数学知识：
- Q 值函数
- 贝尔曼方程
- TD 学习
- ε-greedy 探索

实现内容：
- QLearningAgent: Q-Learning 智能体
- GridWorld: 网格世界环境
- Q 表更新
- 可视化：Q 值热力图、策略演化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional


# ============================================================================
# 环境定义
# ============================================================================

class GridWorld:
    """
    网格世界环境

    状态空间：网格位置 (row, col)
    动作空间：上(0)、下(1)、左(2)、右(3)
    奖励：
        - 到达目标：+1
        - 到达陷阱：-1
        - 其他：-0.01（鼓励快速到达目标）

    数学表示：
        - 状态 s ∈ S = {(row, col) | 0 ≤ row < height, 0 ≤ col < width}
        - 动作 a ∈ A = {上, 下, 左, 右}
        - 转移 P(s'|s,a)：确定性转移（可选随机噪声）
        - 奖励 r(s,a,s')
    """

    def __init__(self,
                 height: int = 5,
                 width: int = 5,
                 start: Tuple[int, int] = (0, 0),
                 goal: Tuple[int, int] = (4, 4),
                 obstacles: List[Tuple[int, int]] = None,
                 traps: List[Tuple[int, int]] = None,
                 noise: float = 0.0):
        """
        初始化网格世界

        Args:
            height: 网格高度
            width: 网格宽度
            start: 起点位置
            goal: 目标位置
            obstacles: 障碍物列表
            traps: 陷阱列表
            noise: 动作噪声概率（随机选择其他动作）
        """
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.noise = noise

        # 障碍物和陷阱
        self.obstacles = obstacles if obstacles else []
        self.traps = traps if traps else []

        # 动作定义
        self.actions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        self.n_actions = 4

        # 状态空间
        self.n_states = height * width

        # 当前状态
        self.state = None

    def reset(self) -> Tuple[int, int]:
        """
        重置环境

        Returns:
            初始状态
        """
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作索引

        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 应用动作噪声
        if np.random.random() < self.noise:
            action = np.random.choice([a for a in range(self.n_actions) if a != action])

        # 计算下一位置
        dr, dc = self.actions[action]
        row, col = self.state
        new_row = np.clip(row + dr, 0, self.height - 1)
        new_col = np.clip(col + dc, 0, self.width - 1)
        next_state = (new_row, new_col)

        # 检查障碍物
        if next_state in self.obstacles:
            next_state = self.state  # 不能移动

        # 计算奖励
        done = False
        reward = -0.01  # 默认小负奖励

        if next_state == self.goal:
            reward = 1.0
            done = True
        elif next_state in self.traps:
            reward = -1.0
            done = True

        self.state = next_state
        return next_state, reward, done, {}

    def render(self) -> None:
        """打印环境"""
        symbols = {
            'empty': '.',
            'start': 'S',
            'goal': 'G',
            'obstacle': '#',
            'trap': 'T',
            'agent': 'A'
        }

        for i in range(self.height):
            row_str = ""
            for j in range(self.width):
                pos = (i, j)
                if pos == self.state:
                    row_str += symbols['agent'] + " "
                elif pos == self.start:
                    row_str += symbols['start'] + " "
                elif pos == self.goal:
                    row_str += symbols['goal'] + " "
                elif pos in self.obstacles:
                    row_str += symbols['obstacle'] + " "
                elif pos in self.traps:
                    row_str += symbols['trap'] + " "
                else:
                    row_str += symbols['empty'] + " "
            print(row_str)
        print()


# ============================================================================
# Q-Learning 智能体
# ============================================================================

class QLearningAgent:
    """
    Q-Learning 智能体

    数学原理：
        Q-Learning 是一种时序差分（TD）控制算法，学习最优动作值函数。

        更新公式：
            Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        其中：
            - α: 学习率
            - γ: 折扣因子
            - r + γ * max_a' Q(s', a'): TD 目标
            - r + γ * max_a' Q(s', a') - Q(s, a): TD 误差

        探索策略：ε-greedy
            - 以概率 ε 随机选择动作（探索）
            - 以概率 1-ε 选择最优动作（利用）
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 0.1):
        """
        初始化 Q-Learning 智能体

        Args:
            n_states: 状态数量
            n_actions: 动作数量
            learning_rate: 学习率 α
            discount_factor: 折扣因子 γ
            epsilon: 探索率 ε
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # 初始化 Q 表
        self.q_table = np.zeros((n_states, n_actions))

    def state_to_index(self, state: Tuple[int, int], env_width: int) -> int:
        """将状态转换为索引"""
        return state[0] * env_width + state[1]

    def get_action(self, state: Tuple[int, int], env_width: int, training: bool = True) -> int:
        """
        选择动作（ε-greedy 策略）

        Args:
            state: 当前状态
            env_width: 环境宽度
            training: 是否在训练模式

        Returns:
            选择的动作
        """
        state_idx = self.state_to_index(state, env_width)

        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择最优动作
            return np.argmax(self.q_table[state_idx])

    def update(self,
               state: Tuple[int, int],
               action: int,
               reward: float,
               next_state: Tuple[int, int],
               done: bool,
               env_width: int) -> float:
        """
        更新 Q 表

        Q-Learning 更新公式：
            Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
            env_width: 环境宽度

        Returns:
            TD 误差
        """
        state_idx = self.state_to_index(state, env_width)
        next_state_idx = self.state_to_index(next_state, env_width)

        # 当前 Q 值
        current_q = self.q_table[state_idx, action]

        # 计算目标 Q 值
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_idx])

        # 计算 TD 误差
        td_error = target_q - current_q

        # 更新 Q 值
        self.q_table[state_idx, action] += self.lr * td_error

        return td_error

    def get_policy(self, env_width: int) -> np.ndarray:
        """
        获取当前策略

        Returns:
            策略数组，每个状态对应最优动作
        """
        return np.argmax(self.q_table, axis=1).reshape(-1, env_width)

    def get_value_function(self, env_width: int) -> np.ndarray:
        """
        获取状态值函数

        V(s) = max_a Q(s, a)

        Returns:
            值函数数组
        """
        return np.max(self.q_table, axis=1).reshape(-1, env_width)


# ============================================================================
# 训练函数
# ============================================================================

def train_qlearning(
    env: GridWorld,
    agent: QLearningAgent,
    n_episodes: int = 1000,
    max_steps: int = 100,
    verbose: bool = True
) -> Dict[str, List]:
    """
    训练 Q-Learning 智能体

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
        'td_errors': []
    }

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_td_errors = []

        for step in range(max_steps):
            # 选择动作
            action = agent.get_action(state, env.width, training=True)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 更新 Q 表
            td_error = agent.update(state, action, reward, next_state, done, env.width)
            episode_td_errors.append(abs(td_error))

            episode_reward += reward
            state = next_state

            if done:
                break

        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(step + 1)
        stats['td_errors'].append(np.mean(episode_td_errors))

        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(stats['episode_rewards'][-100:])
            print(f"Episode {episode + 1}/{n_episodes}, 平均奖励: {avg_reward:.2f}")

    return stats


# ============================================================================
# 可视化
# ============================================================================

def visualize_q_table(
    agent: QLearningAgent,
    env: GridWorld,
    save_path: str = "q_learning_q_table.png"
) -> None:
    """
    可视化 Q 表

    Args:
        agent: Q-Learning 智能体
        env: 环境
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 绘制每个动作的 Q 值
    action_names = ['上', '下', '左', '右']
    for i, name in enumerate(action_names):
        ax = axes[0, i] if i < 2 else axes[1, i - 2]
        q_values = agent.q_table[:, i].reshape(env.height, env.width)

        im = ax.imshow(q_values, cmap='RdYlGn')
        ax.set_title(f'Q(s, {name})')
        ax.set_xlabel('列')
        ax.set_ylabel('行')
        plt.colorbar(im, ax=ax)

        # 标记特殊位置
        ax.scatter(env.goal[1], env.goal[0], c='blue', s=100, marker='*', label='目标')
        ax.scatter(env.start[1], env.start[0], c='green', s=100, marker='o', label='起点')

    # 绘制最优值函数 V(s) = max_a Q(s,a)
    ax = axes[1, 2]
    v_values = agent.get_value_function(env.width)
    im = ax.imshow(v_values, cmap='RdYlGn')
    ax.set_title('V(s) = max_a Q(s,a)')
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Q 表可视化已保存到 {save_path}")


def visualize_policy(
    agent: QLearningAgent,
    env: GridWorld,
    save_path: str = "q_learning_policy.png"
) -> None:
    """
    可视化策略

    Args:
        agent: Q-Learning 智能体
        env: 环境
        save_path: 保存路径
    """
    policy = agent.get_policy(env.width)
    v_values = agent.get_value_function(env.width)

    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制值函数背景
    im = ax.imshow(v_values, cmap='RdYlGn', alpha=0.7)

    # 绘制策略箭头
    action_arrows = {
        0: (0, 0.3),   # 上
        1: (0, -0.3),  # 下
        2: (-0.3, 0),  # 左
        3: (0.3, 0)    # 右
    }

    for i in range(env.height):
        for j in range(env.width):
            if (i, j) == env.goal or (i, j) in env.obstacles:
                continue

            action = policy[i, j]
            dx, dy = action_arrows[action]
            ax.arrow(j, i, dx, dy, head_width=0.1, head_length=0.05, fc='black', ec='black')

    # 标记特殊位置
    ax.scatter(env.goal[1], env.goal[0], c='blue', s=200, marker='*', label='目标')
    ax.scatter(env.start[1], env.start[0], c='green', s=200, marker='o', label='起点')

    for obs in env.obstacles:
        ax.scatter(obs[1], obs[0], c='gray', s=200, marker='s', label='障碍')

    for trap in env.traps:
        ax.scatter(trap[1], trap[0], c='red', s=200, marker='x', label='陷阱')

    ax.set_title('学习到的策略')
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    plt.colorbar(im, ax=ax, label='状态值')

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"策略可视化已保存到 {save_path}")


def visualize_training_stats(
    stats: Dict[str, List],
    save_path: str = "q_learning_stats.png"
) -> None:
    """
    可视化训练统计

    Args:
        stats: 训练统计信息
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 奖励曲线
    axes[0].plot(stats['episode_rewards'], alpha=0.6)
    # 移动平均
    window = min(100, len(stats['episode_rewards']))
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
    axes[2].set_ylabel('平均 TD 误差')
    axes[2].set_title('TD 误差变化')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练统计已保存到 {save_path}")


def visualize_learning_curve(
    rewards_list: List[List[float]],
    labels: List[str],
    save_path: str = "q_learning_comparison.png"
) -> None:
    """
    可视化不同超参数的学习曲线

    Args:
        rewards_list: 多个实验的奖励列表
        labels: 实验标签
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))

    for rewards, label in zip(rewards_list, labels):
        window = min(50, len(rewards))
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(moving_avg, label=label)

    plt.xlabel('Episode')
    plt.ylabel('平均奖励')
    plt.title('不同超参数的学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"学习曲线对比已保存到 {save_path}")


# ============================================================================
# 演示和示例
# ============================================================================

def demo_basic_qlearning():
    """基础 Q-Learning 演示"""
    print("\n" + "=" * 60)
    print("基础 Q-Learning 演示")
    print("=" * 60)

    # 创建环境
    env = GridWorld(
        height=5, width=5,
        start=(0, 0), goal=(4, 4),
        obstacles=[(1, 1), (2, 2), (3, 3)],
        traps=[(3, 0), (0, 3)]
    )

    print("环境布局:")
    env.render()

    # 创建智能体
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1
    )

    # 训练
    print("\n开始训练...")
    stats = train_qlearning(env, agent, n_episodes=1000, verbose=True)

    # 可视化
    visualize_q_table(agent, env)
    visualize_policy(agent, env)
    visualize_training_stats(stats)

    # 测试
    print("\n测试训练后的策略:")
    state = env.reset()
    total_reward = 0

    for step in range(20):
        action = agent.get_action(state, env.width, training=False)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        print(f"  步骤 {step+1}: 状态={state}, 动作={['上','下','左','右'][action]}, 奖励={reward:.2f}")
        state = next_state

        if done:
            print(f"  到达终点！总奖励: {total_reward:.2f}")
            break

    return agent, env, stats


def demo_hyperparameter_comparison():
    """超参数对比实验"""
    print("\n" + "=" * 60)
    print("超参数对比实验")
    print("=" * 60)

    # 创建环境
    env = GridWorld(height=5, width=5, start=(0, 0), goal=(4, 4))

    # 不同学习率
    learning_rates = [0.01, 0.1, 0.5]
    results = []

    for lr in learning_rates:
        agent = QLearningAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            learning_rate=lr,
            discount_factor=0.99,
            epsilon=0.1
        )

        stats = train_qlearning(env, agent, n_episodes=500, verbose=False)
        results.append(stats['episode_rewards'])
        print(f"学习率 {lr}: 最终平均奖励 = {np.mean(stats['episode_rewards'][-50:]):.2f}")

    visualize_learning_curve(results, [f'lr={lr}' for lr in learning_rates])

    # 不同探索率
    epsilons = [0.01, 0.1, 0.3]
    results = []

    for eps in epsilons:
        agent = QLearningAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=eps
        )

        stats = train_qlearning(env, agent, n_episodes=500, verbose=False)
        results.append(stats['episode_rewards'])
        print(f"探索率 {eps}: 最终平均奖励 = {np.mean(stats['episode_rewards'][-50:]):.2f}")

    visualize_learning_curve(results, [f'ε={eps}' for eps in epsilons], "q_learning_epsilon_comparison.png")


def explain_qlearning_math():
    """解释 Q-Learning 数学原理"""
    print("\n" + "=" * 60)
    print("Q-Learning 数学原理")
    print("=" * 60)

    print("""
1. 马尔可夫决策过程 (MDP)

   MDP 由五元组定义: (S, A, P, R, γ)
   - S: 状态空间
   - A: 动作空间
   - P: 转移概率 P(s'|s,a)
   - R: 奖励函数 R(s,a,s')
   - γ: 折扣因子

2. 值函数

   状态值函数:
       V^π(s) = E_π[Σ γ^t r_t | S_0 = s]

   动作值函数:
       Q^π(s,a) = E_π[Σ γ^t r_t | S_0 = s, A_0 = a]

3. 贝尔曼方程

   对于确定性策略 π:
       V^π(s) = Σ_a π(a|s) * Σ_s' P(s'|s,a) * [R(s,a,s') + γV^π(s')]

   最优贝尔曼方程:
       V*(s) = max_a Σ_s' P(s'|s,a) * [R(s,a,s') + γV*(s')]

       Q*(s,a) = Σ_s' P(s'|s,a) * [R(s,a,s') + γ max_a' Q*(s',a')]

4. Q-Learning 更新

   TD 目标:
       y = r + γ * max_a' Q(s', a')

   TD 误差:
       δ = y - Q(s, a)

   更新:
       Q(s, a) ← Q(s, a) + α * δ

5. 收敛性

   Q-Learning 在以下条件下收敛到最优 Q*:
   - 所有状态-动作对被无限次访问
   - 学习率 α_t 满足:
       Σ α_t = ∞, Σ α_t² < ∞
   - 通常使用衰减的学习率

6. 探索与利用

   ε-greedy 策略:
       π(a|s) = {
           1-ε + ε/|A|,  if a = argmax_a Q(s,a)
           ε/|A|,        otherwise
       }

   探索率衰减:
       ε_t = max(ε_min, ε_0 * decay^t)
""")


# ============================================================================
# 测试
# ============================================================================

def test_gridworld():
    """测试网格世界环境"""
    print("测试 GridWorld 环境...")

    env = GridWorld(height=3, width=3, start=(0, 0), goal=(2, 2))

    # 测试重置
    state = env.reset()
    assert state == (0, 0), f"重置状态错误: {state}"

    # 测试动作
    next_state, reward, done, _ = env.step(3)  # 右
    assert next_state == (0, 1), f"右移动错误: {next_state}"

    next_state, reward, done, _ = env.step(1)  # 下
    assert next_state == (1, 1), f"下移动错误: {next_state}"

    print("  ✓ GridWorld 环境正确")


def test_qlearning_agent():
    """测试 Q-Learning 智能体"""
    print("测试 Q-Learning 智能体...")

    env = GridWorld(height=3, width=3, start=(0, 0), goal=(2, 2))
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions
    )

    # 测试状态索引转换
    idx = agent.state_to_index((1, 1), env.width)
    expected = 1 * 3 + 1  # 4
    assert idx == expected, f"状态索引错误: {idx}"

    # 测试动作选择
    action = agent.get_action((0, 0), env.width, training=False)
    assert 0 <= action < 4, f"动作索引错误: {action}"

    print("  ✓ Q-Learning 智能体正确")


def test_qlearning_update():
    """测试 Q-Learning 更新"""
    print("测试 Q-Learning 更新...")

    env = GridWorld(height=3, width=3, start=(0, 0), goal=(2, 2))
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=1.0  # 完全更新
    )

    # 手动设置 Q 值
    state = (0, 0)
    action = 3  # 右
    env.state = state
    next_state, reward, done, _ = env.step(action)

    # 更新
    agent.update(state, action, reward, next_state, done, env.width, env.width)

    state_idx = agent.state_to_index(state, env.width)
    assert agent.q_table[state_idx, action] != 0, "Q 值未更新"

    print("  ✓ Q-Learning 更新正确")


def test_convergence():
    """测试收敛性"""
    print("测试收敛性...")

    env = GridWorld(height=3, width=3, start=(0, 0), goal=(2, 2))
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.5
    )

    # 训练足够长时间
    stats = train_qlearning(env, agent, n_episodes=1000, verbose=False)

    # 检查是否学会到达目标
    final_avg_reward = np.mean(stats['episode_rewards'][-100:])
    assert final_avg_reward > 0.5, f"最终平均奖励太低: {final_avg_reward}"

    print(f"  ✓ 最终平均奖励: {final_avg_reward:.2f}")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行 Q-Learning 测试")
    print("=" * 60)

    test_gridworld()
    test_qlearning_agent()
    test_qlearning_update()
    test_convergence()

    print("\n所有测试通过!")


def main():
    """主函数"""
    print("=" * 60)
    print("Q-Learning 算法详解")
    print("=" * 60)

    # 解释数学原理
    explain_qlearning_math()

    # 基础演示
    demo_basic_qlearning()

    # 超参数对比
    demo_hyperparameter_comparison()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    run_tests()

    # 运行主程序
    main()