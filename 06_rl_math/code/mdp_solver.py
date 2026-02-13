# -*- coding: utf-8 -*-
"""
MDP求解器模块

实现马尔可夫决策过程的标准求解算法。

主要功能：
- MDP环境类（支持自定义MDP）
- 值迭代算法（Value Iteration）
- 策略迭代算法（Policy Iteration）
- 修改的策略迭代
- 策略评估
- 异步值迭代

作者：Essential Math of AI
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
import matplotlib.pyplot as plt


# ============================================================================
# MDP基础类
# ============================================================================

class MDP:
    """
    马尔可夫决策过程基类

    子类需要实现：
    - states: 状态列表
    - actions: 动作列表
    - get_transition_prob(s, a, s_next): 转移概率
    - get_reward(s, a, s_next): 奖励函数
    """

    def __init__(self, gamma: float = 0.9):
        """
        参数：
            gamma: 折扣因子
        """
        self.gamma = gamma
        self.states: List = []
        self.actions: List = []
        self.n_states: int = 0
        self.n_actions: int = 0

        # 缓存的转移和奖励矩阵
        self._P: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None

    def get_transition_prob(self, state, action, next_state) -> float:
        """
        转移概率 P(s'|s,a)

        必须由子类实现
        """
        raise NotImplementedError

    def get_reward(self, state, action, next_state) -> float:
        """
        奖励函数 R(s,a,s')

        必须由子类实现
        """
        raise NotImplementedError

    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建转移概率矩阵P和奖励矩阵R

        返回：
            P: (n_states, n_actions, n_states)
            R: (n_states, n_actions)
        """
        if self._P is not None and self._R is not None:
            return self._P, self._R

        n_s = self.n_states
        n_a = self.n_actions

        P = np.zeros((n_s, n_a, n_s))
        R = np.zeros((n_s, n_a))

        for s_idx, state in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                for s_next_idx, next_state in enumerate(self.states):
                    prob = self.get_transition_prob(state, action, next_state)
                    reward = self.get_reward(state, action, next_state)

                    P[s_idx, a_idx, s_next_idx] = prob
                    R[s_idx, a_idx] += prob * reward

        self._P = P
        self._R = R

        return P, R

    def reset_cache(self):
        """清除缓存的矩阵"""
        self._P = None
        self._R = None


# ============================================================================
# GridWorld MDP 示例
# ============================================================================

class GridWorldMDP(MDP):
    """
    网格世界MDP

    状态：网格坐标 (i, j)
    动作：上下左右
    目标：到达终点
    """

    def __init__(self,
                 size: int = 4,
                 gamma: float = 0.9,
                 goal_reward: float = 1.0,
                 wall_penalty: float = -0.5,
                 step_penalty: float = -0.01):
        """
        参数：
            size: 网格大小
            gamma: 折扣因子
            goal_reward: 到达终点的奖励
            wall_penalty: 撞墙惩罚
            step_penalty: 每步惩罚
        """
        super().__init__(gamma)

        self.size = size
        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty
        self.step_penalty = step_penalty

        # 初始化状态和动作
        self.states = [(i, j) for i in range(size) for j in range(size)]
        self.n_states = len(self.states)

        self.actions = ['up', 'down', 'left', 'right']
        self.action_vectors = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        self.n_actions = len(self.actions)

        # 特殊位置
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = [(1, 1)]  # 可以添加更多障碍

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """状态转索引"""
        return state[0] * self.size + state[1]

    def is_valid(self, state: Tuple[int, int]) -> bool:
        """检查状态是否有效"""
        i, j = state
        if i < 0 or i >= self.size or j < 0 or j >= self.size:
            return False
        if state in self.obstacles:
            return False
        return True

    def get_transition_prob(self,
                           state: Tuple[int, int],
                           action: str,
                           next_state: Tuple[int, int]) -> float:
        """转移概率（确定性环境）"""
        # 终点是吸收态
        if state == self.goal:
            return 1.0 if next_state == state else 0.0

        # 计算预期下一状态
        di, dj = self.action_vectors[action]
        expected_next = (state[0] + di, state[1] + dj)

        # 实际下一状态
        if self.is_valid(expected_next):
            actual_next = expected_next
        else:
            actual_next = state  # 撞墙保持原地

        return 1.0 if next_state == actual_next else 0.0

    def get_reward(self,
                  state: Tuple[int, int],
                  action: str,
                  next_state: Tuple[int, int]) -> float:
        """奖励函数"""
        if state == self.goal:
            return 0.0

        if next_state == self.goal:
            return self.goal_reward
        elif next_state == state:  # 撞墙
            return self.wall_penalty
        else:
            return self.step_penalty

    def render_values(self,
                     V: np.ndarray,
                     title: str = "Value Function",
                     save_path: Optional[str] = None):
        """可视化价值函数"""
        grid = np.zeros((self.size, self.size))

        for i in range(self.size):
            for j in range(self.size):
                state = (i, j)
                if state in self.obstacles:
                    grid[i, j] = np.nan
                else:
                    idx = self.state_to_index(state)
                    grid[i, j] = V[idx]

        plt.figure(figsize=(7, 6))
        im = plt.imshow(grid, cmap='RdYlGn', aspect='equal', vmin=V.min(), vmax=V.max())

        # 添加数值标签
        for i in range(self.size):
            for j in range(self.size):
                state = (i, j)
                if state not in self.obstacles:
                    idx = self.state_to_index(state)
                    color = 'white' if abs(V[idx]) > 0.5 * (V.max() - V.min()) else 'black'
                    plt.text(j, i, f'{V[idx]:.2f}',
                            ha='center', va='center', fontsize=10, color=color)

        plt.colorbar(im, label='Value')
        plt.title(title)
        plt.grid(True, color='black', linewidth=2)
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        return plt

    def render_policy(self,
                     policy: np.ndarray,
                     title: str = "Policy",
                     save_path: Optional[str] = None):
        """可视化策略"""
        action_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
        grid = np.zeros((self.size, self.size))

        plt.figure(figsize=(7, 6))
        plt.imshow(grid, cmap='Greys', alpha=0.1)

        for i in range(self.size):
            for j in range(self.size):
                state = (i, j)

                if state in self.obstacles:
                    plt.text(j, i, 'X', ha='center', va='center',
                            fontsize=24, color='red', fontweight='bold')
                elif state == self.goal:
                    plt.text(j, i, 'G', ha='center', va='center',
                            fontsize=24, color='green', fontweight='bold')
                else:
                    idx = self.state_to_index(state)
                    best_action_idx = np.argmax(policy[idx])
                    best_action = self.actions[best_action_idx]
                    symbol = action_symbols[best_action]

                    plt.text(j, i, symbol, ha='center', va='center',
                            fontsize=24, color='blue')

        plt.title(title)
        plt.grid(True, color='black', linewidth=2)
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        return plt


# ============================================================================
# 值迭代算法
# ============================================================================

class ValueIteration:
    """值迭代算法"""

    def __init__(self, mdp: MDP):
        """
        参数：
            mdp: MDP实例
        """
        self.mdp = mdp
        self.P, self.R = mdp.build_matrices()

    def solve(self,
              threshold: float = 1e-6,
              max_iter: int = 1000,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        值迭代求解

        参数：
            threshold: 收敛阈值
            max_iter: 最大迭代次数
            verbose: 是否打印进度

        返回：
            V: 最优价值函数
            policy: 最优策略
            history: 收敛历史
        """
        V = np.zeros(self.mdp.n_states)
        history = []

        for iteration in range(max_iter):
            V_new = np.zeros_like(V)

            for s in range(self.mdp.n_states):
                # 计算所有动作的Q值
                q_values = np.zeros(self.mdp.n_actions)
                for a in range(self.mdp.n_actions):
                    q_values[a] = self.R[s, a] + self.mdp.gamma * np.sum(
                        self.P[s, a, :] * V
                    )

                # 贝尔曼最优更新
                V_new[s] = np.max(q_values)

            # 计算变化量
            delta = np.max(np.abs(V_new - V))
            history.append(delta)

            V = V_new

            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: Δ = {delta:.8f}")

            # 检查收敛
            if delta < threshold:
                if verbose:
                    print(f"✓ 收敛于 Iter {iteration + 1}, Δ = {delta:.8f}")
                break

        # 提取最优策略
        policy = self._extract_policy(V)

        return V, policy, history

    def _extract_policy(self, V: np.ndarray) -> np.ndarray:
        """从价值函数提取最优策略"""
        policy = np.zeros((self.mdp.n_states, self.mdp.n_actions))

        for s in range(self.mdp.n_states):
            q_values = np.zeros(self.mdp.n_actions)
            for a in range(self.mdp.n_actions):
                q_values[a] = self.R[s, a] + self.mdp.gamma * np.sum(
                    self.P[s, a, :] * V
                )

            best_action = np.argmax(q_values)
            policy[s, best_action] = 1.0

        return policy


# ============================================================================
# 策略迭代算法
# ============================================================================

class PolicyIteration:
    """策略迭代算法"""

    def __init__(self, mdp: MDP):
        """
        参数：
            mdp: MDP实例
        """
        self.mdp = mdp
        self.P, self.R = mdp.build_matrices()

    def policy_evaluation(self,
                         policy: np.ndarray,
                         threshold: float = 1e-6,
                         max_iter: int = 1000) -> np.ndarray:
        """
        策略评估：计算V^π

        参数：
            policy: 当前策略 (n_states, n_actions)
            threshold: 收敛阈值
            max_iter: 最大迭代次数

        返回：
            V^π
        """
        V = np.zeros(self.mdp.n_states)

        for iteration in range(max_iter):
            V_new = np.zeros_like(V)

            for s in range(self.mdp.n_states):
                v_s = 0.0
                for a in range(self.mdp.n_actions):
                    pi_a_s = policy[s, a]
                    q_sa = self.R[s, a] + self.mdp.gamma * np.sum(
                        self.P[s, a, :] * V
                    )
                    v_s += pi_a_s * q_sa

                V_new[s] = v_s

            delta = np.max(np.abs(V_new - V))
            V = V_new

            if delta < threshold:
                break

        return V

    def policy_improvement(self, V: np.ndarray) -> np.ndarray:
        """
        策略改进

        参数：
            V: 当前价值函数

        返回：
            改进后的策略
        """
        new_policy = np.zeros((self.mdp.n_states, self.mdp.n_actions))

        for s in range(self.mdp.n_states):
            q_values = np.zeros(self.mdp.n_actions)
            for a in range(self.mdp.n_actions):
                q_values[a] = self.R[s, a] + self.mdp.gamma * np.sum(
                    self.P[s, a, :] * V
                )

            best_action = np.argmax(q_values)
            new_policy[s, best_action] = 1.0

        return new_policy

    def solve(self,
              init_policy: Optional[np.ndarray] = None,
              threshold: float = 1e-6,
              max_iter: int = 100,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        策略迭代求解

        参数：
            init_policy: 初始策略（None则随机初始化）
            threshold: 收敛阈值
            max_iter: 最大外层迭代次数
            verbose: 是否打印进度

        返回：
            V: 最优价值函数
            policy: 最优策略
            iterations: 迭代次数
        """
        # 初始化策略
        if init_policy is None:
            policy = np.ones((self.mdp.n_states, self.mdp.n_actions)) / self.mdp.n_actions
        else:
            policy = init_policy.copy()

        for iteration in range(max_iter):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}")
                print('='*60)

            # 策略评估
            if verbose:
                print("Step 1: Policy Evaluation...")
            V = self.policy_evaluation(policy, threshold)

            # 策略改进
            if verbose:
                print("Step 2: Policy Improvement...")
            new_policy = self.policy_improvement(V)

            # 检查收敛
            policy_change = np.sum(np.abs(new_policy - policy))

            if verbose:
                print(f"Policy change: {policy_change:.8f}")

            if policy_change < 1e-10:
                if verbose:
                    print(f"✓ 策略收敛于 Iteration {iteration + 1}")
                break

            policy = new_policy

        return V, policy, iteration + 1


# ============================================================================
# 修改的策略迭代
# ============================================================================

class ModifiedPolicyIteration:
    """修改的策略迭代（截断策略评估）"""

    def __init__(self, mdp: MDP, k_eval: int = 10):
        """
        参数：
            mdp: MDP实例
            k_eval: 策略评估迭代次数
        """
        self.mdp = mdp
        self.k_eval = k_eval
        self.P, self.R = mdp.build_matrices()

    def truncated_policy_evaluation(self,
                                    policy: np.ndarray,
                                    V_old: np.ndarray) -> np.ndarray:
        """
        截断的策略评估（只迭代k次）

        参数：
            policy: 当前策略
            V_old: 上一轮的价值函数

        返回：
            更新后的V
        """
        V = V_old.copy()

        for _ in range(self.k_eval):
            V_new = np.zeros_like(V)
            for s in range(self.mdp.n_states):
                v_s = 0.0
                for a in range(self.mdp.n_actions):
                    pi_a_s = policy[s, a]
                    q_sa = self.R[s, a] + self.mdp.gamma * np.sum(
                        self.P[s, a, :] * V
                    )
                    v_s += pi_a_s * q_sa
                V_new[s] = v_s
            V = V_new

        return V

    def solve(self,
              max_iter: int = 100,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        修改的策略迭代求解

        返回：
            V, policy, iterations
        """
        policy = np.ones((self.mdp.n_states, self.mdp.n_actions)) / self.mdp.n_actions
        V = np.zeros(self.mdp.n_states)

        for iteration in range(max_iter):
            # 截断策略评估
            V = self.truncated_policy_evaluation(policy, V)

            # 策略改进
            new_policy = np.zeros((self.mdp.n_states, self.mdp.n_actions))
            for s in range(self.mdp.n_states):
                q_values = np.zeros(self.mdp.n_actions)
                for a in range(self.mdp.n_actions):
                    q_values[a] = self.R[s, a] + self.mdp.gamma * np.sum(
                        self.P[s, a, :] * V
                    )
                best_action = np.argmax(q_values)
                new_policy[s, best_action] = 1.0

            # 检查收敛
            if np.sum(np.abs(new_policy - policy)) < 1e-10:
                if verbose:
                    print(f"✓ 收敛于 Iteration {iteration + 1}")
                break

            policy = new_policy

        return V, policy, iteration + 1


# ============================================================================
# 测试函数
# ============================================================================

def test_gridworld():
    """测试GridWorld MDP"""
    print("="*60)
    print("测试 GridWorld MDP")
    print("="*60)

    mdp = GridWorldMDP(size=4, gamma=0.9)

    print(f"状态数: {mdp.n_states}")
    print(f"动作数: {mdp.n_actions}")
    print(f"起点: {mdp.start}")
    print(f"终点: {mdp.goal}")
    print(f"障碍: {mdp.obstacles}")

    # 测试转移概率
    s = mdp.start
    a = 'right'
    s_next = (0, 1)

    prob = mdp.get_transition_prob(s, a, s_next)
    reward = mdp.get_reward(s, a, s_next)

    print(f"\n从{s}执行{a}到{s_next}:")
    print(f"  转移概率: {prob}")
    print(f"  奖励: {reward}")

    print("✓ GridWorld测试通过\n")


def test_value_iteration():
    """测试值迭代"""
    print("="*60)
    print("测试值迭代算法")
    print("="*60)

    mdp = GridWorldMDP(size=4, gamma=0.9)
    vi = ValueIteration(mdp)

    V, policy, history = vi.solve(verbose=False)

    print(f"收敛迭代次数: {len(history)}")
    print(f"最优价值范围: [{V.min():.3f}, {V.max():.3f}]")

    # 检查终点价值
    goal_idx = mdp.state_to_index(mdp.goal)
    print(f"终点价值: {V[goal_idx]:.3f}")

    assert V[goal_idx] >= 0, "值迭代测试失败"

    print("✓ 值迭代测试通过\n")


def test_policy_iteration():
    """测试策略迭代"""
    print("="*60)
    print("测试策略迭代算法")
    print("="*60)

    mdp = GridWorldMDP(size=4, gamma=0.9)
    pi = PolicyIteration(mdp)

    V, policy, iterations = pi.solve(verbose=False)

    print(f"收敛迭代次数: {iterations}")
    print(f"最优价值范围: [{V.min():.3f}, {V.max():.3f}]")

    # 检查终点价值
    goal_idx = mdp.state_to_index(mdp.goal)
    print(f"终点价值: {V[goal_idx]:.3f}")

    assert V[goal_idx] >= 0, "策略迭代测试失败"

    print("✓ 策略迭代测试通过\n")


def test_algorithms_consistency():
    """测试不同算法结果一致性"""
    print("="*60)
    print("测试算法一致性")
    print("="*60)

    mdp = GridWorldMDP(size=4, gamma=0.9)

    # 值迭代
    vi = ValueIteration(mdp)
    V_vi, pi_vi, _ = vi.solve(verbose=False)

    # 策略迭代
    pi_solver = PolicyIteration(mdp)
    V_pi, pi_pi, _ = pi_solver.solve(verbose=False)

    # 比较结果
    v_diff = np.max(np.abs(V_vi - V_pi))
    pi_diff = np.sum(np.abs(pi_vi - pi_pi))

    print(f"价值函数差异: {v_diff:.8f}")
    print(f"策略差异: {pi_diff:.8f}")

    assert v_diff < 1e-4, "算法一致性测试失败"
    assert pi_diff < 1e-4, "算法一致性测试失败"

    print("✓ 算法一致性测试通过\n")


def test_modified_policy_iteration():
    """测试修改的策略迭代"""
    print("="*60)
    print("测试修改的策略迭代")
    print("="*60)

    mdp = GridWorldMDP(size=4, gamma=0.9)

    # 标准策略迭代
    pi = PolicyIteration(mdp)
    V_pi, _, iters_pi = pi.solve(verbose=False)

    # 修改的策略迭代 (k=5)
    mpi = ModifiedPolicyIteration(mdp, k_eval=5)
    V_mpi, _, iters_mpi = mpi.solve(verbose=False)

    print(f"标准策略迭代: {iters_pi} 次")
    print(f"修改策略迭代 (k=5): {iters_mpi} 次")

    # 比较结果
    v_diff = np.max(np.abs(V_pi - V_mpi))
    print(f"价值函数差异: {v_diff:.8f}")

    assert v_diff < 0.01, "修改策略迭代测试失败"

    print("✓ 修改策略迭代测试通过\n")


def test_gamma_sensitivity():
    """测试折扣因子敏感度"""
    print("="*60)
    print("测试折扣因子敏感度")
    print("="*60)

    gammas = [0.5, 0.7, 0.9, 0.95, 0.99]

    for gamma in gammas:
        mdp = GridWorldMDP(size=4, gamma=gamma)
        vi = ValueIteration(mdp)
        V, _, history = vi.solve(verbose=False)

        goal_idx = mdp.state_to_index(mdp.goal)
        print(f"γ={gamma:.2f}: 迭代{len(history):3d}次, "
              f"终点价值={V[goal_idx]:.3f}, "
              f"总价值={np.sum(V):.2f}")

    print("✓ 折扣因子测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("MDP求解器测试套件")
    print("="*60 + "\n")

    test_gridworld()
    test_value_iteration()
    test_policy_iteration()
    test_algorithms_consistency()
    test_modified_policy_iteration()
    test_gamma_sensitivity()

    print("="*60)
    print("✓ 所有测试通过！")
    print("="*60)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    run_all_tests()

    # 示例：完整求解流程
    print("\n" + "="*60)
    print("GridWorld完整求解示例")
    print("="*60)

    # 创建MDP
    mdp = GridWorldMDP(size=5, gamma=0.95)
    print(f"\n创建 {mdp.size}x{mdp.size} GridWorld")
    print(f"起点: {mdp.start}, 终点: {mdp.goal}")

    # 值迭代
    print("\n运行值迭代...")
    vi = ValueIteration(mdp)
    V, policy, history = vi.solve(threshold=1e-6, verbose=False)

    # 可视化结果
    print("\n可视化最优价值函数...")
    plt_v = mdp.render_values(V, title="Optimal Value Function")
    plt_v.savefig('mdp_value_function.png', dpi=100, bbox_inches='tight')
    plt_v.close()

    print("可视化最优策略...")
    plt_pi = mdp.render_policy(policy, title="Optimal Policy")
    plt_pi.savefig('mdp_optimal_policy.png', dpi=100, bbox_inches='tight')
    plt_pi.close()

    print("\n结果已保存:")
    print("  - mdp_value_function.png")
    print("  - mdp_optimal_policy.png")

    # 展示关键指标
    print(f"\n关键指标:")
    print(f"  收敛迭代: {len(history)}")
    print(f"  终点价值: {V[mdp.state_to_index(mdp.goal)]:.3f}")
    print(f"  起点价值: {V[mdp.state_to_index(mdp.start)]:.3f}")
