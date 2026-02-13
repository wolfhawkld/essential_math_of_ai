# 值迭代与策略迭代

## 为什么需要这两种算法？

已知完整MDP (S, A, P, R, γ)时，如何找到最优策略？

### 两种思路

```
思路1：直接求最优价值（值迭代）
├── 不显式维护策略
├── 迭代更新价值函数
└── 最后提取最优策略

思路2：策略评估+改进交替（策略迭代）
├── 显式维护策略
├── 评估当前策略 → 改进策略 → 再评估
└── 循环直到收敛
```

---

## 1. 值迭代 (Value Iteration)

### 1.1 算法思想

**核心**：直接求解贝尔曼最优方程

```
V_{k+1}(s) = max [R(s,a) + γ Σ P(s'|s,a) V_k(s')]
              a                  s'
```

**停止条件**：||V_{k+1} - V_k|| < θ

**提取策略**：

```
π(s) = argmax [R(s,a) + γ Σ P(s'|s,a) V(s')]
       a                  s'
```

### 1.2 完整算法

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class ValueIteration:
    """值迭代算法"""

    def __init__(self, mdp):
        self.mdp = mdp
        self.P = None
        self.R = None
        self.build_matrices()

    def build_matrices(self):
        """构建转移和奖励矩阵"""
        n_s = self.mdp.n_states
        n_a = self.mdp.n_actions

        self.P = np.zeros((n_s, n_a, n_s))
        self.R = np.zeros((n_s, n_a))

        for s_idx, state in enumerate(self.mdp.states):
            for a_idx, action in enumerate(self.mdp.actions):
                for s_next_idx, next_state in enumerate(self.mdp.states):
                    prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)

                    self.P[s_idx, a_idx, s_next_idx] = prob
                    self.R[s_idx, a_idx] += prob * reward

    def solve(self,
              threshold: float = 1e-6,
              max_iter: int = 1000,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        值迭代求解

        返回：V_star, pi_star, convergence_history
        """
        V = np.zeros(self.mdp.n_states)
        history = []

        for iteration in range(max_iter):
            V_new = np.zeros_like(V)

            for s in range(self.mdp.n_states):
                # 对每个动作计算Q值
                q_values = np.zeros(self.mdp.n_actions)
                for a in range(self.mdp.n_actions):
                    q_values[a] = self.R[s, a] + self.mdp.gamma * np.sum(
                        self.P[s, a, :] * V
                    )

                # 贝尔曼最优更新
                V_new[s] = np.max(q_values)

            # 记录收敛
            delta = np.max(np.abs(V_new - V))
            history.append(delta)

            V = V_new

            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: Δ = {delta:.8f}")

            if delta < threshold:
                if verbose:
                    print(f"✓ 收敛于 Iter {iteration+1}, Δ = {delta:.8f}")
                break

        # 提取最优策略
        policy = self.extract_policy(V)

        return V, policy, history

    def extract_policy(self, V: np.ndarray) -> np.ndarray:
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

# 使用GridWorld MDP测试
class GridWorldMDP:
    """GridWorld MDP"""

    def __init__(self, size=4, gamma=0.9):
        self.size = size
        self.states = [(i, j) for i in range(size) for j in range(size)]
        self.n_states = len(self.states)
        self.actions = ['up', 'down', 'left', 'right']
        self.action_vectors = {
            'up': (-1, 0), 'down': (1, 0),
            'left': (0, -1), 'right': (0, 1)
        }
        self.n_actions = len(self.actions)
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.obstacle = (1, 1)
        self.gamma = gamma

    def state_to_index(self, state):
        return state[0] * self.size + state[1]

    def is_valid(self, state):
        i, j = state
        return (0 <= i < self.size and 0 <= j < self.size and
                state != self.obstacle)

    def get_transition_prob(self, state, action, next_state):
        if state == self.goal:
            return 1.0 if next_state == state else 0.0

        di, dj = self.action_vectors[action]
        expected_next = (state[0] + di, state[1] + dj)
        actual_next = expected_next if self.is_valid(expected_next) else state

        return 1.0 if next_state == actual_next else 0.0

    def get_reward(self, state, action, next_state):
        if state == self.goal:
            return 0.0
        if next_state == self.goal:
            return 1.0
        elif next_state == state:
            return -0.5
        else:
            return -0.01

    def render_values(self, V, title="Value Function"):
        """可视化价值函数"""
        grid = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                idx = i * self.size + j
                if (i, j) == self.obstacle:
                    grid[i, j] = np.nan
                else:
                    grid[i, j] = V[idx]

        plt.figure(figsize=(6, 6))
        im = plt.imshow(grid, cmap='RdYlGn', aspect='equal')

        for i in range(self.size):
            for j in range(self.size):
                if (i, j) != self.obstacle:
                    plt.text(j, i, f'{grid[i, j]:.2f}',
                            ha='center', va='center', fontsize=10)

        plt.colorbar(im, label='Value')
        plt.title(title)
        plt.grid(True, color='black', linewidth=2)
        plt.tight_layout()
        return plt

    def render_policy(self, policy, title="Policy"):
        """可视化策略"""
        action_symbols = ['↑', '↓', '←', '→']
        grid = np.zeros((self.size, self.size))

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='Greys', alpha=0.1)

        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.obstacle:
                    plt.text(j, i, 'X', ha='center', va='center',
                            fontsize=20, color='red', fontweight='bold')
                elif (i, j) == self.goal:
                    plt.text(j, i, 'G', ha='center', va='center',
                            fontsize=20, color='green', fontweight='bold')
                else:
                    idx = i * self.size + j
                    best_action = np.argmax(policy[idx])
                    plt.text(j, i, action_symbols[best_action],
                            ha='center', va='center', fontsize=20, color='blue')

        plt.title(title)
        plt.grid(True, color='black', linewidth=2)
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.tight_layout()
        return plt

# 运行值迭代
mdp = GridWorldMDP(size=4, gamma=0.9)
vi = ValueIteration(mdp)
V_star, pi_star, history = vi.solve()

print("\n最优价值函数 V*:")
mdp.render_values(V_star, "Value Iteration: Optimal Value Function")
plt.savefig('value_iteration_result.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n最优策略 π*:")
mdp.render_policy(pi_star, "Value Iteration: Optimal Policy")
plt.savefig('value_iteration_policy.png', dpi=100, bbox_inches='tight')
plt.show()
```

### 1.3 收敛性分析

```python
def analyze_convergence(history, gamma):
    """分析收敛速度"""
    plt.figure(figsize=(12, 5))

    # 误差曲线
    plt.subplot(121)
    plt.plot(history, linewidth=2, label='Actual Δ')
    plt.xlabel('Iteration')
    plt.ylabel('||V_{k+1} - V_k||_∞')
    plt.yscale('log')
    plt.title('Value Iteration Convergence')
    plt.grid(True, alpha=0.3)

    # 理论收敛线
    theoretical = [gamma ** i for i in range(len(history))]
    plt.plot(theoretical, 'r--', linewidth=2, label=f'Theoretical γ^k')
    plt.legend()

    # 收敛速度分析
    plt.subplot(122)
    ratios = [history[i+1] / history[i] for i in range(len(history)-1)]
    plt.plot(ratios, linewidth=2)
    plt.axhline(y=gamma, color='r', linestyle='--', label=f'γ={gamma}')
    plt.xlabel('Iteration')
    plt.ylabel('Δ_{k+1} / Δ_k')
    plt.title('Convergence Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('value_iteration_convergence.png', dpi=100, bbox_inches='tight')
    plt.show()

    print(f"理论收敛率: γ = {gamma}")
    print(f"实际平均收敛率: {np.mean(ratios):.4f}")

analyze_convergence(history, mdp.gamma)
```

### 1.4 值迭代的特点

**优点**：
- ✅ 概念简单，易于实现
- ✅ 不需要显式维护策略
- ✅ 每次迭代都是贝尔曼最优更新

**缺点**：
- ❌ 收敛可能较慢
- ❌ 最后一步才提取策略

---

## 2. 策略迭代 (Policy Iteration)

### 2.1 算法思想

**两个步骤交替**：

```
1. 策略评估 (Policy Evaluation)
   给定策略π，计算V^π

2. 策略改进 (Policy Improvement)
   根据V^π，改进策略π

循环直到策略不变
```

### 2.2 完整算法

```python
class PolicyIteration:
    """策略迭代算法"""

    def __init__(self, mdp):
        self.mdp = mdp
        self.P = None
        self.R = None
        self.build_matrices()

    def build_matrices(self):
        """构建转移和奖励矩阵"""
        n_s = self.mdp.n_states
        n_a = self.mdp.n_actions

        self.P = np.zeros((n_s, n_a, n_s))
        self.R = np.zeros((n_s, n_a))

        for s_idx, state in enumerate(self.mdp.states):
            for a_idx, action in enumerate(self.mdp.actions):
                for s_next_idx, next_state in enumerate(self.mdp.states):
                    prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)

                    self.P[s_idx, a_idx, s_next_idx] = prob
                    self.R[s_idx, a_idx] += prob * reward

    def policy_evaluation(self,
                         policy: np.ndarray,
                         threshold: float = 1e-6,
                         max_iter: int = 1000) -> np.ndarray:
        """
        策略评估：计算V^π

        V(s) = Σ π(a|s) [R(s,a) + γ Σ P(s'|s,a) V(s')]
               a                  s'
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
        策略改进：根据V生成更优策略

        π'(s) = argmax [R(s,a) + γ Σ P(s'|s,a) V(s')]
                a                  s'
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
              init_policy: np.ndarray = None,
              threshold: float = 1e-6,
              max_iter: int = 100,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        策略迭代求解

        返回：V_star, pi_star, iterations
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
                print(f"Policy change: {policy_change:.6f}")

            if policy_change < 1e-10:
                if verbose:
                    print(f"✓ 策略收敛于 Iteration {iteration + 1}")
                break

            policy = new_policy

        return V, policy, iteration + 1

# 运行策略迭代
pi_solver = PolicyIteration(mdp)
V_pi, pi_pi, n_iters = pi_solver.solve()

print("\n策略迭代结果:")
print(f"收敛迭代次数: {n_iters}")

mdp.render_values(V_pi, f"Policy Iteration: Optimal Value (Converged in {n_iters} iters)")
plt.savefig('policy_iteration_result.png', dpi=100, bbox_inches='tight')
plt.show()

mdp.render_policy(pi_pi, "Policy Iteration: Optimal Policy")
plt.savefig('policy_iteration_policy.png', dpi=100, bbox_inches='tight')
plt.show()

# 验证两种方法结果一致
print("\n验证:")
print(f"V*差异: {np.max(np.abs(V_star - V_pi)):.8f}")
print(f"π*差异: {np.sum(np.abs(pi_star - pi_pi)):.8f}")
```

### 2.3 策略迭代的收敛性

**定理**：策略迭代保证收敛，且每次迭代策略改进或已最优

**证明思路**：

```
1. 策略改进单调性
   V^{π_{k+1}} ≥ V^{π_k}

2. 有限状态空间的有限性
   最多 |A|^{|S|} 种策略

3. 因此有限步收敛
```

### 2.4 策略迭代的优势

```python
# 对比值迭代和策略迭代的迭代次数

print("="*60)
print("值迭代 vs 策略迭代")
print("="*60)
print(f"值迭代迭代次数: {len(history)}")
print(f"策略迭代迭代次数: {n_iters} (每次包含策略评估)")
print()
print("注意事项：")
print("- 值迭代每次迭代简单（只更新V）")
print("- 策略迭代每次迭代复杂（需要完整的策略评估）")
print("- 策略迭代通常总的循环次数少")
```

---

## 3. 两种算法对比

### 3.1 性能对比

```python
def compare_algorithms(mdp, n_runs=5):
    """全面对比两种算法"""

    results = {
        'value_iteration': {'iters': [], 'time': []},
        'policy_iteration': {'iters': [], 'time': []}
    }

    import time

    for _ in range(n_runs):
        # 值迭代
        vi = ValueIteration(mdp)
        start = time.time()
        V_vi, pi_vi, history_vi = vi.solve(verbose=False)
        t_vi = time.time() - start

        results['value_iteration']['iters'].append(len(history_vi))
        results['value_iteration']['time'].append(t_vi)

        # 策略迭代
        pi_solver = PolicyIteration(mdp)
        start = time.time()
        V_pi, pi_pi, n_iters_pi = pi_solver.solve(verbose=False)
        t_pi = time.time() - start

        results['policy_iteration']['iters'].append(n_iters_pi)
        results['policy_iteration']['time'].append(t_pi)

    # 统计
    print("\n" + "="*60)
    print("算法性能对比 (平均 {} 次运行)".format(n_runs))
    print("="*60)
    print("\n值迭代:")
    print(f"  平均迭代次数: {np.mean(results['value_iteration']['iters']):.1f}")
    print(f"  平均时间: {np.mean(results['value_iteration']['time'])*1000:.2f} ms")

    print("\n策略迭代:")
    print(f"  平均外层迭代: {np.mean(results['policy_iteration']['iters']):.1f}")
    print(f"  平均时间: {np.mean(results['policy_iteration']['time'])*1000:.2f} ms")

    return results

results = compare_algorithms(mdp, n_runs=5)
```

### 3.2 适用场景

```
值迭代适合：
✓ 状态空间大
✓ 只需要价值函数，不急着要策略
✓ 每次迭代时间要短

策略迭代适合：
✓ 策略空间相对小
✓ 需要快速找到最优策略
✓ 可以接受每次迭代更复杂
```

---

## 4. 修改策略迭代

### 4.1 思想

策略评估不需要完全收敛，可以用几次迭代代替

```python
class ModifiedPolicyIteration:
    """修改的策略迭代"""

    def __init__(self, mdp, k_eval=10):
        self.mdp = mdp
        self.k_eval = k_eval  # 策略评估迭代次数
        self.P = None
        self.R = None
        self.build_matrices()

    def build_matrices(self):
        """构建矩阵（同上）"""
        n_s = self.mdp.n_states
        n_a = self.mdp.n_actions

        self.P = np.zeros((n_s, n_a, n_s))
        self.R = np.zeros((n_s, n_a))

        for s_idx, state in enumerate(self.mdp.states):
            for a_idx, action in enumerate(self.mdp.actions):
                for s_next_idx, next_state in enumerate(self.mdp.states):
                    prob = self.mdp.get_transition_prob(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)

                    self.P[s_idx, a_idx, s_next_idx] = prob
                    self.R[s_idx, a_idx] += prob * reward

    def truncated_policy_evaluation(self, policy, V_old):
        """截断的策略评估（只迭代k次）"""
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

    def solve(self, max_iter=100, verbose=True):
        """修改的策略迭代求解"""
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

# 对比
mpi = ModifiedPolicyIteration(mdp, k_eval=5)
V_mpi, pi_mpi, n_mpi = mpi.solve()

print("修改策略迭代 (k=5):")
print(f"收敛迭代次数: {n_mpi}")
```

---

## 5. 异步值迭代

### 5.1 思想

不需要每次更新所有状态，可以按任意顺序更新

```python
class AsynchronousValueIteration:
    """异步值迭代"""

    def __init__(self, mdp):
        self.mdp = mdp
        self.P = None
        self.R = None
        self.build_matrices()

    def build_matrices(self):
        """构建矩阵（同上）"""
        # ... 省略，同ValueIteration
        pass

    def solve(self,
              max_iter=10000,
              update_order='random',
              verbose=True):
        """
        异步值迭代

        update_order:
            'random': 随机选择状态更新
            'sweep': 按顺序扫描
            'policy': 根据当前策略选择重要状态
        """
        V = np.zeros(self.mdp.n_states)
        history = []

        for iteration in range(max_iter):
            # 选择要更新的状态
            if update_order == 'random':
                s = np.random.randint(self.mdp.n_states)
            elif update_order == 'sweep':
                s = iteration % self.mdp.n_states
            else:
                s = iteration % self.mdp.n_states

            # 更新单个状态
            q_values = np.zeros(self.mdp.n_actions)
            for a in range(self.mdp.n_actions):
                q_values[a] = self.R[s, a] + self.mdp.gamma * np.sum(
                    self.P[s, a, :] * V
                )

            old_v = V[s]
            V[s] = np.max(q_values)

            # 记录最大变化
            delta = abs(V[s] - old_v)
            history.append(delta)

            if verbose and iteration % 1000 == 0:
                print(f"Iter {iteration}: max Δ = {max(history[-1000:]):.8f}")

            # 检查收敛（检查最近N次更新的最大变化）
            if len(history) > 1000 and max(history[-1000:]) < 1e-6:
                if verbose:
                    print(f"✓ 收敛于 Iter {iteration + 1}")
                break

        policy = self.extract_policy(V)
        return V, policy, history

    def extract_policy(self, V):
        """提取策略（同上）"""
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
```

### 5.2 优势

```
异步值迭代优势：
✓ 可以随时停止，得到部分解
✓ 可以重点更新重要状态
✓ 实现在线学习的基础
```

---

## 6. 实际应用考虑

### 6.1 初值选择

```python
def test_initialization():
    """测试不同初始值的影响"""

    # 不同初始化
    initial_values = {
        'zeros': np.zeros(mdp.n_states),
        'random': np.random.randn(mdp.n_states),
        'optimistic': np.ones(mdp.n_states) * 10,
        'pessimistic': np.ones(mdp.n_states) * -10,
    }

    for name, V_init in initial_values.items():
        vi = ValueIteration(mdp)
        # 从V_init开始
        V = V_init.copy()

        for iteration in range(100):
            V_new = np.zeros_like(V)
            for s in range(mdp.n_states):
                q_values = np.zeros(mdp.n_actions)
                for a in range(mdp.n_actions):
                    q_values[a] = vi.R[s, a] + mdp.gamma * np.sum(vi.P[s, a, :] * V)
                V_new[s] = np.max(q_values)

            delta = np.max(np.abs(V_new - V))
            V = V_new
            if delta < 1e-6:
                print(f"{name:12s}: 收敛于 {iteration+1} 次迭代")
                break

test_initialization()
```

### 6.2 折扣因子影响

```python
def test_gamma_sensitivity():
    """测试折扣因子的影响"""

    gammas = [0.5, 0.7, 0.9, 0.95, 0.99]

    plt.figure(figsize=(12, 5))

    for gamma in gammas:
        mdp_gamma = GridWorldMDP(size=4, gamma=gamma)
        vi = ValueIteration(mdp_gamma)
        V, pi, history = vi.solve(verbose=False)

        plt.subplot(121)
        plt.plot(history, label=f'γ={gamma}', linewidth=2)

        plt.subplot(122)
        plt.plot([np.sum(V)] + [0]*10, 'o', label=f'γ={gamma}')  # 简化示意

    plt.subplot(121)
    plt.xlabel('Iteration')
    plt.ylabel('Δ')
    plt.yscale('log')
    plt.title('收敛速度 vs 折扣因子')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(122)
    plt.xlabel('γ')
    plt.ylabel('Total Value ΣV(s)')
    plt.title('价值范围 vs 折扣因子')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gamma_sensitivity.png', dpi=100, bbox_inches='tight')
    plt.show()

test_gamma_sensitivity()
```

---

## 7. 总结

### 算法对照表

| 特性 | 值迭代 | 策略迭代 | 修改策略迭代 |
|------|--------|----------|--------------|
| **维护对象** | V(s) | π, V^π | π, V^π |
| **每次迭代** | V更新 | 策略评估+改进 | 截断评估+改进 |
| **迭代次数** | 多 | 少 | 中等 |
| **每次复杂度** | O(\|S\|²\|A\|) | 高 | 中等 |
| **适用场景** | 大状态空间 | 小策略空间 | 平衡 |
| **收敛保证** | ✓ | ✓ | ✓ |

### 关键要点

```
✓ 值迭代：直接迭代贝尔曼最优方程
✓ 策略迭代：策略评估和改进交替
✓ 都保证收敛到最优策略
✓ 折扣因子γ决定收敛速度和价值范围
✓ 异步更新提供灵活性
```

### 从表格到深度学习

```
表格方法局限：
├─ 状态空间大时存储困难
└─ 无法泛化到未见状态

深度学习解决方案：
├─ V(s) ≈ NeuralNetwork_θ(s)
├─ Q(s,a) ≈ NeuralNetwork_θ(s,a)
└─ π(a|s) ≈ NeuralNetwork_θ(s) 输出动作概率
```

---

## 下一步

强化学习数学基础已完成。继续学习：
- 07_applications/rl_examples：Q-Learning、Policy Gradient实战
- Sutton & Barto《Reinforcement Learning》第3-4章
- David Silver RL课程
