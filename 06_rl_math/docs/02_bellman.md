# 贝尔曼方程

## 为什么需要贝尔曼方程？

计算价值函数面临挑战：

### 核心问题

```
问题：如何计算V^π(s)？

方法1：蒙特卡洛（不实用）
├── 从s开始，运行无数个episode
├── 计算平均回报
└── 问题：样本效率低，方差大

方法2：动态规划（高效）
├── 利用状态之间的递推关系
├── V(s)依赖V(s')
└── 贝尔曼方程提供递推公式！
```

---

## 1. 贝尔曼期望方程

### 1.1 状态价值的贝尔曼方程

**推导**：

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... | S_t = s]
       = E_π[R_{t+1} + γ(R_{t+2} + γR_{t+3} + ...) | S_t = s]
       = E_π[R_{t+1} + γG_{t+1} | S_t = s]
       = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
```

**贝尔曼期望方程**：

```
V^π(s) = Σ π(a|s) Σ P(s'|s,a) [R(s,a,s') + γV^π(s')]
         a           s'

       = Σ π(a|s) Q^π(s,a)
          a
```

**直觉**：当前价值 = 即时奖励的期望 + 未来价值的折扣期望

### 1.2 动作价值的贝尔曼方程

**推导**：

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
         = E_π[R_{t+1} + γG_{t+1} | S_t = s, A_t = a]
         = Σ P(s'|s,a) [R(s,a,s') + γ E_π[G_{t+1} | S_{t+1} = s']]
            s'
         = Σ P(s'|s,a) [R(s,a,s') + γ Σ π(a'|s') Q^π(s',a')]
            s'                      a'
```

**贝尔曼期望方程（Q函数）**：

```
Q^π(s,a) = Σ P(s'|s,a) [R(s,a,s') + γ Σ π(a'|s') Q^π(s',a')]
            s'                      a'
```

### 1.3 代码实现

```python
import numpy as np
import matplotlib.pyplot as plt

class BellmanEquation:
    """贝尔曼方程计算器"""

    def __init__(self, mdp):
        self.mdp = mdp
        self.P = None  # 转移概率矩阵
        self.R = None  # 奖励矩阵

    def build_matrices(self):
        """构建转移概率和奖励矩阵"""
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

    def compute_v_pi(self, policy, threshold=1e-6, max_iter=1000):
        """
        策略评估：计算V^π

        迭代应用贝尔曼期望方程直到收敛

        V(s) = Σ π(a|s) [R(s,a) + γ Σ P(s'|s,a) V(s')]
                a                 s'
        """
        if self.P is None:
            self.build_matrices()

        V = np.zeros(self.mdp.n_states)

        for iteration in range(max_iter):
            V_new = np.zeros_like(V)

            for s in range(self.mdp.n_states):
                # 贝尔曼期望方程
                v_s = 0.0
                for a in range(self.mdp.n_actions):
                    # π(a|s)
                    pi_a_s = policy.pi[s, a]

                    # R(s,a) + γ Σ P(s'|s,a) V(s')
                    q_sa = self.R[s, a] + self.mdp.gamma * np.sum(
                        self.P[s, a, :] * V
                    )

                    v_s += pi_a_s * q_sa

                V_new[s] = v_s

            # 检查收敛
            delta = np.max(np.abs(V_new - V))
            V = V_new

            if delta < threshold:
                print(f"策略评估收敛: {iteration+1} 次迭代, Δ={delta:.8f}")
                break

        return V

    def compute_q_pi(self, V, policy):
        """
        从V^π计算Q^π

        Q^π(s,a) = R(s,a) + γ Σ P(s'|s,a) V^π(s')
                                s'
        """
        Q = np.zeros((self.mdp.n_states, self.mdp.n_actions))

        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                Q[s, a] = self.R[s, a] + self.mdp.gamma * np.sum(
                    self.P[s, a, :] * V
                )

        return Q

# 使用前面的GridWorld MDP
from typing import List, Tuple

class GridWorldMDP:
    """简化版GridWorld（从01_mdp.md复用）"""

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

# 测试贝尔曼方程
mdp = GridWorldMDP()
bellman = BellmanEquation(mdp)
bellman.build_matrices()

# 创建随机策略
class Policy:
    def __init__(self, n_states, n_actions):
        self.pi = np.ones((n_states, n_actions)) / n_actions

policy = Policy(mdp.n_states, mdp.n_actions)

# 策略评估
V = bellman.compute_v_pi(policy)

print("策略评估结果：")
print("="*60)
print("状态价值函数 V^π:")
for i in range(mdp.size):
    row_values = []
    for j in range(mdp.size):
        idx = i * mdp.size + j
        if (i, j) == mdp.obstacle:
            row_values.append("  X  ")
        else:
            row_values.append(f"{V[idx]:5.2f}")
    print(" ".join(row_values))
```

---

## 2. 贝尔曼最优方程

### 2.1 最优价值函数

**最优状态价值**：

```
V*(s) = max V^π(s)
        π
```

**最优动作价值**：

```
Q*(s,a) = max Q^π(s,a)
          π
```

### 2.2 贝尔曼最优方程

**状态价值**：

```
V*(s) = max Q*(s,a)
        a
      = max [R(s,a) + γ Σ P(s'|s,a) V*(s')]
        a                  s'
```

**动作价值**：

```
Q*(s,a) = R(s,a) + γ Σ P(s'|s,a) max Q*(s',a')
                        s'        a'
```

### 2.3 最优策略

**定理**：存在最优策略π*，使得：

```
π*(s) = argmax Q*(s,a)
        a
```

且：

```
V*(s) = V^{π*}(s)
Q*(s,a) = Q^{π*}(s,a)
```

### 2.4 贝尔曼最优方程的迭代求解

```python
def value_iteration(self, threshold=1e-6, max_iter=1000):
    """
    值迭代：直接求解贝尔曼最优方程

    V(s) = max [R(s,a) + γ Σ P(s'|s,a) V(s')]
           a                  s'
    """
    if self.P is None:
        self.build_matrices()

    V = np.zeros(self.mdp.n_states)

    for iteration in range(max_iter):
        V_new = np.zeros_like(V)

        for s in range(self.mdp.n_states):
            # 对每个动作计算Q值
            q_values = []
            for a in range(self.mdp.n_actions):
                q_sa = self.R[s, a] + self.mdp.gamma * np.sum(
                    self.P[s, a, :] * V
                )
                q_values.append(q_sa)

            #贝尔曼最优：取最大值
            V_new[s] = max(q_values)

        # 检查收敛
        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < threshold:
            print(f"值迭代收敛: {iteration+1} 次迭代, Δ={delta:.8f}")
            break

    # 提取最优策略
    policy = self.extract_policy(V)

    return V, policy

def extract_policy(self, V):
    """从V*提取最优策略"""
    policy = np.zeros((self.mdp.n_states, self.mdp.n_actions))

    for s in range(self.mdp.n_states):
        q_values = []
        for a in range(self.mdp.n_actions):
            q_sa = self.R[s, a] + self.mdp.gamma * np.sum(
                self.P[s, a, :] * V
            )
            q_values.append(q_sa)

        # 贪心策略
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0

    return policy

# 添加到BellmanEquation类
BellmanEquation.value_iteration = value_iteration
BellmanEquation.extract_policy = extract_policy

# 运行值迭代
V_star, pi_star = bellman.value_iteration()

print("\n最优价值函数 V*:")
print("="*60)
for i in range(mdp.size):
    row_values = []
    for j in range(mdp.size):
        idx = i * mdp.size + j
        if (i, j) == mdp.obstacle:
            row_values.append("  X  ")
        else:
            row_values.append(f"{V_star[idx]:5.2f}")
    print(" ".join(row_values))

print("\n最优策略 π*:")
action_symbols = ['↑', '↓', '←', '→']
print("="*60)
for i in range(mdp.size):
    row_actions = []
    for j in range(mdp.size):
        idx = i * mdp.size + j
        if (i, j) == mdp.obstacle:
            row_actions.append("X")
        elif (i, j) == mdp.goal:
            row_actions.append("G")
        else:
            best_action = np.argmax(pi_star[idx])
            row_actions.append(action_symbols[best_action])
    print(" ".join(row_actions))
```

---

## 3. 贝尔曼备份图

### 3.1 可视化

```
V^π(s) 的贝尔曼备份（备份操作）

           s (当前状态)
          /|\
         / | \  π(a|s)
        /  |  \
       a1  a2  a3 (动作)
       |   |   |
       R+γV(s') (Q值)
        \
         求期望

V(s) ← Σ π(a|s) Q(s,a)  (备份操作)
```

```python
def visualize_bellman_backup(mdp, s_idx, V, policy):
    """可视化单个状态的贝尔曼备份"""
    import matplotlib.patches as mpatches

    state = mdp.states[s_idx]
    print(f"\n状态 {state} 的贝尔曼备份:")
    print("="*60)

    total_v = 0.0
    for a_idx, action in enumerate(mdp.actions):
        pi_a = policy.pi[s_idx, a_idx]

        # 找可能的下一状态
        possible_next = []
        for s_next_idx in range(mdp.n_states):
            if mdp.get_transition_prob(state, action, mdp.states[s_next_idx]) > 0:
                possible_next.append(s_next_idx)

        # 计算Q值
        q_value = bellman.R[s_idx, a_idx]
        print(f"\n动作 {action}: π(a|s)={pi_a:.3f}")
        print(f"  R(s,a)={bellman.R[s_idx, a_idx]:.3f}")

        for s_next_idx in possible_next:
            prob = mdp.get_transition_prob(state, action, mdp.states[s_next_idx])
            v_next = V[s_next_idx]
            contribution = prob * v_next

            print(f"  s'={mdp.states[s_next_idx]}: P={prob:.2f}, "
                  f"V(s')={v_next:.3f}, 贡献={contribution:.3f}")
            q_value += mdp.gamma * contribution

        print(f"  Q(s,a)={q_value:.3f}")
        total_v += pi_a * q_value

    print(f"\nV(s) = {total_v:.3f}")

# 示例：可视化起点的贝尔曼备份
s_idx = mdp.state_to_index(mdp.start)
visualize_bellman_backup(mdp, s_idx, V, policy)
```

---

## 4. 收敛性分析

### 4.1 压缩映射定理

**贝尔曼算子T**：

```
T V(s) = max [R(s,a) + γ Σ P(s'|s,a) V(s')]
         a                  s'
```

**定理**：T是压缩映射

```
||T V₁ - T V₂||_∞ ≤ γ ||V₁ - V₂||_∞
```

**含义**：
- 每次迭代，误差至少缩小γ倍
- 保证唯一不动点V*

### 4.2 收敛速度

```python
def value_iteration_with_history(bellman, threshold=1e-6, max_iter=100):
    """记录每次迭代的V"""
    if bellman.P is None:
        bellman.build_matrices()

    V = np.zeros(bellman.mdp.n_states)
    V_history = [V.copy()]

    for iteration in range(max_iter):
        V_new = np.zeros_like(V)

        for s in range(bellman.mdp.n_states):
            q_values = []
            for a in range(bellman.mdp.n_actions):
                q_sa = bellman.R[s, a] + bellman.mdp.gamma * np.sum(
                    bellman.P[s, a, :] * V
                )
                q_values.append(q_sa)

            V_new[s] = max(q_values)

        delta = np.max(np.abs(V_new - V))
        V = V_new
        V_history.append(V.copy())

        if delta < threshold:
            break

    return V, V_history

V_star, V_history = value_iteration_with_history(bellman)

# 绘制收敛曲线
convergence_errors = [np.max(np.abs(V_history[i] - V_star))
                      for i in range(len(V_history))]

plt.figure(figsize=(10, 6))
plt.plot(convergence_errors, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('||V_k - V*||_∞')
plt.yscale('log')
plt.title('值迭代收敛速度\n理论：误差每步缩小γ倍')
plt.grid(True, alpha=0.3)

# 添加理论收敛线
theoretical_line = [1.0 * (mdp.gamma ** i) for i in range(len(convergence_errors))]
plt.plot(theoretical_line, 'r--', linewidth=2, label=f'理论 γ={mdp.gamma}')
plt.legend()

plt.tight_layout()
plt.savefig('bellman_convergence.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

## 5. 贝尔曼方程的重要性

### 5.1 从理论到算法

```
贝尔曼期望方程          贝尔曼最优方程
    ↓                       ↓
策略评估              值迭代/策略迭代
(policy evaluation)   (value/policy iteration)
    ↓                       ↓
已知π求V^π             直接求V*
```

### 5.2 在强化学习中的地位

```
强化学习算法分类：

1. 基于价值的 (Value-based)
   ├── Q-Learning: TD近似贝尔曼最优方程
   ├── DQN: 神经网络拟合Q函数
   └── 核心：学习Q*(s,a)

2. 基于策略的 (Policy-based)
   ├── REINFORCE: 直接优化π_θ
   ├── PPO: 策略梯度+信任域
   └── 核心：学习π*(a|s)

3. Actor-Critic
   ├── A3C, A2C: 同时学习策略和价值
   ├── SAC: 最大化熵
   └── 核心：Critic用贝尔曼方程，Actor用策略梯度
```

---

## 6. 变体与扩展

### 6.1 贝尔曼方程的变体

```
标准贝尔曼方程：
V(s) = E[R + γV(s')]

TD(0) learning:
V(s) ← V(s) + α [R + γV(s') - V(s)]
         ↑
       TD误差 δ_t = R + γV(s') - V(s)

TD(λ):
考虑多步回报
G_t^(n) = R_{t+1} + γR_{t+2} + ... + γ^n V(s_{t+n})
```

### 6.2 连续状态空间

```
表格方法：V(s)存储在表格中
深度RL：V(s) ≈ V_θ(s) = NeuralNetwork(s; θ)

贝尔曼方程变为：
V_θ(s) ≈ max [R(s,a) + γ E[V_θ(s')]]
        a
```

---

## 7. 总结

### 核心公式

```
贝尔曼期望方程（策略评估）：
V^π(s) = Σ π(a|s) [R(s,a) + γ Σ P(s'|s,a) V^π(s')]
         a                        s'

贝尔曼最优方程（求最优）：
V*(s) = max [R(s,a) + γ Σ P(s'|s,a) V*(s')]
        a                  s'
```

### 算法对比

| 方法 | 方程 | 输入 | 输出 |
|------|------|------|------|
| **策略评估** | 贝尔曼期望 | π, MDP | V^π |
| **值迭代** | 贝尔曼最优 | MDP | V*, π* |
| **策略迭代** | 交替使用 | MDP | V*, π* |

### 关键洞察

```
✓ 递归分解：V(s)依赖V(s')
✓ 压缩映射：保证收敛
✓ 最优性原理：最优策略由局部最优动作组成
✓ 实用性：是动态规划和强化学习的理论基础
```

---

## 下一步

继续学习 [03_value_iteration.md](03_value_iteration.md)，掌握值迭代和策略迭代算法的实现。
