# 强化学习数学速查表

## 1. 基础概念

### RL问题设定
```
Agent与Environment交互:

状态 (State): sₜ ∈ S
动作 (Action): aₜ ∈ A
奖励 (Reward): rₜ ∈ R
策略 (Policy): π(a|s)
转移概率: P(s'|s,a)

目标: 最大化累积奖励
```

### 回报 (Return)
```
# 有限视界
Gₜ = rₜ + rₜ₊₁ + ... + r_T

# 无限视界（带折扣）
Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ...
   = Σₖ₌₀^∞ γᵏ rₜ₊ₖ

γ ∈ [0, 1]: 折扣因子
```

**折扣因子的意义**:
- `γ = 0`: 只关心即时奖励（短视）
- `γ → 1`: 重视长期奖励
- 数学保证收敛（有界）

---

## 2. 马尔可夫决策过程 (MDP)

### MDP定义
```
MDP = (S, A, P, R, γ)

S: 状态空间
A: 动作空间
P: 转移概率 P(s'|s,a)
R: 奖励函数 r(s,a) 或 r(s,a,s')
γ: 折扣因子
```

### 马尔可夫性质
```
P(sₜ₊₁ | s₀, a₀, s₁, a₁, ..., sₜ, aₜ) = P(sₜ₊₁ | sₜ, aₜ)

当前状态包含所有必要信息
```

---

## 3. 值函数 (Value Functions)

### 状态值函数 (State-Value Function)
```
V^π(s) = E_π[Gₜ | sₜ = s]
       = E_π[Σₖ₌₀^∞ γᵏ rₜ₊ₖ | sₜ = s]
```

**含义**: 从状态s开始，遵循策略π的期望回报

### 动作值函数 (Action-Value Function)
```
Q^π(s, a) = E_π[Gₜ | sₜ = s, aₜ = a]
          = E_π[Σₖ₌₀^∞ γᵏ rₜ₊ₖ | sₜ = s, aₜ = a]
```

**含义**: 从状态s采取动作a后，遵循策略π的期望回报

### 关系
```
V^π(s) = Σₐ π(a|s) Q^π(s, a)

Q^π(s, a) = r(s, a) + γ Σₛ' P(s'|s,a) V^π(s')
```

---

## 4. 贝尔曼方程 (Bellman Equations)

### 贝尔曼期望方程
```
# 状态值函数
V^π(s) = Σₐ π(a|s) [r(s,a) + γ Σₛ' P(s'|s,a) V^π(s')]

# 动作值函数
Q^π(s,a) = r(s,a) + γ Σₛ' P(s'|s,a) Σₐ' π(a'|s') Q^π(s',a')
```

**直觉**: 当前值 = 即时奖励 + 折扣后续值

### 贝尔曼最优方程
```
# 最优状态值函数
V*(s) = max_a [r(s,a) + γ Σₛ' P(s'|s,a) V*(s')]

# 最优动作值函数
Q*(s,a) = r(s,a) + γ Σₛ' P(s'|s,a) max_a' Q*(s',a')
```

**关系**:
```
V*(s) = max_a Q*(s,a)

最优策略: π*(s) = argmax_a Q*(s,a)
```

---

## 5. 动态规划算法

### 策略迭代 (Policy Iteration)

#### 1. 策略评估 (Policy Evaluation)
```
重复直到收敛:
  V(s) ← Σₐ π(a|s) [r(s,a) + γ Σₛ' P(s'|s,a) V(s')]
```

#### 2. 策略改进 (Policy Improvement)
```
π'(s) = argmax_a [r(s,a) + γ Σₛ' P(s'|s,a) V(s')]
```

#### 3. 迭代
```
重复 评估 → 改进，直到策略不变
```

### 值迭代 (Value Iteration)
```
重复直到收敛:
  V(s) ← max_a [r(s,a) + γ Σₛ' P(s'|s,a) V(s')]

提取策略:
π(s) = argmax_a Q(s,a)
```

**优势**: 将评估和改进合并，收敛更快

---

## 6. 时序差分学习 (TD Learning)

### TD(0)
```
V(sₜ) ← V(sₜ) + α [rₜ + γV(sₜ₊₁) - V(sₜ)]
                    └─────┬──────┘
                      TD误差

α: 学习率
```

**对比蒙特卡洛**:
```
MC:  V(sₜ) ← V(sₜ) + α [Gₜ - V(sₜ)]  (用完整回报)
TD:  V(sₜ) ← V(sₜ) + α [rₜ + γV(sₜ₊₁) - V(sₜ)]  (自举)
```

**TD优势**:
- 在线学习（不需等到episode结束）
- 低方差（但有偏差）

---

## 7. Q-Learning

### Q-Learning更新
```
Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α [rₜ + γ max_a Q(sₜ₊₁, a) - Q(sₜ, aₜ)]
                              └────────┬────────┘
                                   TD目标

关键: max_a → 离策略 (off-policy)
```

### 算法流程
```python
初始化 Q(s,a) 任意值
for episode in episodes:
    s = 初始状态
    while not done:
        a = ε-greedy(Q, s)  # ε-贪心选择动作
        s', r = env.step(a)

        # Q-Learning更新
        Q[s,a] += α * (r + γ * max(Q[s']) - Q[s,a])

        s = s'
```

**特点**:
- **离策略**: 学习最优策略，同时探索（用ε-greedy）
- 收敛到Q*（在满足条件下）

---

## 8. SARSA

### SARSA更新
```
Q(sₜ, aₜ) ← Q(sₜ, aₜ) + α [rₜ + γ Q(sₜ₊₁, aₜ₊₁) - Q(sₜ, aₜ)]

关键: 用实际采取的aₜ₊₁ → 在策略 (on-policy)
```

### Q-Learning vs SARSA
```
Q-Learning:  Q(s,a) ← ... + α [r + γ max_a' Q(s',a') - Q(s,a)]
                                    ↑学习最优

SARSA:       Q(s,a) ← ... + α [r + γ Q(s',a') - Q(s,a)]
                                    ↑学习当前策略

SARSA更保守（考虑探索风险）
```

---

## 9. 策略梯度 (Policy Gradient)

### REINFORCE算法
```
目标: 最大化 J(θ) = E_π[Gₜ]

梯度:
∇_θ J(θ) = E_π[Gₜ ∇_θ log π_θ(aₜ|sₜ)]

更新:
θ ← θ + α Gₜ ∇_θ log π_θ(aₜ|sₜ)
```

**直觉**: 增加好结果的概率，减少坏结果的概率

```python
for episode in episodes:
    states, actions, rewards = [], [], []

    # 采样轨迹
    s = env.reset()
    while not done:
        a = π_θ(s).sample()
        s', r, done = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s'

    # 计算回报
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + γ * G

        # 策略梯度更新
        loss = -G * log π_θ(actions[t] | states[t])
        loss.backward()
        optimizer.step()
```

### 带基线的策略梯度
```
∇_θ J(θ) = E_π[(Gₜ - b(sₜ)) ∇_θ log π_θ(aₜ|sₜ)]

b(sₜ): 基线（通常用V(sₜ)）
```

**好处**: 减少方差，加速学习

---

## 10. Actor-Critic

### 架构
```
Actor: π_θ(a|s)  (策略网络)
Critic: V_w(s)   (价值网络)

优势函数:
A(s,a) = Q(s,a) - V(s)
       ≈ r + γV(s') - V(s)  (TD误差)
```

### 更新规则
```
# Critic更新（TD学习）
δ = r + γV_w(s') - V_w(s)
w ← w + α_w δ ∇_w V_w(s)

# Actor更新（策略梯度）
θ ← θ + α_θ δ ∇_θ log π_θ(a|s)
```

**优势**:
- 比REINFORCE方差小（用V做基线）
- 比Q-Learning稳定（策略平滑更新）

---

## 11. 深度强化学习

### DQN (Deep Q-Network)

#### 核心技巧

**1. 经验回放 (Experience Replay)**
```python
# 存储 (s, a, r, s') 到回放缓冲区
replay_buffer.store(s, a, r, s')

# 随机采样batch训练
batch = replay_buffer.sample(batch_size)
```

**好处**: 打破数据相关性，提高样本效率

**2. 目标网络 (Target Network)**
```
y = r + γ max_a' Q_target(s', a')

Q_target: 周期性从Q复制，保持固定
```

**好处**: 稳定训练（避免追逐移动目标）

#### DQN损失
```
L(θ) = E[(y - Q_θ(s,a))²]

其中:
y = r + γ max_a' Q_θ⁻(s', a')  (目标网络)
```

```python
# PyTorch实现
def dqn_loss(batch, q_network, target_network, γ):
    s, a, r, s_next, done = batch

    # 当前Q值
    q_values = q_network(s)
    q_value = q_values.gather(1, a)

    # 目标Q值
    with torch.no_grad():
        next_q_values = target_network(s_next)
        max_next_q = next_q_values.max(1)[0]
        target = r + γ * max_next_q * (1 - done)

    # MSE损失
    loss = F.mse_loss(q_value, target)
    return loss
```

### PPO (Proximal Policy Optimization)

#### 目标函数
```
L^CLIP(θ) = E[min(
    r_t(θ) Â_t,
    clip(r_t(θ), 1-ε, 1+ε) Â_t
)]

r_t(θ) = π_θ(a|s) / π_old(a|s)  (重要性比率)
Â_t: 优势估计
ε: 裁剪范围（如0.2）
```

**直觉**: 限制策略更新步长，防止性能崩溃

```python
def ppo_loss(states, actions, advantages, old_log_probs, π_θ):
    # 新策略的log概率
    new_log_probs = π_θ.log_prob(actions)

    # 重要性比率
    ratio = (new_log_probs - old_log_probs).exp()

    # 裁剪目标
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages

    # PPO损失（取最小）
    loss = -torch.min(surr1, surr2).mean()

    return loss
```

---

## 12. 探索策略

### ε-greedy
```
π(s) = {
    argmax_a Q(s,a)  概率 1-ε  (利用)
    random action    概率 ε    (探索)
}

常用: ε从1衰减到0.01
```

### Softmax (Boltzmann)
```
π(a|s) = exp(Q(s,a)/τ) / Σₐ' exp(Q(s,a')/τ)

τ: 温度参数
  τ→0: 接近贪心
  τ→∞: 接近均匀
```

### Upper Confidence Bound (UCB)
```
a* = argmax_a [Q(s,a) + c√(ln t / N(s,a))]
                ↑估计值    ↑探索奖励

N(s,a): 访问次数
c: 探索系数
```

---

## 13. Python实现示例

### 简单Q-Learning（表格）
```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, lr=0.1, γ=0.99, ε=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.γ = γ
        self.ε = ε

    def choose_action(self, state):
        if np.random.rand() < self.ε:
            return np.random.randint(self.Q.shape[1])
        else:
            return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next, done):
        target = r + (1 - done) * self.γ * np.max(self.Q[s_next])
        self.Q[s, a] += self.lr * (target - self.Q[s, a])
```

### REINFORCE
```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, γ=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.γ = γ

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[action]

    def update(self, rewards, log_probs):
        # 计算折扣回报
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.γ * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 策略梯度
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

---

## 14. 关键公式对比

| 算法 | 更新公式 | 类型 |
|-----|---------|------|
| **Q-Learning** | `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) - Q(s,a)]` | Off-policy, 值迭代 |
| **SARSA** | `Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]` | On-policy, TD |
| **REINFORCE** | `θ ← θ + α G ∇log π(a\|s)` | On-policy, 策略梯度 |
| **Actor-Critic** | `θ ← θ + α δ ∇log π(a\|s)` | On-policy, 混合 |
| **DQN** | 带经验回放和目标网络的Q-Learning | Off-policy, 深度Q |
| **PPO** | 带裁剪的策略梯度 | On-policy, 近端策略 |

---

## 15. 记忆口诀

**贝尔曼方程**:
```
现在的价值 = 即时奖励 + 折扣未来
V(s) = r + γ V(s')
```

**Q-Learning vs SARSA**:
```
Q-Learning: 学最优（max）→ Off-policy
SARSA: 学当前（实际a'）→ On-policy
```

**Actor-Critic**:
```
Actor说"我做什么" (策略)
Critic说"你做得如何" (评价)
```

---

## 16. 常见陷阱

❌ **忘记探索**
```python
# 错误：总是贪心
a = argmax Q(s,a)

# 正确：ε-greedy
if random() < ε:
    a = random_action()
else:
    a = argmax Q(s,a)
```

❌ **DQN不用目标网络**
```
# 不稳定
target = r + γ max Q_θ(s')

# 稳定
target = r + γ max Q_target(s')
```

❌ **REINFORCE不归一化回报**
```python
# 高方差
loss = -log_prob * G

# 低方差
G_normalized = (G - G.mean()) / G.std()
loss = -log_prob * G_normalized
```

---

**相关**: [概率统计速查表](./probability_statistics_cheatsheet.md) | [优化速查表](./optimization_cheatsheet.md)
