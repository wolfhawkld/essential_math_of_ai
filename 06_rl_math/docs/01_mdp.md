# 马尔可夫决策过程 (MDP)

## 为什么需要MDP？

强化学习研究**序贯决策**问题，需要一个数学框架来描述：

### 现实问题

```
问题1：下围棋
├── 当前状态：棋盘局面
├── 可选动作：落子位置
├── 环境反馈：对手的应对
└── 最终奖励：胜负

问题2：机器人导航
├── 当前状态：位置、障碍物
├── 可选动作：移动方向
├── 环境反馈：新位置
└── 即时奖励：碰撞惩罚、到达奖励

问题3：推荐系统
├── 当前状态：用户历史行为
├── 可选动作：推荐的商品
├── 环境反馈：用户点击/购买
└── 即时奖励：用户满意度

共同特点：
- 序贯决策（一系列动作）
- 延迟奖励（动作效果可能很久后才显现）
- 不确定性（环境的随机性）
```

**解决方案**：马尔可夫决策过程（MDP）

---

## 1. 马尔可夫性质

### 1.1 定义

**马尔可夫性质**：未来只依赖于当前状态，与过去无关

```
P(S_{t+1} | S_t, S_{t-1}, ..., S_0) = P(S_{t+1} | S_t)
```

### 1.2 为什么重要？

```python
# 没有马尔可夫性质：需要记住所有历史
state_0 → action_0 → state_1 → action_1 → ... → state_t → action_t
                                                    ↑
                                              需要所有历史信息

# 有马尔可夫性质：只需当前状态
state_t → action_t → state_{t+1}
    ↑
当前状态已包含所有必要信息
```

### 1.3 例子

```python
import numpy as np

# 例1：棋类游戏 - 马尔可夫
# 当前棋盘状态完全决定未来走向，不需要知道之前的走法
class ChessState:
    """棋盘状态是马尔可夫的"""
    def __init__(self):
        self.board = np.zeros((8, 8))  # 当前棋盘
        # 不需要存储历史走法

    def get_next_state(self, action):
        """下一个状态只依赖当前状态和动作"""
        return self.board + action

# 例2：股票价格 - 非马尔可夫（需要历史趋势）
# 仅知道当前价格不够，需要知道趋势
class StockPrice:
    """股价状态非马尔可夫"""
    def __init__(self):
        self.current_price = 100
        self.history = [95, 97, 99, 100]  # 需要历史趋势

# 例3：增强状态 - 可以将非马尔可夫转为马尔可夫
class StockStateAugmented:
    """增强状态使其马尔可夫化"""
    def __init__(self):
        self.current_price = 100
        self.trend = +1  # 加入趋势信息
        self.momentum = 0.5  # 加入动量信息
```

---

## 2. MDP五元组

### 2.1 正式定义

**MDP** = (S, A, P, R, γ)

| 符号 | 含义 | 说明 |
|------|------|------|
| **S** | 状态空间 | 所有可能的状态集合 |
| **A** | 动作空间 | 所有可能的动作集合 |
| **P** | 转移概率 | P(s'\|s,a): 在状态s执行动作a后转移到s'的概率 |
| **R** | 奖励函数 | R(s,a,s'): 从s经动作a转移到s'的即时奖励 |
| **γ** | 折扣因子 | γ ∈ [0,1)，平衡即时和未来奖励 |

### 2.2 完整示例：网格世界

```python
import numpy as np
import matplotlib.pyplot as plt

class GridWorldMDP:
    """
    4x4网格世界MDP

    目标：从起点(0,0)到达终点(3,3)
    障碍：(1,1)
    动作：上下左右
    """

    def __init__(self, size=4, gamma=0.9):
        # 1. 状态空间 S
        self.size = size
        self.states = [(i, j) for i in range(size) for j in range(size)]
        self.n_states = len(self.states)

        # 2. 动作空间 A
        self.actions = ['up', 'down', 'left', 'right']
        self.action_vectors = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        self.n_actions = len(self.actions)

        # 3. 特殊状态
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.obstacle = (1, 1)

        # 4. 折扣因子
        self.gamma = gamma

    def state_to_index(self, state):
        """状态转索引"""
        return state[0] * self.size + state[1]

    def is_valid(self, state):
        """检查状态是否有效"""
        i, j = state
        if i < 0 or i >= self.size or j < 0 or j >= self.size:
            return False
        if state == self.obstacle:
            return False
        return True

    def get_transition_prob(self, state, action, next_state):
        """
        转移概率 P(s'|s,a)

        这里假设确定性转移：执行动作后一定到达目标状态
        如果撞墙或障碍，保持原位置
        """
        if state == self.goal:  # 终点吸收态
            return 1.0 if next_state == state else 0.0

        # 计算预期下一个状态
        di, dj = self.action_vectors[action]
        expected_next = (state[0] + di, state[1] + dj)

        # 检查是否有效
        if self.is_valid(expected_next):
            actual_next = expected_next
        else:
            actual_next = state  # 撞墙保持原位

        return 1.0 if next_state == actual_next else 0.0

    def get_reward(self, state, action, next_state):
        """
        奖励函数 R(s,a,s')

        规则：
        - 到达终点：+1
        - 撞墙：-0.5
        - 其他：-0.01（鼓励快速到达）
        """
        if state == self.goal:
            return 0.0  # 终点后无奖励

        if next_state == self.goal:
            return 1.0  # 到达终点
        elif next_state == state:  # 撞墙/障碍
            return -0.5
        else:
            return -0.01  # 时间惩罚

    def reset(self):
        """重置到起点"""
        return self.start

    def step(self, state, action):
        """
        执行一步

        返回：next_state, reward, done
        """
        # 计算下一个状态
        di, dj = self.action_vectors[action]
        expected_next = (state[0] + di, state[1] + dj)

        if self.is_valid(expected_next):
            next_state = expected_next
        else:
            next_state = state

        # 计算奖励
        reward = self.get_reward(state, action, next_state)

        # 是否结束
        done = (next_state == self.goal)

        return next_state, reward, done

    def render(self, values=None, policy=None):
        """可视化网格世界"""
        grid = np.zeros((self.size, self.size))

        # 标记特殊位置
        grid[self.start] = 0.5  # 起点
        grid[self.goal] = 1.0   # 终点
        grid[self.obstacle] = -1  # 障碍

        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='RdYlGn', alpha=0.3)

        # 显示价值（如果提供）
        if values is not None:
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) != self.obstacle:
                        plt.text(j, i, f'{values[i*self.size+j]:.2f}',
                                ha='center', va='center', fontsize=10)

        # 显示策略（如果提供）
        if policy is not None:
            action_arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) != self.obstacle and (i, j) != self.goal:
                        action = policy[i*self.size + j]
                        plt.text(j, i+0.3, action_arrows[action],
                                ha='center', va='center', fontsize=14, color='blue')

        plt.grid(True, color='black', linewidth=2)
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.title('GridWorld MDP\nGreen=Goal, Red=Obstacle, Yellow=Start')
        plt.tight_layout()
        plt.savefig('gridworld_mdp.png', dpi=100, bbox_inches='tight')
        plt.show()

# 创建并可视化
mdp = GridWorldMDP()
print("GridWorld MDP 创建成功")
print(f"状态数: {mdp.n_states}")
print(f"动作数: {mdp.n_actions}")
print(f"折扣因子: {mdp.gamma}")
```

### 2.3 转移概率矩阵

```python
# 构建转移概率矩阵
def build_transition_matrix(mdp):
    """
    构建转移概率矩阵 P[n_states, n_actions, n_states]

    P[s, a, s'] = 转移概率
    """
    P = np.zeros((mdp.n_states, mdp.n_actions, mdp.n_states))

    for s_idx, state in enumerate(mdp.states):
        for a_idx, action in enumerate(mdp.actions):
            for s_next_idx, next_state in enumerate(mdp.states):
                prob = mdp.get_transition_prob(state, action, next_state)
                P[s_idx, a_idx, s_next_idx] = prob

    return P

P = build_transition_matrix(mdp)

# 验证：每个(s,a)的转移概率和为1
for s_idx in range(mdp.n_states):
    for a_idx in range(mdp.n_actions):
        total_prob = np.sum(P[s_idx, a_idx, :])
        assert abs(total_prob - 1.0) < 1e-6, f"概率不归一: s={s_idx}, a={a_idx}"

print("转移概率矩阵构建完成")
print(f"形状: {P.shape}")
print("验证：每行求和 = 1 ✓")

# 查看一个状态的转移示例
s = (0, 0)  # 起点
s_idx = mdp.state_to_index(s)
print(f"\n状态 {s} 的转移概率：")
for a_idx, action in enumerate(mdp.actions):
    print(f"  动作 {action}:")
    for s_next_idx in range(mdp.n_states):
        prob = P[s_idx, a_idx, s_next_idx]
        if prob > 0:
            print(f"    → {mdp.states[s_next_idx]}: P={prob:.2f}")
```

### 2.4 奖励矩阵

```python
def build_reward_matrix(mdp):
    """
    构建奖励矩阵 R[n_states, n_actions]

    R[s, a] = E[R(s,a,s')] = Σ P(s'|s,a) × R(s,a,s')
    """
    R = np.zeros((mdp.n_states, mdp.n_actions))

    for s_idx, state in enumerate(mdp.states):
        for a_idx, action in enumerate(mdp.actions):
            expected_reward = 0.0
            for s_next_idx, next_state in enumerate(mdp.states):
                prob = mdp.get_transition_prob(state, action, next_state)
                reward = mdp.get_reward(state, action, next_state)
                expected_reward += prob * reward

            R[s_idx, a_idx] = expected_reward

    return R

R = build_reward_matrix(mdp)

print("奖励矩阵构建完成")
print(f"形状: {R.shape}")
print(f"\n起点 {mdp.start} 的预期奖励:")
s_idx = mdp.state_to_index(mdp.start)
for a_idx, action in enumerate(mdp.actions):
    print(f"  {action}: R={R[s_idx, a_idx]:.3f}")
```

---

## 3. 策略 (Policy)

### 3.1 定义

**策略** π：状态到动作的映射

**确定性策略**：
```
π: S → A
a = π(s)  # 给定状态，输出确定动作
```

**随机策略**：
```
π: S × A → [0,1]
π(a|s) = P(A_t=a | S_t=s)  # 给定状态，输出动作的概率分布
```

### 3.2 策略表示

```python
class Policy:
    """策略类"""

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        # 随机初始化
        self.pi = np.ones((n_states, n_actions)) / n_actions

    def get_action_prob(self, state_idx, action_idx):
        """获取 π(a|s)"""
        return self.pi[state_idx, action_idx]

    def get_action(self, state_idx):
        """根据策略采样动作"""
        probs = self.pi[state_idx]
        return np.random.choice(self.n_actions, p=probs)

    def make_deterministic(self, action_probs):
        """
        设为确定性策略

        参数：
            action_probs: 列表，每个状态的行动作索引
        """
        self.pi = np.zeros((self.n_states, self.n_actions))
        for s_idx, a_idx in enumerate(action_probs):
            self.pi[s_idx, a_idx] = 1.0

    def make_random(self):
        """设为随机策略"""
        self.pi = np.ones((self.n_states, self.n_actions)) / self.n_actions

    def make_epsilon_greedy(self, q_values, epsilon=0.1):
        """
        ε-greedy策略

        π(a|s) = 1-ε+ε/|A|, if a = argmax Q(s,a)
               = ε/|A|,      otherwise
        """
        self.pi = np.ones((self.n_states, self.n_actions)) * epsilon / self.n_actions

        for s_idx in range(self.n_states):
            best_action = np.argmax(q_values[s_idx])
            self.pi[s_idx, best_action] += (1 - epsilon)

# 创建随机策略示例
policy = Policy(mdp.n_states, mdp.n_actions)

print("随机策略示例：")
print(f"状态0的策略分布: {policy.pi[0]}")
print(f"采样的动作: {policy.get_action(0)}")

# ε-greedy策略
q_values_example = np.random.randn(mdp.n_states, mdp.n_actions)
policy.make_epsilon_greedy(q_values_example, epsilon=0.1)
print(f"\nε-greedy策略状态0: {policy.pi[0]}")
```

### 3.3 策略的性能

```python
def evaluate_policy_episode(mdp, policy, max_steps=100):
    """
    评估策略：运行一个episode

    返回：总回报
    """
    state = mdp.reset()
    total_reward = 0.0
    discount = 1.0

    for step in range(max_steps):
        # 根据策略选择动作
        state_idx = mdp.state_to_index(state)
        action_idx = policy.get_action(state_idx)
        action = mdp.actions[action_idx]

        # 执行动作
        next_state, reward, done = mdp.step(state, action)

        # 累积回报
        total_reward += discount * reward
        discount *= mdp.gamma

        if done:
            break

        state = next_state

    return total_reward

# 评估随机策略
policy.make_random()
returns = [evaluate_policy_episode(mdp, policy) for _ in range(100)]

print(f"随机策略性能：")
print(f"平均回报: {np.mean(returns):.3f}")
print(f"标准差: {np.std(returns):.3f}")
```

---

## 4. 价值函数

### 4.1 回报 (Return)

**回报** G_t：从时刻t开始的累积折扣奖励

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ γ^k R_{t+k+1}
```

```python
def compute_return(rewards, gamma):
    """
    计算回报

    参数：
        rewards: 奖励序列 [R_1, R_2, ..., R_T]
        gamma: 折扣因子

    返回：
        G_0
    """
    G = 0.0
    for t, r in enumerate(rewards):
        G += (gamma ** t) * r
    return G

# 示例
rewards = [0, 0, 0, 1]  # 前3步无奖励，第4步到达终点
gamma = 0.9

G = compute_return(rewards, gamma)
print(f"奖励序列: {rewards}")
print(f"折扣因子: {gamma}")
print(f"回报 G_0: {G:.4f}")

# 折扣因子的影响
gammas = [0.5, 0.9, 0.95, 0.99]
for g in gammas:
    G_g = compute_return(rewards, g)
    print(f"γ={g}: G_0 = {G_g:.4f}")
```

### 4.2 状态价值函数 V^π(s)

**定义**：在策略π下，从状态s开始的期望回报

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[Σ γ^k R_{t+k+1} | S_t = s]
```

**直觉**：V^π(s) = 状态s有多"好"

### 4.3 动作价值函数 Q^π(s,a)

**定义**：在策略π下，从状态s采取动作a后的期望回报

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
```

**直觉**：Q^π(s,a) = 在状态s采取动作a有多"好"

### 4.4 V和Q的关系

```
V^π(s) = Σ π(a|s) Q^π(s,a)

Q^π(s,a) = R(s,a) + γ Σ P(s'|s,a) V^π(s')
                  s'
```

```python
# 可视化价值函数
def visualize_value_function(mdp, v_values):
    """可视化状态价值函数"""
    grid_values = np.zeros((mdp.size, mdp.size))

    for i in range(mdp.size):
        for j in range(mdp.size):
            state = (i, j)
            if state == mdp.obstacle:
                grid_values[i, j] = np.nan
            else:
                idx = mdp.state_to_index(state)
                grid_values[i, j] = v_values[idx]

    plt.figure(figsize=(8, 8))

    # 绘制热图
    im = plt.imshow(grid_values, cmap='RdYlGn', aspect='equal')

    # 添加数值标签
    for i in range(mdp.size):
        for j in range(mdp.size):
            if (i, j) != mdp.obstacle:
                text = plt.text(j, i, f'{grid_values[i, j]:.2f}',
                               ha='center', va='center', fontsize=10, color='black')

    plt.colorbar(im, label='State Value V(s)')
    plt.title('State Value Function')
    plt.grid(True, color='black', linewidth=2)
    plt.xticks(range(mdp.size))
    plt.yticks(range(mdp.size))

    plt.tight_layout()
    plt.savefig('value_function.png', dpi=100, bbox_inches='tight')
    plt.show()

# 示例：随机初始化价值
v_random = np.random.rand(mdp.n_states)
visualize_value_function(mdp, v_random)
```

---

## 5. 为什么需要折扣因子？

### 5.1 数学必要性

**无尽地平线问题**：如果持续奖励，回报可能无限大

```
如果 γ = 1:
G_t = R_{t+1} + R_{t+2} + ... = ∞  （如果奖励持续为正）
```

**折扣因子**：保证收敛

```
G_t = Σ γ^k R_{t+k+1} ≤ Σ γ^k R_max = R_max / (1-γ)
```

### 5.2 直觉：未来不确定性

```python
# 折扣因子对决策的影响

def compare_gamma_scenarios():
    """对比不同γ下的最优决策"""

    # 场景：现在得10分 vs 10步后得100分
    R_now = 10
    R_future = 100
    steps = 10

    gammas = [0.5, 0.9, 0.95, 0.99, 0.999]

    print("场景：现在得10分 vs 10步后得100分")
    print("="*60)

    for gamma in gammas:
        G_future = (gamma ** steps) * R_future

        if G_future > R_now:
            choice = "等待未来奖励"
        else:
            choice = "立即拿小奖励"

        print(f"γ={gamma:.3f}: 未来奖励现值={G_future:.2f}, 选择={choice}")

compare_gamma_scenarios()
```

### 5.3 常见取值

```
γ = 0:   短视，只关心即时奖励
γ = 0.9: 常用，适度关注未来
γ = 0.99: 远视，重视长期回报
γ → 1:   近乎无限视界
```

---

## 6. MDP的性质

### 6.1 吸收态

**吸收态**：一旦到达，永远停留

```python
# 终点是吸收态
def get_transition_prob_absorbing(mdp, state, action, next_state):
    """带吸收态的转移概率"""
    if state == mdp.goal:
        # 吸收态：永远停留
        return 1.0 if next_state == state else 0.0

    # 其他状态正常转移
    di, dj = mdp.action_vectors[action]
    expected_next = (state[0] + di, state[1] + dj)

    if mdp.is_valid(expected_next):
        actual_next = expected_next
    else:
        actual_next = state

    return 1.0 if next_state == actual_next else 0.0
```

### 6.2 有限vs无限视界

```python
# 有限视界：有episode长度限制
class FiniteHorizonMDP:
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.current_step = 0

    def step(self, state, action):
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        # ...

# 无限视界：不限制长度（但用γ保证收敛）
class InfiniteHorizonMDP:
    def __init__(self, gamma=0.99):
        self.gamma = gamma  # 必须有折扣因子

    def step(self, state, action):
        done = False  # 可能永不停止
        # ...
```

---

## 7. MDP扩展

### 7.1 部分可观测MDP (POMDP)

**问题**：智能体看不到真实状态，只有观测

```
真实状态: s (隐藏)
观测:     o (可见)
信念:     b(s) = P(S_t=s | 观测历史)
```

**例子**：
- 扑克牌游戏：看不到对手的牌
- 机器人定位：传感器噪声

### 7.2 连续状态/动作 MDP

```python
# 离散MDP
S = {s1, s2, ..., sn}  # 有限个状态
A = {a1, a2, ..., am}  # 有限个动作

# 连续MDP
S = R^n   # 连续状态空间
A = R^m   # 连续动作空间

# 例如：机器人控制
state = (x, y, θ, ẋ, ẏ, θ̇)  # 位置、速度
action = (F_x, F_y)         # 力
```

---

## 8. 总结

### 核心概念

```
MDP五元组： (S, A, P, R, γ)
├─ S: 状态空间（环境描述）
├─ A: 动作空间（智能体选择）
├─ P: 转移概率（环境动力学）
├─ R: 奖励函数（优化目标）
└─ γ: 折扣因子（时间偏好）

策略 π: S → A
├─ 确定性：a = π(s)
└─ 随机性：π(a|s)

价值函数：
├─ V^π(s): 状态价值
├─ Q^π(s,a): 动作价值
└─ 关系：V^π(s) = Σ π(a|s) Q^π(s,a)
```

### 关键性质

```
✓ 马尔可夫性：未来只依赖当前
✓ 折扣因子：保证收敛，体现时间偏好
✓ 吸收态：终点/终止条件
✓ 价值函数：评估状态和策略质量
```

### 从MDP到强化学习

```
已知MDP (S,A,P,R,γ) → 动态规划
├─ 值迭代
└─ 策略迭代

未知MDP → 强化学习
├─ 已知S,A，未知P,R → 无模型RL
└─ 未知S,A → 深度RL
```

---

## 下一步

继续学习 [02_bellman.md](02_bellman.md)，理解贝尔曼方程如何递推计算价值函数。
