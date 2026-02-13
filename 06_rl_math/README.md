# 06 强化学习数学 (Reinforcement Learning Math)

## 为什么需要这些数学？

强化学习与监督学习有本质不同：

### 监督学习
```
输入 → 模型 → 输出
已知标签，直接优化
```

### 强化学习
```
状态 → 动作 → 奖励 → 新状态 → ...
没有直接标签，通过试错学习
```

需要新的数学工具来描述**序贯决策**问题。

## 核心概念

### 马尔可夫决策过程 (MDP)
强化学习的数学框架，包含：
- **状态** S：环境的描述
- **动作** A：智能体的选择
- **转移概率** P(s'|s,a)：动作导致的状态变化
- **奖励** R(s,a,s')：即时反馈
- **策略** π(a|s)：状态到动作的映射

### 贝尔曼方程
核心递推关系，连接当前价值和未来价值：
```
V(s) = max[R(s,a) + γ Σ P(s'|s,a) V(s')]
       a              s'
```

## 最小必要知识

### ✅ 必须掌握
1. **马尔可夫性质**：未来只依赖当前状态
2. **MDP五元组**：(S, A, P, R, γ)
3. **价值函数**：状态价值V(s)和动作价值Q(s,a)
4. **贝尔曼方程**：递推计算价值
5. **策略评估和改进**：如何找到最优策略

### ⚠️ 了解即可
6. 值迭代和策略迭代算法
7. 部分可观测MDP (POMDP)
8. 时间差分学习（TD）

### ❌ 可以跳过（深度RL会用神经网络代替）
- 线性规划求解MDP
- 复杂的POMDP算法
- 理论收敛性证明

## 学习路径

```
第1步：马尔可夫决策过程 (60分钟)
├── docs/01_mdp.md
└── notebooks/gridworld_mdp.ipynb

第2步：贝尔曼方程 (50分钟)
├── docs/02_bellman.md
└── notebooks/bellman_backup.ipynb  # 待创建

第3步：值迭代和策略迭代 (75分钟)
├── docs/03_value_iteration.md
└── notebooks/policy_iteration.ipynb
```

## 目录内容

### 📄 docs/ - 理论文档
- `01_mdp.md` - 马尔可夫决策过程详解
- `02_bellman.md` - 贝尔曼方程推导
- `03_value_iteration.md` - 经典求解算法

### 💻 notebooks/ - 交互式实践
- `gridworld_mdp.ipynb` - 网格世界MDP示例
- `bellman_backup.ipynb` - 贝尔曼更新可视化
- `policy_iteration.ipynb` - 策略迭代算法实现

### 🔧 code/ - 实用代码
- `mdp_solver.py` - MDP求解器（值迭代、策略迭代）

## 快速测试

完成本模块后，你应该能够：

- [ ] 将现实问题建模为MDP
- [ ] 手动计算简单MDP的价值函数
- [ ] 理解贝尔曼方程的递推关系
- [ ] 实现值迭代和策略迭代算法
- [ ] 解释为什么需要折扣因子γ

## 与强化学习算法的连接

| 数学概念 | RL算法 | 特点 |
|---------|--------|------|
| 值迭代 | Value Iteration | 表格法，小状态空间 |
| 策略迭代 | Policy Iteration | 表格法，交替评估和改进 |
| 贝尔曼方程 | Q-Learning | 值函数学习 |
| 策略梯度 | REINFORCE, PPO | 直接优化策略 |
| Actor-Critic | A3C, SAC | 结合值函数和策略 |
| TD学习 | TD(0), TD(λ) | 时间差分更新 |

## 核心公式速查

### MDP定义
```
MDP = (S, A, P, R, γ)
- S: 状态空间
- A: 动作空间
- P(s'|s,a): 转移概率
- R(s,a,s'): 奖励函数
- γ ∈ [0,1): 折扣因子
```

### 回报 (Return)
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ γᵏ R_{t+k+1}
      k=0
```

### 状态价值函数
```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
```

### 动作价值函数
```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
         = E[R_{t+1} + γQ^π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
```

### 贝尔曼最优方程
```
V*(s) = max Q*(s,a)
        a
      = max [R(s,a) + γ Σ P(s'|s,a) V*(s')]
        a             s'

Q*(s,a) = R(s,a) + γ Σ P(s'|s,a) max Q*(s',a')
                      s'          a'
```

### 最优策略
```
π*(s) = argmax Q*(s,a)
        a
```

## 算法速查

### 值迭代 (Value Iteration)
```
初始化: V(s) = 0 for all s
重复:
    for each s in S:
        V_new(s) = max [R(s,a) + γ Σ P(s'|s,a) V(s')]
                   a             s'
    V = V_new
直到收敛

提取策略:
    π(s) = argmax [R(s,a) + γ Σ P(s'|s,a) V(s')]
           a                 s'
```

### 策略迭代 (Policy Iteration)
```
初始化: π(s) = random action for all s

重复:
    # 策略评估
    重复:
        for each s in S:
            V(s) = R(s,π(s)) + γ Σ P(s'|s,π(s)) V(s')
                                 s'
    直到收敛

    # 策略改进
    for each s in S:
        π_new(s) = argmax [R(s,a) + γ Σ P(s'|s,a) V(s')]
                   a                 s'

    if π_new == π:
        break
    π = π_new
```

## 从表格法到深度强化学习

### 表格法的局限
- 状态/动作空间太大无法存储
- 无法泛化到未见过的状态

### 深度RL的解决方案
```python
# 表格法：V(s) 存在表格中
V_table[s] = value

# 深度RL：V(s) 用神经网络近似
V(s) ≈ V_θ(s) = NeuralNetwork(s; θ)
```

### 主要算法
- **DQN**: 用神经网络近似Q(s,a)
- **Policy Gradient**: 直接参数化策略π_θ(a|s)
- **Actor-Critic**: 同时学习V(s)和π(a|s)

## 实践示例：网格世界

```python
# 4x4网格，目标是到达右下角
# S: start, G: goal, .: empty
# S . . .
# . X . .
# . . . .
# . . . G

states = 16
actions = 4  # up, down, left, right
gamma = 0.9

# 奖励
R(goal) = 1
R(wall) = -1
R(other) = 0

# 用值迭代求解最优路径
```

## 推荐资源

### 书籍
- **《Reinforcement Learning: An Introduction》** (Sutton & Barto) - 第3-4章
  - RL领域的圣经，必读
- 《Deep Reinforcement Learning Hands-On》

### 课程
- [David Silver RL Course](https://www.davidsilver.uk/teaching/) - DeepMind创始人的课程
- [UCL RL Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

### 实践环境
- **OpenAI Gym** - 标准RL环境库
- **Gymnasium** - Gym的新版本

## 调试技巧

### 值函数不收敛
- 检查折扣因子γ是否<1
- 检查贝尔曼更新是否正确
- 增加迭代次数

### 策略震荡
- 降低学习率
- 使用ε-greedy探索
- 添加熵正则化

## 下一步

### 继续深度RL
学完经典MDP后，可以学习：
1. **时间差分学习** (TD-Learning)
2. **Q-Learning** - 不需要环境模型
3. **深度Q网络** (DQN)
4. **策略梯度** (Policy Gradient)

### 实践项目
前往 [07_applications/rl_examples](../07_applications/rl_examples) 查看：
- Q-Learning实现
- 策略梯度实现
- 经典环境求解（CartPole, Atari等）
