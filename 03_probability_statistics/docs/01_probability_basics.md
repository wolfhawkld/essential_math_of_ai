# 概率论基础

## 1. 什么是概率？

### 1.1 频率派 vs 贝叶斯派

**频率派**：概率 = 事件发生的频率
```
P(正面) = 抛硬币1000次，正面出现的比例
```

**贝叶斯派**：概率 = 对不确定性的度量（信念程度）
```
P(明天下雨) = 根据历史数据和当前信息的信念
```

深度学习中两种观点都有应用。

### 1.2 概率的基本性质

```
1. 0 ≤ P(A) ≤ 1
2. P(Ω) = 1  (必然事件)
3. P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
4. P(Ā) = 1 - P(A)  (互补事件)
```

**例子**：
```python
import numpy as np

# 模拟抛硬币
n_trials = 10000
coin_flips = np.random.choice(['H', 'T'], size=n_trials)
prob_heads = np.sum(coin_flips == 'H') / n_trials

print(f"正面概率: {prob_heads:.4f}")  # ≈ 0.5
```

---

## 2. 联合概率、边缘概率、条件概率

### 2.1 联合概率 (Joint Probability)

**定义**：多个事件同时发生的概率

```
P(A, B) 或 P(A ∩ B)
```

**例子**：抽牌
```
P(红色, K) = P(抽到红色K) = 2/52
```

**Python模拟**：
```python
# 模拟掷两个骰子
n_rolls = 10000
die1 = np.random.randint(1, 7, size=n_rolls)
die2 = np.random.randint(1, 7, size=n_rolls)

# P(die1=6, die2=6)
prob_both_6 = np.sum((die1 == 6) & (die2 == 6)) / n_rolls
print(f"P(6,6) = {prob_both_6:.4f}")  # ≈ 1/36 = 0.0278
```

### 2.2 边缘概率 (Marginal Probability)

**定义**：对联合分布求和/积分得到单个变量的概率

**离散**：
```
P(X) = Σ_y P(X, Y=y)
```

**连续**：
```
p(x) = ∫ p(x, y) dy
```

**例子**：从联合分布表计算边缘概率

|   | Y=0 | Y=1 | P(X) |
|---|-----|-----|------|
| X=0 | 0.2 | 0.3 | **0.5** |
| X=1 | 0.1 | 0.4 | **0.5** |
| P(Y) | **0.3** | **0.7** | 1.0 |

```python
# 联合概率表
joint_prob = np.array([[0.2, 0.3],
                       [0.1, 0.4]])

# 边缘概率
p_x = joint_prob.sum(axis=1)  # 对Y求和
p_y = joint_prob.sum(axis=0)  # 对X求和

print(f"P(X): {p_x}")  # [0.5, 0.5]
print(f"P(Y): {p_y}")  # [0.3, 0.7]
```

### 2.3 条件概率 (Conditional Probability)

**定义**：在B发生的条件下，A发生的概率

```
P(A|B) = P(A, B) / P(B)
```

前提：`P(B) > 0`

**直觉**：把样本空间缩小到B

**例子**：
```python
# 掷骰子
# P(点数>4 | 点数是偶数)

n_rolls = 10000
rolls = np.random.randint(1, 7, size=n_rolls)

# 条件：点数是偶数
is_even = (rolls % 2 == 0)

# 在偶数中，大于4的概率
prob = np.sum((rolls > 4) & is_even) / np.sum(is_even)
print(f"P(>4 | 偶数) = {prob:.4f}")  # 只有6满足，所以是1/3
```

**乘法法则**：
```
P(A, B) = P(A|B) × P(B) = P(B|A) × P(A)
```

---

## 3. 独立性

### 3.1 独立事件

**定义**：A的发生不影响B的概率

```
P(A, B) = P(A) × P(B)  (独立)

等价于:
P(A|B) = P(A)
P(B|A) = P(B)
```

**例子**：
```python
# 两次独立抛硬币
n_trials = 10000
flip1 = np.random.choice([0, 1], size=n_trials)
flip2 = np.random.choice([0, 1], size=n_trials)

# 联合概率
p_both_1 = np.sum((flip1 == 1) & (flip2 == 1)) / n_trials

# 边缘概率
p_flip1_1 = np.sum(flip1 == 1) / n_trials
p_flip2_1 = np.sum(flip2 == 1) / n_trials

print(f"P(1,1) = {p_both_1:.4f}")
print(f"P(1) × P(1) = {p_flip1_1 * p_flip2_1:.4f}")
# 两者应该接近
```

### 3.2 条件独立

**定义**：给定C，A和B独立

```
P(A, B | C) = P(A|C) × P(B|C)
```

**深度学习应用**：朴素贝叶斯假设
```
# 假设特征在给定类别下条件独立
P(x₁, x₂, ..., xₙ | y) = ∏ᵢ P(xᵢ | y)
```

---

## 4. 期望和方差

### 4.1 期望 (Expectation)

**定义**：随机变量的平均值

**离散**：
```
E[X] = Σ x × P(X=x)
```

**连续**：
```
E[X] = ∫ x × p(x) dx
```

**性质**：
```
E[aX + b] = a E[X] + b  (线性)
E[X + Y] = E[X] + E[Y]  (总成立)
E[XY] = E[X] E[Y]       (仅当X,Y独立)
```

**例子**：
```python
# 掷骰子的期望
outcomes = np.array([1, 2, 3, 4, 5, 6])
probs = np.array([1/6] * 6)

expected_value = np.sum(outcomes * probs)
print(f"期望: {expected_value}")  # 3.5

# 通过采样验证
samples = np.random.randint(1, 7, size=100000)
print(f"样本均值: {samples.mean():.4f}")  # ≈ 3.5
```

### 4.2 方差 (Variance)

**定义**：偏离期望的平均平方

```
Var[X] = E[(X - E[X])²]
       = E[X²] - (E[X])²
```

**标准差**：
```
σ = √Var[X]
```

**性质**：
```
Var[aX + b] = a² Var[X]  (常数b不影响方差)
Var[X + Y] = Var[X] + Var[Y] + 2Cov(X,Y)
```

**例子**：
```python
# 计算方差
samples = np.random.randn(10000)  # 标准正态分布

mean = samples.mean()
variance = ((samples - mean)**2).mean()
std = np.sqrt(variance)

print(f"均值: {mean:.4f}")      # ≈ 0
print(f"方差: {variance:.4f}")  # ≈ 1
print(f"标准差: {std:.4f}")     # ≈ 1

# 或用NumPy内置
print(f"NumPy方差: {samples.var():.4f}")
```

### 4.3 协方差 (Covariance)

**定义**：两个随机变量的联合变化

```
Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
          = E[XY] - E[X]E[Y]
```

**性质**：
```
Cov(X, X) = Var(X)
Cov(X, Y) = Cov(Y, X)  (对称)
Cov(X, Y) = 0 ⟹ X, Y不相关 (但不一定独立)
```

**相关系数**：
```
ρ(X, Y) = Cov(X, Y) / (σ_X σ_Y)

范围: [-1, 1]
ρ = 1: 完全正相关
ρ = 0: 不相关
ρ = -1: 完全负相关
```

**例子**：
```python
# 生成相关的数据
n = 1000
x = np.random.randn(n)
y = 2 * x + np.random.randn(n) * 0.5  # y与x相关

# 计算协方差
cov_matrix = np.cov(x, y)
print(f"协方差矩阵:\n{cov_matrix}")

# 相关系数
corr = np.corrcoef(x, y)
print(f"相关系数:\n{corr}")
```

---

## 5. 深度学习中的应用

### 5.1 数据的统计特性

**标准化 (Standardization)**：
```python
# 将数据转换为均值0，方差1
def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)

# 使用
X = np.random.randn(100, 10) * 3 + 5
X_normalized = standardize(X)

print(f"原始均值: {X.mean(axis=0)[0]:.2f}")
print(f"原始标准差: {X.std(axis=0)[0]:.2f}")
print(f"标准化后均值: {X_normalized.mean(axis=0)[0]:.4f}")  # ≈ 0
print(f"标准化后标准差: {X_normalized.std(axis=0)[0]:.4f}")  # ≈ 1
```

**为什么标准化？**
- 加速收敛
- 避免数值问题
- 让不同特征处于相同尺度

### 5.2 Batch Normalization

**思想**：对每个batch进行标准化

```python
def batch_norm(x, gamma, beta, eps=1e-5):
    """
    x: (N, D) - batch数据
    gamma: (D,) - 缩放参数
    beta: (D,) - 平移参数
    """
    # 计算batch统计量
    mean = x.mean(axis=0)
    var = x.var(axis=0)

    # 标准化
    x_norm = (x - mean) / np.sqrt(var + eps)

    # 缩放和平移
    out = gamma * x_norm + beta

    return out, mean, var

# 使用
x = np.random.randn(32, 128)  # batch_size=32, features=128
gamma = np.ones(128)
beta = np.zeros(128)

x_bn, _, _ = batch_norm(x, gamma, beta)
print(f"BN后均值: {x_bn.mean(axis=0)[0]:.4f}")
print(f"BN后方差: {x_bn.var(axis=0)[0]:.4f}")
```

### 5.3 Dropout的概率解释

**Dropout**：以概率p将神经元置零

```python
def dropout(x, p=0.5, training=True):
    """
    x: 输入
    p: dropout概率
    """
    if not training:
        return x

    # 生成mask（伯努利分布）
    mask = (np.random.rand(*x.shape) > p).astype(float)

    # 缩放（期望保持不变）
    return x * mask / (1 - p)

# 测试
x = np.ones((1000, 100))
x_dropped = dropout(x, p=0.5)

print(f"原始期望: {x.mean():.4f}")
print(f"Dropout后期望: {x_dropped.mean():.4f}")  # 仍然≈1
```

**为什么除以(1-p)？**
- 训练时：期望变为 `E[x] × (1-p)`
- 除以(1-p)：恢复期望 `E[x]`
- 测试时不dropout，期望一致

---

## 6. 随机变量的变换

### 6.1 线性变换

```
Y = aX + b

E[Y] = a E[X] + b
Var[Y] = a² Var[X]
```

**例子**：温度转换
```python
# 摄氏度 → 华氏度: F = 1.8C + 32
celsius = np.random.normal(20, 5, size=1000)  # 均值20，标准差5

fahrenheit = 1.8 * celsius + 32

print(f"摄氏度均值: {celsius.mean():.2f}°C")
print(f"华氏度均值: {fahrenheit.mean():.2f}°F")  # ≈ 68

# 验证公式
expected_f_mean = 1.8 * celsius.mean() + 32
print(f"理论华氏度均值: {expected_f_mean:.2f}°F")
```

### 6.2 非线性变换

**Softmax**：将实数向量转为概率分布

```python
def softmax(x):
    """
    x: (n,) 或 (batch, n)
    """
    exp_x = np.exp(x - np.max(x))  # 数值稳定
    return exp_x / np.sum(exp_x)

# 例子
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print(f"Logits: {logits}")
print(f"概率: {probs}")
print(f"概率和: {probs.sum():.4f}")  # = 1
```

**性质**：
- 输出非负
- 和为1（概率分布）
- 保持相对大小关系

---

## 7. 常见陷阱

### ❌ 陷阱1：混淆独立和不相关

```python
# 不相关但不独立的例子
x = np.random.uniform(-1, 1, 10000)
y = x**2

# 相关系数
corr = np.corrcoef(x, y)[0, 1]
print(f"相关系数: {corr:.4f}")  # ≈ 0（不相关）

# 但明显不独立：y完全由x决定！
```

**记住**：
- 独立 ⟹ 不相关
- 不相关 ⏸ 独立

### ❌ 陷阱2：条件概率的方向

```
P(A|B) ≠ P(B|A)

例如:
P(感冒|发烧) ≠ P(发烧|感冒)
```

### ❌ 陷阱3：期望的乘积

```python
# 错误：E[XY] = E[X] × E[Y]（仅当独立时成立）

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
prob = np.array([0.2, 0.5, 0.3])

# E[XY]
e_xy = np.sum(x * y * prob)

# E[X] × E[Y]
e_x = np.sum(x * prob)
e_y = np.sum(y * prob)
e_x_times_e_y = e_x * e_y

print(f"E[XY] = {e_xy}")
print(f"E[X]×E[Y] = {e_x_times_e_y}")
# 不相等（因为X和Y不独立）
```

---

## 8. 蒙特卡洛方法

**思想**：用采样近似期望

```
E[f(X)] ≈ (1/N) Σᵢ f(xᵢ),  xᵢ ~ P(X)
```

**例子**：估计π
```python
def estimate_pi(n_samples):
    """
    在单位正方形内随机采样，落在1/4圆内的比例 ≈ π/4
    """
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)

    # 距离原点的距离
    inside_circle = (x**2 + y**2) <= 1

    pi_estimate = 4 * inside_circle.mean()
    return pi_estimate

# 不同采样数量
for n in [100, 1000, 10000, 100000]:
    pi_est = estimate_pi(n)
    error = abs(pi_est - np.pi)
    print(f"n={n:6d}: π ≈ {pi_est:.4f}, 误差={error:.4f}")
```

**深度学习应用**：
- 近似复杂积分
- 策略梯度（强化学习）
- 变分推断

---

## 9. 练习

1. **概率计算**：
   - 掷两个骰子，和为7的概率？
   - 抽两张牌，都是A的概率？

2. **期望方差**：
   - 给定分布，手动计算E[X]和Var[X]
   - 用采样验证

3. **实现练习**：
   - 实现协方差矩阵计算
   - 实现数据标准化函数

4. **Monte Carlo**：
   - 用蒙特卡洛估计定积分
   - 比较不同采样数量的精度

---

**下一步**：学习 [02_distributions.md](./02_distributions.md)，了解深度学习中常用的概率分布。
