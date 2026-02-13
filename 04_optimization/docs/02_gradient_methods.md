# 梯度下降方法

## 为什么需要各种梯度方法？

深度学习训练面临独特挑战：

### 实际问题
```
问题1：数据太多
├── 无法每次用全部数据计算梯度（太慢）
└── 解决：随机梯度下降（SGD）

问题2：收敛慢/震荡
├── 在沟壑状损失曲面上震荡
└── 解决：Momentum（动量）

问题3：学习率难调
├── 不同参数需要不同学习率
└── 解决：Adam（自适应学习率）

问题4：局部最优/鞍点
├── 高维空间大量鞍点
└── 解决：SGD的噪声 + Momentum
```

---

## 1. 梯度下降基础

### 1.1 批量梯度下降（Batch GD）

**核心思想**：每次用全部数据计算梯度

```
θ_{t+1} = θ_t - η × ∇_θ L(θ_t)
```

其中：
- θ：模型参数
- η：学习率
- ∇_θ L：损失函数对所有训练数据的梯度

**实现**：
```python
import numpy as np

def batch_gradient_descent(X, y, theta, lr=0.01, epochs=100):
    """
    批量梯度下降

    参数：
        X: (n_samples, n_features) 数据
        y: (n_samples,) 标签
        theta: (n_features,) 参数
        lr: 学习率
        epochs: 迭代次数
    """
    n = len(y)
    history = []

    for epoch in range(epochs):
        # 计算梯度（用全部数据）
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/n) * X.T @ errors

        # 更新参数
        theta = theta - lr * gradient

        # 记录损失
        loss = (1/(2*n)) * np.sum(errors**2)
        history.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return theta, history

# 示例：线性回归
np.random.seed(42)
X = np.random.randn(1000, 5)
true_theta = np.array([1.5, -2.0, 0.5, 1.0, -0.5])
y = X @ true_theta + 0.1 * np.random.randn(1000)

theta_init = np.zeros(5)
theta_final, losses = batch_gradient_descent(X, y, theta_init, lr=0.1, epochs=100)

print(f"真实参数: {true_theta}")
print(f"学习参数: {theta_final}")
print(f"误差: {np.linalg.norm(theta_final - true_theta):.4f}")
```

**缺点**：
- 数据量大时慢（需遍历所有数据才更新一次）
- 无法在线学习（数据流场景）
- 可能陷入局部最优（无随机性）

### 1.2 随机梯度下降（SGD）

**核心思想**：每次用一个样本更新参数

```
θ_{t+1} = θ_t - η × ∇_θ L(θ_t; x_i, y_i)
```

**代码实现**：
```python
def stochastic_gradient_descent(X, y, theta, lr=0.01, epochs=100):
    """
    随机梯度下降（每次1个样本）
    """
    n = len(y)
    history = []

    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(n)
        total_loss = 0

        for i in indices:
            # 单个样本
            xi = X[i:i+1]  # (1, n_features)
            yi = y[i]

            # 预测和梯度
            prediction = xi @ theta
            error = prediction - yi
            gradient = xi.T @ error  # (n_features, 1) -> (n_features,)

            # 立即更新
            theta = theta - lr * gradient.flatten()

            # 累积损失
            total_loss += error**2

        avg_loss = total_loss / n
        history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    return theta, history

# 运行SGD
theta_sgd, losses_sgd = stochastic_gradient_descent(
    X, y, theta_init.copy(), lr=0.01, epochs=100
)
```

**优点**：
- 更新频繁，学习快
- 有噪声，帮助跳出局部最优
- 可在线学习

**缺点**：
- 梯度估计噪声大
- Loss曲线震荡

### 1.3 Mini-batch SGD（最常用）

**核心思想**：批量GD和SGD的折中，每次用一小批数据

```
θ_{t+1} = θ_t - η × (1/|B|) Σ_{i∈B} ∇_θ L(θ_t; x_i, y_i)
```

其中 B 是一个batch（如32、64、128个样本）

```python
def minibatch_sgd(X, y, theta, lr=0.01, epochs=100, batch_size=32):
    """
    Mini-batch SGD（实践标准）
    """
    n = len(y)
    history = []

    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        n_batches = n // batch_size

        for i in range(n_batches):
            # 取一个batch
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # 计算batch梯度
            predictions = X_batch @ theta
            errors = predictions - y_batch
            gradient = (1/batch_size) * X_batch.T @ errors

            # 更新
            theta = theta - lr * gradient

            # 累积损失
            batch_loss = (1/(2*batch_size)) * np.sum(errors**2)
            total_loss += batch_loss

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    return theta, history

# 对比三种方法
import matplotlib.pyplot as plt

theta_bgd, losses_bgd = batch_gradient_descent(
    X, y, theta_init.copy(), lr=0.1, epochs=50
)
theta_sgd, losses_sgd = stochastic_gradient_descent(
    X, y, theta_init.copy(), lr=0.01, epochs=50
)
theta_mbgd, losses_mbgd = minibatch_sgd(
    X, y, theta_init.copy(), lr=0.05, epochs=50, batch_size=32
)

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(losses_bgd, label='Batch GD', linewidth=2)
plt.plot(losses_sgd, label='SGD', linewidth=2, alpha=0.7)
plt.plot(losses_mbgd, label='Mini-batch SGD', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('三种梯度下降对比')
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.plot(losses_bgd, label='Batch GD', linewidth=2)
plt.plot(losses_sgd, label='SGD', linewidth=2, alpha=0.7)
plt.plot(losses_mbgd, label='Mini-batch SGD', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.yscale('log')
plt.legend()
plt.title('对数尺度对比')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_comparison.png', dpi=100, bbox_inches='tight')
```

**实践建议**：
```
数据规模 vs Batch Size：
├── 小数据（<2000）：Batch GD
├── 中等数据：32-128
├── 大数据：64-256
└── 超大数据/分布式：512-4096

注意：
- Batch size影响学习率选择
- Batch size = 2^n 更高效（GPU对齐）
- 小batch更多噪声，有正则化效果
```

---

## 2. Momentum（动量）

### 2.1 问题：梯度下降的震荡

在损失曲面的沟壑中，梯度下降会震荡：

```python
import numpy as np
import matplotlib.pyplot as plt

# 一个沟壑状损失函数
def loss_function(x, y):
    """狭长的山谷（不同方向曲率不同）"""
    return 0.5 * x**2 + 10 * y**2

def gradient(x, y):
    """梯度"""
    return np.array([x, 20*y])

# SGD路径
def sgd_trajectory(start, lr=0.1, n_steps=50):
    path = [start]
    pos = np.array(start, dtype=float)

    for _ in range(n_steps):
        grad = gradient(pos[0], pos[1])
        pos = pos - lr * grad
        path.append(pos.copy())

    return np.array(path)

# 创建等高线图
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = loss_function(X, Y)

# 起点
start_point = [1.5, 0.8]

# 不同学习率的SGD
plt.figure(figsize=(15, 5))

plt.subplot(131)
path = sgd_trajectory(start_point, lr=0.04, n_steps=20)
plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4)
plt.plot(0, 0, 'g*', markersize=15, label='最优解')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SGD (lr=0.04)\n震荡前进')
plt.legend()
plt.axis('equal')

plt.subplot(132)
path = sgd_trajectory(start_point, lr=0.01, n_steps=50)
plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4)
plt.plot(0, 0, 'g*', markersize=15, label='最优解')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SGD (lr=0.01)\n收敛慢')
plt.legend()
plt.axis('equal')

# 理想路径（Momentum的效果）
plt.subplot(133)
plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.plot([1.5, 0], [0.8, 0], 'b->', linewidth=2, markersize=10)
plt.plot(0, 0, 'g*', markersize=15, label='最优解')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Momentum\n抑制震荡，直奔目标')
plt.legend()
plt.axis('equal')

plt.tight_layout()
plt.savefig('momentum_intuition.png', dpi=100, bbox_inches='tight')
```

**问题诊断**：
- y方向梯度大（曲率大）→ 震荡
- x方向梯度小（曲率小）→ 前进慢
- SGD对所有方向用同样的学习率 → 效率低

### 2.2 Momentum原理

**类比**：小球从山坡滚下，有惯性

```
物理直觉：
├── 速度累积：v_t = β·v_{t-1} + gradient
├── 惯性作用：即使梯度改变方向，速度不会立即反向
└── 阻尼：β 控制摩擦力（0 < β < 1）
```

**更新规则**：
```
v_t = β × v_{t-1} + (1-β) × ∇L(θ)    # 速度更新
θ_t = θ_{t-1} - η × v_t              # 参数更新
```

常用：β = 0.9（相当于10步的平均）

**实现**：
```python
def sgd_with_momentum(X, y, theta, lr=0.01, momentum=0.9, epochs=100, batch_size=32):
    """
    带Momentum的SGD
    """
    n = len(y)
    velocity = np.zeros_like(theta)
    history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        n_batches = n // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # 计算梯度
            predictions = X_batch @ theta
            errors = predictions - y_batch
            gradient = (1/batch_size) * X_batch.T @ errors

            # Momentum更新
            velocity = momentum * velocity + gradient
            theta = theta - lr * velocity

            batch_loss = (1/(2*batch_size)) * np.sum(errors**2)
            total_loss += batch_loss

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    return theta, history

# momentum实现（简化版）
class MomentumOptimizer:
    """Momentum优化器类"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        """执行一步更新"""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        # v = β·v + grad
        self.velocity = self.momentum * self.velocity + grads
        # θ = θ - η·v
        params -= self.lr * self.velocity

        return params

# 可视化Momentum效果
def momentum_trajectory(start, lr=0.01, momentum=0.9, n_steps=50):
    """Momentum优化路径"""
    path = [start]
    pos = np.array(start, dtype=float)
    velocity = np.array([0.0, 0.0])

    for _ in range(n_steps):
        grad = gradient(pos[0], pos[1])
        velocity = momentum * velocity + grad
        pos = pos - lr * velocity
        path.append(pos.copy())

    return np.array(path)

plt.figure(figsize=(12, 5))

# 对比不同momentum值
momentum_values = [0.0, 0.5, 0.9, 0.99]

plt.subplot(121)
for beta in momentum_values:
    path = momentum_trajectory(start_point, lr=0.01, momentum=beta, n_steps=50)
    plt.plot(path[:, 0], path[:, 1], 'o-', linewidth=2, markersize=3,
             label=f'β={beta}', alpha=0.7)

plt.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3)
plt.plot(0, 0, 'g*', markersize=15)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Momentum参数对比\nβ越大惯性越强')
plt.legend()
plt.axis('equal')

# 损失曲线对比
plt.subplot(122)
for beta in momentum_values:
    path = momentum_trajectory(start_point, lr=0.01, momentum=beta, n_steps=100)
    losses = [loss_function(p[0], p[1]) for p in path]
    plt.plot(losses, linewidth=2, label=f'β={beta}')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('收敛速度对比')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('momentum_comparison.png', dpi=100, bbox_inches='tight')
```

### 2.3 Momentum的优势

```
✓ 加速收敛：在一致方向积累速度
✓ 抑制震荡：惯性帮助穿越沟壑
✓ 跳出局部最优：有动量更容易冲出小坑

最佳实践：
├── β = 0.9（最常用）
├── β = 0.99（需要更多迭代）
└── 学习率通常比纯SGD小一些
```

---

## 3. 自适应学习率方法

### 3.1 思路：为每个参数定制学习率

**观察**：
- 频繁更新的参数：梯度大 → 需要小学习率
- 稀疏更新的参数：梯度小 → 需要大学习率

```
理想情况：
θ_1 学习率 = η / (历史梯度大小)
θ_2 学习率 = η / (历史梯度大小)
```

### 3.2 AdaGrad

**核心思想**：累积历史梯度平方，调整学习率

```
G_t = G_{t-1} + (∇L)²         # 累积梯度平方
θ_t = θ_{t-1} - η / (√G_t + ε) × ∇L
```

```python
def adagrad(X, y, theta, lr=0.01, epochs=100, batch_size=32, epsilon=1e-8):
    """
    AdaGrad优化器
    """
    n = len(y)
    G = np.zeros_like(theta)  # 累积梯度平方和
    history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        n_batches = n // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            predictions = X_batch @ theta
            errors = predictions - y_batch
            gradient = (1/batch_size) * X_batch.T @ errors

            # AdaGrad更新
            G += gradient ** 2
            adjusted_lr = lr / (np.sqrt(G) + epsilon)
            theta = theta - adjusted_lr * gradient

            batch_loss = (1/(2*batch_size)) * np.sum(errors**2)
            total_loss += batch_loss

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

    return theta, history
```

**优点**：自动调整学习率
**缺点**：G单调递增，学习率会越来越小→最终停止学习

### 3.3 RMSprop（改进AdaGrad）

**核心思想**：用指数移动平均代替累积

```
G_t = β × G_{t-1} + (1-β) × (∇L)²    # 指数衰减平均
θ_t = θ_{t-1} - η / (√G_t + ε) × ∇L
```

```python
def rmsprop(X, y, theta, lr=0.01, decay=0.9, epochs=100, batch_size=32, epsilon=1e-8):
    """
    RMSprop优化器
    """
    n = len(y)
    G = np.zeros_like(theta)  # 梯度平方的指数平均
    history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        n_batches = n // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            predictions = X_batch @ theta
            errors = predictions - y_batch
            gradient = (1/batch_size) * X_batch.T @ errors

            # RMSprop更新
            G = decay * G + (1 - decay) * (gradient ** 2)
            adjusted_lr = lr / (np.sqrt(G) + epsilon)
            theta = theta - adjusted_lr * gradient

            batch_loss = (1/(2*batch_size)) * np.sum(errors**2)
            total_loss += batch_loss

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

    return theta, history
```

**decay = 0.9** 是常用值

---

## 4. Adam（最流行）

### 4.1 Adam = Momentum + RMSprop

**Adam** (Adaptive Moment Estimation) 结合了两者的优点：

```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇L      # 一阶矩（梯度均值）
v_t = β₂ × v_{t-1} + (1-β₂) × (∇L)²   # 二阶矩（梯度方差）

m̂_t = m_t / (1 - β₁^t)               # 偏差修正
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} - η × m̂_t / (√v̂_t + ε)
```

**为什么需要偏差修正？**
```
初始 m_0 = 0
m_1 = 0.9 × 0 + 0.1 × grad = 0.1 × grad  # 太小！

修正后：
m̂_1 = m_1 / (1 - 0.9) = grad  # 正确！
```

### 4.2 Adam实现

```python
def adam(X, y, theta, lr=0.001, beta1=0.9, beta2=0.999,
         epochs=100, batch_size=32, epsilon=1e-8):
    """
    Adam优化器
    """
    n = len(y)
    m = np.zeros_like(theta)  # 一阶矩
    v = np.zeros_like(theta)  # 二阶矩
    t = 0                      # 时间步
    history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        n_batches = n // batch_size

        for i in range(n_batches):
            t += 1  # 更新时间步

            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            predictions = X_batch @ theta
            errors = predictions - y_batch
            gradient = (1/batch_size) * X_batch.T @ errors

            # Adam更新
            # 1. 更新一阶矩和二阶矩
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)

            # 2. 偏差修正
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # 3. 更新参数
            theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)

            batch_loss = (1/(2*batch_size)) * np.sum(errors**2)
            total_loss += batch_loss

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    return theta, history

# Adam优化器类（完整实现）
class AdamOptimizer:
    """Adam优化器类（生产级）"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        参数：
            lr: 学习率
            beta1: 一阶矩衰减率
            beta2: 二阶矩衰减率
            epsilon: 数值稳定性
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grads):
        """执行一步更新"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # 更新矩估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 更新参数
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

# 测试Adam
theta_adam, losses_adam = adam(
    X, y, theta_init.copy(), lr=0.01, epochs=100, batch_size=32
)
```

### 4.3 Adam的超参数

```python
# 推荐配置
adam_config = {
    'lr': 0.001,      # 默认学习率（最重要）
    'beta1': 0.9,     # 一阶矩衰减率（通常不变）
    'beta2': 0.999,   # 二阶矩衰减率（通常不变）
    'epsilon': 1e-8   # 数值稳定性（通常不变）
}

# 学习率调优建议
scenarios = {
    '默认（大多数情况）': 0.001,
    '微调预训练模型': 0.0001,
    '训练困难（RNN等）': 0.0001,
    '大Batch训练': 0.001 * (batch_size / 256)  # 线性缩放
}
```

### 4.4 完整对比

```python
import matplotlib.pyplot as plt

# 对比所有优化器
optimizers = {
    'SGD': (minibatch_sgd, {'lr': 0.05}),
    'Momentum': (sgd_with_momentum, {'lr': 0.01, 'momentum': 0.9}),
    'AdaGrad': (adagrad, {'lr': 0.1}),
    'RMSprop': (rmsprop, {'lr': 0.01, 'decay': 0.9}),
    'Adam': (adam, {'lr': 0.01}),
}

results = {}
for name, (optimizer, params) in optimizers.items():
    print(f"\n训练 {name}...")
    theta_opt, losses = optimizer(X, y, theta_init.copy(), epochs=50, batch_size=32, **params)
    results[name] = {
        'theta': theta_opt,
        'losses': losses,
        'final_loss': losses[-1]
    }

# 可视化对比
plt.figure(figsize=(14, 5))

# 损失曲线
plt.subplot(121)
for name, result in results.items():
    plt.plot(result['losses'], linewidth=2, label=name)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('优化器对比：损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# 最终误差对比
plt.subplot(122)
names = list(results.keys())
final_losses = [results[n]['final_loss'] for n in names]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

bars = plt.bar(names, final_losses, color=colors, alpha=0.7)
plt.ylabel('Final Loss')
plt.title('优化器对比：最终损失')
plt.yscale('log')

# 添加数值标签
for bar, loss in zip(bars, final_losses):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{loss:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# 打印结果
print("\n" + "="*50)
print("优化器性能对比")
print("="*50)
for name in sorted(results.keys(), key=lambda x: results[x]['final_loss']):
    print(f"{name:12s}: Final Loss = {results[name]['final_loss']:.6f}")

print(f"\n真实参数:  {true_theta}")
print(f"Adam参数:  {results['Adam']['theta']}")
```

---

## 5. 学习率调度

### 5.1 为什么需要学习率调度？

```
训练初期：大学习率
├── 快速下降
└── 探索参数空间

训练后期：小学习率
├── 精细调整
└── 避免震荡
```

### 5.2 常见策略

```python
class LRScheduler:
    """学习率调度器集合"""

    @staticmethod
    def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
        """
        阶梯衰减：每N个epoch降低一次
        lr_t = lr_0 × drop_rate^(epoch // epochs_drop)
        """
        return initial_lr * (drop_rate ** (epoch // epochs_drop))

    @staticmethod
    def exponential_decay(initial_lr, epoch, decay_rate=0.95):
        """
        指数衰减：平滑下降
        lr_t = lr_0 × decay_rate^epoch
        """
        return initial_lr * (decay_rate ** epoch)

    @staticmethod
    def cosine_annealing(initial_lr, epoch, total_epochs, min_lr=0):
        """
        余弦退火：平滑下降
        lr_t = min_lr + 0.5*(lr_0 - min_lr)*(1 + cos(π*t/T))
        """
        return min_lr + 0.5 * (initial_lr - min_lr) * \
               (1 + np.cos(np.pi * epoch / total_epochs))

    @staticmethod
    def warmup_cosine(initial_lr, epoch, warmup_epochs, total_epochs):
        """
        预热+余弦衰减
        - 前warmup_epochs线性增加
        - 之后余弦衰减
        """
        if epoch < warmup_epochs:
            # 预热阶段
            return initial_lr * epoch / warmup_epochs
        else:
            # 余弦衰减阶段
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))

# 可视化学习率调度
initial_lr = 0.1
total_epochs = 100

epochs = np.arange(total_epochs)
lr_step = [LRScheduler.step_decay(initial_lr, e, drop_rate=0.5, epochs_drop=20) for e in epochs]
lr_exp = [LRScheduler.exponential_decay(initial_lr, e, decay_rate=0.95) for e in epochs]
lr_cos = [LRScheduler.cosine_annealing(initial_lr, e, total_epochs, min_lr=1e-5) for e in epochs]
lr_warmup = [LRScheduler.warmup_cosine(initial_lr, e, warmup_epochs=10, total_epochs=total_epochs)
             for e in epochs]

plt.figure(figsize=(12, 6))

plt.plot(epochs, lr_step, linewidth=2, label='Step Decay (每20 epoch)')
plt.plot(epochs, lr_exp, linewidth=2, label='Exponential (γ=0.95)')
plt.plot(epochs, lr_cos, linewidth=2, label='Cosine Annealing')
plt.plot(epochs, lr_warmup, linewidth=2, label='Warmup + Cosine')

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('学习率调度策略对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_scheduling.png', dpi=100, bbox_inches='tight')
```

### 5.3 集成到训练中

```python
def train_with_scheduler(X, y, theta, optimizer='adam',
                         lr=0.01, epochs=100, batch_size=32,
                         lr_scheduler='cosine'):
    """
    带学习率调度的训练
    """
    n = len(y)
    t = 0

    if optimizer == 'sgd':
        m = np.zeros_like(theta)  # momentum
    elif optimizer == 'adam':
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)

    history = []
    lr_history = []

    for epoch in range(epochs):
        # 计算当前学习率
        if lr_scheduler == 'step':
            current_lr = LRScheduler.step_decay(lr, epoch)
        elif lr_scheduler == 'cosine':
            current_lr = LRScheduler.cosine_annealing(lr, epoch, epochs)
        elif lr_scheduler == 'warmup_cosine':
            current_lr = LRScheduler.warmup_cosine(lr, epoch, 10, epochs)
        else:
            current_lr = lr

        lr_history.append(current_lr)

        # 训练一个epoch
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        n_batches = n // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            predictions = X_batch @ theta
            errors = predictions - y_batch
            gradient = (1/batch_size) * X_batch.T @ errors

            # 使用当前学习率更新
            if optimizer == 'sgd':
                m = 0.9 * m + gradient
                theta = theta - current_lr * m
            elif optimizer == 'adam':
                t += 1
                m = 0.9 * m + 0.1 * gradient
                v = 0.999 * v + 0.001 * (gradient ** 2)
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                theta = theta - current_lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            batch_loss = (1/(2*batch_size)) * np.sum(errors**2)
            total_loss += batch_loss

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")

    return theta, history, lr_history
```

---

## 6. 梯度裁剪

### 6.1 问题：梯度爆炸

在RNN或深层网络中，梯度可能变得非常大：

```python
# 梯度爆炸示例
import numpy as np

def gradient_explosion_example():
    """演示梯度爆炸"""
    # 深度网络中的梯度
    layers = 50
    grad_initial = 1.0

    # 如果每层梯度放大2倍
    grad = grad_initial
    grad_history = [grad]

    for i in range(layers):
        grad = grad * 2.0  # 梯度在每一层放大
        grad_history.append(grad)

    print("梯度爆炸示例：")
    for i, g in enumerate(grad_history):
        print(f"Layer {i}: grad = {g:.2e}")

    print(f"\n最终梯度 = {grad:.2e}")
    print(f"这是 {layers:.0f} 个数量级的爆炸！")

gradient_explosion_example()

# 输出：
# Layer 0: grad = 1.00e+00
# Layer 1: grad = 2.00e+00
# ...
# Layer 50: grad = 1.13e+15
```

### 6.2 梯度裁剪方法

```python
def clip_gradient_by_norm(gradient, max_norm):
    """
    按范数裁剪梯度

    如果 ||gradient|| > max_norm:
        gradient = gradient × (max_norm / ||gradient||)
    """
    grad_norm = np.linalg.norm(gradient)

    if grad_norm > max_norm:
        gradient = gradient * (max_norm / grad_norm)
        print(f"梯度裁剪: {grad_norm:.2f} -> {max_norm:.2f}")

    return gradient

def clip_gradient_by_value(gradient, min_val, max_val):
    """
    按值裁剪梯度（直接限制范围）
    """
    return np.clip(gradient, min_val, max_val)

# 集成到训练中
def train_with_clipping(X, y, theta, lr=0.01, epochs=100,
                        batch_size=32, max_grad_norm=1.0):
    """
    带梯度裁剪的训练
    """
    n = len(y)
    history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        n_batches = n // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            predictions = X_batch @ theta
            errors = predictions - y_batch
            gradient = (1/batch_size) * X_batch.T @ errors

            # 梯度裁剪
            gradient = clip_gradient_by_norm(gradient, max_grad_norm)

            # 更新
            theta = theta - lr * gradient

            batch_loss = (1/(2*batch_size)) * np.sum(errors**2)
            total_loss += batch_loss

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

    return theta, history
```

**常用值**：max_grad_norm = 1.0 或 5.0

---

## 7. 权重衰减（L2正则化）

### 7.1 原理

```
损失函数 = 原损失 + λ × ||θ||²
         = L(θ) + λ × Σ θ_i²

梯度 = ∇L(θ) + 2λθ

更新：θ_t = θ_{t-1} - η(∇L + 2λθ)
              = (1 - 2ηλ)θ_{t-1} - η∇L
```

**效果**：每步都将参数向零收缩一点

### 7.2 实现

```python
def sgd_with_weight_decay(X, y, theta, lr=0.01, wd=0.01,
                          epochs=100, batch_size=32):
    """
    带权重衰减的SGD
    wd: weight decay系数
    """
    n = len(y)
    history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        n_batches = n // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            predictions = X_batch @ theta
            errors = predictions - y_batch
            gradient = (1/batch_size) * X_batch.T @ errors

            # 加上权重衰减项
            gradient = gradient + wd * theta

            # 更新
            theta = theta - lr * gradient

            batch_loss = (1/(2*batch_size)) * np.sum(errors**2)
            total_loss += batch_loss

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

    return theta, history
```

### 7.3 AdamW（Adam with decoupled Weight decay）

标准Adam中，权重衰减与自适应学习率耦合，AdamW解耦它们：

```python
def adam_w(X, y, theta, lr=0.001, beta1=0.9, beta2=0.999,
           wd=0.01, epochs=100, batch_size=32, epsilon=1e-8):
    """
    AdamW优化器（解耦权重衰减）
    """
    n = len(y)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0
    history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        n_batches = n // batch_size

        for i in range(n_batches):
            t += 1

            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            predictions = X_batch @ theta
            errors = predictions - y_batch
            gradient = (1/batch_size) * X_batch.T @ errors

            # Adam更新
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # AdamW: 权重衰减单独应用
            theta = theta - lr * (m_hat / (np.sqrt(v_hat) + epsilon) + wd * theta)

            batch_loss = (1/(2*batch_size)) * np.sum(errors**2)
            total_loss += batch_loss

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

    return theta, history
```

**Adam vs AdamW**：
- Adam：weight decay在自适应学习率之前
- AdamW：weight decay独立应用，防止参数过度收缩

**实践建议**：Transformer模型用AdamW，CV模型可用Adam

---

## 8. 实战：完整训练循环

```python
import numpy as np

class NeuralNetwork:
    """简单神经网络"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        # 初始化参数
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

        # 优化器状态
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        # ... 其他参数的状态省略

    def forward(self, X):
        """前向传播"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, y_pred):
        """反向传播"""
        m = len(y)

        # 输出层梯度
        dz2 = (y_pred - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # 隐藏层梯度
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)  # ReLU梯度
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

def train_neural_network(X, y, model, optimizer='adam',
                        lr=0.001, epochs=100, batch_size=32,
                        wd=0.01, max_grad_norm=None):
    """
    完整的神经网络训练循环
    """
    n = len(y)
    history = []

    # 优化器状态
    if optimizer == 'adam':
        states = {
            'W1': {'m': 0, 'v': 0},
            'b1': {'m': 0, 'v': 0},
            'W2': {'m': 0, 'v': 0},
            'b2': {'m': 0, 'v': 0}
        }
        t = 0

    for epoch in range(epochs):
        # 学习率调度
        if epoch < 10:
            current_lr = lr * epoch / 10  # Warmup
        else:
            # Cosine annealing
            progress = (epoch - 10) / (epochs - 10)
            current_lr = lr * 0.5 * (1 + np.cos(np.pi * progress))

        # 训练
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        n_batches = n // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # 前向传播
            y_pred = model.forward(X_batch)

            # 计算loss
            loss = np.mean((y_pred - y_batch) ** 2)
            total_loss += loss

            # 反向传播
            grads = model.backward(X_batch, y_batch, y_pred)

            # 梯度裁剪
            if max_grad_norm:
                total_norm = 0
                for param_name in ['W1', 'b1', 'W2', 'b2']:
                    total_norm += np.sum(grads[param_name] ** 2)
                total_norm = np.sqrt(total_norm)

                if total_norm > max_grad_norm:
                    scale = max_grad_norm / total_norm
                    for param_name in grads:
                        grads[param_name] *= scale

            # 参数更新（Adam）
            t += 1
            for param_name in ['W1', 'b1', 'W2', 'b2']:
                param = getattr(model, param_name)
                grad = grads[param_name]

                # 加上权重衰减
                if wd > 0:
                    grad = grad + wd * param

                # Adam更新
                states[param_name]['m'] = 0.9 * states[param_name]['m'] + 0.1 * grad
                states[param_name]['v'] = 0.999 * states[param_name]['v'] + 0.001 * (grad ** 2)

                m_hat = states[param_name]['m'] / (1 - 0.9 ** t)
                v_hat = states[param_name]['v'] / (1 - 0.999 ** t)

                update = current_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                setattr(model, param_name, param - update)

        avg_loss = total_loss / n_batches
        history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")

    return history

# 测试完整训练
np.random.seed(42)
X_train = np.random.randn(1000, 10)
y_train = X_train @ np.random.randn(10, 1) + 0.1 * np.random.randn(1000, 1)

model = NeuralNetwork(10, 20, 1)
losses = train_neural_network(
    X_train, y_train, model,
    optimizer='adam', lr=0.01, epochs=100,
    batch_size=32, wd=0.01, max_grad_norm=1.0
)

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(losses, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('完整训练循环：神经网络训练')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('training_loop.png', dpi=100, bbox_inches='tight')
```

---

## 9. 优化器选择指南

### 9.1 决策树

```
开始
├─ 数据量小（<10K）？
│  └─ 是 → SGD with Momentum
│
├─ 计算机视觉任务？
│  ├─ 是 → SGD with Momentum (lr=0.1, momentum=0.9)
│  │       + 学习率调度
│  └─ 否 → 继续
│
├─ NLP/Transformer？
│  └─ 是 → AdamW (lr=1e-4 到 5e-5)
│
├─ RNN/LSTM？
│  └─ 是 → Adam + 梯度裁剪
│
├─ 默认选择？
│  └─ Adam (lr=0.001)
│
└─ 微调预训练模型？
   └─ Adam (lr=1e-5 到 1e-4)
```

### 9.2 超参数配置模板

```python
# 计算机视觉（CNN）
cv_config = {
    'optimizer': 'SGD',
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_scheduler': 'step',
    'lr_decay': 0.1,
    'decay_epochs': [30, 60, 90]
}

# NLP（Transformer）
nlp_config = {
    'optimizer': 'AdamW',
    'lr': 5e-5,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.01,
    'warmup_epochs': 10
}

# 强化学习
rl_config = {
    'optimizer': 'Adam',
    'lr': 3e-4,
    'epsilon': 1e-8,
    'max_grad_norm': 0.5  # 梯度裁剪很重要
}

# 通用默认
default_config = {
    'optimizer': 'Adam',
    'lr': 0.001,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8
}
```

---

## 10. 调试技巧

### 10.1 梯度检查

```python
def gradient_check(model, X, y, epsilon=1e-7):
    """
    数值梯度检查
    """
    params = ['W1', 'b1', 'W2', 'b2']

    for param_name in params:
        param = getattr(model, param_name)
        numerical_grad = np.zeros_like(param)

        # 对每个参数扰动
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_value = param[idx]

            # +epsilon
            param[idx] = old_value + epsilon
            y_pred = model.forward(X)
            loss_plus = np.mean((y_pred - y) ** 2)

            # -epsilon
            param[idx] = old_value - epsilon
            y_pred = model.forward(X)
            loss_minus = np.mean((y_pred - y) ** 2)

            # 恢复
            param[idx] = old_value

            # 数值梯度
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

            it.iternext()

        # 反向传播梯度
        y_pred = model.forward(X)
        grads = model.backward(X, y, y_pred)
        analytical_grad = grads[param_name]

        # 对比
        diff = np.linalg.norm(numerical_grad - analytical_grad)
        norm = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
        relative_error = diff / norm

        print(f"{param_name}: 相对误差 = {relative_error:.2e}")
        if relative_error > 1e-5:
            print(f"  ⚠️  梯度检查失败！")
        else:
            print(f"  ✓ 梯度检查通过")
```

### 10.2 常见问题与解决

```python
# 问题1：Loss不下降
problem_1 = {
    '症状': 'Loss长期不变或变化极小',
    '可能原因': [
        '学习率太小',
        '梯度消失',
        '初始化问题',
        '数据未归一化'
    ],
    '解决方案': [
        '增大学习率10倍',
        '检查梯度（梯度检查）',
        '使用更好的初始化（Xavier, He）',
        '标准化数据（zero mean, unit variance）'
    ]
}

# 问题2：Loss变成NaN
problem_2 = {
    '症状': '训练中Loss突然变成NaN',
    '可能原因': [
        '学习率太大',
        '梯度爆炸',
        '数值溢出（exp等操作）'
    ],
    '解决方案': [
        '降低学习率',
        '梯度裁剪',
        '使用log-sum-exp技巧',
        '添加epsilon提高数值稳定性'
    ]
}

# 问题3：震荡不收敛
problem_3 = {
    '症状': 'Loss忽高忽低，不稳定',
    '可能原因': [
        '学习率太大',
        'batch size太小',
        '数据噪声大'
    ],
    '解决方案': [
        '降低学习率',
        '增大batch size',
        '使用Momentum',
        '学习率预热'
    ]
}

# 问题4：过拟合
problem_4 = {
    '症状': '训练loss低但验证loss高',
    '可能原因': [
        '模型太大',
        '训练数据太少',
        '过度训练'
    ],
    '解决方案': [
        '增加正则化（weight decay）',
        'Dropout',
        '早停（early stopping）',
        '数据增强'
    ]
}

def print_debug_guide():
    """打印调试指南"""
    print("\n" + "="*60)
    print("神经网络调试指南")
    print("="*60)

    for i, problem in enumerate([problem_1, problem_2, problem_3, problem_4], 1):
        print(f"\n问题{i}: {problem['症状']}")
        print("可能原因:")
        for cause in problem['可能原因']:
            print(f"  - {cause}")
        print("解决方案:")
        for solution in problem['解决方案']:
            print(f"  ✓ {solution}")

print_debug_guide()
```

---

## 11. 总结

### 优化方法演进

```
梯度下降家族树：

Batch GD
    ↓
SGD（随机梯度下降）
    ↓
    ├─ Momentum（加惯性）
    │     ↓
    │     └─ Nesterov Momentum（前瞻）
    │
    ├─ 自适应学习率方法
    │     ├─ AdaGrad（累积历史梯度）
    │     ├─ RMSprop（指数衰减）
    │     └─ Adam（Momentum + 自适应）★最常用
    │           ↓
    │           └─ AdamW（解耦权重衰减）
    │
    └─ 高级技巧
          ├─ 学习率调度
          ├─ 梯度裁剪
          └─ 预热（warmup）
```

### 核心公式速查

| 优化器 | 更新规则 | 关键超参数 |
|--------|---------|-----------|
| **SGD** | θ = θ - η∇L | lr=0.01-0.1 |
| **Momentum** | v = βv + ∇L<br>θ = θ - ηv | lr=0.01, β=0.9 |
| **Adam** | m = β₁m + (1-β₁)∇L<br>v = β₂v + (1-β₂)(∇L)²<br>θ = θ - ηm̂/(√v̂+ε) | lr=0.001<br>β₁=0.9, β₂=0.999 |
| **AdamW** | 同Adam，但解耦权重衰减 | lr=0.001, wd=0.01 |

### 实践建议

```
✓ 默认用Adam (lr=0.001)
✓ CV任务尝试SGD+Momentum
✓ Transformer用AdamW
✓ 总是监控训练和验证loss
✓ 使用学习率调度
✓ 梯度裁剪（RNN/大模型）
✓ 数据归一化
✓ 合理初始化

✗ 不要盲目调参
✗ 不要用过大的学习率
✗ 不要忘记正则化
✗ 不要跳过梯度检查（调试时）
```

---

## 下一步

继续学习 [03_lagrange.md](03_lagrange.md)，了解约束优化和KKT条件。
