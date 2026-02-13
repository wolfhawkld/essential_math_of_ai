# 梯度下降及优化算法

## 1. 梯度下降基础

### 1.1 基本思想

**梯度下降**：沿着函数下降最快的方向（负梯度方向）迭代更新参数

```
θₜ₊₁ = θₜ - η ∇L(θₜ)

θ: 参数
η: 学习率
∇L: 损失函数的梯度
```

**直觉**：
- 梯度指向增长最快的方向
- 负梯度指向下降最快的方向
- 沿着负梯度走，损失逐渐减小

### 1.2 几何解释

```python
import numpy as np
import matplotlib.pyplot as plt

# 简单的二次函数
def f(x):
    return x**2

def df(x):
    return 2*x

# 梯度下降
x = 5.0  # 起点
learning_rate = 0.1
trajectory = [x]

for i in range(20):
    grad = df(x)
    x = x - learning_rate * grad
    trajectory.append(x)

# 可视化
x_vals = np.linspace(-6, 6, 100)
plt.plot(x_vals, f(x_vals), label='f(x) = x²')
plt.plot(trajectory, [f(x) for x in trajectory], 'ro-', label='梯度下降路径')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('梯度下降过程')
plt.show()
```

---

## 2. 批量梯度下降 (Batch Gradient Descent)

### 2.1 标准梯度下降

**每次迭代使用全部数据**计算梯度：

```
θ = θ - η × (1/N) Σᵢ ∇L(θ; xᵢ, yᵢ)
```

```python
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    """
    X: (N, D) 特征
    y: (N,) 标签
    """
    N, D = X.shape
    w = np.zeros(D)
    b = 0

    losses = []

    for epoch in range(epochs):
        # 前向传播
        y_pred = X @ w + b

        # 损失（MSE）
        loss = 0.5 * np.mean((y_pred - y)**2)
        losses.append(loss)

        # 梯度
        grad_pred = (y_pred - y) / N
        grad_w = X.T @ grad_pred
        grad_b = np.sum(grad_pred)

        # 更新
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return w, b, losses

# 测试
X = np.random.randn(100, 5)
w_true = np.array([1, 2, 3, 4, 5])
y = X @ w_true + np.random.randn(100) * 0.1

w, b, losses = batch_gradient_descent(X, y)
```

**优点**：
- 收敛稳定
- 利用向量化（快）

**缺点**：
- 大数据集内存占用高
- 每次迭代慢

---

## 3. 随机梯度下降 (SGD)

### 3.1 单样本更新

**每次只用一个样本**计算梯度：

```
θ = θ - η × ∇L(θ; xᵢ, yᵢ)
```

```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    N, D = X.shape
    w = np.zeros(D)
    b = 0

    for epoch in range(epochs):
        # 随机打乱数据
        indices = np.random.permutation(N)

        for i in indices:
            # 单个样本
            xi = X[i]
            yi = y[i]

            # 前向
            y_pred = xi @ w + b

            # 梯度
            grad_pred = y_pred - yi
            grad_w = xi * grad_pred
            grad_b = grad_pred

            # 更新
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

    return w, b
```

**优点**：
- 内存友好
- 每次迭代快
- 噪声有助于跳出局部最优

**缺点**：
- 收敛不稳定（抖动）
- 无法利用向量化加速

---

## 4. Mini-batch SGD

### 4.1 折中方案

**每次使用小批量数据**（如32、64、128）：

```
θ = θ - η × (1/B) Σᵢ∈batch ∇L(θ; xᵢ, yᵢ)
```

```python
def mini_batch_sgd(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    N, D = X.shape
    w = np.zeros(D)
    b = 0

    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # 分批
        for i in range(0, N, batch_size):
            # 取一个batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # 前向
            y_pred = X_batch @ w + b

            # 梯度
            grad_pred = (y_pred - y_batch) / len(y_batch)
            grad_w = X_batch.T @ grad_pred
            grad_b = np.sum(grad_pred)

            # 更新
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

    return w, b
```

**深度学习标配！**
- 平衡了速度和稳定性
- 利用GPU并行计算
- 典型batch size：32、64、128、256

---

## 5. 学习率调整

### 5.1 学习率的重要性

```python
# 可视化不同学习率的效果
def visualize_learning_rates():
    x_init = 5.0
    lrs = [0.01, 0.1, 0.5, 0.9]

    plt.figure(figsize=(12, 8))
    x_vals = np.linspace(-6, 6, 100)

    for lr in lrs:
        x = x_init
        trajectory = [x]

        for _ in range(20):
            x = x - lr * (2 * x)  # f(x) = x²的梯度
            trajectory.append(x)

        plt.plot(trajectory, label=f'lr={lr}')

    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.title('不同学习率的收敛情况')
    plt.show()
```

**现象**：
- **lr太小**：收敛慢
- **lr适中**：快速收敛
- **lr太大**：震荡、发散

### 5.2 学习率衰减

#### Step Decay
```python
def step_decay(initial_lr, epoch, drop_every=10, drop_rate=0.5):
    return initial_lr * (drop_rate ** (epoch // drop_every))

# 使用
for epoch in range(100):
    lr = step_decay(0.1, epoch)
    # ... 训练 ...
```

#### Exponential Decay
```python
def exponential_decay(initial_lr, epoch, decay_rate=0.95):
    return initial_lr * (decay_rate ** epoch)
```

#### Cosine Annealing
```python
def cosine_annealing(initial_lr, epoch, total_epochs):
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
```

---

## 6. 动量方法 (Momentum)

### 6.1 为什么需要动量？

**问题**：标准SGD在峡谷地形震荡

**解决**：累积过去的梯度方向

```
vₜ = β vₜ₋₁ + (1-β) ∇L(θₜ)
θₜ₊₁ = θₜ - η vₜ

β: 动量系数（通常0.9）
```

```python
def sgd_momentum(X, y, batch_size=32, lr=0.01, momentum=0.9, epochs=100):
    N, D = X.shape
    w = np.zeros(D)
    b = 0

    # 动量项
    v_w = np.zeros(D)
    v_b = 0

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # 前向和梯度
            y_pred = X_batch @ w + b
            grad_w = X_batch.T @ (y_pred - y_batch) / len(y_batch)
            grad_b = np.sum(y_pred - y_batch) / len(y_batch)

            # 更新动量
            v_w = momentum * v_w + (1 - momentum) * grad_w
            v_b = momentum * v_b + (1 - momentum) * grad_b

            # 更新参数
            w -= lr * v_w
            b -= lr * v_b

    return w, b
```

**优势**：
- 加速相同方向
- 抑制震荡方向
- 更快收敛

---

## 7. 自适应学习率方法

### 7.1 AdaGrad

**思想**：频繁更新的参数降低学习率

```
sₜ = sₜ₋₁ + gₜ²
θₜ₊₁ = θₜ - η / √(sₜ + ε) × gₜ
```

```python
def adagrad(X, y, lr=0.01, epochs=100, eps=1e-8):
    N, D = X.shape
    w = np.zeros(D)
    b = 0

    # 累积平方梯度
    s_w = np.zeros(D)
    s_b = 0

    for epoch in range(epochs):
        # ... 前向和梯度计算 ...
        y_pred = X @ w + b
        grad_w = X.T @ (y_pred - y) / N
        grad_b = np.sum(y_pred - y) / N

        # 累积
        s_w += grad_w**2
        s_b += grad_b**2

        # 自适应更新
        w -= lr * grad_w / (np.sqrt(s_w) + eps)
        b -= lr * grad_b / (np.sqrt(s_b) + eps)

    return w, b
```

**问题**：学习率单调递减（后期几乎不更新）

### 7.2 RMSprop

**改进**：使用指数移动平均

```
sₜ = β sₜ₋₁ + (1-β) gₜ²
θₜ₊₁ = θₜ - η / √(sₜ + ε) × gₜ
```

```python
def rmsprop(X, y, lr=0.001, beta=0.9, epochs=100, eps=1e-8):
    N, D = X.shape
    w = np.zeros(D)
    s_w = np.zeros(D)

    for epoch in range(epochs):
        # ... 梯度 ...
        grad_w = compute_gradient(X, y, w)

        # 指数移动平均
        s_w = beta * s_w + (1 - beta) * grad_w**2

        # 更新
        w -= lr * grad_w / (np.sqrt(s_w) + eps)

    return w
```

### 7.3 Adam（最常用）

**结合Momentum + RMSprop**：

```
mₜ = β₁ mₜ₋₁ + (1-β₁) gₜ       (一阶矩)
vₜ = β₂ vₜ₋₁ + (1-β₂) gₜ²      (二阶矩)

m̂ₜ = mₜ / (1 - β₁ᵗ)            (偏差修正)
v̂ₜ = vₜ / (1 - β₂ᵗ)

θₜ₊₁ = θₜ - η m̂ₜ / (√v̂ₜ + ε)
```

```python
def adam(X, y, lr=0.001, beta1=0.9, beta2=0.999, epochs=100, eps=1e-8):
    N, D = X.shape
    w = np.zeros(D)

    # 初始化矩估计
    m_w = np.zeros(D)
    v_w = np.zeros(D)
    t = 0

    for epoch in range(epochs):
        indices = np.random.permutation(N)

        for i in range(0, N, 32):
            t += 1
            X_batch = X[indices[i:i+32]]
            y_batch = y[indices[i:i+32]]

            # 梯度
            grad_w = compute_gradient(X_batch, y_batch, w)

            # 更新矩估计
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w**2

            # 偏差修正
            m_hat = m_w / (1 - beta1**t)
            v_hat = v_w / (1 - beta2**t)

            # 更新参数
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return w
```

**为什么流行**：
- 对超参数不敏感
- 适用于大多数问题
- 默认参数通常就很好

---

## 8. 实战技巧

### 8.1 梯度裁剪

**防止梯度爆炸**：

```python
def clip_gradients(grads, max_norm):
    """按范数裁剪"""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))

    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        grads = [g * clip_coef for g in grads]

    return grads

# 使用
grads = [grad_w, grad_b]
grads = clip_gradients(grads, max_norm=1.0)
```

### 8.2 学习率预热 (Warmup)

**前几个epoch线性增加学习率**：

```python
def warmup_schedule(lr_max, epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return lr_max * (epoch + 1) / warmup_epochs
    else:
        return lr_max

# 常用于Transformer训练
```

### 8.3 权重衰减 (Weight Decay)

**L2正则化的另一种形式**：

```python
def sgd_with_weight_decay(grad_w, w, lr, weight_decay):
    """
    相当于在梯度中加入 weight_decay * w
    """
    w = w - lr * (grad_w + weight_decay * w)
    return w
```

---

## 9. 优化器对比

### 9.1 收敛速度对比

```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2*(1 - x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

# 测试不同优化器
optimizers = {
    'SGD': lambda: sgd_optimizer(lr=0.0001),
    'Momentum': lambda: momentum_optimizer(lr=0.0001, beta=0.9),
    'Adam': lambda: adam_optimizer(lr=0.01)
}

results = {}
for name, opt_fn in optimizers.items():
    x = np.array([-1.0, 1.0])
    trajectory = [x.copy()]

    for _ in range(100):
        grad = rosenbrock_grad(x[0], x[1])
        x = opt_fn().step(x, grad)
        trajectory.append(x.copy())

    results[name] = trajectory

# 可视化
plt.figure(figsize=(10, 8))
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)

plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')

for name, traj in results.items():
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], 'o-', label=name, markersize=3)

plt.plot(1, 1, 'r*', markersize=15, label='Optimum')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('不同优化器的收敛路径')
plt.show()
```

### 9.2 选择建议

| 优化器 | 适用场景 | 典型学习率 |
|-------|---------|-----------|
| **SGD** | 简单问题、已调优 | 0.01 - 0.1 |
| **SGD+Momentum** | CV任务（ResNet等） | 0.01 - 0.1 |
| **Adam** | **默认首选** | 1e-4 - 1e-3 |
| **AdamW** | Transformer | 1e-4 - 3e-4 |

---

## 10. PyTorch实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 选择优化器
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
for epoch in range(10):
    for X_batch, y_batch in dataloader:
        # 前向
        outputs = model(X_batch)
        loss = nn.CrossEntropyLoss()(outputs, y_batch)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
for epoch in range(10):
    # ... 训练 ...
    scheduler.step()
```

---

## 11. 调试技巧

### 11.1 监控梯度

```python
# 检查梯度范数
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm = {grad_norm:.4f}")

# 梯度太小（< 1e-5）→ 梯度消失
# 梯度太大（> 1）→ 可能梯度爆炸
```

### 11.2 损失不下降？

**检查清单**：
```
1. 学习率太大？→ 减小10倍
2. 学习率太小？→ 增大10倍
3. 数据归一化了吗？
4. 梯度爆炸？→ 添加梯度裁剪
5. 模型太复杂？→ 先在小数据上过拟合测试
```

---

## 12. 练习

1. **实现练习**：
   - 从零实现SGD、Momentum、Adam
   - 在简单函数上验证收敛

2. **对比实验**：
   - 在同一数据集上对比不同优化器
   - 绘制损失曲线

3. **超参数调优**：
   - 尝试不同学习率
   - 观察batch size的影响

---

**恭喜！** 你已完成微积分模块。接下来前往 [03_probability_statistics](../../03_probability_statistics/) 学习概率统计。
