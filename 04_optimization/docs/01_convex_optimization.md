# 凸优化基础

## 为什么学习凸优化？

虽然深度学习的损失函数通常是**非凸**的，但理解凸优化仍然重要：

1. **提供理论基础**：凸优化有全局最优解，帮助理解优化问题
2. **局部类似凸**：神经网络训练过程中，局部区域常表现出类似凸的性质
3. **某些问题是凸的**：如线性回归、SVM（原始形式）
4. **优化算法设计**：许多非凸优化算法源自凸优化理论

---

## 1. 凸集 (Convex Set)

### 1.1 定义

集合 C 是凸集，如果对于 C 中任意两点 x, y 和 0 ≤ θ ≤ 1，有：

```
θx + (1-θ)y ∈ C
```

**几何意义**：连接集合内任意两点的线段仍在集合内

**例子**：
```python
import numpy as np
import matplotlib.pyplot as plt

# 凸集示例：圆形
theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)

plt.figure(figsize=(12, 4))

# 凸集：圆
plt.subplot(131)
plt.fill(x, y, alpha=0.3)
plt.plot([0.5, -0.5], [0.3, -0.3], 'ro-', linewidth=2)
plt.title('凸集：圆形\n任意两点连线都在集合内')
plt.axis('equal')

# 凸集：矩形
plt.subplot(132)
rect = plt.Rectangle((-1, -0.5), 2, 1, fill=True, alpha=0.3)
plt.gca().add_patch(rect)
plt.plot([0.5, -0.5], [0.3, -0.3], 'ro-', linewidth=2)
plt.title('凸集：矩形')
plt.axis('equal')

# 非凸集：月牙形
plt.subplot(133)
theta1 = np.linspace(0, 2*np.pi, 100)
theta2 = np.linspace(0, 2*np.pi, 100)
x1 = np.cos(theta1)
y1 = np.sin(theta1)
x2 = 0.5 + 0.6 * np.cos(theta2)
y2 = 0.6 * np.sin(theta2)
plt.fill(x1, y1, alpha=0.3, color='blue')
plt.fill(x2, y2, alpha=1, color='white')
plt.plot([0.8, -0.5], [0.2, -0.3], 'ro-', linewidth=2)
plt.title('非凸集：月牙形\n连线跑出去了')
plt.axis('equal')

plt.tight_layout()
plt.savefig('convex_sets.png', dpi=100, bbox_inches='tight')
```

### 1.2 常见凸集

```
1. 超平面: {x | aᵀx = b}
2. 半空间: {x | aᵀx ≤ b}
3. 欧氏球: {x | ||x - x_c|| ≤ r}
4. 椭球: {x | (x-x_c)ᵀP⁻¹(x-x_c) ≤ 1}
5. 多面体: {x | Ax ≤ b}
```

---

## 2. 凸函数 (Convex Function)

### 2.1 定义

函数 f: ℝⁿ → ℝ 是凸函数，如果其定义域 dom(f) 是凸集，且对于任意 x, y ∈ dom(f) 和 0 ≤ θ ≤ 1，有：

```
f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)
```

**几何意义**：函数图像上任意两点间的线段在函数图像上方

**严格凸**：不等号严格成立 (<)

### 2.2 一阶条件（可微情况）

f 是凸函数 ⟺ 对于所有 x, y：

```
f(y) ≥ f(x) + ∇f(x)ᵀ(y - x)
```

**含义**：一阶泰勒展开是函数的全局下界

**代码验证**：
```python
import numpy as np

def f(x):
    """凸函数示例: f(x) = x²"""
    return x**2

def grad_f(x):
    """梯度"""
    return 2*x

# 选择一个点
x0 = 1.0

# 检查一阶条件
x_test = np.linspace(-2, 3, 100)
f_x = f(x_test)
linear_approx = f(x0) + grad_f(x0) * (x_test - x0)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(x_test, f_x, 'b-', linewidth=2, label='f(x) = x²')
plt.plot(x_test, linear_approx, 'r--', linewidth=2,
         label=f'一阶逼近（在x={x0}）')
plt.plot(x0, f(x0), 'go', markersize=10, label='切点')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('凸函数的一阶条件\n线性逼近是全局下界')
plt.xlabel('x')
plt.ylabel('f(x)')
```

### 2.3 二阶条件（二阶可微）

f 是凸函数 ⟺ Hessian矩阵 ∇²f(x) 半正定（对于所有 x）

```
∇²f(x) ⪰ 0  （所有特征值 ≥ 0）
```

**一维情况**：f''(x) ≥ 0

**例子**：
```python
# 凸函数
f1(x) = x²           # f''(x) = 2 > 0  ✓ 凸
f2(x) = e^x          # f''(x) = e^x > 0  ✓ 凸
f3(x) = -log(x)      # f''(x) = 1/x² > 0  ✓ 凸（x > 0）

# 非凸函数
f4(x) = x³           # f''(x) = 6x  （变号）✗ 非凸
f5(x) = sin(x)       # f''(x) = -sin(x)  ✗ 非凸
```

### 2.4 可视化对比

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)

plt.figure(figsize=(15, 5))

# 凸函数示例
plt.subplot(131)
plt.plot(x, x**2, 'b-', linewidth=2, label='f(x) = x²')
plt.plot(x, np.abs(x), 'r-', linewidth=2, label='f(x) = |x|')
plt.plot(x, np.exp(x), 'g-', linewidth=2, label='f(x) = e^x')
plt.legend()
plt.title('凸函数')
plt.grid(True, alpha=0.3)

# 凹函数示例
plt.subplot(132)
plt.plot(x, -x**2, 'b-', linewidth=2, label='f(x) = -x²')
plt.plot(x, np.log(x[x>0]+0.1), 'r-', linewidth=2, label='f(x) = log(x)')
plt.plot(x, np.sqrt(np.abs(x)), 'g-', linewidth=2, label='f(x) = √|x|')
plt.legend()
plt.title('凹函数（Concave）')
plt.grid(True, alpha=0.3)

# 非凸非凹函数
plt.subplot(133)
plt.plot(x, x**3, 'b-', linewidth=2, label='f(x) = x³')
plt.plot(x, np.sin(x), 'r-', linewidth=2, label='f(x) = sin(x)')
plt.plot(x, x**4 - 2*x**2, 'g-', linewidth=2, label='f(x) = x⁴ - 2x²')
plt.legend()
plt.title('非凸非凹函数')
plt.grid(True, alpha=0.3)

plt.tight_layout()
```

---

## 3. 凸优化问题

### 3.1 标准形式

```
minimize    f(x)
subject to  gᵢ(x) ≤ 0,  i = 1, ..., m    # 不等式约束
            hⱼ(x) = 0,  j = 1, ..., p    # 等式约束
```

**凸优化问题的要求**：
1. 目标函数 f(x) 是凸函数
2. 不等式约束 gᵢ(x) 是凸函数
3. 等式约束 hⱼ(x) 是仿射函数（Ax + b）

### 3.2 凸优化的重要性质

**全局最优性**：凸优化问题的局部最优解就是全局最优解

**证明**（反证法）：
```
假设 x* 是局部最优但不是全局最优
则存在 y 使得 f(y) < f(x*)

由于 dom(f) 是凸集，x* 和 y 之间的点都在定义域内
z = θx* + (1-θ)y，其中 θ 接近 1

由凸性：f(z) ≤ θf(x*) + (1-θ)f(y) < f(x*)

这与 x* 是局部最优矛盾！
```

### 3.3 最优性条件

**一阶最优性条件**（无约束）：

对于可微凸函数，x* 是最优解 ⟺ 梯度为零

```
∇f(x*) = 0
```

**KKT条件**（有约束，将在03_lagrange.md详述）

---

## 4. 深度学习中的非凸问题

### 4.1 为什么神经网络是非凸的？

```python
# 简单的2层神经网络
def f(w1, w2, x):
    h = relu(w1 * x)  # 隐藏层
    y = w2 * h
    return y

# 损失函数对参数不是凸的！
# 原因：
# 1. ReLU等激活函数引入非线性
# 2. 参数之间存在交互（w1 和 w2 相乘）
# 3. 存在对称性（交换神经元不改变输出）
```

**可视化损失曲面**：
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 简单神经网络的损失曲面
def loss_surface(w1, w2):
    """模拟神经网络损失（非凸）"""
    return (w1**2 + w2**2) * (1 + 0.5*np.sin(5*w1) * np.sin(5*w2))

w1 = np.linspace(-2, 2, 100)
w2 = np.linspace(-2, 2, 100)
W1, W2 = np.meshgrid(w1, w2)
Z = loss_surface(W1, W2)

fig = plt.figure(figsize=(12, 5))

# 3D曲面图
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w₁')
ax1.set_ylabel('w₂')
ax1.set_zlabel('Loss')
ax1.set_title('非凸损失曲面（3D）')

# 等高线图
ax2 = fig.add_subplot(122)
contour = ax2.contour(W1, W2, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('w₁')
ax2.set_ylabel('w₂')
ax2.set_title('损失等高线\n可见多个局部最优')
plt.colorbar(contour, ax=ax2)

plt.tight_layout()
```

### 4.2 实践中的策略

虽然神经网络优化是非凸的，但实践中能work：

1. **过参数化**：参数远多于数据，许多不同参数组合都能拟合数据
2. **随机初始化**：避免陷入糟糕的局部最优
3. **Batch Normalization等技巧**：使损失曲面更平滑
4. **局部类似凸**：在优化路径附近，损失常表现出凸性质

```python
# 实验：对比凸和非凸优化

import torch
import torch.nn as nn

# 凸问题：线性回归
X = torch.randn(100, 10)
y_true = X @ torch.randn(10, 1) + 0.1 * torch.randn(100, 1)

# 模型1：线性（凸）
model_linear = nn.Linear(10, 1, bias=False)

# 模型2：深度网络（非凸）
model_deep = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# 训练对比
def train(model, X, y, epochs=100):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

losses_linear = train(model_linear, X, y_true)
losses_deep = train(model_deep, X, y_true)

plt.figure(figsize=(10, 5))
plt.plot(losses_linear, label='线性模型（凸）', linewidth=2)
plt.plot(losses_deep, label='深度模型（非凸）', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.title('凸 vs 非凸优化\n线性模型收敛更稳定')
plt.grid(True, alpha=0.3)
```

---

## 5. 保凸运算

理解哪些运算保持凸性，有助于构造新的凸函数：

### 5.1 基本规则

1. **非负加权和**：
   ```
   若 f₁, f₂ 凸，α₁, α₂ ≥ 0
   则 α₁f₁ + α₂f₂ 也凸
   ```

2. **逐点最大值**：
   ```
   若 f₁, ..., f_m 凸
   则 f(x) = max{f₁(x), ..., f_m(x)} 也凸
   ```

3. **复合（特定情况）**：
   ```
   若 f 凸且单调递增，g 凸
   则 f(g(x)) 凸
   ```

### 5.2 深度学习中的例子

```python
# 1. 损失函数是多个凸函数的和
total_loss = mse_loss + 0.01 * l2_regularization  # ✓ 凸（对于线性模型）

# 2. Hinge loss（SVM）：max(0, 1 - y·f(x))
# 这是 max{f₁, f₂} 的形式，f₁=0 和 f₂=1-y·f(x) 都是凸的（对于线性f）

# 3. 激活函数
# ReLU(x) = max(0, x)  # 凸函数
# 但多层网络不凸！因为参数之间有交互
```

---

## 6. 实用检查清单

**判断凸函数**：
- [ ] 计算二阶导数（Hessian），检查是否半正定
- [ ] 画出函数图像，视觉检查
- [ ] 利用保凸运算规则组合

**判断凸优化问题**：
- [ ] 目标函数是凸的
- [ ] 不等式约束函数是凸的
- [ ] 等式约束是仿射的

---

## 7. 总结

| 概念 | 定义 | 深度学习相关 |
|------|------|-------------|
| **凸集** | 任意两点连线在集合内 | 参数空间的约束 |
| **凸函数** | 函数图像上凸 | 理想的损失函数（但实际不是） |
| **凸优化** | 最小化凸函数 | 线性模型、SVM原始问题 |
| **全局最优** | 凸优化保证 | 深度学习不保证（非凸） |

**核心要点**：
1. 凸优化有全局最优解，非常好解
2. 深度学习是非凸的，但实践中能work
3. 理解凸性有助于理解优化算法

---

## 下一步

继续学习 [02_gradient_methods.md](02_gradient_methods.md)，掌握实际使用的优化算法（SGD、Adam等）。
