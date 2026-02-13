# 导数和偏导数

## 1. 导数的直观理解

### 1.1 什么是导数？

**导数 = 瞬时变化率 = 切线斜率**

```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

**例子**：汽车速度
- 位置函数：`s(t) = t²`
- 速度（导数）：`v(t) = s'(t) = 2t`

在`t=3`时，速度为`v(3) = 6 m/s`

### 1.2 几何意义

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

# 在x=2处的导数
x0 = 2
f_x0 = f(x0)
f_prime_x0 = 2 * x0  # f'(x) = 2x

# 切线方程: y - f(x0) = f'(x0) * (x - x0)
x = np.linspace(0, 4, 100)
y_curve = f(x)
y_tangent = f_prime_x0 * (x - x0) + f_x0

plt.plot(x, y_curve, label='f(x) = x²')
plt.plot(x, y_tangent, '--', label=f'Tangent at x={x0}')
plt.scatter([x0], [f_x0], color='red', s=100, zorder=5)
plt.legend()
plt.grid(True)
plt.show()
```

---

## 2. 基本导数规则

### 2.1 常见函数的导数

| 函数 | 导数 |
|-----|-----|
| `c` (常数) | `0` |
| `x` | `1` |
| `x²` | `2x` |
| `xⁿ` | `n × x^(n-1)` |
| `eˣ` | `eˣ` |
| `ln(x)` | `1/x` |
| `sin(x)` | `cos(x)` |
| `cos(x)` | `-sin(x)` |

### 2.2 运算法则

**常数倍法则**:
```
(c × f)' = c × f'
```

**加法法则**:
```
(f + g)' = f' + g'
```

**乘法法则** (Product Rule):
```
(f × g)' = f' × g + f × g'
```

**除法法则** (Quotient Rule):
```
(f / g)' = (f' × g - f × g') / g²
```

**链式法则** (Chain Rule) - 下一节详细讲:
```
(f(g(x)))' = f'(g(x)) × g'(x)
```

---

## 3. 深度学习中的常见导数

### 3.1 激活函数导数

#### Sigmoid
```
σ(x) = 1 / (1 + e^(-x))

σ'(x) = σ(x) × (1 - σ(x))
```

**推导**:
```python
# 设 y = 1 / (1 + e^(-x))
# 可以写成 y = (1 + e^(-x))^(-1)

# 使用链式法则:
# dy/dx = -1 × (1 + e^(-x))^(-2) × (-e^(-x))
#       = e^(-x) / (1 + e^(-x))²

# 化简（代入 y）:
# dy/dx = y × (1 - y)
```

**代码验证**:
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 数值验证
x = 2.0
h = 1e-7
numerical_grad = (sigmoid(x + h) - sigmoid(x)) / h
analytical_grad = sigmoid_derivative(x)

print(f"数值导数: {numerical_grad:.6f}")
print(f"解析导数: {analytical_grad:.6f}")
# 两者应该非常接近
```

#### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x) = {
    x  if x > 0
    0  if x ≤ 0
}

ReLU'(x) = {
    1  if x > 0
    0  if x ≤ 0
}
```

**注意**: 在`x=0`处不可导，实践中通常定义为0

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

#### Tanh
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

tanh'(x) = 1 - tanh²(x)
```

```python
def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t**2
```

#### Leaky ReLU
```
LeakyReLU(x) = max(α×x, x)  (α通常为0.01)

LeakyReLU'(x) = {
    1  if x > 0
    α  if x ≤ 0
}
```

**优势**: 避免"神经元死亡"问题

---

## 4. 偏导数 (Partial Derivatives)

### 4.1 什么是偏导数？

对于多元函数`f(x, y)`，**偏导数**是对其中一个变量求导，其他变量视为常数。

**符号**:
- `∂f/∂x`: 对x的偏导数
- `∂f/∂y`: 对y的偏导数

**例子**:
```
f(x, y) = x²y + 3xy²

∂f/∂x = 2xy + 3y²  (把y当常数)
∂f/∂y = x² + 6xy   (把x当常数)
```

### 4.2 神经网络中的偏导数

**全连接层**:
```python
# 前向传播
z = w₁x₁ + w₂x₂ + b

# 损失函数（例如MSE）
L = (z - y)²

# 需要计算的偏导数:
∂L/∂w₁, ∂L/∂w₂, ∂L/∂b
```

**计算**:
```
∂L/∂z = 2(z - y)

∂z/∂w₁ = x₁
∂z/∂w₂ = x₂
∂z/∂b = 1

使用链式法则:
∂L/∂w₁ = (∂L/∂z) × (∂z/∂w₁) = 2(z - y) × x₁
∂L/∂w₂ = (∂L/∂z) × (∂z/∂w₂) = 2(z - y) × x₂
∂L/∂b = (∂L/∂z) × (∂z/∂b) = 2(z - y)
```

---

## 5. 梯度 (Gradient)

### 5.1 梯度向量

**梯度**是所有偏导数组成的向量：

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**例子**:
```
f(x, y) = x² + y²

∇f = [∂f/∂x, ∂f/∂y]ᵀ = [2x, 2y]ᵀ

在点(3, 4)处:
∇f(3, 4) = [6, 8]ᵀ
```

### 5.2 梯度的几何意义

**梯度方向 = 函数增长最快的方向**

```python
import numpy as np
import matplotlib.pyplot as plt

# 函数: f(x, y) = x² + y²
def f(x, y):
    return x**2 + y**2

# 梯度: ∇f = [2x, 2y]
def gradient(x, y):
    return np.array([2*x, 2*y])

# 绘制等高线和梯度
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, levels=10)
plt.colorbar(label='f(x, y)')

# 绘制几个点的梯度向量
points = [(1, 1), (-1, 2), (2, -1)]
for px, py in points:
    grad = gradient(px, py)
    plt.arrow(px, py, grad[0]*0.2, grad[1]*0.2,
              head_width=0.2, color='red')
    plt.plot(px, py, 'ro')

plt.xlabel('x')
plt.ylabel('y')
plt.title('梯度指向函数增长最快的方向')
plt.axis('equal')
plt.show()
```

### 5.3 方向导数

沿着单位向量`v`的**方向导数**:

```
D_v f = ∇f · v = ||∇f|| × ||v|| × cos(θ)
```

- 沿梯度方向：`D_∇f f = ||∇f||`（最大）
- 垂直梯度方向：`D_v f = 0`
- 反梯度方向：`D_{-∇f} f = -||∇f||`（最小）

**这就是为什么梯度下降沿着负梯度方向更新参数！**

---

## 6. 数值微分 vs 自动微分

### 6.1 数值微分 (Numerical Differentiation)

**有限差分法**:
```
f'(x) ≈ (f(x + h) - f(x)) / h
```

```python
def numerical_derivative(f, x, h=1e-7):
    """数值求导（前向差分）"""
    return (f(x + h) - f(x)) / h

# 更精确的中心差分
def numerical_derivative_central(f, x, h=1e-5):
    """数值求导（中心差分）"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 测试
def f(x):
    return x**2

print(f"数值导数: {numerical_derivative(f, 3.0)}")  # ≈ 6
print(f"真实导数: {2 * 3.0}")  # = 6
```

**问题**:
- **计算慢**：每个参数需要额外前向传播
- **数值误差**：舍入误差、截断误差
- **不适合高维**：N个参数需要N+1次前向传播

### 6.2 自动微分 (Automatic Differentiation)

**符号微分 + 链式法则 = 自动微分**

PyTorch/TensorFlow使用**反向模式自动微分**（也叫反向传播）：

```python
import torch

# 定义变量（需要梯度）
x = torch.tensor(3.0, requires_grad=True)

# 前向传播
y = x**2 + 2*x + 1

# 反向传播（自动计算梯度）
y.backward()

print(f"dy/dx = {x.grad}")  # 2*3 + 2 = 8
```

**优势**:
- **精确**：机器精度
- **高效**：一次反向传播得到所有梯度
- **通用**：适用于任意可导函数

---

## 7. 梯度检查 (Gradient Checking)

调试反向传播实现时，用数值梯度验证：

```python
def gradient_check(f, x, grad_analytical, epsilon=1e-7):
    """
    检查解析梯度是否正确

    f: 损失函数
    x: 参数
    grad_analytical: 解析计算的梯度
    """
    grad_numerical = np.zeros_like(x)

    # 对每个参数计算数值梯度
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        # f(x + h)
        x[idx] = old_value + epsilon
        fxph = f(x)

        # f(x - h)
        x[idx] = old_value - epsilon
        fxmh = f(x)

        # 数值梯度
        grad_numerical[idx] = (fxph - fxmh) / (2 * epsilon)

        # 恢复原值
        x[idx] = old_value
        it.iternext()

    # 计算相对误差
    numerator = np.linalg.norm(grad_analytical - grad_numerical)
    denominator = np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical)
    relative_error = numerator / denominator

    print(f"相对误差: {relative_error}")
    if relative_error < 1e-7:
        print("✓ 梯度检查通过")
    else:
        print("✗ 梯度可能有误")

    return relative_error

# 示例
x = np.array([1.0, 2.0, 3.0])
def f(x):
    return np.sum(x**2)

# 解析梯度
grad_analytical = 2 * x

gradient_check(f, x, grad_analytical)
```

---

## 8. 实战示例

### 示例1：计算简单神经元的梯度

```python
# 神经元: y = σ(wx + b)
def neuron_forward(x, w, b):
    z = w * x + b
    y = 1 / (1 + np.exp(-z))  # sigmoid
    return y, z

def neuron_backward(x, w, b, y_true):
    """
    计算损失 L = (y - y_true)² 对 w 和 b 的梯度
    """
    # 前向传播
    y, z = neuron_forward(x, w, b)

    # 反向传播
    # dL/dy
    dL_dy = 2 * (y - y_true)

    # dy/dz (sigmoid导数)
    dy_dz = y * (1 - y)

    # dz/dw 和 dz/db
    dz_dw = x
    dz_db = 1

    # 链式法则
    dL_dw = dL_dy * dy_dz * dz_dw
    dL_db = dL_dy * dy_dz * dz_db

    return dL_dw, dL_db

# 测试
x = 2.0
w = 0.5
b = 0.1
y_true = 1.0

dw, db = neuron_backward(x, w, b, y_true)
print(f"dL/dw = {dw}")
print(f"dL/db = {db}")
```

### 示例2：多元函数的梯度

```python
def rosenbrock(x):
    """
    Rosenbrock函数: f(x,y) = (1-x)² + 100(y-x²)²
    最小值在 (1, 1)
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x):
    """解析梯度"""
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

# 验证梯度
x = np.array([0.5, 0.5])
grad_analytical = rosenbrock_gradient(x)

# 数值梯度
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

grad_numerical = numerical_gradient(rosenbrock, x)

print("解析梯度:", grad_analytical)
print("数值梯度:", grad_numerical)
print("差异:", np.linalg.norm(grad_analytical - grad_numerical))
```

---

## 9. 常见错误

### ❌ 错误1：忘记链式法则

```python
# 错误
def backward_wrong(x, w):
    y = w * x**2
    # 错误：只对w求导
    grad_w = x**2  # ✗ 不完整

# 正确
def backward_correct(x, w):
    # L = (y - target)²，需要链式法则
    # dL/dw = dL/dy × dy/dw
    pass
```

### ❌ 错误2：原地操作破坏计算图

```python
# PyTorch中
x = torch.tensor([1.0], requires_grad=True)
y = x + 2
y += 3  # ✗ 原地操作，梯度会出错

# 应该用
y = y + 3  # ✓
```

### ❌ 错误3：在错误的位置detach

```python
# 错误：过早切断梯度
x = torch.randn(10, requires_grad=True)
y = x.detach() + 2  # ✗ 梯度无法回传到x

# 正确：只在必要时detach
y = x + 2
# 只有不需要梯度时才detach
```

---

## 10. 练习

1. **基础练习**：
   - 手动计算 `f(x) = 3x² - 5x + 2` 在 `x=2` 的导数
   - 验证 ReLU 导数的实现

2. **偏导数练习**：
   - 对 `f(x,y) = x³y² + sin(x)×y` 求偏导数

3. **实战练习**：
   - 实现一个两层神经网络，手动计算梯度
   - 用数值梯度验证你的实现

---

**下一步**：学习 [02_chain_rule.md](./02_chain_rule.md)，深入理解链式法则和反向传播算法。
