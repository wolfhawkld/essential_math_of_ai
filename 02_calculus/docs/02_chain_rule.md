# 链式法则与反向传播

## 1. 链式法则 (Chain Rule)

### 1.1 一元函数的链式法则

对于复合函数 `y = f(g(x))`：

```
dy/dx = (dy/dg) × (dg/dx) = f'(g(x)) × g'(x)
```

**例子**:
```
y = (2x + 1)³

设 u = 2x + 1, 则 y = u³

dy/du = 3u²
du/dx = 2

dy/dx = (dy/du) × (du/dx) = 3u² × 2 = 6(2x + 1)²
```

**代码验证**:
```python
import numpy as np

def f(x):
    return (2*x + 1)**3

def df_dx_analytical(x):
    """解析导数"""
    return 6 * (2*x + 1)**2

def df_dx_numerical(x, h=1e-7):
    """数值导数"""
    return (f(x + h) - f(x)) / h

x = 3.0
print(f"解析: {df_dx_analytical(x)}")  # 294
print(f"数值: {df_dx_numerical(x)}")   # ≈ 294
```

### 1.2 多层复合函数

对于 `y = f(g(h(x)))`：

```
dy/dx = (dy/df) × (df/dg) × (dg/dh) × (dh/dx)
```

**这就是深度神经网络的数学基础！**

---

## 2. 计算图 (Computational Graph)

### 2.1 什么是计算图？

计算图是表示数学运算的有向无环图（DAG）：
- **节点**：变量或运算
- **边**：数据流

**例子**：`L = (wx + b - y)²`

```
    x ──→ * ──→ + ──→ - ──→ square ──→ L
          ↑     ↑     ↑
          w     b     y
```

### 2.2 前向传播 (Forward Pass)

从输入到输出计算值：

```python
# L = (wx + b - y)²
def forward(x, w, b, y_true):
    z1 = w * x          # 乘法
    z2 = z1 + b         # 加法
    z3 = z2 - y_true    # 减法
    L = z3**2           # 平方
    return L, (z1, z2, z3)  # 保存中间值供反向传播用
```

### 2.3 反向传播 (Backward Pass)

从输出到输入计算梯度：

```python
def backward(x, w, b, y_true, z1, z2, z3):
    """
    反向传播计算梯度
    """
    # dL/dz3
    dL_dz3 = 2 * z3

    # dL/dz2 = dL/dz3 × dz3/dz2
    dz3_dz2 = 1
    dL_dz2 = dL_dz3 * dz3_dz2

    # dL/dz1 = dL/dz2 × dz2/dz1
    dz2_dz1 = 1
    dL_dz1 = dL_dz2 * dz2_dz1

    # dL/dw = dL/dz1 × dz1/dw
    dz1_dw = x
    dL_dw = dL_dz1 * dz1_dw

    # dL/db = dL/dz2 × dz2/db
    dz2_db = 1
    dL_db = dL_dz2 * dz2_db

    return dL_dw, dL_db

# 测试
x, w, b, y_true = 2.0, 0.5, 0.3, 1.0
L, (z1, z2, z3) = forward(x, w, b, y_true)
dL_dw, dL_db = backward(x, w, b, y_true, z1, z2, z3)

print(f"Loss: {L}")
print(f"dL/dw: {dL_dw}")
print(f"dL/db: {dL_db}")
```

---

## 3. 反向传播算法

### 3.1 算法流程

```
1. 前向传播：
   - 计算所有中间变量
   - 保存用于反向传播

2. 反向传播：
   - 从输出开始
   - 按拓扑逆序计算梯度
   - 使用链式法则累积梯度
```

### 3.2 两层神经网络示例

**网络结构**:
```
输入 x → 隐藏层 h → 输出 y → 损失 L

h = σ(W1 @ x + b1)
y = W2 @ h + b2
L = 0.5 × (y - target)²
```

**前向传播**:
```python
def forward_two_layer(x, W1, b1, W2, b2, target):
    """
    x: (D,) 输入
    W1: (H, D) 第一层权重
    b1: (H,) 第一层偏置
    W2: (K,) 第二层权重
    b2: 标量
    """
    # 第一层
    z1 = W1 @ x + b1  # (H,)
    h = sigmoid(z1)   # (H,) 激活

    # 第二层
    z2 = W2 @ h + b2  # 标量
    y = z2            # 线性输出

    # 损失
    L = 0.5 * (y - target)**2

    # 保存缓存
    cache = (x, z1, h, z2, y, target)
    return L, cache

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

**反向传播**:
```python
def backward_two_layer(W1, W2, cache):
    x, z1, h, z2, y, target = cache

    # dL/dy
    dL_dy = y - target  # 标量

    # dL/dz2 = dL/dy × dy/dz2
    dy_dz2 = 1
    dL_dz2 = dL_dy * dy_dz2

    # dL/dW2 = dL/dz2 × dz2/dW2
    dz2_dW2 = h  # (H,)
    dL_dW2 = dL_dz2 * dz2_dW2  # (H,)

    # dL/db2
    dL_db2 = dL_dz2

    # dL/dh = dL/dz2 × dz2/dh
    dz2_dh = W2  # (H,)
    dL_dh = dL_dz2 * dz2_dh  # (H,)

    # dL/dz1 = dL/dh × dh/dz1
    dh_dz1 = h * (1 - h)  # sigmoid导数
    dL_dz1 = dL_dh * dh_dz1  # (H,)

    # dL/dW1 = dL/dz1 @ dz1/dW1
    dz1_dW1 = x  # (D,)
    dL_dW1 = np.outer(dL_dz1, dz1_dW1)  # (H, D)

    # dL/db1
    dL_db1 = dL_dz1

    grads = {
        'W1': dL_dW1,
        'b1': dL_db1,
        'W2': dL_dW2,
        'b2': dL_db2
    }
    return grads
```

---

## 4. 向量化的反向传播

### 4.1 批量数据处理

**前向传播**:
```python
def forward_batch(X, W1, b1, W2, b2, targets):
    """
    X: (N, D) - N个样本，D维特征
    W1: (H, D)
    b1: (H,)
    W2: (K, H)
    b2: (K,)
    targets: (N, K)
    """
    # 第一层
    Z1 = X @ W1.T + b1  # (N, H)
    H = sigmoid(Z1)     # (N, H)

    # 第二层
    Z2 = H @ W2.T + b2  # (N, K)
    Y = Z2              # (N, K)

    # 损失（MSE）
    L = 0.5 * np.mean((Y - targets)**2)

    cache = (X, Z1, H, Z2, Y, targets)
    return L, cache
```

**反向传播**:
```python
def backward_batch(W1, W2, cache):
    X, Z1, H, Z2, Y, targets = cache
    N = X.shape[0]

    # dL/dY
    dL_dY = (Y - targets) / N  # (N, K) 除以N求均值梯度

    # dL/dW2
    dL_dW2 = dL_dY.T @ H  # (K, H)
    dL_db2 = np.sum(dL_dY, axis=0)  # (K,)

    # dL/dH
    dL_dH = dL_dY @ W2  # (N, H)

    # dL/dZ1
    dH_dZ1 = H * (1 - H)  # sigmoid导数 (N, H)
    dL_dZ1 = dL_dH * dH_dZ1  # (N, H)

    # dL/dW1
    dL_dW1 = dL_dZ1.T @ X  # (H, D)
    dL_db1 = np.sum(dL_dZ1, axis=0)  # (H,)

    grads = {
        'W1': dL_dW1,
        'b1': dL_db1,
        'W2': dL_dW2,
        'b2': dL_db2
    }
    return grads
```

---

## 5. 常见操作的梯度

### 5.1 矩阵乘法

**前向**:
```
Y = X @ W  # X: (N, D), W: (D, H)
```

**反向**:
```python
dL_dX = dL_dY @ W.T  # (N, H) @ (H, D) = (N, D)
dL_dW = X.T @ dL_dY  # (D, N) @ (N, H) = (D, H)
```

**记忆技巧**：保持维度一致
- `dL_dX` 形状与 `X` 相同
- `dL_dW` 形状与 `W` 相同

### 5.2 逐元素操作

**加法**:
```python
# 前向
Z = X + Y

# 反向
dL_dX = dL_dZ  # 梯度直接传递
dL_dY = dL_dZ
```

**乘法**:
```python
# 前向
Z = X * Y  # 逐元素

# 反向
dL_dX = dL_dZ * Y
dL_dY = dL_dZ * X
```

### 5.3 激活函数

**ReLU**:
```python
# 前向
def relu_forward(x):
    return np.maximum(0, x)

# 反向
def relu_backward(dout, x):
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx
```

**Softmax + 交叉熵**:
```python
# 前向
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(probs, targets):
    N = probs.shape[0]
    return -np.sum(targets * np.log(probs + 1e-8)) / N

# 反向（Softmax + CE的组合梯度非常简洁）
def softmax_cross_entropy_backward(probs, targets):
    """
    dL/dz = probs - targets
    """
    N = probs.shape[0]
    return (probs - targets) / N
```

---

## 6. 实现一个模块化的神经网络

### 6.1 层的抽象

```python
class Layer:
    def forward(self, x):
        """前向传播"""
        raise NotImplementedError

    def backward(self, dout):
        """反向传播"""
        raise NotImplementedError
```

### 6.2 全连接层

```python
class Linear(Layer):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        """
        x: (N, in_features)
        返回: (N, out_features)
        """
        self.x = x  # 保存用于反向传播
        return x @ self.W + self.b

    def backward(self, dout):
        """
        dout: (N, out_features) - 来自上层的梯度
        返回: (N, in_features) - 传给下层的梯度
        """
        # 梯度
        self.dW = self.x.T @ dout  # (in, N) @ (N, out) = (in, out)
        self.db = np.sum(dout, axis=0)  # (out,)

        # 传递给下一层
        dx = dout @ self.W.T  # (N, out) @ (out, in) = (N, in)
        return dx
```

### 6.3 ReLU层

```python
class ReLU(Layer):
    def forward(self, x):
        self.mask = (x > 0)  # 保存掩码
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
```

### 6.4 组合成网络

```python
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = [
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, output_size)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_params(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.append((layer.W, layer.dW))
                params.append((layer.b, layer.db))
        return params

# 使用
net = TwoLayerNet(784, 128, 10)

# 前向传播
x = np.random.randn(32, 784)
scores = net.forward(x)

# 假设已计算损失的梯度
dscores = np.random.randn(32, 10)
net.backward(dscores)

# 更新参数
learning_rate = 0.01
for param, grad in net.get_params():
    param -= learning_rate * grad
```

---

## 7. PyTorch中的自动微分

### 7.1 基本用法

```python
import torch

# 创建需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

# 前向传播
y = w * x + b
loss = (y - 5.0)**2

# 反向传播
loss.backward()

# 查看梯度
print(f"dL/dx = {x.grad}")
print(f"dL/dw = {w.grad}")
print(f"dL/db = {b.grad}")
```

### 7.2 神经网络示例

```python
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 训练
net = TwoLayerNet(784, 128, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(100):
    # 前向
    x = torch.randn(32, 784)
    targets = torch.randint(0, 10, (32,))
    scores = net(x)

    # 损失
    loss = nn.CrossEntropyLoss()(scores, targets)

    # 反向
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 自动计算梯度
    optimizer.step()       # 更新参数

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 8. 梯度消失与梯度爆炸

### 8.1 梯度消失 (Vanishing Gradient)

**原因**：
- 链式法则中多个小于1的导数相乘
- 常见于Sigmoid/Tanh激活（导数最大0.25）

```python
# 10层Sigmoid网络
def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

# 假设每层导数都是0.25
grad = 1.0
for i in range(10):
    grad *= 0.25
    print(f"第{i+1}层梯度: {grad}")

# 第10层梯度 ≈ 9.5e-7（几乎为0）
```

**解决方案**:
- 使用ReLU激活（导数为0或1）
- 残差连接（ResNet）
- 批量归一化（Batch Normalization）
- LSTM/GRU（RNN领域）

### 8.2 梯度爆炸 (Exploding Gradient)

**原因**：
- 链式法则中多个大于1的导数相乘
- 权重初始化不当

```python
# 权重过大导致梯度爆炸
W = np.random.randn(100, 100) * 10  # 大权重

grad = np.ones((1, 100))
for i in range(10):
    grad = grad @ W
    print(f"第{i+1}层梯度范数: {np.linalg.norm(grad)}")

# 梯度指数增长
```

**解决方案**:
- 梯度裁剪（Gradient Clipping）
- 合理的权重初始化（Xavier/He初始化）
- 使用Batch Normalization

```python
# 梯度裁剪
def clip_gradients(grads, max_norm):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for g in grads:
            g *= clip_coef
    return grads
```

---

## 9. 调试反向传播

### 9.1 梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    """数值梯度"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + h
        fxh = f(x)

        x[idx] = old_val - h
        fxmh = f(x)

        grad[idx] = (fxh - fxmh) / (2 * h)
        x[idx] = old_val
        it.iternext()
    return grad

def gradient_check(net, x, y):
    """检查网络梯度"""
    # 解析梯度
    loss = net.loss(x, y)
    net.backward()
    grads_analytical = net.get_grads()

    # 数值梯度
    def f(params):
        net.set_params(params)
        return net.loss(x, y)

    grads_numerical = numerical_gradient(f, net.get_params())

    # 比较
    for ga, gn in zip(grads_analytical, grads_numerical):
        diff = np.linalg.norm(ga - gn)
        rel_error = diff / (np.linalg.norm(ga) + np.linalg.norm(gn))
        print(f"相对误差: {rel_error:.2e}")
```

### 9.2 常见错误

**错误1：忘记保存前向传播的中间值**
```python
# ✗ 错误
def backward(self, dout):
    dx = dout @ self.W.T  # self.x哪来的？

# ✓ 正确
def forward(self, x):
    self.x = x  # 保存
    return x @ self.W + self.b
```

**错误2：维度不匹配**
```python
# 检查所有梯度的形状
print(f"dW shape: {dW.shape}, W shape: {W.shape}")
assert dW.shape == W.shape
```

---

## 10. 练习

1. **手动推导**：
   - 推导三层网络的反向传播公式
   - 计算每一层的梯度

2. **实现练习**：
   - 从零实现一个两层网络
   - 用梯度检查验证正确性

3. **调试练习**：
   - 故意引入错误（如忘记保存中间变量）
   - 用梯度检查定位问题

---

**下一步**：学习 [03_gradient_descent.md](./03_gradient_descent.md)，了解如何用梯度优化神经网络。
