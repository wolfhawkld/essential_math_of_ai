# 微积分速查表

## 导数基础

### 定义
```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```
直觉：函数在某点的变化率（斜率）

### 基本导数公式

```python
# 常数
d/dx (c) = 0

# 幂函数
d/dx (xⁿ) = n·xⁿ⁻¹

# 指数函数
d/dx (eˣ) = eˣ
d/dx (aˣ) = aˣ ln(a)

# 对数函数
d/dx (ln x) = 1/x
d/dx (log_a x) = 1/(x ln a)

# 三角函数
d/dx (sin x) = cos x
d/dx (cos x) = -sin x
d/dx (tan x) = sec²x = 1/cos²x
```

## 求导法则

```python
# 加减法则
(f ± g)' = f' ± g'

# 乘法法则
(f·g)' = f'·g + f·g'

# 除法法则
(f/g)' = (f'·g - f·g') / g²

# 链式法则（复合函数）
d/dx f(g(x)) = f'(g(x)) · g'(x)
```

## 常见激活函数导数

```python
# Sigmoid
σ(x) = 1 / (1 + e⁻ˣ)
σ'(x) = σ(x) · (1 - σ(x))

# Tanh
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
tanh'(x) = 1 - tanh²(x)

# ReLU
ReLU(x) = max(0, x)
ReLU'(x) = { 1 if x > 0
           { 0 if x ≤ 0

# Leaky ReLU
LeakyReLU(x) = { x      if x > 0
                { α·x    if x ≤ 0
LeakyReLU'(x) = { 1      if x > 0
                 { α      if x ≤ 0

# Softmax (针对单个输出)
softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ
∂softmax(xᵢ)/∂xⱼ = softmax(xᵢ)·(δᵢⱼ - softmax(xⱼ))
```

## 偏导数

### 定义
```
∂f/∂x = lim[h→0] (f(x+h, y) - f(x, y)) / h
```
对x求偏导时，把其他变量当作常数

### 示例
```python
f(x, y) = x²y + 3xy²

∂f/∂x = 2xy + 3y²
∂f/∂y = x² + 6xy
```

## 梯度

### 定义
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```
梯度指向函数增长最快的方向

### 性质
```python
# 方向导数
D_v f = ∇f · v̂  (v̂是单位向量)

# 最大增长方向
∇f 方向是函数增长最快的方向
||∇f|| 是最大增长率
```

## 链式法则（多元）

### 单变量链式法则
```
z = f(y), y = g(x)
dz/dx = (dz/dy) · (dy/dx)
```

### 多变量链式法则
```
z = f(x, y), x = g(t), y = h(t)
dz/dt = (∂z/∂x)·(dx/dt) + (∂z/∂y)·(dy/dt)
```

### 向量形式
```
y = f(x), x = g(w)
dy/dw = (dy/dx) · (dx/dw)

矩阵维度:
[n×1] = [n×m] @ [m×1]
```

## 反向传播公式

### 简单网络
```
输入 x → 隐藏层 h → 输出 ŷ

h = σ(W₁x + b₁)
ŷ = W₂h + b₂
L = loss(ŷ, y)

# 梯度计算（链式法则）
∂L/∂W₂ = ∂L/∂ŷ · ∂ŷ/∂W₂ = (ŷ - y) · hᵀ
∂L/∂b₂ = ∂L/∂ŷ = (ŷ - y)
∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂h · ∂h/∂W₁
       = (ŷ - y) · W₂ᵀ · σ'(W₁x + b₁) · xᵀ
```

### 通用反向传播
```
前向: z^(l) = W^(l) a^(l-1) + b^(l)
      a^(l) = σ(z^(l))

反向: δ^(l) = (W^(l+1))ᵀ δ^(l+1) ⊙ σ'(z^(l))
      ∂L/∂W^(l) = δ^(l) (a^(l-1))ᵀ
      ∂L/∂b^(l) = δ^(l)
```

## 梯度下降

### 基本梯度下降
```python
θ = θ - η · ∇L(θ)

其中:
θ: 参数
η: 学习率
∇L(θ): 损失对参数的梯度
```

### 批量梯度下降（BGD）
```python
# 使用全部数据计算梯度
θ = θ - η · (1/N) Σᵢ ∇L(θ; xᵢ, yᵢ)
```

### 随机梯度下降（SGD）
```python
# 每次用一个样本
θ = θ - η · ∇L(θ; xᵢ, yᵢ)
```

### Mini-batch SGD
```python
# 使用小批量数据
θ = θ - η · (1/B) Σᵢ∈batch ∇L(θ; xᵢ, yᵢ)
```

## 常见损失函数的梯度

```python
# MSE (均方误差)
L = (1/2)(ŷ - y)²
∂L/∂ŷ = ŷ - y

# Binary Cross-Entropy
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
∂L/∂ŷ = (ŷ - y) / (ŷ(1-ŷ))

# Cross-Entropy + Softmax
L = -Σᵢ yᵢ log(ŷᵢ)
∂L/∂zᵢ = ŷᵢ - yᵢ  (z是softmax的输入)
```

## 数值计算技巧

### 梯度检验（Gradient Check）
```python
# 数值梯度
∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)

# 检查反向传播实现
numerical_grad ≈ analytical_grad
相对误差 < 1e-7 即可
```

### 避免数值不稳定
```python
# Softmax数值稳定版
softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))

# Log-sum-exp技巧
log(Σ exp(xᵢ)) = c + log(Σ exp(xᵢ - c))
其中 c = max(xᵢ)
```

## PyTorch/NumPy实现

```python
import numpy as np

# 手动计算梯度
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

# PyTorch自动微分
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # [2.0, 4.0, 6.0]
```

## 深度学习中的应用

| 概念 | 应用 | 位置 |
|-----|------|------|
| 偏导数 | 计算每个参数的梯度 | 反向传播 |
| 链式法则 | 多层网络的梯度传播 | `loss.backward()` |
| 梯度 | 参数更新方向 | `optimizer.step()` |
| 梯度下降 | 训练算法 | 每次迭代 |

## 常见问题

### 梯度消失
```
原因: 链式法则多次相乘小于1的梯度
解决:
- ReLU替代Sigmoid
- Batch Normalization
- 残差连接
- 更好的权重初始化
```

### 梯度爆炸
```
原因: 梯度过大导致参数更新幅度太大
解决:
- 梯度裁剪: clip_grad_norm_()
- 降低学习率
- Batch Normalization
```

### 学习率选择
```
太大: 震荡或发散
太小: 收敛太慢

策略:
- 学习率预热 (warmup)
- 学习率衰减 (decay)
- 自适应学习率 (Adam)
```

## 技巧总结

✅ **最佳实践**
- 用自动微分框架（PyTorch, TensorFlow）
- 用梯度检验验证手动实现
- 注意数值稳定性

⚠️ **常见错误**
- 忘记清零梯度 (`optimizer.zero_grad()`)
- 在验证时没有关闭梯度计算 (`with torch.no_grad()`)
- 忽略in-place操作对梯度的影响
