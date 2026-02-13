# 向量和矩阵基础

## 1. 向量 (Vector)

### 1.1 什么是向量？

在深度学习中，向量是最基本的数据结构：

**几何视角**：向量是空间中的一个点或箭头
- 2D向量：`[3, 2]` 表示从原点到点(3,2)的箭头

**代数视角**：向量是一列有序的数字
```python
import numpy as np

# 3维向量
v = np.array([1.0, 2.0, 3.0])
print(v.shape)  # (3,)
```

**深度学习视角**：
- 一张灰度图像：向量 `[batch, height×width]`
- 词嵌入：向量 `[300]` 表示一个单词
- 神经网络输出：向量 `[num_classes]` 表示每类的概率

### 1.2 向量运算

#### (1) 向量加法
```
[1]   [4]   [5]
[2] + [5] = [7]
[3]   [6]   [9]
```

**几何意义**：向量首尾相连

**代码实现**：
```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v3 = v1 + v2  # [5, 7, 9]
```

**深度学习应用**：偏置项加法
```python
output = weights @ input + bias  # bias就是向量加法
```

#### (2) 数乘 (Scalar Multiplication)
```
    [1]   [3]
3 × [2] = [6]
    [3]   [9]
```

**几何意义**：缩放向量长度

**深度学习应用**：学习率调整
```python
gradients = compute_gradients()
weights -= learning_rate * gradients  # 数乘
```

#### (3) 点积 (Dot Product)
```
[1]     [4]
[2] · [5] = 1×4 + 2×5 + 3×6 = 32
[3]     [6]
```

**公式**：
```
v · w = Σ vᵢwᵢ = v₁w₁ + v₂w₂ + ... + vₙwₙ
```

**几何意义**：
- 点积 > 0：两向量夹角 < 90°（同向）
- 点积 = 0：两向量垂直
- 点积 < 0：两向量夹角 > 90°（反向）

**深度学习应用**：神经元计算
```python
# 神经元就是输入向量和权重向量的点积
neuron_output = np.dot(inputs, weights) + bias
```

#### (4) 向量范数 (Norm)

**L2范数**（欧几里得距离）：
```
‖v‖₂ = √(v₁² + v₂² + ... + vₙ²)
```

```python
v = np.array([3, 4])
norm = np.linalg.norm(v)  # 5.0
```

**L1范数**（曼哈顿距离）：
```
‖v‖₁ = |v₁| + |v₂| + ... + |vₙ|
```

**深度学习应用**：
- **L2正则化**：`loss += λ × ‖weights‖₂²`
- **梯度裁剪**：`if ‖gradients‖₂ > threshold: ...`

---

## 2. 矩阵 (Matrix)

### 2.1 什么是矩阵？

**矩阵是向量的集合**：

```
     ┌         ┐
A =  │ 1  2  3 │  ← 第1行（行向量）
     │ 4  5  6 │  ← 第2行
     └         ┘
       ↑  ↑  ↑
      列1 列2 列3
```

形状：`(2, 3)` 表示2行3列

**深度学习中的矩阵**：
```python
# 全连接层的权重
weights = torch.randn(512, 256)  # [输入维度, 输出维度]

# 一批图像
images = torch.randn(32, 3, 224, 224)  # [batch, channels, H, W]
images_flat = images.view(32, -1)  # [32, 150528] 矩阵
```

### 2.2 矩阵基本操作

#### (1) 矩阵转置 (Transpose)

行变列，列变行：

```
     ┌     ┐          ┌     ┐
A =  │ 1 2 │   →  Aᵀ= │ 1 3 │
     │ 3 4 │          │ 2 4 │
     └     ┘          └     ┘
   (2×2)            (2×2)
```

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
A_T = A.T  # 或 np.transpose(A)
```

**性质**：
- `(Aᵀ)ᵀ = A`
- `(A + B)ᵀ = Aᵀ + Bᵀ`
- `(AB)ᵀ = BᵀAᵀ`（注意顺序反转）

**深度学习应用**：反向传播
```python
# 前向传播
output = input @ W  # [batch, in] @ [in, out]

# 反向传播：需要W的转置
grad_input = grad_output @ W.T  # [batch, out] @ [out, in]
```

#### (2) 矩阵乘法 (Matrix Multiplication)

**规则**：
- `A` 的列数必须等于 `B` 的行数
- `(m×n)` @ `(n×p)` = `(m×p)`

**计算方式**：
```
┌     ┐   ┌     ┐   ┌           ┐
│ 1 2 │ @ │ 5 6 │ = │ 1×5+2×7  1×6+2×8 │ = │ 19 22 │
│ 3 4 │   │ 7 8 │   │ 3×5+4×7  3×6+4×8 │   │ 43 50 │
└     ┘   └     ┘   └           ┘
(2×2)     (2×2)         (2×2)
```

**深度学习应用**：前向传播
```python
# 全连接层
def forward(x, W, b):
    """
    x: [batch_size, input_dim]
    W: [input_dim, output_dim]
    b: [output_dim]
    """
    return x @ W + b  # [batch_size, output_dim]
```

#### (3) 逐元素乘法 (Element-wise Multiplication)

也叫 Hadamard 积，符号：`⊙`

```
┌     ┐   ┌     ┐   ┌     ┐
│ 1 2 │ ⊙ │ 5 6 │ = │ 5 12│
│ 3 4 │   │ 7 8 │   │21 32│
└     ┘   └     ┘   └     ┘
```

```python
A * B  # NumPy/PyTorch中的 *
```

**深度学习应用**：激活函数、dropout
```python
# ReLU激活
mask = (x > 0).astype(float)  # 掩码矩阵
output = x * mask  # 逐元素乘法

# Dropout
dropout_mask = (np.random.rand(*x.shape) > 0.5)
output = x * dropout_mask
```

---

## 3. 特殊矩阵

### 3.1 单位矩阵 (Identity Matrix)

对角线为1，其余为0：

```
    ┌       ┐
I = │ 1 0 0 │
    │ 0 1 0 │
    │ 0 0 1 │
    └       ┘
```

**性质**：`A @ I = I @ A = A`（就像数字乘以1）

```python
I = np.eye(3)  # 3×3单位矩阵
```

### 3.2 零矩阵 (Zero Matrix)

全为0的矩阵：

```python
Z = np.zeros((2, 3))
```

### 3.3 对角矩阵 (Diagonal Matrix)

只有对角线有值：

```
    ┌       ┐
D = │ 2 0 0 │
    │ 0 3 0 │
    │ 0 0 5 │
    └       ┘
```

```python
D = np.diag([2, 3, 5])
```

**深度学习应用**：
- **批量归一化 (Batch Norm)**：缩放参数 `γ` 可以用对角矩阵表示
- **权重初始化**：某些初始化方法使用对角矩阵

---

## 4. 矩阵的几何意义

### 4.1 矩阵是线性变换

矩阵乘法 = 对向量进行线性变换

**旋转矩阵**（逆时针旋转θ）：
```
    ┌            ┐
R = │ cos(θ) -sin(θ) │
    │ sin(θ)  cos(θ) │
    └            ┘
```

```python
theta = np.pi / 4  # 45度
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

v = np.array([1, 0])
v_rotated = R @ v  # 旋转后的向量
```

**缩放矩阵**：
```
    ┌     ┐
S = │ 2 0 │  # x方向放大2倍，y方向放大3倍
    │ 0 3 │
    └     ┘
```

### 4.2 神经网络是函数组合

```python
# 两层神经网络
h = x @ W1 + b1      # 第一层：线性变换
h = relu(h)          # 激活：非线性变换
y = h @ W2 + b2      # 第二层：线性变换
```

每一层的权重矩阵都在进行几何变换，激活函数引入非线性。

---

## 5. 实战示例

### 例1：用矩阵表示全连接层

```python
import numpy as np

# 输入：3个样本，每个4维特征
X = np.random.randn(3, 4)

# 权重：4个输入神经元 → 2个输出神经元
W = np.random.randn(4, 2)
b = np.random.randn(2)

# 前向传播
output = X @ W + b  # (3, 4) @ (4, 2) + (2,) = (3, 2)
print(output.shape)  # (3, 2)
```

### 例2：批量处理图像

```python
# 100张28×28灰度图像
images = np.random.randn(100, 28, 28)

# 展平成矩阵：每行是一张图
images_flat = images.reshape(100, -1)  # (100, 784)

# 通过全连接层
W = np.random.randn(784, 10)  # 10类分类
logits = images_flat @ W  # (100, 10)
```

---

## 6. 常见错误

### ❌ 错误1：维度不匹配

```python
A = np.random.randn(3, 4)
B = np.random.randn(3, 5)
C = A @ B  # 错误！4 ≠ 3
```

**解决**：检查维度 `(m, n) @ (n, p)` 中间的 `n` 必须相等

### ❌ 错误2：混淆 `@` 和 `*`

```python
A @ B  # 矩阵乘法
A * B  # 逐元素乘法（Hadamard积）
```

### ❌ 错误3：忘记转置

```python
# 权重形状 [input_dim, output_dim]
W = torch.randn(512, 256)

# 错误：x是 [batch, input_dim]，直接乘会报错
x = torch.randn(32, 512)
out = x @ W  # ✓ 正确

# 如果W存储为 [output_dim, input_dim]，需要转置
W_T = torch.randn(256, 512)
out = x @ W_T.T  # ✓ 正确
```

---

## 7. 速查表

| 操作 | NumPy | PyTorch |
|-----|-------|---------|
| 向量点积 | `np.dot(a, b)` | `torch.dot(a, b)` |
| 矩阵乘法 | `A @ B` 或 `np.matmul(A, B)` | `A @ B` 或 `torch.matmul(A, B)` |
| 逐元素乘 | `A * B` | `A * B` |
| 转置 | `A.T` | `A.T` |
| 范数 | `np.linalg.norm(v)` | `torch.norm(v)` |
| 形状 | `A.shape` | `A.shape` |
| 变形 | `A.reshape(...)` | `A.view(...)` |

---

## 练习

1. **基础练习**：计算两个向量的点积、L2范数
2. **维度练习**：给定形状 `(batch, seq_len, hidden_dim)` 的张量，如何通过矩阵乘法得到 `(batch, seq_len, output_dim)`？
3. **实战练习**：实现一个简单的两层神经网络（只用矩阵运算，不用框架）

---

**下一步**：学习 [02_matrix_operations.md](./02_matrix_operations.md)，深入理解矩阵运算在神经网络中的应用。
