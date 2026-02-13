# 矩阵运算与神经网络

## 1. 矩阵求逆 (Matrix Inverse)

### 1.1 什么是逆矩阵？

对于方阵 `A`（行数=列数），如果存在矩阵 `A⁻¹` 使得：

```
A @ A⁻¹ = A⁻¹ @ A = I
```

则称 `A⁻¹` 为 `A` 的逆矩阵。

**例子**：
```python
import numpy as np

A = np.array([[4, 7],
              [2, 6]])

A_inv = np.linalg.inv(A)

# 验证
print(A @ A_inv)  # 近似于单位矩阵 I
# [[1. 0.]
#  [0. 1.]]
```

### 1.2 何时矩阵可逆？

**可逆条件**：
- 必须是方阵（n×n）
- 行列式 `det(A) ≠ 0`
- 所有行（列）线性独立

**不可逆矩阵**（奇异矩阵）：
```python
B = np.array([[1, 2],
              [2, 4]])  # 第二行是第一行的2倍

# np.linalg.inv(B)  # 会报错：LinAlgError
```

### 1.3 深度学习中的应用

#### (1) 最小二乘法

求解线性回归 `y = Xw`：

```
w = (XᵀX)⁻¹ Xᵀy
```

```python
# 最小二乘法求解
X = np.random.randn(100, 5)  # 100个样本，5个特征
y = np.random.randn(100)

w = np.linalg.inv(X.T @ X) @ X.T @ y
```

⚠️ **实践中不建议直接求逆**：
- 计算复杂度高：O(n³)
- 数值不稳定

更好的方法：
```python
# 使用伪逆（更稳定）
w = np.linalg.pinv(X) @ y

# 或使用梯度下降（深度学习常用）
```

#### (2) 协方差矩阵逆

在一些优化算法中（如自然梯度）：
```python
# Fisher信息矩阵的逆
natural_gradient = np.linalg.inv(fisher_matrix) @ gradient
```

---

## 2. 矩阵的秩 (Matrix Rank)

### 2.1 什么是秩？

**秩 = 线性独立的行（列）的最大数目**

```python
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [1, 1, 1]])

rank = np.linalg.matrix_rank(A)  # 2（第二行是第一行的2倍）
```

**满秩矩阵**：秩等于行数和列数的较小值
- `(3, 3)` 矩阵满秩 → rank = 3
- `(5, 3)` 矩阵满秩 → rank = 3

### 2.2 深度学习中的意义

#### (1) 网络表达能力

如果权重矩阵秩很低：
```python
W = np.random.randn(512, 512)
# 如果 rank(W) << 512，说明这一层的表达能力受限
```

**低秩分解**（压缩模型）：
```
W ≈ U @ V
(512, 512) ≈ (512, 64) @ (64, 512)
```

#### (2) 数据秩与过拟合

```python
# 如果输入数据X的秩很低，可能有冗余特征
X = np.random.randn(1000, 100)
print(np.linalg.matrix_rank(X))  # 如果 << 100，说明存在共线性
```

---

## 3. 神经网络中的矩阵运算

### 3.1 全连接层 (Fully Connected Layer)

**前向传播**：
```python
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        """
        x: [batch_size, in_features]
        返回: [batch_size, out_features]
        """
        return x @ self.W + self.b
```

**反向传播**：
```python
def backward(self, x, grad_output):
    """
    x: [batch, in_features]
    grad_output: [batch, out_features] - 来自上一层的梯度

    返回:
    grad_x: [batch, in_features] - 传给下一层的梯度
    grad_W: [in_features, out_features] - 权重梯度
    grad_b: [out_features] - 偏置梯度
    """
    # 梯度计算都是矩阵运算
    grad_W = x.T @ grad_output  # (in, batch) @ (batch, out) = (in, out)
    grad_b = np.sum(grad_output, axis=0)  # (batch, out) -> (out,)
    grad_x = grad_output @ self.W.T  # (batch, out) @ (out, in) = (batch, in)

    return grad_x, grad_W, grad_b
```

**维度分析**：
```
前向: [batch, in] @ [in, out] → [batch, out]
反向: [batch, out] @ [out, in] → [batch, in]
```

### 3.2 批量矩阵乘法 (Batch Matrix Multiplication)

**场景**：Transformer中的注意力机制

```python
# Q, K, V: [batch, num_heads, seq_len, head_dim]

# 计算注意力分数
scores = Q @ K.transpose(-2, -1)  # [batch, num_heads, seq_len, seq_len]

# 加权求和
output = scores @ V  # [batch, num_heads, seq_len, head_dim]
```

**PyTorch实现**：
```python
import torch

Q = torch.randn(32, 8, 100, 64)  # [batch, heads, seq, dim]
K = torch.randn(32, 8, 100, 64)

# bmm只支持3D，einsum更灵活
scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)  # 爱因斯坦求和约定
```

### 3.3 卷积的矩阵视角

卷积可以用矩阵乘法实现（虽然效率低）：

**im2col 方法**：
```python
def conv_as_matmul(image, kernel):
    """
    将卷积转换为矩阵乘法

    image: [H, W]
    kernel: [K, K]
    """
    # 1. 将图像展开成列矩阵（im2col）
    # 每一列是一个感受野

    # 2. 将卷积核展平成行向量
    kernel_flat = kernel.reshape(1, -1)

    # 3. 矩阵乘法
    # output = kernel_flat @ image_cols
    pass
```

实际上卷积库（cuDNN）内部用了这个技巧来利用高度优化的矩阵乘法库。

---

## 4. 注意力机制中的矩阵运算

### 4.1 自注意力 (Self-Attention)

**公式**：
```
Attention(Q, K, V) = softmax(QKᵀ/√dₖ) V
```

**实现**：
```python
def self_attention(Q, K, V):
    """
    Q, K, V: [batch, seq_len, d_model]
    """
    d_k = Q.shape[-1]

    # 计算注意力分数
    scores = Q @ K.T / np.sqrt(d_k)  # [batch, seq_len, seq_len]

    # Softmax归一化
    attn_weights = softmax(scores, axis=-1)

    # 加权求和
    output = attn_weights @ V  # [batch, seq_len, d_model]

    return output, attn_weights
```

**可视化维度变化**：
```
Q: [32, 100, 512]  (batch=32, seq=100, dim=512)
K: [32, 100, 512]
V: [32, 100, 512]

QKᵀ: [32, 100, 512] @ [32, 512, 100] → [32, 100, 100]
     ↑ 每个query与所有key的相似度

Attention @ V: [32, 100, 100] @ [32, 100, 512] → [32, 100, 512]
               ↑ 根据注意力权重聚合所有value
```

### 4.2 多头注意力 (Multi-Head Attention)

**原理**：将Q, K, V投影到多个子空间

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 投影矩阵
        self.W_Q = np.random.randn(d_model, d_model)
        self.W_K = np.random.randn(d_model, d_model)
        self.W_V = np.random.randn(d_model, d_model)
        self.W_O = np.random.randn(d_model, d_model)

    def forward(self, Q, K, V):
        """
        Q, K, V: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = Q.shape

        # 线性投影
        Q = Q @ self.W_Q  # [batch, seq, d_model]
        K = K @ self.W_K
        V = V @ self.W_V

        # 分割成多头: [batch, seq, d_model] → [batch, heads, seq, d_k]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # 计算注意力（每个头独立）
        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)
        attn = softmax(scores, axis=-1)
        output = attn @ V  # [batch, heads, seq, d_k]

        # 合并多头
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

        # 输出投影
        output = output @ self.W_O

        return output
```

---

## 5. 高效矩阵运算技巧

### 5.1 避免不必要的转置

**❌ 低效写法**：
```python
# 反复转置
for i in range(1000):
    result = A.T @ B.T @ C.T
```

**✓ 高效写法**：
```python
# 使用矩阵乘法的结合律: (ABC)ᵀ = CᵀBᵀAᵀ
for i in range(1000):
    result = (C @ B @ A).T  # 只转置一次
```

### 5.2 利用广播 (Broadcasting)

**场景**：给每个样本加偏置

```python
# ❌ 低效：显式循环
for i in range(batch_size):
    output[i] = input[i] @ W + b

# ✓ 高效：广播
output = input @ W + b  # b自动广播到 [batch, out_features]
```

### 5.3 原地操作 (In-place Operations)

**PyTorch示例**：
```python
# 创建新张量
x = x + 1

# 原地修改（节省内存）
x += 1  # 或 x.add_(1)
```

⚠️ **注意**：反向传播时需要保留中间结果，不能随意原地操作

### 5.4 使用专门的BLAS库

**NumPy自动调用**：
- Intel MKL
- OpenBLAS
- ATLAS

**检查当前使用的库**：
```python
import numpy as np
np.show_config()
```

---

## 6. 调试矩阵维度

### 6.1 常见维度错误

**错误1：形状不匹配**
```python
# 错误
x = torch.randn(32, 128)  # [batch, features]
W = torch.randn(256, 64)  # [in, out]
y = x @ W  # RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**修复**：检查中间维度
```python
# 方案1：修正权重形状
W = torch.randn(128, 64)  # [128, 64]
y = x @ W  # ✓ [32, 128] @ [128, 64] = [32, 64]

# 方案2：转置权重
W = torch.randn(64, 128)
y = x @ W.T  # ✓ [32, 128] @ [128, 64] = [32, 64]
```

**错误2：批量维度丢失**
```python
# 错误
x = torch.randn(32, 10, 128)  # [batch, seq, dim]
W = torch.randn(128, 64)
y = x @ W  # ✓ 可以运行，但可能不是你想要的

# 注意：PyTorch会广播
# [32, 10, 128] @ [128, 64] → [32, 10, 64]
# 对最后两个维度做矩阵乘法
```

### 6.2 调试技巧

**技巧1：打印形状**
```python
def debug_shapes(name, tensor):
    print(f"{name}: {tensor.shape}")

x = torch.randn(32, 128)
debug_shapes("x", x)

W = torch.randn(128, 64)
debug_shapes("W", W)

y = x @ W
debug_shapes("y", y)
```

**技巧2：使用einsum（更清晰）**
```python
# 传统写法（容易出错）
output = input @ W1 @ W2.T

# einsum写法（显式指定维度）
output = torch.einsum('bi,ij,kj->bk', input, W1, W2)
#                      ↑   ↑   ↑    ↑
#                      batch × in × out → batch × out
```

---

## 7. 实战案例

### 案例1：实现简单的MLP

```python
class MLP:
    def __init__(self, layers):
        """
        layers: [input_dim, hidden_dim1, hidden_dim2, ..., output_dim]
        """
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            W = np.random.randn(layers[i], layers[i+1]) * 0.01
            b = np.zeros(layers[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        """
        x: [batch, input_dim]
        """
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(0, x @ W + b)  # ReLU

        # 最后一层不加激活
        x = x @ self.weights[-1] + self.biases[-1]
        return x

# 使用
mlp = MLP([784, 256, 128, 10])
x = np.random.randn(32, 784)  # 32张28×28图像
output = mlp.forward(x)  # [32, 10]
```

### 案例2：批量处理序列数据

```python
def batch_process_sequences(sequences, W):
    """
    sequences: List[np.array], 每个shape为 [seq_len_i, d_model]
    W: [d_model, d_out]

    返回: List[np.array], 每个shape为 [seq_len_i, d_out]
    """
    # 方法1：循环（慢）
    outputs = [seq @ W for seq in sequences]

    # 方法2：padding + 批量处理（快）
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    d_model = sequences[0].shape[1]

    # Padding
    padded = np.zeros((batch_size, max_len, d_model))
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    # 批量计算
    output = padded @ W  # [batch, max_len, d_out]

    # 去除padding
    outputs = [output[i, :len(seq)] for i, seq in enumerate(sequences)]

    return outputs
```

---

## 总结

| 概念 | 公式 | 深度学习应用 |
|-----|------|------------|
| 矩阵乘法 | `A @ B` | 全连接层前向传播 |
| 转置 | `Aᵀ` | 反向传播梯度计算 |
| 逆矩阵 | `A⁻¹` | 最小二乘、自然梯度 |
| 矩阵秩 | `rank(A)` | 模型压缩、数据分析 |
| 批量乘法 | `bmm` | Transformer注意力 |

---

**下一步**：学习 [03_eigenvalue_svd.md](./03_eigenvalue_svd.md)，了解特征值分解和SVD在降维和PCA中的应用。
