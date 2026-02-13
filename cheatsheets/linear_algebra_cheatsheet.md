# 线性代数速查表

## 基本概念

### 向量
```
向量: v = [v₁, v₂, ..., vₙ]ᵀ
模长: ||v|| = √(v₁² + v₂² + ... + vₙ²)
单位向量: v̂ = v / ||v||
```

### 矩阵
```
矩阵: A ∈ ℝᵐˣⁿ (m行n列)
转置: Aᵀ[i,j] = A[j,i]
```

## 基本运算

### 向量运算
```python
# 加法
u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]

# 数乘
α·v = [α·v₁, α·v₂, ..., α·vₙ]

# 点积（内积）
u·v = u₁v₁ + u₂v₂ + ... + uₙvₙ = uᵀv

# 叉积（仅3D）
u × v = [u₂v₃-u₃v₂, u₃v₁-u₁v₃, u₁v₂-u₂v₁]
```

### 矩阵运算
```python
# 矩阵乘法
C = AB, where C[i,j] = Σₖ A[i,k] B[k,j]
要求: A的列数 = B的行数
结果维度: (m×k) @ (k×n) → (m×n)

# 逐元素乘法（Hadamard积）
C = A ⊙ B, where C[i,j] = A[i,j] × B[i,j]

# 转置性质
(AB)ᵀ = BᵀAᵀ
(Aᵀ)ᵀ = A
```

## 特殊矩阵

```python
# 单位矩阵
I = diag(1,1,...,1)
AI = IA = A

# 对角矩阵
D = diag(d₁, d₂, ..., dₙ)
D[i,j] = 0 if i≠j

# 对称矩阵
Aᵀ = A

# 正交矩阵
QᵀQ = QQᵀ = I
```

## 逆矩阵

```python
# 定义
AA⁻¹ = A⁻¹A = I

# 性质
(AB)⁻¹ = B⁻¹A⁻¹
(Aᵀ)⁻¹ = (A⁻¹)ᵀ

# 2×2矩阵逆
A = [a b]    A⁻¹ = 1/(ad-bc) [ d -b]
    [c d]                     [-c  a]
```

## 特征值和特征向量

```python
# 定义
Av = λv
其中λ是特征值，v是特征向量

# 特征方程
det(A - λI) = 0

# 性质
tr(A) = Σλᵢ  (迹 = 特征值之和)
det(A) = Πλᵢ  (行列式 = 特征值之积)
```

## 奇异值分解 (SVD)

```python
A = UΣVᵀ
其中:
- U: m×m 正交矩阵（左奇异向量）
- Σ: m×n 对角矩阵（奇异值）
- V: n×n 正交矩阵（右奇异向量）

应用: PCA降维、推荐系统、图像压缩
```

## 范数

```python
# L1范数（曼哈顿距离）
||v||₁ = |v₁| + |v₂| + ... + |vₙ|

# L2范数（欧几里得距离）
||v||₂ = √(v₁² + v₂² + ... + vₙ²)

# L∞范数（最大值范数）
||v||∞ = max(|v₁|, |v₂|, ..., |vₙ|)

# Frobenius范数（矩阵）
||A||_F = √(Σᵢⱼ A[i,j]²)
```

## NumPy实现

```python
import numpy as np

# 创建
v = np.array([1, 2, 3])
A = np.array([[1, 2], [3, 4]])

# 基本运算
np.dot(u, v)           # 向量点积
A @ B                  # 矩阵乘法
A.T                    # 转置
np.linalg.inv(A)       # 逆矩阵

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD
U, S, Vt = np.linalg.svd(A)

# 范数
np.linalg.norm(v, ord=1)    # L1
np.linalg.norm(v, ord=2)    # L2
np.linalg.norm(v, ord=np.inf)  # L∞
```

## 深度学习中的应用

| 概念 | 应用 | 代码示例 |
|-----|------|---------|
| 矩阵乘法 | 全连接层 | `output = input @ weight` |
| 转置 | 反向传播 | `grad_input = grad_output @ weight.T` |
| 特征值分解 | PCA降维 | `np.linalg.eig(cov_matrix)` |
| SVD | 推荐系统 | `U, S, Vt = np.linalg.svd(rating_matrix)` |
| L2范数 | 权重正则化 | `loss + lambda * ||weights||²` |
| 单位矩阵 | 残差连接 | `output = F(x) + x` |

## 常见维度计算

```python
# 全连接层
input: [batch_size, in_features]
weight: [in_features, out_features]
output: [batch_size, out_features]

# 批量矩阵乘法
A: [batch, m, k]
B: [batch, k, n]
C = A @ B: [batch, m, n]

# 注意力机制
Q: [batch, seq_len, d_k]
K: [batch, seq_len, d_k]
V: [batch, seq_len, d_v]
Attention = softmax(QKᵀ/√d_k) V
输出: [batch, seq_len, d_v]
```

## 技巧和陷阱

### ✅ 最佳实践
- 矩阵乘法前检查维度兼容性
- 使用批量操作提高效率
- 注意数值稳定性（避免除以很小的数）

### ⚠️ 常见错误
- 混淆`@`（矩阵乘法）和`*`（逐元素乘法）
- 忘记转置
- 维度不匹配
- 忘记矩阵乘法不满足交换律：AB ≠ BA
