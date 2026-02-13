# 特征值分解与奇异值分解

## 1. 特征值和特征向量

### 1.1 基本概念

对于方阵 `A`，如果存在非零向量 `v` 和标量 `λ` 使得：

```
Av = λv
```

则称：
- `λ` 为**特征值** (eigenvalue)
- `v` 为对应的**特征向量** (eigenvector)

**直观理解**：
- 矩阵 `A` 对向量 `v` 的作用 = 把 `v` 拉伸（或压缩）`λ` 倍
- `v` 的方向不变，只改变长度

### 1.2 计算示例

```python
import numpy as np

A = np.array([[3, 1],
              [0, 2]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值:", eigenvalues)      # [3. 2.]
print("特征向量:\n", eigenvectors)
```

**验证**：
```python
λ1 = eigenvalues[0]  # 3
v1 = eigenvectors[:, 0]  # 第一个特征向量

print(A @ v1)     # [3. 0.] ≈ 3 * [1. 0.]
print(λ1 * v1)    # [3. 0.]
```

### 1.3 几何意义

**例子：旋转+缩放矩阵**

```python
import matplotlib.pyplot as plt

# 矩阵：x方向拉伸2倍，y方向拉伸3倍
A = np.array([[2, 0],
              [0, 3]])

# 特征向量
v1 = np.array([1, 0])  # 特征值2
v2 = np.array([0, 1])  # 特征值3

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 变换前
axes[0].arrow(0, 0, v1[0], v1[1], head_width=0.1, color='red', label='v1')
axes[0].arrow(0, 0, v2[0], v2[1], head_width=0.1, color='blue', label='v2')
axes[0].set_title("原始向量")

# 变换后
v1_transformed = A @ v1
v2_transformed = A @ v2
axes[1].arrow(0, 0, v1_transformed[0], v1_transformed[1], head_width=0.1, color='red')
axes[1].arrow(0, 0, v2_transformed[0], v2_transformed[1], head_width=0.1, color='blue')
axes[1].set_title("变换后（只改变长度）")

plt.show()
```

---

## 2. 特征值分解 (Eigenvalue Decomposition)

### 2.1 矩阵对角化

如果矩阵 `A` 有 `n` 个线性独立的特征向量，可以分解为：

```
A = QΛQᵀ
```

其中：
- `Q`：特征向量矩阵（列是特征向量）
- `Λ`：对角矩阵（对角线是特征值）
- `Qᵀ`：`Q` 的转置（如果 `Q` 是正交矩阵）

**示例**：
```python
A = np.array([[4, -2],
              [-2, 1]])

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(A)

# 构造Λ和Q
Lambda = np.diag(eigenvalues)
Q = eigenvectors

# 验证：A = QΛQᵀ
A_reconstructed = Q @ Lambda @ Q.T
print(np.allclose(A, A_reconstructed))  # True
```

### 2.2 对称矩阵的特殊性质

**对称矩阵** (`A = Aᵀ`) 有特殊性质：
- 所有特征值都是实数
- 不同特征值对应的特征向量正交
- 一定可以对角化

```python
# 对称矩阵
A_sym = np.array([[3, 1],
                  [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A_sym)
Q = eigenvectors

# 验证正交性：QᵀQ = I
print(Q.T @ Q)  # 接近单位矩阵
```

### 2.3 深度学习应用

#### (1) 主成分分析 (PCA)

**思想**：找到数据方差最大的方向

```python
def pca_manual(X, n_components):
    """
    X: [n_samples, n_features]
    """
    # 1. 中心化
    X_centered = X - X.mean(axis=0)

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_centered.T)  # [n_features, n_features]

    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. 按特征值大小排序
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. 选择前n_components个主成分
    principal_components = eigenvectors[:, :n_components]

    # 6. 投影
    X_pca = X_centered @ principal_components

    return X_pca, principal_components, eigenvalues

# 使用示例
X = np.random.randn(100, 50)  # 100个样本，50维
X_pca, components, explained_var = pca_manual(X, n_components=2)

print(f"原始形状: {X.shape}")        # (100, 50)
print(f"降维后: {X_pca.shape}")     # (100, 2)
print(f"解释方差比: {explained_var[:2] / explained_var.sum()}")
```

#### (2) 协方差矩阵分析

```python
# 理解数据的相关性
data = np.random.randn(1000, 3)
data[:, 1] = data[:, 0] + np.random.randn(1000) * 0.1  # 强相关

cov = np.cov(data.T)
eigenvalues, _ = np.linalg.eig(cov)

print("特征值:", eigenvalues)
# 最大特征值 >> 其他特征值，说明存在主方向
```

#### (3) 图神经网络的谱方法

**拉普拉斯矩阵的特征值分解**：
```python
def spectral_graph_convolution(A, X):
    """
    A: 邻接矩阵 [n_nodes, n_nodes]
    X: 节点特征 [n_nodes, n_features]
    """
    # 计算拉普拉斯矩阵
    D = np.diag(A.sum(axis=1))
    L = D - A

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(L)

    # 在谱域进行卷积
    # ...
    pass
```

---

## 3. 奇异值分解 (SVD)

### 3.1 SVD定义

对于**任意**矩阵 `A` (m×n)，都可以分解为：

```
A = UΣVᵀ
```

其中：
- `U`：左奇异向量矩阵 `(m×m)`，列是正交的
- `Σ`：奇异值对角矩阵 `(m×n)`，对角线非负且递减
- `V`：右奇异向量矩阵 `(n×n)`，列是正交的

**关键**：SVD适用于非方阵！

### 3.2 计算SVD

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # (2, 3)矩阵

U, s, VT = np.linalg.svd(A)

print(f"U shape: {U.shape}")     # (2, 2)
print(f"s shape: {s.shape}")     # (2,) - 奇异值
print(f"VT shape: {VT.shape}")   # (3, 3)

# 重建Σ矩阵
Sigma = np.zeros((2, 3))
Sigma[:2, :2] = np.diag(s)

# 验证：A = UΣVᵀ
A_reconstructed = U @ Sigma @ VT
print(np.allclose(A, A_reconstructed))  # True
```

### 3.3 SVD的几何意义

**SVD = 三步线性变换的组合**：

```
A @ v = U @ Σ @ Vᵀ @ v
        └─┬─┘   └┬┘   └┬┘
         3:旋转  2:缩放  1:旋转
```

1. `Vᵀ`：旋转输入空间
2. `Σ`：沿主轴缩放（拉伸/压缩）
3. `U`：旋转到输出空间

### 3.4 低秩近似

**核心思想**：保留最大的 `k` 个奇异值

```python
def low_rank_approximation(A, k):
    """
    用前k个奇异值近似矩阵A
    """
    U, s, VT = np.linalg.svd(A, full_matrices=False)

    # 只保留前k个
    U_k = U[:, :k]
    s_k = s[:k]
    VT_k = VT[:k, :]

    # 近似矩阵
    A_approx = U_k @ np.diag(s_k) @ VT_k

    return A_approx

# 示例
A = np.random.randn(100, 50)
A_approx = low_rank_approximation(A, k=10)

print(f"原始: {A.shape}, 秩 ≈ {np.linalg.matrix_rank(A)}")
print(f"近似: {A_approx.shape}, 秩 = {np.linalg.matrix_rank(A_approx)}")

# 计算重建误差
error = np.linalg.norm(A - A_approx, 'fro')
print(f"Frobenius范数误差: {error:.4f}")
```

---

## 4. SVD在深度学习中的应用

### 4.1 图像压缩

```python
from PIL import Image

# 加载灰度图像
img = np.array(Image.open('image.jpg').convert('L'))
print(f"原始图像: {img.shape}")  # (512, 512)

# SVD分解
U, s, VT = np.linalg.svd(img, full_matrices=False)

# 不同压缩率
for k in [10, 50, 100]:
    img_compressed = U[:, :k] @ np.diag(s[:k]) @ VT[:k, :]

    # 压缩比
    original_size = img.shape[0] * img.shape[1]
    compressed_size = k * (img.shape[0] + img.shape[1] + 1)
    ratio = compressed_size / original_size

    print(f"k={k}: 压缩比={ratio:.2%}, "
          f"误差={(np.linalg.norm(img - img_compressed)):.2f}")
```

### 4.2 推荐系统（协同过滤）

```python
# 用户-物品矩阵
# ratings: [n_users, n_items]
ratings = np.random.randint(0, 6, size=(1000, 500))

# SVD分解
U, s, VT = np.linalg.svd(ratings, full_matrices=False)

# 潜在因子表示
k = 50  # 潜在维度
user_factors = U[:, :k] @ np.diag(s[:k])  # [n_users, k]
item_factors = VT[:k, :].T                 # [n_items, k]

# 预测评分
def predict_rating(user_id, item_id):
    return user_factors[user_id] @ item_factors[item_id]

# 推荐：找相似物品
def recommend(item_id, top_k=10):
    item_vec = item_factors[item_id]
    similarities = item_factors @ item_vec  # 余弦相似度

    # 排序
    top_items = np.argsort(similarities)[::-1][1:top_k+1]
    return top_items
```

### 4.3 模型压缩

**压缩全连接层权重**：

```python
class CompressedLinear:
    def __init__(self, weight_matrix, rank):
        """
        将权重矩阵 W (in, out) 分解为 U (in, rank) @ V (rank, out)
        """
        U, s, VT = np.linalg.svd(weight_matrix, full_matrices=False)

        # 低秩分解
        self.U = U[:, :rank] @ np.diag(np.sqrt(s[:rank]))  # (in, rank)
        self.V = np.diag(np.sqrt(s[:rank])) @ VT[:rank, :]  # (rank, out)

    def forward(self, x):
        """
        x: [batch, in]
        返回: [batch, out]
        """
        # 原始: x @ W = x @ (U @ V)
        # 分解: x @ U @ V （两次较小的矩阵乘法）
        return x @ self.U @ self.V

# 示例
W = np.random.randn(512, 256)  # 131,072参数
compressed = CompressedLinear(W, rank=64)

print(f"原始参数: {512 * 256} = {512 * 256}")
print(f"压缩参数: {512 * 64 + 64 * 256} = {512 * 64 + 64 * 256}")
print(f"压缩比: {(512*64 + 64*256) / (512*256):.2%}")
```

### 4.4 数据去噪

```python
def denoise_via_svd(X, noise_threshold=0.01):
    """
    X: [n_samples, n_features] - 带噪声的数据
    """
    U, s, VT = np.linalg.svd(X, full_matrices=False)

    # 去除小奇异值（认为是噪声）
    s_denoised = s.copy()
    s_denoised[s < noise_threshold] = 0

    # 重建
    X_denoised = U @ np.diag(s_denoised) @ VT

    return X_denoised

# 测试
X_clean = np.random.randn(100, 50)
X_noisy = X_clean + np.random.randn(100, 50) * 0.1

X_denoised = denoise_via_svd(X_noisy, noise_threshold=0.5)

print(f"噪声误差: {np.linalg.norm(X_clean - X_noisy):.4f}")
print(f"去噪后误差: {np.linalg.norm(X_clean - X_denoised):.4f}")
```

---

## 5. 特征值分解 vs SVD

| 特性 | 特征值分解 | SVD |
|-----|-----------|-----|
| 适用矩阵 | 方阵 | 任意矩阵 |
| 分解形式 | `A = QΛQᵀ` | `A = UΣVᵀ` |
| 向量正交性 | 仅对称矩阵保证 | 总是正交 |
| 数值稳定性 | 较差 | 较好 |
| 深度学习应用 | PCA、图谱方法 | 降维、压缩、推荐 |

**如何选择**：
- 协方差矩阵（对称）→ 特征值分解
- 一般矩阵、需要数值稳定性 → SVD

---

## 6. 实战：从零实现PCA

```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        """
        X: [n_samples, n_features]
        """
        # 1. 中心化
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # 2. 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)

        # 3. 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 4. 排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. 保存主成分
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, X):
        """投影到主成分空间"""
        X_centered = X - self.mean
        return X_centered @ self.components

    def inverse_transform(self, X_pca):
        """从主成分空间重建"""
        return X_pca @ self.components.T + self.mean

# 测试
X = np.random.randn(200, 100)  # 200样本，100维

pca = PCA(n_components=10)
pca.fit(X)

X_reduced = pca.transform(X)  # [200, 10]
X_reconstructed = pca.inverse_transform(X_reduced)  # [200, 100]

print(f"降维: {X.shape} → {X_reduced.shape}")
print(f"重建误差: {np.linalg.norm(X - X_reconstructed):.4f}")
print(f"解释方差比: {pca.explained_variance.sum() / np.trace(np.cov(X.T)):.2%}")
```

---

## 7. 高级主题：截断SVD (Truncated SVD)

对于超大矩阵，完整SVD太慢。使用截断SVD只计算前 `k` 个奇异值：

```python
from scipy.sparse.linalg import svds

# 大矩阵
A = np.random.randn(10000, 5000)

# 只计算前50个奇异值
k = 50
U, s, VT = svds(A, k=k)

print(f"U: {U.shape}, s: {s.shape}, VT: {VT.shape}")
# U: (10000, 50), s: (50,), VT: (50, 5000)

# 近似重建
A_approx = U @ np.diag(s) @ VT
```

**应用**：文本挖掘中的LSA（潜在语义分析）

---

## 8. 练习

1. **PCA降维**：对MNIST数据集（784维）进行PCA，可视化前2个主成分
2. **图像压缩**：用SVD压缩一张彩色图像（分别对RGB三通道）
3. **推荐系统**：用MovieLens数据集实现基于SVD的协同过滤
4. **模型压缩**：压缩一个预训练模型的全连接层，对比精度损失

---

## 总结

| 方法 | 公式 | 应用 |
|-----|------|-----|
| 特征值分解 | `A = QΛQᵀ` | PCA、图神经网络 |
| SVD | `A = UΣVᵀ` | 降维、压缩、推荐 |
| 低秩近似 | `A ≈ U_k Σ_k V_k^T` | 模型压缩、去噪 |

**关键要点**：
- 特征值分解揭示矩阵的"主方向"
- SVD是更通用、更稳定的分解方法
- 两者在深度学习中广泛用于降维和压缩

---

**恭喜！** 你已经完成线性代数模块。接下来前往 [02_calculus](../../02_calculus/) 学习微积分，理解反向传播的数学基础。
