"""
线性代数工具函数

常用矩阵操作的实现，用于深度学习中的数学计算。
"""

import numpy as np
from typing import Tuple, Optional


# ==================== 基础矩阵运算 ====================

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    矩阵乘法（教学用，实际请用NumPy的@）

    Args:
        A: shape (m, n)
        B: shape (n, p)

    Returns:
        C: shape (m, p)
    """
    m, n = A.shape
    n2, p = B.shape

    assert n == n2, f"形状不匹配: ({m}, {n}) @ ({n2}, {p})"

    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    return C


def batch_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    批量矩阵乘法

    Args:
        A: shape (batch, m, n)
        B: shape (batch, n, p)

    Returns:
        C: shape (batch, m, p)
    """
    return np.einsum('bmn,bnp->bmp', A, B)


# ==================== 向量运算 ====================

def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """向量点积"""
    return np.dot(v1, v2)


def vector_norm(v: np.ndarray, ord: int = 2) -> float:
    """
    向量范数

    Args:
        v: 向量
        ord: 范数类型（1, 2, np.inf）

    Returns:
        范数值
    """
    if ord == 1:
        return np.sum(np.abs(v))
    elif ord == 2:
        return np.sqrt(np.sum(v ** 2))
    elif ord == np.inf:
        return np.max(np.abs(v))
    else:
        raise ValueError(f"不支持的范数类型: {ord}")


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """向量归一化（单位化）"""
    norm = vector_norm(v, ord=2)
    return v / norm if norm > 0 else v


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    余弦相似度

    返回: [-1, 1]之间的值
    """
    dot = np.dot(v1, v2)
    norm1 = vector_norm(v1)
    norm2 = vector_norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


# ==================== 矩阵分析 ====================

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    计算矩阵的秩

    Args:
        A: 输入矩阵
        tol: 奇异值阈值

    Returns:
        秩
    """
    s = np.linalg.svd(A, compute_uv=False)
    return np.sum(s > tol)


def is_invertible(A: np.ndarray, tol: float = 1e-10) -> bool:
    """判断矩阵是否可逆"""
    if A.shape[0] != A.shape[1]:
        return False
    det = np.linalg.det(A)
    return abs(det) > tol


def condition_number(A: np.ndarray) -> float:
    """
    条件数：衡量矩阵数值稳定性

    条件数越大，矩阵越"病态"，数值计算越不稳定
    """
    s = np.linalg.svd(A, compute_uv=False)
    return s.max() / s.min() if s.min() > 0 else np.inf


# ==================== 特征值分解 ====================

def eigen_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    特征值分解

    Args:
        A: 方阵 (n, n)

    Returns:
        eigenvalues: (n,)
        eigenvectors: (n, n) - 每列是一个特征向量
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # 按特征值大小排序
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def is_positive_definite(A: np.ndarray) -> bool:
    """
    判断是否为正定矩阵

    正定条件：所有特征值 > 0
    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


# ==================== SVD相关 ====================

def svd_decomposition(A: np.ndarray, full_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    奇异值分解

    Args:
        A: (m, n)矩阵
        full_matrices: 是否返回完整的U和V

    Returns:
        U: (m, m) 或 (m, k) - 左奇异向量
        s: (k,) - 奇异值（k = min(m, n)）
        VT: (n, n) 或 (k, n) - 右奇异向量转置
    """
    return np.linalg.svd(A, full_matrices=full_matrices)


def low_rank_approximation(A: np.ndarray, rank: int) -> np.ndarray:
    """
    低秩近似：用前rank个奇异值重建矩阵

    Args:
        A: 原始矩阵
        rank: 目标秩

    Returns:
        近似矩阵
    """
    U, s, VT = svd_decomposition(A, full_matrices=False)

    # 保留前rank个
    U_r = U[:, :rank]
    s_r = s[:rank]
    VT_r = VT[:rank, :]

    return U_r @ np.diag(s_r) @ VT_r


def frobenius_norm(A: np.ndarray) -> float:
    """Frobenius范数：||A||_F = sqrt(sum(A^2))"""
    return np.linalg.norm(A, 'fro')


def reconstruction_error(A: np.ndarray, A_approx: np.ndarray) -> float:
    """计算重建误差（Frobenius范数）"""
    return frobenius_norm(A - A_approx)


# ==================== PCA ====================

class PCA:
    """主成分分析"""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X: np.ndarray) -> 'PCA':
        """
        拟合PCA模型

        Args:
            X: (n_samples, n_features)

        Returns:
            self
        """
        n_samples, n_features = X.shape

        # 中心化
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # 协方差矩阵
        cov_matrix = np.cov(X_centered.T)

        # 特征值分解
        eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)

        # 保存主成分
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / eigenvalues.sum()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        投影到主成分空间

        Args:
            X: (n_samples, n_features)

        Returns:
            X_transformed: (n_samples, n_components)
        """
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        从主成分空间重建

        Args:
            X_pca: (n_samples, n_components)

        Returns:
            X_reconstructed: (n_samples, n_features)
        """
        return X_pca @ self.components.T + self.mean


# ==================== 神经网络相关 ====================

def linear_layer_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    全连接层前向传播

    Args:
        x: (batch, in_features)
        W: (in_features, out_features)
        b: (out_features,)

    Returns:
        output: (batch, out_features)
    """
    return x @ W + b


def linear_layer_backward(
    x: np.ndarray,
    grad_output: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    全连接层反向传播

    Args:
        x: (batch, in_features) - 前向时的输入
        grad_output: (batch, out_features) - 来自上一层的梯度

    Returns:
        grad_x: (batch, in_features)
        grad_W: (in_features, out_features)
        grad_b: (out_features,)
    """
    # 假设forward时已保存W
    # 这里简化实现，实际需要在类中保存状态

    # grad_W = xᵀ @ grad_output
    grad_W = x.T @ grad_output

    # grad_b = sum over batch
    grad_b = grad_output.sum(axis=0)

    # grad_x = grad_output @ Wᵀ
    # 这里需要W，实际使用时需要传入或保存
    # grad_x = grad_output @ W.T

    return None, grad_W, grad_b  # grad_x需要W


def attention_scores(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    计算注意力分数

    Args:
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)

    Returns:
        scores: (batch, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    return scores


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax函数（数值稳定版本）

    Args:
        x: 输入
        axis: 归一化的轴

    Returns:
        softmax结果
    """
    # 减去最大值，防止溢出
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ==================== 初始化方法 ====================

def xavier_init(shape: Tuple[int, int]) -> np.ndarray:
    """
    Xavier/Glorot初始化

    适用于tanh和sigmoid激活函数
    """
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


def he_init(shape: Tuple[int, int]) -> np.ndarray:
    """
    He初始化

    适用于ReLU激活函数
    """
    fan_in, _ = shape
    std = np.sqrt(2 / fan_in)
    return np.random.randn(*shape) * std


# ==================== 实用工具 ====================

def gram_matrix(X: np.ndarray) -> np.ndarray:
    """
    Gram矩阵：XᵀX

    用于风格迁移等任务
    """
    return X.T @ X


def pairwise_distances(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    计算成对欧几里得距离

    Args:
        X: (n, d)
        Y: (m, d) 或 None（默认为X）

    Returns:
        distances: (n, m)
    """
    if Y is None:
        Y = X

    # 使用技巧: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T  # (1, m)
    distances = X_norm + Y_norm - 2 * X @ Y.T  # (n, m)

    # 数值稳定性
    distances = np.maximum(distances, 0)
    return np.sqrt(distances)


def orthogonalize(vectors: np.ndarray) -> np.ndarray:
    """
    Gram-Schmidt正交化

    Args:
        vectors: (n, d) - 每行是一个向量

    Returns:
        orthogonal_vectors: (n, d)
    """
    n, d = vectors.shape
    ortho = np.zeros_like(vectors)

    for i in range(n):
        vec = vectors[i].copy()

        # 减去所有之前向量的投影
        for j in range(i):
            projection = np.dot(vec, ortho[j]) * ortho[j]
            vec = vec - projection

        # 归一化
        norm = vector_norm(vec)
        ortho[i] = vec / norm if norm > 0 else vec

    return ortho


# ==================== 测试函数 ====================

def test_matrix_utils():
    """测试所有工具函数"""
    print("=" * 50)
    print("测试线性代数工具函数")
    print("=" * 50)

    # 测试矩阵乘法
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 2)
    C = matrix_multiply(A, B)
    assert np.allclose(C, A @ B), "矩阵乘法错误"
    print("✓ 矩阵乘法")

    # 测试向量范数
    v = np.array([3, 4])
    assert abs(vector_norm(v, 2) - 5.0) < 1e-6, "L2范数错误"
    print("✓ 向量范数")

    # 测试余弦相似度
    v1 = np.array([1, 0])
    v2 = np.array([1, 1])
    sim = cosine_similarity(v1, v2)
    expected = 1 / np.sqrt(2)
    assert abs(sim - expected) < 1e-6, "余弦相似度错误"
    print("✓ 余弦相似度")

    # 测试PCA
    X = np.random.randn(100, 50)
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    assert X_pca.shape == (100, 10), "PCA形状错误"
    X_reconstructed = pca.inverse_transform(X_pca)
    assert X_reconstructed.shape == X.shape, "PCA重建形状错误"
    print("✓ PCA")

    # 测试低秩近似
    A = np.random.randn(20, 30)
    A_approx = low_rank_approximation(A, rank=5)
    assert matrix_rank(A_approx) <= 5, "低秩近似秩错误"
    print("✓ 低秩近似")

    # 测试Softmax
    x = np.random.randn(10, 5)
    y = softmax(x, axis=1)
    assert np.allclose(y.sum(axis=1), 1.0), "Softmax求和不为1"
    print("✓ Softmax")

    print("\n所有测试通过！")


if __name__ == "__main__":
    test_matrix_utils()
