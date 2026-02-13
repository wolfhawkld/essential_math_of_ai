# -*- coding: utf-8 -*-
"""
信息论度量工具模块

实现深度学习中常用的信息论度量计算工具。

主要功能：
- 信息量（自信息）
- 熵（香农熵、联合熵、条件熵）
- 交叉熵
- KL散度（相对熵）
- 互信息
- 信息增益
- 数值稳定的计算方法

作者：Essential Math of AI
"""

import numpy as np
from typing import Union, Tuple, Optional
from scipy.special import softmax


# ============================================================================
# 基础工具函数
# ============================================================================

def safe_log(x: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    数值稳定的对数函数

    参数：
        x: 输入值
        epsilon: 防止log(0)的小值

    返回：
        log(x)，避免数值错误
    """
    return np.log(np.maximum(x, epsilon))


def safe_divide(numerator: np.ndarray,
                denominator: np.ndarray,
                epsilon: float = 1e-10) -> np.ndarray:
    """
    数值稳定的除法

    参数：
        numerator: 分子
        denominator: 分母
        epsilon: 防止除零的小值

    返回：
        numerator / denominator
    """
    return numerator / np.maximum(denominator, epsilon)


def log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    数值稳定的 log(sum(exp(x)))

    避免 exp 溢出：
    log Σ exp(x_i) = x_max + log Σ exp(x_i - x_max)

    参数：
        x: 输入数组
        axis: 计算轴

    返回：
        log(sum(exp(x)))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else result.item()


# ============================================================================
# 信息量
# ============================================================================

def information(p: Union[float, np.ndarray], base: int = 2) -> Union[float, np.ndarray]:
    """
    计算信息量（自信息）

    I(x) = -log_b P(x)

    参数：
        p: 事件概率
        base: 对数底 (2=bit, e=nat, 10=Hartley)

    返回：
        信息量
    """
    p = np.asarray(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("概率必须在[0, 1]范围内")

    if base == 2:
        log_base = np.log2
    elif base == np.e:
        log_base = np.log
    elif base == 10:
        log_base = np.log10
    else:
        # 通用底
        log_base = lambda x: np.log(x) / np.log(base)

    # 处理 p=0 的情况（信息量为无穷大）
    result = np.where(p > 0, -log_base(np.maximum(p, 1e-10)), np.inf)

    return result.item() if np.isscalar(p) or p.ndim == 0 else result


# ============================================================================
# 熵
# ============================================================================

def entropy(probs: np.ndarray, base: int = 2) -> float:
    """
    计算离散分布的熵（香农熵）

    H(X) = -Σ P(x) log P(x)

    参数：
        probs: 概率分布 [p1, p2, ..., pn]
        base: 对数底 (2=bit, e=nat)

    返回：
        熵值
    """
    probs = np.asarray(probs, dtype=float)

    # 归一化（确保和为1）
    probs = probs / np.sum(probs)

    # 过滤零概率（0*log(0) = 0）
    probs = probs[probs > 0]

    # 计算熵
    if base == 2:
        log_probs = np.log2(probs)
    elif base == np.e:
        log_probs = np.log(probs)
    else:
        log_probs = np.log(probs) / np.log(base)

    return -np.sum(probs * log_probs)


def joint_entropy(joint_probs: np.ndarray, base: int = 2) -> float:
    """
    计算联合熵 H(X,Y)

    H(X,Y) = -Σ Σ P(x,y) log P(x,y)

    参数：
        joint_probs: 联合概率矩阵 P(X,Y)
        base: 对数底

    返回：
        联合熵
    """
    joint_probs = np.asarray(joint_probs, dtype=float)

    # 归一化
    joint_probs = joint_probs / np.sum(joint_probs)

    # 过滤零概率
    probs = joint_probs[joint_probs > 0]

    if base == 2:
        log_probs = np.log2(probs)
    elif base == np.e:
        log_probs = np.log(probs)
    else:
        log_probs = np.log(probs) / np.log(base)

    return -np.sum(probs * log_probs)


def conditional_entropy(joint_probs: np.ndarray,
                       marginal: np.ndarray,
                       base: int = 2) -> float:
    """
    计算条件熵 H(Y|X)

    H(Y|X) = H(X,Y) - H(X)

    参数：
        joint_probs: 联合概率 P(X,Y)
        marginal: X的边缘分布 P(X)
        base: 对数底

    返回：
        条件熵
    """
    H_XY = joint_entropy(joint_probs, base)
    H_X = entropy(marginal, base)
    return H_XY - H_X


def gaussian_entropy(sigma: float, base: int = 2) -> float:
    """
    高斯分布的微分熵

    h(X) = (1/2) log(2πeσ²)

    参数：
        sigma: 标准差
        base: 对数底

    返回：
        微分熵
    """
    variance = sigma ** 2
    if base == 2:
        return 0.5 * np.log2(2 * np.pi * np.e * variance)
    elif base == np.e:
        return 0.5 * np.log(2 * np.pi * np.e * variance)
    else:
        return 0.5 * np.log(2 * np.pi * np.e * variance) / np.log(base)


# ============================================================================
# 交叉熵
# ============================================================================

def cross_entropy(p_true: np.ndarray,
                  p_pred: np.ndarray,
                  base: int = 2,
                  epsilon: float = 1e-10) -> float:
    """
    计算交叉熵 H(P, Q)

    H(P, Q) = -Σ P(x) log Q(x)

    参数：
        p_true: 真实分布 P
        p_pred: 预测分布 Q
        base: 对数底
        epsilon: 数值稳定性参数

    返回：
        交叉熵
    """
    p_true = np.asarray(p_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)

    # 归一化
    p_true = p_true / np.sum(p_true)
    p_pred = p_pred / np.sum(p_pred)

    # 避免log(0)
    p_pred = np.clip(p_pred, epsilon, 1.0)

    # 过滤 p_true=0 的情况
    mask = p_true > 0

    if base == 2:
        log_probs = np.log2(p_pred[mask])
    elif base == np.e:
        log_probs = np.log(p_pred[mask])
    else:
        log_probs = np.log(p_pred[mask]) / np.log(base)

    return -np.sum(p_true[mask] * log_probs)


def binary_cross_entropy(y_true: Union[int, float, np.ndarray],
                         y_pred: Union[float, np.ndarray],
                         epsilon: float = 1e-10) -> Union[float, np.ndarray]:
    """
    二分类交叉熵

    BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

    参数：
        y_true: 真实标签 (0 或 1)
        y_pred: 预测概率 [0, 1]
        epsilon: 数值稳定性

    返回：
        二分类交叉熵
    """
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)


def categorical_cross_entropy(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             epsilon: float = 1e-10) -> float:
    """
    多分类交叉熵

    CE = -Σ y_i · log(ŷ_i)

    参数：
        y_true: 真实标签的one-hot编码 (K,)
        y_pred: 预测概率分布 (K,)
        epsilon: 数值稳定性

    返回：
        分类交叉熵
    """
    y_pred = np.clip(y_pred, epsilon, 1.0)
    return -np.sum(y_true * np.log(y_pred))


def softmax_cross_entropy_with_logits(logits: np.ndarray,
                                      y_true: np.ndarray) -> float:
    """
    Softmax + 交叉熵（联合计算，数值更稳定）

    参数：
        logits: 未归一化的模型输出 (K,)
        y_true: 真实标签的one-hot编码 (K,)

    返回：
        交叉熵损失
    """
    # Log-softmax技巧（避免exp溢出）
    log_probs = logits - log_sum_exp(logits)

    # 交叉熵
    return -np.sum(y_true * log_probs)


# ============================================================================
# KL散度
# ============================================================================

def kl_divergence(p: np.ndarray,
                  q: np.ndarray,
                  base: int = 2,
                  epsilon: float = 1e-10) -> float:
    """
    计算KL散度 KL(P||Q)

    KL(P||Q) = Σ P(x) log[P(x)/Q(x)]

    参数：
        p: 真实分布 P
        q: 近似分布 Q
        base: 对数底
        epsilon: 数值稳定性

    返回：
        KL散度（非负）
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # 归一化
    p = p / np.sum(p)
    q = q / np.sum(q)

    # 避免log(0)和除零
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    # 重新归一化
    p = p / np.sum(p)
    q = q / np.sum(q)

    # 过滤 p=0 的情况（0*log(0/q) = 0）
    mask = p > epsilon

    if base == 2:
        log_ratio = np.log2(p[mask] / q[mask])
    elif base == np.e:
        log_ratio = np.log(p[mask] / q[mask])
    else:
        log_ratio = np.log(p[mask] / q[mask]) / np.log(base)

    return np.sum(p[mask] * log_ratio)


def kl_divergence_gaussian(mu1: float, sigma1: float,
                           mu2: float, sigma2: float,
                           base: int = 2) -> float:
    """
    两个高斯分布之间的KL散度

    KL(N(μ₁,σ₁²) || N(μ₂,σ₂²))

    = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

    参数：
        mu1, sigma1: 第一个高斯的均值和标准差
        mu2, sigma2: 第二个高斯的均值和标准差
        base: 对数底

    返回：
        KL散度
    """
    var1 = sigma1 ** 2
    var2 = sigma2 ** 2

    kl = np.log(sigma2 / sigma1) + \
         (var1 + (mu1 - mu2)**2) / (2 * var2) - 0.5

    if base == 2:
        return kl / np.log(2)
    else:
        return kl


def js_divergence(p: np.ndarray,
                  q: np.ndarray,
                  base: int = 2,
                  epsilon: float = 1e-10) -> float:
    """
    Jensen-Shannon散度（KL散度的对称版本）

    JS(P||Q) = (1/2) KL(P||M) + (1/2) KL(Q||M)
    其中 M = (P + Q) / 2

    参数：
        p: 分布 P
        q: 分布 Q
        base: 对数底
        epsilon: 数值稳定性

    返回：
        JS散度（对称且非负）
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # 归一化
    p = p / np.sum(p)
    q = q / np.sum(q)

    # 中间分布
    m = (p + q) / 2

    return 0.5 * kl_divergence(p, m, base, epsilon) + \
           0.5 * kl_divergence(q, m, base, epsilon)


# ============================================================================
# 互信息
# ============================================================================

def mutual_information(joint_probs: np.ndarray, base: int = 2) -> float:
    """
    计算互信息 I(X;Y)

    I(X;Y) = Σ Σ P(x,y) log[P(x,y)/(P(x)P(y))]

    参数：
        joint_probs: 联合概率矩阵 P(X,Y)
        base: 对数底

    返回：
        互信息
    """
    joint_probs = np.asarray(joint_probs, dtype=float)

    # 归一化
    joint_probs = joint_probs / np.sum(joint_probs)

    # 计算边缘分布
    p_x = joint_probs.sum(axis=1)  # P(X)
    p_y = joint_probs.sum(axis=0)  # P(Y)

    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    H_X = entropy(p_x, base)
    H_Y = entropy(p_y, base)
    H_XY = joint_entropy(joint_probs, base)

    return H_X + H_Y - H_XY


def mutual_information_continuous(x: np.ndarray,
                                  y: np.ndarray,
                                  bins: int = 20,
                                  base: int = 2) -> float:
    """
    连续变量互信息的直方图估计

    参数：
        x, y: 连续变量观测值
        bins: 直方图bin数
        base: 对数底

    返回：
        互信息估计值
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # 计算联合直方图
    joint_hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    # 归一化为概率
    joint_prob = joint_hist / np.sum(joint_hist)

    return mutual_information(joint_prob, base)


def conditional_mutual_information(joint_probs: np.ndarray,
                                   base: int = 2) -> float:
    """
    条件互信息 I(X;Y|Z)

    注：此为简化实现，实际需要三维联合分布

    参数：
        joint_probs: 联合概率（需正确传入三维数组）
        base: 对数底

    返回：
        条件互信息
    """
    # 简化实现，实际应用需要正确处理三维数组
    raise NotImplementedError("完整实现需要三维联合分布处理")


# ============================================================================
# 信息增益
# ============================================================================

def information_gain(y: np.ndarray,
                     feature_values: np.ndarray,
                     base: int = 2) -> float:
    """
    计算特征的信息增益

    IG(Y, X) = H(Y) - H(Y|X) = I(Y; X)

    参数：
        y: 目标变量值
        feature_values: 特征值
        base: 对数底

    返回：
        信息增益
    """
    y = np.asarray(y)
    feature_values = np.asarray(feature_values)

    n = len(y)

    # 构建联合分布
    unique_y = np.unique(y)
    unique_feat = np.unique(feature_values)

    joint = np.zeros((len(unique_feat), len(unique_y)))

    for i, feat_val in enumerate(unique_feat):
        for j, y_val in enumerate(unique_y):
            joint[i, j] = np.sum((feature_values == feat_val) & (y == y_val)) / n

    return mutual_information(joint, base)


# ============================================================================
# 信息论与损失函数
# ============================================================================

def cross_entropy_loss(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       reduction: str = 'mean',
                       epsilon: float = 1e-10) -> float:
    """
    交叉熵损失（支持多分类）

    参数：
        y_true: 真实标签 (N, K) one-hot 或 (N,) 类别索引
        y_pred: 预测概率 (N, K)
        reduction: 'mean', 'sum', 'none'
        epsilon: 数值稳定性

    返回：
        交叉熵损失
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 如果y_true是类别索引，转换为one-hot
    if y_true.ndim == 1:
        n_classes = y_pred.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        y_true = y_true_onehot

    # 避免log(0)
    y_pred = np.clip(y_pred, epsilon, 1.0)

    # 计算每个样本的损失
    losses = -np.sum(y_true * np.log(y_pred), axis=1)

    if reduction == 'mean':
        return np.mean(losses)
    elif reduction == 'sum':
        return np.sum(losses)
    else:
        return losses


def binary_cross_entropy_loss(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              reduction: str = 'mean',
                              epsilon: float = 1e-10) -> float:
    """
    二分类交叉熵损失

    参数：
        y_true: 真实标签 (N,) ∈ {0, 1}
        y_pred: 预测概率 (N,) ∈ [0, 1]
        reduction: 'mean', 'sum', 'none'
        epsilon: 数值稳定性

    返回：
        二分类交叉熵
    """
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    losses = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    if reduction == 'mean':
        return np.mean(losses)
    elif reduction == 'sum':
        return np.sum(losses)
    else:
        return losses


# ============================================================================
# 高级应用
# ============================================================================

def vae_kl_divergence(z_mean: np.ndarray,
                      z_log_var: np.ndarray,
                      reduction: str = 'mean') -> float:
    """
    VAE的KL散度损失：KL(N(μ,σ²) || N(0,1))

    = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

    参数：
        z_mean: 潜在变量均值 (N, latent_dim)
        z_log_var: 潜在变量对数方差 (N, latent_dim)
        reduction: 'mean', 'sum'

    返回：
        KL散度损失
    """
    kl_loss = -0.5 * np.sum(1 + z_log_var - z_mean**2 - np.exp(z_log_var), axis=1)

    if reduction == 'mean':
        return np.mean(kl_loss)
    elif reduction == 'sum':
        return np.sum(kl_loss)
    else:
        return kl_loss


def knowledge_distillation_loss(y_student: np.ndarray,
                                y_teacher: np.ndarray,
                                y_true: np.ndarray,
                                temperature: float = 3.0,
                                alpha: float = 0.7) -> float:
    """
    知识蒸馏损失

    L = α·T²·KL(teacher||student) + (1-α)·CE(true, student)

    参数：
        y_student: 学生模型logits (N, K)
        y_teacher: 教师模型logits (N, K)
        y_true: 真实标签 (N,) 或 (N, K)
        temperature: 蒸馏温度
        alpha: 软标签权重

    返回：
        知识蒸馏损失
    """
    # 软标签损失
    p_student = softmax(y_student / temperature, axis=1)
    p_teacher = softmax(y_teacher / temperature, axis=1)

    kl_loss = 0.0
    for i in range(len(y_student)):
        kl_loss += kl_divergence(p_teacher[i], p_student[i])
    kl_loss /= len(y_student)

    # 硬标签损失
    if y_true.ndim == 1:
        n_classes = y_student.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        y_true = y_true_onehot

    ce_loss = cross_entropy_loss(y_true, softmax(y_student, axis=1))

    return alpha * (temperature**2) * kl_loss + (1 - alpha) * ce_loss


# ============================================================================
# 测试函数
# ============================================================================

def test_information():
    """测试信息量"""
    print("="*60)
    print("测试信息量")
    print("="*60)

    probs = [1.0, 0.5, 0.25, 0.1, 0.01]
    for p in probs:
        info = information(p, base=2)
        print(f"P={p:.2f}: I={info:.4f} bits")

    assert abs(information(0.5, base=2) - 1.0) < 1e-6, "信息量测试失败"
    print("✓ 信息量测试通过\n")


def test_entropy():
    """测试熵"""
    print("="*60)
    print("测试熵")
    print("="*60)

    # 均匀分布
    p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    H_uniform = entropy(p_uniform)
    print(f"均匀分布 {p_uniform}: H={H_uniform:.4f} bits (理论值: 2.0)")

    # 确定分布
    p_certain = np.array([1.0, 0.0, 0.0])
    H_certain = entropy(p_certain)
    print(f"确定分布 {p_certain}: H={H_certain:.4f} bits (理论值: 0.0)")

    assert abs(H_uniform - 2.0) < 0.01, "熵测试失败"
    assert abs(H_certain - 0.0) < 0.01, "熵测试失败"
    print("✓ 熵测试通过\n")


def test_cross_entropy():
    """测试交叉熵"""
    print("="*60)
    print("测试交叉熵")
    print("="*60)

    p_true = np.array([0.7, 0.3])
    p_pred_good = np.array([0.6, 0.4])
    p_pred_bad = np.array([0.3, 0.7])

    ce_good = cross_entropy(p_true, p_pred_good)
    ce_bad = cross_entropy(p_true, p_pred_bad)

    print(f"真实分布: {p_true}")
    print(f"好预测 {p_pred_good}: CE={ce_good:.4f}")
    print(f"差预测 {p_pred_bad}: CE={ce_bad:.4f}")

    assert ce_good < ce_bad, "交叉熵测试失败"
    print("✓ 交叉熵测试通过\n")


def test_kl_divergence():
    """测试KL散度"""
    print("="*60)
    print("测试KL散度")
    print("="*60)

    p = np.array([0.5, 0.5])
    q1 = np.array([0.5, 0.5])
    q2 = np.array([0.9, 0.1])

    kl_same = kl_divergence(p, q1)
    kl_diff = kl_divergence(p, q2)

    print(f"P = {p}")
    print(f"Q相同 {q1}: KL(P||Q)={kl_same:.6f} (理论值: 0)")
    print(f"Q不同 {q2}: KL(P||Q)={kl_diff:.4f}")

    assert abs(kl_same) < 1e-6, "KL散度测试失败"
    assert kl_diff > 0, "KL散度测试失败"
    print("✓ KL散度测试通过\n")


def test_mutual_information():
    """测试互信息"""
    print("="*60)
    print("测试互信息")
    print("="*60)

    # 独立变量
    joint_independent = np.outer([0.5, 0.5], [0.5, 0.5])
    mi_independent = mutual_information(joint_independent)
    print(f"独立变量: I={mi_independent:.6f} (理论值: ≈0)")

    # 完全相关
    joint_dependent = np.array([[0.5, 0.0], [0.0, 0.5]])
    mi_dependent = mutual_information(joint_dependent)
    print(f"完全相关: I={mi_dependent:.4f} (理论值: 1.0)")

    assert abs(mi_independent) < 0.01, "互信息测试失败"
    assert abs(mi_dependent - 1.0) < 0.01, "互信息测试失败"
    print("✓ 互信息测试通过\n")


def test_loss_functions():
    """测试损失函数"""
    print("="*60)
    print("测试损失函数")
    print("="*60)

    # 多分类交叉熵
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred_good = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    y_pred_bad = np.array([[0.3, 0.4, 0.3], [0.4, 0.3, 0.3]])

    loss_good = cross_entropy_loss(y_true, y_pred_good)
    loss_bad = cross_entropy_loss(y_true, y_pred_bad)

    print(f"好预测损失: {loss_good:.4f}")
    print(f"差预测损失: {loss_bad:.4f}")

    assert loss_good < loss_bad, "损失函数测试失败"
    print("✓ 损失函数测试通过\n")


def test_vae_kl():
    """测试VAE的KL散度"""
    print("="*60)
    print("测试VAE KL散度")
    print("="*60)

    z_mean = np.array([[0.0, 0.0], [1.0, -1.0]])
    z_log_var = np.array([[0.0, 0.0], [0.5, 0.5]])

    kl_loss = vae_kl_divergence(z_mean, z_log_var)

    print(f"z_mean: {z_mean}")
    print(f"z_log_var: {z_log_var}")
    print(f"KL损失: {kl_loss:.4f}")

    assert kl_loss > 0, "VAE KL测试失败"
    print("✓ VAE KL测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("信息论模块测试套件")
    print("="*60 + "\n")

    test_information()
    test_entropy()
    test_cross_entropy()
    test_kl_divergence()
    test_mutual_information()
    test_loss_functions()
    test_vae_kl()

    print("="*60)
    print("✓ 所有测试通过！")
    print("="*60)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    run_all_tests()

    # 示例：计算各种信息论度量
    print("\n" + "="*60)
    print("信息论度量示例")
    print("="*60)

    # 例子分布
    p_true = np.array([0.6, 0.3, 0.1])
    p_pred = np.array([0.5, 0.4, 0.1])

    print(f"\n真实分布 P: {p_true}")
    print(f"预测分布 Q: {p_pred}")

    print(f"\n熵 H(P): {entropy(p_true):.4f} bits")
    print(f"熵 H(Q): {entropy(p_pred):.4f} bits")

    print(f"\n交叉熵 H(P,Q): {cross_entropy(p_true, p_pred):.4f} bits")
    print(f"交叉熵 H(Q,P): {cross_entropy(p_pred, p_true):.4f} bits")

    print(f"\nKL散度 KL(P||Q): {kl_divergence(p_true, p_pred):.4f}")
    print(f"KL散度 KL(Q||P): {kl_divergence(p_pred, p_true):.4f}")

    print(f"\nJS散度 JS(P||Q): {js_divergence(p_true, p_pred):.4f}")

    # 验证关系
    print("\n验证 H(P,Q) = H(P) + KL(P||Q):")
    print(f"{cross_entropy(p_true, p_pred):.4f} ≈ {entropy(p_true):.4f} + {kl_divergence(p_true, p_pred):.4f} ✓")
