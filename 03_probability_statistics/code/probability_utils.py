"""
概率统计工具函数

提供常用概率分布、采样、概率计算等实用功能
"""

import numpy as np
from typing import Tuple, List, Union
from scipy import stats


# ====================
# 1. 概率分布类
# ====================

class Distribution:
    """概率分布基类"""

    def sample(self, size: int = 1):
        """采样"""
        raise NotImplementedError

    def pdf(self, x):
        """概率密度函数"""
        raise NotImplementedError

    def cdf(self, x):
        """累积分布函数"""
        raise NotImplementedError


class Gaussian(Distribution):
    """高斯分布 N(μ, σ²)"""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma

    def sample(self, size: int = 1):
        return np.random.normal(self.mu, self.sigma, size=size)

    def pdf(self, x):
        """概率密度函数"""
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

    def log_pdf(self, x):
        """对数概率密度"""
        return -0.5 * np.log(2 * np.pi * self.sigma**2) - \
               0.5 * ((x - self.mu) / self.sigma) ** 2

    def cdf(self, x):
        """累积分布函数"""
        return 0.5 * (1 + np.erf((x - self.mu) / (self.sigma * np.sqrt(2))))

    def mle_fit(self, data: np.ndarray) -> Tuple[float, float]:
        """最大似然估计参数

        Returns:
            (mu_mle, sigma_mle)
        """
        mu_mle = np.mean(data)
        sigma_mle = np.std(data, ddof=0)  # ddof=0 是 MLE
        return mu_mle, sigma_mle


class Bernoulli(Distribution):
    """伯努利分布 Bernoulli(p)"""

    def __init__(self, p: float = 0.5):
        assert 0 <= p <= 1, "p must be in [0, 1]"
        self.p = p

    def sample(self, size: int = 1):
        """返回 0 或 1"""
        return np.random.binomial(1, self.p, size=size)

    def pmf(self, x):
        """概率质量函数"""
        return np.where(x == 1, self.p, 1 - self.p)

    def log_pmf(self, x):
        """对数概率质量函数"""
        return np.where(x == 1, np.log(self.p), np.log(1 - self.p))

    def mle_fit(self, data: np.ndarray) -> float:
        """最大似然估计

        Returns:
            p_mle = 样本中1的比例
        """
        return np.mean(data)


class Binomial(Distribution):
    """二项分布 Binomial(n, p)"""

    def __init__(self, n: int, p: float):
        self.n = n
        self.p = p

    def sample(self, size: int = 1):
        """返回 n 次试验中成功的次数"""
        return np.random.binomial(self.n, self.p, size=size)

    def pmf(self, k):
        """P(X = k)"""
        return stats.binom.pmf(k, self.n, self.p)


class Categorical(Distribution):
    """类别分布（多项分布的单次试验）"""

    def __init__(self, probs: np.ndarray):
        """
        Args:
            probs: 各类别概率 [p1, p2, ..., pk]，和为1
        """
        assert np.isclose(np.sum(probs), 1.0), "Probabilities must sum to 1"
        self.probs = probs
        self.k = len(probs)

    def sample(self, size: int = 1):
        """返回类别索引 0, 1, ..., k-1"""
        return np.random.choice(self.k, size=size, p=self.probs)

    def pmf(self, x):
        """P(X = x)"""
        return self.probs[x]

    def log_pmf(self, x):
        """log P(X = x)"""
        return np.log(self.probs[x])


class Multinomial(Distribution):
    """多项分布 Multinomial(n, p)"""

    def __init__(self, n: int, probs: np.ndarray):
        self.n = n
        self.probs = probs

    def sample(self, size: int = 1):
        """返回各类别的计数 [n1, n2, ..., nk]"""
        return np.random.multinomial(self.n, self.probs, size=size)


# ====================
# 2. 概率计算函数
# ====================

def joint_probability(p_a: float, p_b: float, independent: bool = True) -> float:
    """计算联合概率 P(A, B)

    Args:
        p_a: P(A)
        p_b: P(B) 或 P(B|A)
        independent: 是否独立

    Returns:
        P(A, B) = P(A) * P(B) if independent
               = P(A) * P(B|A) otherwise
    """
    return p_a * p_b


def conditional_probability(p_ab: float, p_b: float) -> float:
    """条件概率 P(A|B) = P(A,B) / P(B)"""
    if p_b == 0:
        raise ValueError("P(B) cannot be zero")
    return p_ab / p_b


def bayes_theorem(p_b_given_a: float, p_a: float, p_b: float) -> float:
    """贝叶斯定理 P(A|B) = P(B|A) * P(A) / P(B)"""
    if p_b == 0:
        raise ValueError("P(B) cannot be zero")
    return (p_b_given_a * p_a) / p_b


def total_probability(p_a_given_b: List[float], p_b: List[float]) -> float:
    """全概率公式 P(A) = Σ P(A|Bᵢ) * P(Bᵢ)

    Args:
        p_a_given_b: [P(A|B1), P(A|B2), ...]
        p_b: [P(B1), P(B2), ...]
    """
    return np.sum(np.array(p_a_given_b) * np.array(p_b))


# ====================
# 3. 统计量计算
# ====================

def expectation(values: np.ndarray, probs: np.ndarray = None) -> float:
    """期望 E[X] = Σ x * P(x)

    Args:
        values: 取值
        probs: 概率（如果为None，假设均匀分布）
    """
    if probs is None:
        return np.mean(values)
    return np.sum(values * probs)


def variance(values: np.ndarray, probs: np.ndarray = None) -> float:
    """方差 Var[X] = E[X²] - (E[X])²"""
    ex = expectation(values, probs)
    ex2 = expectation(values**2, probs)
    return ex2 - ex**2


def covariance(x: np.ndarray, y: np.ndarray) -> float:
    """协方差 Cov(X, Y) = E[(X - E[X])(Y - E[Y])]"""
    return np.mean((x - np.mean(x)) * (y - np.mean(y)))


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """相关系数 ρ = Cov(X, Y) / (σ_X * σ_Y)"""
    return covariance(x, y) / (np.std(x) * np.std(y))


# ====================
# 4. 最大似然估计 (MLE)
# ====================

def gaussian_mle(data: np.ndarray) -> Tuple[float, float]:
    """高斯分布的 MLE 估计

    Returns:
        (mu_mle, sigma_mle)
    """
    mu_mle = np.mean(data)
    sigma_mle = np.std(data, ddof=0)
    return mu_mle, sigma_mle


def bernoulli_mle(data: np.ndarray) -> float:
    """伯努利分布的 MLE 估计

    Returns:
        p_mle = 样本均值
    """
    return np.mean(data)


def categorical_mle(data: np.ndarray, n_classes: int) -> np.ndarray:
    """类别分布的 MLE 估计

    Args:
        data: 类别索引 [0, 1, 2, ...]
        n_classes: 类别数量

    Returns:
        各类别的概率估计 [p1, p2, ..., pk]
    """
    counts = np.bincount(data, minlength=n_classes)
    return counts / len(data)


# ====================
# 5. 信息论相关
# ====================

def entropy(probs: np.ndarray) -> float:
    """熵 H(X) = -Σ p(x) log p(x)"""
    # 避免 log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """交叉熵 H(p, q) = -Σ p(x) log q(x)"""
    q = np.clip(q, 1e-10, 1.0)  # 避免 log(0)
    return -np.sum(p * np.log2(q))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL散度 KL(p||q) = Σ p(x) log(p(x)/q(x))"""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log2(p / q))


# ====================
# 6. 采样方法
# ====================

def rejection_sampling(target_pdf, proposal_pdf, proposal_sampler,
                       M: float, n_samples: int = 1000) -> np.ndarray:
    """拒绝采样

    从复杂分布中采样的通用方法

    Args:
        target_pdf: 目标分布的概率密度函数
        proposal_pdf: 提议分布的概率密度函数
        proposal_sampler: 从提议分布采样的函数
        M: 常数，使得 target_pdf(x) ≤ M * proposal_pdf(x)
        n_samples: 目标样本数

    Returns:
        从目标分布采样的样本
    """
    samples = []
    while len(samples) < n_samples:
        # 从提议分布采样
        x = proposal_sampler(1)[0]
        u = np.random.uniform(0, 1)

        # 接受/拒绝
        if u <= target_pdf(x) / (M * proposal_pdf(x)):
            samples.append(x)

    return np.array(samples)


def importance_sampling(target_pdf, proposal_pdf, proposal_sampler,
                        func, n_samples: int = 1000) -> float:
    """重要性采样

    估计 E_p[f(x)] = ∫ f(x) p(x) dx

    Args:
        target_pdf: 目标分布 p(x)
        proposal_pdf: 提议分布 q(x)
        proposal_sampler: 从 q(x) 采样
        func: 要计算期望的函数 f(x)
        n_samples: 采样数量

    Returns:
        E_p[f(x)] 的估计值
    """
    # 从提议分布采样
    samples = proposal_sampler(n_samples)

    # 计算重要性权重
    weights = target_pdf(samples) / proposal_pdf(samples)

    # 加权平均
    return np.mean(weights * func(samples))


# ====================
# 7. 实用工具
# ====================

def normalize_probs(logits: np.ndarray) -> np.ndarray:
    """将 logits 转换为概率（softmax）"""
    exp_logits = np.exp(logits - np.max(logits))  # 数值稳定
    return exp_logits / np.sum(exp_logits)


def log_sum_exp(x: np.ndarray) -> float:
    """数值稳定的 log(sum(exp(x)))"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


def sample_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> int:
    """带温度的采样（用于语言模型等）

    Args:
        logits: 未归一化的对数概率
        temperature: 温度参数
            - temperature > 1: 输出更随机
            - temperature < 1: 输出更确定
            - temperature → 0: 等价于 argmax

    Returns:
        采样的类别索引
    """
    logits = logits / temperature
    probs = normalize_probs(logits)
    return np.random.choice(len(probs), p=probs)


# ====================
# 8. 示例和测试
# ====================

if __name__ == "__main__":
    print("=" * 60)
    print("概率统计工具函数测试")
    print("=" * 60)

    # 1. 高斯分布
    print("\n1. 高斯分布")
    gaussian = Gaussian(mu=5.0, sigma=2.0)
    samples = gaussian.sample(1000)
    mu_mle, sigma_mle = gaussian.mle_fit(samples)
    print(f"真实参数: μ=5.0, σ=2.0")
    print(f"MLE估计: μ={mu_mle:.3f}, σ={sigma_mle:.3f}")

    # 2. 伯努利分布
    print("\n2. 伯努利分布")
    bernoulli = Bernoulli(p=0.7)
    samples = bernoulli.sample(1000)
    p_mle = bernoulli.mle_fit(samples)
    print(f"真实参数: p=0.7")
    print(f"MLE估计: p={p_mle:.3f}")

    # 3. 贝叶斯定理示例
    print("\n3. 贝叶斯定理示例（疾病诊断）")
    # P(疾病) = 0.01
    # P(阳性|疾病) = 0.99
    # P(阳性|健康) = 0.05
    p_disease = 0.01
    p_pos_given_disease = 0.99
    p_pos_given_healthy = 0.05

    # P(阳性) = P(阳性|疾病)*P(疾病) + P(阳性|健康)*P(健康)
    p_positive = total_probability(
        [p_pos_given_disease, p_pos_given_healthy],
        [p_disease, 1 - p_disease]
    )

    # P(疾病|阳性) = P(阳性|疾病) * P(疾病) / P(阳性)
    p_disease_given_pos = bayes_theorem(p_pos_given_disease, p_disease, p_positive)
    print(f"P(疾病|阳性) = {p_disease_given_pos:.4f}")
    print("即使测试阳性，实际患病概率只有 ~16.5%")

    # 4. 熵和KL散度
    print("\n4. 信息论度量")
    p = np.array([0.5, 0.5])
    q = np.array([0.9, 0.1])
    print(f"p的熵: {entropy(p):.4f}")
    print(f"q的熵: {entropy(q):.4f}")
    print(f"交叉熵 H(p,q): {cross_entropy(p, q):.4f}")
    print(f"KL散度 KL(p||q): {kl_divergence(p, q):.4f}")

    # 5. 温度采样
    print("\n5. 温度采样")
    logits = np.array([2.0, 1.0, 0.1])
    print("logits:", logits)
    print("温度=1.0:", [sample_with_temperature(logits, 1.0) for _ in range(10)])
    print("温度=0.1:", [sample_with_temperature(logits, 0.1) for _ in range(10)])
    print("温度=2.0:", [sample_with_temperature(logits, 2.0) for _ in range(10)])

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
