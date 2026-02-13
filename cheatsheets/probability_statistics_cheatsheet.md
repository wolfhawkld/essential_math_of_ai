# 概率统计速查表

## 1. 基础概率

### 概率公理
```
P(A) ∈ [0, 1]
P(Ω) = 1
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
```

### 条件概率
```
P(A|B) = P(A ∩ B) / P(B)
```

### 贝叶斯定理
```
P(A|B) = P(B|A) × P(A) / P(B)

后验概率 = (似然 × 先验) / 证据
```

### 全概率公式
```
P(B) = Σᵢ P(B|Aᵢ) P(Aᵢ)
```

---

## 2. 随机变量

### 期望 (Expectation)
```
离散: E[X] = Σ xᵢ P(X = xᵢ)
连续: E[X] = ∫ x f(x) dx

性质:
E[aX + b] = a E[X] + b
E[X + Y] = E[X] + E[Y]
```

### 方差 (Variance)
```
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²

性质:
Var(aX + b) = a² Var(X)
```

### 标准差 (Standard Deviation)
```
σ = √Var(X)
```

### 协方差 (Covariance)
```
Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
           = E[XY] - E[X]E[Y]

Cov(X, X) = Var(X)
```

### 相关系数 (Correlation)
```
ρ(X, Y) = Cov(X, Y) / (σ_X σ_Y)

范围: [-1, 1]
```

---

## 3. 常见分布

### 伯努利分布 (Bernoulli)
```
X ~ Bernoulli(p)

P(X=1) = p
P(X=0) = 1-p

E[X] = p
Var(X) = p(1-p)
```

**应用**: 二分类标签

### 二项分布 (Binomial)
```
X ~ Binomial(n, p)

P(X=k) = C(n,k) p^k (1-p)^(n-k)

E[X] = np
Var(X) = np(1-p)
```

**应用**: n次独立试验的成功次数

### 正态分布 (Gaussian/Normal)
```
X ~ N(μ, σ²)

f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

E[X] = μ
Var(X) = σ²

标准正态: Z ~ N(0, 1)
```

**性质**:
- 68-95-99.7规则
- 线性组合仍为正态

**应用**: 无处不在（中心极限定理）

### 多元正态分布
```
X ~ N(μ, Σ)

f(x) ∝ exp(-½(x-μ)ᵀΣ⁻¹(x-μ))

Σ: 协方差矩阵
```

### 指数分布 (Exponential)
```
X ~ Exp(λ)

f(x) = λ e^(-λx),  x ≥ 0

E[X] = 1/λ
Var(X) = 1/λ²
```

**应用**: 等待时间、事件间隔

### 类别分布 (Categorical)
```
X ~ Categorical(p₁, p₂, ..., pₖ)

P(X=i) = pᵢ,  Σᵢ pᵢ = 1
```

**应用**: 多分类标签（one-hot编码）

---

## 4. 参数估计

### 最大似然估计 (MLE)
```
θ_MLE = argmax_θ P(data | θ)
      = argmax_θ Πᵢ P(xᵢ | θ)
      = argmax_θ Σᵢ log P(xᵢ | θ)  (对数似然)
```

**例子**: 正态分布参数估计
```python
μ_MLE = (1/n) Σᵢ xᵢ       # 样本均值
σ²_MLE = (1/n) Σᵢ (xᵢ-μ)²  # 样本方差
```

### 最大后验估计 (MAP)
```
θ_MAP = argmax_θ P(θ | data)
      = argmax_θ P(data | θ) P(θ)
      = argmax_θ [log P(data | θ) + log P(θ)]
```

**与MLE对比**:
- MLE: 只看似然
- MAP: 似然 + 先验（加正则化）

---

## 5. 中心极限定理

```
X₁, X₂, ..., Xₙ 独立同分布，E[Xᵢ] = μ, Var(Xᵢ) = σ²

当 n → ∞:
(X̄ - μ) / (σ/√n) → N(0, 1)

其中 X̄ = (1/n) Σᵢ Xᵢ
```

**意义**: 样本均值近似正态分布（即使原分布不是）

---

## 6. 深度学习中的应用

### 损失函数与概率

#### 交叉熵 ↔ 似然最大化
```
分类交叉熵 = -Σᵢ yᵢ log ŷᵢ
           = -log P(y | x)  (负对数似然)

最小化交叉熵 = 最大化似然
```

#### MSE ↔ 高斯似然
```
MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²

假设 y ~ N(ŷ, σ²):
MLE → MSE
```

### 正则化 ↔ 先验

#### L2正则 ↔ 高斯先验
```
L2: λ ||w||²
先验: w ~ N(0, 1/(2λ))

最小化(loss + L2) = MAP估计
```

#### L1正则 ↔ 拉普拉斯先验
```
L1: λ ||w||₁
先验: w ~ Laplace(0, 1/λ)

效果: 稀疏性
```

### Dropout ↔ 贝叶斯近似
```
Dropout(p) ≈ 变分贝叶斯推断

测试时多次前向传播 → 预测的不确定性估计
```

### 批量归一化 ↔ 分布对齐
```
Batch Norm: 让每层输出 ~ N(0, 1)

减少内部协变量偏移 (Internal Covariate Shift)
```

---

## 7. 常用技巧

### 对数技巧
```python
# 避免下溢
log(Πᵢ pᵢ) = Σᵢ log pᵢ

# Softmax稳定版本
log_softmax = x - log(Σⱼ exp(xⱼ))
            = x - logsumexp(x)
```

### 重参数化技巧 (Reparameterization Trick)
```python
# VAE中采样
z ~ N(μ, σ²)

# 重参数化:
ε ~ N(0, 1)
z = μ + σ * ε  # 可导！
```

### 蒙特卡洛估计
```
E[f(X)] ≈ (1/n) Σᵢ f(xᵢ),  xᵢ ~ P(X)

用于近似复杂积分
```

---

## 8. Python速查

### NumPy
```python
# 采样
np.random.normal(mu, sigma, size)
np.random.binomial(n, p, size)
np.random.choice(a, size, p=probs)

# 统计
np.mean(x)
np.var(x)
np.cov(x)
np.corrcoef(x, y)

# 概率密度
from scipy.stats import norm
norm.pdf(x, mu, sigma)  # PDF
norm.cdf(x, mu, sigma)  # CDF
```

### PyTorch
```python
# 分布
import torch.distributions as dist

normal = dist.Normal(mu, sigma)
samples = normal.sample((batch_size,))
log_prob = normal.log_prob(samples)

# 常用分布
dist.Bernoulli(probs)
dist.Categorical(probs)
dist.MultivariateNormal(mean, cov)
```

---

## 9. 记忆口诀

**贝叶斯定理**:
```
后验 ∝ 似然 × 先验
P(θ|D) ∝ P(D|θ) × P(θ)
```

**方差公式**:
```
"全期望减期望的方"
E[X²] - (E[X])²
```

**协方差对称性**:
```
Cov(X, Y) = Cov(Y, X)
Cov(X, X) = Var(X)
```

**独立 vs 不相关**:
```
独立 → 不相关（Cov=0）
不相关 ≠ 独立（可能非线性相关）
```

---

## 10. 常见陷阱

❌ **混淆独立性和不相关性**
```
X, Y独立 ⟹ Cov(X, Y) = 0
Cov(X, Y) = 0 ⏸ X, Y独立  (反向不成立！)

例: Y = X², X ~ N(0,1)
    Cov(X, Y) = 0 但不独立
```

❌ **忘记对数似然**
```
# 数值稳定性差
loss = -Πᵢ P(xᵢ | θ)

# 应该用
loss = -Σᵢ log P(xᵢ | θ)
```

❌ **MAP vs MLE**
```
MLE: 无先验，容易过拟合
MAP: 有先验（正则化），更稳健
```

---

**相关**: [信息论速查表](./information_theory_cheatsheet.md) | [优化速查表](./optimization_cheatsheet.md)
