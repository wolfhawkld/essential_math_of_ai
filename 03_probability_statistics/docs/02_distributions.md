# 常见概率分布

## 1. 离散分布

### 1.1 伯努利分布 (Bernoulli Distribution)

**场景**：单次二元试验（成功/失败）

```
X ~ Bernoulli(p)

P(X=1) = p
P(X=0) = 1-p

E[X] = p
Var[X] = p(1-p)
```

**深度学习应用**：
- 二分类标签
- Dropout mask

**代码**：
```python
import numpy as np
import matplotlib.pyplot as plt

# 参数
p = 0.3

# 采样
samples = np.random.binomial(1, p, size=10000)

# 统计
prob_1 = samples.mean()
print(f"P(X=1) ≈ {prob_1:.4f}")  # ≈ 0.3

# 可视化
plt.hist(samples, bins=[-0.5, 0.5, 1.5], density=True, alpha=0.7)
plt.xticks([0, 1])
plt.xlabel('X')
plt.ylabel('Probability')
plt.title(f'Bernoulli({p})')
plt.show()
```

### 1.2 二项分布 (Binomial Distribution)

**场景**：n次独立伯努利试验的成功次数

```
X ~ Binomial(n, p)

P(X=k) = C(n,k) × p^k × (1-p)^(n-k)

E[X] = np
Var[X] = np(1-p)
```

**例子**：抛10次硬币，正面次数

```python
# 参数
n = 10  # 试验次数
p = 0.5  # 成功概率

# 采样
samples = np.random.binomial(n, p, size=10000)

# 理论vs实际
print(f"理论期望: {n * p}")
print(f"样本均值: {samples.mean():.4f}")

# 可视化
plt.hist(samples, bins=np.arange(-0.5, n+1.5, 1),
         density=True, alpha=0.7, label='Sample')

# 理论分布
from scipy.stats import binom
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)
plt.plot(x, pmf, 'ro-', label='Theory')

plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.title(f'Binomial({n}, {p})')
plt.legend()
plt.show()
```

### 1.3 类别分布 (Categorical Distribution)

**场景**：多分类问题（骰子、多类别标签）

```
X ~ Categorical(p₁, p₂, ..., pₖ)

P(X=i) = pᵢ,  其中 Σᵢ pᵢ = 1
```

**深度学习应用**：
- 多分类标签
- Softmax输出

```python
# 参数（3类）
probs = np.array([0.5, 0.3, 0.2])

# 采样
samples = np.random.choice(3, size=10000, p=probs)

# 统计
for i in range(3):
    empirical_prob = (samples == i).mean()
    print(f"P(X={i}): 理论={probs[i]:.2f}, 实际={empirical_prob:.4f}")

# 可视化
plt.hist(samples, bins=np.arange(-0.5, 3.5, 1),
         density=True, alpha=0.7)
plt.xticks([0, 1, 2])
plt.xlabel('Class')
plt.ylabel('Probability')
plt.title('Categorical Distribution')
plt.show()
```

---

## 2. 连续分布

### 2.1 均匀分布 (Uniform Distribution)

**场景**：所有值等可能

```
X ~ Uniform(a, b)

p(x) = 1/(b-a),  x ∈ [a, b]

E[X] = (a+b)/2
Var[X] = (b-a)²/12
```

**深度学习应用**：
- 权重初始化
- 数据增强（随机裁剪）

```python
# 参数
a, b = 0, 1

# 采样
samples = np.random.uniform(a, b, size=10000)

# 统计
print(f"理论期望: {(a+b)/2}")
print(f"样本均值: {samples.mean():.4f}")

# 可视化
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Sample')
plt.axhline(y=1/(b-a), color='r', linestyle='--', label='Density')
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Uniform({a}, {b})')
plt.legend()
plt.show()
```

### 2.2 高斯分布 (Gaussian/Normal Distribution)

**最重要的分布！**

```
X ~ N(μ, σ²)

p(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))

E[X] = μ
Var[X] = σ²
```

**性质**：
- 对称，钟形
- 68-95-99.7规则
  - 68%在μ±σ内
  - 95%在μ±2σ内
  - 99.7%在μ±3σ内

**深度学习应用**：
- **权重初始化**（最常用）
- 噪声建模
- Batch Normalization

```python
from scipy.stats import norm

# 参数
mu, sigma = 0, 1  # 标准正态

# 采样
samples = np.random.normal(mu, sigma, size=10000)

# 可视化
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Sample')

# 理论曲线
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Theory')

plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Normal({mu}, {sigma}²)')
plt.legend()
plt.show()

# 验证68-95-99.7规则
within_1sigma = np.sum(np.abs(samples - mu) <= sigma) / len(samples)
within_2sigma = np.sum(np.abs(samples - mu) <= 2*sigma) / len(samples)
within_3sigma = np.sum(np.abs(samples - mu) <= 3*sigma) / len(samples)

print(f"μ±σ: {within_1sigma:.2%} (理论68%)")
print(f"μ±2σ: {within_2sigma:.2%} (理论95%)")
print(f"μ±3σ: {within_3sigma:.2%} (理论99.7%)")
```

**标准正态分布**：
```
Z ~ N(0, 1)

任意正态: X = μ + σZ
```

### 2.3 多元高斯分布 (Multivariate Gaussian)

**高维正态分布**：

```
X ~ N(μ, Σ)

p(x) ∝ exp(-½(x-μ)ᵀΣ⁻¹(x-μ))

μ: 均值向量 (D,)
Σ: 协方差矩阵 (D, D)
```

**协方差矩阵**：
```
Σᵢⱼ = Cov(Xᵢ, Xⱼ)

对角线: 方差
非对角: 协方差
```

```python
# 2D高斯
mean = np.array([0, 0])
cov = np.array([[1, 0.8],
                [0.8, 1]])  # 正相关

# 采样
samples = np.random.multivariate_normal(mean, cov, size=1000)

# 可视化
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Bivariate Gaussian')
plt.axis('equal')
plt.grid(True)
plt.show()

# 协方差矩阵
sample_cov = np.cov(samples.T)
print(f"理论协方差:\n{cov}")
print(f"样本协方差:\n{sample_cov}")
```

### 2.4 指数分布 (Exponential Distribution)

**场景**：等待时间、事件间隔

```
X ~ Exp(λ)

p(x) = λ e^(-λx),  x ≥ 0

E[X] = 1/λ
Var[X] = 1/λ²
```

**无记忆性**：
```
P(X > s+t | X > s) = P(X > t)
```

```python
from scipy.stats import expon

# 参数
lam = 2.0  # 率参数

# 采样
samples = np.random.exponential(1/lam, size=10000)

# 可视化
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Sample')

x = np.linspace(0, 3, 100)
plt.plot(x, expon.pdf(x, scale=1/lam), 'r-', linewidth=2, label='Theory')

plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Exponential(λ={lam})')
plt.legend()
plt.show()
```

---

## 3. 深度学习中的分布应用

### 3.1 权重初始化

#### Xavier/Glorot初始化
```python
def xavier_uniform(shape):
    """
    适用于tanh、sigmoid
    W ~ Uniform(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
    """
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

def xavier_normal(shape):
    """
    W ~ N(0, 2/(fan_in+fan_out))
    """
    fan_in, fan_out = shape
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, std, size=shape)

# 测试
W = xavier_uniform((512, 256))
print(f"均值: {W.mean():.4f}")  # ≈ 0
print(f"标准差: {W.std():.4f}")
```

#### He初始化
```python
def he_normal(shape):
    """
    适用于ReLU
    W ~ N(0, 2/fan_in)
    """
    fan_in, _ = shape
    std = np.sqrt(2 / fan_in)
    return np.random.normal(0, std, size=shape)

# PyTorch默认使用He初始化for ReLU
```

**为什么不同初始化？**
- Xavier：假设激活函数线性（tanh在0附近近似线性）
- He：考虑ReLU会"杀死"一半神经元

### 3.2 数据增强中的随机性

```python
def random_crop(image, crop_size):
    """随机裁剪（均匀分布）"""
    h, w = image.shape[:2]
    ch, cw = crop_size

    # 随机选择左上角
    top = np.random.randint(0, h - ch + 1)
    left = np.random.randint(0, w - cw + 1)

    return image[top:top+ch, left:left+cw]

def add_gaussian_noise(image, sigma=0.1):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, image.shape)
    return image + noise
```

### 3.3 Dropout（伯努利分布）

```python
def dropout(x, p=0.5, training=True):
    """
    每个神经元以概率p被丢弃（伯努利分布）
    """
    if not training:
        return x

    # 伯努利采样
    mask = (np.random.rand(*x.shape) > p).astype(float)

    return x * mask / (1 - p)

# 测试
x = np.ones((100, 50))
x_dropped = dropout(x, p=0.5)

print(f"保留的神经元比例: {(x_dropped > 0).mean():.4f}")  # ≈ 0.5
print(f"期望不变: {x_dropped.mean():.4f}")  # ≈ 1.0
```

---

## 4. 分布的变换

### 4.1 Box-Muller变换

**从均匀分布生成高斯分布**：

```
U₁, U₂ ~ Uniform(0, 1)

Z₁ = √(-2ln U₁) cos(2πU₂)
Z₂ = √(-2ln U₁) sin(2πU₂)

Z₁, Z₂ ~ N(0, 1)
```

```python
def box_muller(n_samples):
    """Box-Muller变换"""
    u1 = np.random.uniform(0, 1, n_samples)
    u2 = np.random.uniform(0, 1, n_samples)

    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

    return z1, z2

# 测试
z1, z2 = box_muller(10000)

print(f"Z1均值: {z1.mean():.4f}, 标准差: {z1.std():.4f}")
print(f"Z2均值: {z2.mean():.4f}, 标准差: {z2.std():.4f}")

# 验证是标准正态
plt.hist(z1, bins=50, density=True, alpha=0.7, label='Z1')
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, 0, 1), 'r-', label='N(0,1)')
plt.legend()
plt.show()
```

### 4.2 重参数化技巧 (Reparameterization Trick)

**VAE中的关键技巧**：

```
# 想要: z ~ N(μ, σ²)

# 不可导的采样:
z = sample_from_normal(μ, σ)  # ✗ 不可微

# 重参数化:
ε ~ N(0, 1)
z = μ + σ * ε  # ✓ 可微！
```

```python
def sample_gaussian(mu, sigma, n_samples=1):
    """
    可微的高斯采样
    """
    epsilon = np.random.randn(n_samples, *mu.shape)
    return mu + sigma * epsilon

# 在VAE中的应用
def vae_encoder(x):
    # 编码器输出μ和log(σ²)
    mu = encoder_mean(x)
    log_var = encoder_log_var(x)

    # 重参数化采样
    sigma = np.exp(0.5 * log_var)
    epsilon = np.random.randn(*mu.shape)
    z = mu + sigma * epsilon  # 可以反向传播！

    return z, mu, log_var
```

---

## 5. 中心极限定理 (Central Limit Theorem)

**定理**：独立同分布随机变量的和趋向正态分布

```
X₁, X₂, ..., Xₙ ~ 任意分布（均值μ，方差σ²）

当n→∞:
(X̄ - μ) / (σ/√n) → N(0, 1)

其中 X̄ = (1/n) Σᵢ Xᵢ
```

**意义**：
- 解释为什么很多现象服从正态分布
- 样本均值的分布

```python
def demonstrate_clt(distribution_func, n_samples, n_experiments):
    """
    演示中心极限定理

    distribution_func: 生成单个样本的函数
    n_samples: 每次实验的样本数
    n_experiments: 实验次数
    """
    means = []
    for _ in range(n_experiments):
        samples = distribution_func(n_samples)
        means.append(samples.mean())

    return np.array(means)

# 测试：从均匀分布开始
uniform_samples = lambda n: np.random.uniform(0, 1, n)

# 不同样本数
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, n in enumerate([2, 5, 30, 100]):
    ax = axes[idx // 2, idx % 2]

    # 样本均值分布
    means = demonstrate_clt(uniform_samples, n, 10000)

    # 绘制直方图
    ax.hist(means, bins=50, density=True, alpha=0.7, label='Sample means')

    # 理论正态分布
    # Uniform(0,1): μ=0.5, σ²=1/12
    mu = 0.5
    sigma = np.sqrt(1/12 / n)  # 标准误差
    x = np.linspace(means.min(), means.max(), 100)
    ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Theory')

    ax.set_title(f'n = {n}')
    ax.legend()

plt.tight_layout()
plt.show()
```

**观察**：
- n越大，样本均值的分布越接近正态
- 方差随n减小：`Var[X̄] = σ²/n`

---

## 6. 比较不同分布

```python
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, norm, expon, uniform

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 伯努利
ax = axes[0, 0]
x = [0, 1]
pmf = bernoulli.pmf(x, 0.3)
ax.bar(x, pmf, alpha=0.7)
ax.set_title('Bernoulli(0.3)')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')

# 2. 二项
ax = axes[0, 1]
x = np.arange(0, 11)
pmf = binom.pmf(x, 10, 0.5)
ax.bar(x, pmf, alpha=0.7)
ax.set_title('Binomial(10, 0.5)')
ax.set_xlabel('x')

# 3. 均匀
ax = axes[0, 2]
x = np.linspace(0, 1, 100)
pdf = uniform.pdf(x, 0, 1)
ax.plot(x, pdf, linewidth=2)
ax.fill_between(x, pdf, alpha=0.3)
ax.set_title('Uniform(0, 1)')
ax.set_xlabel('x')
ax.set_ylabel('p(x)')

# 4. 标准正态
ax = axes[1, 0]
x = np.linspace(-4, 4, 100)
pdf = norm.pdf(x, 0, 1)
ax.plot(x, pdf, linewidth=2)
ax.fill_between(x, pdf, alpha=0.3)
ax.set_title('Normal(0, 1)')
ax.set_xlabel('x')
ax.set_ylabel('p(x)')

# 5. 不同参数的正态
ax = axes[1, 1]
for mu, sigma in [(0, 1), (1, 0.5), (-1, 1.5)]:
    x = np.linspace(-4, 4, 100)
    pdf = norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, label=f'μ={mu}, σ={sigma}', linewidth=2)
ax.set_title('Different Gaussians')
ax.legend()
ax.set_xlabel('x')

# 6. 指数
ax = axes[1, 2]
x = np.linspace(0, 5, 100)
for lam in [0.5, 1, 2]:
    pdf = expon.pdf(x, scale=1/lam)
    ax.plot(x, pdf, label=f'λ={lam}', linewidth=2)
ax.set_title('Exponential')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('p(x)')

plt.tight_layout()
plt.show()
```

---

## 7. 练习

1. **采样练习**：
   - 从不同分布采样，绘制直方图
   - 验证期望和方差

2. **中心极限定理**：
   - 用指数分布验证CLT
   - 观察不同样本数的效果

3. **权重初始化**：
   - 实现Xavier和He初始化
   - 比较对训练的影响

4. **数据增强**：
   - 实现各种随机变换
   - 可视化增强后的数据

---

## 8. 常用分布速查

| 分布 | 参数 | 期望 | 方差 | 应用 |
|-----|------|------|------|------|
| Bernoulli(p) | p | p | p(1-p) | 二分类、Dropout |
| Binomial(n,p) | n, p | np | np(1-p) | n次试验 |
| Categorical(p) | p₁,...,pₖ | - | - | 多分类 |
| Uniform(a,b) | a, b | (a+b)/2 | (b-a)²/12 | 初始化 |
| Normal(μ,σ²) | μ, σ² | μ | σ² | **最常用** |
| Exponential(λ) | λ | 1/λ | 1/λ² | 等待时间 |

---

**下一步**：学习 [03_bayes_mle.md](./03_bayes_mle.md)，了解参数估计和贝叶斯推理。
