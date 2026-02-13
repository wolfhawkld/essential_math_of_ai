# 信息论速查表

## 1. 核心概念

### 信息量 (Self-Information)
```
I(x) = -log P(x) = log(1/P(x))

单位:
- bit (log₂)
- nat (ln)
- dit (log₁₀)
```

**直觉**:
- 概率低的事件 → 信息量大（惊讶程度高）
- 概率高的事件 → 信息量小

**例子**:
```
P(太阳明天升起) = 99.99% → I ≈ 0 (不惊讶)
P(中彩票) = 0.0001% → I 很大 (非常惊讶)
```

---

## 2. 熵 (Entropy)

### 香农熵
```
H(X) = E[-log P(X)]
     = -Σₓ P(x) log P(x)

或连续情况:
H(X) = -∫ p(x) log p(x) dx
```

**含义**:
- 平均信息量
- 不确定性的度量
- 编码所需的最小比特数

**性质**:
```
H(X) ≥ 0
H(X) = 0 ⟺ X是确定的（某个p(x)=1）
H(X) ≤ log |X|（均匀分布时最大）
```

**例子**:
```python
# 均匀分布（熵最大）
P = [0.25, 0.25, 0.25, 0.25]
H = -Σ 0.25 log 0.25 = 2 bits

# 确定分布（熵为0）
P = [1, 0, 0, 0]
H = -1 log 1 = 0
```

### 联合熵
```
H(X, Y) = -Σₓ Σᵧ P(x,y) log P(x,y)
```

### 条件熵
```
H(Y|X) = E_X[H(Y|X=x)]
       = -Σₓ P(x) Σᵧ P(y|x) log P(y|x)

H(Y|X) = H(X,Y) - H(X)  (链式法则)
```

**含义**: 知道X后，Y的剩余不确定性

---

## 3. 交叉熵 (Cross-Entropy)

### 定义
```
H(P, Q) = -Σₓ P(x) log Q(x)
        = E_P[-log Q(X)]

P: 真实分布
Q: 预测分布
```

**含义**:
- 用分布Q编码来自分布P的数据所需的平均比特数
- **深度学习最常用的损失函数**

### 性质
```
H(P, Q) ≥ H(P)（当且仅当P=Q时等号成立）
H(P, Q) ≠ H(Q, P)（不对称）
```

---

## 4. KL散度 (Kullback-Leibler Divergence)

### 定义
```
D_KL(P || Q) = Σₓ P(x) log(P(x)/Q(x))
             = E_P[log P(X) - log Q(X)]
             = H(P, Q) - H(P)
```

**含义**:
- P和Q的"距离"（非对称）
- 用Q近似P的额外代价

### 性质
```
D_KL(P || Q) ≥ 0（Gibbs不等式）
D_KL(P || Q) = 0 ⟺ P = Q
D_KL(P || Q) ≠ D_KL(Q || P)（不对称）
```

### 前向KL vs 反向KL
```
前向: D_KL(P || Q) = ∫ P log(P/Q)
- Q覆盖P的所有支撑（zero-avoiding）
- 变分推断常用

反向: D_KL(Q || P) = ∫ Q log(Q/P)
- Q集中在P的高概率区域（zero-forcing）
- 期望最大化（EM）常用
```

---

## 5. 互信息 (Mutual Information)

### 定义
```
I(X; Y) = D_KL(P(X,Y) || P(X)P(Y))
        = H(X) + H(Y) - H(X,Y)
        = H(X) - H(X|Y)
        = H(Y) - H(Y|X)
```

**含义**:
- X和Y的相关程度
- 知道Y后减少的X的不确定性

**性质**:
```
I(X; Y) ≥ 0
I(X; Y) = 0 ⟺ X, Y独立
I(X; Y) = I(Y; X)（对称）
I(X; X) = H(X)
```

### 数据处理不等式
```
X → Y → Z（马尔可夫链）

则: I(X; Z) ≤ I(X; Y)
```

**含义**: 信息不能通过处理增加

---

## 6. 深度学习中的应用

### 分类任务的交叉熵损失

#### 二分类
```
L = -[y log ŷ + (1-y) log(1-ŷ)]

y: 真实标签（0或1）
ŷ: 预测概率
```

```python
import torch
import torch.nn.functional as F

# PyTorch实现
loss = F.binary_cross_entropy(y_pred, y_true)

# 带logits（数值更稳定）
loss = F.binary_cross_entropy_with_logits(logits, y_true)
```

#### 多分类
```
L = -Σᵢ yᵢ log ŷᵢ

y: one-hot向量
ŷ: softmax输出
```

```python
# PyTorch实现
loss = F.cross_entropy(logits, target)  # 内含softmax

# 等价于
log_probs = F.log_softmax(logits, dim=1)
loss = F.nll_loss(log_probs, target)
```

### KL散度的应用

#### 变分自编码器 (VAE)
```
L = E_q(z|x)[log p(x|z)] - D_KL(q(z|x) || p(z))
    ↑重建损失            ↑正则化项

通常:
q(z|x) = N(μ, σ²)  (编码器输出)
p(z) = N(0, I)     (先验)

D_KL = ½ Σᵢ[μᵢ² + σᵢ² - log σᵢ² - 1]
```

```python
def vae_loss(x, x_recon, mu, logvar):
    # 重建损失
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL散度（解析解）
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_div
```

#### 知识蒸馏 (Knowledge Distillation)
```
L = α × L_CE(y, ŷ) + (1-α) × D_KL(ŷ_teacher || ŷ_student)

学生模型同时学习:
- 硬标签（真实标签）
- 软标签（教师输出）
```

```python
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # 标准交叉熵
    ce_loss = F.cross_entropy(student_logits, labels)

    # KL散度（温度缩放）
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * T * T

    return alpha * ce_loss + (1 - alpha) * kl_loss
```

### 互信息最大化

#### 对比学习 (Contrastive Learning)
```
最大化: I(X; Y)

例: SimCLR
L = -log(exp(sim(z_i, z_j)/τ) / Σₖ exp(sim(z_i, z_k)/τ))

目标: 让正样本对的表示相似（互信息大）
```

#### 信息瓶颈 (Information Bottleneck)
```
minimize I(X; Z) - β I(Z; Y)

Z: 中间表示
目标: 压缩输入X，但保留对Y的预测能力
```

---

## 7. 常用公式推导

### 交叉熵 ↔ 最大似然
```
最小化交叉熵:
minimize H(P, Q) = -Σₓ P(x) log Q(x)

等价于最大化对数似然:
maximize Σₓ P(x) log Q(x) = E_P[log Q(X)]

当P是经验分布（数据）:
= (1/n) Σᵢ log Q(xᵢ)  (MLE)
```

### Softmax + 交叉熵的梯度
```
ŷ = softmax(z) = exp(zᵢ) / Σⱼ exp(zⱼ)
L = -Σᵢ yᵢ log ŷᵢ

梯度:
∂L/∂zᵢ = ŷᵢ - yᵢ

极其简洁！
```

---

## 8. Python实现

### 熵
```python
import numpy as np

def entropy(probs):
    """
    H(X) = -Σ p(x) log p(x)
    """
    # 处理p=0的情况
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# 例子
p = [0.5, 0.25, 0.25]
print(f"H(X) = {entropy(p):.2f} bits")  # 1.5 bits
```

### 交叉熵
```python
def cross_entropy(p, q):
    """
    H(P, Q) = -Σ p(x) log q(x)
    """
    p = np.array(p)
    q = np.array(q)
    # 避免log(0)
    q = np.clip(q, 1e-10, 1)
    return -np.sum(p * np.log(q))

# 例子
p = [1, 0, 0]  # 真实分布
q = [0.7, 0.2, 0.1]  # 预测分布
print(f"H(P,Q) = {cross_entropy(p, q):.4f}")
```

### KL散度
```python
def kl_divergence(p, q):
    """
    D_KL(P || Q) = Σ p(x) log(p(x)/q(x))
    """
    p = np.array(p)
    q = np.array(q)
    # 只在p>0的地方计算
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# 等价于
def kl_divergence_alt(p, q):
    return cross_entropy(p, q) - entropy(p)
```

### 互信息（离散）
```python
def mutual_information(joint_prob):
    """
    I(X; Y) = Σₓ Σᵧ p(x,y) log(p(x,y) / (p(x)p(y)))
    """
    joint = np.array(joint_prob)
    px = joint.sum(axis=1, keepdims=True)  # p(x)
    py = joint.sum(axis=0, keepdims=True)  # p(y)

    # 独立分布
    independent = px @ py

    # 互信息
    mask = joint > 0
    mi = np.sum(joint[mask] * np.log(joint[mask] / independent[mask]))
    return mi
```

### PyTorch版本
```python
import torch
import torch.nn.functional as F

def entropy_torch(probs, dim=-1):
    return -(probs * probs.log()).sum(dim)

def cross_entropy_torch(p, q, dim=-1):
    return -(p * q.log()).sum(dim)

def kl_div_torch(p, q, dim=-1):
    return (p * (p.log() - q.log())).sum(dim)

# 或使用内置
kl = F.kl_div(q.log(), p, reduction='batchmean')
```

---

## 9. 常见变体

### Focal Loss
```
FL = -α(1-ŷ)^γ y log ŷ

γ: 聚焦参数（如γ=2）
α: 平衡参数

用途: 处理类别不平衡
```

### Label Smoothing
```
# 硬标签
y_hard = [0, 0, 1, 0]

# 软标签
y_smooth = (1-ε)y_hard + ε/K

ε: 平滑系数（如0.1）
K: 类别数

效果: 防止过拟合，提升泛化
```

```python
def label_smoothing(target, n_classes, smoothing=0.1):
    """
    target: [batch]
    """
    confidence = 1 - smoothing
    smooth_label = torch.full((batch_size, n_classes), smoothing / (n_classes - 1))
    smooth_label.scatter_(1, target.unsqueeze(1), confidence)
    return smooth_label
```

---

## 10. 记忆口诀

**熵 vs 交叉熵 vs KL散度**
```
H(P)      : P的不确定性
H(P, Q)   : 用Q编码P的代价
D_KL(P||Q): 用Q近似P的额外代价

关系: H(P,Q) = H(P) + D_KL(P||Q)
```

**互信息的理解**
```
I(X; Y) = H(Y) - H(Y|X)
        ↑总不确定性  ↑条件不确定性

= 知道X后减少的Y的不确定性
```

**最小化交叉熵 = 最大化似然**
```
argmin H(P,Q) = argmax E_P[log Q]
```

---

## 11. 常见陷阱

❌ **混淆H(P,Q)和H(Q,P)**
```
H(P, Q) ≠ H(Q, P)  (不对称)
H(P, Q): 用Q编码P
H(Q, P): 用P编码Q（很少用）
```

❌ **KL散度不是距离**
```
D_KL(P||Q) ≠ D_KL(Q||P)  (不对称)
不满足三角不等式

如需对称距离，用:
JS散度: (D_KL(P||M) + D_KL(Q||M))/2, M=(P+Q)/2
```

❌ **log的底数**
```
# 二进制熵（bits）
H = -Σ p log₂ p

# 自然熵（nats）
H = -Σ p ln p

深度学习通常用ln（PyTorch默认）
```

❌ **数值稳定性**
```python
# 不稳定
loss = -torch.log(softmax(logits)[target])

# 稳定
loss = F.cross_entropy(logits, target)  # 内含log_softmax
```

---

## 12. 应用场景总结

| 概念 | 公式 | 深度学习应用 |
|-----|------|------------|
| 熵 | `H(X)` | 数据复杂度度量 |
| 交叉熵 | `H(P,Q)` | **分类损失函数** |
| KL散度 | `D_KL(P\|\|Q)` | VAE、蒸馏、正则化 |
| 互信息 | `I(X;Y)` | 对比学习、特征选择 |
| Focal Loss | `-α(1-p)^γ log p` | 类别不平衡 |
| Label Smoothing | 软化标签 | 防过拟合 |

---

**相关**: [概率统计速查表](./probability_statistics_cheatsheet.md) | [优化速查表](./optimization_cheatsheet.md)
