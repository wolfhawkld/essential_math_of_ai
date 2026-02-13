# 交叉熵与KL散度

## 为什么学习交叉熵和KL散度？

这是深度学习**最核心**的信息论概念：

### 现实应用

```
应用1：分类任务
├── 为什么用交叉熵损失？
└── 本质：最小化预测分布和真实分布的KL散度

应用2：VAE（变分自编码器）
├── 损失函数包含KL散度项
└── 让潜在分布接近先验分布

应用3：知识蒸馏
├── 学生模型学习教师模型的软标签
└── 最小化学生和教师输出的KL散度

应用4：强化学习
├── 策略优化中的熵正则化
└── 防止策略过早收敛到次优解
```

---

## 1. 交叉熵（Cross Entropy）

### 1.1 动机：编码问题

**问题**：用分布Q来编码分布P的数据，平均需要多少比特？

```python
import numpy as np
import matplotlib.pyplot as plt

# 场景：编码天气
# 真实分布 P: 晴天60%, 多云30%, 下雨10%
# 我们使用的编码 Q: 晴天33%, 多云33%, 下雨33%

P = np.array([0.6, 0.3, 0.1])  # 真实分布
Q = np.array([1/3, 1/3, 1/3])  # 我们的编码假设

# 如果用Q的最佳编码
# 编码长度 = -log₂ Q(x)
code_lengths = -np.log2(Q)

# 平均比特数
avg_bits = np.sum(P * code_lengths)

print("编码问题示例：")
print("="*60)
print(f"真实分布 P: {P}")
print(f"编码假设 Q: {Q}")
print(f"各天气的编码长度: {code_lengths}")
print(f"平均编码长度: {avg_bits:.4f} bits")

# 对比：如果知道真实分布P
optimal_lengths = -np.log2(P)
optimal_avg = np.sum(P * optimal_lengths)
print(f"\n如果知道真实分布P，最优平均长度: {optimal_avg:.4f} bits")
print(f"多花费: {avg_bits - optimal_avg:.4f} bits")
```

### 1.2 定义

**交叉熵**：用分布Q编码分布P的平均比特数

```
H(P, Q) = -Σ P(x) log Q(x)
```

**性质**：
1. **不对称**：H(P, Q) ≠ H(Q, P)
2. **H(P, Q) ≥ H(P)**：用错误的Q一定比用正确的P花费更多比特
3. **H(P, Q) = H(P)** 当且仅当 P = Q

### 1.3 可视化理解

```python
def cross_entropy(p, q, base=2):
    """
    计算交叉熵 H(P, Q)

    参数：
        p: 真实分布
        q: 预测/编码分布
        base: 对数底
    """
    p = np.array(p)
    q = np.array(q)

    # 过滤避免log(0)
    mask = (p > 0) & (q > 0)
    return -np.sum(p[mask] * np.log(q[mask]) / np.log(base))

# 可视化：交叉熵与Q的关系
p_true = np.array([0.7, 0.3])

# Q从[0.5, 0.5]变化到[0.9, 0.1]
q1_range = np.linspace(0.01, 0.99, 100)
H_pq = []
H_qp = []

for q1 in q1_range:
    q = np.array([q1, 1-q1])
    H_pq.append(cross_entropy(p_true, q))
    H_qp.append(cross_entropy(q, p_true))

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(q1_range, H_pq, 'b-', linewidth=2, label='H(P,Q)')
plt.plot(q1_range, H_qp, 'r--', linewidth=2, label='H(Q,P)')
plt.axvline(x=0.7, color='g', linestyle=':', label='P的值 (0.7)')
plt.xlabel('Q(x=1)')
plt.ylabel('Cross Entropy')
plt.title('交叉熵的不对称性\nP=[0.7, 0.3]')
plt.legend()
plt.grid(True, alpha=0.3)

# 最小值对比
plt.subplot(122)
# 固定P，变化Q，看哪里最小
H_p_values = []
H_p = entropy(p_true)  # H(P)，下界

for q1 in q1_range:
    q = np.array([q1, 1-q1])
    H_p_values.append(cross_entropy(p_true, q))

plt.plot(q1_range, H_p_values, 'b-', linewidth=2, label='H(P,Q)')
plt.axhline(y=H_p, color='r', linestyle='--', label=f'H(P)={H_p:.4f}')
plt.axvline(x=0.7, color='g', linestyle=':', label='Q=P时最小')
plt.xlabel('Q(x=1)')
plt.ylabel('Cross Entropy')
plt.title('交叉熵最小值\nH(P,Q)最小当且仅当Q=P')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cross_entropy_visualization.png', dpi=100, bbox_inches='tight')
```

### 1.4 与熵的关系

```
H(P, Q) = H(P) + "额外的平均编码长度"
```

这个"额外"正是KL散度！

---

## 2. KL散度（Kullback-Leibler Divergence）

### 2.1 定义

**KL散度**（相对熵）：衡量分布Q偏离分布P的程度

```
KL(P || Q) = Σ P(x) log [P(x) / Q(x)]
           = H(P, Q) - H(P)
```

**关键性质**：
1. **非负性**：KL(P||Q) ≥ 0
2. **KL(P||Q) = 0 ⟺ P = Q**
3. **不对称**：KL(P||Q) ≠ KL(Q||P)
4. **不是距离度量**：不满足三角不等式

### 2.2 为什么KL散度非负？

**证明**（使用Jensen不等式）：

```
KL(P||Q) = Σ P(x) log[P(x)/Q(x)]
         = -Σ P(x) log[Q(x)/P(x)]
         = -E_P[log(Q/P)]

由于 log 是凹函数，由 Jensen 不等式：
E[log(Q/P)] ≤ log E[Q/P] = log Σ P(x)·[Q(x)/P(x)]
                          = log Σ Q(x)
                          = log 1
                          = 0

因此：
KL(P||Q) = -E_P[log(Q/P)] ≥ 0
```

### 2.3 可视化KL散度

```python
def kl_divergence(p, q, base=2):
    """
    计算KL散度 KL(P||Q)

    参数：
        p: 真实分布P
        q: 近似分布Q
        base: 对数底
    """
    p = np.array(p)
    q = np.array(q)

    # 避免除零和log(0)
    mask = (p > 0) & (q > 0)
    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]) / np.log(base))

    return kl

# 验证 KL 散度性质
print("KL散度性质验证：")
print("="*60)

# 非负性
p1 = np.array([0.5, 0.5])
q1 = np.array([0.3, 0.7])
print(f"P={p1}, Q={q1}")
print(f"KL(P||Q) = {kl_divergence(p1, q1):.4f} ≥ 0 ✓")

# P=Q时为0
print(f"\nP=Q时, KL(P||Q) = {kl_divergence(p1, p1):.4f} ✓")

# 不对称性
p2 = np.array([0.7, 0.3])
q2 = np.array([0.3, 0.7])
kl_pq = kl_divergence(p2, q2)
kl_qp = kl_divergence(q2, p2)
print(f"\n不对称性:")
print(f"P={p2}, Q={q2}")
print(f"KL(P||Q) = {kl_pq:.4f}")
print(f"KL(Q||P) = {kl_qp:.4f}")
print(f"KL(P||Q) ≠ KL(Q||P) ✓")
```

### 2.4 KL散度的几何解释

```python
# 可视化KL散度的"距离"（虽然不是真正的距离）

from matplotlib.patches import FancyArrowPatch

# 三个分布
P = np.array([0.7, 0.3])
Q1 = np.array([0.5, 0.5])  # 较接近P
Q2 = np.array([0.1, 0.9])  # 远离P

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：前向KL KL(P||Q)
ax1.bar([0, 1], P, alpha=0.5, label='P (真实)', width=0.3, align='edge')
ax1.bar([0.35, 1.35], Q1, alpha=0.5, label='Q₁', width=0.3, align='edge')
ax1.bar([0.7, 1.7], Q2, alpha=0.5, label='Q₂', width=0.3, align='edge')

ax1.set_xlabel('类别')
ax1.set_ylabel('概率')
ax1.set_title(f'前向KL散度\nKL(P||Q₁)={kl_divergence(P, Q1):.4f}, KL(P||Q₂)={kl_divergence(P, Q2):.4f}')
ax1.legend()
ax1.set_xticks([0.15, 1.15])
ax1.set_xticklabels(['类别1', '类别2'])

# 右图：解释KL散度的含义
ax2.text(0.5, 0.8, 'KL(P||Q)的含义：', fontsize=12, fontweight='bold',
         ha='center', transform=ax2.transAxes)
ax2.text(0.5, 0.65, '用Q近似P引入的"损失"', fontsize=11,
         ha='center', transform=ax2.transAxes)
ax2.text(0.5, 0.5, '编码效率损失（额外比特数）', fontsize=11,
         ha='center', transform=ax2.transAxes)
ax2.text(0.5, 0.35, '信息损失', fontsize=11,
         ha='center', transform=ax2.transAxes)

ax2.text(0.5, 0.15, '注意：不是对称的距离！', fontsize=11,
         ha='center', transform=ax2.transAxes, color='red', fontweight='bold')

ax2.axis('off')
ax2.set_title('KL散度解释')

plt.tight_layout()
plt.savefig('kl_divergence_explanation.png', dpi=100, bbox_inches='tight')
```

### 2.5 前向KL vs 反向KL

```
前向KL: KL(P||Q)  - P是真实分布，Q是近似分布
反向KL: KL(Q||P)  - Q是真实分布，P是近似分布（角色互换）
```

**关键区别**：

```python
# 前向KL vs 反向KL 的不同行为
p_true = np.array([0.9, 0.1])  # 真实分布：类别1占主导

# 尝试不同的Q
q_candidates = {
    'Q1 (平均)': np.array([0.5, 0.5]),
    'Q2 (接近P)': np.array([0.8, 0.2]),
    'Q3 (mode seeking)': np.array([1.0, 0.0]),  # Q将所有质量放在mode上
}

print("前向KL vs 反向KL：")
print("="*70)
print(f"真实分布 P = {p_true}")
print("-"*70)

for name, q in q_candidates.items():
    kl_forward = kl_divergence(p_true, q)   # KL(P||Q)
    kl_reverse = kl_divergence(q, p_true)   # KL(Q||P)

    print(f"\n{name:20s}: Q = {q}")
    print(f"  前向 KL(P||Q) = {kl_forward:.4f}")
    print(f"  反向 KL(Q||P) = {kl_reverse:.4f}")
    print(f"  差异 = {abs(kl_forward - kl_reverse):.4f}")

print("\n" + "="*70)
print("观察：")
print("- 前向KL惩罚Q在哪里的概率小但P大（覆盖所有mode）")
print("- 反向KL惩罚Q在哪里的概率大但P小（mode seeking）")
```

**模式寻求（Mode Seeking）**：

```
前向KL KL(P||Q):
├── Q需要覆盖P的所有支撑集
├── Q(x)=0 但 P(x)>0 → KL(P||Q)=∞
└── 倾向于"平均覆盖"

反向KL KL(Q||P):
├── Q可以忽略P的某些区域
├── Q(x)>0 但 P(x)=0 → KL(Q||P)=∞
└── 倾向于"找主模式"
```

---

## 3. 交叉熵与KL散度的关系

### 3.1 核心关系

```
H(P, Q) = H(P) + KL(P||Q)
```

**推导**：
```
H(P, Q) = -Σ P(x) log Q(x)
        = -Σ P(x) log Q(x) + Σ P(x) log P(x) - Σ P(x) log P(x)
        = Σ P(x) log[P(x)/Q(x)] - Σ P(x) log P(x)
        = KL(P||Q) + H(P)
```

**含义**：
- H(P)：真实分布的熵（固定）
- KL(P||Q)：用Q近似P引入的额外损失

### 3.2 为什么最小化交叉熵？

```
min H(P, Q)  = min [H(P) + KL(P||Q)]
  Q             Q
            = H(P) + min KL(P||Q)
                     Q

因为 H(P) 是常数（真实分布的熵），所以：

min H(P, Q) ⟺ min KL(P||Q)
  Q             Q
```

**结论**：最小化交叉熵等价于让Q接近P！

### 3.3 在深度学习中的应用

```python
# 分类任务的交叉熵损失

def cross_entropy_loss(y_true, y_pred, epsilon=1e-10):
    """
    计算交叉熵损失

    参数：
        y_true: 真实标签的one-hot编码 (N, K)
        y_pred: 预测概率 (N, K)

    返回：
        平均交叉熵损失
    """
    # 避免 log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 交叉熵
    ce = -np.sum(y_true * np.log(y_pred), axis=1)

    return np.mean(ce)

# 例子：3分类任务
y_true = np.array([
    [1, 0, 0],  # 样本1属于类别0
    [0, 1, 0],  # 样本2属于类别1
    [0, 0, 1],  # 样本3属于类别2
])

# 预测1：较好
y_pred_good = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
])

# 预测2：较差
y_pred_bad = np.array([
    [0.3, 0.4, 0.3],
    [0.4, 0.3, 0.3],
    [0.3, 0.4, 0.3],
])

loss_good = cross_entropy_loss(y_true, y_pred_good)
loss_bad = cross_entropy_loss(y_true, y_pred_bad)

print("交叉熵损失在分类任务中：")
print("="*60)
print(f"好预测的损失: {loss_good:.4f}")
print(f"差预测的损失: {loss_bad:.4f}")
print("\n交叉熵损失越小，预测分布越接近真实分布")
```

---

## 4. 二分类交叉熵

### 4.1 公式

对于二分类，真实分布是伯努利分布：

```
真实: P = [y, 1-y]，其中 y ∈ {0, 1}
预测: Q = [ŷ, 1-ŷ]，其中 ŷ ∈ [0, 1]

交叉熵损失:
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### 4.2 可视化

```python
# 二分类交叉熵的可视化

def binary_cross_entropy(y, y_pred, epsilon=1e-10):
    """
    二分类交叉熵

    参数：
        y: 真实标签 (0 或 1)
        y_pred: 预测概率 [0, 1]
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

y_pred_range = np.linspace(0.01, 0.99, 100)

plt.figure(figsize=(12, 5))

plt.subplot(121)
# y=1时的损失
loss_y1 = binary_cross_entropy(1, y_pred_range)
plt.plot(y_pred_range, loss_y1, 'b-', linewidth=2, label='y=1 (正类)')
plt.xlabel('预测概率 ŷ')
plt.ylabel('损失')
plt.title('真实标签 y=1 时的损失')
plt.grid(True, alpha=0.3)
plt.legend()

# 标注
plt.plot(0.5, binary_cross_entropy(1, 0.5), 'ro', markersize=10)
plt.text(0.5, binary_cross_entropy(1, 0.5) + 0.3, f'ŷ=0.5\nLoss={binary_cross_entropy(1, 0.5):.2f}',
         ha='center')

plt.subplot(122)
# y=0时的损失
loss_y0 = binary_cross_entropy(0, y_pred_range)
plt.plot(y_pred_range, loss_y0, 'r-', linewidth=2, label='y=0 (负类)')
plt.xlabel('预测概率 ŷ')
plt.ylabel('损失')
plt.title('真实标签 y=0 时的损失')
plt.grid(True, alpha=0.3)
plt.legend()

# 标注
plt.plot(0.5, binary_cross_entropy(0, 0.5), 'bo', markersize=10)
plt.text(0.5, binary_cross_entropy(0, 0.5) + 0.3, f'ŷ=0.5\nLoss={binary_cross_entropy(0, 0.5):.2f}',
         ha='center')

plt.tight_layout()
plt.savefig('binary_cross_entropy.png', dpi=100, bbox_inches='tight')
```

### 4.3 与最大似然的关系

```python
# 二分类交叉熵 = 负对数似然

print("二分类与最大似然估计：")
print("="*60)

print("\n假设 y ~ Bernoulli(ŷ)，似然函数：")
print("L = ŷ^y · (1-ŷ)^(1-y)")

print("\n对数似然：")
print("log L = y·log(ŷ) + (1-y)·log(1-ŷ)")

print("\n负对数似然（损失）：")
print("-log L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]")
print("        = 交叉熵损失")

print("\n✓ 最小化交叉熵 = 最大化似然")
```

---

## 5. 多分类交叉熵

### 5.1 公式

```
真实分布: y = [y₁, y₂, ..., y_K] (one-hot)
预测分布: ŷ = [ŷ₁, ŷ₂, ..., ŷ_K] (softmax输出)

交叉熵损失:
L = -Σ y_i · log(ŷ_i)
  = -log(ŷ_{true_class})  （one-hot情况）
```

### 5.2 Softmax与交叉熵

```python
def softmax(logits):
    """
    Softmax函数

    参数：
        logits: 未归一化的分数 [z₁, z₂, ..., z_K]

    返回：
        概率分布 [p₁, p₂, ..., p_K]
    """
    # 数值稳定版本
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def softmax_cross_entropy(logits, y_true):
    """
    Softmax + 交叉熵（联合计算，数值更稳定）

    参数：
        logits: 模型输出 (K,)
        y_true: 真实标签的one-hot (K,)
    """
    # Log-softmax技巧避免溢出
    log_probs = logits - np.max(logits) - np.log(np.sum(np.exp(logits - np.max(logits))))

    # 交叉熵
    return -np.sum(y_true * log_probs)

# 示例
logits = np.array([2.0, 1.0, 0.1])
y_true = np.array([1, 0, 0])  # 真实类别0

probs = softmax(logits)
loss = softmax_cross_entropy(logits, y_true)

print("Softmax交叉熵示例：")
print("="*60)
print(f"Logits: {logits}")
print(f"Softmax概率: {probs}")
print(f"真实标签: {y_true}")
print(f"交叉熵损失: {loss:.4f}")

print("\n等价于 -log(预测正确类别的概率):")
print(f"-log(p_0) = {-np.log(probs[0]):.4f} ✓")
```

### 5.3 Softmax的信息论解释

```
Softmax = 最大熵原理

在约束 Σ ŷ_i · z_i = const 的条件下，
熵最大的分布是指数族分布：

ŷ_i ∝ exp(z_i)

这与Softmax完全一致！
```

---

## 6. KL散度在深度学习中的应用

### 6.1 VAE（变分自编码器）

```python
# VAE的损失函数

def vae_loss(x, x_reconstructed, z_mean, z_log_var, beta=1.0):
    """
    VAE损失 = 重建损失 + β·KL散度

    参数：
        x: 原始输入
        x_reconstructed: 重建输出
        z_mean: 潜在变量均值
        z_log_var: 潜在变量对数方差
        beta: KL散度权重（β-VAE）
    """
    # 重建损失（交叉熵或MSE）
    reconstruction_loss = np.mean((x - x_reconstructed)**2)

    # KL散度：KL(N(μ,σ²) || N(0,1))
    # 公式：-0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * np.sum(1 + z_log_var - z_mean**2 - np.exp(z_log_var))

    return reconstruction_loss + beta * kl_loss

print("VAE损失函数组成：")
print("="*60)
print("L = 重建损失 + KL(q(z|x) || p(z))")
print("  = 重建损失 + KL(近似后验 || 先验)")
print("\n作用：")
print("- 重建损失：保证重建质量")
print("- KL散度：让潜在分布接近先验（标准正态）")
```

### 6.2 知识蒸馏

```python
# 知识蒸馏：让小模型学习大模型的软标签

def knowledge_distillation_loss(y_student, y_teacher, y_true, temperature=3.0, alpha=0.7):
    """
    知识蒸馏损失

    参数：
        y_student: 学生模型logits
        y_teacher: 教师模型logits
        y_true: 真实标签
        temperature: 蒸馏温度
        alpha: 软标签权重
    """
    # 软标签损失（KL散度）
    p_student = softmax(y_student / temperature)
    p_teacher = softmax(y_teacher / temperature)

    # KL散度（软标签的知识迁移）
    kl_loss = kl_divergence(p_teacher, p_student)

    # 硬标签损失（传统交叉熵）
    ce_loss = cross_entropy_loss(y_true, softmax(y_student))

    # 综合损失
    total_loss = alpha * (temperature**2) * kl_loss + (1 - alpha) * ce_loss

    return total_loss

print("知识蒸馏：")
print("="*60)
print("学生模型学习：")
print("- α·KL(教师软标签 || 学生输出)：学习教师的知识")
print("- (1-α)·CE(真实标签, 学生输出)：学习真实标签")
print("\n软标签包含类别间的相似性信息！")
```

### 6.3 策略优化的熵正则化

```python
# 强化学习中的熵正则化

def policy_loss_with_entropy(log_probs, advantages, entropy, entropy_coef=0.01):
    """
    带熵正则化的策略损失

    参数：
        log_probs: 动作的对数概率
        advantages: 优势函数
        entropy: 策略的熵
        entropy_coef: 熵系数
    """
    # 策略梯度损失（负号因为要最大化）
    policy_loss = -np.mean(log_probs * advantages)

    # 熵奖励（最大化熵，鼓励探索）
    entropy_bonus = -entropy_coef * np.mean(entropy)

    return policy_loss + entropy_bonus

print("强化学习的熵正则化：")
print("="*60)
print("L = -E[log π(a|s) · A(s,a)] - β·H(π)")
print("  = 策略损失 - 熵奖励")
print("\n作用：")
print("- 高熵 = 随机策略 = 探索多样")
print("- 防止过早收敛到次优确定性策略")
```

---

## 7. 数值稳定性技巧

### 7.1 Log-Sum-Exp技巧

```python
def log_sum_exp(x):
    """
    数值稳定的 log(sum(exp(x)))

    避免溢出：
    log Σ exp(x_i) = x_max + log Σ exp(x_i - x_max)
    """
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))

# 对比：不稳定 vs 稳定
x_large = np.array([1000, 1001, 1002])

print("数值稳定性示例：")
print("="*60)
try:
    unstable = np.log(np.sum(np.exp(x_large)))
    print(f"不稳定版本: {unstable}")
except OverflowError:
    print("不稳定版本: 溢出！")

stable = log_sum_exp(x_large)
print(f"稳定版本: {stable}")
```

### 7.2 避免log(0)

```python
def safe_log(x, epsilon=1e-10):
    """安全的log函数，避免log(0)"""
    return np.log(np.maximum(x, epsilon))

def safe_kl_divergence(p, q, epsilon=1e-10):
    """安全的KL散度计算"""
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    p = p / p.sum()  # 重新归一化
    q = q / q.sum()

    return np.sum(p * safe_log(p / q))
```

---

## 8. 总结

### 核心公式对照

| 概念 | 公式 | 含义 | 应用 |
|------|------|------|------|
| **交叉熵** | H(P,Q)=-ΣP log Q | 用Q编码P的平均比特数 | 分类损失 |
| **KL散度** | KL(P\|\|Q)=ΣP log(P/Q) | Q偏离P的程度 | VAE、知识蒸馏 |
| **关系** | H(P,Q)=H(P)+KL(P\|\|Q) | 核心 | 最小化交叉熵=让Q接近P |

### 为什么用交叉熵损失？

```
理论链条：
1. 最大似然估计 → 对数似然
2. 负对数似然 → 交叉熵
3. 最小化交叉熵 → 最小化KL散度
4. 最小化KL散度 → 让预测分布接近真实分布

✓ 完美！
```

### 前向KL vs 反向KL

| 类型 | KL(P\|\|Q) 前向 | KL(Q\|\|P) 反向 |
|------|----------------|----------------|
| **行为** | 覆盖所有mode | 找主mode |
| **Q=0且P>0** | ∞ (致命) | 可以 |
| **P=0且Q>0** | 可以 | ∞ (致命) |
| **应用** | 标准训练、VAE | 变分推断、GAN |

### 深度学习损失函数总结

```python
loss_functions = {
    '二分类': 'Binary Cross Entropy = -[y·log(ŷ) + (1-y)·log(1-ŷ)]',
    '多分类': 'Categorical Cross Entropy = -Σ y_i·log(ŷ_i)',
    'VAE': 'Reconstruction Loss + KL(q(z|x)||p(z))',
    '知识蒸馏': 'α·KL(teacher||student) + (1-α)·CE(true, student)',
    '策略优化': '-E[log π(a|s)·A] - β·H(π) (熵正则化)',
}
```

---

## 下一步

继续学习 [03_mutual_information.md](03_mutual_information.md)，了解互信息在特征选择和信息瓶颈理论中的应用。
