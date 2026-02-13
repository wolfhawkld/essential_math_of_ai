# 信息量与熵

## 为什么学习熵？

熵是信息论的核心概念，在深度学习中无处不在：

### 现实问题

```
问题1：如何衡量不确定性？
├── 抛硬币 vs 掷骰子，哪个更不确定？
└── 解决：熵量化不确定性

问题2：损失函数如何设计？
├── 为什么分类用交叉熵？
└── 解决：最小化预测分布和真实分布的KL散度

问题3：模型如何表达不确定性？
├── 贝叶斯神经网络
└── 解决：预测分布的熵 = 模型不确定度

问题4：决策树如何选择分裂特征？
├── 信息增益 = 熵的减少
└── 解决：选择熵降低最多的特征
```

---

## 1. 信息量（自信息）

### 1.1 直觉：什么是信息？

**思考**：
- "明天太阳从东边升起" - 有信息量吗？**没有**（必然事件）
- "明天会下雨" - 有信息量吗？**有**（不确定事件）
- "明天会下钻石雨" - 信息量？**巨大**（极不可能）

**结论**：信息量与概率相关。概率越小，信息量越大。

### 1.2 定义

**信息量（自信息）**：

```
I(x) = -log₂ P(x)    (单位：bit，以2为底)
I(x) = -ln P(x)      (单位：nat，以e为底)
```

**性质**：
1. **非负性**：I(x) ≥ 0
2. **单调性**：P(x)越小，I(x)越大
3. **确定性事件**：P=1时，I=0（无信息）
4. **可加性**：I(x,y) = I(x) + I(y)，如果x,y独立

### 1.3 为什么用对数？

```python
import numpy as np
import matplotlib.pyplot as plt

# 可视化信息量函数
p = np.linspace(0.01, 1, 100)
information = -np.log2(p)

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(p, information, 'b-', linewidth=2)
plt.xlabel('Probability P(x)')
plt.ylabel('Information I(x) = -log₂ P(x)')
plt.title('信息量 vs 概率')
plt.grid(True, alpha=0.3)

# 标注几个点
examples = [(1.0, '必然事件'), (0.5, '抛硬币'), (0.1, '不太可能'), (0.01, '极不可能')]
for prob, label in examples:
    info = -np.log2(prob)
    plt.plot(prob, info, 'ro', markersize=8)
    plt.text(prob, info + 0.3, f'{label}\nI={info:.2f} bits',
             ha='center', fontsize=9)

# 验证可加性
plt.subplot(122)
# 两个独立事件
p1 = np.linspace(0.1, 1, 50)
p2 = 0.5
p_joint = p1 * p2  # P(x,y) = P(x)P(y)

I_individual = -np.log2(p1) - np.log2(p2)
I_joint = -np.log2(p_joint)

plt.plot(p1, I_individual, 'b-', linewidth=2, label='I(x) + I(y)')
plt.plot(p1, I_joint, 'r--', linewidth=2, label='I(x,y)')
plt.xlabel('P(x)')
plt.ylabel('Information (bits)')
plt.title('可加性验证：I(x,y) = I(x) + I(y)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('information_content.png', dpi=100, bbox_inches='tight')
```

**对数的好处**：
1. **线性可加**：独立事件的信息量相加
2. **符合直觉**：小概率事件 → 大信息量
3. **数值稳定**：概率乘法 → 对数加法

### 1.4 例子

```python
# 计算各种事件的信息量

def information(p, base=2):
    """
    计算信息量

    参数：
        p: 事件概率
        base: 对数底（2=bit, e=nat, 10=Hartley）
    """
    if p <= 0:
        return float('inf')
    return -np.log(p) / np.log(base)

# 例子
examples = {
    '必然事件（明天太阳升起）': 1.0,
    '抛硬币正面朝上': 0.5,
    '掷骰子得到6': 1/6,
    '抽中彩票（假设概率0.000001）': 1e-6,
}

print("事件的信息量：")
print("="*60)
for event, prob in examples.items():
    info = information(prob)
    print(f"{event:30s}: P={prob:.6f}, I={info:.2f} bits")

# 输出：
# 必然事件（明天太阳升起）    : P=1.000000, I=0.00 bits
# 抛硬币正面朝上            : P=0.500000, I=1.00 bits
# 掷骰子得到6              : P=0.166667, I=2.58 bits
# 抽中彩票（假设概率0.000001）: P=0.000001, I=19.93 bits
```

**直觉**：
- 1 bit信息 = 回答一个yes/no问题
- 抛硬币结果 = 1 bit（需要1个二进制位编码）
- 掷骰子结果 ≈ 2.58 bits（需要约3个二进制位）

---

## 2. 熵（Entropy）

### 2.1 定义

**熵**：随机变量X的平均信息量

```
H(X) = -Σ P(x) log P(x)
     = E[I(x)]
     = Σ P(x) I(x)
```

**直观理解**：
- 熵 = 不确定性
- 熵 = 描述随机变量所需的平均比特数
- 熵 = 系统的"混乱程度"

### 2.2 熵的性质

```python
def entropy(probs, base=2):
    """
    计算离散分布的熵

    参数：
        probs: 概率分布 [p1, p2, ..., pn]
        base: 对数底

    返回：
        H(X) 熵值
    """
    probs = np.array(probs)
    # 过滤掉0概率（0*log(0) = 0）
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs) / np.log(base))

# 验证熵的各种性质
print("熵的性质演示：")
print("="*60)

# 性质1：熵非负
print("\n1. 熵非负")
p1 = [0.5, 0.5]
print(f"   抛硬币分布 {p1}: H = {entropy(p1):.4f} bits")

# 性质2：均匀分布熵最大
print("\n2. 对于n个取值，均匀分布熵最大")
for n in [2, 4, 6, 8]:
    p_uniform = [1/n] * n
    print(f"   {n}个取值均匀分布: H_max = {entropy(p_uniform):.4f} bits = log₂({n})")

# 性质3：确定性分布熵为0
print("\n3. 确定性分布熵为0")
p_certain = [1.0, 0.0, 0.0]
print(f"   确定分布 {p_certain}: H = {entropy(p_certain):.4f} bits")

# 性质4：熵随不确定性增加而增加
print("\n4. 熵随不确定性增加")
distributions = {
    '完全确定': [1.0, 0.0],
    '略微不确定': [0.9, 0.1],
    '中等不确定': [0.7, 0.3],
    '很不确定': [0.6, 0.4],
    '完全不确定': [0.5, 0.5],
}

for name, probs in distributions.items():
    print(f"   {name:12s} {probs}: H = {entropy(probs):.4f} bits")
```

### 2.3 可视化：熵与分布的关系

```python
import numpy as np
import matplotlib.pyplot as plt

# 对于二元分布 P = [p, 1-p]
p = np.linspace(0.001, 0.999, 1000)
H = -p * np.log2(p) - (1-p) * np.log2(1-p)

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(p, H, 'b-', linewidth=2)
plt.xlabel('p (P(X=1) = p)')
plt.ylabel('H(X) (bits)')
plt.title('二元分布的熵\nH(p) = -p log₂ p - (1-p) log₂(1-p)')
plt.grid(True, alpha=0.3)

# 标注关键点
plt.plot(0.5, 1.0, 'ro', markersize=10, label='最大熵 (p=0.5)')
plt.plot([0.1, 0.9], [entropy([0.1, 0.9]), entropy([0.9, 0.1])],
         'g^', markersize=8, label='对称性')
plt.legend()

# 三元分布：可视化熵曲面
plt.subplot(122)
from mpl_toolkits.mplot3d import Axes3D

p1 = np.linspace(0.01, 0.98, 50)
p2 = np.linspace(0.01, 0.98, 50)
P1, P2 = np.meshgrid(p1, p2)

# p3 = 1 - p1 - p2，必须非负
P3 = 1 - P1 - P2
valid = P3 > 0.01

H_3d = np.zeros_like(P1)
H_3d[valid] = -P1[valid] * np.log2(P1[valid]) \
              - P2[valid] * np.log2(P2[valid]) \
              - P3[valid] * np.log2(P3[valid])
H_3d[~valid] = np.nan

fig = plt.gcf()
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(P1, P2, H_3d, cmap='viridis', alpha=0.8)
ax.set_xlabel('p₁')
ax.set_ylabel('p₂')
ax.set_zlabel('H(X)')
ax.set_title('三元分布的熵')

plt.tight_layout()
plt.savefig('entropy_visualization.png', dpi=100, bbox_inches='tight')
```

### 2.4 联合熵和条件熵

**联合熵**：
```
H(X, Y) = -Σ Σ P(x,y) log P(x,y)
```

含义：描述X和Y共同所需的平均比特数

**条件熵**：
```
H(Y|X) = -Σ Σ P(x,y) log P(y|x)
       = H(X,Y) - H(X)
```

含义：已知X后，Y剩余的不确定性

**链式法则**：
```
H(X,Y) = H(X) + H(Y|X)
```

```python
# 实例：计算联合熵和条件熵

def joint_entropy(joint_probs, base=2):
    """
    计算联合熵

    参数：
        joint_probs: 联合概率矩阵 P(x,y)
    """
    joint_probs = np.array(joint_probs)
    probs = joint_probs[joint_probs > 0]
    return -np.sum(probs * np.log(probs) / np.log(base))

def conditional_entropy(joint_probs, marginal_x, base=2):
    """
    计算条件熵 H(Y|X) = H(X,Y) - H(X)

    参数：
        joint_probs: 联合概率 P(x,y)
        marginal_x: X的边缘分布
    """
    H_XY = joint_entropy(joint_probs, base)
    H_X = entropy(marginal_x, base)
    return H_XY - H_X

# 例子：天气(X)和是否带伞(Y)
#        晴天  多云  下雨
# 不带伞  0.4   0.1   0.0
# 带伞    0.1   0.2   0.2

joint = np.array([
    [0.4, 0.1, 0.0],
    [0.1, 0.2, 0.2]
])

# 计算边缘分布
p_x = joint.sum(axis=0)  # 天气分布
p_y = joint.sum(axis=1)  # 带伞分布

H_X = entropy(p_x)
H_Y = entropy(p_y)
H_XY = joint_entropy(joint)
H_Y_given_X = conditional_entropy(joint, p_x)

print("天气和带伞的熵分析：")
print("="*60)
print(f"H(天气) = {H_X:.4f} bits")
print(f"H(带伞) = {H_Y:.4f} bits")
print(f"H(天气, 带伞) = {H_XY:.4f} bits")
print(f"H(带伞|天气) = {H_Y_given_X:.4f} bits")
print(f"\n验证链式法则：H(X,Y) = H(X) + H(Y|X)")
print(f"H(X) + H(Y|X) = {H_X + H_Y_given_X:.4f} bits ✓")
```

---

## 3. 熵的应用

### 3.1 决策树：信息增益

**信息增益** = 熵的减少

```
IG(Y, X) = H(Y) - H(Y|X)
```

含义：知道X后，Y的不确定性降低了多少

```python
# 决策树的信息增益计算

def information_gain(y_probs, conditional_probs):
    """
    计算信息增益

    参数：
        y_probs: Y的先验分布
        conditional_probs: 字典 {x: P(Y|x)}
    """
    H_Y = entropy(y_probs)

    # 计算条件熵
    H_Y_given_X = 0
    for x, probs in conditional_probs.items():
        # P(X=x)
        p_x = sum(probs) / sum([sum(p) for p in conditional_probs.values()])
        # H(Y|X=x)
        H_Y_given_x = entropy(probs)
        H_Y_given_X += p_x * H_Y_given_x

    return H_Y - H_Y_given_X

# 例子：是否打网球
# 特征：天气（晴天、阴天、下雨）
# 标签：打网球（是、否）

# 先验分布：打网球的比例
y_prior = [9/14, 5/14]  # [是, 否]

# 条件分布：给定天气后的打网球分布
# 晴天：2次打，3次不打
# 阴天：4次打，0次不打
# 下雨：3次打，2次不打
conditional = {
    '晴天': [2/5, 3/5],
    '阴天': [4/4, 0/4],
    '下雨': [3/5, 2/5]
}

ig = information_gain(y_prior, conditional)

print(f"信息增益 IG(打网球, 天气) = {ig:.4f} bits")
print("解读：知道天气后，对是否打网球的不确定性减少了{:.4f} bits".format(ig))
```

### 3.2 数据压缩

**熵给出了平均编码长度的下界**：

```
最优平均编码长度 ≥ H(X)
```

```python
# 霍夫曼编码 vs 熵

# 符号概率分布
symbols = {
    'A': 0.4,
    'B': 0.3,
    'C': 0.2,
    'D': 0.1
}

# 熵
H = entropy(list(symbols.values()))
print(f"熵 = {H:.4f} bits")

# 霍夫曼编码（简单示例）
huffman_codes = {
    'A': '0',      # 1 bit
    'B': '10',     # 2 bits
    'C': '110',    # 3 bits
    'D': '111'     # 3 bits
}

# 平均编码长度
avg_length = sum(symbols[s] * len(huffman_codes[s]) for s in symbols)
print(f"霍夫曼平均编码长度 = {avg_length:.4f} bits")
print(f"熵下界 = {H:.4f} bits")
print(f"效率 = {H/avg_length*100:.2f}%")
```

### 3.3 模型不确定性

在深度学习中，预测分布的熵表示模型的不确定性：

```python
# 模型预测的不确定性

def predictive_uncertainty(predictions):
    """
    计算预测分布的熵（模型不确定性）

    参数：
        predictions: 预测概率分布 [p1, p2, ..., pk]
    """
    return entropy(predictions)

# 场景3个样本的分类预测
predictions = {
    '样本1（高置信）': [0.95, 0.03, 0.02],
    '样本2（中等置信）': [0.6, 0.3, 0.1],
    '样本3（低置信）': [0.4, 0.35, 0.25],  # 接近均匀分布
}

print("模型预测的不确定性：")
print("="*60)
for name, probs in predictions.items():
    uncertainty = predictive_uncertainty(probs)
    predicted_class = np.argmax(probs)
    print(f"{name:20s}: 熵={uncertainty:.4f} bits, 预测类别={predicted_class}")

# 高熵 = 模型不确定
# 低熵 = 模型确信
```

### 3.4 最大熵原理

在约束条件下，选择熵最大的分布（最不确定的分布）。

**例子**：给定均值和方差，选什么分布？

答案：高斯分布！

```python
# 最大熵原理示例
# 约束：给定期望，熵最大的分布是什么？

def max_entropy_distribution_with_mean(mean, support):
    """
    给定期望值，求熵最大的分布

    使用指数分布族：p(x) ∝ exp(λx)
    """
    from scipy.optimize import minimize_scalar

    # 这需要求解优化问题，这里简化展示
    # 对于给定支持集，答案是指数分布（或几何分布）

    pass

# 常见结果
print("最大熵原理常见结果：")
print("="*60)
print("约束条件                → 最大熵分布")
print("-"*60)
print("无约束                  → 均匀分布")
print("给定期望（正数）         → 指数分布")
print("给定期望和方差           → 高斯分布")
print("给定范围[a,b]           → 均匀分布")
print("给定期望（离散）         → 几何分布")
```

---

## 4. 连续随机变量的熵

### 4.1 微分熵

对于连续随机变量，**微分熵**：

```
h(X) = -∫ f(x) log f(x) dx
```

**注意**：微分熵可以为负！

```python
# 高斯分布的微分熵

def gaussian_entropy(sigma):
    """
    高斯分布 N(μ, σ²) 的微分熵

    h(X) = (1/2) log(2πeσ²)  (nat)
         = (1/2) log₂(2πeσ²)  (bit)
    """
    return 0.5 * np.log2(2 * np.pi * np.e * sigma**2)

# 不同方差的高斯分布
sigmas = [0.5, 1.0, 2.0, 5.0]

print("高斯分布的微分熵：")
print("="*60)
for sigma in sigmas:
    h = gaussian_entropy(sigma)
    print(f"σ = {sigma:.1f}: h(X) = {h:.4f} bits")

# 当σ很小时，熵为负！
print(f"\nσ = 0.1: h(X) = {gaussian_entropy(0.1):.4f} bits (负值！)")
```

**为什么微分熵可以为负？**

因为连续变量的"信息量"是相对于测度的，不是绝对的。

---

## 5. 交叉熵（预告）

**交叉熵**衡量用分布Q编码分布P所需的平均比特数：

```
H(P, Q) = -Σ P(x) log Q(x)
```

将在下一章详细讲解，重点是：

```
为什么深度学习用交叉熵损失？
→ 最小化 H(P_truth, P_model)
→ 让模型分布接近真实分布
```

---

## 6. 总结

### 核心概念对照表

| 概念 | 定义 | 含义 | 应用 |
|------|------|------|------|
| **信息量** | I(x) = -log P(x) | 单个事件的信息 | 编码理论 |
| **熵** | H(X) = E[I(X)] | 平均信息量/不确定性 | 决策树、模型不确定度 |
| **联合熵** | H(X,Y) | X和Y共同的不确定性 | 多变量分析 |
| **条件熵** | H(Y\|X) | 已知X后Y的不确定性 | 链式法则 |
| **信息增益** | IG = H(Y) - H(Y\|X) | 熵的减少 | 特征选择、决策树 |

### 关键性质

```
✓ 熵 ≥ 0
✓ 熵最大当且仅当分布均匀
✓ 熵 = 0 当且仅当分布确定
✓ H(X,Y) = H(X) + H(Y|X)
✓ H(X,Y) ≤ H(X) + H(Y)  (独立时取等)
```

### 信息单位

```
bit  = log₂  (二进制，最常用)
nat  = ln    (自然对数，理论推导)
Hartley = log₁₀ (十进制)
```

转换关系：
```
1 nat = log₂ e ≈ 1.443 bits
1 Hartley = log₂ 10 ≈ 3.322 bits
```

### 深度学习中的熵

```python
# 应用场景总结

applications = {
    '交叉熵损失': '最小化预测分布与真实分布的差异',
    '模型不确定性': '预测分布的熵 = 不确定度',
    '决策树': '信息增益 = 特征重要性',
    '数据压缩': '熵 = 编码下界',
    '最大熵模型': '在约束下选择最不确定的分布',
    '变分推断': '最大化证据下界（ELBO）',
}
```

---

## 下一步

继续学习 [02_cross_entropy_kl.md](02_cross_entropy_kl.md)，理解交叉熵和KL散度如何成为深度学习损失函数的核心。
