# 互信息

## 为什么学习互信息？

互信息衡量两个变量之间的相关性，在深度学习中有重要应用：

### 实际应用

```
应用1：特征选择
├── 选哪个特征对预测最有用？
└── 选择与目标变量互信息最大的特征

应用2：信息瓶颈理论
├── 神经网络学到了什么表示？
└── 最大化表示与输出的互信息，最小化表示与输入的互信息

应用3：对比学习
├── 自监督学习如何训练？
└── 最大化两个视角的互信息

应用4：因果推断
├── X和Y有因果关系吗？
└── 互信息可以量化依赖程度
```

---

## 1. 互信息的定义

### 1.1 直觉：两个变量共享多少信息？

**思考**：
- 知道天气能告诉你多少关于是否带伞的信息？
- 知道一个人的年龄能告诉你多少关于收入的信息？
- 知道句子的前半部分能告诉你多少关于后半部分的信息？

**互信息**：衡量知道X后，对Y的不确定性降低了多少。

### 1.2 三种等价定义

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义1：KL散度形式
print("定义1：KL散度形式")
print("="*60)
print("I(X;Y) = KL(P(X,Y) || P(X)P(Y))")
print("含义：联合分布与独立假设分布的差异")
print("     如果X和Y独立，P(X,Y) = P(X)P(Y)，MI=0")

print("\n定义2：熵的形式")
print("="*60)
print("I(X;Y) = H(X) + H(Y) - H(X,Y)")
print("含义：两个变量总信息量 - 联合信息量")
print("     = 共享的信息量")

print("\n定义3：条件熵形式")
print("="*60)
print("I(X;Y) = H(X) - H(X|Y)")
print("      = H(Y) - H(Y|X)")
print("含义：知道Y后，X的不确定性减少了多少")
```

### 1.3 实现与计算

```python
def mutual_information(joint_probs, base=2):
    """
    计算互信息 I(X;Y)

    参数：
        joint_probs: 联合概率矩阵 P(X,Y)
        base: 对数底

    返回：
        互信息值
    """
    joint_probs = np.array(joint_probs)

    # 计算边缘分布
    p_x = joint_probs.sum(axis=1)  # P(X)
    p_y = joint_probs.sum(axis=0)  # P(Y)

    # I(X;Y) = ΣΣ P(x,y) log[P(x,y)/(P(x)P(y))]
    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if joint_probs[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += joint_probs[i, j] * np.log(
                    joint_probs[i, j] / (p_x[i] * p_y[j])
                ) / np.log(base)

    return mi

# 例子：天气和带伞
joint = np.array([
    # 不带伞  带伞
    [0.4, 0.1],   # 晴天
    [0.1, 0.2],   # 多云
    [0.0, 0.2],   # 下雨
])

mi = mutual_information(joint)
print(f"\n互信息示例：天气 vs 是否带伞")
print(f"I(天气; 带伞) = {mi:.4f} bits")
print("含义：知道天气后，对是否带伞的不确定性减少了{:.4f} bits".format(mi))
```

---

## 2. 互信息的性质

### 2.1 基本性质

```python
# 验证互信息的性质

joint = np.array([
    [0.4, 0.1],
    [0.1, 0.2],
    [0.0, 0.2]
])

p_x = joint.sum(axis=1)
p_y = joint.sum(axis=0)

# 计算各项熵
def entropy(probs, base=2):
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs) / np.log(base))

def joint_entropy(joint_probs, base=2):
    joint_probs = np.array(joint_probs)
    probs = joint_probs[joint_probs > 0]
    return -np.sum(probs * np.log(probs) / np.log(base))

H_X = entropy(p_x)
H_Y = entropy(p_y)
H_XY = joint_entropy(joint)
I_XY = mutual_information(joint)

print("互信息的性质验证：")
print("="*60)
print(f"H(X) = {H_X:.4f} bits")
print(f"H(Y) = {H_Y:.4f} bits")
print(f"H(X,Y) = {H_XY:.4f} bits")
print(f"I(X;Y) = {I_XY:.4f} bits")

print("\n性质1：非负性")
print(f"I(X;Y) = {I_XY:.4f} ≥ 0 ✓")

print("\n性质2：对称性")
I_YX = mutual_information(joint.T)  # 转置后计算
print(f"I(X;Y) = I(Y;X) → {I_XY:.4f} = {I_YX:.4f} ✓")

print("\n性质3：I(X;Y) = H(X) + H(Y) - H(X,Y)")
print(f"H(X) + H(Y) - H(X,Y) = {H_X + H_Y - H_XY:.4f}")
print(f"I(X;Y) = {I_XY:.4f} ✓")

print("\n性质4：I(X;Y) ≤ min(H(X), H(Y))")
print(f"I(X;Y) = {I_XY:.4f} ≤ min({H_X:.4f}, {H_Y:.4f}) ✓")

print("\n性质5：I(X;Y) = H(X) - H(X|Y)")
H_X_given_Y = H_XY - H_Y
print(f"H(X) - H(X|Y) = {H_X - H_X_given_Y:.4f}")
print(f"I(X;Y) = {I_XY:.4f} ✓")
```

### 2.2 特殊情况

```python
# 情况1：X和Y完全独立
print("\n情况1：X和Y独立")
print("="*60)

# 构造独立分布
p_x_ind = np.array([0.6, 0.4])
p_y_ind = np.array([0.7, 0.3])
joint_independent = np.outer(p_x_ind, p_y_ind)

mi_independent = mutual_information(joint_independent)
print(f"P(X) = {p_x_ind}")
print(f"P(Y) = {p_y_ind}")
print(f"P(X,Y) = P(X)P(Y) (独立)")
print(f"I(X;Y) = {mi_independent:.6f} ≈ 0 ✓")

# 情况2：X和Y完全相关
print("\n情况2：X和Y完全相关")
print("="*60)

joint_dependent = np.array([
    [0.5, 0.0],
    [0.0, 0.5]
])

mi_dependent = mutual_information(joint_dependent)
print(f"联合分布：对角矩阵（完全相关）")
print(f"I(X;Y) = {mi_dependent:.4f} bits")
print(f"此时 I(X;Y) = H(X) = H(Y) = {entropy([0.5, 0.5]):.4f} ✓")

# 情况3：一个变量确定另一个
print("\n情况3：Y完全由X决定")
print("="*60)

joint_deterministic = np.array([
    [0.3, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.2]
])

mi_det = mutual_information(joint_deterministic)
H_X_det = entropy([0.3, 0.5, 0.2])
print(f"联合分布：对角矩阵")
print(f"I(X;Y) = {mi_det:.4f} bits")
print(f"H(X) = {H_X_det:.4f} bits")
print(f"I(X;Y) = H(X) = H(Y) ✓")
```

---

## 3. 互信息的维恩图

### 3.1 可视化理解

```python
# 用维恩图理解互信息

from matplotlib_venn import venn2

plt.figure(figsize=(14, 5))

# 左图：概念关系
plt.subplot(131)
venn = venn2(subsets=(3, 3, 2), set_labels=('H(X)', 'H(Y)'))
plt.title('熵与互信息的关系\n' +
          '交集 = I(X;Y)\n' +
          '左侧非交集 = H(X|Y)\n' +
          '右侧非交集 = H(Y|X)')

# 中图：数值示例
plt.subplot(132)
joint = np.array([[0.4, 0.1], [0.1, 0.2], [0.0, 0.2]])
p_x = joint.sum(axis=1)
p_y = joint.sum(axis=0)

H_X = entropy(p_x)
H_Y = entropy(p_y)
H_XY = joint_entropy(joint)
I_XY = mutual_information(joint)

H_X_given_Y = H_X - I_XY
H_Y_given_X = H_Y - I_XY

# 创建简单的条形图
labels = ['H(X)', 'H(Y)', 'I(X;Y)', 'H(X|Y)', 'H(Y|X)', 'H(X,Y)']
values = [H_X, H_Y, I_XY, H_X_given_Y, H_Y_given_X, H_XY]

bars = plt.bar(range(len(labels)), values, color=['blue', 'red', 'purple', 'lightblue', 'lightcoral', 'gray'])
plt.xticks(range(len(labels)), labels, rotation=45)
plt.ylabel('Entropy (bits)')
plt.title('熵的各项关系')

# 添加数值标签
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# 右图：关系公式
plt.subplot(133)
plt.text(0.5, 0.85, '熵的关系链式法则', fontsize=12, fontweight='bold',
         ha='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.7, 'H(X,Y) = H(X) + H(Y|X)', fontsize=11,
         ha='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.55, 'H(X,Y) = H(Y) + H(X|Y)', fontsize=11,
         ha='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.35, 'I(X;Y) = H(X) - H(X|Y)', fontsize=11,
         ha='center', transform=plt.gca().transAxes, color='purple')
plt.text(0.5, 0.2, 'I(X;Y) = H(Y) - H(Y|X)', fontsize=11,
         ha='center', transform=plt.gca().transAxes, color='purple')
plt.text(0.5, 0.05, 'I(X;Y) = H(X) + H(Y) - H(X,Y)', fontsize=11,
         ha='center', transform=plt.gca().transAxes, color='purple')

plt.axis('off')

plt.tight_layout()
plt.savefig('mutual_information_venn.png', dpi=100, bbox_inches='tight')
```

### 3.2 信息流图

```
        H(X)                H(Y)
    ┌──────────┐        ┌──────────┐
    │          │        │          │
    │   ┌──────┴────────┴──────┐   │
    │   │    I(X;Y)           │   │
    │   │  (共享信息)          │   │
    │   └──────┬────────┬──────┘   │
    │          │        │          │
    │   H(X|Y) │        │ H(Y|X)   │
    │  (X独有) │        │ (Y独有)  │
    └──────────┘        └──────────┘

解释：
- I(X;Y)：X和Y共享的信息
- H(X|Y)：知道Y后，X剩余的不确定性
- H(Y|X)：知道X后，Y剩余的不确定性
```

---

## 4. 条件互信息

### 4.1 定义

**条件互信息**：在已知Z的条件下，X和Y的互信息

```
I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
         = H(Y|Z) - H(Y|X,Z)
         = Σ P(z) I(X;Y|Z=z)
```

**含义**：在给定Z后，X还能提供关于Y的多少信息。

### 4.2 链式法则

```
I(X;Y,Z) = I(X;Y) + I(X;Z|Y)
```

推广到多个变量：

```
I(X;Y₁,Y₂,...,Yₙ) = I(X;Y₁) + I(X;Y₂|Y₁) + ... + I(X;Yₙ|Y₁,...,Yₙ₋₁)
```

```python
# 条件互信息的计算示例

def conditional_mutual_information(joint_xyz, base=2):
    """
    计算条件互信息 I(X;Y|Z)

    参数：
        joint_xyz: 三维联合分布 P(X,Y,Z)
    """
    # 这里简化展示概念
    # 实际实现需要正确处理三维数组
    pass

print("条件互信息示例：")
print("="*60)
print("问题：在已知'季节'的条件下，")
print("      '天气'还能提供多少关于'穿什么衣服'的信息？")
print()
print("I(天气; 穿衣 | 季节)")
print("  = H(天气 | 季节) - H(天气 | 穿衣, 季节)")
print()
print("直觉：")
print("- 如果季节已知，天气的不确定性降低")
print("- 穿衣选择也受季节影响")
print("- 条件互信息衡量'天气的额外信息'")
```

---

## 5. 应用1：特征选择

### 5.1 信息增益

**信息增益** = 互信息

```
IG(Y, X) = I(Y; X) = H(Y) - H(Y|X)
```

选择信息增益最大的特征。

```python
# 特征选择示例

def information_gain(y, feature_values):
    """
    计算特征的信息增益

    参数：
        y: 目标变量值
        feature_values: 特征值

    返回：
        信息增益
    """
    # 构建联合分布
    unique_y = np.unique(y)
    unique_feat = np.unique(feature_values)

    n = len(y)
    joint = np.zeros((len(unique_feat), len(unique_y)))

    for i, feat_val in enumerate(unique_feat):
        for j, y_val in enumerate(unique_y):
            joint[i, j] = np.sum((feature_values == feat_val) & (y == y_val)) / n

    return mutual_information(joint)

# 示例数据：预测是否打网球
data = {
    '天气': ['晴', '晴', '阴', '雨', '雨', '雨', '阴', '晴',
             '晴', '雨', '晴', '阴', '阴', '雨'],
    '温度': ['热', '热', '热', '温', '凉', '凉', '凉', '温',
             '凉', '温', '温', '温', '热', '温'],
    '湿度': ['高', '高', '高', '高', '正常', '正常', '正常', '高',
             '正常', '正常', '正常', '高', '正常', '高'],
    '打球': ['否', '否', '是', '是', '是', '否', '是', '否',
             '是', '是', '是', '是', '是', '否']
}

y = np.array([1 if v == '是' else 0 for v in data['打球']])

print("特征选择 - 信息增益计算：")
print("="*60)

feature_igs = {}
for feature_name in ['天气', '温度', '湿度']:
    # 将类别映射为数字
    feature_values = np.array([hash(v) % 10 for v in data[feature_name]])
    ig = information_gain(y, feature_values)
    feature_igs[feature_name] = ig
    print(f"{feature_name:6s}: 信息增益 = {ig:.4f} bits")

# 选择最佳特征
best_feature = max(feature_igs, key=feature_igs.get)
print(f"\n最佳特征: {best_feature} (信息增益最大)")
```

### 5.2 与其他方法的对比

```python
print("\n特征选择方法对比：")
print("="*60)
print("方法              | 优点                | 缺点")
print("-"*60)
print("互信息            | 非线性关系          | 需要足够样本")
print("相关系数          | 简单快速            | 只能检测线性关系")
print("卡方检验          | 理论基础扎实        | 需要类别变量")
print("递归特征消除      | 考虑特征组合        | 计算量大")
```

---

## 6. 应用2：信息瓶颈理论

### 6.1 核心思想

神经网络学习的是**表示**（representation）。

**信息瓶颈理论**：
- 好的表示应该**保留关于输出的信息**（最大化I(T;Y)）
- 同时**压缩关于输入的信息**（最小化I(T;X)）

**目标函数**：
```
L = I(T;X) - β·I(T;Y)

最小化 L
- 最小化 I(T;X)：压缩输入
- 最大化 I(T;Y)：保留预测能力
```

### 6.2 可视化

```python
# 信息平面的可视化

plt.figure(figsize=(12, 5))

# 信息平面
plt.subplot(121)
# 理想路径：从随机初始化到最优表示
I_TX_path = [0.5, 0.7, 0.9, 1.0, 0.95, 0.85, 0.7]  # 表示与输入的MI
I_TY_path = [0.1, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95]  # 表示与输出的MI

plt.plot(I_TX_path, I_TY_path, 'bo-', linewidth=2, markersize=8)
plt.plot(I_TX_path[0], I_TY_path[0], 'g*', markersize=15, label='初始化')
plt.plot(I_TX_path[-1], I_TY_path[-1], 'r*', markersize=15, label='训练后')

plt.xlabel('I(T;X) - 表示与输入的互信息')
plt.ylabel('I(T;Y) - 表示与输出的互信息')
plt.title('信息瓶颈理论\n信息平面上的训练路径')
plt.legend()
plt.grid(True, alpha=0.3)

# 添加注释
plt.text(1.0, 0.5, '过拟合风险\n(记住太多输入细节)', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
plt.text(0.3, 0.9, '理想表示\n(刚好够预测)', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))

# 右图：压缩 vs 保留
plt.subplot(122)
layers = ['Input', 'Layer1', 'Layer2', 'Layer3', 'Output']
I_TX_layer = [2.0, 1.8, 1.4, 1.0, 0.8]
I_TY_layer = [0.1, 0.5, 0.8, 0.95, 1.0]

x_pos = np.arange(len(layers))
width = 0.35

plt.bar(x_pos - width/2, I_TX_layer, width, label='I(T;X) - 压缩', color='blue', alpha=0.7)
plt.bar(x_pos + width/2, I_TY_layer, width, label='I(T;Y) - 预测', color='red', alpha=0.7)

plt.xlabel('网络层')
plt.ylabel('互信息 (bits)')
plt.title('网络各层的信息流')
plt.xticks(x_pos, layers, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('information_bottleneck.png', dpi=100, bbox_inches='tight')
```

### 6.3 理论意义

```
信息瓶颈理论解释现象：

1. 深度学习的成功
   └─ 深层网络能更好地压缩和表示

2. 过拟合
   └─ I(T;X)太大，记住噪声

3. 泛化能力
   └─ 最小充分统计量：最小I(T;X)，最大I(T;Y)

4. 层次表示
   └─ 逐层提取相关信息
```

---

## 7. 应用3：对比学习

### 7.1 核心思想

**对比学习**：学习表示，使得同一样本的不同视角相似，不同样本的视角不同。

**目标**：最大化同一数据增强版本的互信息

```
max I(Z₁; Z₂)
```

其中Z₁和Z₂是同一输入x的两个不同增强版本。

### 7.2 实现

```python
# 对比学习的互信息视角

def contrastive_loss(z1, z2, temperature=0.1):
    """
    对比损失的简化版本

    参数：
        z1, z2: 两个视角的表示 (batch_size, feature_dim)
        temperature: 温度参数

    返回：
        对比损失（互信息的下界）
    """
    # 归一化
    z1 = z1 / np.linalg.norm(z1, axis=1, keepdims=True)
    z2 = z2 / np.linalg.norm(z2, axis=1, keepdims=True)

    # 相似度矩阵
    sim_matrix = np.dot(z1, z2.T) / temperature

    # InfoNCE损失（互信息的下界）
    # 实际实现需要用softmax
    # 这里简化展示概念

    pass

print("对比学习的信息论解释：")
print("="*60)
print("目标：max I(Z₁; Z₂)")
print("     其中Z₁和Z₂是同一输入的两个增强版本")
print()
print("InfoNCE损失：")
print("L = -log[exp(sim(z₁, z₂)/τ) / Σ_j exp(sim(z₁, zⱼ)/τ)]")
print()
print("理论保证：")
print("- 最小化InfoNCE → 最大化互信息的下界")
print("- 互信息越大，两个视角越一致")
```

---

## 8. 连续变量的互信息

### 8.1 定义

对于连续随机变量：

```
I(X;Y) = ∫∫ p(x,y) log[p(x,y)/(p(x)p(y))] dx dy
```

### 8.2 估计方法

```python
# 连续变量互信息的估计

def mutual_information_continuous(x, y, bins=20):
    """
    连续变量互信息的直方图估计

    参数：
        x, y: 连续变量观测值
        bins: 直方图bin数

    返回：
        互信息估计值
    """
    # 计算联合直方图
    joint_hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    # 归一化为概率
    joint_prob = joint_hist / np.sum(joint_hist)

    # 计算互信息
    return mutual_information(joint_prob)

# 示例：线性相关的高斯变量
np.random.seed(42)
n_samples = 1000

x = np.random.randn(n_samples)
y = x + 0.5 * np.random.randn(n_samples)  # y与x相关

mi_estimate = mutual_information_continuous(x, y, bins=30)

print("连续变量互信息估计：")
print("="*60)
print(f"样本数: {n_samples}")
print(f"X ~ N(0,1), Y = X + N(0, 0.25)")
print(f"估计的互信息: {mi_estimate:.4f} bits")

# 理论值（对于高斯变量）
# I(X;Y) = -0.5 * log(1 - ρ²)
rho = np.corrcoef(x, y)[0, 1]
mi_theory = -0.5 * np.log2(1 - rho**2)
print(f"理论值（高斯假设）: {mi_theory:.4f} bits")
```

### 8.3 更精确的估计方法

```python
print("\n连续变量互信息估计方法：")
print("="*60)

methods = {
    '直方图法': '简单快速，但依赖bin选择',
    '核密度估计': '更平滑，但计算量大',
    'KSG估计器': '基于k近邻，无参数',
    'MINE': '神经网络估计，可微分',
}

for method, desc in methods.items():
    print(f"{method:15s}: {desc}")
```

---

## 9. 互信息与其他度量的关系

### 9.1 与相关系数的关系

```python
# 对于二元高斯分布

rhos = np.linspace(0, 0.99, 20)
mis = [-0.5 * np.log2(1 - rho**2) for rho in rhos]

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(rhos, mis, 'b-', linewidth=2)
plt.xlabel('相关系数 ρ')
plt.ylabel('互信息 I(X;Y) (bits)')
plt.title('高斯变量：互信息 vs 相关系数')
plt.grid(True, alpha=0.3)

# 标注几个点
for rho, mi in [(0.5, -0.5*np.log2(1-0.5**2)),
                 (0.8, -0.5*np.log2(1-0.8**2)),
                 (0.95, -0.5*np.log2(1-0.95**2))]:
    plt.plot(rho, mi, 'ro', markersize=8)
    plt.text(rho, mi + 0.1, f'ρ={rho}\nI={mi:.2f}', ha='center', fontsize=9)

# 与其他依赖度量对比
plt.subplot(122)

phi_coefficient = rhos  # 对于二元变量，φ = ρ
print(f"注意：对于二元高斯变量，φ系数 = ρ")

plt.plot(rhos, mis, 'b-', linewidth=2, label='互信息')
plt.plot(rhos, rhos, 'r--', linewidth=2, label='相关系数')
plt.plot(rhos, rhos**2, 'g:', linewidth=2, label='R²')

plt.xlabel('相关系数 |ρ|')
plt.ylabel('度量值')
plt.title('依赖度量对比')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mi_vs_correlation.png', dpi=100, bbox_inches='tight')
```

### 9.2 总结对比

| 度量 | 范围 | 能检测 | 局限 |
|------|------|--------|------|
| **相关系数** | [-1, 1] | 线性关系 | 非线性关系检测不到 |
| **互信息** | [0, ∞) | 任何依赖关系 | 估计困难，无上界 |
| **τ系数** | [-1, 1] | 单调关系 | 非单调关系 |
| **距离相关** | [0, 1] | 任何依赖 | 计算复杂 |

---

## 10. 总结

### 核心概念

```
互信息 I(X;Y)：
├─ 定义：H(X) - H(X|Y) = H(Y) - H(Y|X)
├─ 含义：X和Y共享的信息量
├─ 性质：非负、对称、I(X;Y)=0 ⟺ 独立
└─ 上界：I(X;Y) ≤ min(H(X), H(Y))
```

### 三种视角

```
1. KL散度视角
   I(X;Y) = KL(P(X,Y) || P(X)P(Y))
   → 联合分布偏离独立假设的程度

2. 熵减视角
   I(X;Y) = H(X) - H(X|Y)
   → 知道Y后，X的不确定性降低了多少

3. 共享信息视角
   I(X;Y) = H(X) + H(Y) - H(X,Y)
   → X和Y共有的信息量
```

### 应用总结

| 应用 | 方法 | 目标 |
|------|------|------|
| **特征选择** | max I(X;Y) | 选最相关特征 |
| **信息瓶颈** | min I(T;X) - β·I(T;Y) | 学到好的表示 |
| **对比学习** | max I(Z₁;Z₂) | 一致性表示 |
| **因果推断** | 比较MI值 | 区分因果方向 |

### 与其他概念的关系

```
熵 H(X)：单个随机变量的不确定性
├─ 联合熵 H(X,Y)：两个变量总不确定性
├─ 条件熵 H(X|Y)：已知Y，X的不确定性
├─ 交叉熵 H(P,Q)：编码损失
├─ KL散度 KL(P||Q)：分布差异
└─ 互信息 I(X;Y)：两个变量的关联

关系：I(X;Y) = H(X) + H(Y) - H(X,Y)
      I(X;Y) = H(X) - H(X|Y)
      H(P,Q) = H(P) + KL(P||Q)
```

### 深度学习中的位置

```
输入 X
  ↓
特征提取 (最大化 I(T;Y))
  ↓
表示 T (信息瓶颈：最小化 I(T;X))
  ↓
预测 Ŷ
  ↓
损失计算 (交叉熵 = 最小化 KL)
```

---

## 下一步

信息论模块已完成。继续学习 [06_rl_math](../../06_rl_math/) 学习强化学习的数学基础，或进入 [07_applications](../../07_applications/) 进行综合实践。
