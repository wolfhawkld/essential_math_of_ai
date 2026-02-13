# 拉格朗日乘数法与约束优化

## 为什么需要约束优化？

深度学习和机器学习中，许多问题带有约束条件：

### 实际问题

```
问题1：SVM（支持向量机）
├── 最大化间隔
└── 约束：正确分类所有样本

问题2：优化问题
├── 最小化损失函数
└── 约束：||θ|| ≤ ε（限制模型复杂度）

问题3：概率分布
├── 最大化熵
└── 约束：期望匹配观测值

问题4：资源分配
├── 最大化收益
└── 约束：总预算固定
```

---

## 1. 约束优化问题

### 1.1 问题形式

**一般形式**：
```
minimize    f(x)
subject to  g_i(x) ≤ 0,  i = 1, ..., m   (不等式约束)
            h_j(x) = 0,  j = 1, ..., p   (等式约束)
```

**几何意义**：在可行域内找最优解

```python
import numpy as np
import matplotlib.pyplot as plt

# 可视化可行域
def f(x, y):
    """目标函数：最小化 x² + y²"""
    return x**2 + y**2

def g1(x, y):
    """约束1：x + y ≥ 1  （即 -(x+y) + 1 ≤ 0）"""
    return -(x + y) + 1

def g2(x, y):
    """约束2：x ≤ 2"""
    return x - 2

# 创建网格
x = np.linspace(-1, 3, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(10, 8))

# 目标函数等高线
contour = plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
plt.clabel(contour, inline=True, fontsize=8)

# 约束区域
# g1 ≤ 0: x + y ≥ 1（即 y ≥ 1 - x）
plt.fill_between(x, 1 - x, 3, alpha=0.3, color='red', label='g₁: x+y≥1')
# g2 ≤ 0: x ≤ 2
plt.axvline(x=2, color='blue', linestyle='--', linewidth=2, label='g₂: x≤2')
plt.fill_betweenx(y, -1, 2, alpha=0.2, color='blue')

# 可行域（同时满足两个约束）
feasible_mask = (X + Y >= 1) & (X <= 2)
plt.contourf(X, Y, feasible_mask, levels=[0.5, 1.5], alpha=0.4, colors=['green'])

# 最优解（无约束时）
plt.plot(0, 0, 'ko', markersize=10, label='无约束最优解(0,0)')

# 最优解（有约束时）- 在边界上
plt.plot(0.5, 0.5, 'r*', markersize=15, label='约束最优解(0.5,0.5)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('约束优化：最小化 x²+y²\ngreen区域=可行域')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.tight_layout()
plt.savefig('constrained_optimization.png', dpi=100, bbox_inches='tight')
```

---

## 2. 等式约束优化

### 2.1 拉格朗日乘数法

**问题**：
```
minimize    f(x)
subject to  h(x) = 0
```

**核心思想**：构造拉格朗日函数

```
L(x, λ) = f(x) + λ · h(x)
```

其中 λ 是拉格朗日乘数

**最优性条件**：
```
∇_x L = 0  →  ∇f(x) + λ∇h(x) = 0
∇_λ L = 0  →  h(x) = 0
```

### 2.2 几何解释

```python
import numpy as np
import matplotlib.pyplot as plt

# 问题：minimize f(x,y) = x² + y²
# 约束：h(x,y) = x + y - 1 = 0 （直线）

def f(x, y):
    return x**2 + y**2

def h(x, y):
    return x + y - 1

# 创建网格
x = np.linspace(-0.5, 1.5, 100)
y = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(12, 5))

# 左图：等高线+约束
plt.subplot(121)
contour = plt.contour(X, Y, Z, levels=15, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)

# 约束线
y_constraint = 1 - x
plt.plot(x, y_constraint, 'r-', linewidth=3, label='约束: x+y=1')

# 最优点
x_opt = 0.5
y_opt = 0.5
plt.plot(x_opt, y_opt, 'r*', markersize=15, label='最优解')

# 梯度
grad_f = np.array([2*x_opt, 2*y_opt])  # ∇f
grad_h = np.array([1, 1])               # ∇h = (1, 1)

# 绘制梯度向量
plt.arrow(x_opt, y_opt, 0.15*grad_f[0], 0.15*grad_f[1],
          head_width=0.05, head_length=0.03, fc='blue', ec='blue', linewidth=2)
plt.arrow(x_opt, y_opt, 0.3*grad_h[0], 0.3*grad_h[1],
          head_width=0.05, head_length=0.03, fc='green', ec='green', linewidth=2)

plt.text(x_opt + 0.05, y_opt + 0.15, '∇f', fontsize=12, color='blue', fontweight='bold')
plt.text(x_opt + 0.2, y_opt + 0.25, '∇h', fontsize=12, color='green', fontweight='bold')

plt.xlabel('x')
plt.ylabel('y')
plt.title('等式约束优化\n∇f = -λ∇h（梯度平行）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 右图：拉格朗日函数
plt.subplot(122)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.gcf()
ax = fig.add_subplot(122, projection='3d')

# 目标函数曲面
x_3d = np.linspace(-1, 2, 50)
y_3d = np.linspace(-1, 2, 50)
X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
Z_3d = f(X_3d, Y_3d)

ax.plot_surface(X_3d, Y_3d, Z_3d, cmap='viridis', alpha=0.6)

# 约束线在曲面上的投影
y_const = 1 - x_3d
z_const = f(x_3d, y_const)
ax.plot(x_3d, y_const, z_const, 'r-', linewidth=4, label='约束曲线')

# 最优点
ax.scatter([0.5], [0.5], [f(0.5, 0.5)], color='red', s=100, marker='*')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('拉格朗日视角\n约束曲线上的最小值')

plt.tight_layout()
plt.savefig('lagrange_equality.png', dpi=100, bbox_inches='tight')
```

### 2.3 求解步骤

**例子**：
```
minimize    f(x, y) = x² + y²
subject to  x + y = 1
```

**步骤**：

1. **构造拉格朗日函数**：
   ```
   L(x, y, λ) = x² + y² + λ(x + y - 1)
   ```

2. **求偏导并令其为0**：
   ```
   ∂L/∂x = 2x + λ = 0    → x = -λ/2
   ∂L/∂y = 2y + λ = 0    → y = -λ/2
   ∂L/∂λ = x + y - 1 = 0  → x + y = 1
   ```

3. **求解方程组**：
   ```
   x = y（对称性）
   2x = 1
   x = y = 0.5

   代入第一式：λ = -1
   ```

4. **验证**：
   ```
   f(0.5, 0.5) = 0.5² + 0.5² = 0.5
   ```

**代码实现**：
```python
from scipy.optimize import minimize

# 使用scipy求解
def objective(x):
    return x[0]**2 + x[1]**2

def constraint_eq(x):
    return x[0] + x[1] - 1

# 定义约束
cons = {'type': 'eq', 'fun': constraint_eq}

# 初始猜测
x0 = [0, 0]

# 求解
solution = minimize(objective, x0, method='SLSQP', constraints=cons)

print(f"最优解: x = {solution.x}")
print(f"最优值: f = {solution.fun}")
print(f"拉格朗日乘数: λ ≈ {-2*solution.x[0]}")  # 从2x+λ=0推出
```

### 2.4 多个等式约束

```
minimize    f(x)
subject to  h₁(x) = 0
            h₂(x) = 0
            ...
            h_m(x) = 0
```

拉格朗日函数：
```
L(x, λ₁, ..., λ_m) = f(x) + λ₁h₁(x) + ... + λ_m h_m(x)
```

**例子**：
```python
# 问题： minimize x² + y² + z²
# 约束： x + y + z = 1
#       x - y = 0

def lagrange_multi_equality():
    """多个等式约束例子"""
    # 隐式求解
    # h1: x + y + z - 1 = 0
    # h2: x - y = 0

    # 从h2得：x = y
    # 代入h1: 2x + z = 1 → z = 1 - 2x

    # 目标函数变为：
    # f(x) = x² + x² + (1-2x)²
    #      = 2x² + 1 - 4x + 4x²
    #      = 6x² - 4x + 1

    # 求导：
    # df/dx = 12x - 4 = 0
    # x = 1/3

    x = 1/3
    y = 1/3
    z = 1 - 2/3

    print(f"最优解: x = y = {x:.3f}, z = {z:.3f}")
    print(f"最优值: f = {x**2 + y**2 + z**2:.3f}")

lagrange_multi_equality()
```

---

## 3. 不等式约束优化

### 3.1 KKT条件

对于一般约束优化问题：
```
minimize    f(x)
subject to  g_i(x) ≤ 0,  i = 1, ..., m
            h_j(x) = 0,  j = 1, ..., p
```

**KKT条件**（Karush-Kuhn-Tucker）：

```
1. 站定条件：
   ∇f(x*) + Σᵢ λᵢ ∇gᵢ(x*) + Σⱼ μⱼ ∇hⱼ(x*) = 0

2. 原始可行性：
   gᵢ(x*) ≤ 0,  ∀i
   hⱼ(x*) = 0,  ∀j

3. 对偶可行性：
   λᵢ ≥ 0,  ∀i

4. 互补松弛条件：
   λᵢ gᵢ(x*) = 0,  ∀i
   (约束不活跃时 λᵢ=0，活跃时 gᵢ=0)
```

### 3.2 理解互补松弛条件

```python
import numpy as np
import matplotlib.pyplot as plt

# 可视化互补松弛条件
def visualize_complementary_slackness():
    """
    例子：minimize f(x) = x²
         约束：g(x) = x - 1 ≤ 0
    """

    x = np.linspace(-1, 3, 100)
    f = x**2
    g = x - 1

    plt.figure(figsize=(14, 5))

    # 情况1：约束不活跃（x*在约束区域内）
    plt.subplot(131)
    plt.plot(x, f, 'b-', linewidth=2, label='f(x) = x²')
    plt.fill_between(x, 0, f, where=(g <= 0), alpha=0.3, color='green', label='可行域')
    plt.axvline(x=1, color='r', linestyle='--', label='约束边界 x=1')
    plt.plot(0, 0, 'ko', markersize=10, label='最优解 x*=0')

    # 标注
    plt.text(0.2, 2, '约束不活跃\nx*=0 在可行域内\nλ=0', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('情况1：约束不活跃\nλ·g = 0 (λ=0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, 3)

    # 情况2：约束活跃（最优解在边界）
    plt.subplot(132)
    # 新问题：minimize f(x) = (x-2)², 约束：x ≤ 1
    f2 = (x - 2)**2
    plt.plot(x, f2, 'b-', linewidth=2, label='f(x) = (x-2)²')
    plt.fill_between(x, 0, f2, where=(g <= 0), alpha=0.3, color='green', label='可行域')
    plt.axvline(x=1, color='r', linestyle='--', label='约束边界 x=1')
    plt.plot(1, 1, 'ko', markersize=10, label='最优解 x*=1')

    plt.text(1.2, 2, '约束活跃\nx*=1 在边界\nλ>0', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('情况2：约束活跃\nλ·g = 0 (g=0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, 3)

    # 情况3：多个约束
    plt.subplot(133)
    g1 = x - 1  # x ≤ 1
    g2 = -x - 0.5  # x ≥ -0.5

    feasible = (g1 <= 0) & (g2 <= 0)
    plt.plot(x, f, 'b-', linewidth=2, label='f(x) = x²')
    plt.fill_between(x, 0, f, where=feasible, alpha=0.3, color='green', label='可行域')
    plt.axvline(x=1, color='r', linestyle='--', label='约束1: x≤1')
    plt.axvline(x=-0.5, color='orange', linestyle='--', label='约束2: x≥-0.5')
    plt.plot(0, 0, 'ko', markersize=10, label='最优解 x*=0')

    plt.text(0.2, 3, '最优解在可行域内\n两个约束都不活跃\nλ₁=λ₂=0', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('情况3：多个约束')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, 3)

    plt.tight_layout()
    plt.savefig('complementary_slackness.png', dpi=100, bbox_inches='tight')

visualize_complementary_slackness()
```

### 3.3 求解示例

**问题**：
```
minimize    f(x, y) = x² + y²
subject to  x + y ≥ 1  （即 g(x,y) = 1 - x - y ≤ 0）
```

**手动求解**：

```python
def solve_kkt_example():
    """
    手动求解KKT条件

    L(x, y, λ) = x² + y² + λ(1 - x - y)

    KKT条件：
    1. ∇L = 0:
       ∂L/∂x = 2x - λ = 0  → x = λ/2
       ∂L/∂y = 2y - λ = 0  → y = λ/2

    2. g ≤ 0:  1 - x - y ≤ 0

    3. λ ≥ 0

    4. λ·g = 0: λ(1 - x - y) = 0
    """

    # 情况1：约束不活跃（λ = 0）
    print("情况1：λ = 0（约束不活跃）")
    lambda1 = 0
    x1 = lambda1 / 2
    y1 = lambda1 / 2
    g1 = 1 - x1 - y1
    print(f"  x = y = {x1}")
    print(f"  g(x,y) = {g1} ≤ 0? {g1 <= 0}")
    if g1 <= 0:
        print(f"  ✓ 可行！f = {x1**2 + y1**2}")
    else:
        print(f"  ✗ 不可行，尝试情况2")

    print()

    # 情况2：约束活跃（g = 0）
    print("情况2：g = 0（约束活跃）")
    # x + y = 1
    # x = y（对称）→ x = y = 0.5
    x2 = 0.5
    y2 = 0.5
    # 从 2x - λ = 0 得
    lambda2 = 2 * x2
    print(f"  x = y = {x2}")
    print(f"  λ = {lambda2}")
    print(f"  λ ≥ 0? {lambda2 >= 0}")
    print(f"  ✓ 可行！f = {x2**2 + y2**2}")

    # 对比
    print("\n结论：情况2是最优解")

solve_kkt_example()
```

**使用scipy求解**：
```python
from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + x[1]**2

def constraint_ineq(x):
    return x[0] + x[1] - 1  # scipy中不等式约束为 >= 0

cons = {'type': 'ineq', 'fun': constraint_ineq}
x0 = [0, 0]

solution = minimize(objective, x0, method='SLSQP', constraints=cons)

print(f"最优解: x = {solution.x}")
print(f"最优值: f = {solution.fun}")
```

---

## 4. 对偶问题

### 4.1 拉格朗日对偶

**原始问题（Primal）**：
```
minimize    f(x)
subject to  g_i(x) ≤ 0
```

**拉格朗日函数**：
```
L(x, λ) = f(x) + Σ λᵢ gᵢ(x)
```

**对偶函数**：
```
d(λ) = min_x L(x, λ)
```

**对偶问题（Dual）**：
```
maximize    d(λ)
subject to  λ ≥ 0
```

### 4.2 弱对偶性

```
对偶问题的最优值 ≤ 原始问题的最优值
```

**对偶间隙**：
```
gap = 原始最优 - 对偶最优 ≥ 0
```

### 4.3 强对偶性（Slater条件）

如果问题是凸的，且存在严格可行点（strictly feasible），则：

```
对偶最优值 = 原始最优值
```

**Slater条件**：
```
存在 x 使得：
  - 所有不等式约束严格满足：g_i(x) < 0
  - 所有等式约束满足：h_j(x) = 0
```

### 4.4 为什么关心对偶？

```python
def example_dual_svm():
    """
    SVM的对偶形式示例

    原始问题：
    minimize    (1/2)||w||²
    subject to  y_i(w·x_i + b) ≥ 1

    对偶问题：
    maximize    Σα_i - (1/2)ΣΣ α_i α_j y_i y_j (x_i·x_j)
    subject to  α_i ≥ 0

    优势：
    1. 对偶问题变量数 = 样本数（与特征维度无关）
    2. 引入核函数（只关心内积 x_i·x_j）
    3. 稀疏解（大多数α_i = 0）
    """
    print("SVM对偶形式的优势：")
    print("1. 高维空间时对偶更易求解")
    print("2. 可以引入核技巧（kernel trick）")
    print("3. 解是稀疏的（支持向量）")

example_dual_svm()
```

---

## 5. 深度学习中的应用

### 5.1 权重衰减（L2正则化）

**问题**：
```
minimize    L(θ)
subject to  ||θ||² ≤ r
```

**拉格朗日形式**：
```
L_aug(θ, λ) = L(θ) + λ(||θ||² - r)
```

**等效于在损失函数中加正则项**：
```
L_total = L(θ) + β||θ||²
```

其中 β = λ（拉格朗日乘数）

```python
import numpy as np
import matplotlib.pyplot as plt

def weight_decay_example():
    """权重衰减示例"""
    np.random.seed(42)

    # 生成数据
    n_samples = 20
    X = np.random.randn(n_samples, 1)
    y_true = 2 * X.squeeze() + 1
    y = y_true + 0.5 * np.random.randn(n_samples)

    # 无正则化
    X_bias = np.c_[np.ones(n_samples), X]
    theta_no_reg = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

    # 有正则化（岭回归）
    def ridge_regression(X, y, alpha):
        n_features = X.shape[1]
        return np.linalg.inv(X.T @ X + alpha * np.eye(n_features)) @ X.T @ y

    alphas = [0, 0.1, 1.0, 10.0]

    # 可视化
    plt.figure(figsize=(15, 5))

    for i, alpha in enumerate(alphas):
        plt.subplot(1, 4, i+1)

        theta = ridge_regression(X_bias, y, alpha)

        # 数据点
        plt.scatter(X, y, alpha=0.6)

        # 拟合线
        X_test = np.linspace(-3, 3, 100)
        y_pred = theta[0] + theta[1] * X_test
        plt.plot(X_test, y_pred, 'r-', linewidth=2)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'λ = {alpha}\nθ = [{theta[0]:.2f}, {theta[1]:.2f}]')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('weight_decay.png', dpi=100, bbox_inches='tight')

weight_decay_example()
```

### 5.2 SVM对偶

**原始问题**：
```
minimize    (1/2)||w||²
subject to  y_i(w·x_i + b) ≥ 1,  ∀i
```

**对偶问题**：
```
maximize    Σα_i - (1/2)ΣΣ α_i α_j y_i y_j K(x_i, x_j)
subject to  α_i ≥ 0, Σα_i y_i = 0

其中 K(x_i, x_j) 是核函数
```

```python
def svm_dual_simple():
    """简单SVM对偶示例"""

    # 2D数据
    X = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [1, 0], [0, 1]])
    y = np.array([1, 1, 1, -1, -1, -1])

    # 对偶问题（简化版，线性核）
    # maximize Σα_i - (1/2)ΣΣ α_i α_j y_i y_j (x_i·x_j)
    # subject to α_i ≥ 0, Σα_i y_i = 0

    from scipy.optimize import minimize

    def dual_objective(alpha):
        """对偶目标函数（取负号因为scipy最小化）"""
        n = len(alpha)
        obj = -np.sum(alpha)  # -Σα_i

        for i in range(n):
            for j in range(n):
                obj += 0.5 * alpha[i] * alpha[j] * y[i] * y[j] * np.dot(X[i], X[j])

        return obj

    def constraint_eq(alpha):
        """等式约束：Σα_i y_i = 0"""
        return np.dot(alpha, y)

    # 初始值
    alpha0 = np.ones(len(y)) * 0.1

    # 约束
    cons = [
        {'type': 'eq', 'fun': constraint_eq},
    ]

    # 边界：α_i ≥ 0
    bounds = [(0, None) for _ in range(len(y))]

    # 求解
    result = minimize(dual_objective, alpha0, method='SLSQP',
                     bounds=bounds, constraints=cons)

    alpha = result.x

    print("拉格朗日乘数（支持向量）：")
    for i, a in enumerate(alpha):
        if a > 1e-5:
            print(f"  α_{i} = {a:.4f}, x_{i} = {X[i]}, y_{i} = {y[i]}")

    # 恢复w和b
    w = np.zeros(2)
    for i in range(len(y)):
        w += alpha[i] * y[i] * X[i]

    # b = mean(y_i - w·x_i) for support vectors
    sv_indices = np.where(alpha > 1e-5)[0]
    b = np.mean([y[i] - np.dot(w, X[i]) for i in sv_indices])

    print(f"\n权重向量 w = {w}")
    print(f"偏置 b = {b:.4f}")

    # 可视化
    plt.figure(figsize=(8, 6))

    # 数据点
    for i in range(len(y)):
        if alpha[i] > 1e-5:
            plt.scatter(X[i, 0], X[i, 1], s=200, c='red' if y[i] > 0 else 'blue',
                       marker='o', alpha=0.5, edgecolors='black', linewidths=2,
                       label='支持向量' if i == sv_indices[0] else '')
        else:
            plt.scatter(X[i, 0], X[i, 1], s=100, c='red' if y[i] > 0 else 'blue',
                       marker='o', alpha=0.7, label='正类' if y[i] > 0 and i == 0 else ('负类' if y[i] < 0 and i == 3 else ''))

    # 决策边界
    x_plot = np.linspace(-0.5, 2.5, 100)
    y_plot = -(w[0] * x_plot + b) / w[1]
    plt.plot(x_plot, y_plot, 'k-', linewidth=2, label='决策边界')

    # 间隔边界
    y_margin_pos = -(w[0] * x_plot + b - 1) / w[1]
    y_margin_neg = -(w[0] * x_plot + b + 1) / w[1]
    plt.plot(x_plot, y_margin_pos, 'k--', linewidth=1, alpha=0.5)
    plt.plot(x_plot, y_margin_neg, 'k--', linewidth=1, alpha=0.5)

    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('SVM对偶求解\n红色圈=支持向量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('svm_dual.png', dpi=100, bbox_inches='tight')

svm_dual_simple()
```

### 5.3 最大熵原理

**问题**：在满足某些矩约束的条件下，找熵最大的分布

```
maximize    H(p) = -Σ p(x) log p(x)
subject to  Σ p(x) = 1              （归一化）
            Σ p(x) f_k(x) = μ_k      （矩约束）
```

**拉格朗日函数**：
```
L(p, λ, α) = -Σ p(x) log p(x) + λ(Σp(x) - 1) + Σα_k(Σp(x)f_k(x) - μ_k)
```

**求解**：
```
∂L/∂p(x) = -log p(x) - 1 + λ + Σα_k f_k(x) = 0

p(x) = exp(λ - 1 + Σα_k f_k(x))
     = (1/Z) exp(Σα_k f_k(x))
```

这是**指数族分布**的形式！

```python
def maximum_entropy_example():
    """最大熵示例：给定均值和方差，求分布"""

    print("最大熵原理应用：")
    print("\n约束1：已知均值 → 高斯分布")
    print("约束2：已知均值和方差 → 高斯分布")
    print("约束3：已知范围[a,b] → 均匀分布")
    print("约束4：已知期望 → 指数分布")

maximum_entropy_example()
```

---

## 6. 增广拉格朗日方法

### 6.1 动机

标准拉格朗日方法在数值上可能不稳定，增广拉格朗日方法增加惩罚项：

```
L_A(x, λ) = f(x) + λ·h(x) + (ρ/2)·h(x)²
                                      ↑
                                   惩罚项
```

### 6.2 交替方向乘子法（ADMM）

**问题形式**：
```
minimize    f(x) + g(z)
subject to  Ax + Bz = c
```

**ADMM迭代**：
```
x^{k+1} = argmin_x L_A(x, z^k, λ^k)
z^{k+1} = argmin_z L_A(x^{k+1}, z, λ^k)
λ^{k+1} = λ^k + ρ(Ax^{k+1} + Bz^{k+1} - c)
```

```python
def admm_example():
    """
    ADMM示例：L1正则化（LASSO）

    minimize (1/2)||Ax - b||² + λ||x||₁

    等价于：
    minimize f(x) + g(z)
    subject to x - z = 0

    其中 f(x) = (1/2)||Ax - b||², g(z) = λ||z||₁
    """

    np.random.seed(42)

    # 生成数据
    n, d = 100, 20
    A = np.random.randn(n, d)
    x_true = np.zeros(d)
    x_true[:5] = np.random.randn(5)  # 稀疏真实解
    b = A @ x_true + 0.1 * np.random.randn(n)

    # ADMM参数
    lam = 0.1  # L1正则化系数
    rho = 1.0  # 惩罚参数
    max_iter = 100

    # 初始化
    x = np.zeros(d)
    z = np.zeros(d)
    u = np.zeros(d)  # 对偶变量

    # 预计算
    A_tA = A.T @ A
    A_tb = A.T @ b
    L = A_tA + rho * np.eye(d)
    L_inv = np.linalg.inv(L)

    def soft_threshold(x, threshold):
        """软阈值算子（L1的proximal operator）"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    # ADMM迭代
    history = []
    for k in range(max_iter):
        # x更新
        x = L_inv @ (A_tb + rho * (z - u))

        # z更新（软阈值）
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)

        # u更新
        u = u + x - z

        # 记录原始残差
        r_norm = np.linalg.norm(x - z)
        history.append(r_norm)

        if k % 20 == 0:
            print(f"Iter {k}: ||x-z|| = {r_norm:.6f}")

    # 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(history, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('||x - z||')
    plt.title('ADMM收敛')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.subplot(122)
    plt.stem(x_true, linefmt='b-', markerfmt='bo', basefmt=' ', label='真实解')
    plt.stem(z, linefmt='r-', markerfmt='ro', basefmt=' ', label='ADMM解')
    plt.xlabel('系数索引')
    plt.ylabel('值')
    plt.title('稀疏解对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('admm_lasso.png', dpi=100, bbox_inches='tight')

admm_example()
```

---

## 7. 投影梯度下降

### 7.1 带约束的梯度下降

对于约束优化：
```
minimize    f(x)
subject to  x ∈ C
```

**投影梯度下降**：
```
x^{k+1} = Proj_C(x^k - η·∇f(x^k))
```

其中 Proj_C 是到集合C的投影

### 7.2 常见投影

```python
def projection_simplex(v, z=1):
    """
    投影到单纯形 {x | x ≥ 0, Σx_i = z}

    参考: Efficient Projections onto the l1-Ball for Learning in High Dimensions
    """
    n = len(v)
    u = np.sort(v)[::-1]  # 降序排列
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - z))[0][-1]
    theta = (cssv[rho] - z) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

def projection_l2_ball(v, radius=1):
    """投影到L2球 {x | ||x|| ≤ radius}"""
    norm = np.linalg.norm(v)
    if norm <= radius:
        return v
    return radius * v / norm

def projection_l1_ball(v, radius=1):
    """投影到L1球 {x | ||x||₁ ≤ radius}"""
    if np.sum(np.abs(v)) <= radius:
        return v

    # 简化版：使用soft thresholding
    # 需要二分搜索阈值
    u = np.abs(v)
    low, high = 0, np.max(u)

    while high - low > 1e-6:
        mid = (low + high) / 2
        w = np.maximum(u - mid, 0)
        if np.sum(w) > radius:
            low = mid
        else:
            high = mid

    return np.sign(v) * np.maximum(u - mid, 0)

def projected_gradient_descent_example():
    """投影梯度下降示例"""

    # 问题：minimize f(x) = x² + y²
    # 约束：x + y = 1, x,y ≥ 0

    def objective(x):
        return np.sum(x**2)

    def grad(x):
        return 2*x

    def project_simplex_2d(x):
        """投影到2D单纯形 x+y=1, x,y≥0"""
        # 先投影到x+y=1
        avg = (x[0] + x[1]) / 2
        x_proj = x - np.array([avg - 0.5, avg - 0.5])
        # 再投影到x,y≥0
        x_proj = np.maximum(x_proj, 0)
        # 重新归一化
        x_proj = x_proj / np.sum(x_proj)
        return x_proj

    # 初始化
    x = np.array([2.0, 3.0])
    lr = 0.1
    path = [x.copy()]

    for k in range(50):
        # 梯度步
        x = x - lr * grad(x)

        # 投影
        x = project_simplex_2d(x)

        path.append(x.copy())

    path = np.array(path)

    print(f"最优解: x = {x}")
    print(f"最优值: f = {objective(x)}")

    # 可视化
    plt.figure(figsize=(8, 8))

    # 等高线
    x_grid = np.linspace(-1, 4, 100)
    y_grid = np.linspace(-1, 4, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = X**2 + Y**2

    plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)

    # 可行域（线段）
    feasible_x = np.linspace(0, 1, 100)
    feasible_y = 1 - feasible_x
    plt.plot(feasible_x, feasible_y, 'g-', linewidth=3, label='可行域')

    # 优化路径
    plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=6, alpha=0.7)
    plt.plot(path[-1, 0], path[-1, 1], 'b*', markersize=20, label='最优解')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('投影梯度下降\n约束: x+y=1, x,y≥0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('projected_gradient.png', dpi=100, bbox_inches='tight')

projected_gradient_descent_example()
```

---

## 8. 内点法（简介）

### 8.1 思路

将不等式约束用障碍函数代替：

```
minimize    f(x) + μ·Σ log(-g_i(x))
subject to  g_i(x) < 0
```

当 μ → 0 时，逼近原始问题

### 8.2 应用

主要用于线性规划、二次规划等凸优化问题，在深度学习中较少直接使用。

---

## 9. 总结

### 方法对比

| 方法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **拉格朗日乘数法** | 等式约束 | 理论清晰 | 需手动求解 |
| **KKT条件** | 一般约束 | 充要条件（凸问题） | 条件复杂 |
| **对偶方法** | SVM等 | 高维优势、核技巧 | 需要强对偶性 |
| **增广拉格朗日** | 数值稳定 | 鲁棒性好 | 需调参 |
| **ADMM** | 分解问题 | 并行化 | 收敛可能慢 |
| **投影梯度** | 简单约束 | 实现简单 | 需要投影公式 |
| **内点法** | 凸优化 | 高效 | 复杂 |

### 深度学习中的相关概念

```
约束优化 ↔ 深度学习应用

等式约束 → Batch Normalization（归一化约束）
不等式约束 → 权重衰减（||θ|| ≤ r）
对偶 → SVM
投影 → 梯度裁剪（投影到L2球）
KKT → 正则化项设计
```

### 核心要点

```
✓ 拉格朗日乘数法：等式约束的最优性条件
✓ KKT条件：一般约束的充要条件（凸问题）
✓ 互补松弛：约束要么活跃，要么乘数为零
✓ 对偶性：原始问题和对偶问题的关系
✓ 投影梯度：简单有效处理约束的方法
✓ 应用：SVM、正则化、最大熵
```

---

## 下一步

优化理论模块已完成，继续学习 [05_information_theory](../../05_information_theory/)，理解信息论在深度学习中的作用（损失函数设计、自编码器等）。
