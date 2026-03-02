# 向量和矩阵基础

## 1. 向量 (Vector)

### 1.1 什么是向量？

在深度学习中，向量是最基本的数据结构：

**几何视角**：向量是空间中的一个点或箭头
- 2D向量：`[3, 2]` 表示从原点到点(3,2)的箭头

**代数视角**：向量是一列有序的数字
```python
import numpy as np

# 3维向量
v = np.array([1.0, 2.0, 3.0])
print(v.shape)  # (3,)
```

**深度学习视角**：
- 一张灰度图像：向量 `[batch, height×width]`
- 词嵌入：向量 `[300]` 表示一个单词
- 神经网络输出：向量 `[num_classes]` 表示每类的概率

### 1.2 向量运算

#### (1) 向量加法
```
[1]   [4]   [5]
[2] + [5] = [7]
[3]   [6]   [9]
```

**几何意义**：向量首尾相连

**代码实现**：
```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v3 = v1 + v2  # [5, 7, 9]
```

**深度学习应用**：偏置项加法
```python
output = weights @ input + bias  # bias就是向量加法
```

#### (2) 数乘 (Scalar Multiplication)
```
    [1]   [3]
3 × [2] = [6]
    [3]   [9]
```

**几何意义**：缩放向量长度

**深度学习应用**：学习率调整
```python
gradients = compute_gradients()
weights -= learning_rate * gradients  # 数乘
```

#### (3) 点积 (Dot Product)
```
[1]     [4]
[2] · [5] = 1×4 + 2×5 + 3×6 = 32
[3]     [6]
```

**公式**：
```
v · w = Σ vᵢwᵢ = v₁w₁ + v₂w₂ + ... + vₙwₙ
```

**几何意义**：
- 点积 > 0：两向量夹角 < 90°（同向）
- 点积 = 0：两向量垂直
- 点积 < 0：两向量夹角 > 90°（反向）

**深度学习应用**：神经元计算
```python
# 神经元就是输入向量和权重向量的点积
neuron_output = np.dot(inputs, weights) + bias
```
<span style="color:red;">注意：对单个神经元来说是算点积，而对一层神经元来说就是算矩阵乘法了</span>

#### (4) 向量范数 (Norm)

**L2范数**（欧几里得距离）：
```
‖v‖₂ = √(v₁² + v₂² + ... + vₙ²)
```

```python
v = np.array([3, 4])
norm = np.linalg.norm(v)  # 5.0
```

**L1范数**（曼哈顿距离）：
```
‖v‖₁ = |v₁| + |v₂| + ... + |vₙ|
```

**深度学习应用**：
- **L2正则化**：`loss += λ × ‖weights‖₂²`
- **梯度裁剪**：`if ‖gradients‖₂ > threshold: ...`
- L1范数的几何形状是菱形八面体（有尖角）；L2范数则是球体（平滑）
- 需要稀疏，快速收敛，抗干扰强，特征选择时用L1范数；需要平滑，可导，对异常敏感，仅改变所有特征权重时用L2范数
- 选择 L1范数​ 当：
    特征数量远大于样本数量（基因数据、文本数据）
    需要模型可解释性（想知道哪些特征真正重要）
    存在大量无关特征（自动特征选择）
    对异常值敏感的场景（金融风控、传感器数据）
- 选择 L2范数​ 当：
    所有特征都可能相关（物理系统建模）
    特征间存在多重共线性（岭回归处理共线性）
    需要稳定、平滑的解决方案（控制系统）
    数据相对干净，异常值少

---

## 2. 矩阵 (Matrix)

### 2.1 什么是矩阵？

**矩阵是向量的集合**：

```
     ┌         ┐
A =  │ 1  2  3 │  ← 第1行（行向量）
     │ 4  5  6 │  ← 第2行
     └         ┘
       ↑  ↑  ↑
      列1 列2 列3
```

形状：`(2, 3)` 表示2行3列

**深度学习中的矩阵**：
```python
# 全连接层的权重
weights = torch.randn(512, 256)  # [输入维度, 输出维度]

# 一批图像
images = torch.randn(32, 3, 224, 224)  # [batch, channels, H, W]
images_flat = images.view(32, -1)  # [32, 150528] 矩阵
```

### 2.2 矩阵基本操作

#### (1) 矩阵转置 (Transpose)

行变列，列变行：

```
     ┌     ┐          ┌     ┐
A =  │ 1 2 │   →  Aᵀ= │ 1 3 │
     │ 3 4 │          │ 2 4 │
     └     ┘          └     ┘
   (2×2)            (2×2)
```

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
A_T = A.T  # 或 np.transpose(A)
```

**性质**：
- `(Aᵀ)ᵀ = A`
- `(A + B)ᵀ = Aᵀ + Bᵀ`
- `(AB)ᵀ = BᵀAᵀ`（注意顺序反转）

**深度学习应用**：反向传播
```python
# 前向传播
output = input @ W  # [batch, in] @ [in, out]

# 反向传播：需要W的转置
grad_input = grad_output @ W.T  # [batch, out] @ [out, in]
```

#### (2) 矩阵乘法 (Matrix Multiplication)

**规则**：
- `A` 的列数必须等于 `B` 的行数
- `(m×n)` @ `(n×p)` = `(m×p)`

**计算方式**：
```
┌     ┐   ┌     ┐   ┌           ┐
│ 1 2 │ @ │ 5 6 │ = │ 1×5+2×7  1×6+2×8 │ = │ 19 22 │
│ 3 4 │   │ 7 8 │   │ 3×5+4×7  3×6+4×8 │   │ 43 50 │
└     ┘   └     ┘   └           ┘
(2×2)     (2×2)         (2×2)
```

**深度学习应用**：前向传播
```python
# 全连接层
def forward(x, W, b):
    """
    x: [batch_size, input_dim]
    W: [input_dim, output_dim]
    b: [output_dim]
    """
    return x @ W + b  # [batch_size, output_dim]
```

#### (3) 逐元素乘法 (Element-wise Multiplication)

也叫 Hadamard 积，符号：`⊙`

```
┌     ┐   ┌     ┐   ┌     ┐
│ 1 2 │ ⊙ │ 5 6 │ = │ 5 12│
│ 3 4 │   │ 7 8 │   │21 32│
└     ┘   └     ┘   └     ┘
```

```python
A * B  # NumPy/PyTorch中的 *
```

**深度学习应用**：激活函数、dropout
```python
# ReLU激活
mask = (x > 0).astype(float)  # 掩码矩阵
output = x * mask  # 逐元素乘法

# Dropout
dropout_mask = (np.random.rand(*x.shape) > 0.5)
output = x * dropout_mask
```

---

## 3. 特殊矩阵

### 3.1 单位矩阵 (Identity Matrix)

对角线为1，其余为0：

```
    ┌       ┐
I = │ 1 0 0 │
    │ 0 1 0 │
    │ 0 0 1 │
    └       ┘
```

**性质**：`A @ I = I @ A = A`（就像数字乘以1）

```python
I = np.eye(3)  # 3×3单位矩阵
```

### 3.2 零矩阵 (Zero Matrix)

全为0的矩阵：

```python
Z = np.zeros((2, 3))
```

### 3.3 对角矩阵 (Diagonal Matrix)

只有对角线有值：

```
    ┌       ┐
D = │ 2 0 0 │
    │ 0 3 0 │
    │ 0 0 5 │
    └       ┘
```

```python
D = np.diag([2, 3, 5])
```

**深度学习应用**：
- **批量归一化 (Batch Norm)**：缩放参数 `γ` 可以用对角矩阵表示
- **权重初始化**：某些初始化方法使用对角矩阵

---

## 4. 矩阵转置与逆：深度对比

**这是初学者最容易混淆的两个概念**。虽然两者都产生一个新矩阵，但本质完全不同。

### 4.1 代数定义对比

| 属性 | 转置 Aᵀ | 逆矩阵 A⁻¹ |
|------|---------|-----------|
| **定义** | 行列互换 | 满足 AA⁻¹ = A⁻¹A = I |
| **符号** | Aᵀ 或 A' | A⁻¹ |
| **存在性** | 总是存在 | 仅方阵且满秩时存在 |
| **形状** | (m×n) → (n×m) | (n×n) → (n×n) |
| **计算复杂度** | O(mn) | O(n³) |

**转置的计算**：
```
     ┌         ┐          ┌         ┐
A =  │ 1  2  3 │    Aᵀ =  │ 1  4    │
     │ 4  5  6 │          │ 2  5    │
     └         ┘          │ 3  6    │
      (2×3)               └         ┘
                           (3×2)
```

**逆矩阵的计算**（以2×2为例）：
```
     ┌     ┐              ┌              ┐
A =  │ a b │    A⁻¹ = 1/(ad-bc) × │ d  -b │
     │ c d │              │ -c  a │
     └     ┘              └              ┘

条件：行列式 det(A) = ad - bc ≠ 0
```

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

# 转置：总是可行
A_T = A.T

# 逆矩阵：需要行列式非零
det = np.linalg.det(A)
if det != 0:
    A_inv = np.linalg.inv(A)
    # 验证：A @ A_inv 应该是单位矩阵
    print(A @ A_inv)  # 接近 [[1, 0], [0, 1]]
```

### 4.2 几何意义对比

#### 转置的几何意义：镜像翻转

**直观理解**：转置相当于沿主对角线"翻折"

```
原矩阵 A（2×3）：              转置 Aᵀ（3×2）：

    列1  列2  列3                  行1→ 列1  列2
    ↓    ↓    ↓                      ↓    ↓
行1 → 1    2    3              行1 → 1    4
行2 → 4    5    6              行2 → 2    5
                               行3 → 3    6

（原来的行变成列，原来的列变成行）
```

**几何变换视角**：
- 转置不是"逆向变换"，而是"交换坐标轴视角"
- 如果 A 表示从空间 X 到空间 Y 的变换，则 Aᵀ 表示从 Y 到 X 的"伴随变换"

**可视化示例**：
```python
import matplotlib.pyplot as plt

# 原矩阵：一个"倾斜"的变换
A = np.array([[1, 0.5],
              [0, 1]])

# 转置矩阵
A_T = A.T

# 绘制单位正方形被变换后的形状
def plot_transformation(M, ax, title, color):
    # 单位正方形的顶点
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    transformed = M @ square
    ax.plot(transformed[0], transformed[1], color + '-o', linewidth=2)
    ax.set_title(title)
    ax.set_xlim(-0.5, 2)
    ax.set_ylim(-0.5, 2)
    ax.grid(True)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plot_transformation(np.eye(2), axes[0], 'Original (I)', 'gray')
plot_transformation(A, axes[1], 'A (shear)', 'blue')
plot_transformation(A_T, axes[2], 'Aᵀ (transpose)', 'red')
plt.tight_layout()
plt.savefig('transpose_visualization.png')
```

#### 逆矩阵的几何意义：逆向变换

**直观理解**：逆矩阵是原矩阵的"撤销操作"

```
原变换 A：将向量 v 变换到 v'
     A @ v = v'

逆变换 A⁻¹：将 v' 变回 v
     A⁻¹ @ v' = v

组合：A⁻¹ @ A @ v = v （相当于没变）
```

**具体例子**：
```python
# 旋转矩阵：逆时针旋转45度
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# 逆矩阵：顺时针旋转45度（撤销旋转）
R_inv = np.linalg.inv(R)

# 验证：R⁻¹ = Rᵀ（旋转矩阵是正交矩阵）
print("R⁻¹ ≈ Rᵀ?", np.allclose(R_inv, R.T))  # True

v = np.array([1, 0])
v_rotated = R @ v           # 旋转后
v_recovered = R_inv @ v_rotated  # 变回来

print("原向量:", v)
print("旋转后:", v_rotated)
print("变回来:", v_recovered)
```

**可视化对比**：
```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# 原始基向量
e1 = np.array([1, 0])  # 蓝色
e2 = np.array([0, 1])  # 红色

A = np.array([[2, 1], [0, 1]])  # 切变+拉伸
A_inv = np.linalg.inv(A)

print(f"矩阵 A:\n{A}")
print(f"逆矩阵 A⁻¹:\n{A_inv}")
print(f"验证 A⁻¹ @ A = I:\n{A_inv @ A}")

# A 变换后的基向量
A_e1 = A @ e1
A_e2 = A @ e2

# 1. 原始向量
ax = axes[0]
ax.arrow(0, 0, e1[0], e1[1], head_width=0.1, color='blue')
ax.arrow(0, 0, e2[0], e2[1], head_width=0.1, color='red')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title('Original\nBasis vectors')
ax.grid(True)

# 2. 经 A 变换后
ax = axes[1]
ax.arrow(0, 0, A_e1[0], A_e1[1], head_width=0.1, color='blue')
ax.arrow(0, 0, A_e2[0], A_e2[1], head_width=0.1, color='red')
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 2)
ax.set_title('After A\n(shear + stretch)')
ax.grid(True)

# 3. 经 Aᵀ 变换后（转置 ≠ 逆）
A_T = A.T
ax = axes[2]
ax.arrow(0, 0, A_T[0,0], A_T[1,0], head_width=0.1, color='blue')
ax.arrow(0, 0, A_T[0,1], A_T[1,1], head_width=0.1, color='red')
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 3)
ax.set_title('After Aᵀ\n(transpose ≠ inverse)')
ax.grid(True)

# 4. A⁻¹ 直接作用于原始基向量（这不是恢复！）
ax = axes[3]
ax.arrow(0, 0, A_inv[0,0], A_inv[1,0], head_width=0.1, color='blue')
ax.arrow(0, 0, A_inv[0,1], A_inv[1,1], head_width=0.1, color='red')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title('A⁻¹ on original\n(different transform)')
ax.grid(True)

# 5. A⁻¹ 作用于 A 变换后的向量（真正的恢复！）
recovered_e1 = A_inv @ A_e1
recovered_e2 = A_inv @ A_e2
ax = axes[4]
ax.arrow(0, 0, recovered_e1[0], recovered_e1[1], head_width=0.1, color='blue')
ax.arrow(0, 0, recovered_e2[0], recovered_e2[1], head_width=0.1, color='red')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title('A⁻¹ after A\n(recovers original!)')
ax.grid(True)

print(f"\n恢复验证:")
print(f"A⁻¹ @ A @ e1 = {recovered_e1} (应为 [1,0])")
print(f"A⁻¹ @ A @ e2 = {recovered_e2} (应为 [0,1])")

plt.tight_layout()
plt.savefig('transpose_vs_inverse_corrected.png', dpi=150)
plt.show()
```

### 4.3 物理含义对比

| 场景 | 转置 Aᵀ | 逆矩阵 A⁻¹ |
|------|---------|-----------|
| **信号处理** | 共轭转置用于功率谱 | 逆滤波器恢复原始信号 |
| **控制理论** | 可控性矩阵转置判断可观性 | 解耦控制器设计 |
| **机器学习** | 特征与样本的视角转换 | 正规方程求解权重 |
| **量子力学** | 厄米算符的厄米共轭 | 时间反演算符 |
| **图像处理** | 图像旋转90° | 图像复原（去模糊） |

**深度学习中的典型应用**：

```python
# ========== 转置的应用 ==========

# 1. 反向传播梯度计算
# 前向：y = W @ x
# 反向：∂L/∂x = Wᵀ @ ∂L/∂y
def backward_linear(grad_output, W):
    grad_input = W.T @ grad_output
    return grad_input

#### ★ 为什么反向传播用转置？—— 切向量与法向量的对偶

**这是理解转置在深度学习中作用的核心问题**。答案在于：**雅可比矩阵的转置建立了切空间与余切空间（法向量空间）的对偶关系**。

##### (1) 前向传播：切向量的变换

考虑线性层 `y = W @ x`，其中 W 是一个 (m, n) 矩阵。

当我们改变输入 x 时，输出 y 如何变化？这由**雅可比矩阵** J = ∂y/∂x = W 描述：

```
dy = J @ dx = W @ dx
```

这里的 `dx` 是输入空间的一个**切向量**——它表示输入的微小变化方向。

```
输入空间 (n维)                    输出空间 (m维)
    ┌─────┐                          ┌─────┐
    │  x  │ ──── W (前向) ───→       │  y  │
    │ dx  │ ──── W (切向量变换) ─→   │ dy  │
    └─────┘                          └─────┘

切向量 dx 沿着 W 的"行方向"被推送到输出空间
```

##### (2) 反向传播：法向量（梯度）的变换

现在考虑损失函数 L 对 y 的梯度 ∂L/∂y。这是一个**法向量**（或称余切向量）——它不是表示"向哪个方向移动"，而是表示"函数值在各个方向上的变化率"。

**关键洞察**：梯度不是沿变换方向传播的，而是沿**对偶方向**传播的！

```
输出空间 (m维)                    输入空间 (n维)
    ┌─────┐                          ┌─────┐
    │  y  │                          │  x  │
    │∂L/∂y│ ──── Wᵀ (反向) ───→     │∂L/∂x│
    └─────┘                          └─────┘

梯度 ∂L/∂y 沿着 Wᵀ 的"列方向"被拉回到输入空间
```

##### (3) 数学推导：链式法则与转置

根据链式法则：
```
∂L/∂x = (∂y/∂x)ᵀ @ (∂L/∂y) = Wᵀ @ (∂L/∂y)
```

**为什么是转置？** 因为：

- `∂y/∂x = W` 是一个 (m, n) 矩阵，描述"输入变化如何影响输出"
- 但我们要计算的是标量 L 对向量 x 的梯度，这需要将 (m,) 维的 ∂L/∂y 映射回 (n,) 维
- 数学上，这需要**对偶映射**，而矩阵的对偶映射就是其转置

##### (4) 几何直观：行空间与列空间的对偶

```
W 的结构：

        列1   列2   ...  列n
       ┌                    ┐
行1  →│ w₁₁  w₁₂  ...  w₁ₙ │
行2  →│ w₂₁  w₂₂  ...  w₂ₙ │
...   │ ...  ...  ...  ... │
行m  →│ wₘ₁  wₘ₂  ...  wₘₙ │
       └                    ┘

前向传播 (W @ x)：输出是 W 各行的线性组合，权重由 x 的分量决定
反向传播 (Wᵀ @ g)：输出是 W 各列的线性组合，权重由 g 的分量决定
```

**几何解释**：
- **前向**：x 的每个分量"激活" W 的一行，输出在行空间中
- **反向**：梯度 g 的每个分量"激活" W 的一列，梯度在列空间中

##### (5) 可视化示例

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义一个 2D → 2D 的线性变换
W = np.array([[2, 1],
              [0, 1]])

# ===== 前向传播：切向量变换 =====
# 输入空间的切向量（微小变化方向）
dx = np.array([1, 0])  # 沿 x₁ 轴方向

# 输出空间的切向量
dy = W @ dx  # = [2, 0]

# ===== 反向传播：梯度变换 =====
# 输出空间的梯度（损失对各输出的敏感度）
grad_y = np.array([1, 0])  # y₁ 方向敏感

# 输入空间的梯度
grad_x = W.T @ grad_y  # = [2, 0]

print("前向变换:")
print(f"  切向量 dx = {dx}")
print(f"  变换后 dy = W @ dx = {dy}")

print("\n反向传播:")
print(f"  梯度 ∂L/∂y = {grad_y}")
print(f"  传回 ∂L/∂x = Wᵀ @ ∂L/∂y = {grad_x}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：前向传播
ax = axes[0]
ax.arrow(0, 0, dx[0], dx[1], head_width=0.1, color='blue', label='dx (input tangent)')
ax.arrow(0, 0, dy[0], dy[1], head_width=0.1, color='red', label='dy = W @ dx')
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 2)
ax.set_aspect('equal')
ax.set_title('Forward: Tangent Vector Transform')
ax.legend()
ax.grid(True)

# 右图：反向传播
ax = axes[1]
ax.arrow(0, 0, grad_y[0], grad_y[1], head_width=0.1, color='green', label='∂L/∂y (output gradient)')
ax.arrow(0, 0, grad_x[0], grad_x[1], head_width=0.1, color='purple', label='∂L/∂x = Wᵀ @ ∂L/∂y')
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 2)
ax.set_aspect('equal')
ax.set_title('Backward: Gradient Transform (uses Wᵀ)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('forward_backward_transpose.png', dpi=150)
plt.show()
```

##### (6) 更一般的视角：对偶空间

在微分几何中：
- **切空间 TₓM**：向量所在的空间，表示"运动方向"
- **余切空间 Tₓ*M**：梯度的空间，表示"函数变化率"

线性映射 `y = W @ x` 诱导：
- **推前映射 (pushforward)**：`W: Tₓ → Ty`，即前向传播
- **拉回映射 (pullback)**：`Wᵀ: T*y → T*x`，即反向传播

**这就是为什么反向传播必须用转置！** 转置不是某种"技巧"，而是数学上对偶映射的自然表达。

##### (7) 深度学习中的启示

| 操作 | 变换 | 空间映射 |
|------|------|----------|
| 前向传播 | `y = W @ x` | 输入空间 → 输出空间 |
| 反向传播 | `∂L/∂x = Wᵀ @ ∂L/∂y` | 输出余切空间 → 输入余切空间 |
| 权重梯度 | `∂L/∂W = ∂L/∂y ⊗ x` | 外积（组合两个空间） |

```python
# 完整的反向传播示例
def linear_forward_backward(x, W):
    """
    完整的线性层前向和反向传播

    前向：y = W @ x
    反向：需要计算三个梯度
    """
    # 前向传播
    y = W @ x

    # 假设来自上层的梯度
    grad_y = np.ones_like(y)

    # 反向传播（关键：理解三个梯度的计算）
    # (1) 对输入的梯度：Wᵀ @ grad_y
    grad_x = W.T @ grad_y

    # (2) 对权重的梯度：grad_y ⊗ x（外积）
    grad_W = np.outer(grad_y, x)

    # (3) 对偏置的梯度：直接传递
    # grad_b = grad_y（如果有的话）

    print("前向传播:")
    print(f"  x = {x}")
    print(f"  W = \n{W}")
    print(f"  y = W @ x = {y}")

    print("\n反向传播:")
    print(f"  ∂L/∂y = {grad_y}")
    print(f"  ∂L/∂x = Wᵀ @ ∂L/∂y = {grad_x}")
    print(f"  ∂L/∂W = ∂L/∂y ⊗ x = \n{grad_W}")

    return y, grad_x, grad_W

# 测试
x = np.array([1.0, 2.0])
W = np.array([[1.0, 2.0],
              [3.0, 4.0]])
y, grad_x, grad_W = linear_forward_backward(x, W)
```

# 2. 注意力机制中的 Key 和 Query
# Q @ Kᵀ：Query 与 Key 的相似度
attention_scores = Q @ K.T / np.sqrt(d_k)

# 3. 特征转置用于不同视角
# X: [samples, features] → Xᵀ: [features, samples]
# 从"样本相似度"视角转到"特征相关性"视角
sample_similarity = X @ X.T      # 样本间的相似度
feature_correlation = X.T @ X   # 特征间的相关性

# ========== 逆矩阵的应用 ==========

# 1. 求解线性方程组 Ax = b
# 解：x = A⁻¹ @ b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.inv(A) @ b  # 或 x = np.linalg.solve(A, b)

# 2. 线性回归的正规方程
# 最优解：W = (XᵀX)⁻¹Xᵀy
def linear_regression_closed_form(X, y):
    # 添加偏置项
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # 正规方程
    W = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return W

# 3. 数据白化（Whitening）
# 使数据协方差矩阵变为单位矩阵
def whitening(X):
    # 计算协方差矩阵
    cov = np.cov(X.T)
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 白化矩阵
    W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return X @ W
```

### 4.4 性质对比

```
┌─────────────────────────────────────────────────────────────────┐
│                         性质对比表                              │
├─────────────────┬─────────────────────┬─────────────────────────┤
│      性质       │       转置 Aᵀ        │        逆矩阵 A⁻¹       │
├─────────────────┼─────────────────────┼─────────────────────────┤
│ 自反性          │ (Aᵀ)ᵀ = A           │ (A⁻¹)⁻¹ = A             │
│ 和的运算        │ (A + B)ᵀ = Aᵀ + Bᵀ  │ (A + B)⁻¹ ≠ A⁻¹ + B⁻¹   │
│ 积的运算        │ (AB)ᵀ = BᵀAᵀ        │ (AB)⁻¹ = B⁻¹A⁻¹         │
│ 数乘            │ (kA)ᵀ = kAᵀ         │ (kA)⁻¹ = (1/k)A⁻¹       │
│ 行列式          │ det(Aᵀ) = det(A)    │ det(A⁻¹) = 1/det(A)     │
│ 秩              │ rank(Aᵀ) = rank(A)  │ rank(A⁻¹) = rank(A) = n │
│ 特征值          │ 与A相同             │ 特征值为倒数             │
│ 正交矩阵        │ Aᵀ = A⁻¹            │ A⁻¹ = Aᵀ                │
└─────────────────┴─────────────────────┴─────────────────────────┘
```

**代码验证**：
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# ===== 转置的性质 =====
# (1) 自反性
print("(Aᵀ)ᵀ == A:", np.allclose(A.T.T, A))

# (2) 和的性质
print("(A+B)ᵀ == Aᵀ+Bᵀ:", np.allclose((A+B).T, A.T + B.T))

# (3) 积的性质（注意顺序反转！）
print("(AB)ᵀ == BᵀAᵀ:", np.allclose((A@B).T, B.T @ A.T))

# ===== 逆矩阵的性质 =====
# (1) 自反性
A_inv = np.linalg.inv(A)
print("(A⁻¹)⁻¹ == A:", np.allclose(np.linalg.inv(A_inv), A))

# (2) 积的性质（注意顺序反转！）
B_inv = np.linalg.inv(B)
print("(AB)⁻¹ == B⁻¹A⁻¹:", np.allclose(np.linalg.inv(A@B), B_inv @ A_inv))

# (3) 行列式
print("det(A⁻¹) == 1/det(A):", np.allclose(np.linalg.det(A_inv), 1/np.linalg.det(A)))
```

### 4.5 特殊情况：正交矩阵

**正交矩阵是转置等于逆的特殊矩阵**：

```
正交矩阵：Qᵀ = Q⁻¹  或等价地  QᵀQ = QQᵀ = I
```

**几何意义**：正交矩阵只做旋转/反射，不做拉伸或剪切

```python
# 旋转矩阵是正交矩阵
theta = np.pi / 6  # 30度
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print("QᵀQ = I?", np.allclose(Q.T @ Q, np.eye(2)))
print("Q⁻¹ = Qᵀ?", np.allclose(np.linalg.inv(Q), Q.T))
print("det(Q) =", np.round(np.linalg.det(Q)))  # 应该是1（纯旋转）或-1（反射）
```

**正交矩阵的优势**：
1. 数值稳定性：求逆只需转置，O(n²) vs O(n³)
2. 保范数：‖Qv‖ = ‖v‖（向量长度不变）
3. 保内积：(Qv)·(Qw) = v·w（向量夹角不变）

### 4.6 常见误区澄清

| 误区 | 正确理解 |
|------|----------|
| ❌ 转置就是逆 | ✓ 只有正交矩阵才有 Aᵀ = A⁻¹ |
| ❌ 转置会"撤销"变换 | ✓ 转置是"换视角"，逆才是"撤销" |
| ❌ 所有方阵都有逆 | ✓ 只有非奇异矩阵（det ≠ 0）才有逆 |
| ❌ AᵀA = I | ✓ 一般不成立，但 AᵀA 是对称矩阵 |
| ❌ 逆矩阵总是存在 | ✓ 不可逆矩阵（奇异矩阵）大量存在 |

**代码演示误区**：
```python
# 误区：认为转置和逆等价
A = np.array([[1, 2], [3, 4]])
print("Aᵀ == A⁻¹?", np.allclose(A.T, np.linalg.inv(A)))  # False!

# 正确：只有正交矩阵才等价
Q = np.array([[0, -1], [1, 0]])  # 90度旋转矩阵
print("Qᵀ == Q⁻¹?", np.allclose(Q.T, np.linalg.inv(Q)))  # True!

# AᵀA 的性质
print("AᵀA 是对称矩阵:", np.allclose((A.T @ A).T, A.T @ A))  # True
print("AᵀA 的形状:", (A.T @ A).shape)  # (2, 2)
```

### 4.7 深度学习中的选择指南

| 场景 | 用转置 Aᵀ | 用逆矩阵 A⁻¹ |
|------|----------|-------------|
| 反向传播计算梯度 | ✓ | ✗ |
| 注意力机制 QKᵀ | ✓ | ✗ |
| 特征/样本视角转换 | ✓ | ✗ |
| 求解线性系统 Ax=b | ✗ | ✓（或用 solve()） |
| 数据白化 | ✗ | ✓ |
| 批归一化的协方差 | ✓（协方差计算） | ✓（白化） |
| 线性回归正规方程 | ✓（XᵀX 部分） | ✓（求逆部分） |

**数值计算建议**：
```python
# ❌ 避免直接求逆（数值不稳定）
x = np.linalg.inv(A) @ b

# ✓ 使用 solve（更稳定）
x = np.linalg.solve(A, b)

# ✓ 使用伪逆（处理不可逆情况）
x = np.linalg.pinv(A) @ b
```

---

## 5. 矩阵的几何意义

### 5.1 矩阵是线性变换

矩阵乘法 = 对向量进行线性变换

**旋转矩阵**（逆时针旋转θ）：
```
    ┌            ┐
R = │ cos(θ) -sin(θ) │
    │ sin(θ)  cos(θ) │
    └            ┘
```

```python
theta = np.pi / 4  # 45度
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

v = np.array([1, 0])
v_rotated = R @ v  # 旋转后的向量
```

**缩放矩阵**：
```
    ┌     ┐
S = │ 2 0 │  # x方向放大2倍，y方向放大3倍
    │ 0 3 │
    └     ┘
```

### 5.2 神经网络是函数组合

```python
# 两层神经网络
h = x @ W1 + b1      # 第一层：线性变换
h = relu(h)          # 激活：非线性变换
y = h @ W2 + b2      # 第二层：线性变换
```

每一层的权重矩阵都在进行几何变换，激活函数引入非线性。

---

## 6. 实战示例

### 例1：用矩阵表示全连接层

```python
import numpy as np

# 输入：3个样本，每个4维特征
X = np.random.randn(3, 4)

# 权重：4个输入神经元 → 2个输出神经元
W = np.random.randn(4, 2)
b = np.random.randn(2)

# 前向传播
output = X @ W + b  # (3, 4) @ (4, 2) + (2,) = (3, 2)
print(output.shape)  # (3, 2)
```

### 例2：批量处理图像

```python
# 100张28×28灰度图像
images = np.random.randn(100, 28, 28)

# 展平成矩阵：每行是一张图
images_flat = images.reshape(100, -1)  # (100, 784)

# 通过全连接层
W = np.random.randn(784, 10)  # 10类分类
logits = images_flat @ W  # (100, 10)
```

---

## 7. 常见错误

### ❌ 错误1：维度不匹配

```python
A = np.random.randn(3, 4)
B = np.random.randn(3, 5)
C = A @ B  # 错误！4 ≠ 3
```

**解决**：检查维度 `(m, n) @ (n, p)` 中间的 `n` 必须相等

### ❌ 错误2：混淆 `@` 和 `*`

```python
A @ B  # 矩阵乘法
A * B  # 逐元素乘法（Hadamard积）
```

### ❌ 错误3：忘记转置

```python
# 权重形状 [input_dim, output_dim]
W = torch.randn(512, 256)

# 错误：x是 [batch, input_dim]，直接乘会报错
x = torch.randn(32, 512)
out = x @ W  # ✓ 正确

# 如果W存储为 [output_dim, input_dim]，需要转置
W_T = torch.randn(256, 512)
out = x @ W_T.T  # ✓ 正确
```

---

## 8. 速查表

| 操作 | NumPy | PyTorch |
|-----|-------|---------|
| 向量点积 | `np.dot(a, b)` | `torch.dot(a, b)` |
| 矩阵乘法 | `A @ B` 或 `np.matmul(A, B)` | `A @ B` 或 `torch.matmul(A, B)` |
| 逐元素乘 | `A * B` | `A * B` |
| 转置 | `A.T` | `A.T` |
| 范数 | `np.linalg.norm(v)` | `torch.norm(v)` |
| 形状 | `A.shape` | `A.shape` |
| 变形 | `A.reshape(...)` | `A.view(...)` |

---

## 练习

1. **基础练习**：计算两个向量的点积、L2范数
2. **维度练习**：给定形状 `(batch, seq_len, hidden_dim)` 的张量，如何通过矩阵乘法得到 `(batch, seq_len, output_dim)`？
3. **实战练习**：实现一个简单的两层神经网络（只用矩阵运算，不用框架）

---

**下一步**：学习 [02_matrix_operations.md](./02_matrix_operations.md)，深入理解矩阵运算在神经网络中的应用。
