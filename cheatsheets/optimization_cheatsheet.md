# 优化理论速查表

## 1. 基础概念

### 优化问题一般形式
```
minimize   f(x)
subject to gᵢ(x) ≤ 0,  i = 1,...,m  (不等式约束)
           hⱼ(x) = 0,  j = 1,...,p  (等式约束)
```

### 无约束优化
```
minimize f(x)

最优条件:
∇f(x*) = 0         (一阶必要条件)
∇²f(x*) ⪰ 0        (二阶必要条件，Hessian半正定)
```

---

## 2. 梯度下降

### 标准梯度下降 (GD)
```
xₜ₊₁ = xₜ - α ∇f(xₜ)

α: 学习率 (learning rate)
```

**特点**:
- 使用全部数据
- 收敛稳定但慢

### 随机梯度下降 (SGD)
```
xₜ₊₁ = xₜ - α ∇f(xₜ; ξₜ)

ξₜ: 随机抽样的样本
```

**特点**:
- 每次只用一个样本
- 快但噪声大

### Mini-batch SGD
```
xₜ₊₁ = xₜ - α ∇f(xₜ; Bₜ)

Bₜ: batch（如32、64个样本）
```

**特点**:
- 平衡速度和稳定性
- **深度学习标配**

---

## 3. 动量方法

### Momentum
```
vₜ = β vₜ₋₁ + (1-β) ∇f(xₜ)
xₜ₊₁ = xₜ - α vₜ

β: 动量系数（典型值0.9）
```

**直觉**:
- 积累过去梯度的"惯性"
- 加速相同方向，减缓震荡方向

### Nesterov Momentum (NAG)
```
# "先跳再看"
xₜₑₘₚ = xₜ + β vₜ₋₁
vₜ = β vₜ₋₁ + ∇f(xₜₑₘₚ)
xₜ₊₁ = xₜ - α vₜ
```

**优势**: 更智能的动量，收敛更快

---

## 4. 自适应学习率

### AdaGrad
```
gₜ = ∇f(xₜ)
sₜ = sₜ₋₁ + gₜ ⊙ gₜ    (累积平方梯度)
xₜ₊₁ = xₜ - α gₜ / (√sₜ + ε)

ε: 防止除零（如1e-8）
```

**特点**:
- 频繁更新的参数 → 小学习率
- 稀疏更新的参数 → 大学习率
- **问题**: 学习率单调递减

### RMSprop
```
sₜ = β sₜ₋₁ + (1-β) gₜ ⊙ gₜ    (指数移动平均)
xₜ₊₁ = xₜ - α gₜ / (√sₜ + ε)

β: 衰减率（典型值0.9）
```

**改进**: 解决AdaGrad学习率递减问题

### Adam (最常用！)
```
# 一阶矩估计（动量）
mₜ = β₁ mₜ₋₁ + (1-β₁) gₜ

# 二阶矩估计（自适应学习率）
vₜ = β₂ vₜ₋₁ + (1-β₂) gₜ ⊙ gₜ

# 偏差修正
m̂ₜ = mₜ / (1 - β₁ᵗ)
v̂ₜ = vₜ / (1 - β₂ᵗ)

# 更新
xₜ₊₁ = xₜ - α m̂ₜ / (√v̂ₜ + ε)

默认超参数:
β₁ = 0.9
β₂ = 0.999
ε = 1e-8
```

**为什么流行**:
- 结合Momentum + RMSprop
- 对超参数不敏感
- 适用于大多数场景

### AdamW (权重衰减)
```
# 标准Adam更新
θₜ₊₁ = θₜ - α m̂ₜ / (√v̂ₜ + ε)

# 额外的权重衰减
θₜ₊₁ = θₜ₊₁ - α λ θₜ

λ: 权重衰减系数（如0.01）
```

**重要**: 与L2正则化不完全等价

---

## 5. 学习率调度

### 常见策略

#### Step Decay
```python
lr = lr₀ × γ^(epoch // step_size)

# 每step_size个epoch衰减γ倍
```

#### Exponential Decay
```python
lr = lr₀ × e^(-λt)
```

#### Cosine Annealing
```python
lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))

T: 总epoch数
```

#### Warmup
```python
# 前几个epoch线性增加学习率
if epoch < warmup_epochs:
    lr = lr_max × (epoch / warmup_epochs)
else:
    lr = lr_max  # 或其他衰减策略
```

**用途**: Transformer训练标配

---

## 6. 凸优化基础

### 凸函数
```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)

∀ x, y, λ ∈ [0,1]
```

**性质**:
- 局部最优 = 全局最优
- 梯度为0的点是最优解

**常见凸函数**:
- 线性函数: `aᵀx + b`
- 二次函数: `½xᵀQx + bᵀx` (Q ⪰ 0)
- 范数: `||x||ₚ`
- 负熵: `-Σᵢ xᵢ log xᵢ`

### 强凸函数
```
f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (m/2)||y-x||²

m > 0: 强凸参数
```

**优势**: 收敛更快（线性收敛率）

---

## 7. 约束优化

### 拉格朗日乘子法
```
L(x, λ) = f(x) + Σᵢ λᵢ hᵢ(x)

最优条件（KKT条件）:
∇ₓL = 0
hᵢ(x) = 0  ∀ i
```

**例子**: 带约束的最小二乘
```
minimize   ||Ax - b||²
subject to Cx = d

L = ||Ax - b||² + λᵀ(Cx - d)
```

### KKT条件（不等式约束）
```
minimize   f(x)
subject to gᵢ(x) ≤ 0

KKT条件:
1. ∇f(x*) + Σᵢ μᵢ ∇gᵢ(x*) = 0  (平稳性)
2. gᵢ(x*) ≤ 0                  (原始可行性)
3. μᵢ ≥ 0                       (对偶可行性)
4. μᵢ gᵢ(x*) = 0                (互补松弛)
```

---

## 8. 深度学习特有技巧

### 梯度裁剪 (Gradient Clipping)
```python
# 按范数裁剪
if ||g|| > threshold:
    g = threshold × g / ||g||

# 按值裁剪
g = clip(g, -threshold, threshold)
```

**用途**: 防止RNN梯度爆炸

### 梯度累积 (Gradient Accumulation)
```python
# 模拟大batch
for i in range(accumulation_steps):
    loss = forward(mini_batch[i])
    loss.backward()  # 累积梯度

optimizer.step()     # 一次更新
optimizer.zero_grad()
```

### 混合精度训练
```python
# FP16前向+反向，FP32更新权重
with autocast():
    loss = model(input)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**加速**: 2-3倍训练速度

---

## 9. 超参数调优

### 学习率选择
```
经验法则:
- Adam: 1e-3 ~ 3e-4
- SGD: 0.01 ~ 0.1 (需动量)
- 较大模型/数据 → 较小lr
```

**找最优lr**: Learning Rate Finder
```python
# 指数增长lr，记录loss
# 选择loss下降最快的lr
```

### Batch Size
```
小batch (32):
- 泛化性好
- 训练慢
- 内存友好

大batch (1024+):
- 训练快
- 需调大lr: lr ∝ √batch_size
- 泛化性可能差
```

---

## 10. Python实现

### PyTorch示例
```python
import torch.optim as optim

# SGD with Momentum
optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      weight_decay=1e-4)

# Adam
optimizer = optim.Adam(model.parameters(),
                       lr=1e-3,
                       betas=(0.9, 0.999))

# AdamW
optimizer = optim.AdamW(model.parameters(),
                        lr=1e-3,
                        weight_decay=0.01)

# 学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs
)

# 训练循环
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()

        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

    scheduler.step()
```

### 从零实现Adam
```python
class Adam:
    def __init__(self, params, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # 更新一阶矩
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # 更新二阶矩
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # 更新参数
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

---

## 11. 优化器对比

| 优化器 | 优点 | 缺点 | 适用场景 |
|-------|------|------|---------|
| **SGD** | 简单稳定 | 需调lr、慢 | 已有良好调参经验 |
| **SGD+Momentum** | 加速收敛 | 需调动量 | CV任务（ResNet等） |
| **Adam** | 自适应、鲁棒 | 有时泛化差 | **默认首选** |
| **AdamW** | Adam+正则 | - | Transformer标配 |
| **RMSprop** | 适合RNN | - | RNN/LSTM（已较少用） |

---

## 12. 调试技巧

### 检查梯度
```python
# 梯度消失/爆炸
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### Loss不下降？
```
1. 学习率太大 → 减小10倍
2. 学习率太小 → 增大10倍
3. 梯度消失 → 检查激活函数、初始化
4. 数据问题 → 检查标签、归一化
5. 模型问题 → 先过拟合小数据集
```

### 过拟合？
```
- 增大权重衰减 (weight_decay)
- 添加Dropout
- 数据增强
- 减小模型容量
- Early Stopping
```

---

## 13. 记忆口诀

**Adam = Momentum + RMSprop**
```
一阶矩（动量） + 二阶矩（自适应lr）
```

**学习率太大/太小**
```
太大: loss震荡、NaN
太小: 收敛慢、卡住
```

**Batch Size规则**
```
batch_size × 2 → lr × √2
（Linear Scaling Rule）
```

---

## 14. 常见陷阱

❌ **忘记zero_grad()**
```python
# 错误：梯度累积
for epoch in epochs:
    loss.backward()
    optimizer.step()  # 梯度会一直累加！

# 正确
for epoch in epochs:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

❌ **学习率衰减过早**
```python
# 可能在验证集上效果差
# 应该用验证集指标决定何时衰减
```

❌ **AdamW vs Adam+L2**
```python
# 不等价！
optimizer = Adam(..., weight_decay=0.01)  # ❌

# 应该用
optimizer = AdamW(..., weight_decay=0.01)  # ✓
```

---

**相关**: [微积分速查表](./calculus_cheatsheet.md) | [线性代数速查表](./linear_algebra_cheatsheet.md)
