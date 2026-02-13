# 04 优化理论 (Optimization)

## 为什么需要优化理论？

深度学习的训练过程就是一个优化问题：

### 核心目标
```
最小化损失函数: min L(θ)
                θ
其中 θ 是模型的所有参数
```

### 实际挑战
1. **高维优化**：现代神经网络有数百万到数十亿参数
2. **非凸优化**：深度网络的损失函数有大量局部最优
3. **数据量大**：无法每次用全部数据计算梯度
4. **训练效率**：需要快速收敛

## 最小必要知识

### ✅ 必须掌握
1. **梯度下降基础**：沿着梯度反方向更新
2. **随机梯度下降SGD**：用mini-batch估计梯度
3. **学习率调整**：学习率对收敛的影响
4. **动量Momentum**：加速收敛，减少震荡
5. **Adam优化器**：自适应学习率（实践中最常用）

### ⚠️ 了解即可
6. 凸优化基础（理解什么是凸函数）
7. 牛顿法和拟牛顿法（二阶优化）
8. 学习率衰减策略
9. 梯度裁剪（处理梯度爆炸）

### ❌ 可以跳过
- 凸优化的详细证明
- 复杂的约束优化算法
- 内点法等传统优化算法

## 学习路径

```
第1步：凸优化基础 (30分钟)
├── docs/01_convex_optimization.md
└── notebooks/convex_functions.ipynb  # 待创建

第2步：梯度方法 (60分钟)
├── docs/02_gradient_methods.md
└── notebooks/optimizer_comparison.ipynb

第3步：约束优化 (45分钟)
├── docs/03_lagrange.md
└── notebooks/constrained_optimization.ipynb  # 待创建
```

## 目录内容

### 📄 docs/ - 理论文档
- `01_convex_optimization.md` - 凸优化基础概念
- `02_gradient_methods.md` - SGD、Momentum、Adam等优化器
- `03_lagrange.md` - 拉格朗日乘子法（约束优化）

### 💻 notebooks/ - 交互式实践
- `convex_functions.ipynb` - 凸函数和非凸函数可视化
- `optimizer_comparison.ipynb` - 对比不同优化器的表现
- `constrained_optimization.ipynb` - 约束优化示例

### 🔧 code/ - 实用代码
- `optimizers.py` - 从零实现SGD、Momentum、Adam

## 快速测试

完成本模块后，你应该能够：

- [ ] 从零实现SGD、Momentum、Adam优化器
- [ ] 理解学习率对训练的影响
- [ ] 选择合适的优化器和超参数
- [ ] 解释为什么Adam在实践中效果好
- [ ] 处理梯度消失/爆炸问题

## 与深度学习的连接

| 优化概念 | 深度学习应用 | PyTorch实现 |
|---------|------------|-------------|
| SGD | 基础优化器 | `torch.optim.SGD()` |
| Momentum | 加速收敛 | `SGD(momentum=0.9)` |
| Adam | 自适应学习率 | `torch.optim.Adam()` |
| 学习率衰减 | 训练后期精调 | `lr_scheduler.StepLR()` |
| 梯度裁剪 | 防止梯度爆炸 | `torch.nn.utils.clip_grad_norm_()` |
| 权重衰减 | L2正则化 | `weight_decay=1e-4` |

## 核心公式速查

### SGD（随机梯度下降）
```
θ_t = θ_{t-1} - η × ∇L(θ_{t-1})
```

### Momentum（动量）
```
v_t = β × v_{t-1} + ∇L(θ_{t-1})
θ_t = θ_{t-1} - η × v_t
```
常用 β = 0.9

### Adam（结合动量和自适应学习率）
```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇L     # 一阶矩估计
v_t = β₂ × v_{t-1} + (1-β₂) × (∇L)²  # 二阶矩估计
θ_t = θ_{t-1} - η × m̂_t / (√v̂_t + ε)
```
常用 β₁=0.9, β₂=0.999, ε=1e-8

### 学习率衰减
```
# Step decay
η_t = η₀ × γ^⌊t/k⌋

# Exponential decay
η_t = η₀ × e^(-λt)

# Cosine annealing
η_t = η_min + (η₀ - η_min) × (1 + cos(πt/T)) / 2
```

## 优化器选择指南

| 优化器 | 适用场景 | 优点 | 缺点 |
|-------|---------|------|------|
| **SGD** | 大规模数据集 | 简单、内存少 | 需要调学习率 |
| **SGD+Momentum** | CV任务 | 收敛快 | 超参数敏感 |
| **Adam** | 通用首选 | 自适应、鲁棒 | 可能泛化略差 |
| **AdamW** | NLP/Transformer | 解耦权重衰减 | 计算稍慢 |
| **RMSprop** | RNN | 处理非平稳目标 | 已被Adam取代 |

**实践建议**：
- 默认用 **Adam** (lr=1e-3)
- CV任务可尝试 **SGD+Momentum** (lr=0.1, momentum=0.9)
- Transformer用 **AdamW**

## 推荐资源

### 论文
- Adam: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- AdamW: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

### 文章
- [CS231n - Optimization](http://cs231n.github.io/optimization-1/)
- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)

### 可视化
- [Gradient Descent Visualization](https://vis.ensmallen.org/)
- [Why Momentum Really Works](https://distill.pub/2017/momentum/)

## 调试技巧

### 学习率太大
- 症状：Loss震荡或NaN
- 解决：降低学习率10倍

### 学习率太小
- 症状：Loss下降太慢
- 解决：增大学习率或用学习率预热

### 梯度消失
- 症状：前层参数不更新
- 解决：Batch Normalization、残差连接、更好的初始化

### 梯度爆炸
- 症状：Loss突然变成NaN
- 解决：梯度裁剪、降低学习率

## 下一步

完成优化理论后，前往 [05_information_theory](../05_information_theory/) 学习信息论，理解损失函数的数学原理。
