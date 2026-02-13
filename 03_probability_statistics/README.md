# 03 概率统计 (Probability & Statistics)

## 为什么需要概率统计？

深度学习本质上是在处理不确定性：

### 1. 数据的不确定性
- 训练数据包含噪声
- 测试数据可能与训练分布不同
- 无法观测到所有可能的样本

### 2. 模型的概率输出
```python
# 分类问题输出概率分布
output = softmax(logits)  # [0.7, 0.2, 0.1] -> 类别概率
```

### 3. 损失函数的概率解释
- **交叉熵损失** = 最大化似然估计
- **MSE损失** = 假设高斯噪声的最大似然

### 4. 贝叶斯推理
- 不确定性估计
- 贝叶斯神经网络
- 主动学习

## 最小必要知识

### ✅ 必须掌握
1. **概率基础**：条件概率、联合概率、边缘概率
2. **期望和方差**：数据的统计特性
3. **常见分布**：
   - 伯努利/二项分布（分类问题）
   - 高斯分布（连续数据）
   - 多项分布（多分类）
4. **最大似然估计MLE**：参数估计的基础
5. **贝叶斯定理**：后验概率推理

### ⚠️ 了解即可
6. 协方差和相关性
7. 中心极限定理
8. MAP（最大后验估计）

### ❌ 可以跳过
- 假设检验的详细步骤
- 复杂的统计推断
- 高级概率论（测度论等）

## 学习路径

```
第1步：概率基础 (45分钟)
├── docs/01_probability_basics.md
└── notebooks/probability_basics.ipynb  # 待创建

第2步：常见分布 (60分钟)
├── docs/02_distributions.md
└── notebooks/distributions.ipynb

第3步：贝叶斯和MLE (75分钟)
├── docs/03_bayes_mle.md
└── notebooks/bayes_classifier.ipynb
```

## 目录内容

### 📄 docs/ - 理论文档
- `01_probability_basics.md` - 概率论基础概念
- `02_distributions.md` - 常见概率分布及其应用
- `03_bayes_mle.md` - 贝叶斯推理和最大似然估计

### 💻 notebooks/ - 交互式实践
- `probability_basics.ipynb` - 概率概念可视化
- `distributions.ipynb` - 各种分布的采样和可视化
- `bayes_classifier.ipynb` - 朴素贝叶斯分类器实现

### 🔧 code/ - 实用代码
- `probability_utils.py` - 概率计算工具函数

## 快速测试

完成本模块后，你应该能够：

- [ ] 理解并计算条件概率和联合概率
- [ ] 从高斯分布、伯努利分布中采样
- [ ] 用最大似然估计求解参数
- [ ] 实现简单的朴素贝叶斯分类器
- [ ] 解释交叉熵损失的概率意义

## 与深度学习的连接

| 概率统计概念 | 深度学习应用 | 示例 |
|------------|------------|------|
| 高斯分布 | 参数初始化、噪声建模 | `torch.randn()` |
| 伯努利分布 | Dropout、二分类 | `nn.Dropout()` |
| 多项分布 | 多分类输出 | `nn.CrossEntropyLoss()` |
| 最大似然估计 | 损失函数优化 | 交叉熵 = 负对数似然 |
| 贝叶斯推理 | 不确定性估计 | 贝叶斯神经网络 |
| 期望 | Batch Normalization | `E[x]` 和 `Var[x]` |

## 核心公式速查

### 条件概率
```
P(A|B) = P(A,B) / P(B)
```

### 贝叶斯定理
```
P(θ|D) = P(D|θ) × P(θ) / P(D)
后验    似然      先验    证据
```

### 期望和方差
```
E[X] = Σ x × P(x)          # 离散
E[X] = ∫ x × p(x) dx       # 连续

Var[X] = E[(X - E[X])²] = E[X²] - (E[X])²
```

### 高斯分布
```
N(x|μ,σ²) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
```

## 推荐资源

### 视频
- [StatQuest - Probability Basics](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
- [3Blue1Brown - Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)

### 书籍
- 《Deep Learning》第3章（Ian Goodfellow）
- 《Pattern Recognition and Machine Learning》第1-2章（Bishop）

### 交互式工具
- [Seeing Theory](https://seeing-theory.brown.edu/) - 概率统计可视化

## 实践建议

1. **多做采样实验** - 理解分布的形状和性质
2. **连接到损失函数** - 理解为什么使用交叉熵
3. **实现简单的贝叶斯分类器** - 加深对条件概率的理解

## 下一步

完成概率统计后，前往 [04_optimization](../04_optimization/) 学习优化理论，理解神经网络训练算法。
