# 贝叶斯推理与最大似然估计

## 为什么重要？

在深度学习中：
- **MLE（最大似然估计）**：几乎所有损失函数的理论基础
- **MAP（最大后验估计）**：正则化的概率解释
- **贝叶斯推理**：不确定性量化、贝叶斯神经网络

---

## 1. 最大似然估计 (MLE)

### 1.1 核心思想

**问题**：给定数据 D = {x₁, x₂, ..., xₙ}，如何估计模型参数 θ？

**MLE 思路**：选择使"观测到这些数据"概率最大的参数

```
θ_MLE = argmax P(D | θ)
         θ
```

### 1.2 数学推导

假设数据独立同分布（i.i.d.），似然函数为：

```
L(θ) = P(D | θ) = ∏ P(xᵢ | θ)
                   i=1

对数似然：
log L(θ) = Σ log P(xᵢ | θ)
           i=1
```

**为什么用对数？**
1. 连乘变连加，数值更稳定
2. 单调性不变（argmax 一样）
3. 方便求导

### 1.3 例子：估计高斯分布参数

**数据**：{x₁, ..., xₙ} ~ N(μ, σ²)

**似然函数**：
```
P(D | μ, σ²) = ∏ (1/√(2πσ²)) exp(-(xᵢ-μ)²/(2σ²))
```

**对数似然**：
```
log L = -n/2 log(2πσ²) - (1/(2σ²)) Σ(xᵢ - μ)²
```

**求导并令导数为0**：
```
∂log L/∂μ = (1/σ²) Σ(xᵢ - μ) = 0

=> μ_MLE = (1/n) Σ xᵢ  (样本均值！)
```

**代码验证**：
```python
import numpy as np

# 生成数据
true_mu, true_sigma = 5.0, 2.0
data = np.random.normal(true_mu, true_sigma, size=1000)

# MLE 估计
mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=0)  # ddof=0 是 MLE

print(f"真实参数: μ={true_mu}, σ={true_sigma}")
print(f"MLE估计: μ={mu_mle:.3f}, σ={sigma_mle:.3f}")
# 输出: μ≈5.0, σ≈2.0
```

---

## 2. MLE 在深度学习中的应用

### 2.1 交叉熵损失 = 负对数似然

**分类问题的概率模型**：
```
P(y = k | x, θ) = softmax(fθ(x))ₖ
```

**似然函数**（数据集 D = {(xᵢ, yᵢ)}）：
```
P(D | θ) = ∏ P(yᵢ | xᵢ, θ)
```

**负对数似然**：
```
-log L(θ) = -Σ log P(yᵢ | xᵢ, θ)
          = Σ CrossEntropy(yᵢ, fθ(xᵢ))
```

**结论**：最小化交叉熵 = 最大化似然！

**PyTorch 实现**：
```python
import torch
import torch.nn as nn

# 模型输出 logits
logits = model(x)  # shape: (batch, n_classes)

# 交叉熵损失 = 负对数似然
loss = nn.CrossEntropyLoss()(logits, labels)

# 等价于：
log_probs = nn.LogSoftmax(dim=1)(logits)
loss = nn.NLLLoss()(log_probs, labels)  # NLL = Negative Log Likelihood
```

### 2.2 回归问题：MSE = 高斯 MLE

**假设**：观测值 y = f(x) + ε，其中 ε ~ N(0, σ²)

**似然**：
```
P(y | x, θ) = N(y | fθ(x), σ²)
            = (1/√(2πσ²)) exp(-(y - fθ(x))²/(2σ²))
```

**负对数似然**：
```
-log L = (n/2)log(2πσ²) + (1/(2σ²)) Σ(yᵢ - fθ(xᵢ))²
```

**忽略常数项并最小化**：
```
argmin Σ(yᵢ - fθ(xᵢ))²  <- 这就是 MSE 损失！
  θ
```

**代码**：
```python
# MSE 损失 = 假设高斯噪声的 MLE
mse_loss = nn.MSELoss()(predictions, targets)
```

---

## 3. 贝叶斯推理

### 3.1 贝叶斯定理

```
P(θ | D) = P(D | θ) × P(θ) / P(D)

后验概率 = 似然 × 先验 / 证据
```

**各部分含义**：
- **P(θ | D)**：后验分布（看到数据后对参数的信念）
- **P(D | θ)**：似然（参数下观测数据的概率）
- **P(θ)**：先验分布（看数据前对参数的信念）
- **P(D)**：边缘似然（归一化常数）

### 3.2 MLE vs MAP vs 贝叶斯

| 方法 | 目标 | 公式 | 深度学习应用 |
|------|------|------|-------------|
| **MLE** | 点估计 | argmax P(D\|θ) | 标准训练 |
| **MAP** | 点估计 | argmax P(θ\|D) = argmax P(D\|θ)P(θ) | L2正则化 |
| **贝叶斯** | 分布估计 | 计算整个 P(θ\|D) | 贝叶斯神经网络 |

### 3.3 MAP = MLE + 正则化

**MAP 推导**：
```
θ_MAP = argmax P(θ | D)
      = argmax P(D | θ) × P(θ)
      = argmax [log P(D | θ) + log P(θ)]
```

**例子：L2 正则化**

假设高斯先验 `P(θ) = N(0, τ²)`：
```
log P(θ) = -θ²/(2τ²) + const

θ_MAP = argmax [Σ log P(yᵢ | xᵢ, θ) - θ²/(2τ²)]
      = argmin [Loss(θ) + λ||θ||²]  <- L2正则化！

其中 λ = 1/(2τ²)
```

**PyTorch 实现**：
```python
# L2 正则化 = 高斯先验的 MAP
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
#                                                          ↑
#                                                    等价于高斯先验
```

### 3.4 贝叶斯神经网络

**标准神经网络**：参数是确定的值
```
y = f(x; θ)  # θ 是点估计
```

**贝叶斯神经网络**：参数是分布
```
P(y | x, D) = ∫ P(y | x, θ) × P(θ | D) dθ
```

**好处**：
- 输出包含不确定性
- 避免过拟合
- 主动学习、安全关键应用

**简化实现：Dropout as Bayesian Approximation**
```python
model.train()  # 保持 dropout 开启
predictions = [model(x) for _ in range(100)]  # 多次采样
mean = torch.mean(torch.stack(predictions), dim=0)
uncertainty = torch.std(torch.stack(predictions), dim=0)
```

---

## 4. 例子：朴素贝叶斯分类器

### 4.1 问题设定

**任务**：给定特征 x = [x₁, x₂, ..., xd]，预测类别 y

**贝叶斯分类**：
```
y_pred = argmax P(y | x)
         y
       = argmax P(x | y) × P(y)  (贝叶斯定理)
         y
```

### 4.2 朴素假设

**假设特征条件独立**：
```
P(x | y) = P(x₁, x₂, ..., xd | y)
         ≈ ∏ P(xᵢ | y)  (朴素假设)
```

**完整公式**：
```
P(y | x) ∝ P(y) × ∏ P(xᵢ | y)
```

### 4.3 训练：估计概率

```python
from collections import defaultdict
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_prob = {}      # P(y)
        self.feature_prob = {}    # P(xᵢ | y)

    def fit(self, X, y):
        n_samples = len(y)
        classes = np.unique(y)

        # 估计 P(y)
        for c in classes:
            self.class_prob[c] = np.sum(y == c) / n_samples

        # 估计 P(xᵢ | y) - 假设高斯分布
        self.feature_prob = {}
        for c in classes:
            X_c = X[y == c]
            self.feature_prob[c] = {
                'mean': X_c.mean(axis=0),
                'std': X_c.std(axis=0) + 1e-6  # 避免除零
            }

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}

            for c, prior in self.class_prob.items():
                # log P(y) + Σ log P(xᵢ | y)
                likelihood = self._gaussian_log_likelihood(
                    x,
                    self.feature_prob[c]['mean'],
                    self.feature_prob[c]['std']
                )
                posteriors[c] = np.log(prior) + likelihood

            predictions.append(max(posteriors, key=posteriors.get))

        return np.array(predictions)

    def _gaussian_log_likelihood(self, x, mean, std):
        """计算高斯分布的对数似然"""
        return -0.5 * np.sum(np.log(2 * np.pi * std**2)) \
               -0.5 * np.sum(((x - mean) / std) ** 2)

# 测试
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

nb = NaiveBayes()
nb.fit(X_train, y_train)
accuracy = np.mean(nb.predict(X_test) == y_test)
print(f"准确率: {accuracy:.3f}")  # 通常 > 0.95
```

---

## 5. 核心公式总结

### MLE
```
θ_MLE = argmax Σ log P(xᵢ | θ)
         θ     i

等价于：最小化负对数似然
```

### MAP
```
θ_MAP = argmax [Σ log P(xᵢ | θ) + log P(θ)]
         θ      i
      = argmin [Loss(θ) + Regularization(θ)]
```

### 贝叶斯定理
```
P(θ | D) = P(D | θ) × P(θ) / P(D)

P(A | B) = P(B | A) × P(A) / P(B)
```

### 朴素贝叶斯
```
P(y | x₁, ..., xd) ∝ P(y) × ∏ P(xᵢ | y)
                              i
```

---

## 6. 与深度学习的连接

| 概念 | 深度学习应用 | 代码 |
|------|-------------|------|
| MLE | 交叉熵损失 | `nn.CrossEntropyLoss()` |
| MLE | MSE损失（高斯假设） | `nn.MSELoss()` |
| MAP | L2正则化 | `weight_decay=1e-4` |
| 贝叶斯 | Dropout不确定性 | `model.train()` + 多次预测 |
| 先验分布 | 权重初始化 | `nn.init.normal_(weight, 0, 0.01)` |
| 后验推理 | 变分自编码器(VAE) | ELBO损失 |

---

## 7. 常见误解

### ❌ 误解1："MLE 就是最小化损失"
- **正确理解**：MLE 是最大化似然，选择合适的概率模型后，等价于最小化特定损失

### ❌ 误解2："正则化只是防止过拟合的技巧"
- **正确理解**：L2正则化有深刻的概率解释（MAP with Gaussian prior）

### ❌ 误解3："贝叶斯方法太复杂，不实用"
- **正确理解**：Dropout、Batch Normalization 等技巧都有贝叶斯解释，且易于实现

---

## 8. 实践建议

1. **理解损失函数的概率意义**
   - 问自己：这个损失假设了什么样的数据分布？

2. **实现朴素贝叶斯分类器**
   - 加深对条件概率和贝叶斯推理的理解

3. **实验 Dropout 不确定性**
   ```python
   # 训练模式下多次预测
   model.train()
   preds = [model(x) for _ in range(100)]
   mean = torch.mean(torch.stack(preds), dim=0)
   std = torch.std(torch.stack(preds), dim=0)  # 不确定性！
   ```

4. **比较 MLE vs MAP**
   - 试试有无 weight_decay 的差异

---

## 9. 延伸阅读

### 理论
- 《Pattern Recognition and Machine Learning》- Bishop (第1-4章)
- 《Deep Learning》- Goodfellow (第5.5节)

### 实践
- [Bayesian Neural Networks](https://arxiv.org/abs/1505.05424)
- [Dropout as Bayesian Approximation](https://arxiv.org/abs/1506.02142)

### 可视化
- [Seeing Theory - Bayesian Inference](https://seeing-theory.brown.edu/bayesian-inference/)

---

## 快速自测

完成后你应该能够：

- [ ] 推导高斯分布的 MLE 估计
- [ ] 解释为什么交叉熵 = 负对数似然
- [ ] 说明 L2 正则化的贝叶斯解释
- [ ] 实现朴素贝叶斯分类器
- [ ] 使用 Dropout 估计预测不确定性

---

## 下一步

完成概率统计模块后，前往 [04_optimization](../../04_optimization/) 学习优化理论，理解梯度下降、Adam等算法原理。
