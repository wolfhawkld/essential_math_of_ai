# 01 线性代数 (Linear Algebra)

## 为什么需要线性代数？

在深度学习中，线性代数无处不在：

### 1. 数据表示
- **图像**：一张28×28的图像 = 784维向量
- **文本**：词嵌入将单词映射为向量（如300维）
- **批量数据**：N个样本的数据 = N×D的矩阵

### 2. 神经网络计算
```python
# 全连接层的本质就是矩阵乘法
output = input @ weights + bias
# input: [batch_size, input_dim]
# weights: [input_dim, output_dim]
# output: [batch_size, output_dim]
```

### 3. 数据变换
- **PCA降维**：使用特征值分解
- **数据增强**：旋转、缩放都是矩阵运算
- **注意力机制**：Query、Key、Value矩阵

## 最小必要知识

你**不需要**学习线性代数的所有内容，只需要掌握：

### ✅ 必须掌握
1. 向量和矩阵的基本操作（加减、数乘、乘法）
2. 矩阵转置和逆矩阵
3. 矩阵乘法的几何意义（线性变换）
4. 特征值和特征向量（PCA用到）

### ⚠️ 了解即可
5. 奇异值分解SVD（降维、推荐系统）
6. 范数（L1/L2正则化）

### ❌ 可以跳过
- 行列式的详细计算
- 线性空间的抽象定义
- 复杂的矩阵分解（QR、Cholesky等）

## 学习路径

```
第1步：向量和矩阵基础 (30分钟)
├── docs/01_vector_matrix.md
└── notebooks/vector_operations.ipynb

第2步：矩阵运算 (45分钟)
├── docs/02_matrix_operations.md
└── notebooks/matrix_for_nn.ipynb

第3步：特征值和SVD (60分钟)
├── docs/03_eigenvalue_svd.md
└── notebooks/pca_from_scratch.ipynb  # 待创建
```

## 目录内容

### 📄 docs/ - 理论文档
- `01_vector_matrix.md` - 向量和矩阵的基本概念
- `02_matrix_operations.md` - 矩阵运算及其在神经网络中的应用
- `03_eigenvalue_svd.md` - 特征值分解和SVD

### 💻 notebooks/ - 交互式实践
- `vector_operations.ipynb` - 向量运算可视化
- `matrix_for_nn.ipynb` - 用矩阵实现简单神经网络
- `pca_from_scratch.ipynb` - 从零实现PCA

### 🔧 code/ - 实用代码
- `matrix_utils.py` - 常用矩阵操作工具函数

## 快速测试

完成本模块后，你应该能够：

- [ ] 用NumPy实现矩阵乘法和转置
- [ ] 理解神经网络前向传播中的矩阵维度变化
- [ ] 用特征值分解实现简单的PCA降维
- [ ] 解释为什么深度学习中大量使用矩阵运算

## 与深度学习的连接

| 线性代数概念 | 深度学习应用 | 位置 |
|------------|------------|------|
| 矩阵乘法 | 全连接层、卷积 | 每一层 |
| 转置 | 反向传播 | 梯度计算 |
| 特征值分解 | PCA降维 | 数据预处理 |
| SVD | 推荐系统、矩阵分解 | 协同过滤 |
| 范数 | 正则化 | 损失函数 |

## 推荐资源

### 视频
- [3Blue1Brown - 线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra) 系列（强烈推荐，直观理解）

### 书籍
- 《Deep Learning》第2章（Ian Goodfellow）
- 《线性代数及其应用》（Gilbert Strang）- MIT公开课配套

### 在线工具
- [Matrix Multiplication Visualizer](http://matrixmultiplication.xyz/) - 矩阵乘法可视化

## 下一步

完成线性代数后，前往 [02_calculus](../02_calculus/) 学习微积分，理解反向传播的数学原理。
