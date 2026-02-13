# Essential Math of AI

面向深度学习和强化学习的数学知识速成手册

## 项目简介

这个项目采用**问题驱动 + 最小必要知识**的方式，帮助你快速掌握深度学习和强化学习所需的数学基础。

### 核心理念

- **从应用反推数学需求** - 不系统学习整个数学体系，只学最必要的
- **代码优先** - 用代码和可视化理解抽象概念
- **实战导向** - 每个知识点都对应实际应用场景

## 学习路径

### 第一阶段：基础数学（深度学习必备）

```
01_linear_algebra → 02_calculus → 03_probability_statistics
```

这三个模块是深度学习的数学基石，建议按顺序学习。

### 第二阶段：深度学习进阶

```
04_optimization → 05_information_theory → 07_applications/dl_examples
```

优化理论和信息论帮助你理解训练过程，最后通过实战案例综合应用。

### 第三阶段：强化学习

```
06_rl_math → 07_applications/rl_examples
```

强化学习需要额外的数学工具（MDP、贝尔曼方程等）。

## 目录结构

### 📐 [01_linear_algebra](./01_linear_algebra/) - 线性代数
神经网络的数据表示和计算基础
- 向量和矩阵
- 矩阵运算（乘法、转置、逆）
- 特征值和SVD

### 📊 [02_calculus](./02_calculus/) - 微积分
反向传播和梯度下降的核心
- 导数和偏导数
- 链式法则
- 梯度下降

### 🎲 [03_probability_statistics](./03_probability_statistics/) - 概率统计
不确定性建模和推理
- 概率基础
- 常见分布
- 贝叶斯推理和最大似然估计

### 🎯 [04_optimization](./04_optimization/) - 优化理论
神经网络训练算法
- 凸优化基础
- SGD、Adam等优化器
- 约束优化

### 📡 [05_information_theory](./05_information_theory/) - 信息论
损失函数的数学原理
- 熵
- 交叉熵和KL散度
- 互信息

### 🤖 [06_rl_math](./06_rl_math/) - 强化学习数学
RL特有的数学工具
- 马尔可夫决策过程
- 贝尔曼方程
- 值迭代和策略迭代

### 🚀 [07_applications](./07_applications/) - 综合应用
理论到实践
- 深度学习案例：从零实现神经网络、CNN原理
- 强化学习案例：Q-Learning、策略梯度

### 📝 [cheatsheets](./cheatsheets/) - 速查表
快速回顾关键公式和概念

### 📚 [resources](./resources/) - 学习资源
推荐书籍、论文和工具

## 快速开始

### 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动Jupyter

```bash
jupyter notebook
```

### 学习建议

1. **先看README** - 每个模块的README都会告诉你为什么需要这个知识
2. **读docs理解概念** - Markdown文档讲解核心原理
3. **运行notebooks实践** - 交互式代码加深理解
4. **参考code实现** - 查看纯净的实用代码

## 使用方式

### 作为学习材料
按照学习路径顺序学习，每个模块包含理论讲解和代码实践。

### 作为参考手册
遇到不理解的数学概念时，查找对应模块快速回顾。

### 作为代码库
直接import `code/`目录下的工具函数到你的项目中。

## 贡献指南

欢迎提交Issue和Pull Request：
- 发现错误或不清晰的地方
- 补充更好的可视化示例
- 添加新的应用案例

## 许可证

MIT License - 详见 [LICENSE](./LICENSE)

## 致谢

本项目灵感来源于深度学习和强化学习实践中的数学痛点，目标是让数学学习更高效、更有针对性。

---

**开始学习**: 从 [01_linear_algebra](./01_linear_algebra/) 开始你的数学之旅吧！
