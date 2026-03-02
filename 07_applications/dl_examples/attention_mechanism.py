#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
注意力机制详解

本模块从零实现注意力机制的核心组件，深入理解 Query-Key-Value 的数学原理。

数学知识：
- 矩阵乘法（Q, K, V 计算）
- Softmax 归一化
- 缩放点积注意力
- 多头注意力

实现内容：
- scaled_dot_product_attention: 基础注意力
- MultiHeadAttention: 多头注意力
- 可视化：注意力权重热力图
- 示例：简单的序列到序列任务
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


# ============================================================================
# 基础注意力机制
# ============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    数值稳定的 Softmax 函数

    数学公式：
        softmax(x)_i = exp(x_i) / sum(exp(x_j))

    Args:
        x: 输入张量
        axis: 计算softmax的轴

    Returns:
        归一化后的概率分布
    """
    # 减去最大值防止溢出
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dropout_rate: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    缩放点积注意力

    数学公式：
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    其中：
        - Q: Query 矩阵 (batch, seq_len_q, d_k)
        - K: Key 矩阵 (batch, seq_len_k, d_k)
        - V: Value 矩阵 (batch, seq_len_v, d_v), 其中 seq_len_k = seq_len_v
        - d_k: Key 的维度

    缩放原因：
        当 d_k 很大时，点积结果会变大，导致 softmax 的梯度变小。
        除以 sqrt(d_k) 可以使点积结果保持在合理范围。

    Args:
        Q: Query 矩阵 (batch, seq_len_q, d_k)
        K: Key 矩阵 (batch, seq_len_k, d_k)
        V: Value 矩阵 (batch, seq_len_v, d_v)
        mask: 可选的掩码矩阵 (batch, seq_len_q, seq_len_k)
        dropout_rate: Dropout 概率

    Returns:
        output: 注意力输出 (batch, seq_len_q, d_v)
        attention_weights: 注意力权重 (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # 计算注意力分数: QK^T
    # (batch, seq_len_q, d_k) @ (batch, d_k, seq_len_k) -> (batch, seq_len_q, seq_len_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1))

    # 缩放: / sqrt(d_k)
    scores = scores / np.sqrt(d_k)

    # 应用掩码（用于解码器的自注意力）
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # 计算注意力权重: softmax
    attention_weights = softmax(scores, axis=-1)

    # 应用 dropout（仅在训练时）
    if dropout_rate > 0:
        dropout_mask = np.random.random(attention_weights.shape) > dropout_rate
        attention_weights = attention_weights * dropout_mask / (1 - dropout_rate)

    # 计算输出: weights * V
    # (batch, seq_len_q, seq_len_k) @ (batch, seq_len_v, d_v) -> (batch, seq_len_q, d_v)
    output = np.matmul(attention_weights, V)

    return output, attention_weights


# ============================================================================
# 多头注意力
# ============================================================================

class MultiHeadAttention:
    """
    多头注意力机制

    数学公式：
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
        where head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^V)

    多头的作用：
        - 每个头可以关注不同的语义信息
        - 类似于 CNN 的多通道卷积
        - 增加模型的表达能力

    参数：
        - W_Q, W_K, W_V: 投影矩阵 (d_model, d_k * num_heads)
        - W_O: 输出投影矩阵 (d_v * num_heads, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.0):
        """
        初始化多头注意力

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout_rate: Dropout 概率
        """
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        self.dropout_rate = dropout_rate

        # 初始化投影矩阵
        self.W_Q = np.random.randn(d_model, d_model).astype(np.float32) * np.sqrt(2.0 / d_model)
        self.W_K = np.random.randn(d_model, d_model).astype(np.float32) * np.sqrt(2.0 / d_model)
        self.W_V = np.random.randn(d_model, d_model).astype(np.float32) * np.sqrt(2.0 / d_model)
        self.W_O = np.random.randn(d_model, d_model).astype(np.float32) * np.sqrt(2.0 / d_model)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播

        Args:
            Q: Query (batch, seq_len, d_model)
            K: Key (batch, seq_len, d_model)
            V: Value (batch, seq_len, d_model)
            mask: 掩码矩阵

        Returns:
            output: 输出 (batch, seq_len, d_model)
            attention_weights: 注意力权重 (batch, num_heads, seq_len, seq_len)
        """
        batch_size = Q.shape[0]

        # 线性投影: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        Q_proj = np.matmul(Q, self.W_Q)
        K_proj = np.matmul(K, self.W_K)
        V_proj = np.matmul(V, self.W_V)

        # 重塑为多头: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q_heads = Q_proj.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K_heads = K_proj.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V_heads = V_proj.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # 计算注意力
        # (batch, num_heads, seq_len_q, d_k), (batch, num_heads, seq_len_k, d_k)
        # -> (batch, num_heads, seq_len_q, seq_len_k)
        scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        attention_weights = softmax(scores, axis=-1)

        # 应用注意力到 Value
        # (batch, num_heads, seq_len_q, seq_len_k) @ (batch, num_heads, seq_len_v, d_k)
        # -> (batch, num_heads, seq_len_q, d_k)
        context = np.matmul(attention_weights, V_heads)

        # 重塑回: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        # 输出投影
        output = np.matmul(context, self.W_O)

        return output, attention_weights


# ============================================================================
# 自注意力层
# ============================================================================

class SelfAttention:
    """
    自注意力层

    在自注意力中，Q、K、V 都来自同一个输入：
        Q = X * W_Q
        K = X * W_K
        V = X * W_V

    自注意力允许序列中的每个位置关注序列中的所有其他位置。
    """

    def __init__(self, d_model: int, num_heads: int = 8):
        """
        初始化自注意力层

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
        """
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播

        Args:
            X: 输入序列 (batch, seq_len, d_model)
            mask: 掩码矩阵

        Returns:
            output: 输出 (batch, seq_len, d_model)
            attention_weights: 注意力权重
        """
        return self.mha.forward(X, X, X, mask)


# ============================================================================
# 可视化
# ============================================================================

def visualize_attention_weights(
    attention_weights: np.ndarray,
    tokens: list = None,
    title: str = "注意力权重",
    save_path: str = "attention_weights.png"
) -> None:
    """
    可视化注意力权重热力图

    Args:
        attention_weights: 注意力权重 (seq_len_q, seq_len_k) 或 (num_heads, seq_len_q, seq_len_k)
        tokens: Token 列表
        title: 图标题
        save_path: 保存路径
    """
    if len(attention_weights.shape) == 3:
        # 多头注意力，只显示第一个头
        attention_weights = attention_weights[0]
        title += " (头 1)"

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(attention_weights, cmap='Blues')

    # 设置刻度标签
    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

    ax.set_xlabel('Key 位置')
    ax.set_ylabel('Query 位置')

    # 添加颜色条
    plt.colorbar(im, ax=ax, label='注意力权重')

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"注意力权重已保存到 {save_path}")


def visualize_multihead_attention(
    attention_weights: np.ndarray,
    tokens: list = None,
    title: str = "多头注意力权重",
    save_path: str = "multihead_attention.png"
) -> None:
    """
    可视化多头注意力权重

    Args:
        attention_weights: 注意力权重 (num_heads, seq_len_q, seq_len_k)
        tokens: Token 列表
        title: 图标题
        save_path: 保存路径
    """
    num_heads = attention_weights.shape[0]
    n_cols = min(4, num_heads)
    n_rows = (num_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if num_heads == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_heads):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        im = ax.imshow(attention_weights[i], cmap='Blues')

        if tokens is not None:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            if i >= num_heads - n_cols:
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_yticklabels(tokens, fontsize=8)
            else:
                ax.set_yticks([])

        ax.set_title(f'头 {i + 1}')

    # 隐藏多余的子图
    for i in range(num_heads, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"多头注意力权重已保存到 {save_path}")


# ============================================================================
# 示例和演示
# ============================================================================

def demo_basic_attention():
    """演示基础注意力"""
    print("\n" + "=" * 60)
    print("基础注意力演示")
    print("=" * 60)

    # 创建示例数据
    batch_size = 1
    seq_len = 4
    d_k = 8

    # 模拟 Query 和 Key-Value
    Q = np.random.randn(batch_size, seq_len, d_k).astype(np.float32)
    K = np.random.randn(batch_size, seq_len, d_k).astype(np.float32)
    V = np.random.randn(batch_size, seq_len, d_k).astype(np.float32)

    # 计算注意力
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")

    # 验证注意力权重和为1
    print(f"注意力权重和: {attention_weights[0].sum(axis=-1)}")

    return attention_weights[0]


def demo_multihead_attention():
    """演示多头注意力"""
    print("\n" + "=" * 60)
    print("多头注意力演示")
    print("=" * 60)

    # 参数
    d_model = 64
    num_heads = 8
    batch_size = 1
    seq_len = 10

    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, num_heads)

    # 创建输入
    X = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

    # 前向传播
    output, attention_weights = mha.forward(X, X, X)

    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")

    return attention_weights[0]


def demo_self_attention_sentence():
    """演示句子上的自注意力"""
    print("\n" + "=" * 60)
    print("句子自注意力演示")
    print("=" * 60)

    # 示例句子
    sentence = "The cat sat on the mat"
    tokens = sentence.split()

    # 参数
    d_model = 16
    num_heads = 4
    seq_len = len(tokens)

    # 创建简单的词嵌入（随机）
    np.random.seed(42)
    embeddings = np.random.randn(1, seq_len, d_model).astype(np.float32)

    # 创建自注意力层
    self_attn = SelfAttention(d_model, num_heads)

    # 前向传播
    output, attention_weights = self_attn.forward(embeddings)

    print(f"句子: '{sentence}'")
    print(f"Tokens: {tokens}")
    print(f"输出形状: {output.shape}")

    # 可视化
    visualize_multihead_attention(
        attention_weights[0],
        tokens,
        "句子自注意力权重",
        "attention_sentence.png"
    )

    return attention_weights[0]


def demo_positional_encoding():
    """演示位置编码"""
    print("\n" + "=" * 60)
    print("位置编码演示")
    print("=" * 60)

    def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
        """
        生成位置编码

        数学公式：
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            max_len: 最大序列长度
            d_model: 模型维度

        Returns:
            位置编码矩阵 (max_len, d_model)
        """
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    # 生成位置编码
    max_len = 50
    d_model = 64
    pe = positional_encoding(max_len, d_model)

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.imshow(pe.T, aspect='auto', cmap='RdBu')
    plt.xlabel('位置')
    plt.ylabel('维度')
    plt.title('位置编码')
    plt.colorbar(label='编码值')
    plt.tight_layout()
    plt.savefig('attention_positional_encoding.png', dpi=150)
    plt.close()
    print("位置编码已保存到 attention_positional_encoding.png")


def explain_attention_math():
    """解释注意力数学原理"""
    print("\n" + "=" * 60)
    print("注意力机制数学原理")
    print("=" * 60)

    print("""
1. 缩放点积注意力
   公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

   步骤分解：
   a) 计算相似度: scores = Q @ K^T  # 点积衡量相似度
   b) 缩放: scores = scores / sqrt(d_k)  # 防止梯度消失
   c) 归一化: weights = softmax(scores)  # 得到注意力权重
   d) 加权求和: output = weights @ V  # 聚合信息

2. 为什么要缩放？
   当 d_k 很大时，点积的方差也很大：
   Var(Q·K) = d_k
   这使得 softmax 的输入值很大，输出接近 one-hot，
   导致梯度接近 0。除以 sqrt(d_k) 可以稳定方差。

3. 多头注意力
   公式: MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W^O
   其中 head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^V)

   好处：
   - 每个头关注不同的语义信息
   - 类似 CNN 的多通道
   - 增加表达能力

4. 自注意力复杂度
   时间复杂度: O(n^2 * d)
   空间复杂度: O(n^2)

   其中 n 是序列长度，d 是维度。
   这限制了自注意力在长序列上的应用。
""")


# ============================================================================
# 测试
# ============================================================================

def test_softmax():
    """测试 Softmax"""
    print("测试 Softmax...")

    x = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float32)
    y = softmax(x)

    # 检查和为1
    assert np.allclose(y.sum(axis=-1), 1.0), "Softmax 和不为 1"

    # 检查所有值为正
    assert np.all(y > 0), "Softmax 存在负值"

    print("  ✓ Softmax 正确")


def test_scaled_dot_product_attention():
    """测试缩放点积注意力"""
    print("测试缩放点积注意力...")

    batch_size, seq_len, d_k = 2, 4, 8

    Q = np.random.randn(batch_size, seq_len, d_k).astype(np.float32)
    K = np.random.randn(batch_size, seq_len, d_k).astype(np.float32)
    V = np.random.randn(batch_size, seq_len, d_k).astype(np.float32)

    output, weights = scaled_dot_product_attention(Q, K, V)

    # 检查形状
    assert output.shape == (batch_size, seq_len, d_k), f"输出形状错误: {output.shape}"
    assert weights.shape == (batch_size, seq_len, seq_len), f"权重形状错误: {weights.shape}"

    # 检查权重和为1
    assert np.allclose(weights.sum(axis=-1), 1.0), "注意力权重和不为 1"

    print("  ✓ 缩放点积注意力正确")


def test_multihead_attention():
    """测试多头注意力"""
    print("测试多头注意力...")

    d_model = 64
    num_heads = 8
    batch_size = 2
    seq_len = 10

    mha = MultiHeadAttention(d_model, num_heads)

    X = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    output, weights = mha.forward(X, X, X)

    # 检查形状
    assert output.shape == (batch_size, seq_len, d_model), f"输出形状错误: {output.shape}"
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len), f"权重形状错误: {weights.shape}"

    print("  ✓ 多头注意力正确")


def test_self_attention():
    """测试自注意力"""
    print("测试自注意力...")

    d_model = 32
    num_heads = 4
    batch_size = 1
    seq_len = 8

    self_attn = SelfAttention(d_model, num_heads)

    X = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    output, weights = self_attn.forward(X)

    # 检查形状
    assert output.shape == X.shape, f"输出形状错误: {output.shape}"

    print("  ✓ 自注意力正确")


def test_masked_attention():
    """测试带掩码的注意力"""
    print("测试带掩码的注意力...")

    batch_size, seq_len, d_k = 1, 4, 8

    Q = np.random.randn(batch_size, seq_len, d_k).astype(np.float32)
    K = np.random.randn(batch_size, seq_len, d_k).astype(np.float32)
    V = np.random.randn(batch_size, seq_len, d_k).astype(np.float32)

    # 创建因果掩码（下三角）
    mask = np.tril(np.ones((seq_len, seq_len)))

    output, weights = scaled_dot_product_attention(Q, K, V, mask)

    # 检查掩码后的权重（上三角应该为0）
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert np.isclose(weights[0, i, j], 0.0, atol=1e-6), f"掩码位置 ({i},{j}) 应该为0"

    print("  ✓ 掩码注意力正确")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行注意力机制测试")
    print("=" * 60)

    test_softmax()
    test_scaled_dot_product_attention()
    test_multihead_attention()
    test_self_attention()
    test_masked_attention()

    print("\n所有测试通过!")


def main():
    """主函数"""
    print("=" * 60)
    print("注意力机制详解")
    print("=" * 60)

    # 解释数学原理
    explain_attention_math()

    # 演示基础注意力
    attention_weights = demo_basic_attention()
    visualize_attention_weights(
        attention_weights,
        title="基础注意力权重",
        save_path="attention_basic.png"
    )

    # 演示多头注意力
    multihead_weights = demo_multihead_attention()
    visualize_multihead_attention(
        multihead_weights,
        title="多头注意力权重",
        save_path="attention_multihead.png"
    )

    # 演示句子自注意力
    demo_self_attention_sentence()

    # 演示位置编码
    demo_positional_encoding()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    run_tests()

    # 运行主程序
    main()