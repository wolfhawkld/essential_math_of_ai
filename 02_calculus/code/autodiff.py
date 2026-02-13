"""
简单的自动微分实现

实现一个类似PyTorch的自动微分系统，用于理解反向传播原理。
"""

import numpy as np
from typing import List, Tuple, Optional, Set


class Tensor:
    """
    支持自动微分的张量类

    类似于torch.Tensor，但更简化
    """

    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        """
        Args:
            data: numpy数组或标量
            requires_grad: 是否需要计算梯度
            _children: 子节点（用于构建计算图）
            _op: 产生该tensor的操作名称
        """
        if isinstance(data, (int, float)):
            self.data = np.array([data], dtype=np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad = requires_grad
        self.grad = None

        # 用于反向传播的内部变量
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        """清零梯度"""
        self.grad = None

    # ==================== 基本运算 ====================

    def __add__(self, other):
        """加法"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other),
                     _op='+')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                # 处理广播
                other.grad += self._reduce_grad(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        """乘法（逐元素）"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other),
                     _op='*')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += self._reduce_grad(other.data * out.grad, self.shape)

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self._reduce_grad(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, power):
        """幂运算"""
        assert isinstance(power, (int, float)), "power must be int or float"
        out = Tensor(self.data ** power,
                     requires_grad=self.requires_grad,
                     _children=(self,),
                     _op=f'**{power}')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        """负号"""
        return self * -1

    def __sub__(self, other):
        """减法"""
        return self + (-other)

    def __truediv__(self, other):
        """除法"""
        return self * (other ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return other * (self ** -1)

    # ==================== 矩阵运算 ====================

    def matmul(self, other):
        """矩阵乘法"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other),
                     _op='@')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad @ other.data.T

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        """求和"""
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,),
                     _op='sum')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                # 广播梯度
                if axis is None:
                    self.grad += np.ones_like(self.data) * out.grad
                else:
                    grad_expanded = np.expand_dims(out.grad, axis)
                    self.grad += np.broadcast_to(grad_expanded, self.data.shape)

        out._backward = _backward
        return out

    def mean(self):
        """均值"""
        return self.sum() / self.data.size

    # ==================== 激活函数 ====================

    def relu(self):
        """ReLU激活"""
        out = Tensor(np.maximum(0, self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,),
                     _op='relu')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """Sigmoid激活"""
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig,
                     requires_grad=self.requires_grad,
                     _children=(self,),
                     _op='sigmoid')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += sig * (1 - sig) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """Tanh激活"""
        t = np.tanh(self.data)
        out = Tensor(t,
                     requires_grad=self.requires_grad,
                     _children=(self,),
                     _op='tanh')

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    # ==================== 反向传播 ====================

    def backward(self):
        """
        反向传播计算梯度

        从当前节点开始，按拓扑逆序计算所有需要梯度的节点
        """
        # 初始化当前节点的梯度为1（链式法则的起点）
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 拓扑排序
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 反向传播
        for node in reversed(topo):
            node._backward()

    # ==================== 辅助方法 ====================

    @staticmethod
    def _reduce_grad(grad, target_shape):
        """
        处理广播导致的梯度形状不匹配

        例如: (3, 1) + (3, 4) → (3, 4)
        反向时需要将 (3, 4) 的梯度reduce到 (3, 1)
        """
        # 如果形状已经匹配，直接返回
        if grad.shape == target_shape:
            return grad

        # 处理标量情况
        if target_shape == () or target_shape == (1,):
            return np.sum(grad)

        # 一般情况：沿着被广播的维度求和
        ndims_added = len(grad.shape) - len(target_shape)
        for i in range(ndims_added):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad


# ==================== 神经网络层 ====================

class Linear:
    """全连接层"""

    def __init__(self, in_features, out_features):
        # He初始化
        self.W = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
                        requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class Sequential:
    """顺序容器"""

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


class ReLU:
    """ReLU激活函数"""

    def __call__(self, x):
        return x.relu()


# ==================== 优化器 ====================

class SGD:
    """随机梯度下降优化器"""

    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """执行一步优化"""
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        """清零所有参数的梯度"""
        for param in self.parameters:
            param.zero_grad()


# ==================== 损失函数 ====================

def mse_loss(y_pred, y_true):
    """均方误差损失"""
    return ((y_pred - y_true) ** 2).mean()


# ==================== 示例和测试 ====================

def test_basic_operations():
    """测试基本运算"""
    print("=" * 50)
    print("测试基本运算")
    print("=" * 50)

    # 创建张量
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)

    # 计算
    z = x * y + x**2
    print(f"z = x*y + x^2 = {z.data[0]}")  # 2*3 + 4 = 10

    # 反向传播
    z.backward()

    print(f"dz/dx = {x.grad[0]}")  # y + 2x = 3 + 4 = 7
    print(f"dz/dy = {y.grad[0]}")  # x = 2

    assert abs(x.grad[0] - 7.0) < 1e-6
    assert abs(y.grad[0] - 2.0) < 1e-6
    print("✓ 基本运算测试通过\n")


def test_neural_network():
    """测试神经网络"""
    print("=" * 50)
    print("测试两层神经网络")
    print("=" * 50)

    # 构建网络
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )

    # 数据
    X = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    y = Tensor([[1.0], [0.0]], requires_grad=False)

    # 优化器
    optimizer = SGD(model.parameters(), lr=0.01)

    # 训练几步
    for epoch in range(100):
        # 前向
        y_pred = model(X)

        # 损失
        loss = mse_loss(y_pred, y)

        # 反向
        optimizer.zero_grad()
        loss.backward()

        # 更新
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data[0]:.6f}")

    print("✓ 神经网络测试通过\n")


def test_gradient_check():
    """梯度检查：对比数值梯度和自动微分梯度"""
    print("=" * 50)
    print("梯度检查")
    print("=" * 50)

    def numerical_gradient(f, x, h=1e-5):
        """数值梯度"""
        grad = np.zeros_like(x.data)
        for i in range(x.data.size):
            old_value = x.data.flat[i]

            x.data.flat[i] = old_value + h
            fxh = f(x).data

            x.data.flat[i] = old_value - h
            fxmh = f(x).data

            grad.flat[i] = (fxh - fxmh) / (2 * h)

            x.data.flat[i] = old_value

        return grad

    # 测试函数: f(x) = x^2 + 2x + 1
    x = Tensor([3.0], requires_grad=True)

    def f(x):
        return x**2 + 2*x + 1

    # 自动微分
    y = f(x)
    y.backward()
    grad_auto = x.grad[0]

    # 数值微分
    x.zero_grad()
    grad_num = numerical_gradient(f, x)[0]

    print(f"自动微分梯度: {grad_auto}")
    print(f"数值梯度: {grad_num}")
    print(f"差异: {abs(grad_auto - grad_num)}")

    assert abs(grad_auto - grad_num) < 1e-6
    print("✓ 梯度检查通过\n")


def example_simple_network():
    """完整示例：训练一个简单的网络"""
    print("=" * 50)
    print("完整示例：XOR问题")
    print("=" * 50)

    # XOR数据
    X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=False)
    y = Tensor([[0], [1], [1], [0]], requires_grad=False)

    # 网络（需要非线性才能解决XOR）
    model = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 1)
    )

    optimizer = SGD(model.parameters(), lr=0.1)

    # 训练
    for epoch in range(1000):
        y_pred = model(X)
        loss = mse_loss(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data[0]:.6f}")

    # 测试
    print("\n预测结果:")
    y_pred = model(X)
    for i, (input_val, pred, target) in enumerate(zip(X.data, y_pred.data, y.data)):
        print(f"Input: {input_val}, Pred: {pred[0]:.4f}, Target: {target[0]}")

    print("\n✓ XOR问题训练完成")


if __name__ == "__main__":
    # 运行所有测试
    test_basic_operations()
    test_neural_network()
    test_gradient_check()
    example_simple_network()

    print("\n" + "=" * 50)
    print("所有测试通过！")
    print("=" * 50)
