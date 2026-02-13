# -*- coding: utf-8 -*-
"""
优化器实现模块

包含深度学习中常用的优化器和学习率调度器。

优化器：
- SGD（随机梯度下降）
- Momentum（动量）
- Nesterov（Nesterov加速梯度）
- AdaGrad（自适应梯度）
- RMSprop（均方根传播）
- Adam（自适应矩估计）
- AdamW（解耦权重衰减的Adam）

学习率调度器：
- StepLR（阶梯衰减）
- ExponentialLR（指数衰减）
- CosineAnnealingLR（余弦退火）
- WarmupLR（预热）

工具：
- 梯度裁剪
- 权重衰减

作者：Essential Math of AI
"""

import numpy as np
from typing import Callable, List, Dict, Optional, Tuple, Union


# ============================================================================
# 基础优化器类
# ============================================================================

class BaseOptimizer:
    """所有优化器的基类"""

    def __init__(self, lr: float = 0.01):
        """
        参数：
            lr: 学习率
        """
        self.lr = lr
        self.t = 0  # 时间步

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        执行一步参数更新

        参数：
            params: 当前参数
            grads: 参数的梯度

        返回：
            更新后的参数
        """
        raise NotImplementedError

    def reset(self):
        """重置优化器状态"""
        self.t = 0


# ============================================================================
# SGD系列优化器
# ============================================================================

class SGD(BaseOptimizer):
    """
    随机梯度下降（Stochastic Gradient Descent）

    最基础的优化器，直接沿梯度反方向更新参数。

    更新规则：
        θ_t = θ_{t-1} - lr × ∇L(θ)

    参数：
        lr: 学习率
    """

    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        SGD更新步骤

        参数：
            params: 参数 θ
            grads: 梯度 ∇L

        返回：
            更新后的参数
        """
        self.t += 1
        return params - self.lr * grads


class Momentum(BaseOptimizer):
    """
    动量优化器（Momentum / SGD with Momentum）

    通过累积历史梯度来加速收敛并减少震荡。

    更新规则：
        v_t = β × v_{t-1} + ∇L
        θ_t = θ_{t-1} - lr × v_t

    参数：
        lr: 学习率
        momentum: 动量系数 β（默认0.9）
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Momentum更新步骤
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.t += 1

        # 更新速度
        self.velocity = self.momentum * self.velocity + grads

        # 更新参数
        return params - self.lr * self.velocity

    def reset(self):
        """重置状态"""
        super().reset()
        self.velocity = None


class Nesterov(Momentum):
    """
    Nesterov加速梯度（Nesterov Accelerated Gradient）

    Momentum的改进版本，先"展望"再计算梯度。

    更新规则：
        v_t = β × v_{t-1} + ∇L(θ - β × v_{t-1})
        θ_t = θ_{t-1} - lr × v_t

    参数：
        lr: 学习率
        momentum: 动量系数（默认0.9）
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(lr, momentum)
        self.lookahead_params = None

    def compute_lookahead(self, params: np.ndarray) -> np.ndarray:
        """
        计算展望位置

        参数：
            params: 当前参数

        返回：
            展望后的参数位置
        """
        if self.velocity is None:
            return params
        return params - self.momentum * self.velocity

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Nesterov更新步骤

        注意：传入的grads应该是在lookahead位置计算的梯度
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.t += 1

        # 更新速度
        self.velocity = self.momentum * self.velocity + grads

        # 更新参数
        return params - self.lr * self.velocity

    def reset(self):
        """重置状态"""
        super().reset()
        self.lookahead_params = None


# ============================================================================
# 自适应学习率优化器
# ============================================================================

class AdaGrad(BaseOptimizer):
    """
    AdaGrad优化器（Adaptive Gradient）

    为不同参数自适应调整学习率，频繁更新的参数学习率更小。

    更新规则：
        G_t = G_{t-1} + (∇L)²
        θ_t = θ_{t-1} - lr / (√G_t + ε) × ∇L

    问题：G单调递增，学习率会持续减小，最终可能停止学习。

    参数：
        lr: 学习率
        epsilon: 数值稳定性参数（默认1e-8）
    """

    def __init__(self, lr: float = 0.01, epsilon: float = 1e-8):
        super().__init__(lr)
        self.epsilon = epsilon
        self.G = None  # 累积梯度平方和

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        AdaGrad更新步骤
        """
        if self.G is None:
            self.G = np.zeros_like(params)

        self.t += 1

        # 累积梯度平方
        self.G += grads ** 2

        # 自适应学习率更新
        adjusted_lr = self.lr / (np.sqrt(self.G) + self.epsilon)

        return params - adjusted_lr * grads

    def reset(self):
        """重置状态"""
        super().reset()
        self.G = None


class RMSprop(BaseOptimizer):
    """
    RMSprop优化器（Root Mean Square Propagation）

    AdaGrad的改进版本，用指数移动平均代替累积，避免学习率持续衰减。

    更新规则：
        G_t = β × G_{t-1} + (1-β) × (∇L)²
        θ_t = θ_{t-1} - lr / (√G_t + ε) × ∇L

    参数：
        lr: 学习率
        decay: 衰减率（默认0.9）
        epsilon: 数值稳定性参数（默认1e-8）
    """

    def __init__(self, lr: float = 0.01, decay: float = 0.9, epsilon: float = 1e-8):
        super().__init__(lr)
        self.decay = decay
        self.epsilon = epsilon
        self.G = None  # 梯度平方的指数移动平均

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        RMSprop更新步骤
        """
        if self.G is None:
            self.G = np.zeros_like(params)

        self.t += 1

        # 梯度平方的指数移动平均
        self.G = self.decay * self.G + (1 - self.decay) * (grads ** 2)

        # 自适应学习率更新
        adjusted_lr = self.lr / (np.sqrt(self.G) + self.epsilon)

        return params - adjusted_lr * grads

    def reset(self):
        """重置状态"""
        super().reset()
        self.G = None


class Adam(BaseOptimizer):
    """
    Adam优化器（Adaptive Moment Estimation）

    结合Momentum和RMSprop的优点，是实践中最常用的优化器。

    更新规则：
        m_t = β₁ × m_{t-1} + (1-β₁) × ∇L          (一阶矩估计)
        v_t = β₂ × v_{t-1} + (1-β₂) × (∇L)²       (二阶矩估计)
        m̂_t = m_t / (1 - β₁^t)                   (偏差修正)
        v̂_t = v_t / (1 - β₂^t)
        θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε)

    参数：
        lr: 学习率（默认0.001）
        beta1: 一阶矩衰减率（默认0.9）
        beta2: 二阶矩衰减率（默认0.999）
        epsilon: 数值稳定性参数（默认1e-8）
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None  # 一阶矩（均值）
        self.v = None  # 二阶矩（未中心化的方差）

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Adam更新步骤
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # 更新一阶矩和二阶矩
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 更新参数
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self):
        """重置状态"""
        super().reset()
        self.m = None
        self.v = None


class AdamW(Adam):
    """
    AdamW优化器（Adam with Decoupled Weight Decay）

    将权重衰减从梯度更新中解耦，比标准Adam的正则化效果更好。

    更新规则：
        （Adam更新）+ weight_decay × θ

    参数：
        lr: 学习率
        beta1: 一阶矩衰减率
        beta2: 二阶矩衰减率
        epsilon: 数值稳定性参数
        weight_decay: 权重衰减系数（默认0.01）
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(lr, beta1, beta2, epsilon)
        self.weight_decay = weight_decay

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        AdamW更新步骤
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # 更新一阶矩和二阶矩
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Adam更新 + 解耦的权重衰减
        return params - self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) +
                                   self.weight_decay * params)


# ============================================================================
# 学习率调度器
# ============================================================================

class LRScheduler:
    """学习率调度器基类"""

    def __init__(self, initial_lr: float):
        """
        参数：
            initial_lr: 初始学习率
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

    def step(self, epoch: int) -> float:
        """
        根据epoch计算当前学习率

        参数：
            epoch: 当前epoch

        返回：
            当前学习率
        """
        raise NotImplementedError

    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.current_lr


class StepLR(LRScheduler):
    """
    阶梯衰减学习率

    每step_size个epoch，学习率乘以gamma。

    lr_t = lr_0 × gamma^(epoch // step_size)

    参数：
        initial_lr: 初始学习率
        step_size: 衰减周期
        gamma: 衰减率
    """

    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, epoch: int) -> float:
        """计算阶梯衰减学习率"""
        self.current_lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
        return self.current_lr


class ExponentialLR(LRScheduler):
    """
    指数衰减学习率

    lr_t = lr_0 × gamma^epoch

    参数：
        initial_lr: 初始学习率
        gamma: 衰减率（每epoch）
    """

    def __init__(self, initial_lr: float, gamma: float = 0.95):
        super().__init__(initial_lr)
        self.gamma = gamma

    def step(self, epoch: int) -> float:
        """计算指数衰减学习率"""
        self.current_lr = self.initial_lr * (self.gamma ** epoch)
        return self.current_lr


class CosineAnnealingLR(LRScheduler):
    """
    余弦退火学习率

    lr_t = min_lr + 0.5 × (lr_0 - min_lr) × (1 + cos(π × epoch / T_max))

    参数：
        initial_lr: 初始学习率
        T_max: 最大epoch数
        min_lr: 最小学习率
    """

    def __init__(self, initial_lr: float, T_max: int, min_lr: float = 0):
        super().__init__(initial_lr)
        self.T_max = T_max
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        """计算余弦退火学习率"""
        self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                         (1 + np.cos(np.pi * epoch / self.T_max))
        return self.current_lr


class WarmupLR(LRScheduler):
    """
    预热学习率

    前warmup_epochs个epoch线性增加学习率，之后保持不变。

    前warmup阶段：lr_t = lr_0 × t / warmup_epochs
    之后：lr_t = lr_0

    参数：
        initial_lr: 目标学习率
        warmup_epochs: 预热epoch数
    """

    def __init__(self, initial_lr: float, warmup_epochs: int):
        super().__init__(initial_lr)
        self.warmup_epochs = warmup_epochs

    def step(self, epoch: int) -> float:
        """计算预热学习率"""
        if epoch < self.warmup_epochs:
            self.current_lr = self.initial_lr * epoch / self.warmup_epochs
        else:
            self.current_lr = self.initial_lr
        return self.current_lr


class WarmupCosineLR(LRScheduler):
    """
    预热+余弦退火

    先线性预热，然后余弦衰减。

    参数：
        initial_lr: 初始学习率
        warmup_epochs: 预热epoch数
        total_epochs: 总epoch数
        min_lr: 最小学习率
    """

    def __init__(self, initial_lr: float, warmup_epochs: int,
                 total_epochs: int, min_lr: float = 0):
        super().__init__(initial_lr)
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        """计算预热+余弦学习率"""
        if epoch < self.warmup_epochs:
            # 预热阶段
            self.current_lr = self.initial_lr * epoch / self.warmup_epochs
        else:
            # 余弦衰减阶段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                             (1 + np.cos(np.pi * progress))
        return self.current_lr


# ============================================================================
# 工具函数
# ============================================================================

def clip_gradient_by_norm(grads: np.ndarray, max_norm: float) -> np.ndarray:
    """
    按范数裁剪梯度

    如果 ||grads|| > max_norm，则：
        grads = grads × (max_norm / ||grads||)

    参数：
        grads: 梯度
        max_norm: 最大范数

    返回：
        裁剪后的梯度
    """
    grad_norm = np.linalg.norm(grads)

    if grad_norm > max_norm:
        grads = grads * (max_norm / grad_norm)

    return grads


def clip_gradient_by_value(grads: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    按值裁剪梯度

    将梯度限制在[min_val, max_val]范围内。

    参数：
        grads: 梯度
        min_val: 最小值
        max_val: 最大值

    返回：
        裁剪后的梯度
    """
    return np.clip(grads, min_val, max_val)


def add_weight_decay(grads: np.ndarray, params: np.ndarray,
                     weight_decay: float) -> np.ndarray:
    """
    添加权重衰减（L2正则化梯度）

    grads_with_wd = grads + weight_decay × params

    参数：
        grads: 原始梯度
        params: 参数
        weight_decay: 权重衰减系数

    返回：
        添加权重衰减后的梯度
    """
    return grads + weight_decay * params


# ============================================================================
# 训练循环
# ============================================================================

def train_with_optimizer(
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    optimizer: BaseOptimizer,
    loss_fn: Callable,
    grad_fn: Callable,
    epochs: int = 100,
    batch_size: int = 32,
    lr_scheduler: Optional[LRScheduler] = None,
    max_grad_norm: Optional[float] = None,
    weight_decay: float = 0.0,
    verbose: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """
    完整的训练循环

    参数：
        X: 训练数据 (n_samples, n_features)
        y: 标签 (n_samples,)
        params: 初始参数
        optimizer: 优化器实例
        loss_fn: 损失函数 loss_fn(params, X, y)
        grad_fn: 梯度函数 grad_fn(params, X, y)
        epochs: 训练epoch数
        batch_size: batch大小
        lr_scheduler: 学习率调度器（可选）
        max_grad_norm: 梯度裁剪阈值（可选）
        weight_decay: 权重衰减系数
        verbose: 是否打印训练信息

    返回：
        params: 训练后的参数
        loss_history: 每个epoch的平均损失
    """
    n_samples = len(y)
    loss_history = []

    for epoch in range(epochs):
        # 更新学习率
        if lr_scheduler is not None:
            optimizer.lr = lr_scheduler.step(epoch)

        # 打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Mini-batch训练
        total_loss = 0.0
        n_batches = n_samples // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # 计算损失和梯度
            loss = loss_fn(params, X_batch, y_batch)
            grads = grad_fn(params, X_batch, y_batch)

            # 权重衰减
            if weight_decay > 0:
                grads = add_weight_decay(grads, params, weight_decay)

            # 梯度裁剪
            if max_grad_norm is not None:
                grads = clip_gradient_by_norm(grads, max_grad_norm)

            # 更新参数
            params = optimizer.step(params, grads)

            total_loss += loss

        # 记录平均损失
        avg_loss = total_loss / n_batches
        loss_history.append(avg_loss)

        if verbose and epoch % 10 == 0:
            lr_str = f", lr={optimizer.lr:.6f}" if lr_scheduler else ""
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}{lr_str}")

    return params, loss_history


# ============================================================================
# 测试函数
# ============================================================================

def test_sgd():
    """测试SGD优化器"""
    print("="*60)
    print("测试 SGD")
    print("="*60)

    # 简单的二次函数：minimize x²
    params = np.array([5.0])
    optimizer = SGD(lr=0.1)

    for i in range(20):
        grads = 2 * params  # f(x) = x² 的梯度
        params = optimizer.step(params, grads)
        if i % 5 == 0:
            print(f"Step {i}: x = {params[0]:.6f}")

    print(f"最终结果: x = {params[0]:.6f} (理论值: 0.0)")
    assert abs(params[0]) < 0.01, "SGD测试失败"
    print("✓ SGD测试通过\n")


def test_momentum():
    """测试Momentum优化器"""
    print("="*60)
    print("测试 Momentum")
    print("="*60)

    # 测试在峡谷中的收敛速度
    # f(x, y) = x² + 100y² （y方向曲率大）
    params = np.array([10.0, 10.0])
    optimizer = Momentum(lr=0.001, momentum=0.9)

    for i in range(100):
        grads = np.array([2 * params[0], 200 * params[1]])
        params = optimizer.step(params, grads)

        if i % 20 == 0:
            loss = params[0]**2 + 100*params[1]**2
            print(f"Step {i}: loss = {loss:.6f}, x = {params}")

    print(f"最终结果: {params}")
    assert np.linalg.norm(params) < 0.1, "Momentum测试失败"
    print("✓ Momentum测试通过\n")


def test_adam():
    """测试Adam优化器"""
    print("="*60)
    print("测试 Adam")
    print("="*60)

    # Rosenbrock函数（经典的优化测试函数）
    # f(x, y) = (a - x)² + b(y - x²)²
    a, b = 1.0, 100.0

    def rosenbrock(params):
        x, y = params
        return (a - x)**2 + b * (y - x**2)**2

    def rosenbrock_grad(params):
        x, y = params
        dx = -2*(a - x) - 4*b*x*(y - x**2)
        dy = 2*b*(y - x**2)
        return np.array([dx, dy])

    params = np.array([-1.0, 1.0])  # 起点
    optimizer = Adam(lr=0.01)

    for i in range(2000):
        grads = rosenbrock_grad(params)
        params = optimizer.step(params, grads)

        if i % 500 == 0:
            loss = rosenbrock(params)
            print(f"Step {i}: loss = {loss:.6f}, params = {params}")

    final_loss = rosenbrock(params)
    print(f"最终结果: params = {params}, loss = {final_loss:.6f}")
    print(f"理论最优: [1.0, 1.0], loss = 0.0")
    assert final_loss < 0.01, "Adam测试失败"
    print("✓ Adam测试通过\n")


def test_lr_schedulers():
    """测试学习率调度器"""
    print("="*60)
    print("测试学习率调度器")
    print("="*60)

    initial_lr = 0.1
    total_epochs = 100

    schedulers = {
        'StepLR': StepLR(initial_lr, step_size=30, gamma=0.1),
        'ExponentialLR': ExponentialLR(initial_lr, gamma=0.95),
        'CosineAnnealingLR': CosineAnnealingLR(initial_lr, T_max=total_epochs),
        'WarmupCosineLR': WarmupCosineLR(initial_lr, warmup_epochs=10, total_epochs=total_epochs)
    }

    for name, scheduler in schedulers.items():
        lrs = [scheduler.step(epoch) for epoch in range(total_epochs)]
        print(f"{name:20s}: 初始={lrs[0]:.6f}, 最终={lrs[-1]:.6f}")
        assert lrs[0] > 0, f"{name}测试失败"

    print("✓ 学习率调度器测试通过\n")


def test_gradient_clipping():
    """测试梯度裁剪"""
    print("="*60)
    print("测试梯度裁剪")
    print("="*60)

    # 测试范数裁剪
    grads = np.array([3.0, 4.0])
    max_norm = 2.0

    original_norm = np.linalg.norm(grads)
    clipped_grads = clip_gradient_by_norm(grads, max_norm)
    clipped_norm = np.linalg.norm(clipped_grads)

    print(f"原始梯度: {grads}, 范数 = {original_norm:.2f}")
    print(f"裁剪后: {clipped_grads}, 范数 = {clipped_norm:.2f}")

    assert clipped_norm <= max_norm + 1e-6, "梯度裁剪测试失败"
    print("✓ 梯度裁剪测试通过\n")


def test_linear_regression():
    """测试完整训练循环（线性回归）"""
    print("="*60)
    print("测试完整训练循环：线性回归")
    print("="*60)

    np.random.seed(42)

    # 生成数据
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    true_params = np.array([1.5, -2.0, 0.5, 1.0, -0.5])
    y = X @ true_params + 0.1 * np.random.randn(n_samples)

    # 定义损失和梯度
    def mse_loss(params, X, y):
        predictions = X @ params
        errors = predictions - y
        return np.mean(errors ** 2)

    def mse_grad(params, X, y):
        predictions = X @ params
        errors = predictions - y
        return (2 / len(y)) * (X.T @ errors)

    # 训练
    params = np.zeros(n_features)
    optimizer = Adam(lr=0.01)

    params, loss_history = train_with_optimizer(
        X, y, params, optimizer,
        loss_fn=mse_loss, grad_fn=mse_grad,
        epochs=100, batch_size=32, verbose=True
    )

    print(f"\n真实参数: {true_params}")
    print(f"学习参数: {params}")
    print(f"误差: {np.linalg.norm(params - true_params):.6f}")

    assert np.linalg.norm(params - true_params) < 0.1, "线性回归测试失败"
    print("✓ 线性回归测试通过\n")


def compare_optimizers():
    """对比不同优化器的性能"""
    print("="*60)
    print("对比不同优化器")
    print("="*60)

    np.random.seed(42)

    # 生成数据
    n_samples = 500
    X = np.random.randn(n_samples, 3)
    true_params = np.array([1.0, -2.0, 0.5])
    y = X @ true_params + 0.1 * np.random.randn(n_samples)

    # 定义损失和梯度
    def loss_fn(params, X, y):
        return np.mean((X @ params - y) ** 2)

    def grad_fn(params, X, y):
        return (2 / len(y)) * (X.T @ (X @ params - y))

    # 不同优化器配置
    optimizers = {
        'SGD': SGD(lr=0.01),
        'Momentum': Momentum(lr=0.01, momentum=0.9),
        'AdaGrad': AdaGrad(lr=0.1),
        'RMSprop': RMSprop(lr=0.01, decay=0.9),
        'Adam': Adam(lr=0.01),
    }

    results = {}
    epochs = 50

    for name, optimizer in optimizers.items():
        params = np.zeros(3)
        _, loss_history = train_with_optimizer(
            X, y, params, optimizer,
            loss_fn=loss_fn, grad_fn=grad_fn,
            epochs=epochs, batch_size=32, verbose=False
        )
        results[name] = loss_history
        print(f"{name:12s}: 初始Loss = {loss_history[0]:.6f}, "
              f"最终Loss = {loss_history[-1]:.6f}")

    print("\n✓ 所有优化器测试完成\n")
    return results


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("优化器模块测试套件")
    print("="*60 + "\n")

    test_sgd()
    test_momentum()
    test_adam()
    test_lr_schedulers()
    test_gradient_clipping()
    test_linear_regression()
    compare_optimizers()

    print("="*60)
    print("✓ 所有测试通过！")
    print("="*60)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    run_all_tests()

    # 可视化对比（可选）
    try:
        import matplotlib.pyplot as plt

        results = compare_optimizers()

        plt.figure(figsize=(12, 5))

        plt.subplot(121)
        for name, losses in results.items():
            plt.plot(losses, linewidth=2, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('优化器对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(122)
        for name, losses in results.items():
            plt.plot(losses, linewidth=2, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('optimizer_comparison_results.png', dpi=100, bbox_inches='tight')
        print("\n可视化结果已保存到 optimizer_comparison_results.png")

    except ImportError:
        print("\n未安装matplotlib，跳过可视化")
