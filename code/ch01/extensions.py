"""第 1 章的高阶导数与 gradient checkpointing 教学实验。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .tensor import Tensor

GradientFunction = Callable[[np.ndarray], np.ndarray]
TensorFunction = Callable[..., Tensor]


def finite_difference_hvp(
    grad_fn: GradientFunction,
    theta: np.ndarray,
    vector: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """以一阶梯度的中心差分近似 ``H(theta) @ vector``。"""

    theta = np.asarray(theta, dtype=np.float64)                    # 确保 theta 是 np.ndarray
    vector = np.asarray(vector, dtype=np.float64)                  # 确保 vector 是 np.ndarray

    if theta.shape != vector.shape:                                # 因为 HVP 的定义是 Hessian 矩阵与向量的乘积，Hessian 矩阵是关于 theta 的二阶导数矩阵，
        raise ValueError("theta 与 vector 必须具有相同 shape")       # 而 vector 是一个与 theta 维度相同的向量，因此它们必须具有相同的 shape 才能进行矩阵乘法运算。
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正数")
    
    positive = grad_fn(theta + epsilon * vector)                   # 计算在 theta + epsilon * vector 处的梯度       
    negative = grad_fn(theta - epsilon * vector)                   # 计算在 theta - epsilon * vector 处的梯度
    return (positive - negative) / (2.0 * epsilon)                 # 通过中心差分公式近似计算 Hessian 向量积 H(theta) @ vector


def polynomial_grad(theta: np.ndarray) -> np.ndarray:
    """用自研 Tensor 引擎求 ``sum(x^4/4 + sin(x))`` 的梯度。"""

    x = Tensor(np.asarray(theta, dtype=np.float64))                # 通过将 theta 转换为 Tensor 对象，利用自研的自动微分功能来计算梯度。
    objective = (x**4 / 4.0 + x.sin()).sum()                       # 计算目标函数的值，即 sum(x^4/4 + sin(x))，并将其存储在 objective 中。
    objective.backward()                                           # 通过调用 objective 的 backward 方法，触发自动微分计算，计算目标函数对输入 x 的梯度，并将结果存储在 x.grad 中。
    return x.grad.copy()                                           # 返回计算得到的梯度值，即目标函数对输入 theta 的梯度。使用 copy() 方法确保返回的是一个独立的副本，避免后续操作对原始梯度的修改。


def gradient_check_curve() -> tuple[np.ndarray, np.ndarray]:
    """返回有限差分步长与 HVP 相对误差，用于绘制 U 形曲线。"""

    theta = np.array([0.4, -0.7, 1.2], dtype=np.float64)                             # 定义一个三维的参数向量 theta
    vector = np.array([1.0, -0.5, 0.25], dtype=np.float64)                           # 定义一个与 theta 维度相同的向量 vector
    exact = (3.0 * theta**2 - np.sin(theta)) * vector                                # 计算精确的 Hessian 向量积 H(theta) @ vector。x^4/4 + sin(x) 的二阶导数是 3x^2 - sin(x)
    epsilons = np.logspace(-12, -1, 24)                                              # 生成从 10^-12 到 10^-1 的 24 个对数间隔的步长 epsilon，用于有限差分近似 HVP。
    denominator = max(float(np.linalg.norm(exact)), np.finfo(np.float64).eps)        # 计算精确 HVP 的范数，用于归一化误差，避免除以零的情况。
    errors = np.array(  
        [
            np.linalg.norm(                                                          # 计算有限差分近似 HVP 与精确 HVP 之间的误差，并归一化
                finite_difference_hvp(polynomial_grad, theta, vector, epsilon)
                - exact
            )
            / denominator
            for epsilon in epsilons
        ]
    )
    return epsilons, errors


@dataclass
class CheckpointStats:
    """记录教学 checkpoint 的激活代理量与前向重算次数。"""

    saved_elements: int = 0
    forward_calls: int = 0


def checkpoint(
    function: TensorFunction,
    *inputs: Tensor,
    stats: CheckpointStats | None = None,
) -> Tensor:
    """只保存边界输入，反向时重跑 ``function`` 的最小 checkpoint。

    ``function`` 必须是确定性的纯函数；这个教学实现只支持单个 Tensor 输出。
    """

    if not inputs:
        raise ValueError("checkpoint 至少需要一个 Tensor 输入")
    
    stats = stats if stats is not None else CheckpointStats()     

    # 正常计算图：x -> scale -> + bias -> tanh -> x -> scale -> + bias -> tanh -> ... -> result
    # 此处计算副本图：x' -> scale -> + bias -> tanh -> x -> scale -> + bias -> tanh -> ... -> result
    #              ↑ （仅复制数据，不保存副本计算图）                                              ↓ （仅复制数据，重建计算图，直接将 x 作为父节点）
    # 此处计算正本图：x                  -> (checkpoint)                                   -> result
    detached = tuple(Tensor(value.data, requires_grad=value.requires_grad) for value in inputs)   # 切断计算图，构建 x'，下面的 function(*detached) 会产生计算图，但我们不想保存这个计算图
    first_output = function(*detached)                                                            # 此处 first_output 计算是按照 x' -> scale -> + bias -> tanh -> x -> scale -> .... 计算图运算的
    if not isinstance(first_output, Tensor):
        raise TypeError("checkpoint function 必须返回单个 Tensor")

    stats.forward_calls += 1                                                                      # 此次 +1 是常规的前向计算

    # 这里模拟的效果是，丢弃了完整的中间所有计算过程，所以不论实际网络多深，这里只保留了输入，值不随 depth 参数变化
    # 但这只是一种极端情况，只展示 demo 效果，因此这需要再反向计算时再完整重算整个计算图，才能得到正确的梯度。
    stats.saved_elements += sum(value.data.size for value in inputs)                              

    # 这里才是最核心的地方，它没有直接返回上面计算出的 first_output 节点，
    # 而是构建了一个新的 Tensor 对象 output，直接掐断了 first_output 的计算图，并把 inputs 作为 output 的 parents，因此整个计算图在此处不复存在了
    # 这会导致反向传播时不会使用 first_output 的梯度，而是触发 output 的 _backward() 函数，从而实现了 checkpointing。
    # 并定义了一个 _backward() 函数来处理反向传播时的梯度计算。这个 _backward() 函数会在反向传播时重新计算函数的输出，并将梯度累加到原始输入 Tensor 的 grad 属性中，从而实现了 checkpointing 的效果。
    output = Tensor(
        first_output.data,
        tuple(inputs),
        "checkpoint",
        requires_grad=any(value.requires_grad for value in inputs),
    )

    def _backward() -> None:
        recompute_inputs = tuple(Tensor(value.data, requires_grad=value.requires_grad) for value in inputs)
        recomputed = function(*recompute_inputs)                                                  # 重建副本计算图 recompute_inputs -> scale -> + bias -> tanh -> ... -> result
        stats.forward_calls += 1                                                                  # 这次 +1 是反向传播时为了重建计算图的前向计算
        recomputed.backward(output.grad)                                                          # 在重建的计算图中可以按常规的方式进行反向传播，计算梯度并累加到 recompute_inputs 的 grad 属性中
        for original, recomputed_input in zip(inputs, recompute_inputs, strict=True):             # 需要把副本计算图 recompute_inputs 的梯度复制回原始输入 Tensor original 的 grad 属性中
            original._add_grad(recomputed_input.grad)

    output._backward = _backward
    return output


def _activation_chain(value: Tensor, depth: int) -> Tensor:
    """构建一个深度为 ``depth`` 的激活链，便于测试 checkpoint。"""
    result = value                                           # 通过将输入 Tensor 赋值给 result，初始化激活链的起点。
    for layer in range(depth):                               # 虚构 depth 个层激活，每一层都对前一层的输出进行缩放、偏移和非线性变换。
        scale = 1.02 + layer * 0.005
        bias = (layer + 1) * 0.01
        result = (result * scale + bias).tanh()
    return result


def checkpoint_demo(depths: Iterable[int] = (2, 4, 8, 16)) -> dict[str, np.ndarray]:
    """比较保存全部激活与整块重计算的输出、梯度和统计量。"""

    depth_values = np.asarray(tuple(depths), dtype=np.int64)                          
    if depth_values.size == 0 or np.any(depth_values < 2):                            
        raise ValueError("depths 必须包含至少一个不小于 2 的整数")

    input_data = np.linspace(-0.8, 0.8, 32, dtype=np.float64).reshape(4, 8)           
    plain_outputs: list[float] = []                                                   # 用于存储在不使用 checkpoint 的情况下计算得到的输出结果。
    checkpoint_outputs: list[float] = []                                              # 用于存储在使用 checkpoint 的情况下计算得到的输出结果。
    plain_grads: list[np.ndarray] = []                                                
    checkpoint_grads: list[np.ndarray] = []                                          
    plain_saved: list[int] = []                                                       
    checkpoint_saved: list[int] = []                                                  
    plain_calls: list[int] = []                                                       
    checkpoint_calls: list[int] = []                                                  

    for depth in depth_values:                  
        # 1. 构建不使用 checkpoint 的激活链，计算输出和梯度,  
        # x -> scale -> + bias -> tanh -> x -> scale -> + bias -> tanh -> ... -> result         
        # 中间节点的梯度会被保存下来，反向传播时直接使用这些梯度，而不需要重新计算前向过程。                       
        plain_input = Tensor(input_data)                                                  
        plain_result = _activation_chain(plain_input, int(depth))                         
        plain_loss = plain_result.sum()                                                   
        plain_loss.backward()         

        # 2. 构建使用 checkpoint 的激活链，计算输出和梯度                                                    
        # x -> checkpoint -> result
        # 此处 checkponit 包装器，把整个 function 对“外层计算图”伪装成了一个单独的算子节点，它有自己的前向和反向传播逻辑。
        stats = CheckpointStats()                                                         
        checkpoint_input = Tensor(input_data)                                             
        checkpoint_result = checkpoint(                                                   
            lambda value, current_depth=int(depth): _activation_chain(value, current_depth),
            checkpoint_input,
            stats=stats,
        )
        checkpoint_loss = checkpoint_result.sum()                                         
        checkpoint_loss.backward()                                                        

        plain_outputs.append(float(plain_loss.data))                                      
        checkpoint_outputs.append(float(checkpoint_loss.data))                            
        plain_grads.append(plain_input.grad.copy())                                       
        checkpoint_grads.append(checkpoint_input.grad.copy())                             
        plain_saved.append(int(depth) * input_data.size)                                  
        checkpoint_saved.append(stats.saved_elements)                                     
        plain_calls.append(1)                                                             
        checkpoint_calls.append(stats.forward_calls)                                      

    return {
        "depths": depth_values,
        "plain_output": np.asarray(plain_outputs),
        "checkpoint_output": np.asarray(checkpoint_outputs),
        "plain_grad": np.asarray(plain_grads),
        "checkpoint_grad": np.asarray(checkpoint_grads),
        "plain_saved": np.asarray(plain_saved),
        "checkpoint_saved": np.asarray(checkpoint_saved),
        "plain_forward_calls": np.asarray(plain_calls),
        "checkpoint_forward_calls": np.asarray(checkpoint_calls),
    }
