"""基于 NumPy 的最小张量 reverse-mode 自动微分引擎。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

ArrayLike: TypeAlias = np.ndarray | list[float] | tuple[float, ...] | float | int


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """把广播后的梯度沿扩张维求和，还原成输入 ``shape``。"""

    grad = np.asarray(grad, dtype=np.float64)              # 转换为 NumPy 数组

    while grad.ndim > len(shape):                          # 如果 grad 维度比 shape 多，说明 grad 是广播后的结果，需要沿最前面的维度求和
        grad = grad.sum(axis=0)                            # 沿第一个维度求和

    for axis, size in enumerate(shape):                    # 如果 shape 中的某个维度是 1，而 grad 对应的维度不是 1，说明 grad 是广播后的结果，需要沿该维度求和
        if size == 1 and grad.shape[axis] != 1:            # 沿该维度求和
            grad = grad.sum(axis=axis, keepdims=True)      # keepdims=True 保持维度不变
    return grad.reshape(shape)                             # 显式地 reshape 成输入 shape，防止广播后维度不一致


class Tensor:
    """记录 NumPy 前向值和局部反向规则的动态计算图节点。"""

    def __init__(
        self,
        data: ArrayLike,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
        *,
        requires_grad: bool = True,
        label: str = "",
    ) -> None:
        array = np.array(data, dtype=np.float64, copy=True)
        array.flags.writeable = False                   # 让前向计算结果不可写，防止意外修改
        self.data = array
        self.grad = np.zeros_like(array)
        self.requires_grad = bool(requires_grad)        # 叶子节点是否需要梯度
        self.label = label
        self._prev = tuple(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @staticmethod
    def _coerce(other: "Tensor | ArrayLike") -> "Tensor":
        """将标量或数组提升为 :class:`Tensor`，便于算子重载。"""
        return (
            other
            if isinstance(other, Tensor)
            else Tensor(other, requires_grad=False)
        )

    def _add_grad(self, grad: np.ndarray) -> None:
        """把广播后的梯度沿扩张维求和，还原成输入 ``shape``，再累加到 ``self.grad``。"""
        if self.requires_grad:
            self.grad += unbroadcast(grad, self.data.shape)

    def __add__(self, other: "Tensor | ArrayLike") -> "Tensor":
        """加法算子重载，支持标量和 :class:`Tensor`。"""
        other = self._coerce(other)
        output = Tensor(
            self.data + other.data,
            (self, other),
            "+",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            self._add_grad(output.grad)
            other._add_grad(output.grad)

        output._backward = _backward
        return output

    def __mul__(self, other: "Tensor | ArrayLike") -> "Tensor":
        """乘法算子重载，支持标量和 :class:`Tensor`。"""
        other = self._coerce(other)
        output = Tensor(
            self.data * other.data,
            (self, other),
            "*",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            self._add_grad(other.data * output.grad)
            other._add_grad(self.data * output.grad)

        output._backward = _backward
        return output

    def __pow__(self, exponent: float) -> "Tensor":
        """幂运算算子重载，支持标量指数。"""
        if not isinstance(exponent, (int, float)):
            raise TypeError("Tensor 只支持常数标量指数")
        
        output = Tensor(
            self.data**exponent,
            (self,),
            f"**{exponent}",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            self._add_grad(exponent * self.data ** (exponent - 1) * output.grad)

        output._backward = _backward
        return output

    def exp(self) -> "Tensor":
        """指数函数算子。"""
        output = Tensor(
            np.exp(self.data),
            (self,),
            "exp",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            self._add_grad(output.data * output.grad)

        output._backward = _backward
        return output

    def tanh(self) -> "Tensor":
        """双曲正切函数算子。"""
        output = Tensor(
            np.tanh(self.data),
            (self,),
            "tanh",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            self._add_grad((1.0 - output.data**2) * output.grad)

        output._backward = _backward
        return output

    def sin(self) -> "Tensor":
        """正弦函数算子。"""
        output = Tensor(
            np.sin(self.data),
            (self,),
            "sin",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            self._add_grad(np.cos(self.data) * output.grad)

        output._backward = _backward
        return output

    def relu(self) -> "Tensor":
        """ReLU 函数算子。"""
        output = Tensor(
            np.maximum(self.data, 0.0),
            (self,),
            "relu",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            self._add_grad((self.data > 0.0) * output.grad)

        output._backward = _backward
        return output

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        """求和算子。"""
        output = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            (self,),
            "sum",
            requires_grad=self.requires_grad,
        )

        def _backward() -> None:
            grad = output.grad

            if axis is not None and not keepdims:
                # 例如 x = [[1, 2, 3], [4, 5, 6]], y = x.sum(axis=1) = [6, 15]
                # x.shape == (2, 3), y.shape == (2,), output.grad.shape == (2,)
                # 需要注意的是，此时不能随便广播，NumPy 广播是从右向左对齐：
                # grad:       (2,)
                # target:     (2,3)
                # 需要把 grad 显式地扩展成 (2,1)，再广播到 (2,3)
                # 需要把 forward 中被 sum 消掉的轴重新插回来。
                # keepdims=True 时不用 expand_dims，yield 的 shape 已经和 x 一致了，直接广播即可
                axes = (axis,) if isinstance(axis, int) else axis         
                normalized = sorted(ax % self.data.ndim for ax in axes)   # 归一化 axis，防止负数 axis，例如 axis=-1, x.ndim=2, -1 % 2 = 1
                for current_axis in normalized:
                    grad = np.expand_dims(grad, axis=current_axis)

            # axis 为 None 时 最简单，例如 x = [[1, 2], [3, 4]], y = x.sum() = 10, sum 对每个元素的局部导数都是 1
            # 因此只需将 output.grad 广播到 self.data 的形状，假设 output.grad = 2.0，
            # 就是直接把 grad 广播到 self.data 的形状，得到 [[2, 2], [2, 2]]，然后累加到 self.grad 上
            self._add_grad(np.broadcast_to(grad, self.data.shape)) 

        output._backward = _backward
        return output

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        """均值算子。"""
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.data.shape[ax] for ax in axes]))     # np.prod 返回一个数组中所有元素的乘积，此处是指获取被求均值的元素个数
        return self.sum(axis=axis, keepdims=keepdims) / count

    def __matmul__(self, other: "Tensor | ArrayLike") -> "Tensor":
        """矩阵乘法算子重载，支持标量和 :class:`Tensor`。"""
        other = self._coerce(other)

        if self.data.ndim != 2 or other.data.ndim != 2:
            raise NotImplementedError("教学版 matmul 目前只支持二维矩阵")
        
        output = Tensor(
            self.data @ other.data,
            (self, other),
            "matmul",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            self._add_grad(output.grad @ other.data.T)
            other._add_grad(self.data.T @ output.grad)

        output._backward = _backward
        return output

    def topological_order(self) -> list["Tensor"]:
        """返回父节点在前、当前节点在后的拓扑序。"""
        order: list[Tensor] = []
        visited: set[Tensor] = set()

        def visit(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                visit(parent)
            order.append(node)

        visit(self)
        return order

    def backward(
        self,
        gradient: ArrayLike | None = None,
        *,
        retain_grad: bool = False,
    ) -> None:
        """计算向量-Jacobian 乘积并把结果写入叶子节点 ``grad``。"""

        if gradient is None:
            if self.data.size != 1:
                raise ValueError("非标量输出调用 backward 时必须显式提供上游梯度")
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(gradient, dtype=np.float64)
            if seed.shape != self.data.shape:
                raise ValueError(
                    f"上游梯度 shape {seed.shape} 与输出 shape {self.data.shape} 不一致"
                )

        order = self.topological_order()
        leaf_grads = (
            {node: node.grad.copy() for node in order if not node._prev}
            if retain_grad
            else {}
        )
        for node in order:
            node.grad.fill(0.0)

        self.grad[...] = seed
        for node in reversed(order):
            node._backward()

        for leaf, previous_grad in leaf_grads.items():
            leaf.grad += previous_grad

    def __neg__(self) -> "Tensor":
        """负号算子重载。"""
        return self * -1.0

    def __sub__(self, other: "Tensor | ArrayLike") -> "Tensor":
        """减法算子重载，支持标量和 :class:`Tensor`。"""
        return self + -self._coerce(other)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        """反向减法算子重载，支持标量和 :class:`Tensor`。"""
        return self._coerce(other) - self

    def __truediv__(self, other: "Tensor | ArrayLike") -> "Tensor":
        """除法算子重载，支持标量和 :class:`Tensor`。"""
        return self * self._coerce(other) ** -1.0

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        """反向除法算子重载，支持标量和 :class:`Tensor`。"""
        return self._coerce(other) / self

    def __radd__(self, other: ArrayLike) -> "Tensor":
        """反向加法算子重载，支持标量和 :class:`Tensor`。"""
        return self + other

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        """反向乘法算子重载，支持标量和 :class:`Tensor`。"""
        return self * other

    def __repr__(self) -> str:
        """返回张量的字符串表示。"""
        label = f", label={self.label!r}" if self.label else ""
        return (
            f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}"
            f"{label})"
        )
