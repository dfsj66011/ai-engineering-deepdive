"""标量级 reverse-mode 自动微分引擎。

这个实现刻意保留最少抽象：每个 :class:`Value` 既是前向结果，也是计算图节点。
算子在前向时记录父节点和局部反向闭包，``backward`` 再按逆拓扑序执行闭包。
"""

from __future__ import annotations

import math
from collections.abc import Callable


class Value:
    """动态标量计算图中的一个节点。"""

    def __init__(
        self,
        data: float,
        _children: tuple["Value", ...] = (),
        _op: str = "",
        *,
        label: str = "",
    ) -> None:
        """初始化一个 :class:`Value` 节点。
	
        Parameters:
            data: 节点的标量值
            _children: 父节点元组
            _op: 生成当前节点的算子名称
            label: 节点标签
        """
        self.data = float(data)                                  # 前向计算结果
        self.grad = 0.0                                          # 反向传播梯度
        self.label = label                                       # 可选标签，便于调试和可视化
        self._prev = tuple(_children)                            # 父节点
        self._op = _op                                           # 生成当前节点的算子
        self._backward: Callable[[], None] = lambda: None        # 反向传播闭包

    @staticmethod
    def _coerce(other: "Value | float") -> "Value":
        """将标量提升为 :class:`Value`，便于算子重载。"""
        return other if isinstance(other, Value) else Value(float(other))

    def __add__(self, other: "Value | float") -> "Value":
        """加法算子重载，支持标量和 :class:`Value`。"""
        other = self._coerce(other)
        output = Value(self.data + other.data, (self, other), "+")  

        def _backward() -> None:
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward
        return output

    def __mul__(self, other: "Value | float") -> "Value":
        """乘法算子重载，支持标量和 :class:`Value`。"""
        other = self._coerce(other)
        output = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output

    def __pow__(self, exponent: float) -> "Value":
        """幂运算算子重载，支持标量指数。"""
        if not isinstance(exponent, (int, float)):
            raise TypeError("Value 只支持常数标量指数")
        
        output = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * output.grad

        output._backward = _backward
        return output

    def exp(self) -> "Value":
        """指数函数算子。"""
        output = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += output.data * output.grad

        output._backward = _backward
        return output

    def tanh(self) -> "Value":
        """双曲正切函数算子。"""
        output = Value(math.tanh(self.data), (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - output.data**2) * output.grad

        output._backward = _backward
        return output

    def relu(self) -> "Value":
        """ReLU 函数算子。"""
        output = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += float(self.data > 0.0) * output.grad

        output._backward = _backward
        return output

    def topological_order(self) -> list["Value"]:
        """返回父节点在前、当前节点在后的拓扑序。
        
        例如如下计算图
         a ─┐
            × → u ─┐
         b ─┘      + → y
               c ──┘
        一个可能的拓扑顺序是：[a, b, u, c, y]
        核心要求是，一个节点一定出现在依赖它的节点之前。
        """

        order: list[Value] = []
        visited: set[Value] = set()

        def visit(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                visit(parent)
            order.append(node)

        visit(self)
        return order

    def backward(self, seed: float = 1.0, *, retain_grad: bool = False) -> None:
        """从当前输出反传。

        默认清空当前图的全部旧梯度。``retain_grad=True`` 只累加叶子节点梯度；
        中间节点仍会清空，防止旧的中间梯度被再次传播。
        """
        # 1. 构建整张计算图的拓扑顺序
        order = self.topological_order()

        # 2. 如果需要梯度累积，先暂存叶子节点旧梯度  
        #    为什么叶子节点可以累积梯度值？
        #    当我们需要“多次反向传播，把梯度叠加到同一个参数上”时，就需要对叶子节点累积梯度。
        #    最典型的场景就是梯度累积，例如单次 batch_size=4，8 次反向传播，累积到 batch_size=32
        #    然后进行一次参数更新，这8次累积的梯度值就会累加到叶子节点上    
        leaf_grads = (
            {node: node.grad for node in order if not node._prev} if retain_grad else {}
        ) 

        # 3. 把整张图当前 grad 全清零, 
        #    防止上一次 backward 留下的中间梯度，再次参与这一次传播。
        #    例如 u=2x, y=3u, 第一次反向传播后，y.grad=1, u.grad=3, x.grad=6
        #    如果不清零，下次再反向传播，y.grad 重新置 1，影响不大，
        #    但 u.grad += y.grad * 3=6, x.grad += u.grad * 2 = 18，计算错误了               
        for node in order:
            node.grad = 0.0

        # 4. 输出节点设置 seed，这相当于 dy/dy=1
        self.grad = float(seed)

        # 5. 按反向拓扑顺序执行每个节点的 _backward()
        for node in reversed(order):
            node._backward()

        # 6. 最后把叶子节点旧梯度加回来
        for leaf, previous_grad in leaf_grads.items():
            leaf.grad += previous_grad

    def __neg__(self) -> "Value":
        """负号算子重载。"""
        return self * -1.0

    def __sub__(self, other: "Value | float") -> "Value":
        """减法算子重载，支持标量和 :class:`Value`。"""
        return self + -self._coerce(other)

    def __rsub__(self, other: float) -> "Value":
        """反向减法算子重载，支持标量和 :class:`Value`。"""
        return self._coerce(other) - self

    def __truediv__(self, other: "Value | float") -> "Value":
        """除法算子重载，支持标量和 :class:`Value`。"""
        return self * self._coerce(other) ** -1.0

    def __rtruediv__(self, other: float) -> "Value":
        """反向除法算子重载，支持标量和 :class:`Value`。"""
        return self._coerce(other) / self

    def __radd__(self, other: float) -> "Value":
        """反向加法算子重载，支持标量和 :class:`Value`。"""
        return self + other

    def __rmul__(self, other: float) -> "Value":
        """反向乘法算子重载，支持标量和 :class:`Value`。"""
        return self * other

    def __repr__(self) -> str:
        """返回节点的字符串表示，便于调试。"""
        label = f", label={self.label!r}" if self.label else ""
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}{label})"
