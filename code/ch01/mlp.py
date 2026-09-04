"""在标量 ``Value`` 引擎之上搭建 Neuron/Layer/MLP，作为第 1 章的收尾闭环：

前面的章节都在验证"梯度算对了没有"；这里用同一个引擎训练一个真正的小网络，
让读者看到 loss 从"一次前向"变成"逐步下降的曲线"。
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from .value import Value

ValueOrFloat = Value | float


class Neuron:
    """一个 ``tanh(w . x + b)``（或线性，若 ``nonlinearity=False``）神经元。"""

    def __init__(
        self,
        n_inputs: int,
        *,
        nonlinearity: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        generator = rng if rng is not None else random                                   # 随机数生成器      
        self.weights = [Value(generator.uniform(-1.0, 1.0)) for _ in range(n_inputs)]    # 设置神经元权重，一个输入值对应一个权重值，初始均匀分布
        self.bias = Value(0.0)                                                           # 设置神经元偏置，初始为 0
        self.nonlinearity = nonlinearity                                                 # 是否应用非线性激活函数

    def __call__(self, inputs: Sequence[ValueOrFloat]) -> Value:
        activation = self.bias        
        for weight, value in zip(self.weights, inputs, strict=True):
            activation = activation + weight * value                                     # y = w.x + b
        return activation.tanh() if self.nonlinearity else activation                    # 如果应用非线性激活，o = tanh(y)

    def parameters(self) -> list[Value]:
        """记录该神经元所有的参数值, 用于遍历参数更新"""
        return [*self.weights, self.bias]         


class Layer:
    """并列的若干个 :class:`Neuron`，共享同一组输入。"""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        *,
        nonlinearity: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        """神经网络的某一层就是多个神经元的组合, 输出维度 n_outputs 是多少, 就意味着该层有多少个神经元
        每个神经元的权重数量就是 n_inputs 
        """
        self.neurons = [
            Neuron(n_inputs, nonlinearity=nonlinearity, rng=rng) for _ in range(n_outputs)
        ]

    def __call__(self, inputs: Sequence[ValueOrFloat]) -> Value | list[Value]:
        """每个神经元独立计算自身的激活值, 如果该层仅有一个神经元(如输出层), 直接取 [0], 提取标量值"""
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self) -> list[Value]:
        """该层共有多少个神经元, 每个神经元有几个参数"""
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]


class MLP:
    """按 ``layer_sizes`` 堆叠若干 :class:`Layer`；最后一层线性输出原始分数。"""

    def __init__(
        self,
        n_inputs: int,
        layer_sizes: Sequence[int],
        *,
        rng: random.Random | None = None,
    ) -> None:
        if not layer_sizes:
            raise ValueError("layer_sizes 至少需要一层")
        
        sizes = [n_inputs, *layer_sizes]                      # 网络每一层有几个神经元，含输入层
        self.layers = [
            Layer(
                sizes[index],                                 # 前层作为输入，后层作为输出，前后排列起来，组成 MLP
                sizes[index + 1],
                nonlinearity=(index != len(layer_sizes) - 1),
                rng=rng,
            )
            for index in range(len(layer_sizes))
        ]

    def __call__(self, inputs: Sequence[ValueOrFloat]) -> Value:
        current: Sequence[ValueOrFloat] | Value | list[Value] = inputs     # 初始输入值
        for layer in self.layers:
            current = layer(current)                                       # 消费当前输入值，输出值作为下一层的输入值
        assert isinstance(current, Value), "最后一层必须只有一个输出神经元"
        return current                                                     # 直至最后返回输出值

    def parameters(self) -> list[Value]:
        """统计一共有多少层, 每层多少个参数"""
        return [parameter for layer in self.layers for parameter in layer.parameters()]


# Karpathy micrograd 教程同款玩具数据集：4 个三维样本，二分类目标 ±1。
TOY_INPUTS: list[list[float]] = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
TOY_TARGETS: list[float] = [1.0, -1.0, -1.0, 1.0]


def train_toy_mlp(
    steps: int = 60,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> dict[str, object]:
    """在玩具数据集上训练一个 3-4-4-1 MLP，返回逐步均方误差和训练前后预测。

    每一步都直接调用 ``loss.backward()``——不需要手动清零参数梯度，因为
    第 1 章 ``Value.backward`` 默认会清空当前图里包括叶子在内的全部旧梯度
    （见 value.py 与第 55 章"两种保留"的讨论），本函数依赖这一点，
    没有重复实现 zero_grad。
    """

    rng = random.Random(seed)
    model = MLP(3, [4, 4, 1], rng=rng)

    initial_predictions = [model(x).data for x in TOY_INPUTS]

    loss_history: list[float] = []
    for _ in range(steps):
        predictions = [model(x) for x in TOY_INPUTS]
        squared_errors = [
            (prediction - target) ** 2                        
            for prediction, target in zip(predictions, TOY_TARGETS, strict=True)
        ]
        loss = sum(squared_errors) / len(squared_errors)          # 采用均方误差损失函数

        loss.backward()                                           # 反向传播求梯度
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad      # 参数更新

        loss_history.append(loss.data)                            # 记录不同步数下的损失值，用于日志作图

    final_predictions = [model(x).data for x in TOY_INPUTS]
    return {
        "model": model,
        "loss_history": loss_history,
        "initial_predictions": initial_predictions,
        "final_predictions": final_predictions,
        "targets": TOY_TARGETS,
    }
