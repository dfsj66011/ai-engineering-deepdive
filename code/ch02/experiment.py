"""CPU 可运行的伪量化训练对照实验。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from code.ch01.tensor import Tensor
from code.ch02.quantization import GradientMode, fake_quantize


@dataclass(frozen=True)
class TrainingResult:
    """单种反向规则的完整、可复现实验轨迹。"""

    losses: np.ndarray
    accuracies: np.ndarray
    initial_weights: np.ndarray
    final_weights: np.ndarray
    weight_update_norm: float


def make_linearly_separable_data(n_samples: int = 160, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """生成带安全间隔的二维二分类数据，标签为 -1/+1。"""

    rng = np.random.default_rng(seed)                                                   # 初始化随机数生成器，用于生成可复现的随机数据
    half = n_samples // 2                                                               # 计算样本总数的一半，用于平均分配负类和正类样本
    negative = rng.normal(loc=(-1.0, 0.8), scale=0.38, size=(half, 2))                  # 生成负类样本，均值在 (-1.0, 0.8)，标准差为 0.38，共 half 个样本，每个样本为 2 维向量
    positive = rng.normal(loc=(1.0, -0.8), scale=0.38, size=(n_samples - half, 2))      # 生成正类样本，均值在 (1.0, -0.8)，标准差为 0.38，共 n_samples - half 个样本，每个样本为 2 维向量
    features = np.vstack((negative, positive))                                          # 将特征和标签沿样本维度堆叠成完整的训练数据集
    targets = np.vstack((-np.ones((half, 1)), np.ones((n_samples - half, 1))))          # 创建标签数组，其中前 half 个样本标签为 -1（负类），后续样本标签为 +1（正类）
    permutation = rng.permutation(n_samples)                                            # 生成随机排列索引，用于打乱样本顺序以增强模型的泛化能力    
    return features[permutation], targets[permutation]                                  # 返回打乱后的特征和标签数组，确保数据集的样本顺序随机化


def _replace_parameter_data(parameter: Tensor, updated: np.ndarray) -> None:
    """在无梯度的优化器阶段替换叶子值，不修改第 1 章引擎。
    把 parameter 里的数据换成新的 updated, 并且把新数据设为不可写
    """

    data = np.array(updated, dtype=np.float64, copy=True) 
    data.flags.writeable = False                                     
    parameter.data = data


def _train_one(
    features: np.ndarray,
    targets: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: np.ndarray,
    *,
    mode: str,
    steps: int,
    learning_rate: float,
    scale: float,
    qmin: int,
    qmax: int,
) -> TrainingResult:
    """使用指定的模式（浮点或量化）训练线性分类器，返回损失、精度和权重更新信息。"""
    x = Tensor(features, requires_grad=False)                             # 将输入特征转换为张量，对 x 无需更新，不需要梯度信息
    y = Tensor(targets, requires_grad=False)                              # 将目标标签转换为张量，对 y 无需更新，不需要梯度信息
    weights = Tensor(initial_weights)                                     # 将初始权重转换为张量，需要梯度信息用于反向传播优化权重参数
    bias = Tensor(initial_bias)                                           # 将初始偏置转换为张量，需要梯度信息用于反向传播优化偏置参数
    losses: list[float] = []                                              # 初始化损失列表，用于记录每个训练步骤的损失值
    accuracies: list[float] = []                                          # 初始化精度列表，用于记录每个训练步骤的分类精度值

    for _ in range(steps):
        if mode == "float":                                               # 当浮点模式时，直接使用原始权重和偏置，无需量化处理
            effective_weights, effective_bias = weights, bias
        else:
            grad_mode = mode
            effective_weights = fake_quantize(                            # 对权重进行伪量化处理
                weights, scale, qmin, qmax, grad_mode=grad_mode
            )
            effective_bias = fake_quantize(                               # 对偏置进行伪量化处理，使用相同的量化参数和梯度模式，确保偏置与权重的量化方式一致
                bias, scale, qmin, qmax, grad_mode=grad_mode
            )

        predictions = (x @ effective_weights + effective_bias).tanh()     # y = tanh(xw+b)
        loss = ((predictions - y) ** 2).mean()                            # 计算均方误差损失，衡量预测值与目标值之间的差异程度
        loss.backward()                                                   # 执行反向传播计算梯度，用于更新权重和偏置参数

        losses.append(float(loss.data))                                                    # 记录损失
        predicted_labels = np.where(predictions.data >= 0.0, 1.0, -1.0)                    # 根据预测值与目标值的符号比较，将预测结果转换为分类标签（1.0 或 -1.0），用于后续精度计算
        accuracies.append(float(np.mean(predicted_labels == targets)))                     # 计算当前训练步骤的分类精度

        _replace_parameter_data(weights, weights.data - learning_rate * weights.grad)      # 使用学习率和梯度进行参数更新，实现随机梯度下降优化算法
        _replace_parameter_data(bias, bias.data - learning_rate * bias.grad)
    
    initial_vector = np.concatenate((initial_weights.ravel(), initial_bias.ravel()))       # 将初始权重和偏置展平并拼接成一维向量，用于计算权重更新的范数，ravel()将多维数组展平为一维向量
    final_vector = np.concatenate((weights.data.ravel(), bias.data.ravel()))

    return TrainingResult(                                                                 # 返回训练结果对象，包含损失曲线、精度曲线、初始权重向量、最终权重向量和权重更新范数等信息
        losses=np.asarray(losses),
        accuracies=np.asarray(accuracies),
        initial_weights=initial_vector,
        final_weights=final_vector,
        weight_update_norm=float(np.linalg.norm(final_vector - initial_vector)),
    )


def run_training_comparison(
    steps: int = 80,
    seed: int = 7,
    learning_rate: float = 0.2,
    scale: float = 0.25,
    qmin: int = -4,
    qmax: int = 4,
) -> dict[str, TrainingResult]:
    """以完全相同的数据和初始化比较 float/zero/identity/clipped。"""

    if steps <= 0:
        raise ValueError("steps 必须为正整数")
    
    features, targets = make_linearly_separable_data(seed=seed)          # 生成数据集用于线性可分类问题的训练，确保所有量化模式使用相同的输入数据
    rng = np.random.default_rng(seed + 1)                                # 初始化随机数生成器，用于生成可重复的初始权重和偏置
    initial_weights = rng.normal(0.0, 0.08, size=(2, 1))                 # 从随机数生成器中采样初始权重，形状为(2, 1)，均值为0，标准差为0.08，用于所有量化模式的一致初始化
    initial_bias = np.zeros((1,), dtype=np.float64)                      # 初始化偏置为零向量，形状为(1,)，数据类型为float64，用于所有量化模式的一致初始化

    return {
        mode: _train_one(
            features,
            targets,
            initial_weights.copy(),
            initial_bias.copy(),
            mode=mode,
            steps=steps,
            learning_rate=learning_rate,
            scale=scale,
            qmin=qmin,
            qmax=qmax,
        )
        for mode in ("float", "zero", "identity", "clipped")             # 遍历四种量化模式，分别训练并返回各模式的训练结果字典
    }
