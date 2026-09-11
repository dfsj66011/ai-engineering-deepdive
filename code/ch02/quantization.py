"""对称 per-tensor 伪量化与三种反向规则。

前向始终模拟 ``round -> clamp -> dequantize``；反向规则由 ``grad_mode``
显式选择。核心实现只依赖第 1 章的 NumPy ``Tensor``，PyTorch 仅用于
独立对拍所需的 detach-trick 示例。
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from code.ch01.tensor import Tensor

GradientMode = Literal["zero", "identity", "clipped"]
VALID_GRADIENT_MODES = frozenset(("zero", "identity", "clipped"))


def _validate_quantization_parameters(
    scale: float,
    qmin: int,
    qmax: int,
    grad_mode: str,
) -> None:
    """验证量化参数的有效性。
    检查 scale 是否为有限正数、qmin/qmax 是否为有效整数对、grad_mode 是否在允许值内。
    """

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale 必须是有限的正数")
    if not isinstance(qmin, int) or not isinstance(qmax, int) or qmin >= qmax:
        raise ValueError("qmin 和 qmax 必须是整数且 qmin < qmax")
    if grad_mode not in VALID_GRADIENT_MODES:
        raise ValueError(
            f"grad_mode 必须是 {sorted(VALID_GRADIENT_MODES)} 之一，收到 {grad_mode!r}"
        )


def quantize_dequantize(
    values: np.ndarray,
    scale: float,
    qmin: int,
    qmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回伪量化值和 clamp 前的整数码。

    NumPy ``round`` 与 PyTorch 在半整数处都采用 round-to-nearest-even。
    保留 clamp 前整数码，是为了让 clipped STE 与官方 fake-quant 的
    backward mask 使用同一个判据。
    """

    integer_codes = np.round(np.asarray(values, dtype=np.float64) / scale)  # 计算整数码向量，通过将浮点值除以 scale 后四舍五入得到，用于后续的 clamp 操作和梯度掩码判据
    fake_quantized = np.clip(integer_codes, qmin, qmax) * scale             # 将 clamp 后的整数码乘以 scale 得到伪量化值，恢复到原始浮点数范围 
    return fake_quantized, integer_codes


def fake_quantize(
    tensor: Tensor,
    scale: float,
    qmin: int,
    qmax: int,
    *,
    grad_mode: GradientMode = "clipped",
) -> Tensor:
    """用真实阶梯前向与指定 surrogate gradient 构造伪量化节点。

    ``zero`` 使用分段常数函数几乎处处为零的真实导数；``identity``
    原样复制上游梯度；``clipped`` 仅在未发生整数码饱和时复制梯度。
    scale 和量化边界在本章视为固定超参数，不参与求导。
    """

    _validate_quantization_parameters(scale, qmin, qmax, grad_mode)              # 验证量化参数的有效性，包括 scale、qmin、qmax 的合法性和 grad_mode 的有效性
    data, integer_codes = quantize_dequantize(tensor.data, scale, qmin, qmax)    # 对伪量化值和整数码进行计算，其中整数码用于后续的梯度掩码判据
    output = Tensor(                                                             # 构建伪量化节点，包含前向的真实阶梯量化和反向的代理梯度 
        data,
        (tensor,),
        f"fake_quant[{grad_mode}]",
        requires_grad=tensor.requires_grad,
    )

    def _backward() -> None:
        if grad_mode == "zero":                                                  # 在 zero 模式下，梯度不传播，用于模拟分段常数量化函数的零导数特性
            return
        if grad_mode == "identity":                                              # 在 identity 模式下，梯度直接复制传播，不应用任何掩码或衰减
            tensor._add_grad(output.grad)
            return
        mask = (qmin <= integer_codes) & (integer_codes <= qmax)                 # 在 clipped 模式下，仅当整数码未饱和时才传播梯度，实现裁剪型代理梯度
        tensor._add_grad(output.grad * mask)                                     

    output._backward = _backward
    return output


def torch_identity_ste(tensor, scale: float, qmin: int, qmax: int):
    """PyTorch detach-trick：量化前向、identity backward。

    ``tensor + (quantized - tensor).detach()`` 的前向等于 ``quantized``；
    被 detach 的差值不贡献梯度，反向只剩恒等路径。
    """

    _validate_quantization_parameters(scale, qmin, qmax, "identity")     
    quantized = (tensor / scale).round().clamp(qmin, qmax) * scale         # 实现量化前向计算，将张量缩放、四舍五入、裁剪到整数范围后恢复原始尺度
    return tensor + (quantized - tensor).detach()                          # 通过 detach 阻断量化差值的梯度，使反向传播仅沿恒等路径进行，实现 identity 型代理梯度
