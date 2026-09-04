"""第 1 章自动微分引擎的可执行验收测试。"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_required_libraries_importable():
    for name in ("numpy", "torch", "matplotlib", "nbformat", "nbclient"):
        assert importlib.import_module(name) is not None


def test_torch_numpy_interoperability():
    """捕获 PyTorch 2.2 与 NumPy 2.x ABI 不兼容导致的对拍失效。"""
    import numpy as np
    import torch

    source = np.array([1.0, 2.0], dtype=np.float64)
    round_trip = torch.tensor(source).numpy()
    np.testing.assert_array_equal(round_trip, source)


def test_value_diamond_graph_accumulates_all_paths():
    """若局部反向用赋值而不是 +=，x 的梯度会遗漏一条路径。"""
    from code.ch01.value import Value

    x = Value(3.0)
    squared = x * x
    output = squared + x
    output.backward()

    assert x.grad == pytest.approx(7.0)


def test_value_branch_and_loop_matches_torch():
    """拓扑遍历必须支持运行时分支和循环展开形成的任意 DAG。"""
    import torch

    from code.ch01.value import Value

    x = Value(1.25)
    output = x
    for _ in range(3):
        output = output * x + x.tanh()
    output.backward()

    torch_x = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)
    torch_output = torch_x
    for _ in range(3):
        torch_output = torch_output * torch_x + torch_x.tanh()
    (expected,) = torch.autograd.grad(torch_output, torch_x)

    assert x.grad == pytest.approx(expected.item(), rel=1e-4, abs=1e-5)


def test_value_backward_resets_intermediates_and_optionally_accumulates_leaves():
    """旧的中间梯度不能再次传播，retain_grad 只累加叶子梯度。"""
    from code.ch01.value import Value

    x = Value(2.0)
    output = x * x

    output.backward()
    assert x.grad == pytest.approx(4.0)

    output.backward()
    assert x.grad == pytest.approx(4.0)

    output.backward(retain_grad=True)
    assert x.grad == pytest.approx(8.0)


def test_value_gradient_matches_central_difference():
    """错误的幂或 exp 局部导数会偏离独立的数值梯度。"""
    import math

    from code.ch01.value import Value

    x0 = 0.7
    x = Value(x0)
    output = (x**3 + 2 / x + x.exp()).relu()
    output.backward()

    def function(value):
        return max(value**3 + 2 / value + math.exp(value), 0.0)

    epsilon = 1e-6
    expected = (function(x0 + epsilon) - function(x0 - epsilon)) / (2 * epsilon)
    assert x.grad == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_tensor_broadcast_gradient_matches_torch():
    """广播维若不求和回收，偏置梯度会保留错误的批次维。"""
    import numpy as np
    import torch

    from code.ch01.tensor import Tensor

    x_data = np.arange(6, dtype=np.float64).reshape(2, 3) / 10
    bias_data = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    x, bias = Tensor(x_data), Tensor(bias_data)
    loss = ((x * bias + bias).tanh()).sum()
    loss.backward()

    torch_x = torch.tensor(x_data, requires_grad=True)
    torch_bias = torch.tensor(bias_data, requires_grad=True)
    torch_loss = torch.tanh(torch_x * torch_bias + torch_bias).sum()
    expected_x, expected_bias = torch.autograd.grad(
        torch_loss, (torch_x, torch_bias)
    )

    np.testing.assert_allclose(x.grad, expected_x.numpy(), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(
        bias.grad, expected_bias.numpy(), atol=1e-5, rtol=1e-4
    )


def test_tensor_non_scalar_backward_requires_matching_seed():
    """非标量输出没有隐含的全 1 上游梯度。"""
    import numpy as np

    from code.ch01.tensor import Tensor

    output = Tensor(np.ones((2, 3))) * 2
    with pytest.raises(ValueError, match="非标量"):
        output.backward()
    with pytest.raises(ValueError, match="shape"):
        output.backward(np.ones(3))


def test_tensor_non_scalar_seed_computes_vector_jacobian_product():
    """显式 seed 应计算 v^T J，而不是把 Jacobian 整体物化。"""
    import numpy as np

    from code.ch01.tensor import Tensor

    x = Tensor(np.array([1.0, 2.0, 3.0]))
    seed = np.array([0.5, -1.0, 2.0])
    (x * x).backward(seed)
    np.testing.assert_allclose(x.grad, 2 * x.data * seed)


def test_tensor_matmul_mean_matches_torch():
    """矩阵乘法左右输入的转置方向或 mean 缩放出错都会被捕获。"""
    import numpy as np
    import torch

    from code.ch01.tensor import Tensor

    rng = np.random.default_rng(7)
    a_data = rng.normal(size=(3, 4))
    weight_data = rng.normal(size=(4, 2))
    a, weight = Tensor(a_data), Tensor(weight_data)
    loss = (a @ weight).tanh().mean()
    loss.backward()

    torch_a = torch.tensor(a_data, requires_grad=True)
    torch_weight = torch.tensor(weight_data, requires_grad=True)
    torch_loss = torch.tanh(torch_a @ torch_weight).mean()
    expected_a, expected_weight = torch.autograd.grad(
        torch_loss, (torch_a, torch_weight)
    )

    np.testing.assert_allclose(a.grad, expected_a.numpy(), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(
        weight.grad, expected_weight.numpy(), atol=1e-5, rtol=1e-4
    )


def test_tensor_axis_reduction_and_read_only_data():
    """axis 规约要恢复维度；只读数据阻止未追踪的原地修改。"""
    import numpy as np

    from code.ch01.tensor import Tensor

    x = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3))
    output = x.sum(axis=1)
    output.backward(np.array([2.0, -1.0]))
    np.testing.assert_array_equal(x.grad, [[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]])

    with pytest.raises(ValueError, match="read-only"):
        x.data[0, 0] = 99.0


def test_finite_difference_hvp_matches_torch():
    """中心差分若方向或分母错误，会偏离 PyTorch 的精确 HVP。"""
    import numpy as np
    import torch

    from code.ch01.extensions import finite_difference_hvp, polynomial_grad

    theta = np.array([0.4, -0.7, 1.2], dtype=np.float64)
    vector = np.array([1.0, -0.5, 0.25], dtype=np.float64)
    actual = finite_difference_hvp(polynomial_grad, theta, vector, 1e-5)

    torch_theta = torch.tensor(theta, requires_grad=True)
    torch_vector = torch.tensor(vector)
    loss = (torch_theta**4 / 4 + torch.sin(torch_theta)).sum()
    (gradient,) = torch.autograd.grad(loss, torch_theta, create_graph=True)
    (expected,) = torch.autograd.grad((gradient * torch_vector).sum(), torch_theta)

    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-5, rtol=1e-4)


def test_gradient_check_curve_exposes_step_size_tradeoff():
    """误差曲线必须真实呈现过大截断误差与过小舍入误差。"""
    import numpy as np

    from code.ch01.extensions import gradient_check_curve

    epsilons, errors = gradient_check_curve()
    assert epsilons.shape == errors.shape == (24,)
    assert np.all(np.isfinite(errors))
    assert errors.min() < errors[0]
    assert errors.min() < errors[-1]


def test_checkpoint_demo_preserves_results_and_exposes_tradeoff():
    """重计算必须保持输出/梯度，同时少存激活并增加前向调用。"""
    import numpy as np

    from code.ch01.extensions import checkpoint_demo

    result = checkpoint_demo(depths=(2, 4, 8))
    np.testing.assert_allclose(
        result["plain_output"], result["checkpoint_output"], atol=1e-12
    )
    np.testing.assert_allclose(
        result["plain_grad"], result["checkpoint_grad"], atol=1e-12
    )
    assert np.all(result["checkpoint_saved"] < result["plain_saved"])
    assert np.all(
        result["checkpoint_forward_calls"] > result["plain_forward_calls"]
    )


def test_generate_all_plots_writes_readable_png_files(tmp_path, monkeypatch):
    """批量脚本必须生成真实可读图片，而不是空占位文件。"""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    import matplotlib.image as mpimg

    from code.ch01.plot_ch01 import generate_all_plots

    paths = generate_all_plots(tmp_path)
    assert {path.name for path in paths} == {
        "finite_difference_error.png",
        "checkpoint_tradeoff.png",
        "computation_graph_example.png",
        "mlp_training_loss.png",
    }
    for path in paths:
        assert path.stat().st_size > 1_000
        assert mpimg.imread(path).size > 0


def test_plot_script_runs_as_documented_cli(tmp_path):
    """脚本直接运行时不能因 package 相对导入而失败。"""
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    result = subprocess.run(
        [sys.executable, "code/ch01/plot_ch01.py"],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "assets/ch01/finite_difference_error.png" in result.stdout
    assert "assets/ch01/checkpoint_tradeoff.png" in result.stdout
    assert "assets/ch01/computation_graph_example.png" in result.stdout
    assert "assets/ch01/mlp_training_loss.png" in result.stdout


def test_draw_graph_renders_readable_expression_graph(tmp_path, monkeypatch):
    """图必须是真的画出来的节点/边，不是空白画布或报错占位图。"""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    import matplotlib.image as mpimg

    from code.ch01.value import Value
    from code.ch01.viz import save_graph

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    output = (a * b + c) ** 2
    output.label = "y"
    output.backward()

    path = save_graph(output, tmp_path / "graph.png")
    assert path.stat().st_size > 1_000
    assert mpimg.imread(path).size > 0


def test_draw_graph_before_and_after_backward_shows_different_grads(tmp_path, monkeypatch):
    """先画一次图（grad 全为 0），backward 之后再画一次，两张图必须不同。"""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    import matplotlib.image as mpimg

    from code.ch01.value import Value
    from code.ch01.viz import save_graph

    a = Value(2.0, label="a")
    b = Value(3.0, label="b")
    output = a * b
    output.label = "y"

    before_path = save_graph(output, tmp_path / "before.png")
    output.backward()
    after_path = save_graph(output, tmp_path / "after.png")

    assert mpimg.imread(before_path).tobytes() != mpimg.imread(after_path).tobytes()


def test_neuron_matches_manual_tanh_computation():
    """Neuron 的前向必须就是 tanh(sum(w_i * x_i) + b)，不是别的什么近似。"""
    import math
    import random

    from code.ch01.mlp import Neuron

    rng = random.Random(0)
    neuron = Neuron(3, rng=rng)
    inputs = [0.5, -1.0, 2.0]
    output = neuron(inputs)

    expected = math.tanh(
        sum(w.data * x for w, x in zip(neuron.weights, inputs)) + neuron.bias.data
    )
    assert output.data == pytest.approx(expected)


def test_mlp_parameters_count_matches_architecture():
    """3-4-4-1 的参数数量必须等于逐层手算的权重加偏置数量。"""
    import random

    from code.ch01.mlp import MLP

    model = MLP(3, [4, 4, 1], rng=random.Random(0))
    expected_count = (3 * 4 + 4) + (4 * 4 + 4) + (4 * 1 + 1)
    assert len(model.parameters()) == expected_count


def test_train_toy_mlp_loss_drops_by_at_least_two_orders_of_magnitude():
    """普通梯度下降在这个玩具数据集上必须能把 MSE 训练到远小于初始值。"""
    from code.ch01.mlp import train_toy_mlp

    result = train_toy_mlp(steps=60, learning_rate=0.05, seed=42)
    loss_history = result["loss_history"]

    assert loss_history[0] > 1.0
    assert loss_history[-1] < loss_history[0] / 100
    # 允许个别步骤因为浮点噪声轻微反弹，但不能是整体震荡发散。
    later_half = loss_history[len(loss_history) // 2 :]
    assert max(later_half) < loss_history[0] / 10


def test_train_toy_mlp_predictions_move_toward_targets():
    """训练后每个样本的预测符号都应该翻转到和目标一致的方向。"""
    from code.ch01.mlp import train_toy_mlp

    result = train_toy_mlp(steps=60, learning_rate=0.05, seed=42)
    for prediction, target in zip(result["final_predictions"], result["targets"], strict=True):
        assert prediction * target > 0


def test_notebook_builder_has_no_invalid_escape_warnings():
    """LaTeX 反斜杠必须使用原始字符串，构建时不应产生 SyntaxWarning。"""
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-Werror::SyntaxWarning",
            "-m",
            "py_compile",
            "code/ch01/build_notebook.py",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_notebook_executes_from_clean_kernel(tmp_path, monkeypatch):
    """逐格运行不能依赖作者机器上的隐藏状态或手工操作。"""
    import nbformat
    from nbclient import NotebookClient

    repository_root = Path(__file__).resolve().parents[2]
    notebook_path = Path(__file__).with_name("01_autograd_engine.ipynb")
    notebook = nbformat.read(notebook_path, as_version=4)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(repository_root)}},
    )
    executed = client.execute()
    assert all(
        output.output_type != "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )
