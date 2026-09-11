"""第 2 章伪量化与 STE 的可执行验收测试。"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_fake_quantize_forward_rounds_and_clips_to_representable_grid():
    """删掉 round 或 clamp 都会让手算量化网格失配。"""
    from code.ch01.tensor import Tensor
    from code.ch02.quantization import fake_quantize

    x = Tensor(np.array([-2.0, -0.74, -0.24, 0.26, 0.74, 2.0]))
    y = fake_quantize(x, scale=0.5, qmin=-2, qmax=2, grad_mode="identity")

    np.testing.assert_array_equal(y.data, [-1.0, -0.5, 0.0, 0.5, 0.5, 1.0])


def test_zero_gradient_mode_blocks_parameter_updates():
    """若无 STE 分支意外透传梯度，分段常数量化就没有被真实模拟。"""
    from code.ch01.tensor import Tensor
    from code.ch02.quantization import fake_quantize

    x = Tensor(np.array([-0.7, 0.2, 1.4]))
    loss = (fake_quantize(x, 0.5, -2, 2, grad_mode="zero") * 3.0).sum()
    loss.backward()

    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 0.0])


def test_identity_ste_copies_the_upstream_gradient():
    """把 identity STE 写成量化值或 scale 倍数会破坏直通语义。"""
    from code.ch01.tensor import Tensor
    from code.ch02.quantization import fake_quantize

    x = Tensor(np.array([-1.4, -0.2, 0.8]))
    seed = np.array([2.0, -3.0, 4.0])
    fake_quantize(x, 0.5, -2, 2, grad_mode="identity").backward(seed)

    np.testing.assert_array_equal(x.grad, seed)


def test_clipped_ste_masks_values_outside_the_quantization_interval():
    """漏掉饱和 mask 会让区间外的参数继续收到梯度。"""
    from code.ch01.tensor import Tensor
    from code.ch02.quantization import fake_quantize

    x = Tensor(np.array([-1.26, -1.24, -0.25, 1.0, 1.24, 1.26]))
    fake_quantize(x, 0.5, -2, 2, grad_mode="clipped").sum().backward()

    np.testing.assert_array_equal(x.grad, [0.0, 1.0, 1.0, 1.0, 1.0, 0.0])


@pytest.mark.parametrize(
    ("scale", "qmin", "qmax", "message"),
    [
        (0.0, -2, 2, "scale"),
        (-0.5, -2, 2, "scale"),
        (0.5, 2, -2, "qmin"),
    ],
)
def test_fake_quantize_rejects_invalid_quantization_parameters(
    scale, qmin, qmax, message
):
    """非法量化参数必须尽早失败，不能产生 NaN 或颠倒的截断。"""
    from code.ch01.tensor import Tensor
    from code.ch02.quantization import fake_quantize

    with pytest.raises(ValueError, match=message):
        fake_quantize(Tensor([0.2]), scale, qmin, qmax)


def test_invalid_gradient_mode_is_rejected():
    """未知的反向约定不能静默退化为任意一种 STE。"""
    from code.ch01.tensor import Tensor
    from code.ch02.quantization import fake_quantize

    with pytest.raises(ValueError, match="grad_mode"):
        fake_quantize(Tensor([0.2]), 0.5, -2, 2, grad_mode="mystery")


def test_fake_quantize_forward_and_clipped_gradient_match_pytorch():
    """量化数值或饱和 mask 偏离官方算子时，对拍必须失败。"""
    import torch

    from code.ch01.tensor import Tensor
    from code.ch02.quantization import fake_quantize

    values = np.array([-1.2, -1.0, -0.6, -0.1, 0.4, 1.0, 1.2])
    x = Tensor(values)
    y = fake_quantize(x, 0.5, -2, 2, grad_mode="clipped")
    y.sum().backward()

    torch_x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    torch_y = torch.fake_quantize_per_tensor_affine(torch_x, 0.5, 0, -2, 2)
    torch_y.sum().backward()

    np.testing.assert_allclose(y.data, torch_y.detach().numpy(), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(x.grad, torch_x.grad.numpy(), atol=1e-5, rtol=1e-4)


def test_detach_trick_has_quantized_forward_and_identity_backward():
    """detach 代数写反时，前向值或梯度至少有一个会失配。"""
    import torch

    from code.ch02.quantization import torch_identity_ste

    x = torch.tensor([-0.74, 0.26, 1.8], dtype=torch.float64, requires_grad=True)
    y = torch_identity_ste(x, scale=0.5, qmin=-2, qmax=2)
    (y * torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)).sum().backward()

    torch.testing.assert_close(y.detach(), torch.tensor([-0.5, 0.5, 1.0], dtype=torch.float64))
    torch.testing.assert_close(x.grad, torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64))


def test_training_comparison_is_deterministic_and_exposes_blocked_learning():
    """若训练没有共享初始化或 zero-gradient 参数发生更新，实验结论不可信。"""
    from code.ch02.experiment import run_training_comparison

    first = run_training_comparison(steps=80, seed=7)
    second = run_training_comparison(steps=80, seed=7)

    for mode in first:
        np.testing.assert_array_equal(first[mode].losses, second[mode].losses)
        np.testing.assert_array_equal(first[mode].accuracies, second[mode].accuracies)

    assert first["zero"].losses[-1] == pytest.approx(first["zero"].losses[0])
    assert first["zero"].weight_update_norm == pytest.approx(0.0)
    assert first["clipped"].losses[-1] < first["clipped"].losses[0] * 0.55
    assert first["clipped"].accuracies[-1] >= 0.9


def test_generate_all_plots_writes_readable_png_files(tmp_path, monkeypatch):
    """绘图入口必须产出两张真实可读 PNG，而不是空白占位文件。"""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    import matplotlib.image as mpimg

    from code.ch02.plot_ch02 import generate_all_plots

    paths = generate_all_plots(tmp_path)
    assert {path.name for path in paths} == {
        "ste_forward_backward.png",
        "training_comparison.png",
    }
    for path in paths:
        assert path.stat().st_size > 1_000
        assert mpimg.imread(path).size > 0


def test_plot_script_runs_as_documented_cli():
    """绘图脚本直接运行时必须正确解析仓库 package 并报告产物。"""
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/tmp/mpl-ch02-cli"
    result = subprocess.run(
        [sys.executable, "code/ch02/plot_ch02.py"],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "assets/ch02/ste_forward_backward.png" in result.stdout
    assert "assets/ch02/training_comparison.png" in result.stdout


def test_notebook_builder_has_no_invalid_escape_warnings():
    """Notebook 中的 LaTeX 反斜杠必须使用原始字符串。"""
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-Werror::SyntaxWarning",
            "-m",
            "py_compile",
            "code/ch02/build_notebook.py",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_notebook_executes_from_clean_kernel(tmp_path, monkeypatch):
    """逐格运行必须复现数值和图表，不能依赖作者会话中的隐藏状态。"""
    import nbformat
    from nbclient import NotebookClient

    repository_root = Path(__file__).resolve().parents[2]
    notebook_path = Path(__file__).with_name("02_backprop_nondifferentiable.ipynb")
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
