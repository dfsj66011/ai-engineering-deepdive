"""批量生成第 2 章的真实实验图表。"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_TEMP_CACHE = Path(tempfile.gettempdir()) / "ai-deepdive-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_TEMP_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TEMP_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .experiment import run_training_comparison
    from .quantization import quantize_dequantize
else:  # 允许 `python code/ch02/plot_ch02.py` 直接运行
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from code.ch02.experiment import run_training_comparison
    from code.ch02.quantization import quantize_dequantize


def _plot_ste_forward_backward(output_path: Path) -> None:
    inputs = np.linspace(-1.8, 1.8, 721)
    quantized, integer_codes = quantize_dequantize(inputs, 0.5, -2, 2)
    zero_gradient = np.zeros_like(inputs)
    identity_gradient = np.ones_like(inputs)
    clipped_gradient = ((integer_codes >= -2) & (integer_codes <= 2)).astype(float)

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(inputs, inputs, "--", color="#888888", label="identity")
    axes[0].step(inputs, quantized, where="mid", color="#2f6f9f", label="fake quantized")
    axes[0].set(ylabel="forward output", title="One forward function, three backward choices")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(inputs, zero_gradient, label="true / zero gradient", linewidth=2.0)
    axes[1].plot(inputs, identity_gradient, label="identity STE", linewidth=2.0)
    axes[1].plot(inputs, clipped_gradient, label="clipped STE", linewidth=2.0)
    axes[1].axvline(-1.25, color="#888888", linestyle=":", linewidth=1.0)
    axes[1].axvline(1.25, color="#888888", linestyle=":", linewidth=1.0)
    axes[1].set(xlabel="input x", ylabel="surrogate derivative", ylim=(-0.08, 1.15))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_training_comparison(output_path: Path) -> None:
    results = run_training_comparison()
    colors = {
        "float": "#3b6ea5",
        "zero": "#d1495b",
        "identity": "#f28e2b",
        "clipped": "#2a9d8f",
    }
    labels = {
        "float": "float baseline",
        "zero": "fake quant + zero grad",
        "identity": "fake quant + identity STE",
        "clipped": "fake quant + clipped STE",
    }

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for mode, result in results.items():
        steps = np.arange(1, result.losses.size + 1)
        loss_style = "steps-post" if mode in {"zero", "identity", "clipped"} else "default"
        axes[0].plot(
            steps,
            result.losses,
            label=labels[mode],
            color=colors[mode],
            drawstyle=loss_style,
        )
        axes[1].plot(steps, result.accuracies, label=labels[mode], color=colors[mode])

    axes[0].set(xlabel="training step", ylabel="mean squared error", title="Loss")
    axes[1].set(xlabel="training step", ylabel="classification accuracy", title="Accuracy", ylim=(0.45, 1.02))
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    # 放大训练早期的低 loss 区域。纵轴范围刻意排除 zero-gradient 曲线，
    # 否则它始终停在高位，会把真正想观察的量化平台与跨格跳降压扁。
    zoom_start = 4
    zoom_end = min(25, next(iter(results.values())).losses.size)
    zoom_slice = slice(zoom_start - 1, zoom_end)
    learning_modes = ("float", "identity", "clipped")
    zoom_values = np.concatenate(
        [results[mode].losses[zoom_slice] for mode in learning_modes]
    )
    zoom_min = float(zoom_values.min())
    zoom_max = float(zoom_values.max())
    zoom_padding = max((zoom_max - zoom_min) * 0.1, 1e-4)

    loss_zoom = axes[0].inset_axes([0.43, 0.12, 0.54, 0.48])
    for mode in learning_modes:
        result = results[mode]
        steps = np.arange(1, result.losses.size + 1)
        loss_zoom.plot(
            steps,
            result.losses,
            color=colors[mode],
            linewidth=1.4 if mode == "float" else 2.0,
            drawstyle="default" if mode == "float" else "steps-post",
        )
    loss_zoom.set(
        xlim=(zoom_start, zoom_end),
        ylim=(max(0.0, zoom_min - zoom_padding), zoom_max + zoom_padding),
        title=f"Steps {zoom_start}–{zoom_end}: quantization plateaus",
    )
    loss_zoom.tick_params(labelsize=7)
    loss_zoom.title.set_fontsize(8)
    loss_zoom.grid(True, alpha=0.25)
    axes[0].indicate_inset_zoom(loss_zoom, edgecolor="#666666", alpha=0.8)

    figure.suptitle("The forward quantizer needs a usable backward surrogate")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_all_plots(output_dir: str | Path) -> list[Path]:
    """生成本章全部 PNG，返回生成后的路径。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "ste_forward_backward.png",
        output_dir / "training_comparison.png",
    ]
    _plot_ste_forward_backward(paths[0])
    _plot_training_comparison(paths[1])
    return paths


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[2]
    for generated in generate_all_plots(repository_root / "assets" / "ch02"):
        print(generated.relative_to(repository_root))
