"""批量生成第 1 章的真实实验图表。"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TEMP_CACHE = Path(tempfile.gettempdir()) / "ai-deepdive-cache"
os.environ.setdefault(
    "MPLCONFIGDIR", str(_TEMP_CACHE / "matplotlib")
)
os.environ.setdefault("XDG_CACHE_HOME", str(_TEMP_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .extensions import checkpoint_demo, gradient_check_curve
    from .mlp import train_toy_mlp
    from .value import Value
    from .viz import draw_graph
else:  # 允许 `python code/ch01/plot_ch01.py` 直接运行
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from code.ch01.extensions import checkpoint_demo, gradient_check_curve
    from code.ch01.mlp import train_toy_mlp
    from code.ch01.value import Value
    from code.ch01.viz import draw_graph


def _plot_finite_difference_error(output_path: Path) -> None:
    epsilons, errors = gradient_check_curve()
    best = int(errors.argmin())

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.loglog(epsilons, errors, marker="o", linewidth=1.8, markersize=4)
    axis.scatter(
        [epsilons[best]],
        [errors[best]],
        color="#d1495b",
        zorder=3,
        label=f"minimum at ε={epsilons[best]:.1e}",
    )
    axis.set(
        xlabel="finite-difference step ε",
        ylabel="relative HVP error",
        title="Truncation error vs. floating-point round-off",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_checkpoint_tradeoff(output_path: Path) -> None:
    result = checkpoint_demo()
    depths = result["depths"]

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].plot(depths, result["plain_saved"], "o-", label="save all")
    axes[0].plot(
        depths,
        result["checkpoint_saved"],
        "o-",
        label="checkpoint",
    )
    axes[0].set(
        xlabel="block depth",
        ylabel="saved activation elements",
        title="Activation storage proxy",
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    width = 0.32
    axes[1].bar(depths - width / 2, result["plain_forward_calls"], width, label="save all")
    axes[1].bar(
        depths + width / 2,
        result["checkpoint_forward_calls"],
        width,
        label="checkpoint",
    )
    axes[1].set(
        xlabel="block depth",
        ylabel="block forward calls",
        title="Recomputation cost",
        xticks=depths,
    )
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    figure.suptitle("Gradient checkpointing trades compute for activation storage")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _build_example_neuron_graph() -> Value:
    """复现 Karpathy micrograd 教程的经典神经元例子，作为可视化的示范图。"""

    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1w1+x2w2"
    n = x1w1x2w2 + b
    n.label = "n"
    o = n.tanh()
    o.label = "o"
    o.backward()
    return o


def _plot_computation_graph(output_path: Path) -> None:
    output = _build_example_neuron_graph()
    figure = draw_graph(output)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_mlp_training(output_path: Path) -> None:
    result = train_toy_mlp()
    loss_history = result["loss_history"]

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.8, color="#3b6ea5")
    axis.set(
        xlabel="training step",
        ylabel="mean squared error",
        title="Training a 3-4-4-1 MLP on the toy dataset with plain gradient descent",
    )
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_all_plots(output_dir: str | Path) -> list[Path]:
    """生成本章全部 PNG，返回生成后的路径。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "finite_difference_error.png",
        output_dir / "checkpoint_tradeoff.png",
        output_dir / "computation_graph_example.png",
        output_dir / "mlp_training_loss.png",
    ]
    _plot_finite_difference_error(paths[0])
    _plot_checkpoint_tradeoff(paths[1])
    _plot_computation_graph(paths[2])
    _plot_mlp_training(paths[3])
    return paths


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[2]
    for generated in generate_all_plots(repository_root / "assets" / "ch01"):
        print(generated.relative_to(repository_root))
