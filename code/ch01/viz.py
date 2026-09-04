"""把标量 ``Value`` 计算图画成分层有向图。

刻意不依赖系统级 Graphviz（``dot`` 可执行文件）：读者的机器上很可能没有装它，
而本书的硬件/环境原则要求每章代码在普通消费级电脑上开箱可跑。这里用
matplotlib 手写一个简化的分层布局——叶子节点在左，输出节点在右，
每条边中点标注产生该值的运算——足够还原"看着表达式长成一棵图"的教学效果。
想要更精细的交互式图（可缩放、可导出 SVG），拓展阅读里指向了真正的 Graphviz 方案。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _trace(root: Any) -> tuple[list[Any], list[tuple[Any, Any]]]:
    """收集 root 所在计算图的全部节点与边（child -> parent）。"""

    nodes: list[Any] = []
    edges: list[tuple[Any, Any]] = []
    visited: set[int] = set()

    def build(node: Any) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        nodes.append(node)
        for child in node._prev:
            edges.append((child, node))
            build(child)

    build(root)
    return nodes, edges


def _depths(nodes: list[Any]) -> dict[Any, int]:
    """按"离叶子的最长路径长度"给每个节点分层，叶子为第 0 层。"""

    cache: dict[int, int] = {}

    def depth_of(node: Any) -> int:
        key = id(node)
        if key in cache:
            return cache[key]
        if not node._prev:
            cache[key] = 0
            return 0
        result = 1 + max(depth_of(parent) for parent in node._prev)
        cache[key] = result
        return result

    return {node: depth_of(node) for node in nodes}


def draw_graph(root: Any, *, figsize: tuple[float, float] = (11.0, 6.0)) -> Figure:
    """绘制 ``root`` 所在的表达式图：叶子在左，输出在右，圆圈标注运算。

    每个矩形节点显示 ``label``、当前 ``data`` 和反向传播后的 ``grad``；
    因此同一张图在调用 ``root.backward()`` 前后会显示不同的 grad 数值，
    适合先画"只有 data"的图，backward 之后再画一次对照。
    """

    nodes, edges = _trace(root)
    depths = _depths(nodes)
    max_depth = max(depths.values()) if depths else 0

    layers: dict[int, list[Any]] = {}
    for node in nodes:
        layers.setdefault(depths[node], []).append(node)

    x_step, y_step = 2.2, 1.4
    positions: dict[Any, tuple[float, float]] = {}
    for depth, layer_nodes in layers.items():
        count = len(layer_nodes)
        for index, node in enumerate(layer_nodes):
            y = (index - (count - 1) / 2.0) * y_step
            positions[node] = (depth * x_step, y)

    figure, axis = plt.subplots(figsize=figsize)

    for child, parent in edges:
        x0, y0 = positions[child]
        x1, y1 = positions[parent]
        axis.annotate(
            "",
            xy=(x1 - 0.75, y1),
            xytext=(x0 + 0.55, y0),
            arrowprops=dict(arrowstyle="-|>", color="#7a7a7a", lw=1.3, shrinkA=0, shrinkB=0),
        )

    for node, (x, y) in positions.items():
        label = node.label or "?"
        text = f"{label}\ndata={node.data:.4f}\ngrad={node.grad:.4f}"
        axis.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eaf1fb", ec="#3b6ea5", lw=1.1),
            zorder=3,
        )
        if node._op:
            axis.text(
                x - 0.75,
                y,
                node._op,
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="circle,pad=0.3", fc="#fdf1de", ec="#c98a2c", lw=1.0),
                zorder=3,
            )

    axis.set_xlim(-1.0, max_depth * x_step + 1.5)
    ys = [pos[1] for pos in positions.values()] or [0.0]
    axis.set_ylim(min(ys) - 1.2, max(ys) + 1.2)
    axis.axis("off")
    axis.set_title("Expression graph — leaves on the left, output on the right")
    figure.tight_layout()
    return figure


def save_graph(root: Any, path: str | Path, **kwargs: Any) -> Path:
    """把 :func:`draw_graph` 的结果保存为 PNG，返回写入路径。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = draw_graph(root, **kwargs)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path
