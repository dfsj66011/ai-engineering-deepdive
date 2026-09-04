# 代码实验说明

## 环境要求

- Python 3.11+
- 安装依赖：在仓库根目录运行 `python -m pip install -r requirements.txt`
- 不要求 GPU：所有章节必须能用小规模实验在 CPU 上跑通。

## 教学入口

打开 `code/chNN/NN_topic.ipynb`，从头按顺序运行单元。仓库提交的 Notebook 保留已执行输出，读者打开即可看到数值与图表。

Notebook 是实验的教学主入口；它按概念拆分单元、解释每一步并调用同目录的真实模块，不复制另一套完整实现。

## 自动化入口

- `python -m pytest code/chNN/ -v`：验证从零实现与主流框架数值一致，默认容差为 `atol=1e-5, rtol=1e-4`。
- `python code/chNN/plot_chNN.py`：无界面批量生成图表到 `assets/chNN/`。
- `.py` 文件是模块化、测试、批处理以及后续章节复用的工程层。

## 第 1 章

```bash
jupyter notebook code/ch01/01_autograd_engine.ipynb
python -m pytest code/ch01/ -v
python code/ch01/plot_ch01.py
```
