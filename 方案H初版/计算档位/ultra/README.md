# ultra 档代码入口

对应方案：**从头生成 H+、G+、E，三轮融合＋局部搜索＋按秒精修**。

已有固定实例的同条件确认结果；历史秒数不预测新设备耗时，也不证明全局最优。

在 `方案H初版/` 中运行：

```bash
uv sync --frozen
uv run python 计算档位/ultra/run.py --show-config
uv run python 计算档位/ultra/run.py --output outputs/effort/ultra-run1 --repeats 2
```

`--show-config` 仅查看参数；第二条 Python 命令才开始完整求解与独立校验。本档包含 G 与多轮融合，全量可能需要数十分钟。实际结果与配置写入新输出目录，默认不调用云端伐谋。

本目录的 `run.py` 是可运行入口，公共实现和依赖来自完整仓库；共同参数解析在 [runner.py](../runner.py)，档位映射在 [effort.py](../../obsplan/effort.py)。修改算法时遵守 [项目约定](../../AGENTS.md)，运行参数与迭代方法见 [目录总说明](../README.md)。
