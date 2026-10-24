# 六档代码入口

每个档位有自己的文件夹、可运行 `run.py` 和说明。它们共享同一仓库中的 `obsplan/` 求解、几何、计分和校验代码，以及 `pyproject.toml`、`uv.lock` 和仓库数据；组员克隆完整仓库即可运行。各档位参数仍只在 `obsplan/effort.py` 定义，文件夹不会另存一套容易失效的参数。

| 文件夹 | 对应求解配置 | 当前证据 |
|---|---|---|
| [instant](instant/README.md) | H+ 1 轮、快速局部搜索、按秒精修 | 同条件确认 |
| [medium](medium/README.md) | H+ 10 轮、快速局部搜索、按秒精修 | 同条件确认 |
| [high](high/README.md) | H+ 30 轮、快速局部搜索、按秒精修 | 探索，待统一耗时校准 |
| [xhigh](xhigh/README.md) | H+ 75 轮、快速局部搜索、按秒精修 | 探索，待统一耗时校准 |
| [max](max/README.md) | H+ 300 轮、完整局部搜索、按秒精修 | 同条件确认 |
| [ultra](ultra/README.md) | 从头生成 H+/G+/E、三轮融合与精修 | 同条件确认 |

## 运行

以下命令在 `方案H初版/` 中执行：

```bash
uv sync --frozen
# 只查看最终完整配置，不运行求解器、不创建输出。
uv run python 计算档位/instant/run.py --show-config
# 从头运行并独立逐秒校验。每次必须选择一个新的输出目录。
uv run python 计算档位/instant/run.py --output outputs/effort/instant-run1 --repeats 2
# 替换档位路径即可使用其余五档，例如：
uv run python 计算档位/high/run.py --show-config
```

也可以使用本项目虚拟环境的 Python 和 `run.py` 的绝对路径从任意目录启动；相对 `--output` 总是相对 `方案H初版/`，不会因当前工作目录改变。

每次运行由公共 `obsplan.benchmark` 启动独立进程，完整记录来源生成、搜索、融合、精修和 Skyfield 1 秒步长校验，计算内部时间加权指标。输出仍在忽略的 `outputs/` 内，包括 `summary.json`、`report.md`、`manifest.json` 及每次运行记录；入口返回后另存 `source-suite.json`，记录当次动态解析的完整配置，便于重新评测。该文件不是旧计划，不会复用旧求解结果。

常用参数：

- `--workers 8`：统一工作线程数。不同线程配置的耗时不能混排。
- `--seed 2026 --repeats 2`：两次重复分别使用 2026、2027；最终候选至少复跑一次。
- `--timeout-s`：每次独立进程的总时限，包含启动、准备、求解和校验；默认前五档 1200 秒、ultra 3600 秒，不是求解器的固定搜索预算。
- `--time-weight`、`--time-scale-s`：默认 3/7 与 120 秒，沿用内部效用口径，不是官方比赛换分公式。

没有跳过校验的开关。非法、未完成校验、超时或环境变化仍由统一评测器排除；退出码 0 表示该方法的全部重复可排名，退出码 2 表示参数失败或没有可排名结果。保留失败记录，不覆盖旧目录。

正式复现标准期固定为 2026-08-01 至 08-07 UTC、全量 10000 目标与 10 站。交互子集和其他规则的探索使用产品界面或自定义 benchmark suite。六档均使用已采纳的本地伐谋规则，运行不需要 API key，也不会自动创建云端实验。

## 组员如何优化

先用 `--show-config` 确认本档参数，再通过公共 solver 接口开发独立候选。核心实现位于 `obsplan/lagrange.py`、`local_search.py`、`portfolio.py`、`fusion_merge.py` 与 `refine.py`；不要在六个目录里各复制一套规则。

历史结果对应当时的代码、功耗条件和参数指纹。本入口生成完整参数，可能与历史最小覆盖参数的指纹不同，即使求解语义相同也不要强行混合旧报告；在冻结的新代码上重跑基线和候选。当前四个确认点与 high/xhigh 两个探索点的边界见 [六档规划](../docs/六档规划.md)。下一阶段计划见 [优化交接](../docs/下一阶段优化交接.md)。
