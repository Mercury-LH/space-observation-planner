# 空间目标自主观测规划

面向百度伐谋杯第四题「开运联合：面向空间目标持续监测的 AI 自主观测任务规划与目标状态分析」（10,000 颗空间目标 × 10 地面站 × 6 天，规划期 `2026-08-01 00:00` 至 `08-07 00:00 UTC`，全部约束口径见 [规则口径.md](规则口径.md)）。

## 主交付方案（方案H）

- **[方案H初版](方案H初版/README.md)**（主交付）：拉格朗日价格协调 + 百度伐谋演化放置规则（稳健模式）+ 按秒精修 + 多线程按秒跨站弹射链与换位置（精确 1D 凸规划加速）+ Skyfield 逐秒独立校验 + 可证明数学上界证书 + 11 页交互工作台（含突发重规划、天气/月光加分项、通用 OpenAI 兼容大模型助手）+ 跨平台一键启动与 Docker。
- **[方案E初版](方案E初版/README.md)**：既有贪心/ALNS 调度基线与伐谋接入。
- **[方案G初版](方案G初版/README.md)**：AI 候选评分 + CP-SAT 滚动时域优化。

### 核心成绩（全量 10,000 目标 × 10 站 × 6 天，全部通过 Skyfield 逐秒校验 0 违规）

<!-- frontier-table:begin -->
| 档位 | 命令 | 全程用时（秒） | 综合 | 覆盖 / 得分 | 平均占用核数 | Skyfield 逐秒校验 |
|---|---|---:|---:|---:|---:|---|
| 快速 + 30 秒 | `python -m obsplan plan --mode fast --chain 0.5` | 54 | 0.9862 | 9999 / 29238 | 2.4 | 1 次全部合法 |
| 快速（不跑弹射链） | `python -m obsplan plan --mode fast --chain 0` | 59 | 0.9690 | 9987 / 28238 | 1.7 | 1 次全部合法 |
| 推荐档（默认，约 2 分钟） | `python -m obsplan plan` | 103 | 0.9899 | 9999 / 29456 | 2.5 | 1 次全部合法 |
| 快速 + 3 分钟 | `python -m obsplan plan --mode fast --chain 3` | 242 | 0.9905 | 9999 / 29495 | 1.6 | 1 次全部合法 |
| 十分钟档（快速 + 9 分钟） | `python -m obsplan plan --mode fast --chain 9` | 565 | 0.9940 | 9999 / 29702 | 2.1 | 1 次全部合法 |

每档从启动进程到写出计划端到端跑 1 次取中位数（括号里是最小–最大），可见窗口已缓存，用时不含 Skyfield 校验；测速机器 macOS-27.2.1-arm64-arm-64bit，10 核。原始记录在 `results/frontier/`，本表由 `scripts/frontier_table.py` 生成。
<!-- frontier-table:end -->

- **离线最好计划**（随仓库提供：`方案H初版/outputs/final_best/longrun_chain_best.csv`，支持 `python -m obsplan validate` 一条命令复核）：覆盖 **9,999 / 9,999** 颗可观测目标（另 1 颗 SGP4 外推异常），得分 **29,924.8**，综合指标 **0.9977**（`0.997667`），离可证明按秒数学上界 **1.0047**（附 10 秒细网格对偶价格证书）差距 **≤ 0.70%**。

### 快速运行（评委一键启动）

进入 `方案H初版/` 目录：
- **网页界面（推荐）**：Windows 双击 `start_app.bat`、macOS 双击 `start_app.command`、Linux 运行 `bash start_app.sh`（仓库已预载推荐档与最好计划，打开即看）。
- **命令行规划与校验**：
  ```bash
  uv sync --extra app
  uv run python -m obsplan plan --validate                                       # 推荐档（约 2 分钟，综合 ~0.990）
  uv run python -m obsplan validate outputs/final_best/longrun_chain_best.csv    # 复核最好计划（0.9977）
  ```
- **Docker**：见 [方案H初版/README.md](方案H初版/README.md#评委运行指南)。
