# 空间目标自主观测规划

面向百度伐谋杯第四题「开运联合：面向空间目标持续监测的 AI 自主观测任务规划与目标状态分析」（10,000 颗空间目标 × 10 地面站 × 7 天，规划期按主办方《提交格式规范》v3.0 为 `2026-07-31 00:00:00` 至 `08-06 23:59:59 UTC`，全部约束口径见 [规则口径.md](规则口径.md)）。

> 规划期 2026-11-27 起按规范 v3.0 改为 7 天；此前按企业老师 10-21 确认的 8 月 1 日起 6 天，旧口径的命令加 `--start 2026-08-01 --days 6` 仍可复现（`--chain 0` 的计划与改动前逐位相同）。7 天里两两隔 48 小时最多能观测 4 次（6 天最多 3 次），规划器按规划期自动取上限。

## 主交付方案（方案H）

- **[方案H初版](方案H初版/README.md)**（主交付）：拉格朗日价格协调 + 百度伐谋演化放置规则（稳健模式）+ 按秒精修 + 多线程按秒跨站弹射链与换位置（精确 1D 凸规划加速）+ Skyfield 逐秒独立校验 + 可证明数学上界证书 + 11 页交互工作台（含突发重规划、天气/月光加分项、通用 OpenAI 兼容大模型助手）+ 跨平台一键启动与 Docker。
- **[方案E初版](方案E初版/README.md)**：既有贪心/ALNS 调度基线与伐谋接入。
- **[方案G初版](方案G初版/README.md)**：AI 候选评分 + CP-SAT 滚动时域优化。

### 正式提交文件（7 天，留 5 秒边界余量）

`方案H初版/outputs/final_best/submission_7d_edge5.json`（主办方格式 v3.0，计划明细 `best_7d_edge5.csv`）：15,030 条观测，按规范复算**全部正常**（0 软错误、0 硬错误，错误率系数 1.0），最终得分 **35,912.3**，覆盖 **9,998** 颗；主办方的预报窗口每端比我们短 5 秒以内时分数不变（短 6 秒降到 31,664.4）；Skyfield 逐秒校验全部合法。由十分钟档（`plan --mode fast --edge-margin 5 --chain 9`）加余量修补得到。

### 各档用时与分数（6 天旧口径实测：全量 10,000 目标 × 10 站 × 6 天，全部通过 Skyfield 逐秒校验 0 违规）

<!-- frontier-table:begin -->
| 档位 | 命令 | 全程用时（秒） | 综合 | 覆盖 / 得分 | 平均占用核数 | Skyfield 逐秒校验 |
|---|---|---:|---:|---:|---:|---|
| 快速（不跑弹射链） | `python -m obsplan plan --mode fast --chain 0` | 53（45–57） | 0.9690（0.9690–0.9690） | 9987 / 28238 | 2.1 | 3 次全部合法 |
| 快速 + 30 秒 | `python -m obsplan plan --mode fast --chain 0.5` | 54（53–54） | 0.9886（0.9885–0.9886） | 9999 / 29380 | 3.1 | 3 次全部合法 |
| 推荐档（默认，约 2 分钟） | `python -m obsplan plan` | 96（96–101） | 0.9913（0.9913–0.9914） | 9999 / 29539 | 3.2 | 3 次全部合法 |
| 快速 + 3 分钟 | `python -m obsplan plan --mode fast --chain 3` | 202（202–202） | 0.9943（0.9940–0.9943） | 9999 / 29720 | 3.8 | 3 次全部合法 |
| 十分钟档（快速 + 9 分钟） | `python -m obsplan plan --mode fast --chain 9` | 562（562–562） | 0.9955（0.9954–0.9955） | 9999 / 29792 | 3.7 | 3 次全部合法 |

每档从启动进程到写出计划端到端跑 3 次取中位数（括号里是最小–最大），可见窗口已缓存，用时不含 Skyfield 校验；测速机器 macOS-27.2.1-arm64-arm-64bit，10 核。原始记录在 `results/frontier/`，本表由 `scripts/frontier_table.py` 生成。
<!-- frontier-table:end -->

- **6 天旧口径的离线最好计划**（随仓库提供：`方案H初版/outputs/final_best/longrun_chain_best.csv`，支持 `python -m obsplan validate` 一条命令复核）：覆盖 **9,999 / 9,999** 颗可观测目标（另 1 颗 SGP4 外推异常），得分 **29,924.8**，综合指标 **0.9977**（`0.997667`），离可证明按秒数学上界 **1.0047**（附 10 秒细网格对偶价格证书）差距 **≤ 0.70%**。

### 快速运行（评委一键启动）

进入 `方案H初版/` 目录：
- **网页界面（推荐）**：Windows 双击 `start_app.bat`、macOS 双击 `start_app.command`、Linux 运行 `bash start_app.sh`（仓库已预载推荐档与最好计划，打开即看）。
- **命令行规划与校验**：
  ```bash
  uv sync --extra app
  uv run python -m obsplan plan --validate                                       # 推荐档（7 天，约 2 分钟）
  uv run python -m obsplan score outputs/final_best/submission_7d_edge5.json --shrink 0,5   # 正式提交文件（7 天，留 5 秒余量，35912.3，0 错误）
  uv run python -m obsplan validate outputs/final_best/best_7d_edge5.csv         # Skyfield 逐秒复核正式计划
  uv run python -m obsplan plan --edge-margin 5 --mode fast --chain 9            # 从头生成同样设置的计划（约 10 分钟）
  uv run python -m obsplan export 某计划.csv -o 某计划.json                        # 任何计划导出成主办方格式 v3.0
  uv run python -m obsplan validate outputs/final_best/longrun_chain_best.csv --start 2026-08-01 --days 6   # 6 天旧口径的最好计划（0.9977）
  ```
  `plan` 也会在 `outputs/plan/` 同时写出提交文件 `plan.json`；规范差异与待确认问题见 [规则口径.md](规则口径.md) 第四节。
- **Docker**：见 [方案H初版/README.md](方案H初版/README.md#评委运行指南)。
