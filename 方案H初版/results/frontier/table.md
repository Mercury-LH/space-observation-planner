| 档位 | 命令 | 全程用时（秒） | 综合 | 覆盖 / 得分 | 平均占用核数 | Skyfield 逐秒校验 |
|---|---|---:|---:|---:|---:|---|
| 快速 + 30 秒 | `python -m obsplan plan --mode fast --chain 0.5` | 54 | 0.9862 | 9999 / 29238 | 2.4 | 1 次全部合法 |
| 快速（不跑弹射链） | `python -m obsplan plan --mode fast --chain 0` | 59 | 0.9690 | 9987 / 28238 | 1.7 | 1 次全部合法 |
| 推荐档（默认，约 2 分钟） | `python -m obsplan plan` | 103 | 0.9899 | 9999 / 29456 | 2.5 | 1 次全部合法 |
| 快速 + 3 分钟 | `python -m obsplan plan --mode fast --chain 3` | 242 | 0.9905 | 9999 / 29495 | 1.6 | 1 次全部合法 |
| 十分钟档（快速 + 9 分钟） | `python -m obsplan plan --mode fast --chain 9` | 565 | 0.9940 | 9999 / 29702 | 2.1 | 1 次全部合法 |

每档从启动进程到写出计划端到端跑 1 次取中位数（括号里是最小–最大），可见窗口已缓存，用时不含 Skyfield 校验；测速机器 macOS-27.2.1-arm64-arm-64bit，10 核。原始记录在 `results/frontier/`，本表由 `scripts/frontier_table.py` 生成。
