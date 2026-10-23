@AGENTS.md

# 运行说明

- 当前主线是 `方案H初版/`（见其 README）。安装和运行都在该目录执行：`uv sync`，`uv run python run_compare.py`（`--no-validate` 跳过 Skyfield 校验，`--h-iters` 调 H 的迭代次数）。
- 赛题数据是仓库根目录的 `data/数据.zip`、`data/优先级.zip`，程序直接读取，不用解压。
- 首次运行会计算可见窗口并缓存到 `方案H初版/cache/`。Skyfield 校验要用 DE421 星历 `cache/de421.bsp`：本地从 `ssd.jpl.nasa.gov` 自动下载；云端访问不了该域名，由启动脚本从 PyPI 的 `skyfield-data` 包取出同一文件。拿不到时告诉用户，不要换星历或跳过校验来绕过。

# 云端会话（`CLAUDE_CODE_REMOTE=true`）

- 会话启动时 `scripts/cloud_setup.sh` 已自动执行 `uv sync` 并准备好 DE421；它报失败时先看 `/tmp/cloud_setup.log`。`.claude/settings.json` 把单条命令的最长超时设为 90 分钟。
- 云端机器 4 核，本地是 10 核 M1 Pro。完整对比（含 CP-SAT 和 8 份计划的 Skyfield 逐秒校验）本地约 30 分钟，云端可能超过 1 小时。
- 会话空闲几分钟后虚拟机会暂停，后台进程也跟着停。长任务必须在同一轮回复里跑完：放后台写日志（`nohup uv run python run_compare.py > results/run.log 2>&1 &`），然后在这一轮里反复查看日志直到结束，不要启动后就结束回复。
- 只改方案 H 时，先用 `--no-validate` 快速看指标，确认有提升再跑带校验的完整对比。
- 做完后把代码和 `results/compare.json` 提交并推送到当前分支，回复里写清：改了什么、跑了哪些命令、结果数字、是否通过 Skyfield 校验。用户会回到本地电脑 `git pull` 继续。
