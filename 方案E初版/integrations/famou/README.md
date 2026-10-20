# 百度伐谋混合云接入

伐谋负责生成调度算法候选，本地程序负责运行候选、独立约束校验和评分。
初始试验使用作者本地保存的 `outputs/congested_300_1h_run01`：
300 个目标、2 个站、1 小时。它是有容量竞争的真实数据开发实例，
不是完整六天赛题或官方验收成绩。

当前已完成一次 10 轮真实试跑。保留上传基线 `init.py`，另将经本地
确定性修复的选中候选保存为 `selected_candidate.py`：
300 目标例覆盖 46→48、得分 91.4→111.6；
未参与云端评分的 1000 目标例覆盖 203→212、得分 408.6→518.0。
全部通过独立约束与几何校验，详细限制见接入验证报告。这些是历史开发
实验结果，原始数据和历史 `outputs/` 没有随仓库发布。

## 离线复用已选中的候选

下列所有命令都从 **`方案E初版/`** 执行。先按本方案
[安装说明](../../README.md)准备 `.venv`，再按
[数据说明](../../data/README.md)放置私有数据或修改配置的 `data_root`。
首次 clone 后，必须先在本机生成一次运行记录：

```bash
.venv/bin/python -m space_planner run \
  --config configs/congested_300_1h.json \
  --output outputs/congested_300_1h_run01

.venv/bin/python scripts/famou_export_plan.py \
  integrations/famou/selected_candidate.py \
  --run-dir outputs/congested_300_1h_run01 \
  --output outputs/famou_selected_new_run
```

复用 `selected_candidate.py` 不需要伐谋 SDK、账号或 API key，也不会创建
云端实验。候选运行和独立几何校验都在本机进行。当前候选隔离执行依赖
macOS `sandbox-exec`，其他平台会拒绝运行；不能将该流程视为已经验证的
跨平台实现。

输出目录必须不存在或为空。重复执行时换一个新的输出目录，并将
`--run-dir` 指向相应的本机运行记录。不要把其他机器的历史 manifest
当成可直接复用的数据包，其中记录的数据路径可能不适用于本机。

## 新云端实验的环境与凭证

按项目提供的《伐谋 混合云SDK 功能手册 v2.5》，单独使用 Python 3.12
和 `famou-sdk==2.5.0`。只有继续创建云端实验时才需要此环境；与本方案的
`.venv` 分开安装：

```bash
python3.12 -m venv .venv-famou
.venv-famou/bin/python -m pip install -r requirements-famou-lock.txt
.venv-famou/bin/python -m pip install -e '.[dev]'
```

在 `方案E初版/` 下自行创建被 Git 忽略的 `.env.famou`，配置
`FAMOU_API_URL` 和 `FAMOU_API_KEY`，并将该文件权限设为 `0600`。
仓库不包含凭证，也不会因 clone 自动获得账号访问权限。可以改用同名
环境变量；环境变量优先于文件中的设置。服务地址为
`https://pro-service.famou.com`。

从 `方案E初版/` 检查 SDK 和账号：

```bash
python3 scripts/famou_ctl.py --version
python3 scripts/famou_ctl.py account info --json
python3 scripts/famou_ctl.py project list --json
```

包装脚本从本地文件读取凭证，不将密钥放到命令参数里，并对终端输出脱敏。
SDK 某些 API 失败仍退出 0，包装脚本会将 JSON `success=false` 转成非零退出码。

## 本地验证、预估和提交

先使用上文生成的本机运行记录。将 `FAMOU_RUN_DIRS` 设置为绝对路径，
并在本地测试和后续 worker 所在终端中保留该变量：

```bash
export FAMOU_RUN_DIRS="$PWD/outputs/congested_300_1h_run01"

python3 scripts/famou_ctl.py test \
  --config integrations/famou/config.yaml --timeout 60 --json

python3 scripts/famou_ctl.py experiment create \
  --config integrations/famou/config.yaml --dry-run --json

# 确认本地 validity=1，且预估返回具体额度并在本次授权预算内后执行：
python3 scripts/famou_ctl.py experiment create \
  --config integrations/famou/config.yaml \
  --experiment-name SpacePlannerSmoke -y --json
```

`config.yaml` 设置 10 轮，这是 SDK 2.5.0 接受的最小轮数。
本版本已忽略 `population_size` 和 `num_islands`，因此没有用它们虚构预算控制。
实测云端实验名称不接受连字符 `-`，这里使用字母数字名称。
`--dry-run` 不创建实验；必须实际返回 `estimated_credits` 和
`available_credits`，不能把缺少额度信息视作免费或已通过。

SDK 的本地测试可能把 `validity=0` 也显示为 “Test passed”。
必须检查返回的 `data.result.validity=1` 与 `geometry_checked=1`；
包装脚本在 `--json` 模式下会对非法候选返回非零退出码。

## 本地评测进程与取回结果

提交成功后，将实际返回的实验 ID 传给专用 worker：

```bash
# 从方案E初版目录执行；若换了终端，重新设置绝对路径。
export FAMOU_RUN_DIRS="$PWD/outputs/congested_300_1h_run01"
.venv-famou/bin/python scripts/famou_worker.py --experiment-id EXP_ID
```

该进程以单并发运行，拒绝云端指定的额外依赖安装，使用上述隔离评测器。
worker 会切换自己的工作目录，因此 `FAMOU_RUN_DIRS` 不应填写相对路径。
它定期检查云端状态，在完成、失败、取消或暂停时停止取任务并提交剩余结果。
日志与退出状态保存在 `outputs/famou_worker/EXP_ID/`。
实验完成以云端返回 `COMPLETED` 为准，不能只看本地进程退出码。
如果本机休眠或评测进程停止，可用同一实验 ID 再次启动。

```bash
python3 scripts/famou_ctl.py experiment status EXP_ID --json
python3 scripts/famou_ctl.py experiment leaderboard EXP_ID --json
python3 scripts/famou_ctl.py experiment results EXP_ID --rank 1 \
  --output outputs/famou_results/EXP_ID --json

# 下载后本地复评，生成现有项目格式的实际观测计划：
.venv-famou/bin/python scripts/famou_export_plan.py \
  outputs/famou_results/EXP_ID/strategy_rank01.py \
  --run-dir outputs/congested_300_1h_run01 \
  --output outputs/famou_plan/EXP_ID
```

计划保存在输出目录的 `plans/famou_candidate.json`，同目录包含候选代码、
评分和来源哈希。输出目录必须不存在或为空；失败时不发布部分有效结果。
云端榜单可能只返回两位小数的分数，最终判断应以本地复评的完整数值为准。

使用专用 worker 是因为 SDK 2.5.0 原生 worker 会在评测之前自动执行
云端依赖安装，并且常规完成状态轮询被关闭。专用进程补足这两处边界；
没有修改安装目录内的 SDK 源码。

## 候选和评分

四个核心文件：

- `init.py`：只依赖标准库的初始调度算法，供云端迭代。
- `prompt.md`：候选接口、约束、优化方向、时间预算。
- `evaluator.py`：本地入口，调用 `space_planner.famou_evaluator`。
- `problem.md`：本地任务契约与开发假设。

候选提供 `solve(instance)`，返回观测列表。输入含窗口、优先级、
规则、时间范围与固定随机种子。原始 TLE 不传给候选。
独立校验包含窗口、最短时长、同站冲突、切换时间、每目标次数及重算仰角。
非法、超时、格式错误的候选得零分。数据损坏等基础设施错误会报错，
不会被当作算法表现差。

开发评分为 `(C + S/(P+1))/(N+1)`，其中 `C` 为覆盖数、`S` 为
优先级与时长得分、`P` 为总优先级权重、`N` 为总目标数。
它保持覆盖优先的字典序，**不等于官方比赛百分制成绩**。

默认候选时间上限为每例 20 秒。可用 `FAMOU_CANDIDATE_TIMEOUT_S`
显式调整，并同步修改提示词。用 `FAMOU_RUN_DIRS` 指定已保存实例，
每个目录使用绝对路径，多个路径用系统路径分隔符连接，macOS 为 `:`。
运行目录必须有本机生成的 `manifest.json`、`windows.json`，原始输入哈希
与几何版本必须匹配。历史摘要不包含这些输入，不能作为评测运行目录。

当前隔离执行支持 macOS `sandbox-exec`；不支持的平台会拒绝运行候选，
不会自动降级成无隔离执行。候选无法访问网络、仓库、凭证、原始数据，
只能读取标准库与临时目录，只能写指定的输出文件，输出及运行时长均有限制。
macOS 当前没有候选 RAM 硬上限，因此评测使用单并发。

## 云端边界与结果

已核对 SDK 源码：混合云创建时只上传 `config.yaml`、`init.py`、
`prompt.md`；不上传 `evaluator.py`、原始数据、保存的窗口或整个仓库。
平台接收候选的数值评分。配置与提示词也不应包含密钥或私人路径。

云端完成不等于算法改进；下载的最佳候选仍需本地复评，并与已有
贪心、ALNS、局部 CP 在相同预算及独立实例上比较。接入试跑本身不证明
创新性，也不自动补齐产品界面和动态重规划功能。

接入记录及本次实际结果见
[接入验证报告](../../docs/research/2026-10-20-famou-integration.md)。
