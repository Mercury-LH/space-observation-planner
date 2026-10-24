# 时间加权优化与统一评测报告 · 2026-10-24

在当前固定实例、AC 低电量模式与质量∶效率=4∶3的内部评价下，**推荐 `hplus_1_fast`**：两次独立运行均覆盖 **9911** 个目标、原始观测分 **27987.64**、Q **0.961049**，完整求解耗时中位数 **72.4645 秒**，内部时间加权分 **81.6383**。只看质量时，三源融合 `fusion_baseline` 更高：覆盖 **9998**、Q **0.978157**，但求解需要 **1515.10 秒**。

H1 相比融合基准少覆盖 87 个目标、观测分低 767.06、Q 相对低约 1.75%；完整求解耗时减少约 95.22%（约 20.91 倍速度）。这是本次同条件样本的取舍，不是跨平台速度保证。包含独立校验的端到端时间分别为 **179.37 秒**和 **1626.72 秒**，不能把 72.46 秒当成完整交付等待时间。主表只采用加入供电检查后的 `confirmation-v4` 与 `winner-repeat-v4`；此前探索结果不参与最终排名。

实现位于 `方案H初版/`，实测代码固定为提交 [`2e41904`](https://github.com/Mercury-LH/space-observation-planner/commit/2e419040557a61d6e0f7b0777322208f5b3ea3f2)。对应 [Planner evaluation contract CI](https://github.com/Mercury-LH/space-observation-planner/actions/runs/40066759739) 成功，**128 项测试通过，2 条 warnings，6.54 秒**。CI 检验接口与规则契约，不代替下列本机全量实验。文档和公开结果后续提交不会改变这次实测记录中的源码版本。

## 题目、规则与评价口径

当前程序解决的是空间目标观测任务规划：10000 个目标、10 个站，规划期统一为 **2026-08-01 00:00:00 至 2026-08-07 00:00:00 UTC**。30 秒是候选构造的时间网格，最终任务按整数秒精修。其他赛题“全球覆盖与区域重访星座构型设计”的直接 Access、沿迹视场 0.81° 公告不加入本题模型。

统一约束包括：全程仰角 10°–85°、站点太阳高度角低于 −6°、目标受照、单站资源互斥、30 秒切换，以及同一目标上一次结束到下一次开始至少 48 小时。标准求解器生成 120–240 秒任务；正式计分在 240 秒饱和，不把这点另行解释为官方禁止更长观测。未明确的容差、球形地影等实现假设仍以仓库根目录 `规则口径.md` 为准。

本实例内部质量指标为：

```text
Q = 0.5 × covered / 10000 + 0.5 × observation_score / 30062
```

`covered` 为合法覆盖的不同目标数；`observation_score` 累加满足条件的观测，允许同一目标间隔 48 小时后的重复得分。单次 120 秒取得优先级分的 60%，120–240 秒线性增加；优先级分为高 5、中 3、低 2。

已核对《说明.docx》及 `规则口径.md` 第 18 项：计算结果占 40%、产品体验占 30%、算法效率占 30%，另有最多 10% 加分。固定产品体验时，我们把质量与效率按 **4∶3** 归一化，令时间权重 `w=3/7`：

```text
speed(T) = 120 / (120 + T)
J(w) = 100 × ((1 − w) × Q + w × speed(T))
```

120 秒是事先固定的团队时间尺度，不随候选集合变化。`T` 包含所有来源计划的生成、训练、搜索、融合与秒级精修。预处理、独立校验、进程端到端时间另列；如果评判的是首次运行等待时间，应同时参考端到端耗时。

**Q 和 J 都是内部指标。** 主办方尚未公布效率分与运行秒数的换算函数，`4∶3` 不能把这个代理公式变成官方评分。允许重复观测使 Q 理论上可超过 1，J 也不承诺以 100 为上限。报告同时展示原始覆盖、观测分、时间和 `w=0、0.1、0.2、0.4、3/7` 的敏感性。

## 方法到底做了什么

所有候选都从相同数据、配置和窗口重新生成，调用：

```python
solve(pb, spec, progress=None) -> (Schedule, info)
```

`pb` 是公共 `Problem`；`Schedule.station/target` 使用公共数据集零基索引，`start_s` 相对规划起点，`dur_s` 为持续秒数，两者均为有限整数秒。`info` 返回实际参数、各阶段耗时和伐谋状态。统一入口是 `obsplan.portfolio:solve`；独立算法可注册相同契约的受信任 `module:function`，仍要通过同一校验器。

- `fusion_baseline`：GA 训练一次并共享给 E、G；生成 E 与 G+，运行最多 300 轮 H+ 及完整局部搜索，执行三轮来源融合和局部搜索，最后秒级精修。所有来源生成都计入求解时间。
- `hplus_1_fast`：只保留 H+ 来源，`h_iters=1`、`h_fast=true`，关闭融合。`lagrange.solve` 从全零站点价格开始；第一次恢复得到的 `best_mu` 仍为零，随后即使用该价格和已采纳的伐谋规则再次恢复，再做快速局部搜索、秒级精修及临界短窗口补入。代码会计算一次子问题，并可能更新 `mu`，但这一更新不会成为此次输出使用的 `best_mu`。因此应准确称为**零初始价格下的伐谋规则恢复与精修基线**，不能称作充分收敛的价格协调方法。
- `hplus_10_fast`：同一条流水线，允许最多 10 轮价格迭代；是否改善必须看最终合法计划，不能仅凭轮数判断。
- `hplus_300_full`：最多 300 轮 H+，启用完整局部搜索，不加入 G/E 或融合，用来测量更多 H 计算的边际收益。

`fast` 没有删除全部局部搜索；它将其中“挤让插入重复观测”限制到高、中优先级目标。秒级精修在真实窗口边界内调整任务起止和时长，以线性规划重新分配时长，并尝试补入 30 秒构建网格遗漏的临界可观测目标。最后由独立 Skyfield 检验所有输出任务。

## 伐谋调用与 G 耗时

默认实际链路为：

```text
portfolio.solve
  → lagrange.solve(rule=load_rule())
  → obsplan/evolved/recover_rule.py 的 score
  → 本地可行恢复、局部搜索和精修
  → 独立 Skyfield 校验
```

这是执行此前通过百度伐谋实验演化、审阅并采纳的本地 Python 规则，**本轮默认实时云端调用数为 0，评测不需要 API key**。规则仍在实际解题的候选排序环节发挥作用，不能描述为“每次计划都有在线伐谋 agent”。继续云端演化是另一个明确触发的流程：`scripts/famou_ctl.py` 创建实验、生成候选、本地 evaluator 回传、下载复评后采纳；本次 benchmark 不自动启动它。

人工贪心与固定优先级贪心只作为对照；它们没有使用伐谋，即使内部时间效用高，也不能据此当作满足赛题伐谋使用要求的提交方案。已采纳规则来源与历史采纳检查见 `obsplan/evolved/recover_rule.py` 头部和 `integrations/famou_h*/`；这不是对其他实例泛化有效的证明。

G 慢主要来自滚动优化的重复成本。默认每次看 4 小时、提交 3 小时，144 小时约有 48 个 CP-SAT 块；每块 10 秒搜索上限已对应约 480 秒，还要加候选构造、建模与局部搜索。三来源之后又进行融合建池、建模、搜索与局部搜索。

本次 AC1 确认集的融合基准完整求解 **1515.10 秒**，其中 G 构造 **487.23 秒**、G 局部搜索 **31.00 秒**（合计 **518.23 秒**），三轮融合求解合计 **598.98 秒**，融合后的局部搜索合计 **133.11 秒**；此外还包括 GA **44.91 秒**、H+ **167.08 秒**等阶段。数值来自同一次运行的 [`stage_timings_s`](confirmation/confirmation-v4/001-fusion_baseline-r1/result.json)，不能把某一阶段的上限当作整个方法耗时。G 的限时搜索与其他阶段的计算行为不同，也不能假设切换功耗后各阶段等比例加速。

## 探索记录与供电混杂

旧批次均按其 manifest 指定的原始 `run/result.json` 读取。`full-v1/summary.json` 后来曾合入 `iteration-v2` 并重算权重，因此不能把该派生总表误认为一个独立的 12 方法批次。

以下仅用于说明筛选过程，**不构成跨批次速度排名或最终推荐**：

| 探索批次 | 方法 | 合法覆盖 | 原始观测分 | Q | 求解秒 |
|---|---|---:|---:|---:|---:|
| full-v1 | fusion_baseline | 9995 | 28743.55 | 0.977821 | 1153.83 |
| full-v1 | hplus_fast（150轮） | 9987 | 28237.78 | 0.969009 | 51.22 |
| iteration-v2 | hplus_1_fast | 9911 | 27987.64 | 0.961049 | 34.27 |
| iteration-v2 | hplus_10_fast | 9933 | 28090.59 | 0.963861 | 36.26 |
| iteration-v2 | hplus_300_full | 9994 | 28538.39 | 0.974359 | 91.29 |
| repeat-v3（2次） | hplus_1_fast | 9911 | 27987.64 | 0.961049 | 68.06–70.15 |
| repeat-v3（2次） | hplus_10_fast | 9933 | 28090.59 | 0.963861 | 73.71–74.72 |

H1/H10 的质量保持一致，但后续耗时约翻倍。排查时实际观察到机器当前使用 AC，且 macOS 的配置为电池 `lowpowermode=0`、AC `lowpowermode=1`。这支持“供电源或功耗模式变化可能影响速度”的推断，但旧 worker 没有逐次记录 `power_profile`，因此不能确认历史上何时发生供电切换，也不能确定每条旧运行的模式。不能事后补造功耗标签，或从这些数据单独分离电源、热状态与其他负载的因果贡献。

因此 `full-v1`、`iteration-v2`、`repeat-v3` 仅保留为各自的探索证据，不跨模式混排、不把旧 J 当最终成绩。正式确认采用新代码和固定供电模式，整个进程每个阶段边界留下 `power_profile`。开始、求解后或验证后任一快照不一致，标记 `environment_changed`，保留原始计划与质量结果但不排名。非 macOS/读取失败为 `unknown`，不等于低耗模式关闭；采样也不检测两次采样之间切换又切回或热降频。

## AC1 主比较：完整确认集与赢家复跑

`confirmation-v4.json` 对四个方法各运行一次，种子 2026、8 线程、单进程超时 3600 秒；`winner-repeat-v4.json` 对 H1 额外运行一次，种子 2027、超时 1200 秒。共 **5 次运行全部通过独立 Skyfield 1 秒校验**，没有违规、超时或运行失败，边界快照未观察到供电模式变化；各次开始、求解后、校验后快照均为 `source=AC, low_power_mode=1`，实时云调用均为 0。

表中 H1 为两次结果的中位数，其他方法各一次；J 按每次运行先计算、再取中位数，没有把一次运行复制为多个样本。预处理约 0.93–1.18 秒；窗口及临界短窗口缓存均为热缓存。Numba 磁盘缓存存在，但不据此保证所有函数签名已预热。

| 方法 | 有效次数/总数 | 覆盖 | 原始观测分 | Q | 求解秒中位数［范围］ | 校验秒中位数 | 端到端秒中位数 | J(3/7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion_baseline | 1/1 | 9998 | 28754.70 | 0.978157 | 1515.1003 | 110.26 | 1626.72 | 59.0400 |
| hplus_1_fast | 2/2 | 9911 | 27987.64 | 0.961049 | 72.4645 ［71.9332–72.9959］ | 105.36 | 179.37 | 81.6383 |
| hplus_10_fast | 1/1 | 9933 | 28090.59 | 0.963861 | 79.2415 | 105.23 | 186.04 | 80.8899 |
| hplus_300_full | 1/1 | 9994 | 28538.39 | 0.974359 | 176.9719 | 106.15 | 284.69 | 72.9953 |

H1 两次求解分别为 71.9332、72.9959 秒，覆盖数、原始观测分与 Q 完全一致；端到端分别为 175.82、182.92 秒。两个种子并不等于两个不同问题实例，也不足以建立统计稳定性。H10、H300 和融合目前各一次，限时多线程 CP-SAT 仍可能产生波动。

权重敏感性保留以下全部数值，避免只挑有利权重：

| 时间权重 w | 融合基准 J | H1 J | H10 J | H300 J | 本组最高 |
|---:|---:|---:|---:|---:|---|
| 0.0 | 97.8157 | 96.1049 | 96.3861 | 97.4359 | fusion_baseline |
| 0.1 | 88.7680 | 92.7293 | 92.7703 | 91.7331 | hplus_10_fast |
| 0.2 | 79.7203 | 89.3538 | 89.1546 | 86.0303 | hplus_1_fast |
| 0.4 | 61.6250 | 82.6028 | 81.9230 | 74.6247 | hplus_1_fast |
| 3/7 | 59.0400 | 81.6383 | 80.8899 | 72.9953 | hplus_1_fast |

只看质量（w=0）选择融合；w=0.1 时 H10 略高于 H1，差约 0.041，当前样本不足以称为稳定优势；w=0.2、0.4 和默认 3/7 时 H1 最高。四种方法都在这四个候选的质量—时间 Pareto 前沿：没有一个方法同时质量更低且更慢。因此“最佳”应附带权重、平台、功耗模式和已测候选范围，不能宣称全局最优。

固定主比较权重 3/7 后，将时间尺度改为 30、60、120、300、600、1200 秒的六种敏感性对照中，H1 均排名第一，见 [time-scale-sensitivity.json](time-scale-sensitivity.json)。但 1200 秒尺度下 H1 相对 H10 仅高约 0.0534，不能据此声称排序对运行波动稳定。

机器为 **Apple M5、arm64、10 逻辑核、16 GiB 内存，Darwin 27.0.0**；求解固定 8 线程。Python **3.12.13**，NumPy **2.5.4**、SGP4 **2.27**、Numba **0.68.0**、OR-Tools **9.15.6755**、Skyfield **1.55**。输入为 10000 目标、10 站，无数据读取问题；1 个目标轨道传播失败而不规划，覆盖分母仍保留全部 10000。

核心复现指纹：

| 项目 | 值 |
|---|---|
| Git revision | `2e419040557a61d6e0f7b0777322208f5b3ea3f2` |
| 公共源码 SHA-256 | `2431d9a2bbbf1cb29edead3eeb794c85d4465680e9de5587b14acf61c755ec8a` |
| 有效输入 SHA-256 | `859f10bbc62c1814a4817c0ecea40ed05f9d1211545a4a9df8516c53b8a21dce` |
| 规则配置 SHA-256（种子单列） | `f67f1c44290b8eb7cfaf25ca57fb331115a111f55dc4412abe414e59d285b7ea` |
| 比较指纹（含 AC1 环境） | `fdf35891ab544c0619d2c92738dd24358961e22d3cd655bdb22048a60d292997` |

主表机器可读数据见 [summary.json](summary.json)，两批逐次结果分别保存在 [confirmation/confirmation-v4](confirmation/confirmation-v4/) 和 [confirmation/winner-repeat-v4](confirmation/winner-repeat-v4/)。质量—时间图见 [tradeoff.png](tradeoff.png)，整轮记录见 [campaign.json](campaign.json)。

## 正常功耗复核：电池供电、低电量模式关闭

AC1 主比较与赢家复跑完成后，按用户授权临时通过系统设置将低电量模式设为 Never，`pmset` 核实初始状态为 AC、`lowpowermode=0`。随后真实供电源发生变化，`normal-power-v5` 没有保持整批相同环境：

- `001-hplus_1_fast-r1`：开始、求解后、校验后三次均为 AC0，计划合法，保留为 AC0 单次证据，不放入电池组。
- `002-hplus_10_fast-r1`：开始与求解后为 AC0，校验后变为 Battery0。独立几何校验 `legal=true`，但状态为 `environment_changed`，不进入时间排名。
- `003-hplus_10_fast-r2` 与 `004-hplus_1_fast-r2`：三次快照均为 Battery0，种子 2027。

因此 v5 的整批 CLI 报告拒绝聚合，提示 `Refusing to compare different inputs/configs/code/environments`。这是正确触发的保护，不是计划丢失，也不是允许手动删除环境字段后混排的理由。全部 manifest 和逐次原始结果保存在 [normal-power-v5](normal_power/normal-power-v5/)。

随后使用 `benchmarks/normal-power-repeat-v6.json`，在 Battery0 下给 H1、H10 各补跑一次，种子 2028。最终正常功耗组按**环境条件**选取：纳入 v5 与 v6 中三次快照全部为 `source=Battery, low_power_mode=0` 的全部运行，然后要求所有纳入运行合法、可排名且没有 worker 退出异常。筛选没有参考分数或耗时，也没有先丢弃不利结果。完整规则、四次纳入运行和两次未纳入记录见 [cohort.json](normal_power/cohort.json)。

| Battery0 方法 | 合法次数/纳入次数 | 覆盖 | 原始观测分 | Q | 求解秒中位数［范围］ | 校验秒中位数 | 端到端秒中位数 | J(3/7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hplus_1_fast | 2/2 | 9911 | 27987.64 | 0.961049 | 38.7711 ［38.1306–39.4116］ | 67.69 | 107.29 | 87.3092 |
| hplus_10_fast | 2/2 | 9933 | 28090.59 | 0.963861 | 43.6537 ［42.1069–45.2004］ | 69.41 | 114.05 | 86.5058 |

两种方法在各自两次 Battery0 复跑中的 Q、覆盖和观测分均相同。默认权重 3/7 下仍由 H1 领先；w=0、0.1 时 H10 领先，w=0.2、0.4 时 H1 领先。两者仍各有质量或速度优势，均为这个两方法集合的 Pareto 候选。完整数值见 [normal_power/summary.json](normal_power/summary.json) 与 [摘要表](normal_power/summary.md)。

这张表只比较稳定 Battery0 的四次运行，**不与 AC1 主表拼接排名**，也不纳入缺少功耗快照的旧实验。正常功耗下没有重跑慢速融合，不能声称融合与 H1/H10 已在 Battery0 下完成对照。跨环境耗时变化只说明环境敏感性，不能记作算法改进收益。两组使用相同公共代码、输入、规则、线程与依赖；电池组比较指纹为 `1c6504013ad007d38fcaa852858a8106cca36d098205862060c20b8e426e85f7`。

本轮共保留 **27 次运行、12 个方法配置**：26 次状态为 `valid`，1 次为 `environment_changed`；**27 次独立几何校验均合法**。其中 AC1 主表 5 次，正常功耗原始记录 6 次、按环境纳入排名 4 次，其余为探索记录。几何合法不代表可以忽略供电变化后进行速度排名。

**原节能配置已恢复并核实。** 系统设置恢复为 Only on Power Adapter（仅使用电源适配器时），`pmset` 验证 AC `lowpowermode=1`、Battery `lowpowermode=0`，恢复核验时间为 **2026-10-24T06:28:20.825278Z**。核验时实际供电仍是 **Battery0**；恢复的是原设置，并不表示物理供电恢复为 AC。操作授权、切换与恢复证据见 [power-transition.json](normal_power/power-transition.json)。benchmark 始终只读取系统状态，不自行更改这些设置。

## 队友及 AI 的复现步骤

先读 `方案H初版/AGENTS.md`、`docs/自动评测与协作.md` 和根目录 `规则口径.md`。在已发布的固定 revision 上执行，正式计时期间不运行测试、其他算法或代码编辑，不自动修改系统供电设置。队友先跑推荐 H1 的单方法配置即可确认接口；需要更高质量且不想承担慢融合成本时，再试 H300。以下两项与完整确认集的输出目录分开：

```bash
cd 方案H初版
uv sync --frozen
uv run python -m obsplan.benchmark run \
  --suite benchmarks/recommended.json --output outputs/benchmarks/recommended
# 可选：只跑 H300 高质量对照；它不是本次质量最高的融合方案。
uv run python -m obsplan.benchmark run \
  --suite benchmarks/quality-hplus.json --output outputs/benchmarks/quality-hplus
```

完整本轮复现命令（从仓库根目录开始）：

```bash
cd 方案H初版
uv sync --frozen

# 合成接口回归；与正式耗时实验分开执行。
uv run python -m pytest -q \
  tests/test_portfolio.py tests/test_benchmark.py tests/test_benchmark_power.py \
  tests/test_benchmark_baselines.py tests/test_validator_contract.py tests/test_config_time.py

# macOS只读查看供电源及两种供电的配置；其他平台保存可获得的系统设置。
pmset -g batt
pmset -g custom

# 每次必须使用不存在的新输出目录。
uv run python -m obsplan.benchmark run \
  --suite benchmarks/confirmation-v4.json \
  --output outputs/benchmarks/confirmation-v4

# 已确认的赢家 H1 额外复跑一次；配置显式使用 seed=2027。
uv run python -m obsplan.benchmark run \
  --suite benchmarks/winner-repeat-v4.json \
  --output outputs/benchmarks/winner-repeat-v4

# 仅合并相同代码、输入、规则、线程、依赖和供电模式的确认记录。
uv run python -m obsplan.benchmark report outputs/benchmarks/confirmation-v4 \
  --include outputs/benchmarks/winner-repeat-v4 \
  --time-weight 0.42857142857142855 --time-scale-s 120

# 正常功耗 v5：本次实际从 AC0 开始，过程中切为 Battery0。
# 混合环境会在全部 worker 结束后拒绝整批聚合；逐次结果仍保存。
uv run python -m obsplan.benchmark run \
  --suite benchmarks/normal-power-v5.json \
  --output outputs/benchmarks/normal-power-v5

# 在已核实稳定的 Battery0 下，各补跑一次；配置明确使用 seed=2028。
uv run python -m obsplan.benchmark run \
  --suite benchmarks/normal-power-repeat-v6.json \
  --output outputs/benchmarks/normal-power-repeat-v6
# 执行者完成后恢复原 AC1/Battery0 配置，并另行记录实际供电源。
```

本次 v5 已是混合批次，不能直接用 `report --include` 合并整批。要从保存的原始记录重建同环境摘要，可按下面的显式条件纳入全部 Battery0 样本，再检查合法性。对于新的复现实验，应保持整批供电稳定，不必刻意重现切换事件。

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from obsplan.benchmark import aggregate, write_json

profile = {"source": "Battery", "low_power_mode": 0}
checkpoints = {"start", "after_solver", "after_validation"}
selected = []
for name in ("normal-power-v5", "normal-power-repeat-v6"):
    folder = Path("outputs/benchmarks") / name
    manifest = json.loads((folder / "manifest.json").read_text())
    for item in manifest["runs"]:
        run = json.loads((folder / item["run_id"] / "result.json").read_text())
        assert run["suite_id"] == manifest["suite_id"] and run["run_id"] == item["run_id"]
        checks = run.get("power_profile_checks", {})
        if set(checks) == checkpoints and all(value == profile for value in checks.values()):
            selected.append(run)  # Environment only; no score/time/validity pre-filter.
assert len(selected) == 4
assert all(r["status"] == "valid" and r["rankable"] and not r.get("worker_exit") for r in selected)
summary = aggregate(selected, weight=3/7, time_scale_s=120)
write_json(Path("outputs/benchmarks/battery0-summary.json"), summary)
PY
```

`summary.json` 和 `report.md` 为派生聚合表；`manifest.json` 记录批次；每个 `result.json` 保留原始指标、实际参数、代码/输入/配置指纹、种子、线程、平台依赖、阶段耗时及供电快照。窗口和临界短窗口使用带数据/配置/几何代码/依赖指纹的缓存，冷热条件另记；Numba 磁盘缓存存在不保证所有签名已预热，实际 JIT 成本不从求解计时中扣除。最终校验默认 Skyfield 1 秒、含任务端点；它是独立数值复核，不是连续时间解析证明。

新方法只能在 `solve` 内进行候选生成、训练和搜索；模块顶层不得预计算、读取旧计划或调用云端来隐藏时间。标准适配器当前限定 30 秒构造网格、120–240 秒动作、30 秒切换、48 小时重复间隔以及质量各 0.5、`sumP` 归一化。修改公共代码/数据/规则后重新建立基准，同名方法改变参数应改试验名称。

不要提交 `outputs/`、`cache/`、完整 `plan.csv`、目标明细或凭据。公开发布配置、源码、相关测试、清洗后的摘要和本报告；CI 使用 2 线程且可能无法读取功耗模式，其耗时不能与本地 8 线程结果混排。当前结论只覆盖已测配置和固定实例，不能承诺全局最优、官方名次或跨日期/站点/目标集的泛化。

## 证据索引

- 规则：根目录 `规则口径.md` 第 1、3、5–14、17–18 项及待确认表；`docs/自动评测与协作.md`。
- H1含义：`obsplan/lagrange.py` 的 `mu=0`、首次恢复保存 `best_mu`、循环后规则恢复；`obsplan/portfolio.py` 的 H+ 调用与统一 `finish`。
- 伐谋：`obsplan/evolved/recover_rule.py` 来源与本地 `score`；各运行 `famou.runtime_cloud_calls`。
- G成本：[confirmation-v4 融合基准原始摘要](confirmation/confirmation-v4/001-fusion_baseline-r1/result.json) 的 `stage_timings_s`。
- 探索：`exploration/{full-v1,iteration-v2,repeat-v3}/` 中各 manifest 所列原始 run/result；不采用已被跨批次合并的旧总表排名。
- 最终确认：[summary.json](summary.json)、`confirmation/confirmation-v4/`、`confirmation/winner-repeat-v4/`；正常功耗单列于 `normal_power/`。
