# 空间目标自主观测任务规划完整系统

## 系统边界

```text
┌─────────────────────────────────────────────────────────────┐
│ 前置确定性计算                                               │
│ TLE/站址/优先级 -> SGP4 -> 仰角与太阳高度角 -> 合法窗口       │
└──────────────────────────┬──────────────────────────────────┘
                           │ ObservationWindow
                           v
┌─────────────────────────────────────────────────────────────┐
│ 通用候选任务层                                               │
│ 30秒时间格 -> 120/240秒候选 -> CandidateTask                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────┐
│ 可替换规划器接口                                             │
│                                                             │
│  [方案G] AI筛选 + CP-SAT + 回流 + 局部搜索                   │
│  [基线]  随机/优先级/AI权重贪心                              │
│  [未来]  新算法只需实现 Planner.plan()                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ PlanResult
                           v
┌─────────────────────────────────────────────────────────────┐
│ 后置确定性处理                                               │
│ 独立校验 -> 指标复算 -> schedule.csv -> metrics.json         │
└─────────────────────────────────────────────────────────────┘
```

## 方案G在哪里

方案G本体位于：

```text
scheme_g/
```

完整项目只通过以下适配器调用方案G：

```text
observation_system/planners/scheme_g_adapter.py
```

规划器注册位置：

```text
observation_system/planners/registry.py
```

如需替换方案G：

1. 新建一个实现 `Planner.plan(tasks)` 的适配器；
2. 在 `registry.py` 注册名称；
3. 将配置文件中的 `planner_name` 改为新名称。

轨道、候选、校验和输出代码都不需要修改。

## 安装依赖

```powershell
cd E:\伐谋杯
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 数据审计

```powershell
.venv\Scripts\python.exe -m observation_system inspect `
  --config observation_system\config\default.json
```

## 小规模端到端测试

先用20颗目标和2个站验证：

```powershell
.venv\Scripts\python.exe -m observation_system run `
  --config observation_system\config\default.json `
  --planner scheme_g `
  --limit-targets 20 `
  --limit-stations 2 `
  --output-dir outputs\small_run `
  --force-windows
```

## 切换规划器

```powershell
# 方案G
--planner scheme_g

# AI权重贪心
--planner ai_greedy

# 固定优先级贪心
--planner priority_greedy

# 随机基线
--planner random_greedy
```

查询所有规划器：

```powershell
.venv\Scripts\python.exe -m observation_system planners
```

在完全相同的窗口和候选任务上比较多种方案：

```powershell
.venv\Scripts\python.exe -m observation_system compare `
  --config observation_system\config\default.json `
  --planners random_greedy priority_greedy ai_greedy scheme_g `
  --limit-targets 100 `
  --limit-stations 3
```

每种算法的任务表和指标会分别写入
`outputs/full_pipeline/planner_comparison/<planner>/`。

## 正式运行

配置确认后，去掉限制参数：

```powershell
.venv\Scripts\python.exe -m observation_system run `
  --config observation_system\config\default.json `
  --planner scheme_g
```

正式运行前应先小规模验证。完整数据包含10000颗目标和10个站，窗口计算与
调度规模都很大。配置中的 `visibility.workers` 控制多进程SGP4窗口计算，
默认值为8；应根据CPU核心数和内存调整。候选压缩和求解时间也需要根据实测调整。

## 输出

默认输出目录：

```text
outputs/full_pipeline/
```

文件包括：

- `windows.csv`：合法可见窗口；
- `windows.meta.json`：缓存指纹和窗口计算异常；
- `schedule.csv`：最终观测计划；
- `metrics.json`：评分、数据量、耗时、分站统计和校验结果；
- `visibility_failures.json`：单目标传播异常记录。

窗口缓存带有数据与参数指纹。修改目标数量、站点数量、规划时间或可见性参数后，
系统会自动重新计算，不会误用旧缓存。

默认累计分归一化上界按“每颗目标在6天内理论最多计分3次”设置为
`3 × 30062 = 90186`。若官方公布不同归一化公式，只需修改配置中的
`coverage_normalizer` 和 `score_normalizer`，无需修改规划器。
