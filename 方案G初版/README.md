# 方案G初版：AI筛选 + CP-SAT自主观测规划

本目录是面向空间目标持续监测赛题的方案G初版实现。核心思路是让AI评分器从大规模候选观测任务中识别高价值候选，再由CP-SAT在硬约束下完成全局组合优化，最后通过候选回流、局部搜索和独立校验得到合法计划。

## 方案流程

```text
TLE、站址、优先级
  -> SGP4与Skyfield合法可见窗口
  -> 120/240秒候选任务
  -> AI任务价值评分
  -> 安全Top-K筛选
  -> CP-SAT第一轮优化
  -> 候选回流与第二轮优化
  -> 局部搜索
  -> 独立约束校验
  -> schedule.csv与metrics.json
```

## 目录

- `observation_system/`：数据读取、窗口计算、候选生成、规划器适配、结果导出。
- `scheme_g/`：方案G评分、筛选、CP-SAT、回流、局部搜索和校验器。
- `docs/`：阶段性压力测试报告。
- `数据/`、`优先级/`：本地比赛数据放置位置；原始数据不随Git提交。

方案G的可替换边界位于 `observation_system/planners/scheme_g_adapter.py`，其他规划器通过 `observation_system/planners/registry.py` 注册。

## 环境安装

在本目录执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 数据放置

将获授权的比赛数据按以下结构放置：

```text
数据/
├── 轨道.txt
└── 站址.txt

优先级/
├── 高优目标.txt
├── 中优目标.txt
└── 低优目标.txt
```

如实际文件名不同，请修改 `observation_system/config/default.json`。

## 测试

```powershell
.venv\Scripts\python.exe -m unittest discover -s scheme_g\tests -v
.venv\Scripts\python.exe -m unittest discover -s observation_system\tests -v
```

未放置比赛数据时，真实数据审计测试会自动跳过，其余单元测试仍可运行。

## 数据审计

```powershell
.venv\Scripts\python.exe -m observation_system inspect `
  --config observation_system\config\default.json
```

## 小规模运行

```powershell
.venv\Scripts\python.exe -m observation_system run `
  --config observation_system\config\default.json `
  --planner scheme_g `
  --limit-targets 100 `
  --limit-stations 3 `
  --output-dir outputs\stress_100x3 `
  --force-windows
```

## 规划器对比

```powershell
.venv\Scripts\python.exe -m observation_system compare `
  --config observation_system\config\default.json `
  --planners random_greedy priority_greedy ai_greedy scheme_g `
  --limit-targets 100 `
  --limit-stations 3
```

## 已完成压力测试

- `100 × 3`：920个窗口、36,800条原始候选、173次观测；全部60颗具有合法窗口的目标均被覆盖，约束错误为0。
- `1000 × 5`：14,759个窗口、589,332条原始候选、1,414次观测；全部778颗具有合法窗口的目标均被覆盖，约束错误为0。

详细记录见：

- [100 × 3压力测试](docs/STRESS_REPORT_100x3.md)
- [1000 × 5压力测试](docs/STRESS_REPORT_1000x5.md)

上述结果是当前开发假设和本地数据下的工程验证结果，不代表官方最终评测分数。完整10000目标 × 10站仍需进一步进行分层或滚动时间域优化与全量验证。

## 当前AI定位

AI评分模型使用优先级、任务收益、机会稀缺性、紧迫度、效率、站点灵活性、冲突压力和资源占用等特征对候选任务排序。CP-SAT负责执行全部硬约束，AI评分不能替代合法性校验。

当前仓库包含默认权重和遗传搜索调参程序；默认权重可直接运行，正式比赛前应使用真实训练/验证场景完成权重训练和消融对比。
