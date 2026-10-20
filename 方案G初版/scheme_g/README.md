# 方案G AI筛选加CP-SAT调度器

本模块实现以下流程：

```text
候选任务CSV
-> 特征计算
-> AI价值评分
-> 安全Top-K筛选
-> 贪心初始解
-> CP-SAT全局优化
-> 候选回流
-> 第二轮CP-SAT
-> 全候选池空隙回填
-> 独立约束校验
-> 结果导出
```

当前模块不负责轨道传播和可见窗口计算。前序模块只需要输出符合
`examples/candidates.csv` 字段定义的候选任务，即可直接接入方案G。

如果前序模块在同一Python进程内运行，也可以直接调用：

```python
from scheme_g import VisibilityWindow, build_candidates_from_windows

candidates = build_candidates_from_windows(
    windows,
    durations_seconds=(120, 240),
    grid_seconds=30,
    max_starts_per_window_duration=20,
)
```

## 安装

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r scheme_g\requirements.txt
```

## 生成演示数据

```powershell
.venv\Scripts\python.exe -m scheme_g demo --output scheme_g\examples\candidates.csv
```

## 运行方案G

```powershell
.venv\Scripts\python.exe -m scheme_g plan `
  --candidates scheme_g\examples\candidates.csv `
  --config scheme_g\config\default.json `
  --output scheme_g\outputs
```

## 对比算法

```powershell
.venv\Scripts\python.exe -m scheme_g compare `
  --candidates scheme_g\examples\candidates.csv `
  --config scheme_g\config\default.json `
  --output scheme_g\outputs\comparison.csv
```

## 训练AI评分权重

遗传搜索使用快速贪心作为内层评价器，避免每组参数都调用CP-SAT。

```powershell
.venv\Scripts\python.exe -m scheme_g tune `
  --candidates scheme_g\examples\candidates.csv `
  --config scheme_g\config\default.json `
  --output scheme_g\config\trained_weights.json `
  --population 24 `
  --generations 20
```

训练后在配置中把 `ai_weights_file` 指向输出文件即可。

## 候选任务CSV字段

必填字段：

- `task_id`
- `window_id`
- `station_id`
- `target_id`
- `priority`，可使用 `HIGH/MEDIUM/LOW` 或 `高/中/低`
- `window_start`
- `window_end`
- `start`
- `end`
- `base_score`

可选字段：

- `duration_seconds`，缺失时由开始结束时间计算
- `task_score`，缺失时按120到240秒线性规则计算
- `min_elevation`
- `max_elevation`
- `max_sun_elevation`

时间使用UTC ISO 8601，例如：

```text
2026-10-01T10:00:00Z
```

## 接入前序窗口模块

前序模块对一个合法窗口生成候选任务后，写入CSV。方案G不会再次判断
轨道传播，但独立校验器会检查任务是否完全落在声明的合法窗口内。

正式接入时推荐由前序模块同时保存：

- 窗口边界；
- 窗口内最小和最大仰角；
- 窗口内最大太阳高度角；
- 用于复核的轨道计算版本和时间系统。
