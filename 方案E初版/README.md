# 方案 E 初版

面向第四题「面向空间目标持续监测的 AI 自主观测任务规划与目标状态分析」的可运行初版，发布日期为 2026-10-20。

本目录可独立安装，包含真实数据加载、轨道传播、可见窗口、贪心组合、ALNS、CP-SAT 局部诊断、独立校验，以及百度伐谋混合云接入和一次演化后选中的算法。动态状态更新、重规划闭环和产品界面尚未完成。

## 先读什么

1. **[从题目到实现的完整说明](docs/方案E完整说明.md)**：题意、规则与假设、公式、算法选择、实验结果、伐谋的作用及后续工作。
2. [运行手册](docs/运行手册.md)：配置、输出格式、几何审计、局部修复和复现细节。
3. [历史实验汇总](results/README.md)：可分享的指标摘要；不包含原始数据和完整任务列表。
4. [伐谋接入说明](integrations/famou/README.md)：离线运行选中候选，以及另行开启云端实验的方法。
5. [本次发布验证](docs/发布验证.md)：对这个独立目录实际完成的检查。

## 安装

先进入本目录，所有后续命令均在这里执行：

```bash
cd 方案E初版
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

通用依赖由 `pyproject.toml` 固定直接版本；`requirements-lock.txt` 保存历史 CPython 3.9/macOS arm64 环境的完整版本。若需复现该环境，在 editable 安装之前安装这个锁文件。基本求解不需要伐谋账号、SDK 或密钥。

当前验证环境为 macOS arm64，Python 3.9 和独立的伐谋 Python 3.12 环境。CLI 使用 Unix `resource`，候选隔离执行依赖 macOS `sandbox-exec`；尚未完成 Windows 移植，不将声明的 Python 最低版本等同于全部平台已验证。

## 放置私有数据

测试使用合成输入，不要求赛题数据。真实实验需要自行将获授权的五个 TXT 文件放到以下位置，详见 [data/README.md](data/README.md)：

```text
data/开运联合/
├── 数据/
│   ├── 轨道.txt
│   └── 站址.txt
└── 优先级/
    ├── 高优先级目标.txt
    ├── 中优先级目标.txt
    └── 低优先级目标.txt
```

也可将配置中的 `data_root` 改为自己机器上的绝对路径。相对数据路径以本方案目录为基准。原始材料、凭证、虚拟环境、缓存和运行输出由本目录自己的 `.gitignore` 排除。

## 最小真实场景

```bash
.venv/bin/python -m space_planner audit --config configs/congested_300_1h.json

.venv/bin/python -m space_planner run \
  --config configs/congested_300_1h.json \
  --output outputs/congested_300_1h_run01

# macOS：离线运行已选中的伐谋候选；无需云端账号或 SDK
.venv/bin/python scripts/famou_export_plan.py \
  integrations/famou/selected_candidate.py \
  --run-dir outputs/congested_300_1h_run01 \
  --output outputs/famou_selected_new_run
```

输出目录必须不存在或为空；再次运行时换一个新目录。先在本机生成 `manifest.json` 和窗口，再导出候选计划；不要直接搬用其他机器上包含绝对数据路径的旧 manifest。

`outputs/…/plans/*.json` 包含任务列表、指标和独立校验结果，`start/end` 为相对配置 UTC 起点的秒数。查看 `valid` 和 `geometry_checked`，不能只看程序是否生成文件。

## 扩大规模

```bash
.venv/bin/python -m space_planner run --config configs/subset_200_1day.json
.venv/bin/python -m space_planner run --config configs/congested_1000_3h.json
.venv/bin/python -m space_planner run --config configs/full_10000_6days.json
```

历史全量实验的窗口冷计算约 12 分钟，含三个 ALNS 种子和独立校验的总流程约 18 分钟；这是单次指定机器的实测量级，不是其他机器的性能承诺。选中伐谋候选尚未在全量实例完成收益验证。

## 已有结果与解释

| 场景 | 算法 | 覆盖目标 | 观测权重分 |
|---|---|---:|---:|
| 10,000 目标、10 站、6 天 | 贪心组合及三个 ALNS 种子 | 10,000 | 30,057.6 |
| 300 目标、2 站、1 小时 | 伐谋上传的初始算法 → 选中候选 | 46 → 48 | 91.4 → 111.6 |
| 1,000 目标、3 站、3 小时 | 同上，额外场景检查 | 203 → 212 | 408.6 → 518.0 |

全量得分距当前数值窗口模型的乐观上界最多 1.6；这不是物理系统最优性证明。伐谋两版本分别使用 180 和 2,500 次搜索迭代，不能据此宣称相同计算量下的算法优势。300 例达到当前单次观测假设的 48 个容量上界，仅覆盖数达到上界。

这里的分数是观测权重累计值，不是比赛百分制总分。起点、每目标最多一次、覆盖优先和中间时长保持 60% 都是显式开发假设；正式重复计分、日夜过滤及最终评分口径仍待确认。

## 目录

```text
src/space_planner/    数据、几何、求解、校验、CLI 和伐谋评测器
tests/               合成用例、约束及隔离策略测试
configs/             5 组可复现实验配置
scripts/             审计、探针、保存计划复验、伐谋 worker 和导出
integrations/famou/   云端初始算法、提示、配置及选中候选
docs/                完整说明、运行手册、方法与历史实验报告
results/             去除本机路径和原始任务的汇总证据
data/                私有输入；仅说明文件提交
outputs/             本机生成结果；不提交
```
