# 方案H的伐谋接入

## 让伐谋做什么

只演化一个小函数：`init.py` 里的 `score(names, rows)`，给每个候选观测打分（规划方案中的"AI 价值函数"）。
排序后的放置、站点冲突、30 秒切换、48 小时间隔全部由本地贪心引擎保证，所以伐谋写出的代码再差也只会得低分，不会产生非法计划。

| 文件 | 作用 | 是否上传云端 |
|---|---|---|
| `init.py` | 初始程序：人工权重的线性打分（与方案 C 相同） | 上传 |
| `prompt.md` | 接口、特征含义、目标函数、限制 | 上传 |
| `config.yaml` | 混合云配置，10 轮（SDK 2.5.0 允许的最少轮数） | 上传 |
| `evaluator.py` → `obsplan/famou_eval.py` | 本地评测 | 不上传 |

评测：按固定种子抽 2 个子场景（20% 目标 × 3 站 × 6 天，规则同答疑口径；规划期跟随配置，现为 2026-08-01 至 08-07 UTC），候选代码在 macOS `sandbox-exec` 中运行（不能联网、读仓库、启动子进程；隔离方式沿用方案E初版），
用它的打分跑贪心构造，`combined_score = 0.5 × 候选综合指标 / 人工权重综合指标`，初始程序得 0.5。

## 运行

在 `方案H初版/` 下：

```bash
uv sync --extra famou
uv run --extra famou python scripts/famou_ctl.py account info --json          # 检查凭证
uv run --extra famou python scripts/famou_ctl.py test --config integrations/famou/config.yaml --json
uv run --extra famou python scripts/famou_ctl.py experiment create --config integrations/famou/config.yaml --dry-run --json
uv run --extra famou python scripts/famou_ctl.py experiment create --config integrations/famou/config.yaml --experiment-name 名称 -y --json
uv run --extra famou python scripts/famou_worker.py --experiment-id 实验ID --evaluator-path integrations/famou/evaluator.py
```

凭证放在被 Git 忽略的 `方案H初版/.env.famou`（`FAMOU_API_URL`、`FAMOU_API_KEY`，权限 0600）。
本地评测进程断网会退出；云端实验仍在运行时用同一实验 ID 重启即可。

## 第一次试验结果（2026-10-21，实验 `SchemeHScore20261020`）

本次试验的 2 个子场景取自原 10 月规划期口径（2026-10-01 起，已作废）；规划期改为 8 月 1–7 日后未重跑，以下数字只反映 10 月子场景上的相对表现。

- 云端完成 10 轮，排行榜 14 条；实际消耗 **3,762 资源点**（预估 10,860）。期间本机网络断开两次，本地评测进程重启过。
- 排行榜前 4 条是初始程序本身；伐谋生成的候选在本地完整精度复评如下：

| 候选 | 改动要点 | 相对人工权重 |
|---|---|---:|
| 初始程序 | 人工线性权重 | 1.000 |
| 第 5 名 | 未覆盖时加"约 3 分覆盖价值"，稀缺度、紧迫度权重翻倍 | 0.984 |
| 第 6 名 | 权重微调 | 0.982 |
| 第 7 名 | 加入窗口紧度等特征 | 0.981 |
| 第 8 名 | 改为"单位时间收益 + 稀缺 + 紧度 − 冲突 − 时间先后"的新公式，max_obs=0/1 特判 | 0.979 |

**结论**：伐谋给出的改写思路都合理，但没有一个超过人工权重（差 1.6%–2.1%）。这类排序启发式的效果很难靠"写得像样"判断，
10 轮的最小演化不足以找到改进。同一问题上，方案 D 的遗传算法调权重和方案 H 的价格协调都明显更有效（见上级 README）。

## 后续建议

- 伐谋适合交给"评测快、范围小、有安全兜底"的子任务；本次接入链路（沙箱、本地评测、断网重启）已经可以复用。
- 若要再试，可考虑把可调参数限定为少量数值常数，或让它演化局部搜索中某一个算子的顺序，每次实验约 4,000 资源点。
