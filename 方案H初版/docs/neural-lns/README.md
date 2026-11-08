# 神经引导局部搜索 v1：实现与运行说明

本实现从发布 tag `synthetic-data-v0.2`（`1d03b9d920c43d20d8e1b46bd30dde9f3745bb4a`）建立独立 clone，分支为 `feat/neural-local-search-v1`。原 planner、auto-eval、dataset 三个工作区及原数据保持只读。平台六档和默认算法未改。

已实现小 MLP 静态预训练、同状态实际收益的动态排序训练、恢复及独立推理、带回滚的局部操作、共享候选的规则/随机/神经对照、反事实样本收集、串行校准/评测、模型冻结和采纳统计。当前状态是**本地开发验证通过，正式训练与采纳评测待云端运行**。本地模型是开发冒烟产物，不能用于正式采纳。没有创建或购买云实例，没有调用伐谋云实验。

## 协议与实现边界

事前协议、split 和特征字典位于 `experiments/neural-lns-v1/`，`FROZEN.json` 保留三文件哈希。`IMPLEMENTATION-v1.json` 单独登记有限候选枚举的实现细化，并进入最终运行冻结；原协议未按结果修改。输入数据下载必须显式使用 `datasets/synthetic-v0.2/release-assets.json`。

- 静态输入为发布的 20 列，编码器 `20→128→64`，静态模型 11,009 参数。动态模型另接 24 列状态/操作特征，共 16,705 参数。未选中教师候选不是非法标签。
- normalizer 只扫描指定 train 子集，按 scene 等权。固定每 scene 16 batch，批量 8192；验证保留自然比例，按 scene 汇总 AP/排名指标。三 seed 为 2026/2027/2028。
- 动态特征和推理共用 `FeatureContext`，用完整窗口池重算价格相关特征。静态模型跨到完整窗口池的消融明确记录 domain transfer。每份动态样本包含 parent/initial/pool/price/代码/依赖指纹。
- 局部候选最多 32 个、相关目标最多 64 个、最多修改 3 条旧任务。固定规则、稀缺和随机来源混合，实际枚举有限；日志记录未访问范围及轮换策略。网络仅改变排序，不把低分当非法。
- 公共结构检查覆盖全规划期、所有站点；48 小时间隔从前一次结束计算。修改在副本中完成，拒绝/超时不改变父计划。最终计划仍调用公共 Skyfield 1 秒与端点校验。
- 反事实标签比较“相同父状态与相同修复预算”的候选结果和未修改 reference。每候选最多 64 次试探、2 秒 spawn 子进程硬上限；启动/传输成本计入。超时保留为删失，不能用零收益代替。reference 和特征耗时是父状态级，跨行汇总须去重。
- 公共 H+ 本身存在块级调用。插件在调用边界检查总预算，返回最近合法 grid；若原局部搜索或精修越过总预算，整次记失败。不能把合法 fallback 的存在当作按时完成。正式外层 worker 另有硬墙钟上限。
- 预算由原 H+300 full 三次完整计时产生：`B=max(5,1.25*median(T0))`。所有方法在 B 内共同预留 `max(0.05,1.25*median(三次finish耗时))`。完整 H+ 初始、特征、模型加载/推理、复制/回滚、候选和精修均计入 B。
- 正式七组都保留强基线的 `finish_near=True`。公共短窗口几何在求解前预备、单列 `near_windows_prepare_s`，缓存按 catalog 隔离；首次云端准备可能需要补算短窗口，必须计入端到端/数据准备成本。本地开发 smoke 用 `near=False`，没有做全量短窗口重算。
- 对照为延长原 H+、经审阅的 H1 multi_bound、规则、随机、神经、仅阶段 A、阶段 B 随机初始化。H1 源文件和测试的来源/哈希在 `h1-import-review.json`；历史速度不能代替本轮复测。
- 主体 60、压力 30、已知赛题实例单独验收，三 seed 按 catalog 配对 bootstrap；缺失、失败、OOM、中断、超时不删。当前没有任何正式 1% 质量收益结论。

## 已执行的本地验证

输出根目录：`outputs/neural-preflight-v1/`。完整原始日志、checkpoint、normalizer、schema、model card 与哈希在该目录，未加入普通 Git 历史。

1. `data-hashes.json`：发布 manifest、events 与 10,891 个登记文件全部哈希一致，420/90/90 个 catalog；只读复用已有数据，没有重新下载或重算 600 场景几何。
2. `static-smoke/`：从冻结 train40 中按输入规模选最小的 `v2-cat-342`，validation 从固定 pilot 中按规模选 `v2-cat-546`。200 步后保存，恢复到固定总步数 202；训练仅用于管线验证。`prediction-output.npz` 证明独立 CLI 保留原 candidate ID 并输出有限分值。
3. `search-smoke/`：该 train catalog 的前 32 个真实目标，真实 TLE/站点和已发布窗口，H+ 1 次迭代、8 搜索轮。rule/random/static_only 三份计划均独立校验通过，C=32、S=279.58，输出相同。预热不同，不能用这里的秒数排名。
4. `counterfactual-dev-smoke.jsonl`（在 search-smoke 内）仅有 4 条开发样本：32 目标、空父状态、0 次 repair 试探。这些偏离正式采样预算的设置写在每行，不能进入正式 pilot；它们用于验证两种实际收益标签、删失接口及训练接线。
5. `dynamic-smoke/`：阶段 A 初始化的动态模型 200+2 步恢复；`dynamic-random-smoke/` 为随机初始化动态模型 200 步。只证明训练和两种初始化路径工作。
6. `dynamic-search-smoke/`：动态 checkpoint 接入真实局部搜索、模型输入版本检查及独立几何校验的开发验证。

关键测试包括跨站/区外历史、结束到开始的 48 小时、事务与截止时间回滚、候选索引、真实 H+ 价格矩阵 padding、训练/推理列序、划分泄漏、逐位一致的 checkpoint 恢复、重复动态样本拒绝、删失排除、超时留档、评测中断恢复和失败不重试。最终测试计数见 `LOCAL_STATUS.json` 与 `outputs/neural-preflight-v1/pytest-full.log`。

## 云端准备与静态训练

以下命令从仓库的 `方案H初版/` 执行。脚本只使用已存在的机器；用户提供实例和明确预算后才启动。`WALL_SECONDS` 是本次已授权的有限秒数，不是自动续租指令。SSH 凭据应保存在安全 SSH 配置中，不写入本目录。

```bash
bash scripts/neural_cloud.sh setup
export DATA_ROOT=/data/synthetic-v0.2
export RUN_ROOT=/data/neural-runs/pilot-2026
export WALL_SECONDS=3600
```

`setup` 用可选 `neural` extra 和锁定依赖安装环境并检查实际 CUDA。锁中的 PyTorch/CUDA 是否兼容租用实例的驱动需要在实例上确认；本机只验证 CPU，不能提前声明 CUDA 已通过。如果必须更换 wheel，先记录新环境锁并重新校准，不伪造原锁兼容性。

云端没有数据时才下载到新目录：

```bash
bash scripts/neural_cloud.sh download
bash scripts/neural_cloud.sh verify
python scripts/neural_make_config.py --dataset-root "$DATA_ROOT" \
  --subset 40 --seed 2026 --device cuda --output "$RUN_ROOT/static-config.json"
bash scripts/neural_cloud.sh train static --config "$RUN_ROOT/static-config.json" \
  --output "$RUN_ROOT/static" --max-wall-s 3500
```

上述一小时只是首轮有界吞吐试跑示例，必须在实际已授权的额度内使用；停止后从 `latest.json` 读取 checkpoint 路径，再恢复同一配置和输出目录：

```bash
bash scripts/neural_cloud.sh resume static --config "$RUN_ROOT/static-config.json" \
  --output "$RUN_ROOT/static" --resume "$RUN_ROOT/static/实际checkpoint文件.pt" \
  --max-wall-s 3500
```

模型/优化器、RNG、下一数据步、normalizer/schema、来源身份均保存。身份变化时拒绝恢复；原 attempt 和 checkpoint 保留。`max-wall-s` 在训练步边界保存，外层 GNU timeout 是独立硬上限，异常时保留已有 checkpoint 和失败 ledger。短测试吞吐不能直接预测全量成本。

## 反事实校准与动态训练

先在 train40 上至多 100 个操作，保留所有 reference/候选超时、峰值内存和启动开销。看到这些成本后才决定扩展，不自动增加 GPU：

```bash
bash scripts/neural_cloud.sh collect --dataset-root "$DATA_ROOT" \
  --output "$RUN_ROOT/cf-calibration" --cache-dir "$RUN_ROOT/cache" \
  --instance-id 实际实例标识 --workers 8 --max-evaluations 100 --max-states 8 --max-wall-s 3500
```

正式首批上限 40 catalog × 8 状态 × 16 操作=5120。只有完成校准并确认瓶颈后，才在新输出目录把 `--max-evaluations` 改为 5120；重复父状态会记录并跳过，因而实际行数可以更少。此处不保证能获得足够不同收益的成对标签，若无有效 pair，训练会明确拒绝。

```bash
python scripts/neural_make_config.py --dataset-root "$DATA_ROOT" \
  --subset 40 --seed 2026 --device cuda --samples "$RUN_ROOT/cf-full/samples.jsonl" \
  --init-checkpoint "$RUN_ROOT/static/经validation选择的checkpoint.pt" \
  --output "$RUN_ROOT/dynamic-config.json"
bash scripts/neural_cloud.sh train dynamic --config "$RUN_ROOT/dynamic-config.json" \
  --output "$RUN_ROOT/dynamic" --max-wall-s 3500
```

阶段 B 随机初始化对照用相同样本、seed、步数，生成配置时省略 `--init-checkpoint`，使用独立输出目录。阶段 A 单独消融直接使用其 checkpoint。动态 checkpoint 只可根据固定 validation 的实际规划分数选择，不能按训练损失或 test 选择。

## 串行评测、恢复与正式冻结

在空闲实例上固定线程、GPU/CPU、RAM、实例 ID、缓存和预热策略。环境记录区分 Linux 的实际 GPU/CUDA/容器配额与 macOS 电源字段，不伪造 AC0/Battery0。

```bash
bash scripts/neural_cloud.sh calibrate --dataset-root "$DATA_ROOT" --selection pilot \
  --cache-dir "$RUN_ROOT/cache" --instance-id 实际实例标识 --workers 8 \
  --output "$RUN_ROOT/pilot-calibration"
```

准备 `models.json`，键为 `neural`、`static_only`、`dynamic_random_init`，每个值为 seed 字符串到 checkpoint 路径的映射。第一轮允许仅 seed `2026`，正式为全部三个 seed。

```bash
bash scripts/neural_cloud.sh evaluate --dataset-root "$DATA_ROOT" --selection pilot \
  --cache-dir "$RUN_ROOT/cache" --instance-id 实际实例标识 --workers 8 --seeds 2026 \
  --calibration "$RUN_ROOT/pilot-calibration/calibration.json" \
  --models "$RUN_ROOT/models.json" --output "$RUN_ROOT/pilot-evaluation" --device cpu
```

`--resume-campaign` 只续跑未开始的条目，已完成、失败、中断和超时都保留且不重试筛选。原始记录按追加账本保存。恢复输出新的汇总文件，使用命令返回的 `records` 路径进行后续评测。40-catalog pilot 未比规则取得正的平均 S 收益，或新搜索输给延长原 H+，就停止扩大。

对动态 checkpoint 的选择证据：

```bash
.venv/bin/python -m obsplan.neural_lns.protocol select-dynamic \
  --protocol experiments/neural-lns-v1 --records "$RUN_ROOT/pilot-evaluation/runs.json" \
  --checkpoint "$RUN_ROOT/dynamic/候选checkpoint.pt" --method neural --seed 2026 \
  --output "$RUN_ROOT/dynamic/validation_selection.json"
```

完整三 seed 及所有对照准备好后，再对 `--selection final` 校准：它包括 90 个 synthetic test 的**基线计时**和已知赛题应用实例，须传 `--application-data-dir` 指向原生输入。此步骤不读取 test 候选标签来选模型。然后执行正式冻结：

```bash
.venv/bin/python -m obsplan.neural_lns.protocol freeze-runtime \
  --protocol experiments/neural-lns-v1 --models "$RUN_ROOT/models-three-seeds.json" \
  --calibration "$RUN_ROOT/final-calibration/calibration.json" \
  --pilot-records "$RUN_ROOT/pilot-evaluation/runs.json" --dependency-lock uv.lock \
  --instance-id 实际实例标识 --output "$RUN_ROOT/runtime-freeze.json"
```

冻结器验证真实 checkpoint、完成 10000 步的同 trial 记录、所选 validation checkpoint、normalizer/schema/依赖/代码/校准/环境哈希和实际 pilot 门槛。开发 smoke 不能通过正式冻结。正式 `evaluate --selection final` 额外传 `--runtime-freeze`、`--application-data-dir`，完整运行七种方法和三 seed；该模型集合只绑定一个正式 campaign，不能换输出目录重新试 test。

最后用 `python -m obsplan.neural_lns.evaluation --protocol experiments/neural-lns-v1 --records 完整runs.json --models runtime-freeze.json --output 新的acceptance.json` 汇总主体/压力/应用实例、95% catalog bootstrap、C/S/Q/J、Pareto 与失败清单。全部门槛通过前保持原平台算法。正式产出应另附实际 GPU/CPU 小时、权重及部署推理成本；目前这些正式云端成本和收益均未知。
