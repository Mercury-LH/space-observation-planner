# 空间目标观测调度训练数据 synthetic-v0.1

这是一批用于学习**地面光学站如何安排空间目标观测任务**的原创合成数据。每个场景提供轨道、站点、规则、可见窗口、20 维候选特征、低成本教师选择标签，以及独立逐秒校验的计划证据。首个适合的用途是训练小型 MLP 学习候选排序，再由已有放置器保证资源约束。

本版本每个目标最多保留 12 个窗口作为候选短表，教师使用 3 轮价格协调和本地已演化规则，不运行局部搜索。标签表示该教师的选择，**不表示最优动作；未选中也不等于非法**。它不包含完整强化学习状态转移。原始合成 TLE 与完整几何窗口一并保留，可以另做扩池实验。

生成前固定规模为 30 个独立轨道 catalog、120 个六天场景；按 catalog 隔离为 train/validation/test = 20/5/5，对应 80/20/20 个场景。目标规模覆盖 1,000、5,000、10,000；共有 103,000 条独立合成轨道，四种变体合计 412,000 个目标实例。`cat-027` 和 `cat-029` 是预先指定的高倾角 LEO 分布偏移测试组。

**发布状态：完整生产 120 场景、21,791,917 候选行、536,751 条最终任务；正式记录失败 0、超时 0；240/240 份独立逐秒计划报告通过，删除修复 0 条任务。31 个分片共 744,282,558 字节。以 Release 资产分发；上传及匿名下载的核验记录见 Release 发布说明。** 生成代码冻结在 `85bc712b3035b103dfd72adaaa90e578002b043e`；Release tag 为 `synthetic-data-v0.1`。正式资产清单以本目录的 `release-assets.json` 为准，不用生成前目标代替实际验收结果。

## 下载并读取

以下命令需要 Git、uv 和网络；数据下载与 NumPy 加载器**不依赖 PyTorch**，也不需要 Hugging Face 账号、赛题原始数据包或伐谋 API key。完整数据位于同仓库的 [GitHub Release](https://github.com/Mercury-LH/space-observation-planner/releases/tag/synthetic-data-v0.1)，按 catalog 分片；Git 只跟踪代码、说明、摘要、预览和下载清单。

```bash
git clone --branch synthetic-data-v0.1 https://github.com/Mercury-LH/space-observation-planner.git
cd space-observation-planner/方案H初版
uv sync --frozen
uv run python -m obsplan.training_data.download outputs/synthetic-v0.1
uv run python -m obsplan.training_data.loader outputs/synthetic-v0.1 --smoke
uv run python -m obsplan.training_data.audit outputs/synthetic-v0.1 --output outputs/synthetic-v0.1-audit.json
```

下载器先按 `release-assets.json` 校验全部分片的字节数及 SHA-256，再安全解压。已校验的下载缓存可以复用；目标目录必须是新目录，已有目录不会被覆盖。`--smoke` 会先遍历全部训练行拟合标准化，然后各划分读取一个批次，适合验证接口，不是完整机器学习评测。

审计输出写在数据根目录之外，因为向完整数据根目录新增文件会破坏清单完整性。`audit` 检查文件哈希、划分、输入、标签、计划和保存的逐秒校验证据，**不会再运行一遍 Skyfield**。需要重新模拟时应调用项目独立校验器，并按清单准备相同星历。

若只想先看字段和加载效果，Git 中包含约 **1.74 MB** 的 [preview](preview/manifest.json)，无需下载完整 Release：

```bash
uv run python -m obsplan.training_data.loader datasets/synthetic-v0.1/preview --smoke
```

这个预览来自 200 目标校准 catalog 的 4 个变体，全部属于 train；没有 validation/test，不能用于泛化评估或替代正式 80/20/20 场景划分。

## 在训练代码中使用

```python
from obsplan.training_data.loader import DatasetReader

reader = DatasetReader("outputs/synthetic-v0.1")
normalizer = reader.fit_normalizer()  # 只使用 train；验证和测试复用同一统计

for epoch in range(10):
    for batch in reader.iter_batches(
        "train", batch_size=4096, normalizer=normalizer,
        shuffle=True, seed=2026 + epoch,
    ):
        features = batch["features"]       # float32 [B, 20]
        selected = batch["grid_selected"]  # bool [B]，教师选择标签
        scene_id = batch["scene_id"]
        candidate_id = batch["candidate_id"]  # actions.npz 中原始候选行号
        # 在已安装 PyTorch 的训练环境中：
        # x = torch.from_numpy(features)
        # y = torch.from_numpy(selected.astype("float32"))
```

加载器一次加载一个压缩场景，再在场景内部切批，不要求整批数据同时驻留内存。训练时建议启用 `shuffle=True`，避免按原始时长分块顺序连续喂给模型；随机化仅作用于指定划分中的场景顺序和场景内候选顺序，各字段同步重排，不跨划分且不合并场景。`scene_id` 加 `candidate_id` 可以恢复原候选位置。validation/test 复用训练标准化，不重新拟合。

正式完整数据可以复用仓库提供的 [normalizer.json](normalizer.json)：`from obsplan.training_data.loader import Normalizer` 后调用 `Normalizer.load("datasets/synthetic-v0.1/normalizer.json")`。该文件只由正式训练集拟合，并绑定训练清单指纹；不适用于小型 preview 或另行生成的数据。

`iter_batches()` 返回候选级数组；要读取长度为任务数的 `grid_task_candidate`，使用 `iter_scenes()`。不要随机拆分候选行来重新划分训练集，避免同一轨道 catalog 泄漏。这里提供的是读取和训练接口示例，尚未据此训练或评估神经网络。

读取计划时注意两种索引：`candidate_id` 指向 `actions.npz` 的候选行；`source_grid_row` 指向**修复前原始教师的插入序号**。由于网格计划与精修计划分别校验并可能分别删去非法任务，`plan.npz` 的 `source_grid_row` 不能直接当作发布后 `grid_plan.npz` 的行号。两个计划应按原始序号关联。

## 文档和复现

- [数据卡](DATASET_CARD.md)：生成假设、划分、20 维特征、标签语义、完整产物树和生成/审计/打包命令。
- [机器可读 schema](schema.json)：NPZ 数组类型、维度、关联索引、计划溯源和元数据字段。
- [冻结协议](PROTOCOL.md)：规模、校验、候选预算及失败保留的事前约定。
- [来源记录](SOURCES.md)：为何未直接采用调查到的 HF 数据，以及 SGP4/Skyfield 方法依据。
- [数据许可](DATA_LICENSE.md)：本次原创数据采用 CC BY 4.0；不改变仓库已有代码或第三方依赖的许可。

实际使用效果需要在保留测试集上比较完整计划的覆盖、原始观测分、质量与总计算时间。仅提高行级分类准确率，不能说明神经网络已经提升调度质量或真实任务泛化能力。
