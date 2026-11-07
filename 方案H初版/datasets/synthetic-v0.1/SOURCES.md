# 数据来源与方法依据

本文件记录 `synthetic-v0.1` 的来源筛选，核验日期为 **2026-11-07**。本版本的目标轨道参数、站点配置和优先级由本项目生成；不并入以下候选的卫星记录、观测记录、问题实例或解。这里列出它们，是为了说明取舍和保留后续复核入口。

“合成”指空间目标及场景是人为设定的，不表示随意填写可见窗口或教师分数。窗口由轨道传播和观测几何计算得到，教师计划由本地求解器生成。SGP4、Skyfield 和太阳位置计算所需的天文星历属于计算依赖，不能据此把合成目标称为真实卫星。

## 检查的三个候选

| 候选 | 一手来源所述内容 | 与本题的差异 | 本版本处理 |
|---|---|---|---|
| [AstroAgentBench](https://huggingface.co/datasets/kaupane/AstroAgentBench) | 多类空间任务规划的 Harbor 任务包，包含对地成像、星座设计、深空网络天线调度等；当前版本只有 7 个 smoke 任务附 oracle 解。 | 对地成像和深空通信的资源、几何及任务模型不同；不能直接替代地面光学站观测空间目标的数据，也不是大量专家动作轨迹。 | 仅参考场景组织和独立验证方式，不复制实例或解。 |
| [Space-Track TLE History](https://huggingface.co/datasets/juliensimon/space-track-tle-history) | 历史轨道要素档案，可支持轨道分布研究、轨道预测等；不包含本题站点、观测任务和排程标签。 | 轨道记录数不等于独立调度场景数；同一目标不同历元的记录还需去重、按历元匹配并重建观测场景。 | 保留为今后真实轨道外部测试的候选；本版本不下载或合并。 |
| [satdatagen](https://github.com/ahgolden/satdatagen) | 地面站空间目标观测数据生成器，输出采样时刻、仰角、方位角、TLE、视星等和云量；需要 Space-Track 账号。 | 领域最接近，但输出观测几何输入，不提供本题时长、切换、重访约束下的大规模排程监督。 | 本项目已有传播、窗口、求解和校验核心，本版本复用现有核心，不引入该生成器的在线依赖。 |

### 版本、规模与许可核验

**AstroAgentBench。** 检查的 HF 提交为 `ac02325e2d9d15314ef87b02ceda0f5723dadaa6`，发布版 `v2.0.0` 对应源码提交 `6f5b9d21b1efb2ca2ee6fcabdc338a9d077612cb`。对其发布清单逐项统计，131 个任务目录、2,400 个列入清单的文件，合计 141,390,006 字节；只有 7 个 `solution/solution.json`。这些是整个任务包的规模，不是监督样本数。[发布清单](https://huggingface.co/datasets/kaupane/AstroAgentBench/blob/v2.0.0/release-manifest.json)

当前数据卡为 **mixed-source-licenses**：项目代码采用 MIT，第三方输入保留各自条款；SPOT5 数据来自 Zequn Wei、Jin-Kao Hao 的 Mendeley 发布版，采用 CC BY 4.0。旧搜索摘要里的单一 MIT 标签不能代表整个当前数据包。[当前许可说明](https://huggingface.co/datasets/kaupane/AstroAgentBench/blob/v2.0.0/README.md#licensing-and-attribution)、[SPOT5 原始数据发布](https://data.mendeley.com/datasets/2kbzg9nw3b/1)

**Space-Track TLE History。** 检查时 HF 元数据提交为 `3f9e1667f9a98c6b72449a4d2135d60e878dc4e9`，卡片声明 CC BY 4.0，API 返回 `gated="auto"`；匿名 raw 请求返回 401，需要账号填写访问表单。页面显示约 2.53 亿条、12.5 GB，而较旧 README 文字仍为 2.38 亿、11 GB。因此这些数字仅供估算，不作为本数据集的规模或依赖。本轮没有申请访问，也没有下载该档案。[数据卡与来源说明](https://huggingface.co/datasets/juliensimon/space-track-tle-history/blob/main/README.md?code=true)、[HF 元数据接口](https://huggingface.co/api/datasets/juliensimon/space-track-tle-history)

其上游 Space-Track 的官方文档在 “Redistribution of Basic SSA Information” 一节明确说明：TLE/OMM、SATCAT、衰减和再入等 Basic SSA 数据可在适当引用条件下转发和再分发。不能把所有 Space-Track 产品一概视作禁止转载，也不能只凭下游作者的“public domain”文字推及其他产品。[Space-Track 官方条款和数据说明](https://www.space-track.org/documentation)

**satdatagen。** 检查的 PyPI 版本为 `1.0.1`，源码包约 2.9 MB、wheel 约 3.1 MB；这是程序包体积，不是预生成训练集体积。软件采用 MIT，复用实质代码时需保留版权和许可。软件许可不会自动替代 Space-Track 轨道及气象服务的数据条件。本版本没有复制该项目代码或输出。[PyPI 发布页](https://pypi.org/project/satdatagen/1.0.1/)、[MIT 许可](https://raw.githubusercontent.com/ahgolden/satdatagen/main/LICENSE)、[生成输出的源码](https://github.com/ahgolden/satdatagen/blob/main/src/satdatagen/helpers.py)

## 本版本采用的方法依据

1. **SGP4 轨道传播。** 根据合成的平均轨道要素求给定时刻的位置；不把这些要素当作实测定轨结果，也不使用简单开普勒轨道代替 SGP4 的要素语义。方法文献为 Vallado、Crawford、Hujsak、Kelso（2006），*Revisiting Spacetrack Report #3*，AIAA 2006-6753。[作者发布页](https://celestrak.org/publications/AIAA/2006-6753/)、[Python SGP4 包说明](https://pypi.org/project/sgp4/)
2. **Skyfield 独立几何校验。** 对最终任务重新计算站心仰角、太阳高度和目标受照，不读取规划器给出的可见窗口作为校验真值。库本身仍使用 SGP4 传播，故“独立”指校验路径与窗口数据分离，不表示采用另一套高精度动力学真值。[Skyfield 地球卫星文档](https://rhodesmill.org/skyfield/earth-satellites.html)
3. **本仓库的约束与教师。** 规则来自项目统一口径；候选特征复用 `obsplan.famou_h_eval.FEATURES`。本版本在按日×站点分层抽取、每目标最多 12 个窗口的候选池内，固定运行 3 轮本地价格协调，再使用已演化排序规则；不做局部搜索，后续以 `near=False` 按秒精修。该过程不请求云端伐谋，不保证价格收敛或教师全局最优，也不是全窗口搜索。完整窗口另行保留以支持后续扩池。[规则口径](../../../规则口径.md)、[共同评测流程](../../docs/自动评测与协作.md)、[特征实现](../../obsplan/famou_h_eval.py)、[教师实现](../../obsplan/training_data/teacher.py)

## 原创部分与发布许可

合成轨道和站点参数、场景配置、候选特征以及本次求解得到的教师计划属于本次数据生产流程的输出。**这些原创数据采用 [Creative Commons Attribution 4.0 International（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/)**，允许按许可条款复制、分发及改作，需保留适当归属、许可链接并说明修改。发布资产与 Git 元数据应一致标明这一许可。

这一授权只作用于本次原创数据，不变更仓库已有代码的许可，也不重新授权第三方依赖或天文星历。本文件对第三方条款的说明是来源记录，不将第三方许可证自动套用到原创数据。

原创数据许可见 [DATA_LICENSE.md](DATA_LICENSE.md)。生产依赖为 Python 3.12.13、NumPy 2.5.4、SGP4 2.27、Skyfield 1.55、Numba 0.68.0、OR-Tools 9.15.6755；各值也保存在 `manifest.json`。DE421 文件的 SHA-256 为 `a20a7139da04cbc462454634918e9a9ca69127044e2cc9d4f9c16e238d2deedc`，不随 Release 分片打包。数据生成代码和协议的逐文件指纹随数据清单发布。
