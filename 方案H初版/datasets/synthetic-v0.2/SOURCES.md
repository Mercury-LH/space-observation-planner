# synthetic-v0.2 来源与方法依据

v0.2 的轨道参数、虚构站点、优先级和场景配置由本项目按冻结协议生成，没有并入外部 Hugging Face 记录、真实卫星定轨记录或第三方排程解。可见窗口来自 SGP4 传播和观测几何，教师标签来自本地固定预算求解器，不是人工填写分数或由语言模型猜测。

## 来源边界

v0.1 于 2026-11-07 对 AstroAgentBench、Space-Track TLE History 和 satdatagen 作过来源调查，核验的提交、规模和许可记录保留在 [v0.1/SOURCES.md](../synthetic-v0.1/SOURCES.md)。这是带日期的既有调查，不表示本页重新核对了这些服务的最新状态。v0.2 延续自建物理一致调度场景的方案，不下载或混入这些候选数据。

`v2_sampling.py` 中的截断正态、月份配额、轨道族及站点锚区都是**本项目的合成实验设计假设**，没有借用某真实卫星群体的实测概率分布。新增压力组也不对应真实任务事故频率。原赛题数据没有复制进本版本数据分片，v0.1 仅作为轨道不复用检查的参照。

## 方法依据与代码

| 内容 | 依据和实际作用 |
|---|---|
| SGP4 传播 | Vallado、Crawford、Hujsak、Kelso（2006），*Revisiting Spacetrack Report #3*，AIAA 2006-6753。由平均轨道要素计算时刻位置；本数据不是高精度实测定轨。[作者发布页](https://celestrak.org/publications/AIAA/2006-6753/) |
| SGP4 实现 | Python SGP4 包支持 TLE 输入及批量传播；本项目先导出再回读 TLE，使发布输入与传播输入一致。[软件说明](https://pypi.org/project/sgp4/) |
| Skyfield 独立校验 | 使用卫星传播、站心坐标和太阳几何重新检查任务，不把规划器窗口作为真值；仍使用 SGP4 轨道模型。“独立”是计算路径和数据来源分离，不是另一套动力学真值。[官方卫星文档](https://rhodesmill.org/skyfield/earth-satellites.html) |
| 冻结场景设计 | [采样协议](PROTOCOL.sampling.md)、[采样实现](../../obsplan/training_data/v2_sampling.py)；控制参数逆 CDF 分层抽样，压力目录另列，周期角均匀抽样。 |
| 约束和教师 | [统一规则](../../../规则口径.md)、[教师实现](../../obsplan/training_data/teacher.py)、[本地已演化规则](../../obsplan/evolved/recover_rule.py)。每目标最多 12 个窗口、3 轮价格协调、无局部搜索、按秒精修；不发起云端伐谋请求。 |
| 生成与审计 | [构建器](../../obsplan/training_data/v2_build.py)以事务保存独立目录及中断证据；[审计器](../../obsplan/training_data/v2_audit.py)核对文件、划分、标签与已保存校验报告，不重新执行 Skyfield。 |

SGP4、Skyfield、CelesTrak 方法页和 CC BY 4.0 摘要于 2026-11-07 整理本页时核对。这里不引用第三方图表、观测目录或求解结果作为本数据。

太阳几何计算使用 DE421 天文星历；冻结输入 SHA-256 为 `a20a7139da04cbc462454634918e9a9ca69127044e2cc9d4f9c16e238d2deedc`，该星历不随数据 Release 打包。实际生成提交、逐源文件指纹、协议、依赖版本和并发配置以 `manifest.json.identity` 为准，并在[数据卡](DATASET_CARD.md)的生产证据栏回填。静态绘图只读取冻结规格，不读取或修改生成进程、模型输出及旧结果。

## 原创数据许可

原创合成轨道、站点参数、配置、窗口、候选特征与教师计划按 [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) 发布。该许可允许在保留适当署名、许可链接并说明修改的条件下分享和改作；不暗示作者背书。[具体许可范围](DATA_LICENSE.md)

该许可不重新授权程序代码、第三方依赖、DE421 星历或外部数据。全部发布物的来源和许可边界以本页、数据许可与资产清单共同记录；外部工具的开源许可不会自动转化为对其可能访问的全部数据的授权。
