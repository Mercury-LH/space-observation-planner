# 自动评测结果

这是团队内部质量—时间效用，不是官方分数，也不保证上限为 100。
时间权重 0.428571；固定时间尺度 120 秒。
求解计时包括所有候选生成、融合和精修；预处理与独立校验另列。

| 方法 | 状态 | Q中位数 | 覆盖数 | 累计得分 | 求解秒 | 内部效用 | Pareto |
|---|---|---:|---:|---:|---:|---:|---|
| fusion_baseline | 全部合法 (1) | 0.978157 | 9998 | 28754.703 | 1515.100 | 59.0400 | 是 |
| hplus_10_fast | 全部合法 (1) | 0.963861 | 9933 | 28090.587 | 79.242 | 80.8899 | 是 |
| hplus_1_fast | 全部合法 (2) | 0.961049 | 9911 | 27987.640 | 72.465 | 81.6383 | 是 |
| hplus_300_full | 全部合法 (1) | 0.974359 | 9994 | 28538.393 | 176.972 | 72.9953 | 是 |

时间权重敏感性（效用越高越好）：

| 时间权重 | 排名 |
|---:|---|
| 0.0 | fusion_baseline > hplus_300_full > hplus_10_fast > hplus_1_fast |
| 0.1 | hplus_10_fast > hplus_1_fast > hplus_300_full > fusion_baseline |
| 0.2 | hplus_1_fast > hplus_10_fast > hplus_300_full > fusion_baseline |
| 0.4 | hplus_1_fast > hplus_10_fast > hplus_300_full > fusion_baseline |
| 0.42857142857142855 | hplus_1_fast > hplus_10_fast > hplus_300_full > fusion_baseline |

每个 run/result.json 保存原始指标、配置、数据/代码指纹、随机种子、线程和阶段耗时。
plan.csv 和本地日志含目标明细，仅保存在忽略的 outputs 中。
单次试验不能证明统计稳定或全局最优；缓存冷热和平台会影响耗时。
