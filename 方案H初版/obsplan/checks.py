"""快速结构检查（不做几何计算）：用于测试和开发时的即时反馈。

检查项：时长范围、规划期、站点冲突与切换时间、同目标 48 小时间隔（结束到开始）、
任务是否完整落在该目标在该站的某个合法窗口内。几何上的最终把关仍由 validator.py 用 Skyfield 完成。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .problem import Problem
from .schedule import Schedule


def structural_violations(pb: Problem, sch: Schedule) -> dict[str, list[int]]:
    cfg = pb.cfg
    bad: dict[str, list[int]] = defaultdict(list)
    horizon = cfg.days * 86400
    for i in range(len(sch)):
        if not (cfg.obs_min_s <= sch.dur_s[i] <= cfg.obs_max_s):
            bad["时长越界"].append(i)
        if sch.start_s[i] < 0 or sch.start_s[i] + sch.dur_s[i] > horizon:
            bad["超出规划期"].append(i)

    # 窗口包含：任务 [a, a+m]（采样点）须落在同目标同站某个窗口 [k0, k1] 内
    a = sch.start_s // cfg.cell_s
    m = sch.dur_s // cfg.cell_s
    for i in range(len(sch)):
        t, s = int(sch.target[i]), int(sch.station[i])
        ws = pb.t_win[pb.t_ptr[t]:pb.t_ptr[t + 1]]
        ws = ws[pb.w_st[ws] == s]
        if not np.any((pb.w_k0[ws] <= a[i]) & (a[i] + m[i] <= pb.w_k1[ws])):
            bad["不在合法窗口内"].append(i)
        if sch.start_s[i] % cfg.cell_s or sch.dur_s[i] % cfg.cell_s:
            bad["未对齐时间网格"].append(i)

    for s in np.unique(sch.station):
        idx = np.nonzero(sch.station == s)[0]
        idx = idx[np.argsort(sch.start_s[idx], kind="stable")]
        for p, q in zip(idx[:-1], idx[1:]):
            if sch.start_s[q] < sch.start_s[p] + sch.dur_s[p] + cfg.switch_s:
                bad["站点冲突或切换不足"].append(int(q))

    gap = cfg.repeat_gap_h * 3600
    for t in np.unique(sch.target):
        idx = np.nonzero(sch.target == t)[0]
        idx = idx[np.argsort(sch.start_s[idx], kind="stable")]
        if len(idx) > 3:
            bad["同目标超过 3 次"].append(int(idx[3]))
        for p, q in zip(idx[:-1], idx[1:]):
            if sch.start_s[q] < sch.start_s[p] + sch.dur_s[p] + gap:
                bad["48 小时间隔不足"].append(int(q))
    return dict(bad)
