"""快速结构检查（不做几何计算）：用于测试和开发时的即时反馈。

检查项：时长范围、规划期、站点冲突与切换时间、同目标 48 小时间隔（结束到开始）、
任务是否完整落在该目标在该站的某个合法窗口内。几何上的最终把关仍由 validator.py 用 Skyfield 完成。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .problem import Problem
from .schedule import Schedule


def structural_violations(pb: Problem, sch: Schedule, by_second: bool = False,
                          extra_windows: dict | None = None) -> dict[str, list[int]]:
    """by_second：按秒精修后的计划——不要求对齐网格；任务可以越出网格窗口两端各至多 cell_s−1 秒
    （精修按秒外推的范围，那几秒是否真的可见由 Skyfield 校验把关）。
    extra_windows：网格上没有窗口、按秒补覆盖插入的目标的真实可见区间 {"station", "target", "L", "R"}（秒），
    见 refine.near_windows。"""
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
    slack = cfg.cell_s - 1
    extra = defaultdict(list)
    if extra_windows is not None:
        for s, t, lo, hi in zip(extra_windows["station"], extra_windows["target"], extra_windows["L"], extra_windows["R"]):
            extra[(int(t), int(s))].append((int(lo), int(hi)))
    for i in range(len(sch)):
        t, s = int(sch.target[i]), int(sch.station[i])
        ws = pb.t_win[pb.t_ptr[t]:pb.t_ptr[t + 1]]
        ws = ws[pb.w_st[ws] == s]
        if by_second:
            s0, e0 = sch.start_s[i], sch.start_s[i] + sch.dur_s[i]
            ok = np.any((pb.w_k0[ws] * cfg.cell_s - slack <= s0) & (e0 <= pb.w_k1[ws] * cfg.cell_s + slack)) or \
                any(lo <= s0 and e0 <= hi for lo, hi in extra.get((t, s), ()))
            if not ok:
                bad["不在合法窗口内"].append(i)
            continue
        if not np.any((pb.w_k0[ws] <= a[i]) & (a[i] + m[i] <= pb.w_k1[ws])):
            bad["不在合法窗口内"].append(i)
        if sch.start_s[i] % cfg.cell_s or sch.dur_s[i] % cfg.cell_s:
            bad["未对齐时间网格"].append(i)

    # 同站相邻任务（按开始时间）之间留切换时间；同一目标至多 3 次，相邻两次观测间隔 48 小时
    gap = cfg.repeat_gap_h * 3600
    order = np.lexsort((sch.start_s, sch.target))
    first = np.r_[True, sch.target[order[1:]] != sch.target[order[:-1]]]
    rank = np.arange(len(order)) - np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    if (rank == 3).any():
        bad["同目标超过 3 次"] = [int(i) for i in order[rank == 3]]
    for key, need, name in ((sch.station, cfg.switch_s, "站点冲突或切换不足"), (sch.target, gap, "48 小时间隔不足")):
        order = np.lexsort((sch.start_s, key))
        p, q = order[:-1], order[1:]
        hit = (key[p] == key[q]) & (sch.start_s[q] < sch.start_s[p] + sch.dur_s[p] + need)
        if hit.any():
            bad[name] = [int(i) for i in q[hit]]
    return dict(bad)
