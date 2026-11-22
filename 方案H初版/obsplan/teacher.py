"""训练神经网络放置规则用的"老师"：任意场景上的精简版长跑优化（与 scripts/long_run.py 同一套做法）。

从稳健模式的计划出发，反复"换一组设置生成新的 H 计划 → 解合并（CP-SAT 分块）→ 局部搜索"，只接受变好的，
直到时间用完。赛题上这套做法 35 分钟能把 0.9744 提到 0.9822；老师必须明显强于默认流水线，
网络才可能学到默认流水线之外的东西。
"""
from __future__ import annotations

import time

import numpy as np

from . import fusion_merge as fm
from . import lagrange
from .checks import structural_violations
from .lns import state_from_schedule
from .local_search import improve
from .problem import Problem
from .schedule import Schedule

# 新计划的设置轮换（与 scripts/long_run.py 相同）：迭代次数、恢复时机偏移、快速局部搜索、是否用伐谋规则
VARIANTS = [
    {"iters": 300, "offset": 0, "fast": False, "rule": True},
    {"iters": 400, "offset": 17, "fast": False, "rule": True},
    {"iters": 300, "offset": 0, "fast": False, "rule": False},
    {"iters": 200, "offset": 29, "fast": False, "rule": True},
    {"iters": 500, "offset": 41, "fast": False, "rule": True},
    {"iters": 150, "offset": 0, "fast": True, "rule": True},
    {"iters": 350, "offset": 23, "fast": False, "rule": False},
    {"iters": 250, "offset": 11, "fast": False, "rule": True},
    {"iters": 600, "offset": 7, "fast": False, "rule": True},
    {"iters": 450, "offset": 33, "fast": False, "rule": False},
]


def grid_objective(pb: Problem, sch: Schedule) -> float:
    """网格计划的目标值（与 lagrange.objective 同口径：覆盖奖励 + 得分）。"""
    _, tg, _, m = fm._cells(pb, sch)
    return float(pb.vc * np.unique(tg).size + pb.vs * (pb.P[tg] * pb.frac[m]).sum())


def make_variant(pb: Problem, k: int, rule) -> tuple[Schedule, dict]:
    v = VARIANTS[k % len(VARIANTS)]
    it = v["iters"]
    offset = (v["offset"] + 13 * (k // len(VARIANTS))) % 50
    recover_at = tuple(sorted({0, *range(offset, it, 50), it - 1}))
    sch, _ = lagrange.solve(pb, iters=it, fast=v["fast"], verbose=False, recover_at=recover_at,
                            rule=rule if v["rule"] else None)
    return sch, v


def teach(pb: Problem, rule, minutes: float = 30.0, workers: int = 8, block_limit: float = 30.0,
          block_budget: float = 360.0, keep: int = 2, log=None) -> tuple[Schedule, dict]:
    """返回 (最好网格计划, 信息)。信息里有起点（稳健模式）与最终的目标值、轮数、每轮记录。"""
    t_start = time.time()
    deadline = t_start + minutes * 60
    start, info0 = lagrange.solve(pb, iters=300, verbose=False, rule=rule, portfolio=True)
    t_start_plan = time.time() - t_start
    best, best_obj = start, grid_objective(pb, start)
    start_obj = best_obj
    recent: list[Schedule] = []
    rounds = []
    k = 0
    while time.time() < deadline:
        t0 = time.time()
        var, vcfg = make_variant(pb, k, rule)
        recent = (recent + [var])[-keep:]
        pool = fm.build_pool(pb, [best] + recent, durations=(4, 5, 6, 7, 8), shifts=(-1, 1))
        cur = fm.select_of(pool, fm._cells(pb, best), fm.pool_index(pool))
        remaining = max(deadline - time.time(), 30.0)
        cur, _ = fm.merge_blocks(pb, pool, cur, time_limit=block_limit, workers=workers, seed=k,
                                 budget_s=min(block_budget, remaining), verbose=False)
        merged = fm.to_schedule(pb, pool, cur, "老师：解合并")
        st = state_from_schedule(pb, merged)
        improve(pb, st)
        after = st.schedule(pb, "老师：解合并 + 局部搜索", 0.0)
        improved = False
        for cand in (merged, after):
            o = grid_objective(pb, cand)
            if o > best_obj + 1e-9 and not structural_violations(pb, cand):
                best, best_obj, improved = cand, o, True
        row = {"round": k, "variant": vcfg, "pool": len(pool), "variant_obj": round(grid_objective(pb, var), 4),
               "best_obj": round(best_obj, 4), "improved": improved, "seconds": round(time.time() - t0, 1)}
        rounds.append(row)
        if log is not None:
            log(row)
        k += 1
    best.algo = "老师（精简长跑）"
    info = {"start_obj": start_obj, "best_obj": best_obj, "gain_rel": best_obj / start_obj - 1.0,
            "start_picked": info0.get("portfolio", {}).get("picked"), "start_seconds": round(t_start_plan, 1),
            "rounds": rounds, "seconds": round(time.time() - t_start, 1), "start": start}
    return best, info
