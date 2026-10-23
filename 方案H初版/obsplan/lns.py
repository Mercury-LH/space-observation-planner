"""大邻域搜索（LNS）：逐段用 CP-SAT 精确重排。

每次取一个站点的一段时间（默认 2 小时），把完全落在段内的任务拿掉，其余任务固定；
在段内为所有可见目标建立可选区间（开始时间由求解器决定，时长 120–240 秒），
满足站点不重叠、与该目标其他观测的 48 小时间隔，最大化（覆盖奖励 + 得分），变好才接受。
全量 CP-SAT 解不动（见方案 G 的实验），但单站 2 小时的子问题通常几秒内就能求到最优。
"""
from __future__ import annotations

import time

import numpy as np
from ortools.sat.python import cp_model

from .greedy import State
from .local_search import _attach, _detach
from .problem import Problem

SCALE = 1e8


def state_from_schedule(pb: Problem, sch, state: State | None = None) -> State:
    """把 Schedule 装回 State（找回每个任务所在的窗口）；给定 state 时在其基础上追加。"""
    st = state if state is not None else State(pb)
    cell = pb.cfg.cell_s
    for s, t, a0, d in zip(sch.station, sch.target, sch.start_s, sch.dur_s):
        a, m = int(a0 // cell), int(d // cell)
        ws = pb.t_win[pb.t_ptr[t]:pb.t_ptr[t + 1]]
        hit = ws[(pb.w_st[ws] == s) & (pb.w_k0[ws] <= a) & (pb.w_k1[ws] >= a + m)]
        k = st.n
        st.st[k], st.tg[k], st.a[k], st.m[k], st.w[k] = s, t, a, m, int(hit[0])
        st.alive[k] = True
        _attach(k, st.st, st.tg, st.a, st.m, st.owner, st.t_cnt, st.t_obs_a, st.t_obs_m, st.t_task)
        st.n += 1
    return st


def _value(pb: Problem, t: int, m: int, first: bool) -> float:
    return pb.vs * pb.P[t] * pb.frac[m] + (pb.vc if first else 0.0)


def improve_slice(pb: Problem, st: State, s: int, c0: int, c1: int, time_limit: float = 3.0,
                  durations=(4, 5, 6, 7, 8), workers: int = 8) -> float:
    """在站点 s 的 [c0, c1) 内用 CP-SAT 重排，返回目标函数提升量（未提升返回 0，状态不变）。"""
    owner = st.owner[s]
    inside = sorted({int(k) for k in owner[c0:c1] if k >= 0
                     and st.a[k] >= c0 and st.a[k] + st.m[k] < c1})
    # 段内被边界任务占用的格视为不可用
    blocked = []
    for c in range(c0, c1):
        if owner[c] >= 0 and int(owner[c]) not in inside:
            if blocked and blocked[-1][1] == c - 1:
                blocked[-1] = (blocked[-1][0], c)
            else:
                blocked.append((c, c))
    # 摘下段内任务，记录它们的价值（按摘下后其余观测是否存在判断首次）
    removed = [(k, int(st.tg[k]), int(st.a[k]), int(st.m[k]), int(st.w[k])) for k in inside]
    for k, *_ in removed:
        _detach(k, st.st, st.tg, st.a, st.m, st.owner, st.t_cnt, st.t_obs_a, st.t_obs_m, st.t_task)
    base = 0.0
    seen = set()
    for k, t, a, m, w in removed:
        base += _value(pb, t, m, st.t_cnt[t] == 0 and t not in seen)
        seen.add(t)

    model = cp_model.CpModel()
    vars_ = []
    per_target: dict[int, list] = {}
    intervals = []
    sel = np.nonzero((pb.w_st == s) & (pb.w_k1 >= c0 + 4) & (pb.w_k0 <= c1 - 5))[0]
    hint = {(t, w): (a, m) for k, t, a, m, w in removed}
    for w in sel:
        t = int(pb.w_tg[w])
        if st.t_cnt[t] >= 3:
            continue
        lo0, hi0 = max(int(pb.w_k0[w]), c0), min(int(pb.w_k1[w]), c1 - 1)
        for m in durations:
            lo, hi = lo0, hi0 - m
            for x in range(st.t_cnt[t]):
                b, mb = int(st.t_obs_a[t, x]), int(st.t_obs_m[t, x])
                if b < lo:
                    lo = max(lo, b + mb + pb.gap)
                else:
                    hi = min(hi, b - pb.gap - m)
            if hi < lo:
                continue
            x_ = model.NewBoolVar("")
            a_ = model.NewIntVar(lo, hi, "")
            intervals.append(model.NewOptionalFixedSizeIntervalVar(a_, m + 1, x_, ""))
            val = _value(pb, t, m, st.t_cnt[t] == 0)
            vars_.append((x_, a_, t, m, int(w), val))
            per_target.setdefault(t, []).append(x_)
            h = hint.get((t, int(w)))
            if h is not None and h[1] == m and lo <= h[0] <= hi:
                model.AddHint(x_, 1)
                model.AddHint(a_, h[0])
    for b0, b1 in blocked:
        intervals.append(model.NewIntervalVar(b0, b1 - b0 + 1, b1 + 1, ""))
    model.AddNoOverlap(intervals)
    for xs in per_target.values():
        if len(xs) > 1:
            model.AddAtMostOne(xs)
    model.Maximize(sum(int(round(v * SCALE)) * x for x, _, _, _, _, v in vars_))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.Solve(model)
    gain = 0.0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        new = [(t, solver.Value(a_), m, w, v) for x_, a_, t, m, w, v in vars_ if solver.Value(x_)]
        gain = sum(v for *_, v in new) - base
    if gain > 1e-12:
        for t, a, m, w, v in new:
            k = st.n
            st.st[k], st.tg[k], st.a[k], st.m[k], st.w[k] = s, t, a, m, w
            st.alive[k] = True
            _attach(k, st.st, st.tg, st.a, st.m, st.owner, st.t_cnt, st.t_obs_a, st.t_obs_m, st.t_task)
            st.n += 1
        for k, *_ in removed:
            st.alive[k] = False
        return gain
    # 没有提升：恢复原状
    for k, *_ in removed:
        _attach(k, st.st, st.tg, st.a, st.m, st.owner, st.t_cnt, st.t_obs_a, st.t_obs_m, st.t_task)
    return 0.0


def slices(pb: Problem, hours: float = 2.0) -> list[tuple[int, int, int]]:
    L = int(hours * 3600 / pb.cfg.cell_s)
    out = []
    for s, runs in enumerate(pb.win.dark):
        for a, b in runs:
            for c0 in range(a, b, L):
                out.append((s, c0, min(c0 + L, b + 1)))
    return out


def run_lns(pb: Problem, st: State, budget_s: float = 120.0, hours: float = 2.0, time_limit: float = 3.0,
            seed: int = 0, verbose: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    sl = slices(pb, hours)
    order = rng.permutation(len(sl))
    t0 = time.time()
    total, tried, improved = 0.0, 0, 0
    for i in order:
        if time.time() - t0 > budget_s:
            break
        s, c0, c1 = sl[i]
        g = improve_slice(pb, st, s, c0, c1, time_limit=time_limit)
        tried += 1
        if g > 0:
            improved += 1
            total += g
        if verbose and tried % 20 == 0:
            print(f"  [LNS] {tried}/{len(sl)} 段，改进 {improved} 段，累计 +{total:.5f}，{time.time() - t0:.0f}s")
    return {"slices": len(sl), "tried": tried, "improved": improved, "gain": total, "seconds": time.time() - t0}
