"""加分项：结合真实历史预报的动态重规划模拟（"状态更新 → 动态重规划"闭环）。

每天 0 点（UTC）重新规划剩余时段：
- 状态：已执行任务按真实云量（再分析）判定成败；成功的目标视为已覆盖（不再给覆盖奖励）、
  剩余可观测次数相应减少、下一次须在上次成功观测结束 48 小时后；失败的可以再排；过去的站点时段封锁；
- 天气：未来 24 小时用"提前 1 天发布"的预报，24–48 小时用"提前 2 天"的预报，之后用气候统计；
  预报云量换算成晴空概率时，用另一年的预报与实况校准（避免自我验证）；
- 只执行当天的任务，次日再重排。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import lagrange
from .greedy import State, candidates, run_greedy
from .problem import Problem
from .visibility import Windows

DAY_H = 24


def calibrate(forecast: np.ndarray, truth: np.ndarray, threshold: float, bins=np.arange(0, 101, 10)):
    """预报云量 → 实况云量 ≤ 阈值的概率（按 10% 分箱），返回映射函数。"""
    ok = ~np.isnan(forecast) & ~np.isnan(truth)
    f, t = forecast[ok], truth[ok] <= threshold
    idx = np.clip(np.digitize(f, bins) - 1, 0, len(bins) - 2)
    overall = t.mean()
    p = np.array([t[idx == b].mean() if (idx == b).sum() >= 20 else overall for b in range(len(bins) - 1)])

    def fn(x: np.ndarray) -> np.ndarray:
        out = np.full(x.shape, overall)
        good = ~np.isnan(x)
        out[good] = p[np.clip(np.digitize(x[good], bins) - 1, 0, len(bins) - 2)]
        return out
    fn.table = p
    return fn


@dataclass
class Status:
    succ: dict = field(default_factory=dict)        # 目标 -> [(a, m)] 成功观测（格）
    c_now: np.ndarray | None = None                  # 各站下一个可用格
    executed: int = 0
    failed: int = 0
    tasks: list = field(default_factory=list)        # (站, 目标, a, m, 成功)


def residual_problem(pb: Problem, status: Status) -> Problem:
    k0 = np.maximum(pb.w_k0, status.c_now[pb.w_st])
    for t, obs in status.succ.items():
        sel = pb.w_tg == t
        if len(obs) >= 3:
            k0[sel] = pb.w_k1[sel] + 1
        else:
            last_end = max(a + m for a, m in obs)
            k0[sel] = np.maximum(k0[sel], last_end + pb.gap)
    keep = pb.w_k1 - k0 >= pb.m_min
    w = pb.win
    w2 = Windows(pb.w_st[keep].astype(np.int16), pb.w_tg[keep].astype(np.int32), k0[keep].astype(np.int32),
                 pb.w_k1[keep].astype(np.int32), w.dark, w.n_samples, w.sgp4_failed)
    return Problem.build(pb.cfg, pb.ds, w2)


def clip_windows(pb: Problem, usable_h: np.ndarray) -> Problem:
    """把可见窗口裁剪到"可用"的小时（usable_h：(站, 小时) bool，例如已知或预报为晴的时段）。
    观测占用的每个时间格都必须落在可用小时里；裁剪后短于最短观测时长的窗口丢弃。"""
    cell = pb.cfg.cell_s
    T = pb.n_samples + 2
    hour_of_cell = (np.arange(T) * cell) // 3600
    H = usable_h.shape[1]
    ok = np.ones((pb.n_stations, T), bool)
    inside = hour_of_cell < H
    ok[:, inside] = usable_h[:, hour_of_cell[inside]]
    st, tg, k0, k1 = [], [], [], []
    for s in range(pb.n_stations):
        sel = np.nonzero(pb.w_st == s)[0]
        if len(sel) == 0:
            continue
        d = np.diff(np.concatenate([[0], ok[s].astype(np.int8), [0]]))
        r0, r1 = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]        # 可用格 [r0, r1)
        a0, a1 = pb.w_k0[sel], pb.w_k1[sel]
        for x0, x1 in zip(r0, r1):
            hit = (a0 < x1) & (a1 > x0)
            if not hit.any():
                continue
            n0, n1 = np.maximum(a0[hit], x0), np.minimum(a1[hit], x1)
            good = n1 - n0 >= pb.m_min
            st.append(np.full(good.sum(), s))
            tg.append(pb.w_tg[sel][hit][good])
            k0.append(n0[good])
            k1.append(n1[good])
    st, tg, k0, k1 = (np.concatenate(x) if x else np.zeros(0, np.int64) for x in (st, tg, k0, k1))
    order = np.lexsort((k0, st))
    w = pb.win
    w2 = Windows(st[order].astype(np.int16), tg[order].astype(np.int32), k0[order].astype(np.int32),
                 k1[order].astype(np.int32), w.dark, w.n_samples, w.sgp4_failed)
    return Problem.build(pb.cfg, pb.ds, w2)


def _fill(pb: Problem, status: Status, sch, q: np.ndarray, vct: np.ndarray):
    """在核心计划之外，把剩余站点时间按"期望价值 = 价值 × 窗口平均晴空概率"贪心补满。"""
    from .lns import state_from_schedule
    st = state_from_schedule(pb, sch, state_factory(pb, status)())
    cw, cm = candidates(pb)
    qp = np.concatenate([np.zeros((q.shape[0], 1)), np.cumsum(q, axis=1)], axis=1)
    qbar = (qp[pb.w_st[cw], pb.w_k1[cw] + 1] - qp[pb.w_st[cw], pb.w_k0[cw]]) / (pb.w_k1[cw] - pb.w_k0[cw] + 1)
    v = pb.vs * pb.P[pb.w_tg[cw]] * pb.frac[cm]
    run_greedy(pb, cw, cm, (v + vct[pb.w_tg[cw]]) * qbar, v * qbar, best_fit=True, state=st)
    return st.schedule(pb, sch.algo, 0.0)


def state_factory(pb: Problem, status: Status):
    def make() -> State:
        st = State(pb)
        for s in range(pb.n_stations):
            st.owner[s, :status.c_now[s]] = -3                 # 过去的时段不可再用
        for t, obs in status.succ.items():
            for x, (a, m) in enumerate(obs[:3]):
                st.t_obs_a[t, x], st.t_obs_m[t, x] = a, m
            st.t_cnt[t] = min(len(obs), 3)
        return st
    return make


def cell_probability(pb: Problem, hourly_q: np.ndarray) -> np.ndarray:
    """(站, 小时) 晴空概率 → (站, 格)。"""
    T = pb.n_samples + 2
    q = np.repeat(hourly_q, int(3600 / pb.cfg.cell_s), axis=1)
    if q.shape[1] < T:
        q = np.concatenate([q, np.repeat(q[:, -1:], T - q.shape[1], axis=1)], axis=1)
    return q[:, :T]


def task_succeeds(truth: np.ndarray, s: int, a: int, m: int, cell_s: int, threshold: float) -> bool:
    h0, h1 = (a * cell_s) // 3600, ((a + m) * cell_s - 1) // 3600
    seg = truth[s, h0:h1 + 1]
    return bool(np.all(seg <= threshold))


def execute(pb: Problem, status: Status, sch, truth, threshold, until_cell: int | None):
    cell = pb.cfg.cell_s
    order = np.argsort(sch.start_s, kind="stable")
    for i in order:
        s, t = int(sch.station[i]), int(sch.target[i])
        a, m = int(sch.start_s[i] // cell), int(sch.dur_s[i] // cell)
        if until_cell is not None and a >= until_cell:
            continue
        ok = task_succeeds(truth, s, a, m, cell, threshold)
        status.executed += 1
        status.tasks.append((s, t, a, m, ok))
        status.c_now[s] = max(status.c_now[s], a + m + 1)
        if ok:
            status.succ.setdefault(t, []).append((a, m))
        else:
            status.failed += 1


def daily(pb: Problem, status: Status, days: int) -> dict:
    """逐日累计：成功覆盖数、得分、当天尝试与失败次数。"""
    day_cells = DAY_H * int(3600 / pb.cfg.cell_s)
    seen, score = set(), 0.0
    out = {"covered": [], "score": [], "attempts": [], "failed": []}
    tasks = sorted(status.tasks, key=lambda x: x[2])
    k = 0
    for d in range(days):
        att = fail = 0
        while k < len(tasks) and tasks[k][2] < (d + 1) * day_cells:
            s, t, a, m, ok = tasks[k]
            att += 1
            if ok:
                seen.add(t)
                score += float(pb.P[t] * pb.frac[m])
            else:
                fail += 1
            k += 1
        out["covered"].append(len(seen))
        out["score"].append(score)
        out["attempts"].append(att)
        out["failed"].append(fail)
    return out


def realized(pb: Problem, status: Status) -> dict:
    cov = len(status.succ)
    score = sum(pb.P[t] * pb.frac[m] for t, obs in status.succ.items() for a, m in obs)
    return {"covered": cov, "score": float(score), "executed": status.executed, "failed": status.failed,
            "combined": 0.5 * cov / pb.n_targets + 0.5 * score / float(pb.P.sum())}


def simulate(pb: Problem, truth: np.ndarray, policy: str, clim: np.ndarray, threshold: float = 30.0,
             fc1: np.ndarray | None = None, fc2: np.ndarray | None = None, cal1=None, cal2=None,
             iters: int = 150, fast: bool = True, verbose: bool = False, return_status: bool = False,
             clip_p: float | None = None, clip_days: int = 1, fill: bool = False, on_day=None):
    """policy: 'static'（不看天气，一次规划）、'static_clim'（按气候概率一次规划）、
    'dynamic_clim'（每天重排，只用气候概率）、'dynamic_forecast'（每天重排，用预报）、
    'hindsight'（事后最优参照：6 天的真实天气全部已知，一次规划；现实中不可实现）。
    clip_p：每天重排时，把未来 clip_days 天内晴空概率 < clip_p 的时段从窗口中裁掉（只在确定性高的预报下有意义）。
    fill：裁剪后再把核心计划装回未裁剪的问题，用按晴空概率加权的贪心补满其余时段（"碰运气"的观测）。
    注意：局部搜索不看晴空概率，所以"已知天气"用裁剪窗口表达，比用权重表达更可靠。
    on_day(d, days)：每天执行完后回调（产品界面显示进度用）。"""
    t0 = time.time()
    H = truth.shape[1]
    days = int(np.ceil(H / DAY_H))
    cells_per_hour = int(3600 / pb.cfg.cell_s)
    status = Status(c_now=np.zeros(pb.n_stations, np.int64))
    clim_h = np.repeat(clim[:, None], H, axis=1)
    if policy.startswith("static") or policy == "hindsight":
        kw, p0 = {}, pb
        if policy == "static_clim":
            kw = dict(cell_weight=cell_probability(pb, clim_h), backup=np.zeros(pb.n_targets))
        elif policy == "hindsight":
            p0 = clip_windows(pb, truth <= threshold)
        sch, _ = lagrange.solve(p0, iters=iters, fast=fast, verbose=False, **kw)
        execute(pb, status, sch, truth, threshold, None)
        if on_day is not None:
            on_day(days, days)
    else:
        for d in range(days):
            now_cell = d * DAY_H * cells_per_hour
            status.c_now = np.maximum(status.c_now, now_cell)
            q_h = clim_h.copy()
            if policy == "dynamic_forecast":
                a, b = d * DAY_H, min((d + 1) * DAY_H, H)
                q_h[:, a:b] = cal1(fc1[:, a:b])
                a2, b2 = b, min((d + 2) * DAY_H, H)
                if a2 < b2:
                    q_h[:, a2:b2] = cal2(fc2[:, a2:b2])
            full = pr = residual_problem(pb, status)
            if clip_p is not None:
                usable = np.ones(q_h.shape, bool)
                a, b = d * DAY_H, min((d + clip_days) * DAY_H, H)
                usable[:, a:b] = q_h[:, a:b] >= clip_p
                pr = clip_windows(full, usable)
            vct = np.full(pb.n_targets, pb.vc)
            left = np.full(pb.n_targets, 3, np.int64)
            for t, obs in status.succ.items():
                vct[t] = 0.0
                left[t] = 3 - len(obs)
            qc = cell_probability(pr, q_h)
            sch, _ = lagrange.solve(pr, iters=iters, fast=fast, verbose=False,
                                    cell_weight=qc, backup=np.zeros(pb.n_targets),
                                    vc_target=vct, max_obs_left=left, init_state=state_factory(pr, status))
            if fill and clip_p is not None:
                sch = _fill(full, status, sch, qc, vct)
            execute(pb, status, sch, truth, threshold, (d + 1) * DAY_H * cells_per_hour)
            if on_day is not None:
                on_day(d + 1, days)
            if verbose:
                r = realized(pb, status)
                print(f"    第 {d + 1} 天后：成功覆盖 {r['covered']}，得分 {r['score']:.0f}，已执行 {r['executed']}（失败 {r['failed']}）",
                      flush=True)
    out = realized(pb, status)
    out["seconds"] = round(time.time() - t0, 1)
    return (out, status) if return_status else out
