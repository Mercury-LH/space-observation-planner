"""按秒大邻域搜索：每次只动一个站的一段时间（其余任务全部固定），用 CP-SAT 在这段时间里按秒重新排。

和网格上的长跑互补：长跑在 30 秒格子上决定选哪些任务，最后的按秒精修只能平移、拉长，不能换任务、改顺序；
这里在秒的层面同时决定选谁、先后和时长。
- 候选：这个站在这段时间里的全部可见窗口——网格窗口两端按秒外推后的真实区间，加上网格上不到 120 秒、
  按秒外推后够 120 秒的短窗口（refine.probe_bounds，和按秒精修同一套判定与余量）；
- 变量：每个候选选不选、开始秒、时长（120–240 秒，整秒）；
- 约束：同站相邻留 30 秒切换（区间长度 = 时长 + 30 秒，NoOverlap），落在段内且不碰段外的固定任务；
  同一目标本段内至多一次，并与固定任务保持 48 小时间隔，总次数不超过 3；
- 目标：和 combined 成比例（覆盖奖励只给别处没有观测的目标）。以当前计划为提示，变好才接受，
  接受前用 schedule.evaluate 重新精确计分。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

from .problem import Problem
from .refine import near_windows, probe_bounds
from .schedule import Schedule, evaluate

SCALE = 1e12          # 目标函数系数放大后取整（combined 的单项约 1e-4）


@dataclass
class Context:
    pb: Problem
    # 候选窗口：网格窗口 + 短窗口，各自的站、目标、粗略区间（秒）；真实区间按需探查后缓存
    st: np.ndarray
    tg: np.ndarray
    lo0: np.ndarray        # 粗略区间（网格两端各放宽 cell−1 秒），用来筛选
    hi0: np.ndarray
    k0: np.ndarray
    k1: np.ndarray
    L: np.ndarray          # 真实区间，未探查为 −1
    R: np.ndarray
    by_station: list = field(default_factory=list)   # 每站的候选下标（按 lo0 排序）

    @classmethod
    def build(cls, pb: Problem) -> "Context":
        from .bounds import _short_windows
        cfg = pb.cfg
        cell = cfg.cell_s
        sst, stg, sk0, sk1 = (np.asarray(x).astype(np.int64) for x in _short_windows(pb))
        keep = sk1 - sk0 == pb.m_min - 1                  # 只要差一格的短窗口（两端外推最多补回 58 秒）
        st = np.concatenate([pb.w_st, sst[keep]])
        tg = np.concatenate([pb.w_tg, stg[keep]])
        k0 = np.concatenate([pb.w_k0, sk0[keep]])
        k1 = np.concatenate([pb.w_k1, sk1[keep]])
        n = len(st)
        ctx = cls(pb, st, tg, k0 * cell - (cell - 1), k1 * cell + (cell - 1), k0, k1,
                  np.full(n, -1, np.int64), np.full(n, -1, np.int64))
        # 按秒才可见的目标（near_windows）本来就是短窗口的一部分，这里不用另加
        for s in range(pb.n_stations):
            idx = np.nonzero(st == s)[0]
            ctx.by_station.append(idx[np.argsort(ctx.lo0[idx], kind="stable")])
        return ctx

    def bounds(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        need = idx[self.L[idx] < 0]
        if len(need):
            L, R = probe_bounds(self.pb.cfg, self.pb.ds, self.st[need], self.tg[need], self.k0[need], self.k1[need])
            self.L[need], self.R[need] = L, R
        return self.L[idx], self.R[idx]

    def extra_windows(self) -> dict:
        """已探查过的短窗口（真实区间 ≥ 120 秒），给 checks.structural_violations 当额外合法窗口。"""
        base = len(self.pb.w_st)
        sel = np.nonzero((np.arange(len(self.st)) >= base) & (self.L >= 0)
                         & (self.R - self.L >= self.pb.cfg.obs_min_s))[0]
        nw = near_windows(self.pb)
        return {"station": np.concatenate([self.st[sel], nw["station"]]),
                "target": np.concatenate([self.tg[sel], nw["target"]]),
                "L": np.concatenate([self.L[sel], nw["L"]]), "R": np.concatenate([self.R[sel], nw["R"]])}


def _coef(pb: Problem, P: float, cov: bool) -> tuple[int, int]:
    """一次观测的价值 = a + b × 时长（秒），已放大取整。frac(d) = 0.6 + 0.4 (d − 120) / 120 = 0.2 + d / 300。"""
    a = (pb.vc if cov else 0.0) + pb.vs * P * 0.2
    b = pb.vs * P / 300.0
    return int(round(a * SCALE)), int(round(b * SCALE))


def solve_slice(ctx: Context, sch: Schedule, s: int, A: int, B: int, time_limit: float = 10.0, workers: int = 8,
                seed: int = 0, hint_keep: float = 1.0) -> tuple[Schedule, dict]:
    """重排站 s 在 [A, B]（秒）内的任务。返回（可能更新的）计划和统计。
    hint_keep：提示里随机保留多少比例的段内任务（"拆掉一部分再重建"）。整份当前计划作提示时，
    CP-SAT 往往停在提示上找不到更好的；只给一部分，它会从不同的起点搜，常能超过原计划。"""
    pb, cfg = ctx.pb, ctx.pb.cfg
    sw, dmin, dmax = cfg.switch_s, cfg.obs_min_s, cfg.obs_max_s
    gap = int(round(cfg.repeat_gap_h * 3600))
    t0 = time.time()
    end = sch.start_s + sch.dur_s
    on_s = sch.station == s
    free = on_s & (sch.start_s >= A) & (end <= B)
    before = on_s & ~free & (sch.start_s < A)
    after = on_s & ~free & (sch.start_s >= A)
    lo = max(A, int(end[before].max()) + sw) if before.any() else A
    hi = min(B, int(sch.start_s[after].min()) - sw) if after.any() else B          # 最晚结束
    info = {"station": s, "A": A, "B": B, "free": int(free.sum())}
    if hi - lo < dmin:
        return sch, {**info, "skipped": "段太短"}

    # 候选窗口：粗略区间与 [lo, hi] 相交的
    idx = ctx.by_station[s]
    j1 = np.searchsorted(ctx.lo0[idx], hi, side="right")
    idx = idx[:j1]
    idx = idx[ctx.hi0[idx] >= lo + dmin]
    if not len(idx):
        return sch, {**info, "skipped": "没有候选"}
    L, R = ctx.bounds(idx)
    cl, cr = np.maximum(L, lo), np.minimum(R, hi)

    # 同目标的固定任务（所有站，不含本段要重排的任务）：48 小时间隔与次数上限
    fixed = ~free
    fx_t, fx_s, fx_e = sch.target[fixed], sch.start_s[fixed], end[fixed]
    order = np.argsort(fx_t, kind="stable")
    fx_t, fx_s, fx_e = fx_t[order], fx_s[order], fx_e[order]
    tg = ctx.tg[idx]
    p0 = np.searchsorted(fx_t, tg, side="left")
    p1 = np.searchsorted(fx_t, tg, side="right")
    cov = p1 == p0
    for i in range(len(idx)):
        if p1[i] - p0[i] >= 3:
            cr[i] = cl[i] - 1
            continue
        for k in range(p0[i], p1[i]):
            fs, fe = fx_s[k], fx_e[k]
            if fe + gap <= cl[i] or cr[i] + gap <= fs:
                continue
            left = (cl[i], min(cr[i], fs - gap))               # 放在固定任务之前
            right = (max(cl[i], fe + gap), cr[i])              # 放在之后
            cl[i], cr[i] = max((left, right), key=lambda x: x[1] - x[0])
    ok = cr - cl >= dmin
    idx, cl, cr, tg, cov = idx[ok], cl[ok], cr[ok], tg[ok], cov[ok]
    info["candidates"] = int(len(idx))
    if not len(idx):
        return sch, {**info, "skipped": "没有可行候选"}

    # 当前段内任务映射到候选（提示 + 当前值）
    fr = np.nonzero(free)[0]
    hint = {}
    for i in fr:
        m = np.nonzero((tg == sch.target[i]) & (cl <= sch.start_s[i]) & (end[i] <= cr))[0]
        if not len(m):
            return sch, {**info, "skipped": f"段内任务 {int(i)} 找不到对应候选"}
        hint[int(m[0])] = (int(sch.start_s[i]), int(sch.dur_s[i]))

    cur_hint = dict(hint)
    if hint_keep < 1.0:
        rng = np.random.default_rng(seed)
        hint = {k: v for k, v in hint.items() if rng.random() < hint_keep}
    P = pb.P[tg]
    md = cp_model.CpModel()
    xs, ss, ds_, ivs, obj = [], [], [], [], []
    for i in range(len(idx)):
        x = md.NewBoolVar("")
        st = md.NewIntVar(int(cl[i]), int(cr[i]) - dmin, "")
        d = md.NewIntVar(0, int(min(dmax, cr[i] - cl[i])), "")
        e = md.NewIntVar(int(cl[i]) + dmin + sw, int(cr[i]) + sw, "")
        md.Add(d >= dmin).OnlyEnforceIf(x)
        md.Add(d == 0).OnlyEnforceIf(x.Not())
        md.Add(st + d <= int(cr[i])).OnlyEnforceIf(x)
        ivs.append(md.NewOptionalIntervalVar(st, d + sw, e, x, ""))
        a, b = _coef(pb, float(P[i]), bool(cov[i]))
        obj += [a * x, b * d]
        xs.append(x), ss.append(st), ds_.append(d)
        if i in hint:
            md.AddHint(x, 1), md.AddHint(st, hint[i][0]), md.AddHint(d, hint[i][1])
            md.AddHint(e, hint[i][0] + hint[i][1] + sw)
        else:
            md.AddHint(x, 0), md.AddHint(d, 0)
    md.AddNoOverlap(ivs)
    for t in np.unique(tg):
        same = np.nonzero(tg == t)[0]
        if len(same) > 1:
            md.AddAtMostOne([xs[k] for k in same])
    md.Maximize(sum(obj))
    cur = sum(_coef(pb, float(P[i]), bool(cov[i]))[0] + _coef(pb, float(P[i]), bool(cov[i]))[1] * h[1]
              for i, h in cur_hint.items())
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    status = solver.Solve(md)
    info.update(status=solver.StatusName(status), seconds=round(time.time() - t0, 1))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return sch, info
    new_obj = solver.ObjectiveValue()
    info.update(obj_gain=(new_obj - cur) / SCALE, bound_gap=(solver.BestObjectiveBound() - new_obj) / SCALE)
    if new_obj <= cur:
        return sch, info
    sel = [i for i in range(len(idx)) if solver.Value(xs[i])]
    keep = ~free
    out = Schedule(np.concatenate([sch.station[keep], np.full(len(sel), s, np.int64)]),
                   np.concatenate([sch.target[keep], tg[sel].astype(np.int64)]),
                   np.concatenate([sch.start_s[keep], np.array([solver.Value(ss[i]) for i in sel], np.int64)]),
                   np.concatenate([sch.dur_s[keep], np.array([solver.Value(ds_[i]) for i in sel], np.int64)]),
                   algo=sch.algo, runtime_s=sch.runtime_s, info=sch.info)
    o = np.lexsort((out.start_s, out.station))
    out = Schedule(out.station[o], out.target[o], out.start_s[o], out.dur_s[o], algo=out.algo,
                   runtime_s=out.runtime_s, info=out.info)
    e0, e1 = evaluate(cfg, pb.ds, sch)["combined"], evaluate(cfg, pb.ds, out)["combined"]
    info.update(combined_before=e0, combined_after=e1, tasks_after=int(len(sel)))
    if e1 <= e0:
        return sch, {**info, "rejected": "精确计分没有变好"}
    info["accepted"] = True
    return out, info


def slices(pb: Problem, hours: float = 2.0, step_h: float = 1.0, stations=None):
    """每站每个黑夜段，按 hours 长、step_h 步长切片，返回 (站, A 秒, B 秒) 列表。"""
    cell = pb.cfg.cell_s
    out = []
    for s in (range(pb.n_stations) if stations is None else stations):
        for a, b in pb.win.dark[s]:
            A0, B0 = a * cell - cell, b * cell + cell
            t = A0
            while True:
                B = min(t + int(hours * 3600), B0)
                out.append((s, t, B))
                if B >= B0:
                    break
                t += int(step_h * 3600)
    return out
