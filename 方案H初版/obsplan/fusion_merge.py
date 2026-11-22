"""解合并（solution merging）：把几份网格计划里的任务合成一个受限候选池，用 CP-SAT 在池上重新求解原问题。

候选 = (站点, 目标, 开始格 a, 时长 m)，开始时间固定，所以全部约束都是"区间图"上的冲突：
- 站点：任务占用格 [a, a+m]（含 30 秒切换的 1 格），同站两个候选占用格相交即冲突；
- 同目标 48 小时：把候选看成 [a, a+m+gap) 的"延长区间"，两个延长区间相交 ⇔ 间隔不足 48 小时；
区间图的最大团可以扫描线一次求出，每个团加一条 AtMostOne，既精确又给线性松弛很紧的界。
每颗目标至多 3 次（6 天内本来就被 48 小时间隔隐含，仍显式加上）。
覆盖用布尔 y_t：y_t ⇒ 至少选了一个该目标的候选；目标函数 = 覆盖奖励 + 得分，系数取整后与 combined 严格成比例：
    combined × 2 × N × sumP = 覆盖数 × sumP + Σ P × (0.6 + 0.4(m−4)/4) × N
                            = 覆盖数 × sumP + Σ P × (2000 + 1000 m)   （N = 10000，m 以格计，m=4..8）
为了腾出空间，池中每个任务另补同一窗口内的变体：左对齐（同开始）和右对齐（同结束）的若干较短/较长时长。

两种求解模式：
- 全局：整个池一个模型，以当前最好计划为提示；
- 分块：按时间块滚动，块外任务固定为当前解，块内重排（冲突的候选预先剔除），变好才接受。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

from .problem import Problem
from .schedule import Schedule


# ---------------------------------------------------------------- 候选池
@dataclass
class Pool:
    st: np.ndarray       # 站点
    tg: np.ndarray       # 目标
    a: np.ndarray        # 开始格
    m: np.ndarray        # 时长（格）
    w: np.ndarray        # 所在窗口
    src: np.ndarray      # 来源位掩码：1,2,4… 对应输入计划；0 = 补的变体

    def __len__(self):
        return len(self.st)


def _cells(pb: Problem, sch: Schedule):
    cell = pb.cfg.cell_s
    a = np.rint(np.asarray(sch.start_s, float) / cell).astype(np.int64)
    m = np.rint(np.asarray(sch.dur_s, float) / cell).astype(np.int64)
    return np.asarray(sch.station, np.int64), np.asarray(sch.target, np.int64), a, m


def find_window(pb: Problem, s: int, t: int, a: int, m: int) -> int:
    ws = pb.t_win[pb.t_ptr[t]:pb.t_ptr[t + 1]]
    hit = ws[(pb.w_st[ws] == s) & (pb.w_k0[ws] <= a) & (pb.w_k1[ws] >= a + m)]
    return int(hit[0]) if len(hit) else -1


def build_pool(pb: Problem, plans: list[Schedule], durations=(4, 6, 8), align=("left", "right"),
               shifts=()) -> Pool:
    """plans 里所有任务去重合并；每个任务再按 durations 补左/右对齐变体（须落在原窗口内）；
    shifts 非空时，再把（原时长和 durations 的）左对齐位置平移 shifts 格。"""
    rows: dict[tuple, list] = {}           # (s,t,a,m) -> [w, src]
    for bit, sch in enumerate(plans):
        for s, t, a, m in zip(*_cells(pb, sch)):
            key = (int(s), int(t), int(a), int(m))
            if key in rows:
                rows[key][1] |= 1 << bit
                continue
            w = find_window(pb, *key)
            if w < 0:
                raise ValueError(f"计划 {bit} 的任务 {key} 不在任何合法窗口内")
            rows[key] = [w, 1 << bit]
    base = list(rows.items())
    for (s, t, a, m), (w, _) in base:
        k0, k1 = int(pb.w_k0[w]), int(pb.w_k1[w])
        for d in durations:
            for side in align:
                b = a if side == "left" else a + m - d
                if b < k0 or b + d > k1 or d < pb.m_min or d > pb.m_max:
                    continue
                rows.setdefault((s, t, b, d), [w, 0])
        for d in set(durations) | {m}:
            for sh in shifts:
                b = a + sh
                if b < k0 or b + d > k1 or d < pb.m_min or d > pb.m_max:
                    continue
                rows.setdefault((s, t, b, d), [w, 0])
    keys = np.array(list(rows.keys()), np.int64).reshape(-1, 4)
    vals = np.array(list(rows.values()), np.int64).reshape(-1, 2)
    return Pool(keys[:, 0], keys[:, 1], keys[:, 2], keys[:, 3], vals[:, 0], vals[:, 1])


def pool_index(pool: Pool) -> dict:
    return {(int(s), int(t), int(a), int(m)): i for i, (s, t, a, m) in enumerate(zip(pool.st, pool.tg, pool.a, pool.m))}


def select_of(pool: Pool, sch_cells, idx: dict | None = None) -> np.ndarray:
    """把一份计划映射到池下标（计划须已在池中）。"""
    idx = idx or pool_index(pool)
    return np.array([idx[(int(s), int(t), int(a), int(m))] for s, t, a, m in zip(*sch_cells)], np.int64)


def to_schedule(pb: Problem, pool: Pool, sel: np.ndarray, algo: str, runtime: float = 0.0) -> Schedule:
    sel = np.asarray(sel, np.int64)
    sch = Schedule.from_cells(pb.cfg, pool.st[sel], pool.tg[sel], pool.a[sel], pool.m[sel], algo=algo,
                              runtime_s=runtime)
    sch.info["windows"] = pool.w[sel]
    return sch


# ---------------------------------------------------------------- 区间图最大团
def interval_cliques(lo: np.ndarray, hi: np.ndarray, ids: np.ndarray) -> list[list[int]]:
    """半开区间 [lo, hi) 的全部极大团（只返回大小 ≥ 2 的）。扫描线：结束事件排在同一时刻的开始事件之前。"""
    if len(ids) < 2:
        return []
    ev = np.concatenate([np.stack([hi, np.zeros_like(hi), ids], 1), np.stack([lo, np.ones_like(lo), ids], 1)])
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
    active: set[int] = set()
    out, grew = [], False
    for _, kind, i in ev:
        if kind == 1:
            active.add(int(i))
            grew = True
        else:
            if grew and len(active) >= 2:
                out.append(list(active))
            grew = False
            active.discard(int(i))
    return out


# ---------------------------------------------------------------- 模型
def coefficients(pb: Problem, pool: Pool) -> tuple[np.ndarray, int]:
    """整数系数：任务 P×(2000+1000m)（按 cell=30s、120–240s 规则），覆盖 sumP。与 combined 成严格比例。"""
    N = pb.n_targets
    sref = pb.cfg.w_score / pb.vs                                          # = score_ref（sumP）
    task = pb.P[pool.tg] * pb.frac[pool.m] * N                             # 得分 × N
    cov = sref
    # 校验能否取整（默认规则下必然是整数）
    ti = np.rint(task).astype(np.int64)
    ci = int(round(cov))
    if np.abs(ti - task).max() > 1e-6 or abs(ci - cov) > 1e-6:
        scale = 1e3
        return np.rint(task * scale).astype(np.int64), int(round(cov * scale))
    return ti, ci


def combined_of(pb: Problem, obj: float, ci: int) -> float:
    N = pb.n_targets
    return obj / (2.0 * N * ci) if ci else 0.0


def solve_subset(pb: Problem, pool: Pool, free: np.ndarray, fixed: np.ndarray, hint: np.ndarray | None,
                 time_limit: float, workers: int = 8, seed: int = 0, verbose: bool = False,
                 log: bool = False) -> dict:
    """free：可选候选（池下标）；fixed：固定选中的任务（池下标）。返回 {"sel": 新选中的 free 子集, ...}。
    与 fixed 冲突（站点占用 / 48 小时）的 free 候选先剔除；覆盖奖励只给 fixed 未覆盖的目标。"""
    t0 = time.time()
    ti, ci = coefficients(pb, pool)
    gap = pb.gap
    free = np.asarray(free, np.int64)
    fixed = np.asarray(fixed, np.int64)
    # ---- 剔除与固定任务冲突的候选
    T = pb.n_samples + 2
    occ = np.zeros((pb.n_stations, T + 1), np.int32)
    for k in fixed:
        occ[pool.st[k], pool.a[k]:pool.a[k] + pool.m[k] + 1] += 1
    pref = np.concatenate([np.zeros((pb.n_stations, 1), np.int64), np.cumsum(occ, axis=1)], axis=1)
    s_, a_, m_ = pool.st[free], pool.a[free], pool.m[free]
    ok = (pref[s_, a_ + m_ + 1] - pref[s_, a_]) == 0
    fix_cnt = np.bincount(pool.tg[fixed], minlength=pb.n_targets)
    fix_obs: dict[int, list] = {}
    for k in fixed:
        fix_obs.setdefault(int(pool.tg[k]), []).append((int(pool.a[k]), int(pool.m[k])))
    ok &= fix_cnt[pool.tg[free]] < 3
    for j in np.nonzero(ok)[0]:
        t = int(pool.tg[free[j]])
        if t in fix_obs:
            a, m = int(a_[j]), int(m_[j])
            for b, mb in fix_obs[t]:
                if not (a >= b + mb + gap or a + m + gap <= b):
                    ok[j] = False
                    break
    cand = free[ok]
    # ---- 建模
    model = cp_model.CpModel()
    x = [model.NewBoolVar("") for _ in range(len(cand))]
    pos = {int(k): j for j, k in enumerate(cand)}
    n_amo = 0
    st_c, tg_c, a_c, m_c = pool.st[cand], pool.tg[cand], pool.a[cand], pool.m[cand]
    for s in range(pb.n_stations):
        j = np.nonzero(st_c == s)[0]
        for cl in interval_cliques(a_c[j], a_c[j] + m_c[j] + 1, j):
            model.AddAtMostOne([x[q] for q in cl])
            n_amo += 1
    order = np.argsort(tg_c, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(tg_c[order]))[0] + 1, len(order)]
    obj = [ti[cand[j]] * x[j] for j in range(len(cand))]
    n_cov = 0
    ys = {}
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        j = order[b0:b1]
        if len(j) == 0:
            continue
        t = int(tg_c[j[0]])
        for cl in interval_cliques(a_c[j], a_c[j] + m_c[j] + gap, j):
            model.AddAtMostOne([x[q] for q in cl])
            n_amo += 1
        lim = 3 - int(fix_cnt[t])
        if len(j) > lim:
            model.Add(sum(x[q] for q in j) <= lim)
        if fix_cnt[t] == 0:
            y = model.NewBoolVar("")
            model.AddBoolOr([x[q] for q in j] + [y.Not()])
            obj.append(ci * y)
            ys[t] = (y, j)
            n_cov += 1
    model.Maximize(sum(obj))
    # ---- 提示
    hint_set = set()
    if hint is not None:
        hint_set = {pos[int(k)] for k in hint if int(k) in pos}
        for j in range(len(cand)):
            model.AddHint(x[j], 1 if j in hint_set else 0)
        for t, (y, j) in ys.items():
            model.AddHint(y, 1 if any(q in hint_set for q in j) else 0)
    hint_obj = sum(int(ti[cand[j]]) for j in hint_set) + ci * sum(
        1 for t, (y, j) in ys.items() if any(q in hint_set for q in j))
    build_s = time.time() - t0
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log
    status = solver.Solve(model)
    res = {"status": solver.StatusName(status), "n_free": int(len(free)), "n_cand": int(len(cand)),
           "n_amo": n_amo, "n_cov_vars": n_cov, "build_s": round(build_s, 1),
           "solve_s": round(time.time() - t0 - build_s, 1), "hint_obj": int(hint_obj)}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sel = cand[[j for j in range(len(cand)) if solver.Value(x[j])]]
        res.update(sel=sel, obj=int(solver.ObjectiveValue()), bound=float(solver.BestObjectiveBound()))
    else:
        res.update(sel=None, obj=None, bound=None)
    if verbose:
        print(f"  [merge] 候选 {len(cand)}/{len(free)} 团约束 {n_amo} 覆盖变量 {n_cov} 建模 {build_s:.1f}s "
              f"{res['status']} 目标 {res['obj']} (提示 {hint_obj}) 界 {res['bound']} {res['solve_s']}s", flush=True)
    return res


def objective(pb: Problem, pool: Pool, sel: np.ndarray) -> int:
    ti, ci = coefficients(pb, pool)
    return int(ti[sel].sum()) + ci * len(np.unique(pool.tg[sel]))


# ---------------------------------------------------------------- 两种求解模式
def merge_global(pb: Problem, pool: Pool, incumbent: np.ndarray, time_limit=300.0, workers=8, seed=0,
                 verbose=True, log=False) -> tuple[np.ndarray, dict]:
    allc = np.arange(len(pool))
    r = solve_subset(pb, pool, allc, np.zeros(0, np.int64), incumbent, time_limit, workers, seed, verbose, log)
    if r["sel"] is not None and objective(pb, pool, r["sel"]) > objective(pb, pool, incumbent):
        return r["sel"], r
    return incumbent, r


def merge_blocks(pb: Problem, pool: Pool, incumbent: np.ndarray, block_h=24.0, step_h=12.0, time_limit=40.0,
                 workers=8, seed=0, budget_s=600.0, passes=1, verbose=True) -> tuple[np.ndarray, list]:
    """滚动时间块：开始格落在 [B0, B1) 的候选可选，其余固定为当前解。"""
    cell = pb.cfg.cell_s
    B, S = int(block_h * 3600 / cell), int(step_h * 3600 / cell)
    T = pb.n_samples
    cur = np.asarray(incumbent, np.int64)
    logs = []
    t0 = time.time()
    for p in range(passes):
        B0 = 0
        while B0 < T:
            if time.time() - t0 > budget_s:
                return cur, logs
            B1 = min(B0 + B, T + 1)
            in_blk = (pool.a >= B0) & (pool.a < B1)
            free = np.nonzero(in_blk)[0]
            cur_in = in_blk[cur]
            fixed, hint = cur[~cur_in], cur[cur_in]
            r = solve_subset(pb, pool, free, fixed, hint, time_limit, workers, seed + p, verbose=False)
            old = objective(pb, pool, cur)
            acc = False
            if r["sel"] is not None:
                new = np.concatenate([fixed, r["sel"]])
                if objective(pb, pool, new) > old:
                    cur, acc = new, True
            row = {k: v for k, v in r.items() if k != "sel"}
            row.update(pass_=p, B0_h=B0 * cell / 3600, B1_h=B1 * cell / 3600, accepted=acc,
                       combined=combined_of(pb, objective(pb, pool, cur), coefficients(pb, pool)[1]))
            logs.append(row)
            if verbose:
                print(f"  [merge 块] 第{p + 1}遍 [{row['B0_h']:5.1f}h,{row['B1_h']:5.1f}h) 候选 {r['n_cand']:6d} "
                      f"{r['status']:8s} 目标 {r['obj']} 提示 {r['hint_obj']} 界 {r['bound']} "
                      f"{'接受' if acc else '—'} combined {row['combined']:.5f} {time.time() - t0:.0f}s", flush=True)
            if B1 > T:
                break
            B0 += S
    return cur, logs
