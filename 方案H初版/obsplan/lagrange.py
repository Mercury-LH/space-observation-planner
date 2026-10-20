"""新算法 H：拉格朗日价格协调（可看作多智能体竞价）。

把"每个站点每个 30 秒格最多被一个任务占用"这条耦合约束松弛掉，给每个格一个价格 μ。
松弛后问题按目标完全拆开：每颗目标在价格下独立选出最划算的 ≤3 次观测（满足 48 小时间隔），
相当于每颗目标作为一个智能体对时间格出价。被争抢的格（需求 > 1）涨价，没人要的格降价，
用次梯度法迭代。价格收敛后：
- 对偶值给出一个上界（所有合法计划都不可能超过它），用来衡量各算法离最优还有多远；
- 用"价值 - 价格"（约减收益）给候选排序做贪心，得到合法计划，再做局部搜索。
价格的含义很直观：黄昏/黎明低轨卫星受照又天黑的时段最贵，深夜只有高轨卫星可见的时段便宜。
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from .greedy import State, candidates, run_greedy
from .local_search import improve
from .problem import Problem
from .schedule import Schedule


@njit(parallel=True, cache=True)
def _subproblem(pref, t_ptr, t_win, w_st, w_k0, w_k1, Pv, frac, vc, gap, m_min, m_max,
                out_total, out_task, out_cnt):
    """每颗目标独立求解：窗口内最优 (开始, 时长)，再在窗口间做 ≤3 次、间隔 ≥48h 的链式 DP。
    窗口间隔用放松的判定（前一窗口最早结束 + 48h <= 后一窗口最晚开始），保证对偶值是合法上界。"""
    N = len(t_ptr) - 1
    for i in prange(N):
        p0, p1 = t_ptr[i], t_ptr[i + 1]
        nw = p1 - p0
        out_cnt[i] = 0
        out_total[i] = 0.0
        if nw == 0:
            continue
        pi = np.empty(nw)
        ba = np.empty(nw, np.int64)
        bm = np.empty(nw, np.int64)
        k0s = np.empty(nw, np.int64)
        for q in range(nw):
            w = t_win[p0 + q]
            s, k0, k1 = w_st[w], w_k0[w], w_k1[w]
            k0s[q] = k0
            best, besta, bestm = -1e18, k0, m_min
            for m in range(m_min, m_max + 1):
                if k1 - k0 < m:
                    break
                val = Pv[i] * frac[m]
                for a in range(k0, k1 - m + 1):
                    r = val - (pref[s, a + m + 1] - pref[s, a])
                    if r > best:
                        best, besta, bestm = r, a, m
            pi[q], ba[q], bm[q] = best, besta, bestm
        # r_q：满足放松间隔的前序窗口个数（窗口按 k0 排序，前序窗口是前缀）
        rq = np.empty(nw, np.int64)
        for q in range(nw):
            lim = w_k1[t_win[p0 + q]] - m_min - gap - m_min
            lo, hi = 0, nw
            while lo < hi:
                mid = (lo + hi) // 2
                if k0s[mid] <= lim:
                    lo = mid + 1
                else:
                    hi = mid
            rq[q] = lo
        f1 = pi.copy()
        pm1 = np.full(nw + 1, -1e18)
        am1 = np.full(nw + 1, -1, np.int64)
        for q in range(nw):
            if f1[q] > pm1[q]:
                pm1[q + 1], am1[q + 1] = f1[q], q
            else:
                pm1[q + 1], am1[q + 1] = pm1[q], am1[q]
        f2 = np.full(nw, -1e18)
        for q in range(nw):
            if rq[q] > 0:
                f2[q] = pi[q] + pm1[rq[q]]
        pm2 = np.full(nw + 1, -1e18)
        am2 = np.full(nw + 1, -1, np.int64)
        for q in range(nw):
            if f2[q] > pm2[q]:
                pm2[q + 1], am2[q + 1] = f2[q], q
            else:
                pm2[q + 1], am2[q + 1] = pm2[q], am2[q]
        best, bq, bk = -1e18, -1, 0
        for q in range(nw):
            if f1[q] > best:
                best, bq, bk = f1[q], q, 1
            if f2[q] > best:
                best, bq, bk = f2[q], q, 2
            if rq[q] > 0 and pm2[rq[q]] > -1e17 and pi[q] + pm2[rq[q]] > best:
                best, bq, bk = pi[q] + pm2[rq[q]], q, 3
        total = best + vc
        if total <= 0:
            continue
        out_total[i] = total
        q = bq
        for k in range(bk, 0, -1):
            w = t_win[p0 + q]
            c = out_cnt[i]
            out_task[i, c, 0], out_task[i, c, 1], out_task[i, c, 2] = w_st[w], ba[q], bm[q]
            out_cnt[i] = c + 1
            if k == 3:
                q = am2[rq[q]]
            elif k == 2:
                q = am1[rq[q]]


@njit(cache=True)
def _demand(out_task, out_cnt, S, T):
    cnt = np.zeros((S, T))
    for i in range(len(out_cnt)):
        for c in range(out_cnt[i]):
            s, a, m = out_task[i, c, 0], out_task[i, c, 1], out_task[i, c, 2]
            for x in range(a, a + m + 1):
                cnt[s, x] += 1.0
    return cnt


@njit(parallel=True, cache=True)
def _candidate_cost(pref, c_win, c_m, w_st, w_k0, w_k1):
    """每个候选 (窗口, 时长) 在窗口内最便宜的放置成本。"""
    out = np.empty(len(c_win))
    for j in prange(len(c_win)):
        w, m = c_win[j], c_m[j]
        s, k0, k1 = w_st[w], w_k0[w], w_k1[w]
        best = 1e18
        for a in range(k0, k1 - m + 1):
            v = pref[s, a + m + 1] - pref[s, a]
            if v < best:
                best = v
        out[j] = best
    return out


def objective(pb: Problem, st: State) -> float:
    alive = st.alive[:st.n]
    tg, m = st.tg[:st.n][alive], st.m[:st.n][alive]
    return float(pb.vc * np.unique(tg).size + pb.vs * (pb.P[tg] * pb.frac[m]).sum())


def recover(pb: Problem, mu: np.ndarray, cand) -> State:
    """价格引导贪心：先按约减收益（价值 - 价格）放置收益为正的候选，再按真实价值补空。"""
    cw, cm = cand
    pref = np.concatenate([np.zeros((mu.shape[0], 1)), np.cumsum(mu, axis=1)], axis=1)
    cost = _candidate_cost(pref, cw, cm, pb.w_st, pb.w_k0, pb.w_k1)
    v = pb.vs * pb.P[pb.w_tg[cw]] * pb.frac[cm]
    st = run_greedy(pb, cw, cm, v + pb.vc - cost, v - cost, best_fit=True, cutoff=0.0, price=mu)
    run_greedy(pb, cw, cm, v + pb.vc, v, best_fit=True, state=st, price=mu)
    return st


def solve(pb: Problem, iters: int = 150, recover_at=None, local_search=True,
          verbose=True, name="H 价格协调（新算法）", progress=None) -> tuple[Schedule, dict]:
    """progress(fraction, text)：可选的进度回调（产品界面用）。"""
    t0 = time.time()
    if recover_at is None:                      # 在整个迭代过程中均匀做可行解恢复
        recover_at = tuple(range(0, iters, 50)) + (iters - 1,)
    S, T = pb.n_stations, pb.n_samples + 2
    mu = np.zeros((S, T))
    cand = candidates(pb)
    N = pb.n_targets
    out_total = np.zeros(N)
    out_task = np.zeros((N, 3, 3), np.int64)
    out_cnt = np.zeros(N, np.int64)
    Pv = pb.vs * pb.P
    best_ub, best_lb, best_state, best_mu = np.inf, -np.inf, None, mu.copy()
    theta, stall, hist = 1.0, 0, []
    for it in range(iters):
        if it in recover_at:
            st = recover(pb, mu, cand)
            lb = objective(pb, st)
            if lb > best_lb:
                best_lb, best_state, best_mu = lb, st, mu.copy()
        pref = np.concatenate([np.zeros((S, 1)), np.cumsum(mu, axis=1)], axis=1)
        _subproblem(pref, pb.t_ptr, pb.t_win, pb.w_st, pb.w_k0, pb.w_k1, Pv, pb.frac, pb.vc, pb.gap,
                    pb.m_min, pb.m_max, out_total, out_task, out_cnt)
        L = mu.sum() + out_total.sum()
        if L < best_ub - 1e-9:
            best_ub, stall = L, 0
        else:
            stall += 1
            if stall >= 8:
                theta, stall = max(theta / 2, 1e-3), 0
        g = _demand(out_task, out_cnt, S, T) - 1.0
        g[(mu <= 0) & (g < 0)] = 0.0
        norm = float((g * g).sum())
        hist.append((it, L, best_lb, theta))
        if progress is not None and it % 10 == 0:
            progress(0.9 * it / iters, f"价格迭代 {it}/{iters}，当前最好计划 {best_lb:.4f}")
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"  [H] it {it:3d}  对偶上界 {L:.4f}  最好可行解 {best_lb:.4f}  步长系数 {theta:.3f}  {time.time() - t0:.0f}s")
        if norm == 0:
            break
        mu = np.maximum(0.0, mu + theta * (L - best_lb) / norm * g)
    ls_stats = {}
    if progress is not None:
        progress(0.92, "局部搜索：挤让插入、压紧、补空、延长……")
    if local_search:
        ls_stats = improve(pb, best_state)
    sch = best_state.schedule(pb, name, time.time() - t0)
    info = {"ub_relaxed": best_ub, "lb_before_ls": best_lb, "history": hist, "ls": ls_stats, "mu": best_mu}
    return sch, info
