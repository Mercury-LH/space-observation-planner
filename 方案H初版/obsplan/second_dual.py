"""更紧的按秒上界：用每个窗口的按秒真实可见区间，在细网格（默认 10 秒）上算拉格朗日对偶上界。

bounds.py 里原来的按秒上界把每个 30 秒网格窗口两端各放宽一整格（共 60 秒），而真实能多用的平均只有约 29 秒，
所以偏松（1.005828）。这里改成：
1. 每个窗口用 second_bounds.all_bounds 探查出的真实区间 [L, R]（秒），向外取整到细网格（每端最多多算 c−1 秒）；
   网格上只有 4 个采样点、按秒够 120 秒的短窗口同样探查后加入；
2. 细网格上任务占用 = 观测时长 + 30 秒切换（30 秒网格上切换正好一格，细网格上是 30/c 格，所以子问题单独写了一份）；
3. 其余与 bounds.dual_upper_bound 相同：给每个站每个细格一个价格 mu >= 0，
   L(mu) = 价格之和 + 每颗目标在这组价格下各自最优的"价值 − 价格"之和，对任何 mu >= 0 都是上界。

为什么对按秒计划成立：按秒合法的任务一定落在某个（含短窗口的）真实区间里，向外取整后的窗口包含它；
价格在细格内是常数、得分对时长是线性的，120 / 240 / 30 秒都是细格的整数倍，所以单个窗口内"价值 − 价格"的
最大值在细格点上取到；48 小时间隔用窗口级的放松判定。"合法"指本程序的可见性模型（含安全余量）。
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from numba import njit, prange

from .bounds import _short_windows
from .lagrange import balanced_order, target_work
from .problem import Problem
from .refine import probe_bounds
from .visibility import Windows


def fine_problem(pb: Problem, wL: np.ndarray, wR: np.ndarray, cell: int = 10, verbose: bool = True) -> Problem:
    """细网格上的放宽问题（窗口 = 真实区间向外取整，含按秒才够长的短窗口）。"""
    cfg = pb.cfg
    assert cfg.cell_s % cell == 0 and cfg.obs_min_s % cell == 0 and cfg.obs_max_s % cell == 0 and cfg.switch_s % cell == 0
    st, tg, k0, k1 = (np.asarray(x).astype(np.int64) for x in _short_windows(pb))
    keep = k1 - k0 == pb.m_min - 1
    st, tg, k0, k1 = st[keep], tg[keep], k0[keep], k1[keep]
    sL, sR = probe_bounds(cfg, pb.ds, st, tg, k0, k1)
    ok = sR - sL >= cfg.obs_min_s
    if verbose:
        print(f"  [细网格] 短窗口 {len(st)} 个，其中按秒够 {cfg.obs_min_s} 秒的 {int(ok.sum())} 个", flush=True)
    w_st = np.concatenate([pb.w_st, st[ok]])
    w_tg = np.concatenate([pb.w_tg, tg[ok]])
    L = np.concatenate([wL, sL[ok]])
    R = np.concatenate([wR, sR[ok]])
    fcfg = replace(cfg, cell_s=cell)
    last = fcfg.n_cells
    f0 = np.maximum(L // cell, 0)
    f1 = np.minimum(-(-R // cell), last)
    order = np.lexsort((f0, w_st))
    r = cfg.cell_s // cell
    dark = [[(max(a * r - r, 0), min(b * r + r, last)) for a, b in runs] for runs in pb.win.dark]
    win = Windows(w_st[order].astype(np.int16), w_tg[order].astype(np.int32), f0[order].astype(np.int32),
                  f1[order].astype(np.int32), dark, last + 1, pb.win.sgp4_failed)
    return Problem.build(fcfg, pb.ds, win)


@njit(parallel=True, cache=True)
def _kernel(order, pref, t_ptr, t_win, w_st, w_k0, w_k1, Pv, frac, vc, gap, m_min, m_max, sw,
            out_total, out_task, out_cnt):
    """每颗目标独立求解（同 lagrange._subproblem_kernel 的无天气分支，切换时间为 sw 格）。"""
    N = len(t_ptr) - 1
    for j in prange(N):
        i = order[j]
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
            b, a1, m1 = -1e18, k0, m_min
            for m in range(m_min, m_max + 1):
                if k1 - k0 < m:
                    break
                val = Pv[i] * frac[m]
                for a in range(k0, k1 - m + 1):
                    r = val - (pref[s, a + m + sw] - pref[s, a])
                    if r > b:
                        b, a1, m1 = r, a, m
            pi[q], ba[q], bm[q] = b, a1, m1
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
        pm1 = np.full(nw + 1, -1e18)
        am1 = np.full(nw + 1, -1, np.int64)
        for q in range(nw):
            if pi[q] > pm1[q]:
                pm1[q + 1], am1[q + 1] = pi[q], q
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
            if pi[q] > best:
                best, bq, bk = pi[q], q, 1
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
def _demand(out_task, out_cnt, S, T, sw):
    cnt = np.zeros((S, T))
    for i in range(len(out_cnt)):
        for c in range(out_cnt[i]):
            s, a, m = out_task[i, c, 0], out_task[i, c, 1], out_task[i, c, 2]
            for x in range(a, a + m + sw):
                cnt[s, x] += 1.0
    return cnt


class Dual:
    def __init__(self, pb: Problem):
        if 3 * (pb.gap + pb.m_min) + pb.m_min <= pb.n_samples - 1:
            raise NotImplementedError("规划期长到一颗目标可能观测 4 次以上；子问题只算到 3 次")
        self.pb = pb
        self.sw = pb.cfg.switch_s // pb.cfg.cell_s
        self.S, self.T = pb.n_stations, pb.n_samples + self.sw + 1
        N = pb.n_targets
        self.order = balanced_order(target_work(pb.t_ptr, pb.t_win, pb.w_k0, pb.w_k1))
        self.out_total = np.zeros(N)
        self.out_task = np.zeros((N, 3, 3), np.int64)
        self.out_cnt = np.zeros(N, np.int64)

    def value(self, mu: np.ndarray) -> float:
        """L(mu)：任何 mu >= 0 都给出合法上界。"""
        pb = self.pb
        mu = np.asarray(mu, np.float64)
        if mu.shape != (self.S, self.T) or not np.isfinite(mu).all() or (mu < 0).any():
            raise ValueError("价格必须是 (站点数, 细格数) 的非负有限数组")
        pref = np.concatenate([np.zeros((self.S, 1)), np.cumsum(mu, axis=1)], axis=1)
        _kernel(self.order, pref, pb.t_ptr, pb.t_win, pb.w_st, pb.w_k0, pb.w_k1, pb.vs * pb.P, pb.frac, pb.vc,
                pb.gap, pb.m_min, pb.m_max, self.sw, self.out_total, self.out_task, self.out_cnt)
        return float(mu.sum() + self.out_total.sum())

    def subgradient(self, mu: np.ndarray) -> np.ndarray:
        g = _demand(self.out_task, self.out_cnt, self.S, self.T, self.sw) - 1.0
        g[(mu <= 0) & (g < 0)] = 0.0
        return g


def dual_upper_bound(pb: Problem, target: float, iters: int = 2000, mu0: np.ndarray | None = None, theta: float = 1.0,
                     patience: int = 20, decay: float = 0.7, checkpoint=None, verbose: bool = True,
                     log_every: int = 50, deadline: float | None = None, beta: float = 0.0) -> dict:
    """次梯度法（同 bounds.dual_upper_bound）。返回过程中最小的 L 和取到它的价格。
    beta > 0 时用带惯性的方向（本次次梯度 + beta × 上次方向），价格多的时候收敛快一些；不影响上界的合法性。"""
    t0 = time.time()
    d = Dual(pb)
    mu = np.zeros((d.S, d.T)) if mu0 is None else np.asarray(mu0, np.float64).copy()
    best, best_mu, stall, hist = np.inf, mu.copy(), 0, []
    dirn = None

    def save(it):
        if checkpoint is not None:
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(checkpoint, mu=best_mu, best=best, theta=theta, it=it, target=target)

    it = 0
    for it in range(iters):
        L = d.value(mu)
        if L < best - 1e-7:
            best, best_mu, stall = L, mu.copy(), 0
        else:
            stall += 1
            if stall >= patience:
                theta, stall = max(theta * decay, 1e-4), 0
        g = d.subgradient(mu)
        if beta > 0:
            dirn = g if dirn is None else g + beta * dirn
            g = dirn
        norm = float((g * g).sum())
        if it % log_every == 0 or it == iters - 1:
            hist.append((it, L, best, theta))
            if verbose:
                print(f"  [细网格对偶上界] it {it:5d}  L {L:.6f}  最好 {best:.6f}  步长系数 {theta:.4f}  "
                      f"超载格 {int((g > 0).sum())}  {time.time() - t0:.0f}s", flush=True)
        if it % 100 == 99:
            save(it + 1)
        if norm == 0 or L <= target or (deadline is not None and time.time() > deadline):
            break
        mu = np.maximum(0.0, mu + theta * (L - target) / norm * g)
    save(it + 1)
    return {"combined_ub": float(best), "mu": best_mu, "iters": it + 1, "theta": theta, "target": float(target),
            "history": hist, "seconds": time.time() - t0}


def warm_start(pb_fine: Problem, mu_grid: np.ndarray, grid_cell: int) -> np.ndarray:
    """把 30 秒网格上的价格摊到细格上（每个细格 = 所在粗格价格 × 细格长度 / 粗格长度）。"""
    c = pb_fine.cfg.cell_s
    sw = pb_fine.cfg.switch_s // c
    T = pb_fine.n_samples + sw + 1
    idx = np.minimum(np.arange(T) * c // grid_cell, mu_grid.shape[1] - 1)
    return np.maximum(mu_grid[:, idx] * (c / grid_cell), 0.0)
