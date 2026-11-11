"""按秒跨站弹射链：在一份按秒计划上，成批地"多塞进一次高优先级重复观测"。

为什么做这个：当前最好计划里各站黑夜时间已经 100% 用满，低、中优先级目标基本都只观测一次 120 秒，
多出来的时间全在高优先级任务的"延长"上（延长 150 秒值 2.5 分）。同样 150 秒如果换成一次新的高优先级
重复观测（120 秒 + 30 秒切换）值 3 分——每换成一次涨 0.5 分（综合约 +0.000008）。难点是新观测只能放在
它自己的可见窗口里，那里通常被别的任务占着。

做法（和二分图匹配里找增广路是一个思路）：
1. 每个"站 × 夜晚"是一条时间线，任务顺序固定。把全部任务压到 120 秒，前向算每个任务最早结束、
   后向算最晚开始，于是"在第 k 个空当里再塞一个窗口为 [a, b] 的任务行不行""把第 k 个任务换成它行不行"
   都是 O(1) 的判断。
2. 对一颗还没观测满 3 次的高优先级目标 R 做广度优先搜索：R 的某个窗口能直接塞进去 → 成功；
   否则 R 顶掉那里的任务 X，X 去它自己的其他窗口（可以换站、换夜晚）找位置，还不行就再顶掉 Y……
   一条链上每个"站 × 夜晚"只动一次。
3. 找到链后，对链上涉及的时间线解一次线性规划（顺序固定，重新分配起止时间；同目标 48 小时间隔：
   两次观测都在这次规划里的写成约束，另一头不在的按它当前的起止时间卡死），总得分变高才接受。

所有时间都是整秒；窗口用 obsplan/second_bounds.py 探查的真实可见区间（与按秒精修同一判定）。
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
from numba import njit

from .problem import Problem
from .schedule import Schedule

BIG = 10 ** 12


@njit(cache=True)
def _unit_ab(seq, n, tk_w, tk_s, tk_e, tk_tg, tg_tasks, wL, wR, gap, dmin, sw, A, B):
    """一条时间线：全部任务压到最短时长时，空当 k（第 k−1 与第 k 个任务之间）可用的时间范围 [A[k], B[k]]。
    每个任务的可用范围 = 它的窗口 ∩ 同目标其他观测（按当前起止）留出的 48 小时间隔。返回这条线本身是否可行。"""
    ok = True
    prev = -BIG
    A[0] = -BIG
    lo = np.empty(n, np.int64)
    hi = np.empty(n, np.int64)
    for j in range(n):
        p = seq[j]
        if tk_w[p] >= 0:
            l, h = wL[tk_w[p]], wR[tk_w[p]]
        else:
            l, h = tk_s[p], tk_e[p]
        x = tk_tg[p]
        for c in range(3):
            q = tg_tasks[x, c]
            if q < 0 or q == p:
                continue
            if tk_s[q] > tk_s[p]:
                if tk_s[q] - gap < h:
                    h = tk_s[q] - gap
            elif tk_e[q] + gap > l:
                l = tk_e[q] + gap
        lo[j], hi[j] = l, h
        s = max(l, prev + sw)
        ef = s + dmin
        if ef > h:
            ok = False
        prev = ef
        A[j + 1] = ef + sw
    nxt = BIG
    B[n] = BIG
    for j in range(n - 1, -1, -1):
        e = min(hi[j], nxt - sw)
        ls = e - dmin
        if ls < lo[j]:
            ok = False
        nxt = ls
        B[j] = ls - sw
    return ok


@njit(cache=True)
def _direct_fit(u, a, b, A2, B2, ulen, dmin):
    """窗口 [a, b] 的新任务能直接塞进时间线 u 的哪个空当；没有返回 −1。"""
    n = ulen[u]
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if B2[u, mid] >= a + dmin:
            hi = mid
        else:
            lo = mid + 1
    k = lo
    while k <= n and A2[u, k] <= b - dmin:
        if min(b, B2[u, k]) - max(a, A2[u, k]) >= dmin:
            return k
        k += 1
    return -1


@njit(cache=True)
def _slots(x, skip, w, tk_s, tk_e, tg_tasks, wL, wR, gap, dmin, out_a, out_b):
    """目标 x（不算任务 skip）再加一次观测、用窗口 w：与已有观测保持 48 小时后剩下的可用区间。返回区间个数。"""
    os_ = np.empty(3, np.int64)
    oe_ = np.empty(3, np.int64)
    c = 0
    for j in range(3):
        q = tg_tasks[x, j]
        if q >= 0 and q != skip:
            os_[c], oe_[c] = tk_s[q], tk_e[q]
            c += 1
    if c >= 3:
        return 0
    for i in range(c):                       # 按开始时间排序（最多 3 个）
        for j in range(i + 1, c):
            if os_[j] < os_[i]:
                os_[i], os_[j] = os_[j], os_[i]
                oe_[i], oe_[j] = oe_[j], oe_[i]
    n = 0
    for g in range(c + 1):
        a = wL[w]
        b = wR[w]
        if g > 0 and oe_[g - 1] + gap > a:
            a = oe_[g - 1] + gap
        if g < c and os_[g] - gap < b:
            b = os_[g] - gap
        if b - a >= dmin:
            out_a[n], out_b[n] = a, b
            n += 1
    return n


@njit(cache=True)
def _search(new_tid, max_depth, max_q, t_ptr, t_win, w_unit, wL, wR, tk_tg, tk_s, tk_e, tk_w, tg_tasks,
            seq2, ulen, A2, B2, unit_ok, gap, dmin, visited, stamp,
            q_task, q_par, q_w, q_k, q_a, q_unit, q_depth, P, min_eject_p, max_eject_p):
    """广度优先找弹射链。队列第 e 项 = "任务 q_task[e] 需要找位置"；它是被父项的任务顶出来的：
    父项任务用窗口 q_w[e]、放在时间线 q_unit[e] 的第 q_k[e] 个位置（参考开始时间 q_a[e]）。
    返回 (末项 e, 末项任务直接塞进去用的窗口, 空当下标, 参考开始时间)；找不到返回 (−1, …)。"""
    q_task[0], q_par[0], q_unit[0], q_depth[0] = new_tid, -1, -1, 0
    head, tail = 0, 1
    sa = np.empty(4, np.int64)
    sb = np.empty(4, np.int64)
    while head < tail:
        e = head
        head += 1
        T = q_task[e]
        x = tk_tg[T]
        # 第一遍：能不能直接塞进去
        for wi in range(t_ptr[x], t_ptr[x + 1]):
            w = t_win[wi]
            u = w_unit[w]
            if u < 0 or not unit_ok[u]:
                continue
            used = False
            p = e
            while p > 0:
                if q_unit[p] == u:
                    used = True
                    break
                p = q_par[p]
            if used:
                continue
            ns = _slots(x, T, w, tk_s, tk_e, tg_tasks, wL, wR, gap, dmin, sa, sb)
            for g in range(ns):
                k = _direct_fit(u, sa[g], sb[g], A2, B2, ulen, dmin)
                if k >= 0:
                    return e, w, k, sa[g]
        if q_depth[e] >= max_depth:
            continue
        # 第二遍：顶掉一个任务
        for wi in range(t_ptr[x], t_ptr[x + 1]):
            w = t_win[wi]
            u = w_unit[w]
            if u < 0 or not unit_ok[u]:
                continue
            used = False
            p = e
            while p > 0:
                if q_unit[p] == u:
                    used = True
                    break
                p = q_par[p]
            if used:
                continue
            ns = _slots(x, T, w, tk_s, tk_e, tg_tasks, wL, wR, gap, dmin, sa, sb)
            n = ulen[u]
            for g in range(ns):
                a, b = sa[g], sb[g]
                lo, hi = 0, n
                while lo < hi:                      # 第一个 B[k+1] >= a + dmin 的 k
                    mid = (lo + hi) // 2
                    if B2[u, mid + 1] >= a + dmin:
                        hi = mid
                    else:
                        lo = mid + 1
                k = lo
                while k < n and A2[u, k] <= b - dmin:
                    Y = seq2[u, k]
                    if visited[Y] != stamp and tk_w[Y] >= 0 and min_eject_p <= P[tk_tg[Y]] <= max_eject_p \
                            and min(b, B2[u, k + 1]) - max(a, A2[u, k]) >= dmin:
                        visited[Y] = stamp
                        if tail < max_q:
                            q_task[tail], q_par[tail], q_w[tail], q_k[tail] = Y, e, w, k
                            q_a[tail], q_unit[tail], q_depth[tail] = a, u, q_depth[e] + 1
                            tail += 1
                    k += 1
    return -1, -1, -1, -1


class Timeline:
    """按秒计划的可变表示：每个"站 × 夜晚"一条有序任务线。"""

    def __init__(self, pb: Problem, wL: np.ndarray, wR: np.ndarray, sch: Schedule, extra: int = 4000):
        cfg = pb.cfg
        self.pb = pb
        self.wL, self.wR = wL.astype(np.int64), wR.astype(np.int64)
        self.dmin, self.dmax, self.sw = int(cfg.obs_min_s), int(cfg.obs_max_s), int(cfg.switch_s)
        self.gap = int(round(cfg.repeat_gap_h * 3600))
        # 站 × 夜晚
        self.unit_off = np.zeros(pb.n_stations + 1, np.int64)
        starts = []
        for s, runs in enumerate(pb.win.dark):
            self.unit_off[s + 1] = self.unit_off[s] + len(runs)
            starts.append(np.array([a for a, _ in runs], np.int64))
        self.n_units = int(self.unit_off[-1])
        self.w_unit = np.full(len(pb.w_st), -1, np.int64)
        for s in range(pb.n_stations):
            m = pb.w_st == s
            r = np.searchsorted(starts[s], pb.w_k0[m], side="right") - 1
            self.w_unit[m] = self.unit_off[s] + np.maximum(r, 0)
        n = len(sch)
        cap = n + extra
        self.n = n
        self.tk_st = np.zeros(cap, np.int64)
        self.tk_tg = np.zeros(cap, np.int64)
        self.tk_s = np.zeros(cap, np.int64)
        self.tk_e = np.zeros(cap, np.int64)
        self.tk_w = np.full(cap, -1, np.int64)
        self.tk_unit = np.full(cap, -1, np.int64)
        self.alive = np.zeros(cap, bool)
        self.tk_st[:n], self.tk_tg[:n] = sch.station, sch.target
        self.tk_s[:n] = np.rint(sch.start_s).astype(np.int64)
        self.tk_e[:n] = self.tk_s[:n] + np.rint(sch.dur_s).astype(np.int64)
        self.alive[:n] = True
        self.P = pb.P.astype(np.float64)
        self.unmatched = 0
        for i in range(n):
            t, s = int(sch.target[i]), int(sch.station[i])
            ws = pb.t_win[pb.t_ptr[t]:pb.t_ptr[t + 1]]
            hit = ws[(pb.w_st[ws] == s) & (self.wL[ws] <= self.tk_s[i]) & (self.tk_e[i] <= self.wR[ws])]
            if len(hit):
                self.tk_w[i] = hit[0]
                self.tk_unit[i] = self.w_unit[hit[0]]
            else:                                   # 不在任何网格窗口里（按秒才可见的短窗口）：原样固定
                self.unmatched += 1
                r = np.searchsorted(starts[s], self.tk_s[i] // cfg.cell_s, side="right") - 1
                self.tk_unit[i] = self.unit_off[s] + max(int(r), 0)
        self.tg_tasks = np.full((pb.n_targets, 3), -1, np.int64)
        self.tg_cnt = np.zeros(pb.n_targets, np.int64)
        for i in range(n):
            t = self.tk_tg[i]
            self.tg_tasks[t, self.tg_cnt[t]] = i
            self.tg_cnt[t] += 1
        self.seq = [[] for _ in range(self.n_units)]
        for i in np.argsort(self.tk_s[:n], kind="stable"):
            self.seq[self.tk_unit[i]].append(int(i))
        self.maxlen = max(len(s) for s in self.seq) + 80
        self.seq2 = np.full((self.n_units, self.maxlen), -1, np.int64)
        self.ulen = np.zeros(self.n_units, np.int64)
        self.A2 = np.zeros((self.n_units, self.maxlen + 1), np.int64)
        self.B2 = np.zeros((self.n_units, self.maxlen + 1), np.int64)
        self.unit_ok = np.ones(self.n_units, bool)
        for u in range(self.n_units):
            self.refresh(u)
        self.visited = np.zeros(cap, np.int64)
        self.stamp = 0
        mq = 60000
        self.q = [np.zeros(mq, np.int64) for _ in range(7)]
        self.max_q = mq
        self.stats = defaultdict(int)

    # ---------- 基础 ----------
    def refresh(self, u: int) -> None:
        seq = self.seq[u]
        n = len(seq)
        if n > self.maxlen - 2:
            self.unit_ok[u] = False
            return
        self.seq2[u, :n] = seq
        self.ulen[u] = n
        ok = _unit_ab(self.seq2[u], n, self.tk_w, self.tk_s, self.tk_e, self.tk_tg, self.tg_tasks,
                      self.wL, self.wR, self.gap, self.dmin, self.sw, self.A2[u], self.B2[u])
        self.unit_ok[u] = bool(ok)

    def score_points(self) -> float:
        a = np.nonzero(self.alive[:self.n])[0]
        d = self.tk_e[a] - self.tk_s[a]
        return float((self.P[self.tk_tg[a]] * (0.6 + 0.4 * (d - self.dmin) / (self.dmax - self.dmin))).sum())

    def schedule(self, algo: str = "按秒跨站弹射链") -> tuple[Schedule, np.ndarray, np.ndarray]:
        """返回计划和每个任务的窗口真实区间 (L, R)（给全局按秒精修用）。"""
        a = np.nonzero(self.alive[:self.n])[0]
        sch = Schedule(self.tk_st[a].copy(), self.tk_tg[a].copy(), self.tk_s[a].copy(),
                       (self.tk_e[a] - self.tk_s[a]).copy(), algo=algo)
        w = self.tk_w[a]
        L = np.where(w >= 0, self.wL[np.maximum(w, 0)], self.tk_s[a])
        R = np.where(w >= 0, self.wR[np.maximum(w, 0)], self.tk_e[a])
        return sch, L, R

    # ---------- 线性规划：给定各时间线的新顺序，重新定起止 ----------
    def lp(self, new_seqs: dict, override: dict):
        """new_seqs：{时间线: 任务编号的新顺序}；override：{任务: (新窗口, 参考开始时间)}（新任务或换了位置的任务）。
        返回 (新旧得分差, {任务: (开始, 结束)})；不可行返回 None。"""
        from ortools.linear_solver import pywraplp
        dmin, dmax, sw, gap = self.dmin, self.dmax, self.sw, self.gap
        tasks = [p for seq in new_seqs.values() for p in seq]
        inlp = set(tasks)
        lp = pywraplp.Solver.CreateSolver("GLOP")
        sv, dv = {}, {}
        ref, lim = {}, {}
        by_t = defaultdict(list)
        obj = lp.Objective()
        for p in tasks:
            if p in override:
                w, r = override[p]
            else:
                w, r = int(self.tk_w[p]), int(self.tk_s[p])
            ref[p] = r
            if w >= 0:
                l, h = int(self.wL[w]), int(self.wR[w])
            else:
                l, h = int(self.tk_s[p]), int(self.tk_e[p])
            x = int(self.tk_tg[p])
            for q in self.tg_tasks[x]:
                q = int(q)
                if q < 0 or q == p or q in inlp:
                    continue
                if self.tk_s[q] > r:
                    h = min(h, int(self.tk_s[q]) - gap)
                else:
                    l = max(l, int(self.tk_e[q]) + gap)
            if h - l < dmin:
                return None
            lim[p] = (l, h)
            sv[p] = lp.NumVar(float(l), float(h - dmin), "")
            dv[p] = lp.NumVar(float(dmin), float(dmax), "")
            c = lp.RowConstraint(-lp.infinity(), float(h))
            c.SetCoefficient(sv[p], 1.0)
            c.SetCoefficient(dv[p], 1.0)
            obj.SetCoefficient(dv[p], float(self.P[x]))
            by_t[x].append(p)
        pairs = []
        for seq in new_seqs.values():
            for p, q in zip(seq[:-1], seq[1:]):
                pairs.append((p, q, sw))
        for x, ps in by_t.items():
            if len(ps) > 1:
                ps = sorted(ps, key=lambda p: ref[p])
                for p, q in zip(ps[:-1], ps[1:]):
                    pairs.append((p, q, gap))
        for p, q, g in pairs:
            c = lp.RowConstraint(float(g), lp.infinity())
            c.SetCoefficient(sv[q], 1.0)
            c.SetCoefficient(sv[p], -1.0)
            c.SetCoefficient(dv[p], -1.0)
        obj.SetMaximization()
        if lp.Solve() != pywraplp.Solver.OPTIMAL:
            return None
        out = {}
        for p in tasks:
            s = int(round(sv[p].solution_value()))
            d = int(round(dv[p].solution_value()))
            if d < dmin or d > dmax:
                return None
            out[p] = (s, s + d)
        for p, q, g in pairs:                       # 取整后复核
            if out[q][0] - out[p][1] < g:
                return None
        for p in tasks:
            if out[p][0] < lim[p][0] or out[p][1] > lim[p][1]:
                return None
        k = 0.4 / (dmax - dmin)
        new = sum(self.P[self.tk_tg[p]] * (0.6 + k * (out[p][1] - out[p][0] - dmin)) for p in tasks)
        return new, out

    def units_score(self, units) -> float:
        k = 0.4 / (self.dmax - self.dmin)
        return sum(self.P[self.tk_tg[p]] * (0.6 + k * (self.tk_e[p] - self.tk_s[p] - self.dmin))
                   for u in units for p in self.seq[u])

    def commit(self, new_seqs: dict, override: dict, times: dict) -> None:
        touched = set(new_seqs)
        for u, seq in new_seqs.items():
            self.seq[u] = list(seq)
        for p, (w, _) in override.items():
            self.tk_w[p] = w
            self.tk_unit[p] = self.w_unit[w]
            self.tk_st[p] = self.pb.w_st[w]
        for p, (s, e) in times.items():
            if self.tk_s[p] != s or self.tk_e[p] != e:
                self.tk_s[p], self.tk_e[p] = s, e
                for q in self.tg_tasks[self.tk_tg[p]]:
                    if q >= 0 and q != p:
                        touched.add(int(self.tk_unit[q]))
        for u in touched:
            self.refresh(u)

    # ---------- 弹射链 ----------
    def try_add(self, x: int, max_depth: int = 2, min_gain: float = 1e-6, eject_p=(0.0, 9.0)) -> float:
        """给目标 x 多加一次观测。成功返回得分增量（分），否则 0。"""
        if self.tg_cnt[x] >= 3:
            return 0.0
        tid = self.n
        self.tk_tg[tid] = x
        self.tk_w[tid] = -1
        self.stamp += 1
        qt, qp, qw, qk, qa, qu, qd = self.q
        e, w, k, a = _search(tid, max_depth, self.max_q, self.pb.t_ptr, self.pb.t_win, self.w_unit, self.wL, self.wR,
                             self.tk_tg, self.tk_s, self.tk_e, self.tk_w, self.tg_tasks, self.seq2, self.ulen,
                             self.A2, self.B2, self.unit_ok, self.gap, self.dmin, self.visited, self.stamp,
                             qt, qp, qw, qk, qa, qu, qd, self.P, eject_p[0], eject_p[1])
        if e < 0:
            self.stats["找不到链"] += 1
            return 0.0
        new_seqs, override = {}, {}
        u = int(self.w_unit[w])
        seq = list(self.seq[u])
        seq.insert(int(k), int(qt[e]))
        new_seqs[u] = seq
        override[int(qt[e])] = (int(w), int(a))
        depth = 0
        while e > 0:
            par = int(qp[e])
            u = int(qu[e])
            seq = list(self.seq[u])
            assert seq[int(qk[e])] == int(qt[e])
            seq[int(qk[e])] = int(qt[par])
            new_seqs[u] = seq
            override[int(qt[par])] = (int(qw[e]), int(qa[e]))
            e = par
            depth += 1
        old = self.units_score(new_seqs.keys())
        res = self.lp(new_seqs, override)
        if res is None:
            self.stats["规划不可行"] += 1
            return 0.0
        new, times = res
        gain = new - old
        if gain < min_gain:
            self.stats["不划算"] += 1
            return 0.0
        # 接受：登记新任务
        self.n += 1
        self.alive[tid] = True
        self.tg_tasks[x, self.tg_cnt[x]] = tid
        self.tg_cnt[x] += 1
        self.commit(new_seqs, override, times)
        self.stats[f"成功_深度{depth}"] += 1
        return float(gain)


def run_pass(tl: Timeline, targets, max_depth: int = 2, verbose: bool = True, log_every: int = 100,
             deadline: float | None = None, eject_p=(0.0, 9.0)) -> dict:
    t0 = time.time()
    total, ok = 0.0, 0
    for n, x in enumerate(targets):
        if deadline is not None and time.time() > deadline:
            break
        g = tl.try_add(int(x), max_depth=max_depth, eject_p=eject_p)
        if g > 0:
            total += g
            ok += 1
        if verbose and (n + 1) % log_every == 0:
            print(f"  [弹射链] {n + 1}/{len(targets)}  成功 {ok}  累计 +{total:.1f} 分  {dict(tl.stats)}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    return {"tried": len(targets), "added": ok, "gain_points": total, "seconds": time.time() - t0}
