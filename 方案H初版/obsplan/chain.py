"""按秒跨站弹射链：在一份按秒计划上，成批地"多塞进一次高优先级重复观测"。

为什么做这个：当前最好计划里各站黑夜时间已经 100% 用满，低、中优先级目标基本都只观测一次 120 秒，
多出来的时间全在高优先级任务的"延长"上（延长 150 秒值 2.5 分）。同样 150 秒如果换成一次新的高优先级
重复观测（120 秒 + 30 秒切换）值 3 分——每换成一次涨 0.5 分（综合约 +0.000008）。难点是新观测只能放在
它自己的可见窗口里，那里通常被别的任务占着。

做法（和二分图匹配里找增广路是一个思路）：
1. 每个"站 × 夜晚"是一条时间线，任务顺序固定；也可以把一夜按任务数切成几段（piece），段与段之间用一堵
   固定的"墙"隔开，这样一条链可以在同一站同一夜的不同段里各动一次。把全部任务压到 120 秒，前向算每个任务
   最早结束、后向算最晚开始，于是"在第 k 个空当里再塞一个窗口为 [a, b] 的任务行不行""拿掉第 j 个任务后
   行不行"都是 O(1) 级的判断。
2. 对一颗还没观测满 3 次的高优先级目标 R 做广度优先搜索：R 的某个窗口能直接塞进去 → 成功；
   否则 R 顶掉那里的任务 X，X 去它自己的其他窗口（可以换站、换夜晚）找位置，还不行就再顶掉 Y……
   被顶掉的如果是优先级更低的重复观测，可以直接删掉。一条链上每条时间线（段）只动一次。
3. 找到链后，对链上涉及的时间线解一次线性规划（顺序固定，重新分配起止时间；同目标 48 小时间隔：
   两次观测都在这次规划里的写成约束，另一头不在的按它当前的起止时间卡死），总得分变高才接受；
   不划算就禁用丢分最多的那一步再搜。

所有时间都是整秒；窗口用 obsplan/second_bounds.py 探查的真实可见区间（与按秒精修同一判定）。
"""
from __future__ import annotations

import bisect
import os
import time
from collections import defaultdict

import numpy as np
from numba import njit

from .problem import Problem
from .schedule import Schedule

BIG = 10 ** 12


@njit(cache=True)
def _unit_ab(seq, n, tk_w, tk_s, tk_e, tk_tg, tg_tasks, wL, wR, gap, dmin, sw, wall_lo, wall_hi, A, B, lo, hi):
    """一条时间线：全部任务压到最短时长时，空当 k（第 k−1 与第 k 个任务之间）可用的时间范围 [A[k], B[k]]。
    每个任务的可用范围 = 它的窗口 ∩ 这段时间线的墙 ∩ 同目标其他观测（按当前起止）留出的 48 小时间隔。
    返回这条线本身是否可行。"""
    ok = True
    prev = wall_lo - sw
    A[0] = wall_lo
    for j in range(n):
        p = seq[j]
        if tk_w[p] >= 0:
            l, h = wL[tk_w[p]], wR[tk_w[p]]
        else:
            l, h = tk_s[p], tk_e[p]
        if wall_lo > l:
            l = wall_lo
        if wall_hi - sw < h:
            h = wall_hi - sw
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
    nxt = wall_hi
    B[n] = wall_hi - sw
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
def _fit_without(u, j, a, b, A2, B2, lo2, hi2, ulen, dmin, sw, D):
    """拿掉时间线 u 的第 j 个任务后，窗口 [a, b] 的新任务能插在原第 k 个任务之前（k 在 j 前后 D 个以内）。
    返回 k；不行返回 −1。k = j 就是原地顶替。"""
    n = ulen[u]
    if min(b, B2[u, j + 1]) - max(a, A2[u, j]) >= dmin:
        return j
    nxt = B2[u, j + 1]                  # 第 j−1 个任务的最晚结束（后面直接接第 j+1 个）
    k = j - 1
    while k >= 0 and k >= j - D:        # 插到前面：原第 k..j−1 个任务排在新任务之后
        ls = min(hi2[u, k], nxt) - dmin
        if ls < lo2[u, k]:
            break
        nxt = ls - sw
        if min(b, nxt) - max(a, A2[u, k]) >= dmin:
            return k
        if nxt < a + dmin:
            break
        k -= 1
    prev = A2[u, j]                     # 第 j+1 个任务的最早开始
    m = j + 1
    while m < n and m <= j + D:         # 插到后面：原第 j+1..m 个任务排在新任务之前
        ef = max(lo2[u, m], prev) + dmin
        if ef > hi2[u, m]:
            break
        prev = ef + sw
        if min(b, B2[u, m + 1]) - max(a, prev) >= dmin:
            return m + 1
        if prev > b - dmin:
            break
        m += 1
    return -1


@njit(cache=True)
def _slots(x, skip, cl, cr, tk_s, tk_e, tg_tasks, gap, dmin, out_a, out_b):
    """目标 x（不算任务 skip）再加一次观测、可用区间 [cl, cr]：与已有观测保持 48 小时后剩下的区间。返回区间个数。"""
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
        a = cl
        b = cr
        if g > 0 and oe_[g - 1] + gap > a:
            a = oe_[g - 1] + gap
        if g < c and os_[g] - gap < b:
            b = os_[g] - gap
        if b - a >= dmin:
            out_a[n], out_b[n] = a, b
            n += 1
    return n


@njit(cache=True)
def _three_ok(lo, hi, dur, gap):
    """三次观测（各自的可用区间 [lo, hi] 与时长）能不能排成两两间隔 gap。"""
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    for pm in perms:
        t = -BIG
        ok = True
        for i in pm:
            s = max(lo[i], t)
            e = s + dur[i]
            if e > hi[i]:
                ok = False
                break
            t = e + gap
        if ok:
            return True
    return False


@njit(cache=True)
def _pattern_mask(x, ks, ke, w_skip, c_ptr, c_w, c_l, c_r, gap, dmin, mask):
    """目标 x 保留一次观测 K（固定在 [ks, ke]）、把另一次挪走：挪到哪些候选之后还存在放第三次的窗口。
    结果写进 mask（只动 x 的候选那一段），返回可行候选个数。w_skip：被挪走的那次原来所在的窗口。"""
    lo = np.empty(3, np.int64)
    hi = np.empty(3, np.int64)
    dur = np.empty(3, np.int64)
    lo[0], hi[0], dur[0] = ks, ke, ke - ks
    dur[1] = dmin
    dur[2] = dmin
    n = 0
    for ci in range(c_ptr[x], c_ptr[x + 1]):
        mask[ci] = 0
        if c_w[ci] == w_skip:
            continue
        lo[1], hi[1] = c_l[ci], c_r[ci]
        for cj in range(c_ptr[x], c_ptr[x + 1]):
            if c_w[cj] == c_w[ci]:
                continue
            lo[2], hi[2] = c_l[cj], c_r[cj]
            if _three_ok(lo, hi, dur, gap):
                mask[ci] = 1
                n += 1
                break
    return n


@njit(cache=True, nogil=True)
def _search(new_tid, max_depth, max_q, D, c_ptr, c_unit, c_l, c_r, tk_tg, tk_s, tk_e, tk_w, tg_tasks, tg_cnt,
            seq2, ulen, A2, B2, lo2, hi2, unit_ok, gap, dmin, sw, visited, stamp, ban_unit, ban_task, attempt,
            q_task, q_par, q_c, q_k, q_a, q_unit, q_depth, P, drop_below, root_ok):
    """广度优先找弹射链。root_ok[ci]：链头任务（新观测）允许用哪些候选。候选 ci =（目标的某个窗口, 某条时间线, 可用区间 [c_l, c_r]）。
    队列第 e 项 = "任务 q_task[e] 需要找位置"；它是被父项的任务顶出来的：
    父项任务用候选 q_c[e]、插在时间线 q_unit[e] 原第 q_k[e] 个任务之前（参考开始时间 q_a[e]）。
    返回 (末项 e, 候选, 位置 k, 参考开始时间, 方式, 被删任务)：
      方式 0 = 末项任务直接塞进空当 k；方式 1 = 末项任务顶掉一个低价值的重复观测（直接删掉，不再安置）。
    找不到返回 e = −1。"""
    q_task[0], q_par[0], q_unit[0], q_depth[0] = new_tid, -1, -1, 0
    head, tail = 0, 1
    sa = np.empty(4, np.int64)
    sb = np.empty(4, np.int64)
    while head < tail:
        e = head
        head += 1
        T = q_task[e]
        x = tk_tg[T]
        for phase in range(2):                  # 0：直接塞进去；1：顶掉一个任务
            if phase == 1 and q_depth[e] >= max_depth and drop_below <= 0:
                break
            for ci in range(c_ptr[x], c_ptr[x + 1]):
                u = c_unit[ci]
                if not unit_ok[u] or (e == 0 and not root_ok[ci]):
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
                ns = _slots(x, T, c_l[ci], c_r[ci], tk_s, tk_e, tg_tasks, gap, dmin, sa, sb)
                n = ulen[u]
                for g in range(ns):
                    a, b = sa[g], sb[g]
                    if phase == 0:
                        if ban_unit[u] != attempt:
                            k = _direct_fit(u, a, b, A2, B2, ulen, dmin)
                            if k >= 0:
                                return e, ci, k, a, 0, -1
                        continue
                    lo, hi = 0, n
                    while lo < hi:              # 第一个 B[k+1] >= a + dmin 的 k
                        mid = (lo + hi) // 2
                        if B2[u, mid + 1] >= a + dmin:
                            hi = mid
                        else:
                            lo = mid + 1
                    j = max(lo - D, 0)
                    while j < n and (j <= lo + D or A2[u, max(j - D, 0)] <= b - dmin):
                        Y = seq2[u, j]
                        if visited[Y] != stamp and tk_w[Y] >= 0 and ban_task[Y] != attempt:
                            y = tk_tg[Y]
                            can_drop = P[y] < drop_below and tg_cnt[y] >= 2
                            if can_drop or q_depth[e] < max_depth:
                                k = _fit_without(u, j, a, b, A2, B2, lo2, hi2, ulen, dmin, sw, D)
                                if k >= 0:
                                    if can_drop:
                                        return e, ci, k, a, 1, Y
                                    visited[Y] = stamp
                                    if tail < max_q:
                                        q_task[tail], q_par[tail], q_c[tail], q_k[tail] = Y, e, ci, k
                                        q_a[tail], q_unit[tail], q_depth[tail] = a, u, q_depth[e] + 1
                                        tail += 1
                        j += 1
    return -1, -1, -1, -1, 0, -1


class _Scratch:
    """一次搜链要用的临时数组（访问标记、队列、禁用标记）。多线程时每个线程一份，互不干扰。"""

    def __init__(self, cap: int, n_units: int, mq: int):
        self.visited = np.zeros(cap, np.int64)
        self.stamp = 0
        self.q = [np.zeros(mq, np.int64) for _ in range(7)]
        self.ban_unit = np.zeros(n_units, np.int64)
        self.ban_task = np.zeros(cap, np.int64)
        self.attempt = 0


class Timeline:
    """按秒计划的可变表示：每个"站 × 夜晚"（或其中一段）一条有序任务线。

    piece > 0：每夜按任务数切段，每段约 piece 个任务（第一段 offset 个）；段之间的墙 = 后一段第一个任务当前的开始时间。
    extra_windows：网格上不够 120 秒、按秒才够的短窗口 {"station", "target", "k0", "L", "R"}，作为额外候选。"""

    def __init__(self, pb: Problem, wL: np.ndarray, wR: np.ndarray, sch: Schedule, extra: int = 4000,
                 piece: int = 0, offset: int = 0, extra_windows: dict | None = None):
        cfg = pb.cfg
        self.pb = pb
        self.dmin, self.dmax, self.sw = int(cfg.obs_min_s), int(cfg.obs_max_s), int(cfg.switch_s)
        self.gap = int(round(cfg.repeat_gap_h * 3600))
        sw = self.sw
        # 窗口表（可含额外的短窗口）
        w_st, w_tg, w_k0 = pb.w_st, pb.w_tg, pb.w_k0
        wL, wR = np.asarray(wL, np.int64), np.asarray(wR, np.int64)
        self.n_grid_windows = len(w_st)
        if extra_windows is not None and len(extra_windows["station"]):
            w_st = np.concatenate([w_st, extra_windows["station"]])
            w_tg = np.concatenate([w_tg, extra_windows["target"]])
            w_k0 = np.concatenate([w_k0, extra_windows["k0"]])
            wL = np.concatenate([wL, extra_windows["L"]])
            wR = np.concatenate([wR, extra_windows["R"]])
        self.w_st, self.w_tg = w_st.astype(np.int64), w_tg.astype(np.int64)
        self.wL, self.wR = wL.astype(np.int64), wR.astype(np.int64)
        W = len(self.w_st)
        # 站 × 夜晚
        night_off = np.zeros(pb.n_stations + 1, np.int64)
        starts = []
        for s, runs in enumerate(pb.win.dark):
            night_off[s + 1] = night_off[s] + len(runs)
            starts.append(np.array([a for a, _ in runs], np.int64))
        n_nights = int(night_off[-1])
        w_night = np.zeros(W, np.int64)
        for s in range(pb.n_stations):
            m = self.w_st == s
            r = np.searchsorted(starts[s], w_k0[m], side="right") - 1
            w_night[m] = night_off[s] + np.maximum(r, 0)
        # 任务
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
        # 目标 → 窗口
        order = np.argsort(self.w_tg, kind="stable")
        t_ptr = np.zeros(pb.n_targets + 1, np.int64)
        np.cumsum(np.bincount(self.w_tg, minlength=pb.n_targets), out=t_ptr[1:])
        self.unmatched = 0
        tk_night = np.zeros(n, np.int64)
        for i in range(n):
            t, s = int(self.tk_tg[i]), int(self.tk_st[i])
            ws = order[t_ptr[t]:t_ptr[t + 1]]
            hit = ws[(self.w_st[ws] == s) & (self.wL[ws] <= self.tk_s[i]) & (self.tk_e[i] <= self.wR[ws])]
            if len(hit):
                self.tk_w[i] = hit[0]
                tk_night[i] = w_night[hit[0]]
            else:                                   # 不在任何已知窗口里：原样固定
                self.unmatched += 1
                r = np.searchsorted(starts[s], self.tk_s[i] // cfg.cell_s, side="right") - 1
                tk_night[i] = night_off[s] + max(int(r), 0)
        self.tg_tasks = np.full((pb.n_targets, 3), -1, np.int64)
        self.tg_cnt = np.zeros(pb.n_targets, np.int64)
        for i in range(n):
            t = self.tk_tg[i]
            self.tg_tasks[t, self.tg_cnt[t]] = i
            self.tg_cnt[t] += 1
        # 时间线：每夜一条，或按任务数切段
        night_seq = [[] for _ in range(n_nights)]
        for i in np.argsort(self.tk_s[:n], kind="stable"):
            night_seq[tk_night[i]].append(int(i))
        self.seq, wall_lo, wall_hi = [], [], []
        night_units = []                             # 每夜：(各段的墙起点列表, 第一段的时间线编号)
        for seq in night_seq:
            cuts = []
            if piece > 0:
                c = offset if 0 < offset < piece else piece
                while c < len(seq) - piece // 3:
                    cuts.append(c)
                    c += piece
            bounds = [0] + cuts + [len(seq)]
            walls = [-BIG] + [int(self.tk_s[seq[c]]) for c in cuts] + [BIG]
            night_units.append((walls, len(self.seq)))
            for j in range(len(bounds) - 1):
                self.seq.append(seq[bounds[j]:bounds[j + 1]])
                wall_lo.append(walls[j])
                wall_hi.append(walls[j + 1])
        self.n_units = len(self.seq)
        self.wall_lo, self.wall_hi = np.array(wall_lo, np.int64), np.array(wall_hi, np.int64)
        for u, seq in enumerate(self.seq):
            self.tk_unit[seq] = u
        # 候选：（窗口, 时间线, 可用区间）
        c_w, c_unit, c_l, c_r = [], [], [], []
        c_ptr = np.zeros(pb.n_targets + 1, np.int64)
        wLl, wRl, wnl = self.wL.tolist(), self.wR.tolist(), w_night.tolist()
        for t in range(pb.n_targets):
            for w in order[t_ptr[t]:t_ptr[t + 1]].tolist():
                walls, u0 = night_units[wnl[w]]
                L, R = wLl[w], wRl[w]
                j = max(bisect.bisect_right(walls, L) - 1, 0)
                while j < len(walls) - 1 and walls[j] < R:
                    a, b = max(L, walls[j]), min(R, walls[j + 1] - sw)
                    if b - a >= self.dmin:
                        c_w.append(w); c_unit.append(u0 + j); c_l.append(a); c_r.append(b)
                    j += 1
            c_ptr[t + 1] = len(c_w)
        self.c_ptr = c_ptr
        self.c_w, self.c_unit = np.array(c_w, np.int64), np.array(c_unit, np.int64)
        self.c_l, self.c_r = np.array(c_l, np.int64), np.array(c_r, np.int64)
        self.maxlen = max(len(s) for s in self.seq) + 150
        self.seq2 = np.full((self.n_units, self.maxlen), -1, np.int64)
        self.ulen = np.zeros(self.n_units, np.int64)
        self.A2 = np.zeros((self.n_units, self.maxlen + 1), np.int64)
        self.B2 = np.zeros((self.n_units, self.maxlen + 1), np.int64)
        self.lo2 = np.zeros((self.n_units, self.maxlen), np.int64)
        self.hi2 = np.zeros((self.n_units, self.maxlen), np.int64)
        self.all_ok = np.ones(len(self.c_w), np.uint8)
        self.root_mask = np.zeros(len(self.c_w), np.uint8)
        self.unit_ok = np.ones(self.n_units, bool)
        for u in range(self.n_units):
            self.refresh(u)
        mq = 60000
        self.cap = cap
        self.max_q = mq
        self.scr = _Scratch(cap, self.n_units, mq)
        self.fail_count = np.zeros(pb.n_targets, np.int64)             # 每颗目标试过几次没成功（AI 预测器的特征）
        self.stats = defaultdict(int)
        self.lp_k = int(os.environ.get("OBSPLAN_CHAIN_LPK", "40"))   # 线性规划只重排改动前后各几个任务（0 = 整条时间线）

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
                      self.wL, self.wR, self.gap, self.dmin, self.sw, self.wall_lo[u], self.wall_hi[u],
                      self.A2[u], self.B2[u], self.lo2[u], self.hi2[u])
        self.unit_ok[u] = bool(ok)

    def score_points(self) -> float:
        a = np.nonzero(self.alive[:self.n])[0]
        d = self.tk_e[a] - self.tk_s[a]
        return float((self.P[self.tk_tg[a]] * (0.6 + 0.4 * (d - self.dmin) / (self.dmax - self.dmin))).sum())

    def schedule(self, algo: str = "按秒跨站弹射链"):
        """返回计划、每个任务的窗口真实区间 (L, R)（给全局按秒精修用）、用到的额外短窗口（结构检查用）。"""
        a = np.nonzero(self.alive[:self.n])[0]
        sch = Schedule(self.tk_st[a].copy(), self.tk_tg[a].copy(), self.tk_s[a].copy(),
                       (self.tk_e[a] - self.tk_s[a]).copy(), algo=algo)
        w = self.tk_w[a]
        L = np.where(w >= 0, self.wL[np.maximum(w, 0)], self.tk_s[a])
        R = np.where(w >= 0, self.wR[np.maximum(w, 0)], self.tk_e[a])
        ex = np.unique(w[w >= self.n_grid_windows])
        used = {"station": self.w_st[ex], "target": self.w_tg[ex], "L": self.wL[ex], "R": self.wR[ex]}
        return sch, L, R, used

    # ---------- 线性规划：给定各时间线的新顺序，重新定起止 ----------
    def lp(self, new_seqs: dict, override: dict, dropped=()):
        """new_seqs：{时间线: 任务编号的新顺序}；override：{任务: (新窗口, 参考开始时间)}（新任务或换了位置的任务）。
        返回 (这些时间线的新得分, {任务: (开始, 结束)})；不可行返回 None。"""
        from ortools.linear_solver import pywraplp
        dmin, dmax, sw, gap = self.dmin, self.dmax, self.sw, self.gap
        kx = 0.4 / (dmax - dmin)
        # 只重排改动附近的任务：每条时间线取"新旧顺序不同的那一段"前后各 lp_k 个任务，其余按当前起止固定，
        # 固定任务成为这一段的边界。lp_k = 0 时整条时间线都进线性规划（旧行为）。
        local, fixed_pts = {}, 0.0
        for u, seq in new_seqs.items():
            n_new = len(seq)
            if self.lp_k <= 0:
                local[u] = (0, n_new, int(self.wall_lo[u]), int(self.wall_hi[u]) - sw)
                continue
            cur = self.seq[u]
            m = min(n_new, len(cur))
            i0 = 0
            while i0 < m and seq[i0] == cur[i0] and seq[i0] not in override:
                i0 += 1
            j0 = 0
            while j0 < m - i0 and seq[n_new - 1 - j0] == cur[len(cur) - 1 - j0] and seq[n_new - 1 - j0] not in override:
                j0 += 1
            a, b = max(0, i0 - self.lp_k), min(n_new, n_new - j0 + self.lp_k)
            if b <= a:
                a, b = max(0, i0 - self.lp_k), min(n_new, i0 + self.lp_k)
            wlo, whi = int(self.wall_lo[u]), int(self.wall_hi[u]) - sw
            if a > 0:
                wlo = max(wlo, int(self.tk_e[seq[a - 1]]) + sw)
            if b < n_new:
                whi = min(whi, int(self.tk_s[seq[b]]) - sw)
            local[u] = (a, b, wlo, whi)
            for p in seq[:a] + seq[b:]:
                fixed_pts += self.P[self.tk_tg[p]] * (0.6 + kx * (self.tk_e[p] - self.tk_s[p] - dmin))
        tasks = [p for u, seq in new_seqs.items() for p in seq[local[u][0]:local[u][1]]]
        inlp = set(tasks)
        lp = pywraplp.Solver.CreateSolver("GLOP")
        sv, dv = {}, {}
        ref, lim = {}, {}
        by_t = defaultdict(list)
        obj = lp.Objective()
        inf = lp.infinity()
        for u, seq in new_seqs.items():
            a_, b_, wlo, whi = local[u]
            for p in seq[a_:b_]:
                if p in override:
                    w, r = override[p]
                else:
                    w, r = int(self.tk_w[p]), int(self.tk_s[p])
                ref[p] = r
                if w >= 0:
                    l, h = int(self.wL[w]), int(self.wR[w])
                else:
                    l, h = int(self.tk_s[p]), int(self.tk_e[p])
                l, h = max(l, wlo), min(h, whi)
                x = int(self.tk_tg[p])
                for q in self.tg_tasks[x]:
                    q = int(q)
                    if q < 0 or q == p or q in inlp or q in dropped:
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
                c = lp.RowConstraint(-inf, float(h))
                c.SetCoefficient(sv[p], 1.0)
                c.SetCoefficient(dv[p], 1.0)
                obj.SetCoefficient(dv[p], float(self.P[x]))
                by_t[x].append(p)
        pairs = []
        for u, seq in new_seqs.items():
            sub = seq[local[u][0]:local[u][1]]
            for p, q in zip(sub[:-1], sub[1:]):
                pairs.append((p, q, sw))
        for x, ps in by_t.items():
            if len(ps) > 1:
                ps = sorted(ps, key=lambda p: ref[p])
                for p, q in zip(ps[:-1], ps[1:]):
                    pairs.append((p, q, gap))
        for p, q, g in pairs:
            c = lp.RowConstraint(float(g), inf)
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
        new = fixed_pts + sum(self.P[self.tk_tg[p]] * (0.6 + kx * (out[p][1] - out[p][0] - dmin)) for p in tasks)
        for u, seq in new_seqs.items():                 # 固定的任务按当前起止一并返回
            a_, b_ = local[u][0], local[u][1]
            for p in seq[:a_] + seq[b_:]:
                out[p] = (int(self.tk_s[p]), int(self.tk_e[p]))
        return new, out

    def units_score(self, units) -> float:
        k = 0.4 / (self.dmax - self.dmin)
        return sum(self.P[self.tk_tg[p]] * (0.6 + k * (self.tk_e[p] - self.tk_s[p] - self.dmin))
                   for u in units for p in self.seq[u])

    def commit(self, new_seqs: dict, override: dict, times: dict, dropped=()) -> None:
        touched = set(new_seqs)
        for u, seq in new_seqs.items():
            self.seq[u] = list(seq)
            for p in seq:
                self.tk_unit[p] = u
        for y in dropped:
            x = self.tk_tg[y]
            keep = [int(q) for q in self.tg_tasks[x] if q >= 0 and q != y]
            self.tg_tasks[x] = -1
            self.tg_tasks[x, :len(keep)] = keep
            self.tg_cnt[x] = len(keep)
            self.alive[y] = False
            touched.update(int(self.tk_unit[q]) for q in keep)
        for p, (w, _) in override.items():
            self.tk_w[p] = w
            self.tk_st[p] = self.w_st[w]
        for p, (s, e) in times.items():
            if self.tk_s[p] != s or self.tk_e[p] != e:
                self.tk_s[p], self.tk_e[p] = s, e
                for q in self.tg_tasks[self.tk_tg[p]]:
                    if q >= 0 and q != p:
                        touched.add(int(self.tk_unit[q]))
        for u in touched:
            self.refresh(u)

    # ---------- 弹射链 ----------
    def try_add(self, x: int, max_depth: int = 2, min_gain: float = 1e-6, shift: int = 4, retries: int = 6,
                drop: bool = True, root_mask=None) -> float:
        """给目标 x 多加一次观测。成功返回得分增量（分），否则 0。root_mask：只允许新观测用这些候选。
        shift：顶出一个任务后，新任务可以插在它前后几个位置以内；retries：一条链不划算时禁用出问题的那一步再搜几次；
        drop：允许直接删掉优先级更低的重复观测来腾位置。"""
        if self.tg_cnt[x] >= 3:
            return 0.0
        prop = self._propose(x, self.n, self.scr, max_depth, min_gain, shift, retries, drop, root_mask, self.stats)
        if prop is None:
            self.fail_count[x] += 1
            return 0.0
        return self._apply(prop)

    def _propose(self, x: int, tid: int, scr: _Scratch, max_depth: int, min_gain: float, shift: int, retries: int,
                 drop: bool, root_mask, stats) -> dict | None:
        """在当前状态上给目标 x 找一条划算的链（含"不划算就禁用一步再搜"），只读状态、不改计划。
        tid：新观测临时用的任务编号。返回提交所需的全部内容，找不到返回 None。多个线程可以同时调用（各用各的 scr）。"""
        self.tk_tg[tid] = x
        self.tk_w[tid] = -1
        scr.attempt += 1
        qt, qp, qc, qk, qa, qu, qd = scr.q
        kx = 0.4 / (self.dmax - self.dmin)
        for _ in range(retries):
            scr.stamp += 1
            e, ci, k, a, mode, Y = _search(
                tid, max_depth, self.max_q, shift, self.c_ptr, self.c_unit, self.c_l, self.c_r,
                self.tk_tg, self.tk_s, self.tk_e, self.tk_w, self.tg_tasks, self.tg_cnt, self.seq2, self.ulen,
                self.A2, self.B2, self.lo2, self.hi2, self.unit_ok, self.gap, self.dmin, self.sw,
                scr.visited, scr.stamp, scr.ban_unit, scr.ban_task, scr.attempt,
                qt, qp, qc, qk, qa, qu, qd, self.P, float(self.P[x]) if drop else 0.0,
                self.all_ok if root_mask is None else root_mask)
            if e < 0:
                stats["找不到链"] += 1
                return None
            new_seqs, override, steps = {}, {}, []          # steps：(时间线, 被顶出/删掉的任务或 −1)
            u = int(self.c_unit[ci])
            seq = list(self.seq[u])
            dropped = ()
            if mode == 0:
                seq.insert(int(k), int(qt[e]))
                steps.append((u, -1))
            else:
                j = seq.index(int(Y))
                seq.insert(int(k), int(qt[e]))
                del seq[j + 1 if k <= j else j]
                dropped = (int(Y),)
                steps.append((u, int(Y)))
            new_seqs[u] = seq
            override[int(qt[e])] = (int(self.c_w[ci]), int(a))
            depth = 0
            while e > 0:
                par = int(qp[e])
                u = int(qu[e])
                seq = list(self.seq[u])
                j = seq.index(int(qt[e]))
                kk = int(qk[e])
                seq.insert(kk, int(qt[par]))
                del seq[j + 1 if kk <= j else j]
                new_seqs[u] = seq
                override[int(qt[par])] = (int(self.c_w[qc[e]]), int(qa[e]))
                steps.append((u, int(qt[e])))
                e = par
                depth += 1
            old = self.units_score(new_seqs.keys())
            res = self.lp(new_seqs, override, dropped)
            if res is None:
                stats["规划不可行"] += 1
                bad = 0
            else:
                new, times = res
                gain = new - old
                if gain >= min_gain:
                    return {"x": int(x), "tid": int(tid), "new_seqs": new_seqs, "override": override, "times": times,
                            "dropped": dropped, "gain": float(gain), "key": f"成功_深度{depth}" + ("_删" if mode else "")}
                stats["不划算"] += 1
                # 哪一步丢的延长最多，就禁用哪一步再搜
                loss = []
                for u_, _y in steps:
                    o = sum(self.P[self.tk_tg[p]] * kx * (self.tk_e[p] - self.tk_s[p] - self.dmin) for p in self.seq[u_])
                    nw = sum(self.P[self.tk_tg[p]] * kx * (times[p][1] - times[p][0] - self.dmin) for p in new_seqs[u_])
                    loss.append(o - nw)
                bad = int(np.argmax(loss))
            u_, y_ = steps[bad]
            if y_ < 0:
                scr.ban_unit[u_] = scr.attempt
            else:
                scr.ban_task[y_] = scr.attempt
        return None

    def _apply(self, prop: dict) -> float:
        """提交 _propose 给出的方案：新观测换成正式的任务编号，改顺序和起止。"""
        x, tmp, tid = prop["x"], prop["tid"], self.n
        new_seqs, override, times = prop["new_seqs"], prop["override"], prop["times"]
        if tmp != tid:
            new_seqs = {u: [tid if p == tmp else p for p in seq] for u, seq in new_seqs.items()}
            override = {(tid if p == tmp else p): v for p, v in override.items()}
            times = {(tid if p == tmp else p): v for p, v in times.items()}
        moved = {p for p, (a_, b_) in times.items() if p != tid and (self.tk_s[p] != a_ or self.tk_e[p] != b_)}
        self.last_change = (set(new_seqs.keys()), moved | set(prop["dropped"]) | {tid})   # 多线程提交时查冲突用
        self.tk_tg[tid] = x
        self.tk_w[tid] = -1
        self.n += 1
        self.alive[tid] = True
        self.tg_tasks[x, self.tg_cnt[x]] = tid
        self.tg_cnt[x] += 1
        self.commit(new_seqs, override, times, prop["dropped"])
        self.stats[prop["key"]] += 1
        return prop["gain"]

    def _still_valid(self, prop: dict, changed_units: set, changed_tasks: set) -> bool:
        """同一批里前面已经提交了别的方案之后，这个方案还能不能原样提交：
        它要改的时间线都没被动过、要删的重复观测还在且目标还有别的观测、和被挪动过的同目标观测仍隔 48 小时。"""
        if any(u in changed_units for u in prop["new_seqs"]):
            return False
        if self.tg_cnt[prop["x"]] >= 3:
            return False
        for y in prop["dropped"]:
            if not self.alive[y] or self.tg_cnt[self.tk_tg[y]] < 2:
                return False
        if changed_tasks:
            times, tmp, drop = prop["times"], prop["tid"], set(prop["dropped"])
            for p, (s_, e_) in times.items():
                tx = prop["x"] if p == tmp else int(self.tk_tg[p])
                for q in self.tg_tasks[tx]:
                    q = int(q)
                    if q < 0 or q == p or q in times or q in drop or q not in changed_tasks:
                        continue
                    if not (self.tk_s[q] >= e_ + self.gap or s_ >= self.tk_e[q] + self.gap):
                        return False
        return True

    def try_drop(self, y: int, min_gain: float = 1e-6, force: bool = False):
        """删掉一个重复观测，把腾出的时间交给同一条时间线上的其他任务延长；总分变高才接受。
        force：不管划不划算都删（复合动作的第一步用），返回得分变化；规划不可行返回 None。"""
        x = int(self.tk_tg[y])
        if not self.alive[y] or self.tg_cnt[x] < 2 or self.tk_w[y] < 0:
            return None if force else 0.0
        u = int(self.tk_unit[y])
        seq = [p for p in self.seq[u] if p != y]
        old = self.units_score([u])
        res = self.lp({u: seq}, {}, (y,))
        if res is None:
            return None if force else 0.0
        new, times = res
        if not force and new - old < min_gain:
            return 0.0
        self.commit({u: seq}, {}, times, (y,))
        if not force:
            self.stats["删重复换延长"] += 1
        return float(new - old)

    # ---------- 快照（复合动作失败时整体回退） ----------
    def snapshot(self):
        return (self.n, [a.copy() for a in (self.tk_st, self.tk_tg, self.tk_s, self.tk_e, self.tk_w, self.tk_unit,
                                            self.alive, self.tg_tasks, self.tg_cnt, self.seq2, self.ulen, self.A2,
                                            self.B2, self.lo2, self.hi2, self.unit_ok)],
                [list(q) for q in self.seq], dict(self.stats))

    def restore(self, snap) -> None:
        self.n, arrs, seq, stats = snap
        for dst, src in zip((self.tk_st, self.tk_tg, self.tk_s, self.tk_e, self.tk_w, self.tk_unit, self.alive,
                             self.tg_tasks, self.tg_cnt, self.seq2, self.ulen, self.A2, self.B2, self.lo2, self.hi2,
                             self.unit_ok), arrs):
            dst[...] = src
        self.seq = seq
        self.stats = defaultdict(int, stats)

    def repattern(self, x: int, max_depth: int = 3, needy_in_unit: dict | None = None, fill: int = 12,
                  min_total: float = 1e-6, **kw) -> float:
        """重排节奏：目标 x 已有两次观测、按现在的时间放不下第三次时，把其中一次挪到别的窗口再补第三次。
        三步（删一次 → 在"挪过去后还放得下第三次"的窗口里重新放 → 补第三次）作为一个整体，总分变高才保留；
        挪走后腾出的位置如果只换来低价值的延长，再试着让附近其他缺观测的高优先级目标用上（fill 个以内）。"""
        if self.tg_cnt[x] != 2:
            return 0.0
        obs = sorted((int(q) for q in self.tg_tasks[x] if q >= 0), key=lambda q: self.tk_s[q])
        for O, K in ((obs[1], obs[0]), (obs[0], obs[1])):
            if self.tk_w[O] < 0:
                continue
            nm = _pattern_mask(x, int(self.tk_s[K]), int(self.tk_e[K]), int(self.tk_w[O]), self.c_ptr, self.c_w,
                               self.c_l, self.c_r, self.gap, self.dmin, self.root_mask)
            if nm == 0:
                continue
            snap = self.snapshot()
            u_old = int(self.tk_unit[O])
            total = self.try_drop(O, force=True)
            ok = total is not None
            if ok:
                g = self.try_add(x, max_depth=max_depth, root_mask=self.root_mask, **kw)
                ok = g > 0
                total = total + g if ok else 0.0
            if ok:
                g = self.try_add(x, max_depth=max_depth, **kw)
                ok = g > 0
                total += g
            if ok and total < min_total and needy_in_unit is not None:
                tried = 0
                for x2 in needy_in_unit.get(u_old, ()):
                    if x2 == x or self.tg_cnt[x2] >= 3:
                        continue
                    total += self.try_add(int(x2), max_depth=min(max_depth, 2), **kw)
                    tried += 1
                    if total >= min_total or tried >= fill:
                        break
            self.root_mask[self.c_ptr[x]:self.c_ptr[x + 1]] = 0
            if ok and total >= min_total:
                self.stats["重排节奏成功"] += 1
                return float(total)
            self.restore(snap)
            self.stats["重排节奏回退"] += 1
        return 0.0


def run_pass(tl: Timeline, targets, max_depth: int = 2, verbose: bool = True, log_every: int = 200,
             deadline: float | None = None, **kw) -> dict:
    t0 = time.time()
    total, ok = 0.0, 0
    for n, x in enumerate(targets):
        if deadline is not None and time.time() > deadline:
            break
        g = tl.try_add(int(x), max_depth=max_depth, **kw)
        if g > 0:
            total += g
            ok += 1
        if verbose and (n + 1) % log_every == 0:
            print(f"  [弹射链] {n + 1}/{len(targets)}  成功 {ok}  累计 +{total:.1f} 分  {dict(tl.stats)}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    return {"tried": len(targets), "added": ok, "gain_points": total, "seconds": time.time() - t0}


def default_threads() -> int:
    """弹射链的线程数：环境变量 OBSPLAN_CHAIN_THREADS，否则取 CPU 核数（最多 8）。1 = 单线程（旧行为）。"""
    v = os.environ.get("OBSPLAN_CHAIN_THREADS")
    if v:
        return max(1, int(v))
    return max(1, min(8, os.cpu_count() or 1))


def run_pass_parallel(tl: Timeline, targets, threads: int = 8, max_depth: int = 2, verbose: bool = True,
                      log_every: int = 200, deadline: float | None = None, min_gain: float = 1e-6, shift: int = 4,
                      retries: int = 6, drop: bool = True, batch: int | None = None, redo: bool | None = None) -> dict:
    """多线程版 run_pass：每批 threads 颗目标，各线程在同一个状态上同时找链、解线性规划（只读，不改计划），
    然后主线程按原来的顺序逐个提交；提交前检查是否与本批前面已提交的方案冲突（动过同一条时间线、要删的观测已不在、
    同目标观测不再隔 48 小时），冲突的就在最新状态上单独重找。搜链（numba，不占全局解释器锁）占大头，所以能并行。
    批次和提交顺序固定，结果与线程调度、核数都无关，可复现（与单线程版不完全相同：同一批里后面的目标看到的是批开始时的状态）。
    实测（M1 Pro 8 线程，快速流水线的计划）：一整遍 30 秒对单线程 33 秒、综合 0.9849 对 0.9847；同样给 30 秒 / 1 分钟，
    综合比单线程高 0.0003～0.0004。提速不大，因为同一站同一夜的时间线大家都在抢，一批里约三成方案冲突、要单线程重找。"""
    import queue
    from concurrent.futures import ThreadPoolExecutor
    batch = int(os.environ.get("OBSPLAN_CHAIN_BATCH", 0)) or (batch or 8)     # 批次固定 8：结果与核数无关
    redo_on = (os.environ.get("OBSPLAN_CHAIN_REDO", "1") != "0") if redo is None else redo
    t0 = time.time()
    total, ok, redo = 0.0, 0, 0
    t_prop = t_redo = 0.0
    targets = [int(x) for x in targets]
    pool = queue.Queue()                                   # 每个正在算的方案借一份临时数组，算完归还
    for _ in range(threads):
        pool.put(_Scratch(tl.cap, tl.n_units, tl.max_q))

    def job(x, tid, st):
        scr = pool.get()
        try:
            return tl._propose(x, tid, scr, max_depth, min_gain, shift, retries, drop, None, st)
        finally:
            pool.put(scr)
    with ThreadPoolExecutor(threads) as ex:
        i, n_done = 0, 0
        while i < len(targets):
            if deadline is not None and time.time() > deadline:
                break
            cur_b = [x for x in targets[i:i + batch] if tl.tg_cnt[x] < 3]
            i += batch
            if tl.n + 2 * batch >= tl.cap:                   # 临时编号放在数组尾部，快满时退回单线程
                for x in cur_b:
                    g = tl.try_add(x, max_depth=max_depth, min_gain=min_gain, shift=shift, retries=retries, drop=drop)
                    total, ok = total + g, ok + (g > 0)
                continue
            stats = [defaultdict(int) for _ in cur_b]
            tp = time.time()
            futs = [ex.submit(job, x, tl.cap - 1 - j, stats[j]) for j, x in enumerate(cur_b)]
            props = [f.result() for f in futs]
            t_prop += time.time() - tp
            changed_units, changed_tasks = set(), set()
            for j, (x, prop) in enumerate(zip(cur_b, props)):
                if prop is None:                             # 批开始时的状态下找不到划算的链：算作这一遍没成功
                    for k_, v_ in stats[j].items():
                        tl.stats[k_] += v_
                    tl.fail_count[x] += 1
                    continue
                if not tl._still_valid(prop, changed_units, changed_tasks):
                    redo += 1
                    tl.stats["并行冲突重找"] += 1
                    if not redo_on:                          # 冲突的这一遍放弃，下一遍在新状态上再试
                        continue
                    tr = time.time()
                    g = tl.try_add(x, max_depth=max_depth, min_gain=min_gain, shift=shift, retries=retries, drop=drop)
                    t_redo += time.time() - tr
                    if g > 0:
                        changed_units.update(tl.last_change[0])
                        changed_tasks.update(tl.last_change[1])
                        total, ok = total + g, ok + 1
                    continue
                for k_, v_ in stats[j].items():
                    tl.stats[k_] += v_
                g = tl._apply(prop)
                changed_units.update(tl.last_change[0])
                changed_tasks.update(tl.last_change[1])
                total, ok = total + g, ok + 1
            n_done += len(cur_b)
            if verbose and n_done % log_every < batch:
                print(f"  [弹射链·{threads} 线程] {n_done}/{len(targets)}  成功 {ok}  重找 {redo}  累计 +{total:.1f} 分  "
                      f"{time.time() - t0:.0f}s", flush=True)
    return {"tried": len(targets), "added": ok, "gain_points": total, "seconds": time.time() - t0, "redo": redo,
            "t_propose": round(t_prop, 1), "t_redo": round(t_redo, 1)}


def drop_pass(tl: Timeline, max_priority: float = 3.0) -> dict:
    """对优先级不高于 max_priority 的重复观测逐个试"删掉换延长"。"""
    total, ok = 0.0, 0
    for y in range(tl.n):
        x = tl.tk_tg[y]
        if tl.alive[y] and tl.tg_cnt[x] >= 2 and tl.P[x] <= max_priority:
            g = tl.try_drop(int(y))
            if g > 0:
                total += g
                ok += 1
    return {"dropped": ok, "gain_points": total}


def repattern_pass(tl: Timeline, targets, max_depth: int = 3, verbose: bool = True, log_every: int = 100,
                   deadline: float | None = None, **kw) -> dict:
    """对已有两次观测、还缺一次的目标逐个试"重排节奏"。"""
    t0 = time.time()
    needy = defaultdict(list)
    for x in targets:
        for u in np.unique(tl.c_unit[tl.c_ptr[x]:tl.c_ptr[x + 1]]):
            needy[int(u)].append(int(x))
    total, ok = 0.0, 0
    for n, x in enumerate(targets):
        if deadline is not None and time.time() > deadline:
            break
        g = tl.repattern(int(x), max_depth=max_depth, needy_in_unit=needy, **kw)
        if g > 0:
            total += g
            ok += 1
        if verbose and (n + 1) % log_every == 0:
            print(f"  [重排节奏] {n + 1}/{len(targets)}  成功 {ok}  累计 +{total:.1f} 分  {time.time() - t0:.0f}s", flush=True)
    return {"tried": len(targets), "repatterned": ok, "gain_points": total, "seconds": time.time() - t0}


def improve(pb: Problem, sch: Schedule, minutes: float = 5.0, depth: int = 4, rounds: str = "0:0,40:0,40:20,0:0",
            passes: int = 4, shift: int = 4, retries: int = 6, drop: bool = True, repattern: bool = True,
            repattern_depth: int = 3, min_priority: float = 5.0, seed: int = 0, extra_windows: dict | None = None,
            bounds=None, workers: int = 4, verbose: bool = False, on_best=None,
            threads: int | None = None) -> tuple[Schedule, dict]:
    """弹射链的整套流程（scripts/chain_run.py 与 `obsplan plan --chain` 共用）：依次按 rounds 里的时间线切法，
    每种切法做若干遍"弹射链 → 重排节奏"，每种切法结束后全局按秒精修一次，只保留变好的。
    minutes：总时长上限（0 = 不限）。bounds：全部窗口的按秒真实区间 (L, R)，不给时读缓存或现算（第一次约十几秒）。
    on_best(sch, ev, used)：每次变好时回调（存盘用）。返回最好计划和统计。"""
    from pathlib import Path
    from .refine import refine
    from .schedule import evaluate
    from .second_bounds import all_bounds
    cfg, ds = pb.cfg, pb.ds
    threads = default_threads() if threads is None else max(1, int(threads))
    # AI 预测器（obsplan/chain_ai.py）：跳过"大概率又失败、又白白耗时"的目标；OBSPLAN_CHAIN_AI=1 打开
    pred, ai_thr, ai_skipped = None, float(os.environ.get("OBSPLAN_CHAIN_AI_THR", "0.7")), 0
    if os.environ.get("OBSPLAN_CHAIN_AI", "0") != "0":                   # 实测固定时间内不提分，默认关闭
        from .chain_ai import features as ai_features, load_predictor
        pred = load_predictor()
    t_start = time.time()
    deadline = t_start + minutes * 60 if minutes > 0 else None
    late = lambda: deadline is not None and time.time() > deadline          # noqa: E731
    say = print if verbose else (lambda *a, **k: None)

    def ev(s):
        e = evaluate(cfg, ds, s)
        return {"covered": int(e["covered"]), "score": round(float(e["score"]), 3),
                "combined": round(float(e["combined"]), 6), "tasks": int(e["tasks"])}

    if bounds is None:
        bounds = all_bounds(pb, Path(cfg.cache_dir) / f"bounds_all_{cfg.scenario_key()}.npz", workers=workers,
                            verbose=verbose)
    wL, wR = bounds
    rng = np.random.default_rng(seed)
    e0 = ev(sch)
    best, best_ev, best_used = sch, e0, None
    stats_all, log = {}, []
    for spec in [r for r in rounds.split(",") if r]:
        if late():
            break
        piece, offset = (int(v) for v in spec.split(":"))
        tl = Timeline(pb, wL, wR, best, piece=piece, offset=offset, extra_windows=extra_windows)
        if tl.unmatched or not tl.unit_ok.all():
            say(f"  [弹射链] 切法 {spec}：{tl.unmatched} 个任务对不上窗口、{int((~tl.unit_ok).sum())} 条时间线不可行")
        drop_pass(tl)
        for k in range(passes):
            cand = np.nonzero((pb.P >= min_priority) & (tl.tg_cnt < np.minimum(3, pb.max_obs)) & (tl.tg_cnt > 0))[0]
            cand = cand[rng.permutation(len(cand))]
            cand = cand[np.argsort(-pb.P[cand], kind="stable")]
            if pred is not None and len(cand):
                keep = pred.prob(ai_features(tl, cand)) <= ai_thr
                ai_skipped += int((~keep).sum())
                cand = cand[keep]
            if threads > 1:
                r = run_pass_parallel(tl, cand, threads=threads, max_depth=depth, verbose=verbose, deadline=deadline,
                                      shift=shift, retries=retries, drop=drop)
            else:
                r = run_pass(tl, cand, max_depth=depth, verbose=verbose, deadline=deadline, shift=shift,
                             retries=retries, drop=drop)
            r.update(spec=spec, step="chain", **{"pass": k})
            log.append(r)
            if repattern and not late():
                cand = np.nonzero((pb.P >= min_priority) & (tl.tg_cnt == 2) & (pb.max_obs >= 3))[0]
                cand = cand[rng.permutation(len(cand))]
                rp = repattern_pass(tl, cand, max_depth=repattern_depth, verbose=verbose, deadline=deadline,
                                    shift=shift, retries=retries, drop=drop)
                rp.update(spec=spec, step="repattern", **{"pass": k})
                log.append(rp)
                r["added"] += rp["repatterned"]
            say(f"  [弹射链] 切法 {spec} 第 {k} 遍：加入 {r['added']} 次（{time.time() - t_start:.0f}s）")
            if r["added"] == 0 or late():
                break
        for k_, v_ in tl.stats.items():
            stats_all[k_] = stats_all.get(k_, 0) + v_
        cur, L, R, used = tl.schedule()
        e1 = ev(cur)
        ref, _ = refine(pb, cur, bounds=(L, R))           # 全局按秒精修：同目标 48 小时间隔按真实约束重新定时
        e2 = ev(ref)
        if e2["combined"] >= e1["combined"]:
            cur, e1 = ref, e2
        if e1["combined"] > best_ev["combined"]:
            best, best_ev, best_used = cur, e1, used
            if on_best is not None:
                on_best(best, best_ev, best_used)
    best.algo = (sch.algo + " + " if sch.algo else "") + "按秒跨站弹射链"
    return best, {"start": e0, "best": best_ev, "extra_windows": best_used, "stats": stats_all, "passes": log,
                  "seconds": round(time.time() - t_start, 1), "threads": threads,
                  "ai_skipped": ai_skipped if pred is not None else None}
