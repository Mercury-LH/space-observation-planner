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


@njit(cache=True, nogil=True)
def _solve_1d_exact(l, h, P, dmin, dmax, sw, out_s, out_e):
    """单条时间线精确求解：最大化 sum P[i] * (e[i] - s[i])，约束 l[i] <= s[i]、e[i] <= h[i]、
    dmin <= e[i] - s[i] <= dmax、s[i+1] >= e[i] + sw。凸分段线性动态规划（slope trick），
    与线性规划 GLOP 数学上完全等价、输出精确整秒且释放 GIL。"""
    n = len(l)
    if n == 0:
        return True, 0.0
    m = 2 * n
    cap = 2 * m + 8
    bx = np.empty(cap, np.int64)
    bs = np.empty(cap, np.float64)
    nbx = np.empty(cap, np.int64)
    nbs = np.empty(cap, np.float64)
    mlo = np.empty(m, np.int64)
    mhi = np.empty(m, np.int64)

    dlo = l[0]
    dhi = h[0] - dmin
    if dlo > dhi:
        return False, 0.0
    nb = 0
    bs[0] = P[0]
    mlo[0] = dlo
    mhi[0] = dlo

    for k in range(1, m):
        i = k >> 1
        if (k & 1) == 1:
            a, b, inf_b = dmin, dmax, False
            w = -P[i]
            L, U = l[i] + dmin, h[i]
        else:
            a, b, inf_b = sw, 0, True
            w = P[i]
            L, U = l[i], h[i] - dmin

        ndlo = dlo + a
        ndhi = U if inf_b else (dhi + b)
        if ndlo > U or (not inf_b and ndhi < L) or ndhi < ndlo:
            return False, 0.0

        nn = 0
        pos_idx = 0
        while pos_idx <= nb and bs[pos_idx] < -1e-12:
            pos_idx += 1
        for j in range(pos_idx):
            nbs[nn] = bs[j]
            nbx[nn] = (bx[j] if j < nb else dhi) + a
            nn += 1
        if inf_b:
            nbs[nn] = 0.0
            n_seg = nn + 1
        else:
            gt_idx = pos_idx
            while gt_idx <= nb and bs[gt_idx] <= 1e-12:
                gt_idx += 1
            plat_r = mhi[k - 1] + b
            if gt_idx <= nb:
                if plat_r > (nbx[nn - 1] if nn > 0 else ndlo):
                    nbs[nn] = 0.0
                    nbx[nn] = plat_r
                    nn += 1
                for j in range(gt_idx, nb + 1):
                    nbs[nn] = bs[j]
                    if j < nb:
                        nbx[nn] = bx[j] + b
                        nn += 1
                n_seg = nn + 1
            else:
                nbs[nn] = 0.0
                n_seg = nn + 1

        for j in range(n_seg):
            nbs[j] += w

        cdlo = L if L > ndlo else ndlo
        cdhi = U if U < ndhi else ndhi
        if cdlo > cdhi:
            return False, 0.0

        nb = 0
        first = 0
        while first < n_seg - 1 and nbx[first] <= cdlo:
            first += 1
        bs[0] = nbs[first]
        for j in range(first, n_seg - 1):
            x_pt = nbx[j]
            if x_pt >= cdhi:
                break
            sl = nbs[j + 1]
            if sl > bs[nb] + 1e-12:
                bx[nb] = x_pt
                nb += 1
                bs[nb] = sl
        dlo, dhi = cdlo, cdhi

        j = 0
        while j <= nb and bs[j] < -1e-12:
            j += 1
        mlo[k] = dlo if j == 0 else (bx[j - 1] if j <= nb else dhi)
        j2 = j
        while j2 <= nb and bs[j2] <= 1e-12:
            j2 += 1
        mhi[k] = dlo if j2 == 0 else (bx[j2 - 1] if j2 <= nb else dhi)

    cur = mlo[m - 1]
    out_e[n - 1] = cur
    for k in range(m - 2, -1, -1):
        if (k & 1) == 0:
            lo_x, hi_x = cur - dmax, cur - dmin
            if mlo[k] > hi_x:
                cur = hi_x
            elif mhi[k] < lo_x:
                cur = lo_x
            else:
                cur = mlo[k] if mlo[k] >= lo_x else lo_x
            out_s[k >> 1] = cur
        else:
            hi_x = cur - sw
            cur = hi_x if mlo[k] > hi_x else mlo[k]
            out_e[k >> 1] = cur

    obj = 0.0
    for i in range(n):
        obj += P[i] * (out_e[i] - out_s[i])
    return True, obj


@njit(cache=True, nogil=True)
def _in_sorted(a, v):
    """v 是否在升序数组 a 里（二分查找）。"""
    i = np.searchsorted(a, v)
    return i < len(a) and a[i] == v


@njit(cache=True, nogil=True)
def _lp_fast(flat, ptr, cflat, cptr, wl0, wh0, ovr_t, ovr_w, ovr_r, drop, lp_k,
             tk_tg, tk_w, tk_s, tk_e, tg_tasks, wL, wR, P, gap, dmin, dmax, sw, out_t, out_s, out_e):
    """Timeline._lp 的快速路径（编译、释放全局锁）。flat[ptr[k]:ptr[k+1]]：第 k 条改动的时间线的新顺序；
    cflat[cptr[k]:cptr[k+1]]：它现在的顺序；wl0 / wh0：它的墙。只重排"新旧顺序不同的那一段"前后各 lp_k 个任务
    （lp_k <= 0 时整条），其余按当前起止固定。与原来的 Python 实现逐项同序计算，结果逐位相同。
    同一目标有两个任务同时进了规划（不同的夜，要隔 48 小时）时，先不管这条间隔、各条时间线分别精确求解：
    解出来恰好满足间隔，它就是带间隔的问题的最优解（放松问题的最优解可行即最优），直接用；不满足才交给 GLOP 联合求解。
    返回 (状态, 新得分, 输出个数)：状态 1 = 可行，0 = 无解，−1 = 需要 GLOP 联合求解。
    out_t/out_s/out_e 依次是各时间线参与规划的任务、然后各时间线固定的任务及其起止。"""
    nu = len(ptr) - 1
    kx = 0.4 / (dmax - dmin)
    A = np.empty(nu, np.int64)
    B = np.empty(nu, np.int64)
    WLO = np.empty(nu, np.int64)
    WHI = np.empty(nu, np.int64)
    fixed = 0.0
    n_lp = 0
    for k in range(nu):
        p0, p1 = ptr[k], ptr[k + 1]
        n_new = p1 - p0
        wlo, whi = wl0[k], wh0[k]
        if lp_k <= 0:
            a, b = 0, n_new
        else:
            c0, c1 = cptr[k], cptr[k + 1]
            n_cur = c1 - c0
            m = min(n_new, n_cur)
            i0 = 0
            while i0 < m and flat[p0 + i0] == cflat[c0 + i0]:
                hit = False
                for t in range(len(ovr_t)):
                    if ovr_t[t] == flat[p0 + i0]:
                        hit = True
                        break
                if hit:
                    break
                i0 += 1
            j0 = 0
            while j0 < m - i0 and flat[p1 - 1 - j0] == cflat[c1 - 1 - j0]:
                hit = False
                for t in range(len(ovr_t)):
                    if ovr_t[t] == flat[p1 - 1 - j0]:
                        hit = True
                        break
                if hit:
                    break
                j0 += 1
            a, b = max(0, i0 - lp_k), min(n_new, n_new - j0 + lp_k)
            if b <= a:
                a, b = max(0, i0 - lp_k), min(n_new, i0 + lp_k)
            if a > 0:
                wlo = max(wlo, tk_e[flat[p0 + a - 1]] + sw)
            if b < n_new:
                whi = min(whi, tk_s[flat[p0 + b]] - sw)
            for j in range(n_new):
                if j < a or j >= b:
                    q = flat[p0 + j]
                    fixed += P[tk_tg[q]] * (0.6 + kx * (tk_e[q] - tk_s[q] - dmin))
        A[k], B[k], WLO[k], WHI[k] = a, b, wlo, whi
        n_lp += b - a
    lpt = np.empty(n_lp, np.int64)
    i = 0
    for k in range(nu):
        for j in range(A[k], B[k]):
            lpt[i] = flat[ptr[k] + j]
            i += 1
    srt = np.sort(lpt)
    tg_lp = tk_tg[lpt]
    t_ord = np.argsort(tg_lp)
    coupled = False
    for j in range(1, n_lp):
        if tg_lp[t_ord[j]] == tg_lp[t_ord[j - 1]]:
            coupled = True
            break
    lo = np.empty(n_lp, np.int64)
    hi = np.empty(n_lp, np.int64)
    ref = np.empty(n_lp, np.int64)
    i = 0
    for k in range(nu):
        for j in range(A[k], B[k]):
            p = flat[ptr[k] + j]
            w, r = tk_w[p], tk_s[p]
            for t in range(len(ovr_t)):
                if ovr_t[t] == p:
                    w, r = ovr_w[t], ovr_r[t]
                    break
            if w >= 0:
                l, h = wL[w], wR[w]
            else:
                l, h = tk_s[p], tk_e[p]
            l, h = max(l, WLO[k]), min(h, WHI[k])
            x = tk_tg[p]
            for c in range(tg_tasks.shape[1]):
                q = tg_tasks[x, c]
                if q < 0 or q == p or _in_sorted(srt, q):
                    continue
                skip = False
                for t in range(len(drop)):
                    if drop[t] == q:
                        skip = True
                        break
                if skip:
                    continue
                if tk_s[q] > r:
                    h = min(h, tk_s[q] - gap)
                else:
                    l = max(l, tk_e[q] + gap)
            if h - l < dmin:
                return 0, 0.0, 0
            lo[i], hi[i], ref[i] = l, h, r
            i += 1
    new = fixed
    i = 0
    for k in range(nu):
        m_ = B[k] - A[k]
        if m_ == 0:
            continue
        Ps = np.empty(m_, np.float64)
        for j in range(m_):
            Ps[j] = P[tk_tg[lpt[i + j]]]
        os_ = np.empty(m_, np.int64)
        oe_ = np.empty(m_, np.int64)
        ok, _ = _solve_1d_exact(lo[i:i + m_], hi[i:i + m_], Ps, dmin, dmax, sw, os_, oe_)
        if not ok:
            return 0, 0.0, 0
        for j in range(m_):
            out_t[i + j], out_s[i + j], out_e[i + j] = lpt[i + j], os_[j], oe_[j]
            new += Ps[j] * (0.6 + kx * (oe_[j] - os_[j] - dmin))
        i += m_
    if coupled:                                       # 同目标的几个任务按参考开始时间排序，相邻的要隔 gap
        j = 0
        while j < n_lp:
            j2 = j + 1
            while j2 < n_lp and tg_lp[t_ord[j2]] == tg_lp[t_ord[j]]:
                j2 += 1
            if j2 - j > 1:
                idx = t_ord[j:j2]
                idx = idx[np.argsort(ref[idx], kind="mergesort")]
                for c in range(1, len(idx)):
                    if out_s[idx[c]] - out_e[idx[c - 1]] < gap:
                        return -1, 0.0, 0
            j = j2
    for k in range(nu):
        for j in range(ptr[k + 1] - ptr[k]):
            if j < A[k] or j >= B[k]:
                q = flat[ptr[k] + j]
                out_t[i], out_s[i], out_e[i] = q, tk_s[q], tk_e[q]
                i += 1
    return 1, new, i


@njit(cache=True, nogil=True)
def _units_pts(units, seq2, ulen, tk_tg, tk_s, tk_e, P, dmin, k):
    """这些时间线上全部任务的得分之和（与 Python 逐个相加同序，结果逐位相同）。"""
    tot = 0.0
    for u in units:
        for j in range(ulen[u]):
            p = seq2[u, j]
            tot += P[tk_tg[p]] * (0.6 + k * (tk_e[p] - tk_s[p] - dmin))
    return tot


@njit(cache=True, nogil=True)
def _ext_pts(ids, s, e, P, tk_tg, kx, dmin):
    """一串任务按给定起止算的"延长分"之和（同序相加）。"""
    tot = 0.0
    for j in range(len(ids)):
        tot += P[tk_tg[ids[j]]] * kx * (e[j] - s[j] - dmin)
    return tot


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True)
def _open_mask(targets, c_ptr, c_l, c_r, tk_s, tk_e, tg_tasks, gap, dmin):
    """每颗目标：它的候选里有没有"与已有观测隔 48 小时后还放得下一次"的时间段（只看时间，不看那里有没有空位）。"""
    out = np.zeros(len(targets), np.bool_)
    sa = np.empty(4, np.int64)
    sb = np.empty(4, np.int64)
    for i in range(len(targets)):
        x = targets[i]
        for ci in range(c_ptr[x], c_ptr[x + 1]):
            if _slots(x, -1, c_l[ci], c_r[ci], tk_s, tk_e, tg_tasks, gap, dmin, sa, sb) > 0:
                out[i] = True
                break
    return out


@njit(cache=True, nogil=True)
def _search(new_tid, max_depth, max_q, D, c_ptr, c_unit, c_l, c_r, tk_tg, tk_s, tk_e, tk_w, tg_tasks, tg_cnt,
            seq2, ulen, A2, B2, lo2, hi2, unit_ok, gap, dmin, sw, visited, stamp, ban_unit, ban_task, attempt,
            q_task, q_par, q_c, q_k, q_a, q_unit, q_depth, P, drop_below, root_ok, skip_u):
    """广度优先找弹射链。root_ok[ci]：链头任务（新观测）允许用哪些候选。skip_u >= 0 时整条链禁止触碰该时间线。
    候选 ci =（目标的某个窗口, 某条时间线, 可用区间 [c_l, c_r]）。
    队列第 e 项 = "任务 q_task[e] 需要找位置"；它是被父项的任务顶出来的：
    父项任务用候选 q_c[e]、插在时间线 q_unit[e] 原第 q_k[e] 个任务之前（参考开始时间 q_a[e]）。
    返回 (末项 e, 候选, 位置 k, 参考开始时间, 方式, 被删任务)：
      方式 0 = 末项任务直接塞进空当 k；方式 1 = 末项任务顶掉一个低价值的重复观测（直接删掉，不再安置）。
    找不到返回 e = −1。"""
    q_task[0], q_par[0], q_unit[0], q_depth[0] = new_tid, -1, skip_u, 0
    root_tg = tk_tg[new_tid]
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
                while p >= 0:
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
                            can_drop = P[y] < drop_below and tg_cnt[y] >= 2 and y != root_tg
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
        mq = int(os.environ.get("OBSPLAN_CHAIN_MAXQ", "60000"))     # 一次搜链最多展开多少个"待安置任务"
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
        返回 (这些时间线的新得分, {任务: (开始, 结束)})；不可行返回 None。
        先只重排改动前后各 lp_k 个任务；这一带排得太满、空隙在更远处时局部规划会无解，这时放宽到整条时间线再解一次。"""
        res = self._lp(new_seqs, override, dropped, self.lp_k)
        if res is None and self.lp_k > 0:
            res = self._lp(new_seqs, override, dropped, 0)
        return res

    def _lp(self, new_seqs: dict, override: dict, dropped, lp_k: int):
        """局部重排。99%+ 的调用走编译版 _lp_fast（释放全局锁，多线程能真正并行）；
        同一目标有两个任务同时进规划时回退到 _lp_glop 联合求解。两条路径的结果与原来的 Python 实现逐位相同。"""
        units = list(new_seqs)
        lens = [len(new_seqs[u]) for u in units]
        ptr = np.zeros(len(units) + 1, np.int64)
        ptr[1:] = np.cumsum(lens)
        total = int(ptr[-1])
        flat = np.fromiter((p for u in units for p in new_seqs[u]), np.int64, total)
        if lp_k > 0:
            curs = [self.seq2[u, :self.ulen[u]] if self.ulen[u] == len(self.seq[u]) else np.asarray(self.seq[u], np.int64)
                    for u in units]
            cptr = np.zeros(len(units) + 1, np.int64)
            cptr[1:] = np.cumsum([len(c) for c in curs])
            cflat = np.concatenate(curs).astype(np.int64, copy=False) if curs else np.zeros(0, np.int64)
        else:
            cptr, cflat = np.zeros(len(units) + 1, np.int64), np.zeros(0, np.int64)
        u_arr = np.asarray(units, np.int64)
        wl0 = self.wall_lo[u_arr]
        wh0 = self.wall_hi[u_arr] - self.sw
        no = len(override)
        ovr_t = np.fromiter(override.keys(), np.int64, no)
        ovr_w = np.fromiter((v[0] for v in override.values()), np.int64, no)
        ovr_r = np.fromiter((v[1] for v in override.values()), np.int64, no)
        drop = np.asarray(dropped, np.int64).reshape(-1)
        out_t = np.empty(total, np.int64)
        out_s = np.empty(total, np.int64)
        out_e = np.empty(total, np.int64)
        st, new, n_out = _lp_fast(flat, ptr, cflat, cptr, wl0, wh0, ovr_t, ovr_w, ovr_r, drop, lp_k,
                                  self.tk_tg, self.tk_w, self.tk_s, self.tk_e, self.tg_tasks, self.wL, self.wR, self.P,
                                  self.gap, self.dmin, self.dmax, self.sw, out_t, out_s, out_e)
        if st == 0:
            return None
        if st < 0:
            return self._lp_glop(new_seqs, override, dropped, lp_k)
        return new, dict(zip(out_t[:n_out].tolist(), zip(out_s[:n_out].tolist(), out_e[:n_out].tolist())))

    def _lp_glop(self, new_seqs: dict, override: dict, dropped, lp_k: int):
        """原来的 Python 实现（含 GLOP 联合求解）：同一目标有两个任务同时进规划时用。"""
        dmin, dmax, sw, gap = self.dmin, self.dmax, self.sw, self.gap
        kx = 0.4 / (dmax - dmin)
        # 只重排改动附近的任务：每条时间线取"新旧顺序不同的那一段"前后各 lp_k 个任务，其余按当前起止固定，
        # 固定任务成为这一段的边界。lp_k = 0 时整条时间线都进线性规划（旧行为）。
        local, fixed_pts = {}, 0.0
        for u, seq in new_seqs.items():
            n_new = len(seq)
            if lp_k <= 0:
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
            a, b = max(0, i0 - lp_k), min(n_new, n_new - j0 + lp_k)
            if b <= a:
                a, b = max(0, i0 - lp_k), min(n_new, i0 + lp_k)
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
        ref, lim = {}, {}
        by_t = defaultdict(list)
        unit_subs = []
        for u, seq in new_seqs.items():
            a_, b_, wlo, whi = local[u]
            sub = seq[a_:b_]
            unit_subs.append((u, sub))
            for p in sub:
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
                by_t[x].append(p)
        # 快速路径（99%+ 的调用）：本次规划的任务里没有同一目标出现两次，各时间线完全解耦，
        # 用精确的 1D 凸分段线性动态规划求解（与 GLOP 数学等价，快约 95 倍且释放 GIL）。
        if len(by_t) == len(tasks):
            out = {}
            new = fixed_pts
            for _u, sub in unit_subs:
                m_ = len(sub)
                if m_ == 0:
                    continue
                ls = np.empty(m_, np.int64)
                hs = np.empty(m_, np.int64)
                Ps = np.empty(m_, np.float64)
                for idx, p in enumerate(sub):
                    ls[idx], hs[idx] = lim[p]
                    Ps[idx] = self.P[self.tk_tg[p]]
                os_ = np.empty(m_, np.int64)
                oe_ = np.empty(m_, np.int64)
                ok, _ = _solve_1d_exact(ls, hs, Ps, dmin, dmax, sw, os_, oe_)
                if not ok:
                    return None
                for idx, p in enumerate(sub):
                    s_, e_ = int(os_[idx]), int(oe_[idx])
                    out[p] = (s_, e_)
                    new += Ps[idx] * (0.6 + kx * (e_ - s_ - dmin))
            for u, seq in new_seqs.items():
                a_, b_ = local[u][0], local[u][1]
                for p in seq[:a_] + seq[b_:]:
                    out[p] = (int(self.tk_s[p]), int(self.tk_e[p]))
            return new, out
        # 少数跨时间线同目标耦合情形：回退到 GLOP 联合求解
        from ortools.linear_solver import pywraplp
        lp = pywraplp.Solver.CreateSolver("GLOP")
        sv, dv = {}, {}
        obj = lp.Objective()
        inf = lp.infinity()
        for p in tasks:
            l, h = lim[p]
            sv[p] = lp.NumVar(float(l), float(h - dmin), "")
            dv[p] = lp.NumVar(float(dmin), float(dmax), "")
            c = lp.RowConstraint(-inf, float(h))
            c.SetCoefficient(sv[p], 1.0)
            c.SetCoefficient(dv[p], 1.0)
            obj.SetCoefficient(dv[p], float(self.P[self.tk_tg[p]]))
        pairs = []
        for _u, sub in unit_subs:
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
        us = list(units)
        if all(self.ulen[u] == len(self.seq[u]) for u in us):
            return _units_pts(np.asarray(us, np.int64), self.seq2, self.ulen, self.tk_tg, self.tk_s, self.tk_e, self.P,
                              self.dmin, k)
        return sum(self.P[self.tk_tg[p]] * (0.6 + k * (self.tk_e[p] - self.tk_s[p] - self.dmin))
                   for u in us for p in self.seq[u])

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
                drop: bool = True, root_mask=None, drop_below: float | None = None, pattern: int = 0) -> float:
        """给目标 x 多加一次观测。成功返回得分增量（分），否则 0。root_mask：只允许新观测用这些候选。
        shift：顶出一个任务后，新任务可以插在它前后几个位置以内；retries：一条链不划算时禁用出问题的那一步再搜几次；
        drop：允许直接删掉优先级更低的重复观测来腾位置。
        pattern：留节奏（见 _propose_pattern）。0 = 不管；1 = 优先放在留得住节奏的位置，放不进再随便放；2 = 只放这样的位置。"""
        if self.tg_cnt[x] >= 3:
            return 0.0
        if pattern and root_mask is None:
            prop = self._propose_pattern(x, self.n, self.scr, max_depth, min_gain, shift, retries, drop, self.stats,
                                         strict=int(pattern) >= 2)
        else:
            prop = self._propose(x, self.n, self.scr, max_depth, min_gain, shift, retries, drop, root_mask, self.stats,
                                 drop_below)
        if prop is None:
            self.fail_count[x] += 1
            return 0.0
        return self._apply(prop)

    def _propose_pattern(self, x: int, tid: int, scr: _Scratch, max_depth: int, min_gain: float, shift: int,
                         retries: int, drop: bool, stats, strict: bool = False) -> dict | None:
        """留节奏：目标 x 只有一次观测、最多能排三次时，第二次只放在"放完之后第三次还有窗口可用"的候选里
        （6 天里排三次要两两隔 48 小时，第二次随手一放，第三次常常就无处可放了）。这样的候选都放不进去时，
        strict 为假就再不加限制地找一次——先拿到第二次观测的分；为真就放弃（留给后面更深的搜索）。
        其他情况与 _propose 相同。"""
        if self.tg_cnt[x] == 1 and self.pb.max_obs[x] >= 3:
            K = int(self.tg_tasks[x, 0])
            nm = _pattern_mask(x, int(self.tk_s[K]), int(self.tk_e[K]), -1, self.c_ptr, self.c_w, self.c_l, self.c_r,
                               self.gap, self.dmin, self.root_mask)       # 只写 x 自己那一段，线程之间不冲突
            if nm > 0:
                prop = self._propose(x, tid, scr, max_depth, min_gain, shift, retries, drop, self.root_mask, stats)
                if prop is not None:
                    stats["留节奏"] += 1
                    return prop
                stats["留节奏失败"] += 1
                if strict:
                    return None
        return self._propose(x, tid, scr, max_depth, min_gain, shift, retries, drop, None, stats)

    def _propose(self, x: int, tid: int, scr: _Scratch, max_depth: int, min_gain: float, shift: int, retries: int,
                 drop: bool, root_mask, stats, drop_below: float | None = None) -> dict | None:
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
                qt, qp, qc, qk, qa, qu, qd, self.P,
                (float(self.P[x]) if drop_below is None else float(drop_below)) if drop else 0.0,
                self.all_ok if root_mask is None else root_mask, -1)
            if e < 0:
                stats["找不到链"] += 1
                return None
            new_seqs, override, steps = {}, {}, []          # steps：(时间线, 被顶出/删掉的任务或 −1)
            recipe = []                                     # (时间线, 插入的任务, 插在谁前面（−1 = 末尾）, 拿掉的任务或 −1)
            u = int(self.c_unit[ci])
            seq = list(self.seq[u])
            dropped = ()
            recipe.append((u, int(qt[e]), seq[int(k)] if k < len(seq) else -1, int(Y) if mode else -1))
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
                recipe.append((u, int(qt[par]), seq[kk] if kk < len(seq) else -1, int(qt[e])))
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
                            "dropped": dropped, "gain": float(gain), "recipe": recipe, "min_gain": min_gain,
                            "key": f"成功_深度{depth}" + ("_删" if mode else "")}
                stats["不划算"] += 1
                # 哪一步丢的延长最多，就禁用哪一步再搜
                loss = []
                for u_, _y in steps:
                    ids = np.asarray(self.seq[u_], np.int64)
                    o = _ext_pts(ids, self.tk_s[ids], self.tk_e[ids], self.P, self.tk_tg, kx, self.dmin)
                    ns = new_seqs[u_]
                    se = np.array([times[p] for p in ns], np.int64).reshape(-1, 2)
                    nw = _ext_pts(np.asarray(ns, np.int64), se[:, 0], se[:, 1], self.P, self.tk_tg, kx, self.dmin)
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

    def _rebase(self, prop: dict) -> dict | None:
        """同一批里前面的方案已经改动了状态，这个方案不能原样提交时：按它的配方（每条时间线上插入谁、插在谁前面、
        拿掉谁）在最新的顺序上重做一遍，只重解线性规划——线性规划会按最新的起止时间重新检查窗口、切换和 48 小时间隔，
        可行且仍然划算就返回新方案。比重新搜链便宜得多（约 2 毫秒）。做不了返回 None。"""
        x, tid = prop["x"], prop["tid"]
        if self.tg_cnt[x] >= 3:
            return None
        for y in prop["dropped"]:
            if not self.alive[y] or self.tg_cnt[self.tk_tg[y]] < 2:
                return None
        new_seqs = {}
        for u, ins, anchor, rem in prop["recipe"]:
            if u in new_seqs or not self.unit_ok[u]:
                return None
            seq = list(self.seq[u])
            if rem >= 0:
                if not self.alive[rem] or self.tk_unit[rem] != u:
                    return None
            if anchor < 0:
                k = len(seq)
            elif self.alive[anchor] and self.tk_unit[anchor] == u:
                k = seq.index(anchor)
            else:
                return None
            seq.insert(k, ins)
            if rem >= 0:
                seq.remove(rem)
            if len(seq) > self.maxlen - 2:
                return None
            new_seqs[u] = seq
        self.tk_tg[tid] = x
        self.tk_w[tid] = -1
        old = self.units_score(new_seqs.keys())
        res = self.lp(new_seqs, prop["override"], prop["dropped"])
        if res is None:
            return None
        new, times = res
        if new - old < prop["min_gain"]:
            return None
        out = dict(prop)
        out.update(new_seqs=new_seqs, times=times, gain=float(new - old))
        return out

    def try_drop(self, y: int, min_gain: float = 1e-6, force: bool = False, last: bool = False):
        """删掉一个重复观测，把腾出的时间交给同一条时间线上的其他任务延长；总分变高才接受。
        force：不管划不划算都删（复合动作的第一步用），返回得分变化；规划不可行返回 None。
        last：连目标唯一的一次观测也允许删（只给"换位置"用：删完马上在别处重新放，放不回去就整体回退）。"""
        x = int(self.tk_tg[y])
        if not self.alive[y] or (self.tg_cnt[x] < 2 and not last) or self.tk_w[y] < 0:
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

    # ---------- 副本（多线程重排节奏：每个线程在自己的副本上试，试成了再落到正本） ----------
    _STATE = ("tk_st", "tk_tg", "tk_s", "tk_e", "tk_w", "tk_unit", "alive", "tg_tasks", "tg_cnt", "seq2", "ulen",
              "A2", "B2", "lo2", "hi2", "unit_ok", "root_mask", "fail_count")

    def clone(self) -> "Timeline":
        """可变状态各自一份、窗口和候选等只读数据共用的副本。"""
        import copy
        c = copy.copy(self)
        for k in self._STATE:
            setattr(c, k, getattr(self, k).copy())
        c.seq = [list(q) for q in self.seq]
        c.scr = _Scratch(self.cap, self.n_units, self.max_q)
        c.stats = defaultdict(int)
        return c

    def sync_from(self, src: "Timeline") -> None:
        """把自己的计划状态改成和 src 一样（统计数字不动）。"""
        self.n = src.n
        for k in self._STATE:
            getattr(self, k)[...] = getattr(src, k)
        self.seq = [list(q) for q in src.seq]

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

    def _can_absorb(self, y: int, min_absorb: float) -> bool:
        """O(1) 必要条件预过滤：拿走任务 y 后，同一时间线局部窗口内的其余任务按自身窗口与 240 秒上限最多还能延长多少分。
        如果上限都达不到 min_absorb，线性规划必然也达不到（零漏判）。"""
        u = int(self.tk_unit[y])
        seq = self.seq[u]
        n = len(seq)
        try:
            idx = seq.index(y)
        except ValueError:
            return False
        a = max(0, idx - self.lp_k) if self.lp_k > 0 else 0
        b = min(n, idx + self.lp_k + 1) if self.lp_k > 0 else n
        kx = 0.4 / (self.dmax - self.dmin)
        avail = 0.0
        for j in range(a, b):
            p = seq[j]
            if p == y:
                continue
            dur = int(self.tk_e[p] - self.tk_s[p])
            if dur >= self.dmax:
                continue
            w = int(self.tk_w[p])
            wcap = min(self.dmax, int(self.wR[w] - self.wL[w])) if w >= 0 else dur
            if wcap > dur:
                avail += float(self.P[self.tk_tg[p]]) * kx * (wcap - dur)
                if avail >= min_absorb - 1e-9:
                    return True
        return False

    def relocate(self, y: int, max_depth: int = 1, min_total: float = 1e-6, shift: int = 4, retries: int = 6,
                 min_absorb: float = 0.0) -> float:
        """换位置：把任务 y 从原处拿走（腾出的时间交给旁边的任务延长，且允许别的任务换进这个空位形成双站交换），
        再给同一颗目标在别的窗口重新放一次，放的时候可以顶掉优先级不高于它的重复观测。两步合起来总分变高才保留，
        否则微秒级局部还原（不拷贝全量数组）。"""
        x = int(self.tk_tg[y])
        if not self.alive[y] or self.tk_w[y] < 0 or self.n >= self.cap - 2:
            return 0.0
        if min_absorb > 0.0 and not self._can_absorb(y, min_absorb):
            return 0.0
        cnt, w_old = int(self.tg_cnt[x]), int(self.tk_w[y])
        lo, hi = int(self.c_ptr[x]), int(self.c_ptr[x + 1])
        mask = self.c_w[lo:hi] != w_old                              # 不放回原来那个窗口
        if not mask.any():
            return 0.0
        u = int(self.tk_unit[y])
        old_seq = self.seq[u]
        seq_wo = [p for p in old_seq if p != y]
        old_u = self.units_score([u])
        res = self.lp({u: seq_wo}, {}, (y,))
        if res is None:
            return 0.0
        new_u, times = res
        kx = 0.4 / (self.dmax - self.dmin)
        val = float(self.P[x]) * (0.6 + kx * (self.tk_e[y] - self.tk_s[y] - self.dmin))
        g0 = float(new_u - old_u)
        if min_absorb > 0.0 and g0 + val < min_absorb:
            return 0.0
        self.stats["换位置试探"] += 1
        old_tg_tasks = self.tg_tasks[x].copy()
        saved_times = [(p, int(self.tk_s[p]), int(self.tk_e[p]))
                       for p, (s_, e_) in times.items() if self.tk_s[p] != s_ or self.tk_e[p] != e_]
        touched = {u}
        for q in old_tg_tasks:
            if q >= 0 and q != y:
                touched.add(int(self.tk_unit[q]))
        for p, _, _ in saved_times:
            for q in self.tg_tasks[self.tk_tg[p]]:
                if q >= 0 and q != p:
                    touched.add(int(self.tk_unit[q]))
        self.commit({u: seq_wo}, {}, times, (y,))
        self.root_mask[lo:hi] = mask
        prop = self._propose(x, self.n, self.scr, max_depth, min_total - g0, shift, retries, True, self.root_mask,
                             self.stats, float(self.P[x]) + 1e-9)
        self.root_mask[lo:hi] = 0
        if prop is not None:
            g1 = self._apply(prop)
            if self.tg_cnt[x] == cnt:
                self.stats["换位置成功"] += 1
                return float(g0 + g1)
        self.seq[u] = old_seq
        self.tg_tasks[x] = old_tg_tasks
        self.tg_cnt[x] = cnt
        self.alive[y] = True
        for p, os_, oe_ in saved_times:
            self.tk_s[p], self.tk_e[p] = os_, oe_
        for u2 in touched:
            self.refresh(u2)
        self.stats["换位置回退"] += 1
        return 0.0


def relocate_pass(tl: Timeline, tasks=None, max_depth: int = 1, min_absorb: float = 0.05, deadline: float | None = None,
                  **kw) -> dict:
    """逐个任务试"换位置"（允许双站交换，失败时微秒级局部还原）。tasks 默认是全部任务，优先级低的先试。"""
    t0 = time.time()
    if tasks is None:
        tasks = np.nonzero(tl.alive[:tl.n] & (tl.tk_w[:tl.n] >= 0))[0]
        tasks = tasks[np.argsort(tl.P[tl.tk_tg[tasks]], kind="stable")]
    ok, total = 0, 0.0
    st0 = tl.stats["换位置试探"]
    for y in tasks:
        if deadline is not None and time.time() > deadline:
            break
        if tl.n >= tl.cap - 2:
            break
        g = tl.relocate(int(y), max_depth=max_depth, min_absorb=min_absorb, **kw)
        if g > 0:
            ok += 1
            total += g
    return {"tried": int(tl.stats["换位置试探"] - st0), "scanned": int(len(tasks)),
            "relocated": ok, "gain_points": total, "seconds": time.time() - t0}


def relocate_pass_parallel(tl: Timeline, tasks=None, threads: int | None = None, max_depth: int = 1,
                           min_absorb: float = 0.05, min_total: float = 1e-6, shift: int = 4, retries: int = 6,
                           deadline: float | None = None, batch: int | None = None) -> dict:
    """多线程版"换位置"（完整支持双站交换）：每个线程持有一份计划副本并发试探，失败时微秒级局部还原、不标脏；
    一批内第一个成功的直接把副本状态同步到正本，其余成功的在正本最新状态上重做。结果与线程调度和核数无关。"""
    from concurrent.futures import ThreadPoolExecutor
    threads = default_threads() if threads is None else max(1, int(threads))
    if threads <= 1:
        return relocate_pass(tl, tasks=tasks, max_depth=max_depth, min_absorb=min_absorb,
                             min_total=min_total, shift=shift, retries=retries, deadline=deadline)
    t0 = time.time()
    if tasks is None:
        tasks = np.nonzero(tl.alive[:tl.n] & (tl.tk_w[:tl.n] >= 0))[0]
        tasks = tasks[np.argsort(tl.P[tl.tk_tg[tasks]], kind="stable")]
    tasks = [int(y) for y in tasks]
    bsize = batch or threads
    clones = [tl.clone() for _ in range(bsize)]
    dirty = [False] * bsize
    ok, total, tried, copied, redone = 0, 0.0, 0, 0, 0
    tm = defaultdict(float)
    with ThreadPoolExecutor(threads) as ex:
        for i in range(0, len(tasks), bsize):
            if deadline is not None and time.time() > deadline:
                break
            if tl.n >= tl.cap - 2 * bsize:
                break
            cur = [y for y in tasks[i:i + bsize] if tl.alive[y] and tl.tk_w[y] >= 0]
            if not cur:
                continue
            t_ = time.perf_counter()
            for j in range(len(cur)):
                if dirty[j]:
                    clones[j].sync_from(tl)
                    dirty[j] = False
            tm["sync"] += time.perf_counter() - t_
            t_ = time.perf_counter()
            futs = [ex.submit(clones[j].relocate, y, max_depth, min_total, shift, retries, min_absorb)
                    for j, y in enumerate(cur)]
            gains = [f.result() for f in futs]
            tm["try"] += time.perf_counter() - t_
            t_ = time.perf_counter()
            changed = False
            winner_j = -1
            for j, (y, g) in enumerate(zip(cur, gains)):
                tried += clones[j].stats.pop("换位置试探", 0)
                for k_, v_ in clones[j].stats.items():
                    tl.stats[k_] += v_
                clones[j].stats.clear()
                if g <= 0:
                    continue
                if not changed:
                    tl.sync_from(clones[j])
                    changed = True
                    winner_j = j
                    copied += 1
                    ok += 1
                    total += g
                else:
                    g2 = tl.relocate(y, max_depth=max_depth, min_total=min_total, shift=shift, retries=retries,
                                     min_absorb=min_absorb)
                    redone += 1
                    if g2 > 0:
                        ok += 1
                        total += g2
                        winner_j = -1                                # 正本已比任何副本都新，全部副本下次需同步
            if changed:
                dirty = [j != winner_j for j in range(bsize)]
            tm["commit"] += time.perf_counter() - t_
    if os.environ.get("OBSPLAN_RELOC_TIMING"):
        print("  [换位置计时]", {k: round(v, 2) for k, v in tm.items()}, flush=True)
    return {"tried": tried, "scanned": len(tasks), "relocated": ok, "gain_points": total,
            "copied": copied, "redone": redone, "threads": threads, "seconds": time.time() - t0}


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
                      retries: int = 6, drop: bool = True, batch: int | None = None, redo: bool | None = None,
                      pattern: int = 0) -> dict:
    """多线程版 run_pass：每批 threads 颗目标，各线程在同一个状态上同时找链、解线性规划（只读，不改计划），
    然后主线程按原来的顺序逐个提交；提交前检查是否与本批前面已提交的方案冲突（动过同一条时间线、要删的观测已不在、
    同目标观测不再隔 48 小时），冲突的就在最新状态上单独重找。搜链（numba，不占全局解释器锁）占大头，所以能并行。
    批次和提交顺序固定，结果与线程调度、核数都无关，可复现（与单线程版不完全相同：同一批里后面的目标看到的是批开始时的状态）。
    实测（M1 Pro 8 线程，快速流水线的计划）：一整遍 30 秒对单线程 33 秒、综合 0.9849 对 0.9847；同样给 30 秒 / 1 分钟，
    综合比单线程高 0.0003～0.0004。提速不大，因为同一站同一夜的时间线大家都在抢，一批里约三成方案冲突、要单线程重找。"""
    import queue
    from concurrent.futures import ThreadPoolExecutor
    batch = batch or int(os.environ.get("OBSPLAN_CHAIN_BATCH", 0)) or 8       # 批次固定（默认 8）：结果与核数无关
    redo_on = (os.environ.get("OBSPLAN_CHAIN_REDO", "1") != "0") if redo is None else redo
    rebase_on = os.environ.get("OBSPLAN_CHAIN_REBASE", "1") != "0"
    defer_on = os.environ.get("OBSPLAN_CHAIN_DEFER", "1") != "0"
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
            if pattern:
                return tl._propose_pattern(x, tid, scr, max_depth, min_gain, shift, retries, drop, st,
                                           strict=int(pattern) >= 2)
            return tl._propose(x, tid, scr, max_depth, min_gain, shift, retries, drop, None, st)
        finally:
            pool.put(scr)
    with ThreadPoolExecutor(threads) as ex:
        i, n_done = 0, 0
        n_first = len(targets)                               # 之后追加在队尾的是"冲突后推迟重找"的目标
        while i < len(targets):
            if deadline is not None and time.time() > deadline:
                break
            cur_b = [x for x in targets[i:i + batch] if tl.tg_cnt[x] < 3]
            defer = defer_on and i < n_first                 # 第一轮的冲突推迟到队尾（仍然多线程找）；队尾那一轮再冲突才当场重找
            i += batch
            if tl.n + 2 * batch >= tl.cap:                   # 临时编号放在数组尾部，快满时退回单线程
                for x in cur_b:
                    g = tl.try_add(x, max_depth=max_depth, min_gain=min_gain, shift=shift, retries=retries, drop=drop,
                                   pattern=pattern)
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
                    rb = tl._rebase(prop) if rebase_on else None
                    if rb is not None:                       # 配方在最新状态上仍然可行、划算：直接提交
                        for k_, v_ in stats[j].items():
                            tl.stats[k_] += v_
                        tl.stats["并行冲突改期"] += 1
                        g = tl._apply(rb)
                        changed_units.update(tl.last_change[0])
                        changed_tasks.update(tl.last_change[1])
                        total, ok = total + g, ok + 1
                        continue
                    redo += 1
                    tl.stats["并行冲突重找"] += 1
                    if defer:
                        targets.append(x)
                        continue
                    if not redo_on:                          # 冲突的这一遍放弃，下一遍在新状态上再试
                        continue
                    tr = time.time()
                    g = tl.try_add(x, max_depth=max_depth, min_gain=min_gain, shift=shift, retries=retries, drop=drop,
                                   pattern=pattern)
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
    return {"tried": n_first, "added": ok, "gain_points": total, "seconds": time.time() - t0, "redo": redo,
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


def cover_pass(tl: Timeline, max_depth: int = 4, shift: int = 4, retries: int = 6,
               deadline: float | None = None) -> dict:
    """没覆盖到的目标逐个找链。多覆盖一颗在综合指标里值 vc / vs 分（赛题口径约 3 分），所以丢的得分只要比这少就接受；
    腾位置时可以直接删掉任何优先级的重复观测（一次 120 秒的高优先级重复观测也只值 3 分，还不算新观测自己的得分）。"""
    pb = tl.pb
    bonus = float(pb.vc / pb.vs)
    unc = np.nonzero((tl.tg_cnt == 0) & (pb.max_obs > 0) & (np.diff(tl.c_ptr) > 0))[0]
    unc = unc[np.argsort(-pb.P[unc], kind="stable")]
    t0 = time.time()
    ok, pts = 0, 0.0
    for x in unc:
        if deadline is not None and time.time() > deadline:
            break
        g = tl.try_add(int(x), max_depth=max_depth, min_gain=-bonus + 1e-6, shift=shift, retries=retries,
                       drop_below=1e9)
        if tl.tg_cnt[x] > 0:
            ok += 1
            pts += g
    return {"tried": int(len(unc)), "covered": ok, "gain_points": pts, "seconds": time.time() - t0}


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


def repattern_pass_parallel(tl: Timeline, targets, threads: int = 8, max_depth: int = 3, verbose: bool = True,
                            deadline: float | None = None, **kw) -> dict:
    """多线程版 repattern_pass。重排节奏是"删一次 → 换位置重放 → 补第三次"的复合动作，要改状态，不能像弹射链那样只读地找；
    所以每个线程拿一份计划副本，各自在副本上试一颗目标（绝大多数会失败并回退，时间都花在失败的搜链上，这部分不占全局锁）。
    一批试完后按顺序处理成功的：这一批里第一个成功的，正本还没变，直接把副本的状态抄过来；后面成功的在正本的最新状态上重做一遍。
    批次固定为 8、处理顺序固定，结果与线程调度和核数无关。"""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    batch = 8
    targets = [int(x) for x in targets]
    needy = defaultdict(list)
    for x in targets:
        for u in np.unique(tl.c_unit[tl.c_ptr[x]:tl.c_ptr[x + 1]]):
            needy[int(u)].append(int(x))
    clones = [tl.clone() for _ in range(batch)]
    dirty = [False] * batch
    total, ok, copied, redone = 0.0, 0, 0, 0
    with ThreadPoolExecutor(threads) as ex:
        for i in range(0, len(targets), batch):
            if deadline is not None and time.time() > deadline:
                break
            cur = [x for x in targets[i:i + batch] if tl.tg_cnt[x] == 2]
            for j in range(len(cur)):
                if dirty[j]:
                    clones[j].sync_from(tl)
                    dirty[j] = False
            futs = [ex.submit(clones[j].repattern, x, max_depth, needy, **kw) for j, x in enumerate(cur)]
            gains = [f.result() for f in futs]
            changed = False
            for j, (x, g) in enumerate(zip(cur, gains)):
                tl.stats["重排节奏回退"] += clones[j].stats.pop("重排节奏回退", 0)
                clones[j].stats.clear()
                if g <= 0:
                    continue
                dirty[j] = True
                if not changed:                              # 正本还是这一批开始时的状态：直接采用副本的结果
                    tl.sync_from(clones[j])
                    tl.stats["重排节奏成功"] += 1
                    copied += 1
                else:                                        # 正本已经变了：在最新状态上重做
                    g = tl.repattern(x, max_depth=max_depth, needy_in_unit=needy, **kw)
                    redone += 1
                if g > 0:
                    changed = True
                    total += g
                    ok += 1
            if changed:
                dirty = [True] * batch
    return {"tried": len(targets), "repatterned": ok, "gain_points": total, "seconds": time.time() - t0,
            "copied": copied, "redone": redone}


def improve(pb: Problem, sch: Schedule, minutes: float = 5.0, depth: int = 4, rounds: str = "0:0",
            passes: int = 1, shift: int = 4, retries: int = 6, drop: bool = True, repattern: bool = True,
            repattern_depth: int = 3, min_priority: float = 5.0, seed: int = 0, extra_windows: dict | None = None,
            bounds=None, workers: int = 4, verbose: bool = False, on_best=None,
            threads: int | None = None, cover: bool = True, pattern: bool = True,
            cycles: int | None = None, staged: bool = True, fill: bool = True,
            reserve_s: float = 4.0, relocate: bool = True, escalate_gain: float = 3e-4) -> tuple[Schedule, dict]:
    """弹射链的整套流程（scripts/chain_run.py 与 `obsplan plan --chain` 共用）：按 rounds 里的时间线切法轮换，
    每种切法：删低价值重复 → 补覆盖 → 弹射链一遍（留节奏）→ 删低价值重复 → 重排节奏（先做"节奏卡死"的目标）
    → 全局按秒精修，只保留变好的。每轮结束后（以及限时收尾阶段）做多线程"换位置"（relocate）把低/中优先级任务
    从高优先级旁边挪走腾出延长空间，再轮换回弹射链；多轮预算充足时自动加深搜索深度。
    minutes：总时长上限（0 = 不限）。bounds：全部窗口的按秒真实区间 (L, R)，不给时读缓存或现算（第一次约十几秒）。
    passes：每种切法里"弹射链 → 重排节奏"最多做几遍（实测同一种切法做第 2 遍以后收获很少，换切法更划算，所以默认 1）。
    cycles：切法最多轮换几轮（默认：限时的时候不限轮数，不限时的时候最多 6 轮）。
    cover：先给没覆盖到的目标找链；pattern：放第二次观测时给第三次留位置。
    staged：弹射链先浅后深——先只找深度 1 以内的链，再深度 2，最后才用到 depth。浅的链又快又不折腾别的任务，
    实测深度 1、2 两层几秒钟就拿到一遍里八成多的分。
    fill：每种切法的最后，给优先级低于 min_priority 的目标补重复观测（只往空当里直接塞），再删一遍不划算的重复观测。
    reserve_s：限时的时候给每轮收尾的便宜步骤（补低优先级、删重复、全局精修）留出的秒数——耗时的搜索提前这么多秒停。
    relocate：是否启用多线程"换位置"收尾与轮间交替（默认开启）。
    rounds 默认只用整夜一种切法：实测（rep5 起点，9 分钟）四种切法轮换时后三种每种 30～70 秒只涨 0～7 分，
    换成"整夜弹射链 → 换位置"更频繁地交替，240 秒就到了原来 9 分钟的分数。
    escalate_gain：一轮（弹射链 + 换位置）综合指标涨不到这么多（默认 3e-4，约 18 分）时，之后每轮只做深度 2 的
    换位置（贵，但在弹射链和浅层换位置都做到头的计划上仍然有效：实测 90 秒 +39 分，而再做一轮 8 层弹射链 47 秒 +0）；
    深度 2 也不涨了才停。
    限时收尾的换位置做完还有时间就再做一遍（换过之后别的任务又有了新位置）：推荐档实测（rep5 起点，77 秒）
    0.9899 → 0.9913，原来收尾只做一遍、门槛也更高，常常提前二十几秒就停了。
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
    hard = t_start + minutes * 60 if minutes > 0 else None
    deadline = None if hard is None else max(t_start + 0.5 * minutes * 60, hard - reserve_s)   # 耗时搜索的截止时间
    reloc_reserve = 0.0
    if relocate and deadline is not None and minutes * 60 >= 20.0:
        reloc_reserve = min(34.0, max(10.0, 0.38 * minutes * 60))
    chain_deadline = None if deadline is None else max(t_start + 8.0, deadline - reloc_reserve)
    late = lambda: chain_deadline is not None and time.time() > chain_deadline                 # noqa: E731
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
    specs = [r for r in rounds.split(",") if r]
    deep_batch = int(os.environ.get("OBSPLAN_CHAIN_DEEP_BATCH", "32"))   # 深层搜索大多失败、不改状态：批次大一些，线程不空等
    n_cycles = cycles if cycles is not None else (10 ** 6 if deadline is not None else 6)

    prof, mark = defaultdict(lambda: [0.0, 0.0]), [time.time(), time.process_time()]

    def tick(name):
        """把上次打点以来的墙钟时间和全进程 CPU 时间记到 name 名下（CPU / 墙钟 = 平均占用核数）。"""
        w, c = time.time(), time.process_time()
        prof[name][0] += w - mark[0]
        prof[name][1] += c - mark[1]
        mark[0], mark[1] = w, c

    def note(rec, spec, step, k, tl):
        rec.update(spec=spec, step=step, elapsed=round(time.time() - t_start, 1), points=round(tl.score_points(), 1),
                   **{"pass": k})
        log.append(rec)

    def run_relocate(min_abs: float, r_retries: int, dl: float | None, tag: str, r_depth: int = 1) -> int:
        nonlocal best, best_ev, best_used
        tick("其他")
        tl_r = Timeline(pb, wL, wR, best, piece=0, offset=0, extra_windows=extra_windows)
        tick("建时间线")
        rr = relocate_pass_parallel(tl_r, threads=threads, max_depth=r_depth, min_absorb=min_abs,
                                    shift=shift, retries=r_retries, deadline=dl)
        rr["depth"] = r_depth
        tick("换位置")
        dp_r = drop_pass(tl_r)
        tick("删重复")
        rr["dropped"] = dp_r["dropped"]
        note(rr, "0:0", tag, 0, tl_r)
        for k_, v_ in tl_r.stats.items():
            stats_all[k_] = stats_all.get(k_, 0) + v_
        cur_r, L_r, R_r, used_r = tl_r.schedule()
        e1_r = ev(cur_r)
        ref_r, _ = refine(pb, cur_r, bounds=(L_r, R_r))
        e2_r = ev(ref_r)
        if e2_r["combined"] >= e1_r["combined"]:
            cur_r, e1_r = ref_r, e2_r
        tick("全局精修")
        if e1_r["combined"] > best_ev["combined"]:
            best, best_ev, best_used = cur_r, e1_r, used_r
            if on_best is not None:
                on_best(best, best_ev, best_used)
        say(f"  [换位置·深度{r_depth}] {tag}：挪动 {rr['relocated']} 次，删 {dp_r['dropped']} 次，综合 {best_ev['combined']:.6f}"
            f"（{rr['seconds']:.1f}s）")
        return int(rr["relocated"] + dp_r["dropped"])

    reloc_done_since_change = False
    reloc_depth = 1
    for cyc in range(n_cycles):
        cyc_best = best_ev["combined"]
        cur_depth = depth if cyc == 0 else min(depth + 2 * cyc, 8)
        for spec in (specs if reloc_depth < 2 else []):          # 升级之后每轮只做深度 2 的换位置
            if late():
                break
            piece, offset = (int(v) for v in spec.split(":"))
            tick("其他")
            tl = Timeline(pb, wL, wR, best, piece=piece, offset=offset, extra_windows=extra_windows)
            tick("建时间线")
            if tl.unmatched or not tl.unit_ok.all():
                say(f"  [弹射链] 切法 {spec}：{tl.unmatched} 个任务对不上窗口、{int((~tl.unit_ok).sum())} 条时间线不可行")
            drop_pass(tl)
            tick("删重复")
            if cover:
                cp = cover_pass(tl, max_depth=max(cur_depth, 4), shift=shift, retries=retries, deadline=chain_deadline)
                tick("补覆盖")
                note(cp, spec, "cover", 0, tl)
                say(f"  [弹射链] 切法 {spec}：补覆盖 {cp['covered']}/{cp['tried']} 颗（{cp['seconds']:.1f}s）")
            for k in range(passes):
                cand = np.nonzero((pb.P >= min_priority) & (tl.tg_cnt < np.minimum(3, pb.max_obs)) & (tl.tg_cnt > 0))[0]
                cand = cand[rng.permutation(len(cand))]
                cand = cand[np.argsort(-pb.P[cand], kind="stable")]
                if pattern:                                  # 只有一次观测的先做：它们的第二次要挑位置
                    cand = cand[np.argsort(tl.tg_cnt[cand], kind="stable")]
                if pred is not None and len(cand):
                    keep = pred.prob(ai_features(tl, cand)) <= ai_thr
                    ai_skipped += int((~keep).sum())
                    cand = cand[keep]
                added = [0]

                def chain_stage(d, pat, batch):
                    left = cand[tl.tg_cnt[cand] < np.minimum(3, pb.max_obs[cand])]
                    if late() or not len(left):
                        return
                    if threads > 1:
                        rd = run_pass_parallel(tl, left, threads=threads, max_depth=d, verbose=verbose,
                                               deadline=chain_deadline, shift=shift, retries=retries, drop=drop,
                                               pattern=pat, batch=batch)
                    else:
                        rd = run_pass(tl, left, max_depth=d, verbose=verbose, deadline=chain_deadline, shift=shift,
                                      retries=retries, drop=drop, pattern=pat)
                    rd["depth"] = d
                    tick(f"弹射链·深度{d}")
                    added[0] += rd["added"]
                    note(rd, spec, "chain", k, tl)

                def repattern_stage(only_stuck):
                    if not repattern or late():
                        return
                    two = np.nonzero((pb.P >= min_priority) & (tl.tg_cnt == 2) & (pb.max_obs >= 3))[0]
                    two = two[rng.permutation(len(two))]
                    opened = _open_mask(two, tl.c_ptr, tl.c_l, tl.c_r, tl.tk_s, tl.tk_e, tl.tg_tasks, tl.gap, tl.dmin)
                    groups = [("repattern_stuck", two[~opened])] + ([] if only_stuck else [("repattern_open", two[opened])])
                    for name, grp in groups:
                        if late() or not len(grp):
                            continue
                        if threads > 1:
                            rp = repattern_pass_parallel(tl, grp, threads=threads, max_depth=repattern_depth,
                                                         verbose=verbose, deadline=chain_deadline, shift=shift,
                                                         retries=retries, drop=drop, pattern=1 if pattern else 0)
                        else:
                            rp = repattern_pass(tl, grp, max_depth=repattern_depth, verbose=verbose,
                                                deadline=chain_deadline, shift=shift, retries=retries, drop=drop,
                                                pattern=1 if pattern else 0)
                        tick("重排节奏")
                        note(rp, spec, name, k, tl)
                        added[0] += rp["repatterned"]

                dp = {"dropped": 0, "gain_points": 0.0}

                def drop_stage():
                    tick("其他")
                    d_ = drop_pass(tl)                       # 新插进来的高优先级任务让旁边的低价值重复观测不再划算
                    tick("删重复")
                    dp["dropped"] += d_["dropped"]
                    dp["gain_points"] += d_["gain_points"]

                def fill_stage():
                    """高优先级放不进去的零碎空当：给其余目标补重复观测（只直接塞，不顶别人），再删一遍不划算的重复观测。"""
                    if not fill:
                        return
                    low = np.nonzero((pb.P < min_priority) & (tl.tg_cnt < np.minimum(3, pb.max_obs)) & (tl.tg_cnt > 0))[0]
                    low = low[rng.permutation(len(low))]
                    low = low[np.argsort(-pb.P[low], kind="stable")]
                    if len(low):
                        if threads > 1:
                            rf = run_pass_parallel(tl, low, threads=threads, max_depth=0, verbose=False, shift=shift,
                                                   retries=retries, drop=False)
                        else:
                            rf = run_pass(tl, low, max_depth=0, verbose=False, shift=shift, retries=retries, drop=False)
                        tick("补其余优先级")
                        note(rf, spec, "fill", k, tl)
                    drop_stage()

                if staged and cur_depth > 2:
                    # 实测几种顺序里这一种最好：浅层弹射链 → 最深一层 → 重排节奏（先卡死的）→ 补其余目标。
                    # （把重排节奏或深度 3 提到最深一层之前、或者提前补其余目标，同样时间里都更差，见 README。）
                    for d in (1, 2):                         # 浅层：第二次观测只放留得住节奏的位置
                        chain_stage(d, 2 if pattern else 0, None)
                    drop_stage()
                    # 最深一层也只放留得住节奏的位置：否则只有一次观测的目标每颗要搜两遍（先限定、失败了再不限定），而深层搜索很贵
                    chain_stage(cur_depth, 2 if pattern else 0, deep_batch)
                    repattern_stage(False)
                    if pattern:                              # 实在留不住节奏的：浅层、不限定位置，先拿到第二次观测的分
                        chain_stage(2, 0, None)
                else:
                    chain_stage(cur_depth, 1 if pattern else 0, None)
                    drop_stage()
                    repattern_stage(False)
                drop_stage()
                fill_stage()
                note(dp, spec, "drop", k, tl)
                say(f"  [弹射链] 切法 {spec} 第 {k} 遍：加入 {added[0]} 次（{time.time() - t_start:.0f}s）")
                if added[0] == 0 or late():
                    break
            for k_, v_ in tl.stats.items():
                stats_all[k_] = stats_all.get(k_, 0) + v_
            cur, L, R, used = tl.schedule()
            e1 = ev(cur)
            ref, _ = refine(pb, cur, bounds=(L, R))           # 全局按秒精修：同目标 48 小时间隔按真实约束重新定时
            e2 = ev(ref)
            if e2["combined"] >= e1["combined"]:
                cur, e1 = ref, e2
            tick("全局精修")
            if e1["combined"] > best_ev["combined"]:
                best, best_ev, best_used = cur, e1, used
                reloc_done_since_change = False
                if on_best is not None:
                    on_best(best, best_ev, best_used)
        if relocate and not late():
            moved = run_relocate(0.5, min(retries, 3), deadline, f"reloc_cyc{cyc}", reloc_depth)
            reloc_done_since_change = moved == 0
        if late():
            break
        if best_ev["combined"] <= cyc_best + escalate_gain:
            if relocate and reloc_depth < 2:                 # 浅层做到头了：换位置升到深度 2 再试
                reloc_depth = 2
                reloc_done_since_change = False
                continue
            if best_ev["combined"] <= cyc_best + 1e-9:
                break
    if relocate and not reloc_done_since_change and (deadline is None or time.time() < deadline - 1.0):
        moved, n_final = run_relocate(0.5, min(retries, 3), deadline, "reloc_final", reloc_depth), 1
        # 还有时间就接着换：换过之后别的任务又有了新位置（每遍几秒，做到没得换或到点为止）
        while moved > 0 and deadline is not None and time.time() < deadline - 5.0:
            moved = run_relocate(0.5, min(retries, 3), deadline, f"reloc_final{n_final}", reloc_depth)
            n_final += 1
    tick("其他")
    best.algo = (sch.algo + " + " if sch.algo else "") + "按秒跨站弹射链"
    profile = {k: {"wall_s": round(w, 1), "cores": round(c / w, 2) if w > 0 else None}
               for k, (w, c) in sorted(prof.items(), key=lambda kv: -kv[1][0])}
    return best, {"start": e0, "best": best_ev, "extra_windows": best_used, "stats": stats_all, "passes": log,
                  "profile": profile,
                  "seconds": round(time.time() - t_start, 1), "threads": threads,
                  "ai_skipped": ai_skipped if pred is not None else None}
