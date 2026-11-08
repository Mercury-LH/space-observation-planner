"""Three explicit exact pruning rules: ``benchmarks.h_acceleration_bounds:solve``.

This self-contained diagnostic candidate copies only three operators from
obsplan/local_search.py and obsplan/greedy.py at a5d7835. It adds zero-loss
repair upper-bound exits, partial-loss blocker pruning, and a relocation-only
best-fit exit after a free run achieves zero waste. All other solver, local
search, greedy placement and refinement code remains shared. No candidate1
helper is imported, so the benchmark's plugin fingerprint covers this code.
Enable explicitly with acceleration='multi_bound' in an isolated worker.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from numba import njit

from obsplan import local_search as ls
from obsplan import portfolio
from obsplan.greedy import _sep_ok
from obsplan.local_search import RESERVED, _attach, _detach, _sep_ok_excl, _task_value
from obsplan.problem import Problem
from obsplan.schedule import Schedule


@njit(cache=True)
def _find_start_exact(owner_row, k0, k1, m, cnt, obs_a, obs_m, gap, best_fit, T):
    """在窗口 [k0,k1] 内为 m 格观测找开始格，占用 [a, a+m]。找不到返回 -1。"""
    best_a, best_w = -1, 1 << 30
    c = k0
    while c <= k1:
        if owner_row[c] != -1:
            c += 1
            continue
        r0 = c
        while c <= k1 and owner_row[c] == -1:
            c += 1
        r1 = c - 1                         # 窗口内空档 [r0, r1]
        if r1 - r0 < m:
            continue
        g0 = r0                            # 真实空档（不受窗口裁剪），只需看碎片是否 <= m 格
        while g0 - 1 >= 0 and owner_row[g0 - 1] == -1 and r0 - g0 <= m + 1:
            g0 -= 1
        g1 = r1
        while g1 + 1 < T and owner_row[g1 + 1] == -1 and g1 - r1 <= m + 1:
            g1 += 1
        # 候选位置：左对齐、右对齐、以及 48h 约束边界
        for k in range(2 + 2 * cnt):
            if k == 0:
                a = r0
            elif k == 1:
                a = r1 - m
            elif k % 2 == 0:
                x = (k - 2) // 2
                a = obs_a[x] + obs_m[x] + gap
            else:
                x = (k - 3) // 2
                a = obs_a[x] - gap - m
            if a < r0 or a + m > r1:
                continue
            if not _sep_ok(a, m, cnt, obs_a, obs_m, gap):
                continue
            if not best_fit:
                if best_a < 0 or a < best_a:
                    best_a = a
                continue
            left, right = a - g0, g1 - (a + m)
            w = 0
            if 0 < left < m + 1:
                w += left
            if 0 < right < m + 1:
                w += right
            if w < best_w or (w == best_w and a < best_a):
                best_w, best_a = w, a
        if not best_fit and best_a >= 0:
            return best_a
        # Exact pruning 3: all positions in this run have been checked. Later
        # free runs start later and cannot beat (zero waste, earliest start).
        if best_fit and best_w == 0:
            return best_a
    return best_a



@njit(cache=True)
def _relocate_option_exact(k, st, tg, a, m, w, owner, t_cnt, t_obs_a, t_obs_m, t_task,
                     w_st, w_k0, w_k1, t_ptr, t_win, gap, m_min, T):
    """任务 k 已被摘下。在它目标的所有窗口里找新位置，时长尽量不缩短。
    返回 (新窗口, 新开始, 新时长)，找不到返回 (-1, -1, -1)。"""
    i = tg[k]
    best_w, best_a, best_m = -1, -1, -1
    for mm in range(m[k], m_min - 1, -1):
        for p in range(t_ptr[i], t_ptr[i + 1]):
            ww = t_win[p]
            if w_k1[ww] - w_k0[ww] < mm:
                continue
            aa = _find_start_exact(owner[w_st[ww]], w_k0[ww], w_k1[ww], mm, t_cnt[i], t_obs_a[i], t_obs_m[i], gap, True, T)
            if aa >= 0:
                return ww, aa, mm
    return best_w, best_a, best_m



@njit(cache=True)
def repair_coverage_multi_bound(targets, st, tg, a, m, w, alive, n, owner, t_cnt, t_obs_a, t_obs_m, t_task,
                              w_st, w_k0, w_k1, t_ptr, t_win, P, frac, vc, vs, gap, m_min, max_pos):
    """Baseline repair with first-optimum-preserving exact pruning.

    Requires finite nonnegative P, vc, vs and finite nonnegative nondecreasing
    frac. solve() validates this contract before installing the operator.
    """
    T = owner.shape[1]
    inserted = 0
    cap = len(st)
    for u in targets:
        if t_cnt[u] >= 3 or n + 4 >= cap:
            continue
        gain_base = (vc if t_cnt[u] == 0 else 0.0) + vs * P[u] * frac[m_min]
        best_gain = 1e-12
        best = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
        # best = (w, a, b1, b1_action, b1_w, b1_a, b1_m, b2, b2_action, b2_w, b2_a, b2_m)
        for p in range(t_ptr[u], t_ptr[u + 1]):
            if best_gain >= gain_base:  # Exact pruning 1a: no later window can improve zero loss.
                break
            wu = t_win[p]
            s = w_st[wu]
            k0, k1 = w_k0[wu], w_k1[wu]
            L = k1 - k0 - m_min
            if L < 0:
                continue
            step = 1 if L < max_pos else (L + max_pos - 1) // max_pos
            for au in range(k0, k1 - m_min + 1, step):
                if best_gain >= gain_base:  # Exact pruning 1b: keep the first maximum-gain placement.
                    break
                if t_cnt[u] > 0 and not _sep_ok_excl(u, -1, au, m_min, t_cnt, t_obs_a, t_obs_m, t_task, gap):
                    continue
                b1, b2, nb = -1, -1, 0
                for c in range(au, au + m_min + 1):
                    o = owner[s, c]
                    if o >= 0 and o != b1 and o != b2:
                        if nb == 0:
                            b1 = o
                        elif nb == 1:
                            b2 = o
                        nb += 1
                if nb == 0:
                    if gain_base > best_gain:
                        best_gain = gain_base
                        best = (wu, au, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
                    continue
                if nb > 2:
                    continue
                # 依次处理挡路任务：摘下 -> 预留 u 的格子 -> 找缩短/挪走/删除中损失最小的办法
                loss = 0.0
                ok = True
                acts = np.full((2, 5), -1, np.int64)
                for bi in range(nb):
                    b = b1 if bi == 0 else b2
                    vb = _task_value(b, tg, m, t_cnt, P, frac, vc, vs)
                    _detach(b, st, tg, a, m, owner, t_cnt, t_obs_a, t_obs_m, t_task)
                    for c in range(au, au + m_min + 1):
                        if owner[s, c] == -1:
                            owner[s, c] = RESERVED
                    best_loss, act, nw, na, nm = 1e18, -1, -1, -1, -1
                    # 缩短：保留不与 [au, au+m_min] 相交的一段
                    if st[b] == s:
                        e = a[b] + m[b]
                        if a[b] < au:
                            mm = min(m[b], au - 1 - a[b])
                            if mm >= m_min:
                                lo = vs * P[tg[b]] * (frac[m[b]] - frac[mm])
                                if lo < best_loss:
                                    best_loss, act, nw, na, nm = lo, 1, w[b], a[b], mm
                        if e > au + m_min:
                            na2 = au + m_min + 1
                            mm = e - na2
                            if mm >= m_min and na2 >= w_k0[w[b]]:
                                lo = vs * P[tg[b]] * (frac[m[b]] - frac[min(mm, m[b])])
                                if lo < best_loss:
                                    best_loss, act, nw, na, nm = lo, 1, w[b], na2, min(mm, m[b])
                    # 挪走（可换站）
                    rw, ra, rm = _relocate_option_exact(b, st, tg, a, m, w, owner, t_cnt, t_obs_a, t_obs_m, t_task,
                                                  w_st, w_k0, w_k1, t_ptr, t_win, gap, m_min, T)
                    if rw >= 0:
                        lo = vs * P[tg[b]] * (frac[m[b]] - frac[rm])
                        if lo < best_loss:
                            best_loss, act, nw, na, nm = lo, 2, rw, ra, rm
                    # 删除
                    if vb < best_loss:
                        best_loss, act, nw, na, nm = vb, 3, -1, -1, -1
                    # 恢复
                    for c in range(au, au + m_min + 1):
                        if owner[s, c] == RESERVED:
                            owner[s, c] = -1
                    _attach(b, st, tg, a, m, owner, t_cnt, t_obs_a, t_obs_m, t_task)
                    if act < 0:
                        ok = False
                        break
                    loss += best_loss
                    # Exact pruning 2: remaining blocker losses cannot be
                    # negative, so this position cannot improve the incumbent.
                    if gain_base - loss <= best_gain:
                        ok = False
                        break
                    acts[bi, 0], acts[bi, 1], acts[bi, 2], acts[bi, 3], acts[bi, 4] = b, act, nw, na, nm
                if not ok:
                    continue
                # 两个挡路任务都选"挪走"时可能互相冲突，保守起见只允许其中一个挪走
                if nb == 2 and acts[0, 1] == 2 and acts[1, 1] == 2:
                    continue
                g = gain_base - loss
                if g > best_gain:
                    best_gain = g
                    best = (wu, au, acts[0, 0], acts[0, 1], acts[0, 2], acts[0, 3], acts[0, 4],
                            acts[1, 0], acts[1, 1], acts[1, 2], acts[1, 3], acts[1, 4])
        if best[0] < 0:
            continue
        wu, au = best[0], best[1]
        s = w_st[wu]
        ok = True
        for bi in range(2):
            b, act, nw, na, nm = best[2 + 5 * bi], best[3 + 5 * bi], best[4 + 5 * bi], best[5 + 5 * bi], best[6 + 5 * bi]
            if b < 0:
                continue
            _detach(b, st, tg, a, m, owner, t_cnt, t_obs_a, t_obs_m, t_task)
            if act == 3:
                alive[b] = False
                continue
            # 重新检查新位置仍然可行（另一个挡路任务的处理可能已改变占用）
            free = True
            for c in range(na, na + nm + 1):
                if owner[w_st[nw], c] != -1 or (w_st[nw] == s and au <= c <= au + m_min):
                    free = False
                    break
            if not free:
                _attach(b, st, tg, a, m, owner, t_cnt, t_obs_a, t_obs_m, t_task)
                ok = False
                break
            st[b], a[b], m[b], w[b] = w_st[nw], na, nm, nw
            _attach(b, st, tg, a, m, owner, t_cnt, t_obs_a, t_obs_m, t_task)
        if not ok:
            continue
        clear = True
        for c in range(au, au + m_min + 1):
            if owner[s, c] != -1:
                clear = False
        if not clear or not _sep_ok_excl(u, -1, au, m_min, t_cnt, t_obs_a, t_obs_m, t_task, gap):
            continue
        st[n], tg[n], a[n], m[n], w[n] = s, u, au, m_min, wu
        alive[n] = True
        _attach(n, st, tg, a, m, owner, t_cnt, t_obs_a, t_obs_m, t_task)
        n += 1
        inserted += 1
    return n, inserted


def _check_objective(pb: Problem) -> None:
    """Reject objectives for which blocker loss may be negative."""
    if (not np.isfinite(pb.P).all() or np.any(pb.P < 0)
            or not np.isfinite(pb.vc) or pb.vc < 0 or not np.isfinite(pb.vs) or pb.vs < 0
            or not np.isfinite(pb.frac).all() or np.any(pb.frac < 0)
            or np.any(np.diff(pb.frac) < 0)):
        raise ValueError("multi_bound requires finite nonnegative P/vc/vs and nonnegative nondecreasing frac")


def solve(pb: Problem, spec: dict, progress=None) -> tuple[Schedule, dict]:
    """Temporarily install one operator, restore it on every return/exception."""
    started = perf_counter()
    if not isinstance(spec, dict):
        raise ValueError("acceleration spec must be a dict")
    solver_spec = dict(spec)
    if solver_spec.pop("acceleration", None) != "multi_bound":
        raise ValueError("explicit acceleration='multi_bound' is required")
    _check_objective(pb)
    original = ls.repair_coverage
    try:
        ls.repair_coverage = repair_coverage_multi_bound
        schedule, info = portfolio.solve(pb, solver_spec, progress=progress)
    finally:
        ls.repair_coverage = original
    runtime = perf_counter() - started
    schedule.runtime_s = runtime
    info = {**info, "runtime_s": runtime,
            "acceleration": {"candidate": "multi_bound", "changed_factors": 3,
                             "changes": ["repair loops stop at zero-loss gain upper bound",
                                         "skip remaining blockers when partial loss cannot improve incumbent",
                                         "relocation best-fit returns after a zero-waste free run"],
                             "baseline": "obsplan.portfolio:solve with original obsplan.local_search.repair_coverage",
                             "objective_contract": "finite nonnegative P/vc/vs; finite nonnegative nondecreasing frac",
                             "scope": "all shared local-search calls during this isolated portfolio solve"}}
    return schedule, info
