"""局部搜索（方案 E、G 和新算法共用同一套，保证对比公平）。

三个算子，按顺序执行：
1. 覆盖修复：为未覆盖目标找一个 120 秒位置；位置上若有 1~2 个任务挡路，
   尝试把挡路任务缩短、挪到它自己的其他窗口（可换站），或删掉低价值的重复观测；
   净收益为正才接受。对应规划方案 E 的"转移地面站 / 调整开始时间 / 用首次覆盖替换重复观测 / 240 秒缩短为 120 秒"。
2. 重复观测换延长：删掉一个重复观测，把相邻任务延长占用腾出的时间，净收益为正才接受。
3. 延长：用剩余空闲时间把任务延长到 240 秒，高优先级先延长。
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .greedy import State, _find_start
from .problem import Problem

RESERVED = -2


@njit(cache=True)
def _task_value(k, tg, m, t_cnt, P, frac, vc, vs):
    v = vs * P[tg[k]] * frac[m[k]]
    if t_cnt[tg[k]] == 1:
        v += vc
    return v


@njit(cache=True)
def _detach(k, st, tg, a, m, owner, t_cnt, t_obs_a, t_obs_m, t_task):
    """把任务 k 从站点占用和目标记录中摘下（不改 alive）。"""
    s = st[k]
    for c in range(a[k], a[k] + m[k] + 1):
        if owner[s, c] == k:
            owner[s, c] = -1
    i = tg[k]
    cnt = t_cnt[i]
    for x in range(cnt):
        if t_task[i, x] == k:
            t_task[i, x] = t_task[i, cnt - 1]
            t_obs_a[i, x] = t_obs_a[i, cnt - 1]
            t_obs_m[i, x] = t_obs_m[i, cnt - 1]
            t_task[i, cnt - 1] = -1
            t_cnt[i] = cnt - 1
            break


@njit(cache=True)
def _attach(k, st, tg, a, m, owner, t_cnt, t_obs_a, t_obs_m, t_task):
    s = st[k]
    for c in range(a[k], a[k] + m[k] + 1):
        owner[s, c] = k
    i = tg[k]
    cnt = t_cnt[i]
    t_task[i, cnt] = k
    t_obs_a[i, cnt] = a[k]
    t_obs_m[i, cnt] = m[k]
    t_cnt[i] = cnt + 1


@njit(cache=True)
def _sep_ok_excl(i, k, aa, mm, t_cnt, t_obs_a, t_obs_m, t_task, gap):
    for x in range(t_cnt[i]):
        if t_task[i, x] == k:
            continue
        b, mb = t_obs_a[i, x], t_obs_m[i, x]
        if not (aa >= b + mb + gap or aa + mm + gap <= b):
            return False
    return True


@njit(cache=True)
def _relocate_option(k, st, tg, a, m, w, owner, t_cnt, t_obs_a, t_obs_m, t_task,
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
            aa = _find_start(owner[w_st[ww]], w_k0[ww], w_k1[ww], mm, t_cnt[i], t_obs_a[i], t_obs_m[i], gap, True, T)
            if aa >= 0:
                return ww, aa, mm
    return best_w, best_a, best_m


@njit(cache=True)
def repair_coverage(targets, st, tg, a, m, w, alive, n, owner, t_cnt, t_obs_a, t_obs_m, t_task,
                    w_st, w_k0, w_k1, t_ptr, t_win, P, frac, vc, vs, gap, m_min, max_pos):
    """对每个未覆盖目标尝试一次最优插入。返回 (新任务数 n, 成功插入数)。"""
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
            wu = t_win[p]
            s = w_st[wu]
            k0, k1 = w_k0[wu], w_k1[wu]
            L = k1 - k0 - m_min
            if L < 0:
                continue
            step = 1 if L < max_pos else (L + max_pos - 1) // max_pos
            for au in range(k0, k1 - m_min + 1, step):
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
                    rw, ra, rm = _relocate_option(b, st, tg, a, m, w, owner, t_cnt, t_obs_a, t_obs_m, t_task,
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


@njit(cache=True)
def _set_obs(i, k, aa, mm, t_cnt, t_obs_a, t_obs_m, t_task):
    for x in range(t_cnt[i]):
        if t_task[i, x] == k:
            t_obs_a[i, x] = aa
            t_obs_m[i, x] = mm


@njit(cache=True)
def extend_all(order, st, tg, a, m, w, alive, n, owner, t_cnt, t_obs_a, t_obs_m, t_task,
               w_k0, w_k1, gap, m_max):
    """按给定顺序把任务向后、向前延长到最多 m_max 格。返回延长的总格数。"""
    total = 0
    for k in order:
        if not alive[k]:
            continue
        s, i = st[k], tg[k]
        while m[k] < m_max:
            e = a[k] + m[k] + 1               # 延长后新增占用的格
            if a[k] + m[k] + 1 <= w_k1[w[k]] and owner[s, e] == -1 and \
                    _sep_ok_excl(i, k, a[k], m[k] + 1, t_cnt, t_obs_a, t_obs_m, t_task, gap):
                owner[s, e] = k
                m[k] += 1
            elif a[k] - 1 >= w_k0[w[k]] and owner[s, a[k] - 1] == -1 and \
                    _sep_ok_excl(i, k, a[k] - 1, m[k] + 1, t_cnt, t_obs_a, t_obs_m, t_task, gap):
                a[k] -= 1
                owner[s, a[k]] = k
                m[k] += 1
            else:
                break
            total += 1
            _set_obs(i, k, a[k], m[k], t_cnt, t_obs_a, t_obs_m, t_task)
    return total



@njit(cache=True)
def compact(order, direction, st, tg, a, m, w, alive, owner, t_cnt, t_obs_a, t_obs_m, t_task, w_k0, w_k1, gap):
    """同站任务按时间顺序整体左移（direction=-1）或右移（+1），把碎片空隙并到一起。返回移动的任务数。"""
    moved = 0
    T = owner.shape[1]
    last_s, bound = -1, 0
    for k in order:
        if not alive[k]:
            continue
        s = st[k]
        if s != last_s:
            last_s = s
            bound = 0 if direction < 0 else T - 1
        if direction < 0:
            target = max(w_k0[w[k]], bound)
            if target < a[k] and _sep_ok_excl(tg[k], k, target, m[k], t_cnt, t_obs_a, t_obs_m, t_task, gap):
                for c in range(a[k], a[k] + m[k] + 1):
                    owner[s, c] = -1
                a[k] = target
                for c in range(a[k], a[k] + m[k] + 1):
                    owner[s, c] = k
                _set_obs(tg[k], k, a[k], m[k], t_cnt, t_obs_a, t_obs_m, t_task)
                moved += 1
            bound = a[k] + m[k] + 1
        else:
            target = min(w_k1[w[k]] - m[k], bound - m[k])
            if target > a[k] and _sep_ok_excl(tg[k], k, target, m[k], t_cnt, t_obs_a, t_obs_m, t_task, gap):
                for c in range(a[k], a[k] + m[k] + 1):
                    owner[s, c] = -1
                a[k] = target
                for c in range(a[k], a[k] + m[k] + 1):
                    owner[s, c] = k
                _set_obs(tg[k], k, a[k], m[k], t_cnt, t_obs_a, t_obs_m, t_task)
                moved += 1
            bound = a[k] - 1
    return moved

@njit(cache=True)
def swap_repeat_for_extension(st, tg, a, m, w, alive, n, owner, t_cnt, t_obs_a, t_obs_m, t_task,
                              w_k0, w_k1, P, frac, vs, gap, m_max):
    """删除一个重复观测 r，让同站前后相邻任务延长占用腾出的格子；收益为正才接受。"""
    T = owner.shape[1]
    swaps = 0
    for r in range(n):
        if not alive[r] or t_cnt[tg[r]] < 2:
            continue
        s = st[r]
        lose = vs * P[tg[r]] * frac[m[r]]
        # 找左邻（占用在 r 之前紧挨着）和右邻
        left = owner[s, a[r] - 1] if a[r] - 1 >= 0 else -1
        right = owner[s, a[r] + m[r] + 1] if a[r] + m[r] + 1 < T else -1
        if left < 0 and right < 0:
            continue
        gain = 0.0
        # 左邻向后延长
        lext = 0
        if left >= 0:
            while m[left] + lext < m_max and a[left] + m[left] + lext + 1 <= w_k1[w[left]] \
                    and a[left] + m[left] + lext + 1 <= a[r] + m[r] and \
                    _sep_ok_excl(tg[left], left, a[left], m[left] + lext + 1, t_cnt, t_obs_a, t_obs_m, t_task, gap):
                lext += 1
            gain += vs * P[tg[left]] * (frac[m[left] + lext] - frac[m[left]])
        rext = 0
        if right >= 0:
            lim = a[r] - 1 + lext
            while m[right] + rext < m_max and a[right] - rext - 1 >= w_k0[w[right]] \
                    and a[right] - rext - 1 > lim and \
                    _sep_ok_excl(tg[right], right, a[right] - rext - 1, m[right] + rext + 1, t_cnt, t_obs_a, t_obs_m, t_task, gap):
                rext += 1
            gain += vs * P[tg[right]] * (frac[m[right] + rext] - frac[m[right]])
        if gain <= lose + 1e-12:
            continue
        _detach(r, st, tg, a, m, owner, t_cnt, t_obs_a, t_obs_m, t_task)
        alive[r] = False
        if left >= 0 and lext > 0:
            for c in range(a[left] + m[left] + 1, a[left] + m[left] + lext + 1):
                owner[s, c] = left
            m[left] += lext
            _set_obs(tg[left], left, a[left], m[left], t_cnt, t_obs_a, t_obs_m, t_task)
        if right >= 0 and rext > 0:
            for c in range(a[right] - rext, a[right]):
                owner[s, c] = right
            a[right] -= rext
            m[right] += rext
            _set_obs(tg[right], right, a[right], m[right], t_cnt, t_obs_a, t_obs_m, t_task)
        swaps += 1
    return swaps


def improve(pb: Problem, state: State, rounds: int = 3, max_pos: int = 40, verbose=False) -> dict:
    """局部搜索主流程：
    每轮：覆盖修复（可挤让） -> 左压紧 + 补空 -> 重复观测插入（可挤让） -> 右压紧 + 补空；
    最后：重复观测换延长 -> 延长到 240 秒。"""
    from .greedy import candidates, run_greedy
    stats = {"inserted": 0, "repeats": 0, "moved": 0, "swaps": 0, "extended_cells": 0}
    cw, cm = candidates(pb, [pb.m_min])
    v = pb.vs * pb.P[pb.w_tg[cw]] * pb.frac[cm]
    has_win = np.diff(pb.t_ptr) > 0

    def args():
        return (state.st, state.tg, state.a, state.m, state.w, state.alive, state.owner,
                state.t_cnt, state.t_obs_a, state.t_obs_m, state.t_task)

    def station_order(rev=False):
        idx = np.nonzero(state.alive[:state.n])[0]
        key = state.a[idx] if not rev else -state.a[idx]
        return idx[np.lexsort((key, state.st[idx]))]

    def fill():
        """轮内只补新覆盖；重复观测和延长留到最后按单位时间收益分档处理。"""
        n0 = state.n
        run_greedy(pb, cw, cm, v + pb.vc, np.full(len(cw), -1.0), best_fit=True, state=state, cutoff=0.0)
        return state.n - n0

    def tiered_fill():
        """按"每格收益"从高到低交替做：某优先级的重复观测（120 秒）或某优先级任务的延长。"""
        acts = []
        for P in np.unique(pb.P):
            acts.append((pb.vs * P * pb.frac[pb.m_min] / (pb.m_min + 1), 0, P))
            acts.append((pb.vs * P * (pb.frac[pb.m_max] - pb.frac[pb.m_min]) / (pb.m_max - pb.m_min), 1, P))
        acts.sort(key=lambda x: -x[0])
        added = ext = 0
        for _, kind, P in acts:
            if kind == 0:
                sel = pb.P[pb.w_tg[cw]] == P
                n0 = state.n
                run_greedy(pb, cw[sel], cm[sel], v[sel] + pb.vc, v[sel], best_fit=True, state=state)
                added += state.n - n0
            else:
                idx = np.nonzero(state.alive[:state.n] & (pb.P[state.tg[:state.n]] == P))[0]
                ext += extend_all(idx, state.st, state.tg, state.a, state.m, state.w, state.alive, state.n,
                                  state.owner, state.t_cnt, state.t_obs_a, state.t_obs_m, state.t_task,
                                  pb.w_k0, pb.w_k1, pb.gap, pb.m_max)
        return added, ext

    for r in range(rounds):
        unc = np.nonzero((state.t_cnt == 0) & has_win)[0]
        unc = unc[np.argsort(-pb.P[unc], kind="stable")]
        state.n, ins = repair_coverage(unc, state.st, state.tg, state.a, state.m, state.w, state.alive, state.n,
                                       state.owner, state.t_cnt, state.t_obs_a, state.t_obs_m, state.t_task,
                                       pb.w_st, pb.w_k0, pb.w_k1, pb.t_ptr, pb.t_win, pb.P, pb.frac,
                                       pb.vc, pb.vs, pb.gap, pb.m_min, max_pos)
        st_, tg_, a_, m_, w_, alive_, owner_, tc, toa, tom, tt = args()
        mv = compact(station_order(), -1, st_, tg_, a_, m_, w_, alive_, owner_, tc, toa, tom, tt, pb.w_k0, pb.w_k1, pb.gap)
        f1 = fill()
        rep = np.nonzero((state.t_cnt > 0) & (state.t_cnt < 3))[0]
        rep = rep[np.argsort(-pb.P[rep], kind="stable")]
        state.n, ins2 = repair_coverage(rep, state.st, state.tg, state.a, state.m, state.w, state.alive, state.n,
                                        state.owner, state.t_cnt, state.t_obs_a, state.t_obs_m, state.t_task,
                                        pb.w_st, pb.w_k0, pb.w_k1, pb.t_ptr, pb.t_win, pb.P, pb.frac,
                                        pb.vc, pb.vs, pb.gap, pb.m_min, max_pos)
        st_, tg_, a_, m_, w_, alive_, owner_, tc, toa, tom, tt = args()
        mv += compact(station_order(rev=True), 1, st_, tg_, a_, m_, w_, alive_, owner_, tc, toa, tom, tt, pb.w_k0, pb.w_k1, pb.gap)
        f2 = fill()
        stats["inserted"] += ins + f1 + f2
        stats["repeats"] += ins2
        stats["moved"] += mv
        if verbose:
            print(f"  [LS] 第{r + 1}轮 覆盖插入 {ins} 重复插入 {ins2} 压紧移动 {mv} 补空 {f1 + f2}")
        if ins + ins2 + f1 + f2 == 0:
            break
    stats["tier_added"], stats["tier_ext"] = tiered_fill()
    stats["swaps"] = swap_repeat_for_extension(state.st, state.tg, state.a, state.m, state.w, state.alive, state.n,
                                               state.owner, state.t_cnt, state.t_obs_a, state.t_obs_m, state.t_task,
                                               pb.w_k0, pb.w_k1, pb.P, pb.frac, pb.vs, pb.gap, pb.m_max)
    order = np.argsort(-pb.P[state.tg[:state.n]], kind="stable")
    stats["extended_cells"] = extend_all(order, state.st, state.tg, state.a, state.m, state.w, state.alive, state.n,
                                         state.owner, state.t_cnt, state.t_obs_a, state.t_obs_m, state.t_task,
                                         pb.w_k0, pb.w_k1, pb.gap, pb.m_max)
    return stats
