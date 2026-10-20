"""贪心类方案（A 随机 / B 优先级 / C 人工多因素 / D AI 权重）的共用引擎。

候选 = (窗口, 观测格数 m)。开始时间不预先枚举，而是在插入时于窗口内找空档放置：
- first-fit：最早可行位置；
- best-fit：在可行空档里选产生"碎片"（1~4 格、放不下任何任务的空隙）最少的位置。
覆盖奖励是动态的（目标被覆盖后候选价值下降），用两条有序流归并实现精确的动态贪心。
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from .problem import Problem
from .schedule import Schedule

FEATURES = ["优先级", "首次覆盖", "任务得分", "机会稀缺度", "窗口紧迫度", "单位资源收益", "冲突代价", "资源占用"]


# ---------------------------------------------------------------- 引擎
@njit(cache=True)
def _sep_ok(a, m, cnt, obs_a, obs_m, gap):
    for x in range(cnt):
        b, mb = obs_a[x], obs_m[x]
        if not (a >= b + mb + gap or a + m + gap <= b):
            return False
    return True


@njit(cache=True)
def _find_start(owner_row, k0, k1, m, cnt, obs_a, obs_m, gap, best_fit, T):
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
    return best_a


@njit(cache=True)
def _find_start_priced(owner_row, pref_row, k0, k1, m, cnt, obs_a, obs_m, gap, T):
    """价格感知放置：在窗口内所有可行位置中选占用格价格之和最小的，平局时选碎片少的。"""
    best_a, best_c = -1, 1e300
    c = k0
    while c <= k1:
        if owner_row[c] != -1:
            c += 1
            continue
        r0 = c
        while c <= k1 and owner_row[c] == -1:
            c += 1
        r1 = c - 1
        if r1 - r0 < m:
            continue
        g0 = r0
        while g0 - 1 >= 0 and owner_row[g0 - 1] == -1 and r0 - g0 <= m + 1:
            g0 -= 1
        g1 = r1
        while g1 + 1 < T and owner_row[g1 + 1] == -1 and g1 - r1 <= m + 1:
            g1 += 1
        for a in range(r0, r1 - m + 1):
            if not _sep_ok(a, m, cnt, obs_a, obs_m, gap):
                continue
            left, right = a - g0, g1 - (a + m)
            wst = 0
            if 0 < left < m + 1:
                wst += left
            if 0 < right < m + 1:
                wst += right
            cost = (pref_row[a + m + 1] - pref_row[a]) * (1.0 + 1e-9) + 1e-12 * wst
            if cost < best_c:
                best_c, best_a = cost, a
    return best_a


@njit(cache=True)
def greedy_core(order1, order2, v1, v2, c_win, c_m, w_st, w_tg, w_k0, w_k1,
                owner, t_cnt, t_obs_a, t_obs_m, t_task, gap, best_fit, out_st, out_tg, out_a, out_m, out_w, n_out,
                cutoff, pref, use_price):
    """归并两条流：order1 按"未覆盖价值"降序，order2 按"已覆盖价值"降序。价值低于 cutoff 即停止。"""
    T = owner.shape[1]
    p1, p2 = 0, 0
    n1, n2 = len(order1), len(order2)
    while p1 < n1 or p2 < n2:
        # 流 1 里目标已被覆盖的候选可以直接跳过（覆盖是永久的，它在流 2 里还有一份）
        while p1 < n1 and t_cnt[w_tg[c_win[order1[p1]]]] > 0:
            p1 += 1
        if p1 >= n1 and p2 >= n2:
            break
        if p2 >= n2 or (p1 < n1 and v1[order1[p1]] >= v2[order2[p2]]):
            c = order1[p1]
            if v1[c] < cutoff:
                break
            p1 += 1
        else:
            c = order2[p2]
            if v2[c] < cutoff:
                break
            p2 += 1
            # 流 2 队首的目标仍未覆盖：它在流 1 的那份价值更高、已经处理过且放不下，
            # 放不下的原因是站点冲突（永久），所以可以安全丢弃
            if t_cnt[w_tg[c_win[c]]] == 0:
                continue
        w = c_win[c]
        i = w_tg[w]
        cnt = t_cnt[i]
        if cnt >= 3:
            continue
        m = c_m[c]
        s = w_st[w]
        if use_price:
            a = _find_start_priced(owner[s], pref[s], w_k0[w], w_k1[w], m, cnt, t_obs_a[i], t_obs_m[i], gap, T)
        else:
            a = _find_start(owner[s], w_k0[w], w_k1[w], m, cnt, t_obs_a[i], t_obs_m[i], gap, best_fit, T)
        if a < 0:
            continue
        for k in range(a, a + m + 1):
            owner[s, k] = n_out
        t_obs_a[i, cnt] = a
        t_obs_m[i, cnt] = m
        t_task[i, cnt] = n_out
        t_cnt[i] = cnt + 1
        out_st[n_out], out_tg[n_out], out_a[n_out], out_m[n_out], out_w[n_out] = s, i, a, m, w
        n_out += 1
    return n_out


class State:
    """站点占用 + 目标观测记录，贪心和局部搜索共用。"""

    def __init__(self, pb: Problem, cap: int = 60000):
        T = pb.n_samples + 2
        self.owner = np.full((pb.n_stations, T), -1, np.int32)
        self.t_cnt = np.zeros(pb.n_targets, np.int64)
        self.t_obs_a = np.zeros((pb.n_targets, 3), np.int64)
        self.t_obs_m = np.zeros((pb.n_targets, 3), np.int64)
        self.t_task = np.full((pb.n_targets, 3), -1, np.int64)
        self.st = np.zeros(cap, np.int64)
        self.tg = np.zeros(cap, np.int64)
        self.a = np.zeros(cap, np.int64)
        self.m = np.zeros(cap, np.int64)
        self.w = np.zeros(cap, np.int64)
        self.alive = np.zeros(cap, np.bool_)
        self.n = 0

    def schedule(self, pb: Problem, algo: str, runtime: float) -> Schedule:
        sel = np.nonzero(self.alive[:self.n])[0]
        return Schedule.from_cells(pb.cfg, self.st[sel], self.tg[sel], self.a[sel], self.m[sel],
                                   algo=algo, runtime_s=runtime)


def run_greedy(pb: Problem, c_win, c_m, v1, v2, best_fit=True, state: State | None = None,
               cutoff: float = -np.inf, price: np.ndarray | None = None) -> State:
    """price: (站点, 格) 价格；给定时按价格最低的位置放置（新算法 H 用）。"""
    st = state or State(pb)
    if price is None:
        pref, use_price = np.zeros((1, 1)), False
    else:
        pref, use_price = np.concatenate([np.zeros((price.shape[0], 1)), np.cumsum(price, axis=1)], axis=1), True
    order1 = np.argsort(-v1, kind="stable")
    order2 = np.argsort(-v2, kind="stable")
    n0 = st.n
    st.n = greedy_core(order1, order2, v1, v2, c_win, c_m, pb.w_st, pb.w_tg, pb.w_k0, pb.w_k1,
                       st.owner, st.t_cnt, st.t_obs_a, st.t_obs_m, st.t_task, pb.gap, best_fit,
                       st.st, st.tg, st.a, st.m, st.w, st.n, cutoff, pref, use_price)
    st.alive[n0:st.n] = True
    return st


# ---------------------------------------------------------------- 候选与特征
def candidates(pb: Problem, durations=None):
    durations = list(durations or range(pb.m_min, pb.m_max + 1))
    L = pb.w_k1 - pb.w_k0
    cw, cm = [], []
    for m in durations:
        idx = np.nonzero(L >= m)[0]
        cw.append(idx)
        cm.append(np.full(len(idx), m, np.int64))
    return np.concatenate(cw), np.concatenate(cm)


def window_features(pb: Problem) -> dict:
    """与观测时长无关的窗口级特征。"""
    W = len(pb.w_k0)
    n_win = np.diff(pb.t_ptr)
    remaining = np.zeros(W)
    next48 = np.zeros(W)
    for i in range(pb.n_targets):
        ws = pb.t_win[pb.t_ptr[i]:pb.t_ptr[i + 1]]
        if len(ws) == 0:
            continue
        k0, k1 = pb.w_k0[ws], pb.w_k1[ws]
        ends_sorted = np.sort(k1)
        starts_sorted = np.sort(k0)
        # 本窗口结束后还剩多少窗口、之后 48 小时内有多少窗口
        after = len(ws) - np.searchsorted(starts_sorted, k1, side="right")
        within = np.searchsorted(starts_sorted, k1 + pb.gap, side="right") - np.searchsorted(starts_sorted, k1, side="right")
        remaining[ws] = after
        next48[ws] = within
    # 站点负载：每格被多少个窗口覆盖
    load = np.zeros((pb.n_stations, pb.n_samples + 1))
    np.add.at(load, (pb.w_st, pb.w_k0), 1)
    np.add.at(load, (pb.w_st, np.minimum(pb.w_k1 + 1, pb.n_samples)), -1)
    load = np.cumsum(load, axis=1)
    pref = np.concatenate([np.zeros((pb.n_stations, 1)), np.cumsum(load, axis=1)], axis=1)
    mean_load = (pref[pb.w_st, pb.w_k1 + 1] - pref[pb.w_st, pb.w_k0]) / (pb.w_k1 - pb.w_k0 + 1)
    return {
        "scarcity": 1.0 / (1.0 + remaining),
        "urgency": 1.0 / (1.0 + next48),
        "conflict": mean_load / mean_load.max(),
        "n_win": n_win,
    }


def feature_matrix(pb: Problem, wf: dict, c_win, c_m) -> np.ndarray:
    """(C, 8) 候选特征，第 2 列（首次覆盖）在打分时动态处理，这里置 1。"""
    tg = pb.w_tg[c_win]
    P = pb.P[tg]
    sc = P * pb.frac[c_m]
    F = np.empty((len(c_win), 8), np.float32)
    F[:, 0] = P / 5.0
    F[:, 1] = 1.0
    F[:, 2] = sc / 5.0
    F[:, 3] = wf["scarcity"][c_win]
    F[:, 4] = wf["urgency"][c_win]
    F[:, 5] = (sc / (c_m + 1)) / (5.0 * pb.frac[pb.m_min] / (pb.m_min + 1))
    F[:, 6] = wf["conflict"][c_win]
    F[:, 7] = (c_m + 1) / (pb.m_max + 1)
    return F


def ai_values(F: np.ndarray, w: np.ndarray):
    """AIValue = w1..w6 正向，w7、w8 为代价项（减）。返回 (未覆盖时价值, 已覆盖时价值)。"""
    sign = np.array([1, 1, 1, 1, 1, 1, -1, -1], np.float32)
    ww = (np.asarray(w, np.float32) * sign)
    base = F[:, [0, 2, 3, 4, 5, 6, 7]] @ ww[[0, 2, 3, 4, 5, 6, 7]]
    return (base + ww[1]).astype(np.float64), base.astype(np.float64)


MANUAL_WEIGHTS = np.array([1.0, 1.0, 0.5, 0.3, 0.2, 0.5, 0.2, 0.2])


# ---------------------------------------------------------------- 方案 A/B/C/D
def algo_random(pb: Problem, seed=0) -> Schedule:
    t = time.time()
    cw, cm = candidates(pb)
    v = np.random.default_rng(seed).random(len(cw))
    st = run_greedy(pb, cw, cm, v, v, best_fit=False)
    return st.schedule(pb, "A 随机可行调度", time.time() - t)


def algo_priority(pb: Problem) -> Schedule:
    t = time.time()
    cw, cm = candidates(pb)
    v = pb.P[pb.w_tg[cw]] * 10 + pb.frac[cm]
    st = run_greedy(pb, cw, cm, v, v, best_fit=False)
    return st.schedule(pb, "B 固定优先级贪心", time.time() - t)


def algo_weighted(pb: Problem, w, name: str, wf=None, cand=None, return_state=False):
    t = time.time()
    cw, cm = cand if cand is not None else candidates(pb)
    wf = wf if wf is not None else window_features(pb)
    F = feature_matrix(pb, wf, cw, cm)
    v1, v2 = ai_values(F, w)
    st = run_greedy(pb, cw, cm, v1, v2, best_fit=True)
    sch = st.schedule(pb, name, time.time() - t)
    return (sch, st) if return_state else sch
