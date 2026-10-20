"""方案 G：AI 筛选候选 + CP-SAT 滚动时域求解（规划方案 8-G、9.1 节）。

每个时间块：
- AI 价值函数（D 学到的权重）给候选打分，每颗目标在块内只保留价值最高的 K 个窗口、每个窗口保留价值最高的 2 种时长；
- 每个候选是一个可选区间变量，开始时间在窗口内由求解器决定，长度 = 观测 + 30 秒切换；
- 每站 NoOverlap；每颗目标块内至多一次（块长 < 48 小时），并满足与已提交观测的 48 小时间隔；
- 目标函数 = 得分 + 首次覆盖奖励（按当前已覆盖状态），用 D 的贪心结果做初始提示；
求解后只提交块前段开始的任务，后段留到下一块重新优化（重叠区）。
"""
from __future__ import annotations

import time

import numpy as np
from ortools.sat.python import cp_model

from .greedy import State, ai_values, candidates, feature_matrix, window_features
from .problem import Problem
from .schedule import Schedule

SCALE = 1e7


def solve(pb: Problem, weights, block_h=4.0, commit_h=3.0, k_win=1, k_dur=1, time_limit=10.0,
          workers=8, hint: Schedule | None = None, verbose=True, name="G AI筛选+CP-SAT", max_blocks=None) -> Schedule:
    t0 = time.time()
    cfg = pb.cfg
    cw, cm = candidates(pb)
    F = feature_matrix(pb, window_features(pb), cw, cm)
    ai1, _ = ai_values(F, weights)
    # 每个窗口保留 AI 价值最高的 k_dur 种时长
    order = np.lexsort((-ai1, cw))
    rank = np.zeros(len(cw), np.int64)
    first = np.r_[True, cw[order][1:] != cw[order][:-1]]
    grp = np.cumsum(first) - 1
    starts = np.nonzero(first)[0]
    rank[order] = np.arange(len(order)) - starts[grp]
    keep = rank < k_dur
    cw, cm, ai1 = cw[keep], cm[keep], ai1[keep]
    win_ai = np.full(len(pb.w_k0), -np.inf)
    np.maximum.at(win_ai, cw, ai1)

    hint_map = {}
    if hint is not None:
        for s, t, a, m in zip(hint.station, hint.target, hint.start_s // cfg.cell_s, hint.dur_s // cfg.cell_s):
            hint_map[(int(t), int(s))] = (int(a), int(m))

    block, commit = int(block_h * 3600 / cfg.cell_s), int(commit_h * 3600 / cfg.cell_s)
    T = pb.n_samples
    committed = []            # (station, target, a, m, window)
    t_obs = [[] for _ in range(pb.n_targets)]
    busy = [[] for _ in range(pb.n_stations)]   # 已提交任务在本站的占用区间
    B0 = 0
    nblk = 0
    while B0 < T - pb.m_min and (max_blocks is None or nblk < max_blocks):
        B1 = min(B0 + block, T)
        C1 = B1 if B1 >= T else B0 + commit
        # 块内候选：窗口与 [B0, B1] 相交
        sel = (pb.w_k1[cw] >= B0 + cm) & (pb.w_k0[cw] <= B1 - cm)
        idx = np.nonzero(sel)[0]
        # 每颗目标只保留 AI 价值最高的 k_win 个窗口
        tg = pb.w_tg[cw[idx]]
        wv = win_ai[cw[idx]]
        o = np.lexsort((-wv, tg))
        idx, tg = idx[o], tg[o]
        wins = cw[idx]
        # 目标内窗口去重后排名
        keep_idx = []
        cur_t, seen = -1, []
        for j, t, w in zip(idx, tg, wins):
            if t != cur_t:
                cur_t, seen = t, []
            if w in seen:
                keep_idx.append(j)
            elif len(seen) < k_win:
                seen.append(w)
                keep_idx.append(j)
        model = cp_model.CpModel()
        per_station = [[] for _ in range(pb.n_stations)]
        per_target = {}
        obj = []
        vars_ = []
        hinted = set()
        for j in keep_idx:
            w, m = int(cw[j]), int(cm[j])
            s, i = int(pb.w_st[w]), int(pb.w_tg[w])
            if len(t_obs[i]) >= 3:
                continue
            lo, hi = max(int(pb.w_k0[w]), B0), min(int(pb.w_k1[w]), B1) - m
            for (pa, pm) in t_obs[i]:
                if pa < lo:
                    lo = max(lo, pa + pm + pb.gap)
                else:
                    hi = min(hi, pa - pb.gap - m)
            if hi < lo:
                continue
            x = model.NewBoolVar("")
            a = model.NewIntVar(lo, hi, "")
            iv = model.NewOptionalFixedSizeIntervalVar(a, m + 1, x, "")
            per_station[s].append(iv)
            per_target.setdefault(i, []).append(x)
            val = pb.vs * pb.P[i] * pb.frac[m] + (pb.vc if not t_obs[i] else 0.0)
            obj.append((x, int(round(val * SCALE))))
            vars_.append((x, a, s, i, m, w))
            h = hint_map.get((i, s))
            if h is not None and h[1] == m and lo <= h[0] <= hi and (i, s) not in hinted:
                model.AddHint(x, 1)
                model.AddHint(a, h[0])
                hinted.add((i, s))
            else:
                model.AddHint(x, 0)
                model.AddHint(a, lo)
        for s in range(pb.n_stations):
            for (ba, bb) in busy[s]:
                if bb >= B0 and ba <= B1:
                    per_station[s].append(model.NewIntervalVar(ba, bb - ba + 1, bb + 1, ""))
            if per_station[s]:
                model.AddNoOverlap(per_station[s])
        for i, xs in per_target.items():
            if len(xs) > 1:
                model.AddAtMostOne(xs)
        model.Maximize(sum(c * x for x, c in obj))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = workers
        status = solver.Solve(model)
        n_commit = 0
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for x, a, s, i, m, w in vars_:
                if solver.Value(x):
                    av = solver.Value(a)
                    if av < C1:
                        committed.append((s, i, av, m, w))
                        t_obs[i].append((av, m))
                        busy[s].append((av, av + m))
                        n_commit += 1
        nblk += 1
        if verbose:
            print(f"  [G] 块{nblk:2d} [{B0 * cfg.cell_s / 3600:5.1f}h,{B1 * cfg.cell_s / 3600:5.1f}h) 变量 {len(vars_):6d} "
                  f"{solver.StatusName(status):8s} 提交 {n_commit:4d}  {time.time() - t0:.0f}s")
        B0 = C1
    c = np.array(committed, np.int64).reshape(-1, 5)
    sch = Schedule.from_cells(cfg, c[:, 0], c[:, 1], c[:, 2], c[:, 3], algo=name, runtime_s=time.time() - t0)
    sch.info["windows"] = c[:, 4]
    return sch


def to_state(pb: Problem, sch: Schedule) -> State:
    """把 G 的结果装进 State，便于接局部搜索。"""
    st = State(pb)
    cfg = pb.cfg
    for s, i, a, m, w in zip(sch.station, sch.target, sch.start_s // cfg.cell_s, sch.dur_s // cfg.cell_s,
                             sch.info["windows"]):
        k = st.n
        st.st[k], st.tg[k], st.a[k], st.m[k], st.w[k] = s, i, a, m, w
        st.alive[k] = True
        st.owner[s, a:a + m + 1] = k
        c = st.t_cnt[i]
        st.t_obs_a[i, c], st.t_obs_m[i, c], st.t_task[i, c] = a, m, k
        st.t_cnt[i] = c + 1
        st.n += 1
    return st
