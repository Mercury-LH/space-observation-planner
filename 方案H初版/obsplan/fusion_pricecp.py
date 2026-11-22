"""融合方案：价格引导的 CP-SAT 分块（H 的拉格朗日价格 + G 的滚动分块 CP-SAT）。

不修改 obsplan/cpsat.py；本文件复制并改写其 solve()/to_state()，只把候选筛选那一步换掉：
原 G 用 AI 权重（D 训练得到）给每个候选 (窗口, 时长) 打分，每颗目标每块只保留价值最高的
k_win 个窗口、每个窗口只保留价值最高的 k_dur 种时长，再交给 CP-SAT 分块求解。

这里改成用 H（拉格朗日价格协调，obsplan/lagrange.py）收敛后的价格 mu（站点 × 格）：
对每个候选 (窗口, 时长)，先在窗口内找价格最低的放置位置（复用 lagrange._candidate_cost），
得到该候选的"成本"；候选价值仍是 vs*P*frac[m] + vc（近似按首次观测算，和 AI 方案里
"首次覆盖"特征恒为 1 的处理一致）；价值 − 成本 即"约减收益"，用它代替 AI 价值做同样的筛选。
价格来自对偶问题，约减收益为正的候选大致对应"全局看合算"的候选，理论上应比固定权重的
启发式筛选更贴近当前资源稀缺状况（晚上黑暗段价格高、候选会被压到更挑剔的位置）。

CP-SAT 分块求解、滚动提交、结果格式与 obsplan/cpsat.py 完全一致，以保证和 G/G+ 公平对比。
"""
from __future__ import annotations

import threading
import time

import numpy as np
from ortools.sat.python import cp_model

from .greedy import State, candidates
from .lagrange import _candidate_cost
from .problem import Problem
from .schedule import Schedule

SCALE = 1e7


def price_guided_candidates(pb: Problem, mu: np.ndarray, k_win: int = 1, k_dur: int = 1):
    """按"价值 − 价格"（约减收益）筛选候选，复刻 cpsat.solve 里 AI 筛选那段逻辑。

    返回 (cw, cm, win_rb)：筛选后保留下来的候选（窗口下标、时长格数），以及每个窗口的
    最高约减收益（win_rb，用于块内按目标选 k_win 个窗口）。
    """
    cw, cm = candidates(pb)
    pref = np.concatenate([np.zeros((mu.shape[0], 1)), np.cumsum(mu, axis=1)], axis=1)
    cost = _candidate_cost(pref, cw, cm, pb.w_st, pb.w_k0, pb.w_k1)
    val = pb.vs * pb.P[pb.w_tg[cw]] * pb.frac[cm] + pb.vc
    rb = val - cost
    # 每个窗口只保留约减收益最高的 k_dur 种时长（与 cpsat.solve 对 ai1 做的事完全相同）
    order = np.lexsort((-rb, cw))
    rank = np.zeros(len(cw), np.int64)
    first = np.r_[True, cw[order][1:] != cw[order][:-1]]
    grp = np.cumsum(first) - 1
    starts = np.nonzero(first)[0]
    rank[order] = np.arange(len(order)) - starts[grp]
    keep = rank < k_dur
    cw, cm, rb = cw[keep], cm[keep], rb[keep]
    win_rb = np.full(len(pb.w_k0), -np.inf)
    np.maximum.at(win_rb, cw, rb)
    return cw, cm, win_rb


def solve(pb: Problem, mu: np.ndarray, block_h=4.0, commit_h=3.0, k_win=1, k_dur=1, time_limit=10.0,
          workers=8, hint: Schedule | None = None, verbose=True, name="PriceCP 价格引导CP-SAT分块",
          max_blocks=None) -> Schedule:
    t0 = time.time()
    cfg = pb.cfg
    cw, cm, win_rb = price_guided_candidates(pb, mu, k_win=k_win, k_dur=k_dur)

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
        # 每颗目标只保留约减收益最高的 k_win 个窗口
        tg = pb.w_tg[cw[idx]]
        wv = win_rb[cw[idx]]
        o = np.lexsort((-wv, tg))
        idx, tg = idx[o], tg[o]
        wins = cw[idx]
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
        # 价格引导的候选往往把多颗目标挤到同一批"低价"格子上，NoOverlap/AtMostOne 的分支搜索
        # 比 AI 筛选更容易遇到退化实例，观测到 max_time_in_seconds 偶尔被显著超出；
        # 这里加一个定时器兜底，到 2×time_limit（至少 +5 秒）仍没返回就强制停止搜索，
        # 保证单块求解时间有硬上限（代价是可能只拿到当前最优的可行解，不影响结果合法性）。
        guard = threading.Timer(max(time_limit * 2, time_limit + 5), solver.StopSearch)
        guard.daemon = True
        guard.start()
        try:
            status = solver.Solve(model)
        finally:
            guard.cancel()
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
            print(f"  [PriceCP] 块{nblk:2d} [{B0 * cfg.cell_s / 3600:5.1f}h,{B1 * cfg.cell_s / 3600:5.1f}h) 变量 {len(vars_):6d} "
                  f"{solver.StatusName(status):8s} 提交 {n_commit:4d}  {time.time() - t0:.0f}s")
        B0 = C1
    c = np.array(committed, np.int64).reshape(-1, 5)
    sch = Schedule.from_cells(cfg, c[:, 0], c[:, 1], c[:, 2], c[:, 3], algo=name, runtime_s=time.time() - t0)
    sch.info["windows"] = c[:, 4]
    return sch


def to_state(pb: Problem, sch: Schedule) -> State:
    """同 cpsat.to_state；复制一份避免依赖/修改 obsplan/cpsat.py。"""
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
