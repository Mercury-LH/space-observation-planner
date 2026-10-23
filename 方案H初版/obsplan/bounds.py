"""网络流上界：用来衡量任何算法离最优还有多远。

松弛掉的约束：48 小时间隔的具体位置、任务必须是连续的格、时段内的精确对齐。
保留的约束：每颗目标只能用它可见的站点时段（按 bucket 聚合）、每个站点时段的格数容量、
每颗目标最多观测次数、计分规则。

每颗目标"用掉 x 个格"能拿到的最大价值是 x 的分段线性凹函数：
  前 5 格（一次 120 秒观测 + 30 秒切换）每格 (覆盖奖励 + 0.6P)/5，
  再往后每 5 格一次重复观测，每格 0.12P（直到最多观测次数），
  再往后延长，每格 0.1P（每次观测最多延长 4 格）。
凹函数 = 若干条成本递增的平行弧，于是整个松弛就是一个最小费用最大流问题，可精确求解。
任何合法计划都对应这个网络里的一个可行流，所以流的最优值是合法计划的上界。
"""
from __future__ import annotations

import time

import numpy as np
from ortools.graph.python import min_cost_flow

from .problem import Problem

SCALE = 1e10


def flow_upper_bound(pb: Problem, bucket_cells: int = 10, verbose=True) -> dict:
    t0 = time.time()
    S, T = pb.n_stations, pb.n_samples
    bc = bucket_cells
    nb = (T + 1 + bc - 1) // bc
    # 每个 bucket 的容量 = 其中属于黑夜段的格数（任务占用的格都落在黑夜采样点上）
    dark = np.zeros((S, nb * bc), bool)
    for s, runs in enumerate(pb.win.dark):
        for a, b in runs:
            dark[s, a:b + 1] = True
    cap = dark.reshape(S, nb, bc).sum(axis=2)
    # 目标 -> (站点, bucket) 弧，容量 = 该目标在该时段内可见的格数（且不超过一次最长任务的占用）
    b0 = pb.w_k0 // bc
    b1 = pb.w_k1 // bc
    n_per = b1 - b0 + 1
    rep = np.repeat(np.arange(len(b0)), n_per)
    off = np.arange(n_per.sum()) - np.repeat(np.cumsum(n_per) - n_per, n_per)
    bb = b0[rep] + off
    overlap = np.minimum(pb.w_k1[rep], (bb + 1) * bc - 1) - np.maximum(pb.w_k0[rep], bb * bc) + 1
    key_all = pb.w_tg[rep] * (S * nb) + pb.w_st[rep] * nb + bb
    key, inv = np.unique(key_all, return_inverse=True)
    ov = np.bincount(inv, weights=overlap).astype(np.int64)
    arc_t, arc_b = key // (S * nb), key % (S * nb)
    arc_cap = np.minimum(ov, pb.m_max + 1)

    N = pb.n_targets
    src, sink = 0, 1
    t_node = 2 + np.arange(N)
    b_node = 2 + N + np.arange(S * nb)
    smcf = min_cost_flow.SimpleMinCostFlow()
    tails, heads, caps, costs = [], [], [], []
    m_min, m_ext = pb.m_min + 1, pb.m_max - pb.m_min
    for i in range(N):
        k = int(pb.max_obs[i])
        if k == 0:
            continue
        P = pb.P[i]
        d1 = (pb.vc + pb.vs * P * pb.frac[pb.m_min]) / m_min
        d2 = pb.vs * P * pb.frac[pb.m_min] / m_min
        d3 = pb.vs * P * (pb.frac[pb.m_max] - pb.frac[pb.m_min]) / m_ext
        for c, d in ((m_min, d1), (m_min * (k - 1), d2), (m_ext * k, d3)):
            if c > 0:
                tails.append(src); heads.append(t_node[i]); caps.append(c); costs.append(-int(round(d * SCALE)))
    tails += list(t_node[arc_t]); heads += list(b_node[arc_b])
    caps += list(arc_cap); costs += [0] * len(arc_t)
    flat = cap.ravel()
    used = np.unique(arc_b)
    tails += list(b_node[used]); heads += [sink] * len(used)
    caps += list(flat[used].astype(np.int64)); costs += [0] * len(used)
    smcf.add_arcs_with_capacity_and_unit_cost(np.array(tails, np.int64), np.array(heads, np.int64),
                                              np.array(caps, np.int64), np.array(costs, np.int64))
    big = int(sum(c for t, c in zip(tails, caps) if t == src))
    smcf.set_node_supply(src, big)
    smcf.set_node_supply(sink, -big)
    status = smcf.solve_max_flow_with_min_cost()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"min cost flow status {status}")
    ub = -smcf.optimal_cost() / SCALE
    out = {"combined_ub": float(ub), "bucket_min": bucket_cells * pb.cfg.cell_s / 60,
           "arcs": int(len(tails)), "seconds": time.time() - t0}
    if verbose:
        print(f"[UB] 网络流上界 {ub:.4f}（bucket {out['bucket_min']:.0f} 分钟，{len(tails)} 条弧，{out['seconds']:.0f}s）")
    return out
