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

对偶上界（dual_upper_bound）：给每个站的每个 30 秒格定一个价格 mu >= 0，
  L(mu) = 全部价格之和 + 每颗目标在这组价格下各自最优的"价值 − 价格"之和，
对任何 mu >= 0 都是合法计划的上界（价格协调算法用的就是它）。把 mu 用次梯度法迭代到接近收敛，
保存下来的 mu 就是一份可复核的证书：dual_value(pb, mu) 一次子问题计算即可重算出同一个数。
它保留了"任务是连续的格、在哪个窗口里"，理论上不比网络流上界松，但要迭代很多次才收敛。

按秒口径（by_second_problem）：上面两个上界原本只对起止落在 30 秒格点上的计划成立。按秒精修后的计划
起止是任意整秒，还能用"格点上不够 120 秒、按秒才够"的短窗口。把每个窗口两端各放宽一格、再加上
4 个采样点的短窗口，得到一个放宽的问题；它的两个上界对按秒的计划同样成立：
  1. 按秒合法的任务，它覆盖的采样点都可见且相邻采样点之间逐秒可见，所以落在某个网格窗口内，
     两端各超出不到一格；时长 >= 120 秒的任务至少盖住 4 个采样点，所以只需补上 4 个采样点的短窗口。
  2. 同站任务按"开始到结束后 30 秒切换"互不重叠，即每个格被占用的总时长不超过 30 秒；
     任务的价格 = 价格在它占用时段上的积分。得分对时长是线性的、价格在格内是常数，
     所以单个窗口内"价值 − 价格"的最大值在格点上取到（tests/test_bounds.py 有数值检验）。
  3. 48 小时间隔在子问题里用的是窗口级的放松判定，放宽后的窗口同样满足。
注意"合法"指本程序的可见性模型（含仰角、太阳高度、地影的安全余量）。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ortools.graph.python import min_cost_flow

from .problem import Problem
from .visibility import Windows

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


def _short_windows(pb: Problem):
    """比最短观测少一格的网格窗口 (station, target, k0, k1)。get_windows 顺带缓存；没有缓存时现算。"""
    cfg = pb.cfg
    path = Path(cfg.cache_dir) / f"short_{cfg.scenario_key()}.npz"
    if path.exists():
        z = np.load(path)
        return z["station"], z["target"], z["k0"], z["k1"]
    from dataclasses import replace
    from .visibility import compute_windows
    w = compute_windows(replace(cfg, obs_min_s=(pb.m_min - 1) * cfg.cell_s), pb.ds, verbose=False)
    return w.station, w.target, w.k0, w.k1


def by_second_problem(pb: Problem, short=None, targets: np.ndarray | None = None) -> Problem:
    """按秒口径的放宽问题：每个窗口两端各放宽一格，加上 4 个采样点的短窗口，黑夜段两端也各放宽一格。
    按秒合法的计划都是它的（松弛后的）可行解，所以它的上界对按秒的计划成立，理由见模块说明。
    short：(station, target, k0, k1) 短窗口，默认读缓存；只取 pb 里有窗口的站。
    targets：(N,) bool，只给这些目标加短窗口（演示用的子问题传参与的目标；默认全部，上界仍合法、略松）。"""
    st, tg, k0, k1 = _short_windows(pb) if short is None else short
    st, tg, k0, k1 = (np.asarray(x).astype(np.int64) for x in (st, tg, k0, k1))
    keep = (k1 - k0 == pb.m_min - 1) & np.isin(st, np.unique(pb.w_st))
    if targets is not None:
        keep &= np.asarray(targets, bool)[tg]
    last = pb.n_samples - 1
    w_st = np.concatenate([pb.w_st, st[keep]])
    w_tg = np.concatenate([pb.w_tg, tg[keep]])
    w_k0 = np.maximum(np.concatenate([pb.w_k0, k0[keep]]) - 1, 0)
    w_k1 = np.minimum(np.concatenate([pb.w_k1, k1[keep]]) + 1, last)
    order = np.lexsort((w_k0, w_st))
    dark = [[(max(a - 1, 0), min(b + 1, last)) for a, b in runs] for runs in pb.win.dark]
    win = Windows(w_st[order].astype(np.int16), w_tg[order].astype(np.int32), w_k0[order].astype(np.int32),
                  w_k1[order].astype(np.int32), dark, pb.n_samples, pb.win.sgp4_failed)
    return Problem.build(pb.cfg, pb.ds, win)


def _dual_setup(pb: Problem):
    from . import lagrange
    if 3 * (pb.gap + pb.m_min) + pb.m_min <= pb.n_samples - 1:
        raise NotImplementedError("规划期长到一颗目标可能观测 4 次以上；价格子问题只算到 3 次，对偶上界不适用")
    N = pb.n_targets
    order = lagrange.balanced_order(lagrange.target_work(pb.t_ptr, pb.t_win, pb.w_k0, pb.w_k1))
    bufs = (np.zeros(N), np.zeros((N, 3, 3), np.int64), np.zeros(N, np.int64))
    const = (pb.vs * pb.P, np.full(N, pb.vc), np.full(N, 3, np.int64), np.zeros((1, 1)), np.zeros(N))
    return order, bufs, const


def _dual_eval(pb: Problem, mu: np.ndarray, setup) -> float:
    """L(mu)；各目标的最优选择留在 setup 的缓冲区里（供次梯度用）。"""
    from . import lagrange
    order, (out_total, out_task, out_cnt), (Pv, vct, maxk, qpref, beta) = setup
    S = pb.n_stations
    pref = np.concatenate([np.zeros((S, 1)), np.cumsum(mu, axis=1)], axis=1)
    lagrange._subproblem(pref, pb.t_ptr, pb.t_win, pb.w_st, pb.w_k0, pb.w_k1, Pv, pb.frac, vct, pb.gap,
                         pb.m_min, pb.m_max, out_total, out_task, out_cnt, qpref, False, beta, maxk, order=order)
    return float(mu.sum() + out_total.sum())


def _check_mu(pb: Problem, mu) -> np.ndarray:
    mu = np.asarray(mu, np.float64)
    if mu.shape != (pb.n_stations, pb.n_samples + 2) or not np.isfinite(mu).all() or (mu < 0).any():
        raise ValueError("价格必须是 (站点数, 采样点数 + 2) 的非负有限数组")
    return mu


def dual_value(pb: Problem, mu: np.ndarray) -> float:
    """给定价格 mu（>= 0）的对偶值 L(mu)：任何 mu 都给出合法上界。用来复核保存下来的价格。"""
    return _dual_eval(pb, _check_mu(pb, mu), _dual_setup(pb))


def dual_upper_bound(pb: Problem, target: float, iters: int = 2000, mu0: np.ndarray | None = None,
                     theta: float = 1.0, patience: int = 20, decay: float = 0.7, checkpoint=None,
                     verbose: bool = True, log_every: int = 100) -> dict:
    """次梯度法把对偶上界迭代到接近收敛。
    target：步长用的目标值，取已知最好计划的综合值（上界不会低于它）；mu0：起始价格（可接着上次的结果）。
    步长 = theta × (L − target) / |g|²；连续 patience 次没有改进就把 theta 乘以 decay。
    checkpoint：路径，每 200 次迭代保存一次（mu 为当前最好价格，theta 与 it 供续算）。
    返回 combined_ub（过程中最小的 L）和取到它的价格 mu——dual_value(pb, mu) 可重算。"""
    from . import lagrange
    t0 = time.time()
    S, T = pb.n_stations, pb.n_samples + 2
    setup = _dual_setup(pb)
    out_task, out_cnt = setup[1][1], setup[1][2]
    mu = np.zeros((S, T)) if mu0 is None else _check_mu(pb, mu0).copy()
    best, best_mu, stall, hist = np.inf, mu.copy(), 0, []

    def save(it):
        if checkpoint is not None:
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(checkpoint, mu=best_mu, best=best, theta=theta, it=it, target=target)

    it = 0
    for it in range(iters):
        L = _dual_eval(pb, mu, setup)
        if L < best - 1e-7:
            best, best_mu, stall = L, mu.copy(), 0
        else:
            stall += 1
            if stall >= patience:
                theta, stall = max(theta * decay, 1e-4), 0
        g = lagrange._demand(out_task, out_cnt, S, T) - 1.0
        g[(mu <= 0) & (g < 0)] = 0.0
        norm = float((g * g).sum())
        if it % log_every == 0 or it == iters - 1:
            hist.append((it, L, best, theta))
            if verbose:
                print(f"  [对偶上界] it {it:5d}  L {L:.6f}  最好 {best:.6f}  步长系数 {theta:.4f}  "
                      f"超载格 {int((g > 0).sum())}  {time.time() - t0:.0f}s", flush=True)
        if it % 200 == 199:
            save(it + 1)
        if norm == 0 or L <= target:
            break
        mu = np.maximum(0.0, mu + theta * (L - target) / norm * g)
    save(it + 1)
    return {"combined_ub": float(best), "mu": best_mu, "iters": it + 1, "theta": theta, "target": float(target),
            "history": hist, "seconds": time.time() - t0}
