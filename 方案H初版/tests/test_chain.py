"""按秒跨站弹射链与细网格对偶上界：链只在总分变高时改计划，改完的计划必须满足全部时间约束。"""
from dataclasses import replace

import numpy as np
import pytest

from obsplan import bounds, lagrange
from obsplan.chain import BIG, Timeline, _direct_fit, _three_ok, _unit_ab, repattern_pass, run_pass
from obsplan.checks import structural_violations
from obsplan.problem import Problem
from obsplan.refine import refine
from obsplan.schedule import evaluate
from obsplan.second_bounds import all_bounds
from obsplan.second_dual import Dual, warm_start
from obsplan.visibility import Windows


def test_three_ok():
    gap = 48 * 3600
    dur = np.array([120, 120, 120], np.int64)
    day = 86400
    lo = np.array([0, 2 * day + 200, 4 * day + 400], np.int64)
    assert _three_ok(lo, lo + 300, dur, gap)
    assert _three_ok(lo[::-1].copy(), lo[::-1] + 300, dur, gap)             # 顺序无关
    lo2 = np.array([0, 2 * day, 4 * day - 3600], np.int64)                  # 第三次离第二次不到 48 小时
    assert not _three_ok(lo2, lo2 + 300, dur, gap)


def test_direct_fit_matches_brute_force():
    """空当判断（全部任务压到 120 秒后的最早结束 / 最晚开始）与逐秒暴力枚举一致。"""
    rng = np.random.default_rng(3)
    dmin, sw, gap = 120, 30, 48 * 3600
    for _ in range(200):
        n = int(rng.integers(1, 6))
        t, lo, hi = 0, [], []
        for _j in range(n):                                                  # 先造一条可行的线，再给每个任务一点活动余地
            s = t + int(rng.integers(0, 80))
            lo.append(s - int(rng.integers(0, 60)))
            hi.append(s + dmin + int(rng.integers(0, 60)))
            t = s + dmin + sw
        tk_w = np.arange(n, dtype=np.int64)
        wL, wR = np.array(lo, np.int64), np.array(hi, np.int64)
        tk_s = wL.copy()
        tk_e = tk_s + dmin
        tk_tg = np.arange(n, dtype=np.int64)
        tg_tasks = np.full((n, 3), -1, np.int64)
        tg_tasks[:, 0] = np.arange(n)
        A = np.zeros((1, n + 1), np.int64)
        B = np.zeros((1, n + 1), np.int64)
        l2 = np.zeros(n, np.int64)
        h2 = np.zeros(n, np.int64)
        assert _unit_ab(np.arange(n, dtype=np.int64), n, tk_w, tk_s, tk_e, tk_tg, tg_tasks, wL, wR, gap, dmin, sw,
                        -BIG, BIG, A[0], B[0], l2, h2)
        a = int(rng.integers(lo[0] - 300, hi[-1] + 100))
        b = a + int(rng.integers(120, 400))
        got = _direct_fit(0, a, b, A, B, np.array([n], np.int64), dmin) >= 0

        def feasible(k):                                                     # 新任务插在第 k 个任务之前：贪心排最早结束
            order = [(lo[i], hi[i]) for i in range(k)] + [(a, b)] + [(lo[i], hi[i]) for i in range(k, n)]
            t0 = -10 ** 9
            for l, h in order:
                s = max(l, t0)
                if s + dmin > h:
                    return False
                t0 = s + dmin + sw
            return True
        assert got == any(feasible(k) for k in range(n + 1))


@pytest.fixture(scope="module")
def chained(sub):
    sch, _ = lagrange.solve(sub, iters=20, verbose=False)
    ref, info = refine(sub, sch)
    assert info["refined"]
    wL, wR = all_bounds(sub, None, workers=2, verbose=False)
    tl = Timeline(sub, wL, wR, ref)
    cand = np.nonzero((tl.tg_cnt > 0) & (tl.tg_cnt < np.minimum(3, sub.max_obs)) & (sub.P >= 5))[0]
    before = tl.score_points()
    r1 = run_pass(tl, cand, max_depth=3, verbose=False)
    two = np.nonzero((tl.tg_cnt == 2) & (sub.max_obs >= 3) & (sub.P >= 5))[0]
    r2 = repattern_pass(tl, two, max_depth=2, verbose=False)
    return ref, tl, before, r1, r2


def test_chain_keeps_constraints_and_gain_is_real(cfg, ds, sub, chained):
    ref, tl, before, r1, r2 = chained
    out, L, R, used = tl.schedule()
    assert not structural_violations(sub, out, by_second=True)
    assert (out.start_s >= L).all() and (out.start_s + out.dur_s <= R).all()
    assert tl.unmatched == 0 and len(used["station"]) == 0
    gain = tl.score_points() - before
    assert gain == pytest.approx(r1["gain_points"] + r2["gain_points"], abs=1e-6)     # 报的增量 = 实际增量
    e0, e1 = evaluate(cfg, ds, ref), evaluate(cfg, ds, out)
    assert e1["covered"] == e0["covered"]
    assert e1["score"] == pytest.approx(e0["score"] + gain, abs=1e-6)
    assert len(out) == int(tl.tg_cnt.sum()) > len(ref)


def test_fine_dual_equals_grid_dual_on_same_windows(sub):
    """同样的窗口放到 10 秒细网格上（切换占 3 格），同一组价格的对偶值与 30 秒网格上相同：
    价格在 30 秒格内是常数时，最优的起止落在 30 秒格点上。"""
    rng = np.random.default_rng(5)
    mu = rng.exponential(1e-4, (sub.n_stations, sub.n_samples + 2)) * (rng.random((sub.n_stations, sub.n_samples + 2)) < 0.7)
    mu[:, -2:] = 0.0                                  # 最后两格没有任务会用到，两种网格的补齐长度不同
    grid = bounds.dual_value(sub, mu)
    c = 10
    r = sub.cfg.cell_s // c
    fcfg = replace(sub.cfg, cell_s=c)
    win = Windows(sub.win.station, sub.win.target, (sub.win.k0.astype(np.int64) * r).astype(np.int32),
                  (sub.win.k1.astype(np.int64) * r).astype(np.int32), sub.win.dark, fcfg.n_cells + 1, sub.win.sgp4_failed)
    pf = Problem.build(fcfg, sub.ds, win)
    assert Dual(pf).value(warm_start(pf, mu, sub.cfg.cell_s)) == pytest.approx(grid, abs=1e-9)
