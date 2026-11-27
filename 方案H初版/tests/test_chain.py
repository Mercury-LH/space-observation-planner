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


def test_parallel_pass_is_legal_and_deterministic(sub):
    """多线程弹射链：结果合法（结构检查 0 违规），报的增量等于实际增量，两次运行完全相同（与线程调度无关）。"""
    from obsplan.chain import run_pass_parallel
    sch, _ = lagrange.solve(sub, iters=20, verbose=False)
    ref, _ = refine(sub, sch)
    wL, wR = all_bounds(sub, None, workers=2, verbose=False)
    outs = []
    for _ in range(2):
        tl = Timeline(sub, wL, wR, ref)
        before = tl.score_points()
        cand = np.nonzero((tl.tg_cnt > 0) & (tl.tg_cnt < np.minimum(3, sub.max_obs)) & (sub.P >= 5))[0]
        r = run_pass_parallel(tl, cand, threads=4, max_depth=3, verbose=False, batch=8)
        out, L, R, used = tl.schedule()
        assert not structural_violations(sub, out, by_second=True)
        assert tl.score_points() - before == pytest.approx(r["gain_points"], abs=1e-6)
        assert r["added"] > 0
        outs.append((r["added"], sorted(zip(out.station.tolist(), out.target.tolist(), out.start_s.tolist()))))
    assert outs[0] == outs[1]


def test_relocate_parallel_is_legal_and_deterministic(sub):
    """多线程换位置：结果合法（结构检查 0 违规），报的增量等于实际增量，两次运行完全相同。"""
    from obsplan.chain import relocate_pass_parallel
    sch, _ = lagrange.solve(sub, iters=20, verbose=False)
    ref, _ = refine(sub, sch)
    wL, wR = all_bounds(sub, None, workers=2, verbose=False)
    outs = []
    for _ in range(2):
        tl = Timeline(sub, wL, wR, ref)
        before = tl.score_points()
        r = relocate_pass_parallel(tl, threads=4, max_depth=1, min_absorb=0.05)
        out, L, R, used = tl.schedule()
        assert not structural_violations(sub, out, by_second=True)
        assert (out.start_s >= L).all() and (out.start_s + out.dur_s <= R).all()
        assert tl.score_points() - before == pytest.approx(r["gain_points"], abs=1e-6)
        assert r["gain_points"] >= 0.0
        outs.append((r["relocated"], round(r["gain_points"], 6),
                     sorted(zip(out.station.tolist(), out.target.tolist(), out.start_s.tolist(), out.dur_s.tolist()))))
    assert outs[0] == outs[1]


def test_chain_ai_features_and_model(sub):
    """AI 预测器：特征有限、维数对；随包的模型能加载并给出 0～1 的概率。"""
    from obsplan.chain_ai import FEATURES, features, load_predictor
    sch, _ = lagrange.solve(sub, iters=20, verbose=False)
    ref, _ = refine(sub, sch)
    wL, wR = all_bounds(sub, None, workers=2, verbose=False)
    tl = Timeline(sub, wL, wR, ref)
    xs = np.nonzero((tl.tg_cnt > 0) & (tl.tg_cnt < 3))[0][:50]
    F = features(tl, xs)
    assert F.shape == (len(xs), len(FEATURES)) and np.isfinite(F).all()
    pred = load_predictor()
    if pred is not None:
        p = pred.prob(F)
        assert ((p >= 0) & (p <= 1)).all()


def test_compiled_local_lp_matches_glop(sub):
    """局部重排的编译版（_lp_fast，释放全局锁）与原来的 GLOP 实现给出同样的最优得分；
    同一目标两次观测同时进规划（跨夜、要隔 48 小时）的耦合情形也一样。时间线得分的编译版与逐个相加逐位相同。"""
    sch, _ = lagrange.solve(sub, iters=20, verbose=False)
    ref, _ = refine(sub, sch)
    wL, wR = all_bounds(sub, None, workers=2, verbose=False)
    tl = Timeline(sub, wL, wR, ref)
    k = 0.4 / (tl.dmax - tl.dmin)
    for u in range(tl.n_units):
        py = sum(tl.P[tl.tk_tg[p]] * (0.6 + k * (tl.tk_e[p] - tl.tk_s[p] - tl.dmin)) for p in tl.seq[u])
        assert tl.units_score([u]) == py
    checked = coupled = 0
    rng = np.random.default_rng(0)
    for y in rng.permutation(np.nonzero(tl.alive[:tl.n] & (tl.tk_w[:tl.n] >= 0))[0])[:60]:
        u = int(tl.tk_unit[y])
        seq = [p for p in tl.seq[u] if p != y]
        for lp_k in (tl.lp_k, 0):
            a = tl._lp({u: seq}, {}, (int(y),), lp_k)
            b = tl._lp_glop({u: seq}, {}, (int(y),), lp_k)
            assert (a is None) == (b is None)
            if a is not None:
                assert a[0] == pytest.approx(b[0], abs=1e-6)
                assert set(a[1]) == set(b[1])
                checked += 1
    for x in np.nonzero(tl.tg_cnt >= 2)[0][:40]:              # 两次观测在不同的夜：两条时间线整条进规划，耦合
        p, q = (int(v) for v in tl.tg_tasks[x, :2])
        ua, ub = int(tl.tk_unit[p]), int(tl.tk_unit[q])
        if ua == ub:
            continue
        new_seqs = {ua: list(tl.seq[ua]), ub: list(tl.seq[ub])}
        a = tl._lp(new_seqs, {}, (), 0)
        b = tl._lp_glop(new_seqs, {}, (), 0)
        assert (a is None) == (b is None)
        if a is not None:
            assert a[0] == pytest.approx(b[0], abs=1e-6)
            s_p, e_p = a[1][p]
            s_q, e_q = a[1][q]
            assert s_q >= e_p + tl.gap or s_p >= e_q + tl.gap
            coupled += 1
    assert checked > 50 and coupled > 5
