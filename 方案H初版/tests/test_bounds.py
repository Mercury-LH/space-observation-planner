"""上界：对偶上界、按秒口径的放宽问题。上界必须不小于任何合法计划（网格的、按秒精修的）。"""
import numpy as np
import pytest

from obsplan import lagrange
from obsplan.bounds import _short_windows, by_second_problem, dual_upper_bound, dual_value, flow_upper_bound
from obsplan.refine import probe_bounds, refine
from obsplan.schedule import evaluate


def test_by_second_value_minus_price_peaks_on_grid():
    """按秒口径的依据（不需要数据）：价格在每个 30 秒格内是常数、得分对时长是线性的，
    那么真实可见区间内任意起止（整秒或小数）的任务，"价值 − 价格"不超过窗口两端各放宽一格后格点任务的最大值。"""
    rng = np.random.default_rng(1)
    C, m_min, m_max, T = 30.0, 4, 8, 60
    worst = -np.inf
    for _ in range(400):
        mu = rng.exponential(1.0, T) * (rng.random(T) < rng.uniform(0.2, 1.0))
        F = np.concatenate([[0.0], np.cumsum(mu)])                  # F[x] = 前 x 格的价格和

        def cum(t):
            x = min(int(np.floor(t / C)), T - 1)
            return F[x] + mu[x] * (t / C - x)

        P = rng.uniform(0.5, 10)
        k0 = int(rng.integers(2, 30))
        k1 = k0 + int(rng.integers(3, 20))                          # 采样点 k0..k1 可见（至少 4 个）
        t_lo, t_hi = C * k0 - rng.uniform(0, 29.999), C * k1 + rng.uniform(0, 29.999)   # 真实可见区间
        if t_hi - t_lo < 120:
            continue
        best = max(P * (0.6 + 0.4 * (m - m_min) / (m_max - m_min)) - (F[a + m + 1] - F[a])
                   for m in range(m_min, m_max + 1) for a in range(k0 - 1, k1 + 1 - m + 1))
        for j in range(120):
            d = rng.uniform(120, min(240, t_hi - t_lo))
            s = rng.uniform(t_lo, t_hi - d)
            if j % 3 == 0:                                          # 贴边、整格、整秒等特殊位置
                s = [t_lo, t_hi - d, C * np.ceil(t_lo / C), np.floor(s)][j // 3 % 4]
                d = np.floor(d) if j % 2 else d
                if s < t_lo or s + d > t_hi or d < 120:
                    continue
            value = P * (0.6 + 0.4 * (d - 120) / 120) - (cum(s + d + C) - cum(s))    # 占用 [开始, 结束 + 切换)
            worst = max(worst, value - best)
    assert worst <= 1e-9


@pytest.fixture(scope="module")
def plans(sub):
    sch, _ = lagrange.solve(sub, iters=40, verbose=False)
    ref, info = refine(sub, sch)
    assert info["refined"]
    return sch, ref


@pytest.fixture(scope="module")
def ext(sub):
    return by_second_problem(sub, targets=np.isin(np.arange(sub.n_targets), sub.w_tg))


def test_by_second_problem_contains_refined_tasks(cfg, sub, ext, plans):
    _, ref = plans
    assert len(ext.w_st) > len(sub.w_st)                            # 加进了短窗口
    cell = cfg.cell_s
    end = ref.start_s + ref.dur_s
    for i in range(len(ref)):
        w = ext.t_win[ext.t_ptr[ref.target[i]]:ext.t_ptr[ref.target[i] + 1]]
        w = w[ext.w_st[w] == ref.station[i]]
        assert ((ext.w_k0[w] * cell <= ref.start_s[i]) & (end[i] <= ext.w_k1[w] * cell)).any()
    # 占用的格（含切换）都在放宽后的黑夜段里
    dark = np.zeros((ext.n_stations, ext.n_samples + 1), bool)
    for s, runs in enumerate(ext.win.dark):
        for a, b in runs:
            dark[s, a:b + 1] = True
    first, last = ref.start_s // cell, (end + cfg.switch_s - 1) // cell
    for i in range(len(ref)):
        assert dark[ref.station[i], first[i]:last[i] + 1].all()


def test_by_second_problem_contains_true_visible_intervals(cfg, ds, sub, ext):
    """按秒探查出的真实可见区间（常规窗口、短窗口各抽一批）都落在放宽问题里同站同目标的某个窗口内。"""
    rng = np.random.default_rng(3)
    st, tg, k0, k1 = (np.asarray(x).astype(np.int64) for x in _short_windows(sub))
    short = np.nonzero((k1 - k0 == sub.m_min - 1) & np.isin(st, np.unique(sub.w_st)) & np.isin(tg, sub.w_tg))[0]
    assert len(short)
    short = rng.choice(short, min(150, len(short)), replace=False)
    reg = rng.choice(len(sub.w_st), min(150, len(sub.w_st)), replace=False)
    st, tg = np.concatenate([st[short], sub.w_st[reg]]), np.concatenate([tg[short], sub.w_tg[reg]])
    k0, k1 = np.concatenate([k0[short], sub.w_k0[reg]]), np.concatenate([k1[short], sub.w_k1[reg]])
    L, R = probe_bounds(cfg, ds, st, tg, k0.astype(np.int64), k1.astype(np.int64))
    cell = cfg.cell_s
    assert (R - L >= (k1 - k0) * cell).all()
    assert (R[:len(short)] - L[:len(short)] >= cfg.obs_min_s).any()        # 短窗口里确实有按秒才够长的
    for i in range(len(st)):
        w = ext.t_win[ext.t_ptr[tg[i]]:ext.t_ptr[tg[i] + 1]]
        w = w[ext.w_st[w] == st[i]]
        assert ((ext.w_k0[w] * cell <= L[i]) & (R[i] <= ext.w_k1[w] * cell)).any()


def test_bounds_hold_for_grid_and_refined_plans(cfg, ds, sub, ext, plans):
    sch, ref = plans
    grid_obj, ref_obj = evaluate(cfg, ds, sch)["combined"], evaluate(cfg, ds, ref)["combined"]
    assert ref_obj >= grid_obj - 1e-9
    f_grid = flow_upper_bound(sub, verbose=False)["combined_ub"]
    f_ext = flow_upper_bound(ext, verbose=False)["combined_ub"]
    assert f_ext >= f_grid - 1e-9 and f_grid >= grid_obj - 1e-9 and f_ext >= ref_obj - 1e-9

    r = dual_upper_bound(sub, target=grid_obj, iters=60, verbose=False)
    assert r["combined_ub"] >= grid_obj - 1e-9
    assert dual_value(sub, r["mu"]) == pytest.approx(r["combined_ub"], abs=1e-9)        # 价格是可复核的证书
    assert dual_value(ext, r["mu"]) >= r["combined_ub"] - 1e-9                          # 放宽的问题只会更大

    r2 = dual_upper_bound(ext, target=ref_obj, iters=60, mu0=r["mu"], verbose=False)
    assert r2["combined_ub"] >= ref_obj - 1e-9
    assert r2["combined_ub"] <= dual_value(ext, r["mu"]) + 1e-9                         # 热启动后不会变差
    assert dual_value(ext, r2["mu"]) == pytest.approx(r2["combined_ub"], abs=1e-9)
    assert dual_value(ext, r2["mu"].astype(np.float32)) >= ref_obj - 1e-9               # 存成单精度仍是合法上界


def test_dual_rejects_bad_prices(sub):
    shape = (sub.n_stations, sub.n_samples + 2)
    with pytest.raises(ValueError):
        dual_value(sub, np.zeros((shape[0], shape[1] - 1)))
    with pytest.raises(ValueError):
        dual_value(sub, np.full(shape, -1.0))
    assert dual_value(sub, np.zeros(shape)) >= flow_upper_bound(sub, verbose=False)["combined_ub"] - 1e-9
