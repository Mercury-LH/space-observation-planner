"""解合并：区间团必须覆盖全部冲突对；合并结果合法、不比提示差、系数与 combined 成比例。"""
import numpy as np
import pytest

from obsplan import fusion_merge as fm
from obsplan import lagrange
from obsplan.checks import structural_violations
from obsplan.greedy import algo_priority
from obsplan.schedule import evaluate


def test_interval_cliques_cover_every_overlapping_pair():
    rng = np.random.default_rng(0)
    lo = rng.integers(0, 60, 40)
    hi = lo + rng.integers(1, 12, 40)
    ids = np.arange(40)
    cl = [set(c) for c in fm.interval_cliques(lo, hi, ids)]
    for i in range(40):
        for j in range(i + 1, 40):
            overlap = lo[i] < hi[j] and lo[j] < hi[i]
            assert overlap == any(i in c and j in c for c in cl), (i, j)
    for c in cl:                                    # 团内两两相交
        c = list(c)
        assert max(lo[c]) < min(hi[c])


@pytest.fixture(scope="module")
def merged(sub):
    a, _ = lagrange.solve(sub, iters=20, verbose=False)
    b = algo_priority(sub)
    for s in (a, b):
        s.start_s = np.rint(s.start_s).astype(np.int64)
        s.dur_s = np.rint(s.dur_s).astype(np.int64)
    pool = fm.build_pool(sub, [a, b], durations=(4, 6, 8), shifts=(-1, 1))
    hint = fm.select_of(pool, fm._cells(sub, a))
    sel, logs = fm.merge_blocks(sub, pool, hint, block_h=24, step_h=12, time_limit=10, verbose=False)
    return a, pool, hint, sel


def test_pool_contains_inputs_and_stays_in_windows(sub, merged):
    a, pool, hint, _ = merged
    assert len(np.unique(hint)) == len(a)
    assert (pool.a >= sub.w_k0[pool.w]).all() and (pool.a + pool.m <= sub.w_k1[pool.w]).all()
    assert ((pool.m >= sub.m_min) & (pool.m <= sub.m_max)).all()
    assert (sub.w_st[pool.w] == pool.st).all() and (sub.w_tg[pool.w] == pool.tg).all()


def test_merge_is_legal_and_not_worse_than_hint(cfg, ds, sub, merged):
    a, pool, hint, sel = merged
    sch = fm.to_schedule(sub, pool, sel, "merge")
    assert not structural_violations(sub, sch)
    assert fm.objective(sub, pool, sel) >= fm.objective(sub, pool, hint)
    ci = fm.coefficients(sub, pool)[1]
    for s, idx in ((a, hint), (sch, sel)):        # 整数目标与 evaluate 的 combined 一致
        assert fm.combined_of(sub, fm.objective(sub, pool, idx), ci) == pytest.approx(
            evaluate(cfg, ds, s)["combined"], abs=1e-9)
