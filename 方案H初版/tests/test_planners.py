"""各规划器的输出必须满足全部结构约束；上界必须不小于任何可行解。"""
import numpy as np

from obsplan import cpsat, lagrange
from obsplan.bounds import flow_upper_bound
from obsplan.checks import structural_violations
from obsplan.greedy import MANUAL_WEIGHTS, algo_priority, algo_random, algo_weighted
from obsplan.lagrange import objective
from obsplan.local_search import improve


def test_greedy_variants_valid(sub):
    for sch in (algo_random(sub, seed=1), algo_priority(sub), algo_weighted(sub, MANUAL_WEIGHTS, "C")):
        assert len(sch) > 0
        assert structural_violations(sub, sch) == {}


def test_local_search_keeps_validity_and_improves(sub):
    sch, st = algo_weighted(sub, MANUAL_WEIGHTS, "C", return_state=True)
    before = objective(sub, st)
    improve(sub, st)
    after = objective(sub, st)
    assert after >= before - 1e-12
    assert structural_violations(sub, st.schedule(sub, "C+LS", 0)) == {}


def test_price_coordination_valid_and_bounded(sub):
    sch, info = lagrange.solve(sub, iters=40, verbose=False)
    assert structural_violations(sub, sch) == {}
    tg = sch.target
    obj = sub.vc * np.unique(tg).size + sub.vs * (sub.P[tg] * sub.frac[sch.dur_s // sub.cfg.cell_s]).sum()
    assert info["ub_relaxed"] >= obj - 1e-9
    assert flow_upper_bound(sub, verbose=False)["combined_ub"] >= obj - 1e-9


def test_cpsat_blocks_valid(sub):
    sch = cpsat.solve(sub, MANUAL_WEIGHTS, time_limit=2.0, max_blocks=3, verbose=False)
    assert structural_violations(sub, sch) == {}
