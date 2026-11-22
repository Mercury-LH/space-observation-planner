"""提速改动的等价性：并行顺序、伐谋规则向量化、覆盖修复剪枝都不能改变结果。"""
import copy

import numpy as np

from obsplan import lagrange
from obsplan.evolved import load_rule, rule_from_module
from obsplan.evolved import recover_rule as original_rule
from obsplan.famou_h_eval import FEATURES, feature_rows
from obsplan.greedy import candidates
from obsplan.local_search import repair_coverage


def test_balanced_order_is_permutation():
    work = np.random.default_rng(0).exponential(size=1001)
    for c in (1, 3, 8, 10):
        order = lagrange.balanced_order(work, c)
        assert np.array_equal(np.sort(order), np.arange(len(work)))
    order = lagrange.balanced_order(work, 4)          # 4 个分块的工作量接近
    shares = [work[ch].sum() for ch in np.array_split(order, 4)]
    assert max(shares) / min(shares) < 1.2


def _sub_args(pb, mu):
    n = pb.n_targets
    pref = np.concatenate([np.zeros((pb.n_stations, 1)), np.cumsum(mu, axis=1)], axis=1)
    outs = (np.zeros(n), np.zeros((n, 3, 3), np.int64), np.zeros(n, np.int64))
    return pref, outs


def test_subproblem_result_does_not_depend_on_order(sub):
    mu = np.random.default_rng(1).random((sub.n_stations, sub.n_samples + 2)) * 0.02
    res = []
    for order in (np.arange(sub.n_targets, dtype=np.int64), None):
        pref, (tot, task, cnt) = _sub_args(sub, mu)
        lagrange._subproblem(pref, sub.t_ptr, sub.t_win, sub.w_st, sub.w_k0, sub.w_k1, sub.vs * sub.P, sub.frac,
                             np.full(sub.n_targets, sub.vc), sub.gap, sub.m_min, sub.m_max, tot, task, cnt,
                             np.zeros((1, 1)), False, np.zeros(sub.n_targets), np.full(sub.n_targets, 3, np.int64),
                             order=order)
        res.append((tot, task, cnt))
    for x, y in zip(*res):
        assert np.array_equal(x, y)


def test_vectorized_rule_matches_original(sub):
    _, info = lagrange.solve(sub, iters=20, verbose=False, local_search=False)
    cw, cm = candidates(sub)
    X = feature_rows(sub, info["mu"], cw, cm)
    rng = np.random.default_rng(2)
    R = rng.normal(size=(5000, len(FEATURES))) * 5
    R[:100] = 0.0                                     # 恰好为 0 的分支
    for M in (X, R):
        u0, c0 = original_rule.score(FEATURES, M.tolist())
        u, c = load_rule()(M)
        assert np.array_equal(u, np.asarray(u0, float)) and np.array_equal(c, np.asarray(c0, float))


def test_rule_falls_back_when_vectorized_disagrees():
    X = np.random.default_rng(3).normal(size=(50, len(FEATURES)))
    wrong = rule_from_module(original_rule, lambda names, M: (np.zeros(len(M)), np.zeros(len(M))))
    u0, c0 = original_rule.score(FEATURES, X.tolist())
    u, c = wrong(X)
    assert np.array_equal(u, np.asarray(u0, float)) and np.array_equal(c, np.asarray(c0, float))


def _repair(pb, state, targets, prune):
    s = copy.deepcopy(state)
    s.n, ins = repair_coverage(targets, s.st, s.tg, s.a, s.m, s.w, s.alive, s.n, s.owner, s.t_cnt, s.t_obs_a,
                               s.t_obs_m, s.t_task, pb.w_st, pb.w_k0, pb.w_k1, pb.t_ptr, pb.t_win, pb.P, pb.frac,
                               pb.vc, pb.vs, pb.gap, pb.m_min, 40, prune)
    return s, ins


def _same_state(x, y):
    n = x.n
    assert n == y.n
    for f in ("st", "tg", "a", "m", "w", "alive"):
        assert np.array_equal(getattr(x, f)[:n], getattr(y, f)[:n]), f
    assert np.array_equal(x.owner, y.owner) and np.array_equal(x.t_cnt, y.t_cnt)
    for i in np.nonzero(x.t_cnt)[0]:                  # 观测记录按集合比较（剪枝只会改变排列顺序）
        k = x.t_cnt[i]
        rx = sorted(zip(x.t_task[i, :k], x.t_obs_a[i, :k], x.t_obs_m[i, :k]))
        ry = sorted(zip(y.t_task[i, :k], y.t_obs_a[i, :k], y.t_obs_m[i, :k]))
        assert rx == ry


def test_repair_pruning_gives_identical_state(sub):
    _, info = lagrange.solve(sub, iters=40, verbose=False, local_search=False)
    state = lagrange.recover(sub, info["mu"], candidates(sub))
    has_win = np.diff(sub.t_ptr) > 0
    unc = np.nonzero((state.t_cnt == 0) & has_win)[0]
    unc = unc[np.argsort(-sub.P[unc], kind="stable")]
    a, ia = _repair(sub, state, unc, True)
    b, ib = _repair(sub, state, unc, False)
    assert ia == ib
    _same_state(a, b)
    rep = np.nonzero((a.t_cnt > 0) & (a.t_cnt < 3))[0]   # 重复观测插入（收益小，剪枝最多）
    rep = rep[np.argsort(-sub.P[rep], kind="stable")]
    a2, ia2 = _repair(sub, a, rep, True)
    b2, ib2 = _repair(sub, a, rep, False)
    assert ia2 == ib2 and ia2 > 0
    _same_state(a2, b2)
