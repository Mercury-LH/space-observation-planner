"""伐谋放置规则：特征与人工规则一致，带演化规则的 H 仍输出合法计划。"""
import numpy as np

from obsplan import famou_h_eval as fh
from obsplan import lagrange
from obsplan.checks import structural_violations
from obsplan.greedy import candidates


def test_features_and_baseline_rule(sub):
    _, info = lagrange.solve(sub, iters=20, verbose=False, local_search=False)
    cw, cm = candidates(sub)
    X = fh.feature_rows(sub, info["mu"], cw, cm)
    assert X.shape == (len(cw), len(fh.FEATURES)) and np.isfinite(X).all()
    c = {n: i for i, n in enumerate(fh.FEATURES)}
    assert set(np.unique(X[:, c["sub_pick"]])) <= {0.0, 1.0}
    assert (X[:, c["sub_pick_same_duration"]] <= X[:, c["sub_pick"]]).all()
    u, v = fh.baseline_rule(X)
    assert np.allclose(u - v, sub.vc / sub.vs, atol=1e-3)      # 未覆盖比已覆盖多一份覆盖价值


def test_solve_with_rule_is_legal(sub):
    k = fh.FEATURES.index("sub_pick")
    rule = lambda X: tuple(x + 3.0 * X[:, k] for x in fh.baseline_rule(X))  # noqa: E731
    sch, info = lagrange.solve(sub, iters=20, verbose=False, rule=rule)
    assert not structural_violations(sub, sch)
    assert "lb_manual_rule" in info
