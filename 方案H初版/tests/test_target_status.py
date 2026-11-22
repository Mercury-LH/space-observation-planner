"""目标状态分析：与计分口径一致（覆盖数、观测次数、得分），状态互斥且完整。"""
import numpy as np

from obsplan import lagrange
from obsplan.schedule import evaluate
from obsplan.target_status import STATUS, summarize, target_status


def test_status_matches_evaluate(cfg, ds, sub):
    sch, _ = lagrange.solve(sub, iters=20, verbose=False)
    rows = target_status(sub, sch)
    ev = evaluate(cfg, ds, sch)
    assert len(rows) == sub.n_targets
    assert sum(r["n_obs"] > 0 for r in rows) == ev["covered"]
    assert sum(r["n_obs"] for r in rows) == ev["tasks"]
    assert np.isclose(sum(r["task_score"] for r in rows), ev["score"], atol=0.05)
    for r in rows:
        assert r["status"] in STATUS and r["status_cn"] == STATUS[r["status"]]
        assert r["n_obs"] <= max(r["max_obs"], 0) or r["status"] == "SGP4_FAILED"
        assert (r["n_windows"] == 0) == (r["status"] in ("NO_WINDOW", "SGP4_FAILED")) or r["status"] == "SGP4_FAILED"
    sm = summarize(rows)
    assert sum(sum(v.values()) for v in sm["by_priority"].values()) == sub.n_targets
    assert sm["covered"] == ev["covered"]
