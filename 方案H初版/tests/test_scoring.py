import numpy as np
import pytest

from obsplan.schedule import Schedule, evaluate, task_score


def test_task_score_linear(cfg):
    P = np.full(5, 5.0)
    got = task_score(cfg, P, np.array([90, 120, 180, 240, 300]))
    assert got == pytest.approx([0.0, 3.0, 4.0, 5.0, 5.0])


def test_repeat_gap_end_to_start(cfg, ds):
    """答疑 #30/#32：上次观测结束 48 小时后才能再计分。"""
    t = int(np.nonzero(ds.priority == 5.0)[0][0])
    gap = int(cfg.repeat_gap_h * 3600)
    ok = Schedule(np.array([0, 1]), np.array([t, t]), np.array([0, 120 + gap]), np.array([120, 120]))
    early = Schedule(np.array([0, 1]), np.array([t, t]), np.array([0, 120 + gap - 30]), np.array([120, 120]))
    assert evaluate(cfg, ds, ok)["repeat_tasks"] == 1
    assert evaluate(cfg, ds, early)["repeat_tasks"] == 0
    assert evaluate(cfg, ds, ok)["covered"] == evaluate(cfg, ds, early)["covered"] == 1
