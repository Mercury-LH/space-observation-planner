"""外部基线插件合约；纯合成窗口，几何 finish 使用显式测试替身。"""
import json
from types import SimpleNamespace

import numba
import numpy as np
import pytest

from benchmarks import baselines as b
from obsplan.benchmark import resolve_callable
from obsplan.checks import structural_violations
from obsplan.config import Config
from obsplan.problem import Problem
from obsplan.schedule import Schedule
from obsplan.visibility import Windows


@pytest.fixture
def tiny():
    cfg = Config(days=1 / 24, use_sun=False, use_sunlit=False)
    ds = SimpleNamespace(n_targets=3, n_stations=1, priority=np.array([5.0, 3.0, 2.0]),
                         priority_label=np.array(["高", "中", "低"]))
    win = Windows(np.zeros(3, np.int16), np.arange(3, dtype=np.int32),
                  np.array([0, 10, 20], np.int32), np.array([8, 18, 28], np.int32),
                  [[(0, 119)]], 121, np.zeros(3, bool))
    return Problem.build(cfg, ds, win)


@pytest.mark.parametrize("mode", ["manual_greedy", "priority"])
def test_real_greedy_modes_are_legal_and_use_shared_finish(tiny, monkeypatch, mode):
    finished = []

    def fake_finish(pb, sch, **kwargs):
        finished.append((sch, kwargs))
        return sch, {"refine": {"refined": True}, "synthetic_identity": True}

    monkeypatch.setattr(b, "finish", fake_finish)
    old_threads = numba.get_num_threads()
    events = []
    solve = resolve_callable("benchmarks.baselines:solve")
    sch, info = solve(tiny, {"mode": mode, "workers": 1, "seed": 17}, events.append)
    assert len(sch) == 3
    assert not structural_violations(tiny, sch)
    assert len(finished) == 1
    assert finished[0][1] == {"near": True, "verbose": False}
    assert set(info["stage_timings_s"]) == {"construct", "grid_check", "finish", "final_summary"}
    assert info["runtime_s"] >= sum(info["stage_timings_s"].values())
    assert sch.runtime_s == info["runtime_s"]
    assert info["actual_config"]["mode"] == mode
    assert info["seed_used"] is False
    assert info["famou"]["runtime_cloud_calls"] == 0
    assert info["grid_final"]["covered"] == 3
    assert numba.get_num_threads() == old_threads
    assert {event["event"] for event in events} == {"start", "complete"}
    json.dumps(info, allow_nan=False)


def test_each_solve_constructs_fresh_and_routes_only_selected_algorithm(tiny, monkeypatch):
    calls = []
    expected = b.MANUAL_WEIGHTS.copy()

    def manual(pb, weights, name):
        assert np.array_equal(weights, expected)
        assert weights is not b.MANUAL_WEIGHTS
        calls.append("manual")
        return Schedule.from_cells(pb.cfg, [0], [0], [0], [8])

    def priority(pb):
        calls.append("priority")
        return Schedule.from_cells(pb.cfg, [0], [1], [10], [8])

    monkeypatch.setattr(b, "algo_weighted", manual)
    monkeypatch.setattr(b, "algo_priority", priority)
    monkeypatch.setattr(b, "finish", lambda pb, sch, **kw: (sch, {"synthetic_identity": True}))
    first, _ = b.solve(tiny, {"mode": "manual_greedy", "workers": 1})
    second, _ = b.solve(tiny, {"mode": "manual_greedy", "workers": 1})
    third, _ = b.solve(tiny, {"mode": "priority", "workers": 1})
    assert calls == ["manual", "manual", "priority"]
    assert first is not second
    assert third.target.tolist() == [1]
    assert np.array_equal(b.MANUAL_WEIGHTS, expected)


@pytest.mark.parametrize("spec", [
    {"mode": "unknown"}, {"workers": 0}, {"workers": True}, {"seed": -1},
    {"seed": 1.5}, {"finish_near": "false"}, {"unrecognized": 1},
])
def test_rejects_invalid_specs_before_constructing(tiny, monkeypatch, spec):
    def unexpected(*args, **kwargs):
        raise AssertionError("不应开始构造")

    monkeypatch.setattr(b, "algo_weighted", unexpected)
    monkeypatch.setattr(b, "algo_priority", unexpected)
    with pytest.raises(ValueError):
        b.solve(tiny, {"workers": 1, **spec})


def test_rejects_invalid_grid_and_restores_threads(tiny, monkeypatch):
    monkeypatch.setattr(b, "algo_priority",
                        lambda pb: Schedule.from_cells(pb.cfg, [0, 0], [0, 1], [0, 0], [8, 8]))
    previous_threads = numba.get_num_threads()
    with pytest.raises(ValueError, match="非法网格"):
        b.solve(tiny, {"mode": "priority", "workers": 1})
    assert numba.get_num_threads() == previous_threads


def test_refinement_failure_propagates_and_restores_threads(tiny, monkeypatch):
    monkeypatch.setattr(b, "algo_priority",
                        lambda pb: Schedule.from_cells(pb.cfg, [0], [0], [0], [8]))

    def failure(*args, **kwargs):
        raise RuntimeError("refinement failure")

    monkeypatch.setattr(b, "finish", failure)
    previous_threads = numba.get_num_threads()
    with pytest.raises(RuntimeError, match="refinement failure"):
        b.solve(tiny, {"mode": "priority", "workers": 1})
    assert numba.get_num_threads() == previous_threads
